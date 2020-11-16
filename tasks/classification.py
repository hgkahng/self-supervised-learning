# -*- coding: utf-8 -*-

import os
import collections

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from rich.progress import Progress

from tasks.base import Task
from utils.metrics import TopKAccuracy
from utils.logging import make_epoch_description
from utils.optimization import get_optimizer
from utils.optimization import get_cosine_scheduler
from utils.optimization import get_multi_step_scheduler


class Classification(Task):
    def __init__(self, backbone: nn.Module, classifier: nn.Module):
        super(Classification, self).__init__()

        self.backbone = backbone
        self.classifier = classifier

        self.scaler = None
        self.optimizer = None
        self.scheduler = None
        self.loss_function = None
        self.writer = None

        self.prepared = False

    def prepare(self,
                ckpt_dir: str,
                optimizer: str = 'lars',
                learning_rate: float = 1.0,
                weight_decay: float = 0.0,
                cosine_warmup: int = 0,
                epochs: int = 100,
                batch_size: int = 256,
                num_workers: int = 0,
                distributed: bool = False,
                local_rank: int = 0,
                mixed_precision: bool = True,
                **kwargs):  # pylint: disable=unused-argument
        """Add function docstring."""

        # Set attributes
        self.ckpt_dir = ckpt_dir                # pylint: disable=attribute-defined-outside-init
        self.epochs = epochs                    # pylint: disable=attribute-defined-outside-init
        self.batch_size = batch_size            # pylint: disable=attribute-defined-outside-init
        self.num_workers = num_workers          # pylint: disable=attribute-defined-outside-init
        self.distributed = distributed          # pylint: disable=attribute-defined-outside-init
        self.local_rank = local_rank            # pylint: disable=attribute-defined-outside-init
        self.mixed_precision = mixed_precision  # pylint: disable=attribute-defined-outside-init

        # Distributed training (optional)
        if distributed:
            self.backbone = DistributedDataParallel(
                nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone).to(local_rank),
                device_ids=[local_rank]
            )
            self.classifier = DistributedDataParallel(
                nn.SyncBatchNorm.convert_sync_batchnorm(self.classifier).to(local_rank),
                device_ids=[local_rank]
            )
        else:
            self.backbone.to(local_rank)
            self.classifier.to(local_rank)

        # Mixed precision training (optional)
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

        # Optimization (TODO: freeze)
        self.optimizer = get_optimizer(
            params=[
                {'params': self.backbone.parameters()},
                {'params': self.classifier.parameters()},
            ],
            name=optimizer,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = get_cosine_scheduler(self.optimizer, epochs=epochs, warmup_steps=cosine_warmup)

        # Loss function
        self.loss_function = nn.CrossEntropyLoss()

        # TensorBoard
        self.writer = SummaryWriter(ckpt_dir) if local_rank == 0 else None

        # Ready to train!
        self.prepared = True

    def run(self,
            train_set,
            eval_set,
            test_set: torch.utils.data.Dataset = None,
            save_every: int = 10,
            **kwargs):  # pylint: disable=unused-argument

        epochs = self.epochs
        batch_size = self.batch_size
        num_workers = self.num_workers

        if not self.prepared:
            raise RuntimeError("Training not prepared.")

        # DataLoader (train, val, test)
        sampler = DistributedSampler(train_set) if self.distributed else None
        shuffle = not self.distributed
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True
        )
        eval_loader = DataLoader(
            eval_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True
        )

        # Logging
        logger = kwargs.get('logger', None)

        # Supervised training
        best_eval_loss = float('inf')
        best_epoch     = 0

        for epoch in range(1, epochs + 1):

            if self.distributed:
                sampler.set_epoch(epoch)

            # Train & evaluate
            train_history = self.train(train_loader)
            eval_history  = self.evaluate(eval_loader)
            epoch_history = collections.defaultdict(dict)
            for k, v1 in train_history.items():
                epoch_history[k]['train'] = v1
                try:
                    v2 = eval_history[k]
                    epoch_history[k]['eval'] = v2
                except KeyError:
                    continue

            # Write TensorBoard summary
            if self.writer is not None:
                for k, v in epoch_history.items():
                    self.writer.add_scalars(k, v, global_step=epoch)
                if self.scheduler is not None:
                    lr = self.scheduler.get_last_lr()[0]
                    self.writer.add_scalar('lr', lr, global_step=epoch)

            # Save best model checkpoint
            eval_loss = eval_history['loss']
            if eval_loss <= best_eval_loss:
                best_eval_loss = eval_loss
                best_epoch = epoch
                if self.local_rank == 0:
                    ckpt = os.path.join(self.ckpt_dir, f"ckpt.best.pth.tar")
                    self.save_checkpoint(ckpt, epoch=epoch)

            # Save intermediate model checkpoints
            if (epoch % save_every == 0) & (self.local_rank == 0):
                ckpt = os.path.join(self.ckpt_dir, f"ckpt.{epoch}.pth.tar")
                self.save_checkpoint(ckpt, epoch=epoch)

            # Write logs
            log = make_epoch_description(
                history=epoch_history,
                current=epoch,
                total=epochs,
                best=best_epoch,
            )
            if logger is not None:
                logger.info(log)

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

        # Save final model checkpoint
        ckpt = os.path.join(self.ckpt_dir, f"ckpt.last.pth.tar")
        self.save_checkpoint(ckpt, epoch=epoch)

        # Test (optional)
        if test_set is not None:
            test_loader = DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=False
            )
        test_history = self.evaluate(test_loader)
        if (self.local_rank == 0) & (logger is not None):
            log = "Test: "
            for k, v in test_history.items():
                log += f" {k}: {v:.4f} |"
            logger.info(log)

    def train(self, data_loader):
        """Training defined for a single epoch."""

        steps = len(data_loader)
        self._set_learning_phase(train=True)
        result = {
            'loss': torch.zeros(steps, device=self.local_rank),
            'top@1': torch.zeros(steps, device=self.local_rank),
        }

        with Progress(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Training...", total=steps)

            for i, batch in enumerate(data_loader):

                with torch.cuda.amp.autocast(self.mixed_precision):
                    x = batch['x'].to(self.local_rank)
                    y = batch['y'].to(self.local_rank)
                    logits = self.predict(x)
                    loss = self.loss_function(logits, y)

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self.optimizer.zero_grad()

                result['loss'][i] = loss.detach()
                result['top@1'][i] = TopKAccuracy(k=1)(logits, y).detach()

                if self.local_rank == 0:
                    desc = f"[bold green] [{i+1}/{steps}]: "
                    for k, v in result.items():
                        desc += f" {k} : {v[:i+1].mean():.4f} |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

        return {k: v.mean().item() for k, v in result.items()}

    @torch.no_grad()
    def evaluate(self, data_loader):
        """Evaluation defined for a single epoch."""

        steps = len(data_loader)
        self._set_learning_phase(train=False)
        result = {
            'loss': torch.zeros(steps, device=self.local_rank),
            'top@1': torch.zeros(steps, device=self.local_rank),
        }

        for i, batch in enumerate(data_loader):

            x = batch['x'].to(self.local_rank)
            y = batch['y'].to(self.local_rank)
            logits = self.predict(x)
            loss = self.loss_function(logits, y)

            result['loss'][i] = loss
            result['top@1'][i] = TopKAccuracy(k=1)(logits, y)

        return {k: v.mean().item() for k, v in result.items()}

    def predict(self, x: torch.FloatTensor):
        """Make a prediction provided a batch of samples."""
        return self.classifier(self.backbone(x))

    def _set_learning_phase(self, train=False):
        if train:
            self.backbone.train()
            self.classifier.train()
        else:
            self.backbone.eval()
            self.classifier.eval()

    def save_checkpoint(self, path: str, **kwargs):
        ckpt = {
            'backbone': self.backbone.state_dict(),
            'classifier': self.classifier.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if kwargs:
            ckpt.update(kwargs)
        torch.save(ckpt, path)

    def load_model_from_checkpoint(self, path: str):
        ckpt = torch.load(path)
        self.backbone.load_state_dict(ckpt['backbone'])
        self.classifier.load_state_dict(ckpt['classifier'])
        self.optimizer.load_state_dict(ckpt['optimizer'])

    def load_history_from_checkpoint(self, path: str):
        ckpt = torch.load(path)
        del ckpt['backbone']
        del ckpt['classifier']
        del ckpt['optimizer']
        return ckpt
