# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from rich.progress import Progress

from tasks.base import Task
from utils.loss import SimCLRLoss
from utils.metrics import TopKAccuracy
from utils.optimization import get_optimizer
from utils.optimization import get_cosine_scheduler


class SimCLR(Task):
    def __init__(self, backbone: nn.Module, projector: nn.Module):
        super(SimCLR, self).__init__()

        self.backbone = backbone
        self.projector = projector

        self.scaler = None
        self.optimizer = None
        self.loss_function = None
        self.writer = None

    def prepare(self,
                ckpt_dir: str,
                optimizer: str = 'lars',
                learning_rate: float = 1.0,
                weight_decay: float = 0.0,
                temperature: float = 0.07,
                distributed: bool = False,
                local_rank: int = 0,
                mixed_precision: bool = True,
                **kwargs):  # pylint: disable=unused-argument
        """Prepare training."""

        # Distributed training (optional)
        if distributed:
            self.backbone = DistributedDataParallel(
                nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone).to(local_rank),
                device_ids=[local_rank]
            )
            self.projector = DistributedDataParallel(
                nn.SyncBatchNorm.convert_sync_batchnorm(self.projector).to(local_rank),
                device_ids=[local_rank]
            )
        else:
            self.backbone.to(local_rank)
            self.projector.to(local_rank)

        # Mixed precision training (optional)
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

        # Optimization
        self.optimizer = get_optimizer(
            params=[
                {'params': self.backbone.parameters()},
                {'params': self.projector.parameters()},
            ],
            name=optimizer,
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Loss function
        self.loss_function = SimCLRLoss(
            temperature=temperature,
            distributed=distributed,
            local_rank=local_rank
        )

        # TensorBoard
        self.writer = SummaryWriter(ckpt_dir) if local_rank == 0 else None

        self.ckpt_dir = ckpt_dir                # pylint: disable=attribute-defined-outside-init
        self.distributed = distributed          # pylint: disable=attribute-defined-outside-init
        self.local_rank = local_rank            # pylint: disable=attribute-defined-outside-init
        self.mixed_precision = mixed_precision  # pylint: disable=attribute-defined-outside-init
        self.prepared = True                    # pylint: disable=attribute-defined-outside-init

    def run(self,
            dataset,
            epochs: int = 100,
            batch_size: int = 1024,
            num_workers: int = 0,
            cosine_warmup: int = -1,
            save_every: int = 100,
            **kwargs):

        if not self.prepared:
            raise RuntimeError("Training not prepared.")

        # Data loading
        sampler = DistributedSampler(dataset) if self.distributed else None
        shuffle = not self.distributed
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True
        )

        # Learning rate scheduling (optional)
        if cosine_warmup >= 0:
            scheduler = get_cosine_scheduler(
                optimizer=self.optimizer,
                epochs=epochs,
                warmup_steps=cosine_warmup
            )
        else:
            scheduler = None

        # Logging
        logger = kwargs.get('logger', None)

        # Pre-train
        for epoch in range(1, epochs + 1):

            if self.distributed:
                sampler.set_epoch(epoch)

            history = self.train(data_loader)
            log = " | ".join([f"{k} : {v:.4f}" for k, v in history.items()])
            if logger is not None:
                logger.info(f"Epoch [{epoch:>4}/{epochs:>4}] - " + log)

            if self.writer is not None:
                for k, v in history.items():
                    self.writer.add_scalar(k, v, global_step=epoch)
                if scheduler is not None:
                    lr = scheduler.get_last_lr()[0]
                    self.writer.add_scalar('lr', lr, global_step=epoch)

            if (epoch % save_every == 0) & (self.local_rank == 0):
                ckpt = os.path.join(self.ckpt_dir, f"ckpt.{epoch}.pth.tar")
                self.save_checkpoint(ckpt, epoch=epoch)

            if scheduler is not None:
                scheduler.step()

    def train(self, data_loader):
        """SimCLR training."""

        steps = len(data_loader)
        self._set_learning_phase(train=True)
        result = {
            'loss': torch.zeros(steps, device=self.local_rank),
            'rank@1': torch.zeros(steps, device=self.local_rank)
        }

        with Progress(transient=True, auto_refresh=False) as pg:
            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Waiting...", total=steps)

            for i, batch in enumerate(data_loader):

                with torch.cuda.amp.autocast(self.mixed_precision):
                    x1 = batch['x1'].to(self.local_rank)
                    x2 = batch['x2'].to(self.local_rank)
                    z = self.predict(torch.cat([x1, x2], dim=0))
                    z = torch.stack(z.chunk(2, dim=0), dim=1)
                    loss, logits, pos_mask = self.loss_function(z)

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self.optimizer.zero_grad()

                result['loss'][i] = loss.detach()
                with torch.no_grad():
                    preds = logits.detach()
                    labels = pos_mask.detach().eq(1).nonzero(as_tuple=True)[1]
                result['rank@1'][i] = TopKAccuracy(k=2)(preds, labels)

                if self.local_rank == 0:
                    desc = f"[bold green] [{i+1}/{steps}]: "
                    for k, v in result.items():
                        desc += f" {k} : {v[:i+1].mean():.4f} |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

        return {k: v.mean().item() for k, v in result.items()}

    def predict(self, x: torch.Tensor):
        return self.projector(self.backbone(x))

    def _set_learning_phase(self, train=False):
        if train:
            self.backbone.train()
            self.projector.train()
        else:
            self.backbone.eval()
            self.projector.eval()

    def save_checkpoint(self, path: str, **kwargs):
        ckpt = {
            'backbone': self.backbone.state_dict(),
            'projector': self.projector.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if kwargs:
            ckpt.update(kwargs)
        torch.save(ckpt, path)

    def load_model_from_checkpoint(self, path: str):
        ckpt = torch.load(path)
        self.backbone.load_state_dict(ckpt['backbone'])
        self.projector.load_state_dict(ckpt['projector'])
        self.optimizer.load_state_dict(ckpt['optimizer'])

    def load_history_from_checkpoint(self, path: str):
        ckpt = torch.load(path)
        del ckpt['backbone']
        del ckpt['projector']
        del ckpt['optimizer']
        return ckpt
