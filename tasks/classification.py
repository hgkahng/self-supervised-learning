# -*- coding: utf-8 -*-

import os
import typing
import logging
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp.grad_scaler import GradScaler
from layers.batchnorm import SplitBatchNorm2d
from models.backbone import ResNetBackbone
from models.head import LinearClassifier

from tasks.base import Task
from utils.metrics import TopKAccuracy
from utils.optimization import get_optimizer, WarmupCosineDecayLR
from utils.logging import get_rich_logger, get_rich_pbar, make_epoch_message
from utils.decorators import timer


num_classes_of_dataset = {
    'cifar10': 10,
    'cifar100': 100,
    'stl10': 10,
    'imagenet': 1000,
}


class Classification(Task):
    def __init__(self, config: object, local_rank: int):
        super(Classification, self).__init__()
        self.config = config
        self.local_rank = local_rank

        self._init_logger()
        self._init_modules()
        self._init_cuda()
        self._init_criterion()
        self._init_optimization()
    
    def _init_logger(self):
        if self.local_rank == 0:
            logfile = os.path.join(self.config.checkpoint_dir, 'main.log')
            self.logger = get_rich_logger(logfile)
            self.logger.info(f'Checkpoint directory: {self.config.checkpoint_dir}')
        else:
            self.logger = None

    def _init_modules(self):
        self.encoder = ResNetBackbone(
            name=self.config.backbone_type,
            data=self.config.data,
            in_channels=3
        )
        self.encoder = SplitBatchNorm2d.revert_batchnorm(self.encoder)
        self.classifier = LinearClassifier(
            in_channels=self.encoder.out_channels,
            num_classes=num_classes_of_dataset[self.config.data]
        )
        
        if self.config.pretrained_file is not None:
            try:
                self.encoder.load_weights_from_checkpoint(
                    self.config.pretrained_file, key='encoder'
                )
            except KeyError:
                self.encoder.load_weights_from_checkpoint(
                    self.config.pretrained_file, key='backbone'
                )
            finally:
                if self.logger is not None:
                    self.logger.info(
                        f'Loaded pretrained model from {self.config.pretrained_file}')
        
        if not self.config.finetune:
            self.encoder.freeze_weights()

        if self.logger is not None:
            self.logger.info(f"Encoder ({self.config.backbone_type}): {self.encoder.num_parameters:,}")
            self.logger.info(f"Classifier (linear)): {self.classifier.num_parameters:,}")
        
    def _init_cuda(self):
        if self.config.distributed:
            self.encoder = DistributedDataParallel(
                module=nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder).to(self.local_rank),
                device_ids=[self.local_rank],
            )
            self.classifier = DistributedDataParallel(
                module=nn.SyncBatchNorm.convert_sync_batchnorm(self.classifier).to(self.local_rank),
                device_ids=[self.local_rank],
            )
        else:
            self.encoder.to(self.local_rank)
            self.classifier.to(self.local_rank)

    def _init_criterion(self):
        self.criterion=nn.CrossEntropyLoss()

    def _init_optimization(self):
        self.optimizer = get_optimizer(
            params=[{'params': self.encoder.parameters()},
                    {'params': self.classifier.parameters()}],
            name=self.config.optimizer,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.scheduler = WarmupCosineDecayLR(
            optimizer=self.optimizer,
            total_epochs=self.config.epochs,
            warmup_epochs=self.config.lr_warmup,
            warmup_start_lr=1e-4,
            min_decay_lr=1e-4,
        )
        self.amp_scaler = GradScaler() if self.config.mixed_precision else None

    def run(self,
            dataset: torch.utils.data.Dataset,
            eval_set: torch.utils.data.Dataset,
            test_set: torch.utils.data.Dataset):
        """Classification training and evaulation."""

        # For distributed training, we must provide an explicit sampler to
        # avoid duplicates across devices. Each node (i.e., GPU) will be trained on
        # `len(dataset) // world_size' samples.
        sampler = DistributedSampler(dataset, shuffle=True) if self.config.distributed else None
        shuffle = not self.config.distributed
        train_loader = DataLoader(dataset,
                                  batch_size=self.config.batch_size,
                                  sampler=sampler,
                                  shuffle=shuffle,
                                  num_workers=self.config.num_workers,
                                  drop_last=True,
                                  pin_memory=True,
                                  persistent_workers=True)
        eval_loader = DataLoader(eval_set,
                                 batch_size=self.config.batch_size,
                                 shuffle=False,
                                 num_workers=self.config.num_workers,
                                 drop_last=False,
                                 pin_memory=True,
                                 persistent_workers=True,)

        best_eval_acc: float = 0.
        best_epoch:      int = 0

        # Iterations of epochs start from 1, for simplicity.
        for epoch in range(1, self.config.epochs + 1):
            
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)

            # Train & evaluate
            train_history: dict = self.train(train_loader)
            eval_history:  dict = self.evaluate(eval_loader)
            history = dict(train=train_history, eval=eval_history)

            # Logging (https://wandb.ai)
            if self.local_rank == 0:
                if self.scheduler is not None:
                    history['misc'] = dict(lr=self.scheduler.get_last_lr()[0])
                wandb.log(data=history, step=epoch)
                        
            # Logging (terminal)
            if self.logger is not None:
                msg_train: str = make_epoch_message(train_history, epoch, self.config.epochs, best_epoch)
                msg_eval:  str = make_epoch_message(eval_history, epoch, self.config.epochs, best_epoch)
                self.logger.info(f"(Train) {msg_train}")
                self.logger.info(f"( Eval) {msg_eval} ")

            # Save intermediate model checkpoints
            if (epoch % self.config.save_every == 0) & (self.local_rank == 0):
                ckpt = os.path.join(self.config.checkpoint_dir, f"ckpt.{epoch}.pth.tar")
                self.save_checkpoint(ckpt, epoch=epoch)
                
            # Save best model checkpoint
            eval_acc = eval_history['top@1']
            if eval_acc <= best_eval_acc:
                best_eval_loss = eval_acc
                best_epoch = epoch
                if self.local_rank == 0:
                    ckpt = os.path.join(self.config.checkpoint_dir, f"ckpt.best.pth.tar")
                    self.save_checkpoint(ckpt, epoch=epoch)

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

        # Save final model checkpoint
        ckpt = os.path.join(self.config.checkpoint_dir, f"ckpt.final.pth.tar")
        self.save_checkpoint(ckpt, epoch=epoch)
        
        # Test (optional when `test_set` is provided)
        if (self.local_rank == 0) and (test_set is not None):
            # Instantiate test loader
            test_loader = DataLoader(test_set,
                                     batch_size=self.config.batch_size,
                                     shuffle=False,
                                     num_workers=self.config.num_workers,
                                     drop_last=False,
                                     pin_memory=False,
                                     persistent_workers=False)
            # Last model (last epoch)
            last_history = self.evaluate(test_loader)
            if self.logger is not None:
                last_msg = " | ".join([f"{k}:{v:.4f}" for k, v in last_history.items()])
                last_msg = "(Final) : " + last_msg
                self.logger.info(last_msg)
            # Best model (with highest validation accuracy)
            self.load_model_from_checkpoint(os.path.join(self.config.checkpoint_dir, "ckpt.best.pth.tar"))
            best_history = self.evaluate(test_loader)
            if self.logger is not None:
                best_msg = " | ".join([f"{k} : {v:.4f}" for k, v in best_history.items()])
                best_msg = "( Best) : " + best_msg
                self.logger.info(best_msg)

            wandb.log({
                'test.top@1': wandb.Table(
                    columns=['last', 'best'],
                    data=[[last_history['top@1'], best_history['top@1']]]
                ),
            })

    def train(self, data_loader: DataLoader) -> typing.Dict[str,torch.FloatTensor]:
        """Iterates over the `data_loader' once for training."""

        steps = len(data_loader)
        self._set_learning_phase(train=True, finetune=self.config.finetune)

        result = {
            'loss': torch.zeros(steps, device=self.local_rank),
            'top@1': torch.zeros(steps, device=self.local_rank),
        }

        with get_rich_pbar(transient=True,
                           auto_refresh=False,
                           disable=self.local_rank != 0) as pbar:
            job = pbar.add_task(f":thread:", total=steps)
            
            for i, batch in enumerate(data_loader):
                
                # Forward pass
                with torch.cuda.amp.autocast(self.config.mixed_precision):
                    # Get data & assign device
                    x = batch['x'].to(self.local_rank)
                    y = batch['y'].to(self.local_rank)
                    # Compute logits & loss
                    logits = self.predict(x)
                    loss = self.criterion(logits, y)
                
                # Backpropagation
                if self.amp_scaler is not None:
                    self.amp_scaler.scale(loss).backward()
                    self.amp_scaler.step(self.optimizer)
                    self.amp_scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Update progress bar
                result['loss'][i] = loss.detach()
                result['top@1'][i] = TopKAccuracy(k=1)(logits, y).detach()
                msg = " | ".join([f"{k} : {v[:i+1].mean():.4f}" for k, v in result.items()])
                msg = f":thread:(Rank: {self.local_rank}) [{i+1}/{steps}]: " + msg
                pbar.update(job, advance=1., description=msg)
                pbar.refresh()

        return {k: v.mean().item() for k, v in result.items()}

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> typing.Dict[str,torch.FloatTensor]:
        """Iterates over the `data_loader' once for evaluation."""

        steps = len(data_loader)
        self._set_learning_phase(train=False, finetune=self.config.finetune)
        result = {
            'loss': torch.zeros(steps, device=self.local_rank),
            'top@1': torch.zeros(steps, device=self.local_rank),
        }

        for i, batch in enumerate(data_loader):
            x = batch['x'].to(self.local_rank)
            y = batch['y'].to(self.local_rank)
            logits = self.predict(x)
            loss = self.criterion(logits, y)
            result['loss'][i] = loss
            result['top@1'][i] = TopKAccuracy(k=1)(logits, y)

        return {k: v.mean().item() for k, v in result.items()}

    def predict(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Make a prediction provided a batch of samples."""
        return self.classifier(self.encoder(x))

    def _set_learning_phase(self, train: bool = False, finetune: bool = False):
        if train:
            self.classifier.train()
            if finetune:
                self.encoder.train()
            else:
                self.encoder.eval()
        else:
            self.classifier.eval()
            self.encoder.eval()

    def save_checkpoint(self, path: str, epoch: int, **kwargs):
        """Save model states."""
        if isinstance(self.encoder, DistributedDataParallel):
            encoder = self.encoder.module
            classifier = self.classifier.module
        else:
            encoder = self.encoder
            classifier = self.classifier

        ckpt = {
            'encoder': encoder.state_dict(),
            'classifier': classifier.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch
        }
        if kwargs:
            ckpt.update(kwargs)
        torch.save(ckpt, path)

    def load_model_from_checkpoint(self, path: str) -> int:
        """
        Loading model from a checkpoint. Be sure to have
        all modules properly initialized prior to executing this function.
        Returns the epoch of the checkpoint + 1.
        """
        device = torch.device(f'cuda:{self.local_rank}')
        ckpt = torch.load(path, map_location=device)

        if isinstance(self.encoder, DistributedDataParallel):
            self.encoder.module.load_state_dict(ckpt['encoder'])
            self.classifier.module.load_state_dict(ckpt['classifier'])
        else:
            self.encoder.load_state_dict(ckpt['encoder'])
            self.classifier.load_state_dict(ckpt['classifier'])

        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.move_optimizer_states(self.optimizer, device)

        if 'scheduler' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler'])
        
        return ckpt['epoch'] + 1
        



class ClassificationOld(Task):
    def __init__(self,
                 backbone: nn.Module,
                 classifier: nn.Module,
                 criterion: nn.Module,
                 finetune: bool = False):
        super(Classification, self).__init__()

        self.backbone = backbone
        self.classifier = classifier
        self.criterion = criterion
        self.finetune = finetune

        self.amp_scaler: GradScaler = None
        self.optimizer: torch.optim.Optimizer = None
        self.scheduler: WarmupCosineDecayLR = None

        self.prepared = False

    def prepare(self,  # TODO: remove default parameters
                ckpt_dir: str,
                optimizer: str = 'lars',
                learning_rate: float = 1.0,
                weight_decay: float = 0.0,
                cosine_warmup: int = 0,
                epochs: int = 100,
                batch_size: int = 256,
                num_workers: int = 4,
                distributed: bool = False,
                local_rank: int = 0,
                world_size: int = 1,
                mixed_precision: bool = False,
                logger: logging.Logger = None):
        """Prepare training."""

        # Set attributes
        self.ckpt_dir:        str = ckpt_dir         # pylint: disable=attribute-defined-outside-init
        self.learning_rate: float = learning_rate    # pylint: disable=attribute-defined-outside-init
        self.weight_decay:  float = weight_decay     # pylint: disable=attribute-defined-outside-init
        self.cosine_warmup:   int = cosine_warmup    # pylint: disable=attribute-defined-outside-init
        self.epochs:          int = epochs           # pylint: disable=attribute-defined-outside-init
        self.batch_size:      int = batch_size       # pylint: disable=attribute-defined-outside-init
        self.num_workers:     int = num_workers      # pylint: disable=attribute-defined-outside-init
        self.distributed:    bool = distributed      # pylint: disable=attribute-defined-outside-init
        self.local_rank:      int = local_rank       # pylint: disable=attribute-defined-outside-init
        self.world_size:      int = world_size       # pylint: disable=attribute-defined-outside-init
        self.mixed_precision: int = mixed_precision  # pylint: disable=attribute-defined-outside-init
        self.logger:       object = logger

        # Warp the encoder & classifier with `DistributedDataParallel` which
        # automatically synchronizes parameter gradients when `.step()`
        # is called on the optimizer.
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

        # Optimizer; SGD+momentum, AdamW, or LARS
        self.optimizer = get_optimizer(
            params=[
                {'params': self.backbone.parameters()},
                {'params': self.classifier.parameters()},
            ],
            name=optimizer,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Learning rate scheduler; linear warmup + cosine decay
        self.scheduler = WarmupCosineDecayLR(
            optimizer=self.optimizer,
            total_epochs=self.epochs,
            warmup_epochs=self.cosine_warmup,
            warmup_start_lr=1e-4,
            min_decay_lr=1e-4,
        )

        if self.criterion is not None:
            self.criterion.to(self.local_rank)
        else:
            raise NotImplementedError("Work in progress to support a default loss.")

        # Mixed precision training (optional, disabled by default)
        self.amp_scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None

        self.prepared = True

    @timer
    def run(self,
            dataset: torch.utils.data.Dataset,
            eval_set: torch.utils.data.Dataset,
            test_set: torch.utils.data.Dataset,
            save_every: int = 10):
        """Classification training and evaulation."""
        
        if not self.prepared:
            raise RuntimeError("Training not prepared.")

        # For distributed training, we must provide an explicit sampler to
        # avoid duplicates across devices. Each node (i.e., GPU) will be trained on
        # `len(dataset) // world_size' samples.
        sampler = DistributedSampler(dataset, shuffle=True) if self.distributed else None
        shuffle = not self.distributed
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  sampler=sampler,
                                  shuffle=shuffle,
                                  num_workers=self.num_workers,
                                  drop_last=True,
                                  pin_memory=True,
                                  persistent_workers=True)
        eval_loader = DataLoader(eval_set,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers,
                                 drop_last=False,
                                 pin_memory=True,
                                 persistent_workers=True,)

        best_eval_acc: float = 0.
        best_epoch:      int = 0

        # Iterations of epochs start from 1, for simplicity.
        for epoch in range(1, self.epochs + 1):
            
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)

            # Train & evaluate
            train_history: dict = self.train(train_loader)
            eval_history:  dict = self.evaluate(eval_loader)
            history = dict(train=train_history, eval=eval_history)

            # Logging (https://wandb.ai)
            if self.local_rank == 0:
                if self.scheduler is not None:
                    history['misc'] = dict(lr=self.scheduler.get_last_lr()[0])
                wandb.log(data=history, step=epoch)
                        
            # Logging (terminal)
            if self.logger is not None:
                msg_train: str = make_epoch_message(train_history, epoch, self.epochs, best_epoch)
                msg_eval:  str = make_epoch_message(eval_history, epoch, self.epochs, best_epoch)
                self.logger.info(f"(Train) {msg_train}")
                self.logger.info(f"( Eval) {msg_eval} ")

            # Save intermediate model checkpoints
            if (epoch % save_every == 0) & (self.local_rank == 0):
                ckpt = os.path.join(self.ckpt_dir, f"ckpt.{epoch}.pth.tar")
                self.save_checkpoint(ckpt, epoch=epoch)
                
            # Save best model checkpoint
            eval_acc = eval_history['top@1']
            if eval_acc <= best_eval_acc:
                best_eval_loss = eval_acc
                best_epoch = epoch
                if self.local_rank == 0:
                    ckpt = os.path.join(self.ckpt_dir, f"ckpt.best.pth.tar")
                    self.save_checkpoint(ckpt, epoch=epoch)

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

        # Save final model checkpoint
        ckpt = os.path.join(self.ckpt_dir, f"ckpt.final.pth.tar")
        self.save_checkpoint(ckpt, epoch=epoch)
        
        # Test (optional when `test_set` is provided)
        if (self.local_rank == 0) and (test_set is not None):
            # Instantiate test loader
            test_loader = DataLoader(test_set,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     num_workers=self.num_workers,
                                     drop_last=False,
                                     pin_memory=False,
                                     persistent_workers=False)
            # Last model (last epoch)
            last_history = self.evaluate(test_loader)
            if self.logger is not None:
                last_msg = " | ".join([f"{k}:{v:.4f}" for k, v in last_history.items()])
                last_msg = "(Final) : " + last_msg
                self.logger.info(last_msg)
            # Best model (with highest validation accuracy)
            self.load_model_from_checkpoint(os.path.join(self.ckpt_dir, "ckpt.best.pth.tar"))
            best_history = self.evaluate(test_loader)
            if self.logger is not None:
                best_msg = " | ".join([f"{k} : {v:.4f}" for k, v in best_history.items()])
                best_msg = "( Best) : " + best_msg
                self.logger.info(best_msg)

            wandb.log({
                'test.top@1': wandb.Table(
                    columns=['last', 'best'],
                    data=[[last_history['top@1'], best_history['top@1']]]
                ),
            })

    def train(self, data_loader: DataLoader) -> typing.Dict[str,torch.FloatTensor]:
        """Iterates over the `data_loader' once for training."""

        steps = len(data_loader)
        self._set_learning_phase(train=self.finetune)
        result = {
            'loss': torch.zeros(steps, device=self.local_rank),
            'top@1': torch.zeros(steps, device=self.local_rank),
        }

        with get_rich_pbar(transient=True,
                           auto_refresh=False,
                           disable=self.local_rank != 0) as pbar:
            job = pbar.add_task(f":thread:", total=steps)
            
            for i, batch in enumerate(data_loader):
                
                # Forward pass
                with torch.cuda.amp.autocast(self.mixed_precision):
                    # Get data & assign device
                    x = batch['x'].to(self.local_rank)
                    y = batch['y'].to(self.local_rank)
                    # Compute logits & loss
                    logits = self.predict(x)
                    loss = self.criterion(logits, y)
                
                # Backpropagation
                if self.amp_scaler is not None:
                    self.amp_scaler.scale(loss).backward()
                    self.amp_scaler.step(self.optimizer)
                    self.amp_scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Update progress bar
                result['loss'][i] = loss.detach()
                result['top@1'][i] = TopKAccuracy(k=1)(logits, y).detach()
                msg = " | ".join([f"{k} : {v[:i+1].mean():.4f}" for k, v in result.items()])
                msg = f":thread:(Rank: {self.local_rank}) [{i+1}/{steps}]: " + msg
                pbar.update(job, advance=1., description=msg)
                pbar.refresh()

        return {k: v.mean().item() for k, v in result.items()}

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> typing.Dict[str,torch.FloatTensor]:
        """Iterates over the `data_loader' once for evaluation."""

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
            loss = self.criterion(logits, y)
            result['loss'][i] = loss
            result['top@1'][i] = TopKAccuracy(k=1)(logits, y)

        return {k: v.mean().item() for k, v in result.items()}

    def predict(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Make a prediction provided a batch of samples."""
        return self.classifier(self.backbone(x))

    def _set_learning_phase(self, train: bool = False):
        if train:
            self.backbone.train()
            self.classifier.train()
        else:
            self.backbone.eval()
            self.classifier.eval()

    def save_checkpoint(self, path: str, **kwargs):
        """Save model states."""
        ckpt = {
            'backbone': self.backbone.state_dict(),
            'classifier': self.classifier.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        if kwargs:
            ckpt.update(kwargs)
        torch.save(ckpt, path)

    def load_model_from_checkpoint(self, path: str):
        """Load model states."""
        ckpt = torch.load(path)
        self.backbone.load_state_dict(ckpt['backbone'])
        self.classifier.load_state_dict(ckpt['classifier'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt['scheduler'])
