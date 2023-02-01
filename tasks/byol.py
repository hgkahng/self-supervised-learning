# -*- coding: utf-8 -*-

import os
import copy
import math
import typing

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp.grad_scaler import GradScaler

from tasks.base import Task
from models.backbone import ResNetBackbone
from models.head import BYOLProjectionHead, BYOLPredictionHead

from utils.optimization import get_optimizer, WarmupCosineDecayLR
from utils.metrics import TopKAccuracy
from utils.knn import KNNEvaluator
from utils.logging import get_rich_logger, get_rich_pbar
from utils.decorators import timer, suppress_logging_info


class BYOLLoss(nn.Module):
    def __init__(self, reduction: str = 'none'):
        super(BYOLLoss, self).__init__()
        self.reduction: str = reduction
    
    def forward(self, a: torch.FloatTensor, b: torch.FloatTensor) -> torch.FloatTensor:
        a = F.normalize(a, dim=1)
        b = F.normalize(b, dim=1)
        out = 2 - 2 * (a * b).sum(dim=1)  # shape; (B, )
        if self.reduction == 'none':
            return out
        elif self.reduction == 'sum':
            return out.sum()
        elif self.reduction == 'mean':
            return out.mean()
        else:
            raise ValueError


class BYOL(Task):
    """BYOL trainer."""
    def __init__(self, config: object, local_rank: int):
        super(BYOL, self).__init__()

        self.config: object = config
        self.local_rank: int = local_rank
        self.start_epoch: int = 1
        self.target_decay: float = config.target_decay_base

        self._init_logger()
        self._init_modules()
        self._init_cuda()
        self._init_criterion()
        self._init_optimization()
        self._resume_training_from_checkpoint()

    def _init_logger(self):
        """Add function docstring."""
        if self.local_rank == 0:
            logfile = os.path.join(self.config.checkpoint_dir, 'main.log')
            self.logger = get_rich_logger(logfile)
            self.logger.info(f"Checkpoint directory: {os.path.dirname(logfile)}")
        else:
            self.logger = None

    def _init_modules(self):
        """Add function docstring."""
        encoder = ResNetBackbone(name=self.config.backbone_type,
                                 data=self.config.data)
        projector = BYOLProjectionHead(in_channels=encoder.out_channels,
                                       hidden_size=self.config.projector_hid_dim,
                                       output_size=self.config.projector_out_dim)
        predictor = BYOLPredictionHead(input_size=self.config.projector_out_dim,
                                       hidden_size=self.config.projector_hid_dim,
                                       output_size=self.config.projector_out_dim,)
        
        self.online_net = nn.Sequential()
        self.online_net.add_module('encoder', encoder)
        self.online_net.add_module('projector', projector)
        self.target_net = copy.deepcopy(self.online_net)
        self.online_predictor = predictor
        self._freeze_target_net_params()

        if self.logger is not None:
            self.logger.info(f"Encoder ({self.config.backbone_type}): {encoder.num_parameters:,}")
            self.logger.info(f"Projector: {projector.num_parameters:,}")
            self.logger.info(f"Predictor: {predictor.num_parameters:,}")

    def _init_cuda(self):
        """Add function docstring."""
        if self.config.distributed:
            self.online_net = DistributedDataParallel(
                module=nn.SyncBatchNorm.convert_sync_batchnorm(self.online_net).to(self.local_rank),
                device_ids=[self.local_rank]
            )
            self.online_predictor = DistributedDataParallel(
                module=nn.SyncBatchNorm.convert_sync_batchnorm(self.online_predictor).to(self.local_rank),
                device_ids=[self.local_rank]
            )
        else:
            self.online_net.to(self.local_rank)
            self.online_predictor.to(self.local_rank)
        self.target_net.to(self.local_rank)
    
    def _init_criterion(self):
        """Add function docstring."""
        self.criterion = BYOLLoss()
    
    def _init_optimization(self):
        """Add function docstring."""
        self.optimizer = get_optimizer(
            params=[{'params': self.online_net.parameters()},
                    {'params': self.online_predictor.parameters()}],
            name=self.config.optimizer,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay)
        self.scheduler = WarmupCosineDecayLR(optimizer=self.optimizer,
                                             total_epochs=self.config.epochs,
                                             warmup_epochs=self.config.cosine_warmup,
                                             warmup_start_lr=1e-4,
                                             min_decay_lr=1e-4)
        self.amp_scaler = GradScaler() if self.config.mixed_precision else None

    def _resume_training_from_checkpoint(self):
        """Add function docstring."""
        if self.config.resume is not None:
            if os.path.exists(self.config.resume):
                self.start_epoch = self.load_model_from_checkpoint(
                    self.config.resume, self.local_rank)
            else:
                self.start_epoch = 1
        else:
            self.start_epoch = 1

    @timer
    def run(self,
            dataset: torch.utils.data.Dataset,
            finetune_set: torch.utils.data.Dataset,
            test_set: torch.utils.data.Dataset,
            save_every: typing.Optional[int] = 100,
            eval_every: typing.Optional[int] = 1):
        """Training & evaluation."""

        if self.logger is not None:
            self.logger.info(f"Data: {dataset.__class__.__name__}")
            self.logger.info(f"Number of training examples: {len(dataset):,}")

        if self.config.distributed:
            sampler = DistributedSampler(dataset, shuffle=True)
        else:
            sampler = None
        shuffle: bool = sampler is None
        data_loader = DataLoader(dataset,
                                 batch_size=self.config.batch_size,
                                 sampler=sampler,
                                 shuffle=shuffle,
                                 num_workers=self.config.num_workers,
                                 drop_last=True,
                                 pin_memory=True,
                                 persistent_workers=self.config.num_workers > 0)
        
        if self.local_rank == 0:
            eval_loader_config = dict(batch_size=self.config.batch_size * self.config.world_size,
                                      num_workers=self.config.num_workers * self.config.world_size,
                                      pin_memory=True,
                                      persistent_workers=self.config.num_workers > 0)
            finetune_loader = DataLoader(finetune_set, **eval_loader_config)
            test_loader     = DataLoader(test_set, **eval_loader_config)
            knn_evaluator   = KNNEvaluator(num_neighbors=[5, 200], num_classes=dataset.num_classes)
        else:
            finetune_loader = None
            test_loader     = None
            knn_evaluator   = None

        for epoch in range(1, self.config.epochs + 1):

            if epoch < self.start_epoch:
                if self.logger is not None:
                    self.logger.info(f"Skipping epoch {epoch} (< {self.start_epoch})")
                continue

            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)

            # A single epoch of training
            history = self.train(data_loader)

            # Intermediate evaluation of the quality of representations
            if (knn_evaluator is not None) & (epoch % eval_every == 0):
                knn_scores: dict = knn_evaluator.evaluate(self.online_net, finetune_loader, test_loader)  # FIXME
            else:
                knn_scores = None
            
            # Logging; https://wandb.ai
            history.update({'target_decay': self.target_decay})
            if self.scheduler is not None:
                history.update({'lr': self.scheduler.get_last_lr()[0]})
            if isinstance(knn_scores, dict):
                history.update({f'knn@{k}': v for k, v in knn_scores.items()})
            if self.local_rank == 0:
                wandb.log(data=history, step=epoch)

            def maxlen_fmt(total: int):
                return f">0{int(math.log10(total))+1}d"
            
            def history_to_log_message(history: dict, step: int, total: int, fmt: str = '>04d'):
                msg: str = " | ".join([f"{k} : {v:.4f}" for k, v in history.items()])
                return f"Epoch [{step:{fmt}}/{total:{fmt}}] - " + msg 
            
            # Logging; terminal
            if self.logger is not None:
                fmt = maxlen_fmt(self.config.epochs)
                msg = history_to_log_message(history, epoch, self.config.epochs, fmt)
                self.logger.info(msg)
            
            # Save intermediate model checkpoints
            if (self.local_rank == 0) & (epoch & save_every == 0):
                fmt = maxlen_fmt(self.config.epochs)
                ckpt = os.path.join(self.config.checkpoint_dir, f"ckpt.{epoch:{fmt}}.pth.tar")
                self.save_checkpoint(ckpt, epoch=epoch, history=history)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # Update target decay rate (gradually increased)
            self._update_target_decay_rate(step=epoch, total=self.config.epochs, base=self.config.target_decay_base)

    @suppress_logging_info
    def train(self, data_loader: DataLoader) -> typing.Dict[str, float]:
        """Iterates over the `data_loader` once for BYOL training."""
        
        steps = len(data_loader)
        self._set_learning_phase(train=True)
        metrics = {'loss': torch.zeros(steps, device=self.local_rank),}

        with get_rich_pbar(transient=True, auto_refresh=False, disable=self.local_rank != 0) as pbar:
            job = pbar.add_task(f":dizzy:", total=steps)

            for i, batch in enumerate(data_loader):
                # Single batch iteration
                loss = self.train_step(batch)
                metrics['loss'][i] = loss.detach()
                msg = f':dizzy: [{i+1}/{steps}]: ' + '| '.join(
                    [f"{k} : {self.nanmean(v[:i+1]):.4f}" for k, v in metrics.items()]
                )
                pbar.update(job, advance=1., description=msg)
                pbar.refresh()
        
        return {k: self.nanmean(v).item() for k, v in metrics.items()}
    
    def train_step(self, batch: dict) -> torch.FloatTensor:
        """
        A single forward & backward pass using a batch of examples.
        Symmetric training is enabled by default for data efficiency.
        """
        with torch.cuda.amp.autocast(self.amp_scaler is not None):
            # Fetch two positive views
            x1 = batch['x1'].to(self.local_rank, non_blocking=True)
            x2 = batch['x2'].to(self.local_rank,  non_blocking=True)
            # Forward computation
            z1_onl = self.online_net(x1)
            z2_onl = self.online_net(x2)
            with torch.no_grad():
                z1_tgt = self.target_net(x1)
                z2_tgt = self.target_net(x2)
            loss = self.criterion(self.online_predictor(z1_onl), z2_tgt.detach()) + \
                self.criterion(self.online_predictor(z2_onl), z1_tgt.detach())
            loss = loss.mean()
            # Backpropagate & update online network
            self.backprop(loss)
            # EMA update target network
            self._update_target_net_params(gamma=self.target_decay)
        
        return loss

    def backprop(self, loss: torch.FloatTensor):
        self.optimizer.zero_grad()
        if self.amp_scaler is not None:
            self.amp_scaler.scale(loss).backward()
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()
        else:
            loss.backward(retain_graph=True)
            self.optimizer.step()

    def _set_learning_phase(self, train: bool = False):
        if train:
            self.online_net.train()
            self.online_predictor.train()
            self.target_net.train()
        else:
            self.online_net.eval()
            self.online_predictor.eval()
            self.target_net.eval()
        
    @torch.no_grad()
    def _freeze_target_net_params(self):
        for p in self.target_net.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _update_target_net_params(self, gamma: float = 0.996):
        for p_onl, p_tgt in zip(self.online_net.parameters(), self.target_net.parameters()):
            p_tgt.data = p_tgt.data * gamma + p_onl.data * (1. - gamma)

    @torch.no_grad()
    def _update_target_decay_rate(self, step: int, total: int, base: float = 0.996):
        self.target_decay = 1 - (1 - base) * (math.cos(math.pi * step / total) + 1) * 0.5

    def save_checkpoint(self, path: str, **kwargs):
        """Save model to a `.tar` checkpoint file."""
        if self.config.distributed:
            encoder = self.online_net.module.encoder
            projector = self.online_net.module.projector
            predictor = self.online_predictor.module
        else:
            encoder = self.online_net.encoder
            projector = self.online_net.projector
            predictor = self.online_predictor

        ckpt = {
            'encoder': encoder.state_dict(),
            'projector': projector.state_dict(),
            'predictor': predictor.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        if kwargs:
            ckpt.update(kwargs)
        torch.save(ckpt, path)

    def load_model_from_checkpoint(self, path: str, local_rank: int = 0) -> int:
        """
        Loading model from a checkpoint. Be sure to have
        all modules properly initialized prior to executing this function.
        Returns the epoch of the checkpoint + 1 (used as `self.start_epoch`).
        """
        device = torch.device(f'cuda:{local_rank}')
        ckpt = torch.load(path, map_location=device)

        # Load online network
        if isinstance(self.online_net, DistributedDataParallel):
            self.online_net.module.encoder.load_state_dict(ckpt['encoder'])
            self.online_net.module.projector.load_state_dict(ckpt['projector'])
            self.online_predictor.module.load_state_dict(ckpt['predictor'])
        else:
            self.online_net.encoder.load_state_dict(ckpt['encoder'])
            self.online_net.projector.load_state_dict(ckpt['projector'])
            self.online_predictor.load_state_dict(ckpt['predictor'])
        
        # Load target network
        self.target_net.load_state_dict(ckpt['target_net'])

        # Load optimizer states
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.move_optimizer_states(self.optimizer, device)

        # Load scheduler states
        if 'scheduler' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler'])
        
        return ckpt['epoch'] + 1

    @staticmethod
    def nanmean(x: torch.FloatTensor) -> torch.FloatTensor:
        nan_mask = torch.isnan(x)
        denominator = (~nan_mask).sum()
        if denominator.eq(0):
            return torch.full((1, ), fill_value=float('nan'), device=x.device)
        else:
            numerator = x[~nan_mask].sum()
            return torch.true_divide(numerator, denominator)
