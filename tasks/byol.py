# -*- coding: utf-8 -*-

import os
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from tasks.base import Task
from utils.metrics import TopKAccuracy
from utils.knn import KNNEvaluator
from utils.optimization import get_optimizer
from utils.optimization import get_cosine_scheduler
from utils.logging import get_rich_pbar


class BYOLLoss(nn.Module):
    def __init__(self):
        super(BYOLLoss, self).__init__()
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = F.normalize(a, dim=1)
        b = F.normalize(b, dim=1)
        return 2 - 2 * (a * b).sum(dim=1)  # shape; (B, )


class BYOL(Task):
    def __init__(self,
                 encoder: nn.Module,
                 projector: nn.Module,
                 predictor: nn.Module,
                 loss_function: nn.Module,
                ):
        super(BYOL, self).__init__()

        self.online_net = nn.Sequential()
        self.online_net.add_module('encoder', encoder)
        self.online_net.add_module('projector', projector)
        self.online_predictor = predictor

        self.target_net = copy.deepcopy(self.online_net)
        self._freeze_target_net_params()

        if isinstance(loss_function, BYOLLoss):
            self.loss_function = loss_function
        else:
            raise ValueError

        self.scaler = None
        self.optimizer = None
        self.scheduler = None
        self.writer = None

        self.prepared = False
    
    def prepare(self,
                ckpt_dir: str,
                optimizer: str = 'lars',
                learning_rate: float = 0.2,
                weight_decay: float = 1.5 * 1e-6,
                cosine_warmup: int = 10,
                cosine_cycles: int = 1,
                cosine_min_lr: float = 0.,
                epochs: int = 1000,
                batch_size: int = 256,
                num_workers: int = 0,
                distributed: bool = False,
                local_rank: int = 0,
                mixed_precision: bool = True,
                resume: str = None):
        """Prepare BYOL pre-training."""

        # Set attributes
        self.ckpt_dir = ckpt_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed = distributed
        self.local_rank = local_rank
        self.mixed_precision = mixed_precision
        self.resume = resume

        self.optimizer = get_optimizer(
            params=[
                {'params': self.online_net.parameters()},
                {'params': self.online_predictor.parameters()},
            ],
            name=optimizer,
            lr=learning_rate,
            weight_decay=weight_decay  # TODO: remove params from batch norm
        )

        self.scheduler = get_cosine_scheduler(
            self.optimizer,
            epochs=self.epochs,
            warmup_steps=cosine_warmup,
            cycles=cosine_cycles,
            min_lr=cosine_min_lr,
        )

        # Resuming from previous checkpoint (optional)
        if resume is not None:
            if not os.path.exists(resume):
                raise FileNotFoundError
            self.load_model_from_checkpoint(resume)

        # Distributed training (optional, disabled by default.)
        if distributed:
            self.online_net = DistributedDataParallel(
                module=self.online_net.to(local_rank),
                device_ids=[local_rank]
            )
            self.online_predictor = DistributedDataParallel(
                module=self.online_predictor.to(local_rank),
                device_ids=[local_rank]
            )
        else:
            self.online_net.to(local_rank)
            self.online_predictor.to(local_rank)

        # No DDP wrapping for target network; no gradient updates
        self.target_net.to(local_rank)

        # Mixed precision training (optional, enabled by default)
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

        # TensorBoard
        self.writer = SummaryWriter(ckpt_dir) if local_rank == 0 else None

        # Ready to train
        self.prepared = True

    def run(self,
            dataset: torch.utils.data.Dataset,
            memory_set: torch.utils.data.Dataset = None,
            query_set: torch.utils.data.Dataset = None,
            save_every: int = 100,
            **kwargs):

        if not self.prepared:
            raise RuntimeError("Training not prepared.")
        
        # DataLoader (for self-supervised pre-training)
        sampler = DistributedSampler(dataset) if self.distributed else None
        shuffle = not self.distributed
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )

        if (memory_set is not None) and (query_set is not None):
            memory_loader = DataLoader(memory_set, batch_size=self.batch_size*2, num_workers=self.num_workers)
            query_loader = DataLoader(query_set, batch_size=self.batch_size*2)
            knn_eval = True
        else:
            query_loader = None
            memory_loader = None
            knn_eval = False

        # Logging
        logger = kwargs.get('logger', None)

        for epoch in range(1, self.epochs + 1):

            if self.distributed and (sampler is not None):
                sampler.set_epoch(epoch)

            # Train
            history = self.train(data_loader, current_epoch=epoch - 1)  # for loop starts with `epoch=1`.
            log = " | ".join([f"{k} : {v:.4f}" for k, v in history.items()])

            # Evaluate
            if (self.local_rank == 0) and knn_eval:
                knn_k = kwargs.get('knn_k', [5, 200])
                knn = KNNEvaluator(knn_k, num_classes=query_loader.dataset.num_classes)
                knn_scores = knn.evaluate(self.online_net,
                                          memory_loader=memory_loader,
                                          query_loader=query_loader)
                for k, score in knn_scores.items():
                    log += f" | knn@{k}: {score*100:.2f}%"
            else:
                knn_scores = None

            # Logging
            if logger is not None:
                logger.info(f"Epoch [{epoch:>4}/{self.epochs:>4}] - " + log)

            # TensorBoard
            if self.writer is not None:
                for k, v in history.items():
                    self.writer.add_scalar(k, v, global_step=epoch)
                if knn_scores is not None:
                    for k, score in knn_scores.items():
                        self.writer.add_scalar(f'knn@{k}', score, global_step=epoch)
                if self.scheduler is not None:
                    lr = self.scheduler.get_last_lr()[0]
                    self.writer.add_scalar('lr', lr, global_step=epoch)

            if (epoch % save_every == 0) & (self.local_rank == 0):
                ckpt = os.path.join(self.ckpt_dir, f"ckpt.{epoch}.pth.tar")
                self.save_checkpoint(ckpt, epoch=epoch, history=history)

            if self.scheduler is not None:
                self.scheduler.step()

    def train(self, data_loader: torch.utils.data.DataLoader, current_epoch: int):
        """BYOL training of an epoch."""
        
        steps = len(data_loader)
        self._set_learning_phase(train=True)
        result = {
            'loss': torch.zeros(steps, device=self.local_rank)
        }

        with get_rich_pbar(transient=True, auto_refresh=False) as pg:
            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Waiting... ", total=steps)

            for i, batch in enumerate(data_loader):
                gamma = self.gamma_by_step(current_step=current_epoch * steps + i,
                                           total_steps=self.epochs * steps,
                                           base=0.996)
                loss = self.train_step(batch, gamma=gamma)
                result['loss'][i] = loss.detach()
                
                if self.local_rank == 0:
                    desc = f"[bold green] [{i+1}/{steps}]: "
                    for k, v in result.items():
                        desc += f" {k} : {v[:i+1].mean():.4f} |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()
        
        return {k: v.mean().item() for k, v in result.items()}
    
    def train_step(self, batch: dict, gamma: float = 0.996):
        """A single forward & backward pass."""
        with torch.cuda.amp.autocast(self.mixed_precision):

            # Two views    
            x1 = batch['x1'].to(self.local_rank)
            x2 = batch['x2'].to(self.local_rank)

            # Online projections
            z1_onl = self.online_net(x1)
            z2_onl = self.online_net(x2)
            
            # Target projections
            with torch.no_grad():
                z1_tgt = self.target_net(x1).detach()
                z2_tgt = self.target_net(x2).detach()

            loss = \
                self.loss_function(self.online_predictor(z1_onl), z2_tgt) + \
                self.loss_function(self.online_predictor(z2_onl), z1_tgt)
            loss = loss.mean()

            # Backpropagate & update
            self.backprop(loss)

            # Update target network
            total_steps = self.epochs 
            self._ema_update_target_net_params(gamma=gamma)
        
        return loss

    def backprop(self, loss: torch.FloatTensor):
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad()

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
    def _ema_update_target_net_params(self, gamma: float):
        for p_onl, p_tgt in zip(self.online_net.parameters(), self.target_net.parameters()):
            p_tgt.data = p_tgt.data * gamma + p_onl.data * (1. - gamma)

    @staticmethod
    def gamma_by_step(current_step: int, total_steps: int, base: float = 0.996):
        """Gradually increases ema update parameter."""
        progress = current_step / total_steps
        return 1 - (1 - base) * math.cos(math.pi * progress) / 2

    def save_checkpoint(self, path: str, **kwargs):
        """Save model to a `.tar` checkpoint file."""
        if self.distributed:
            encoder = self.online_net.module.encoder
            projector = self.online_net.module.projector
            predictor = self.online_predictor.module  # XXX: check
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

    def load_model_from_checkpoint(self, path: str):
        """
        Loading model from checkpoint.
        If resuming training, ensure that all modules have been properly initialized.
        For distributed training, call this function before using DataDistributedParallel.
        """
        ckpt = torch.load(path, map_location='cpu')
        self.online_net.encoder.load_state_dict(ckpt['encoder'])
        self.online_net.projector.load_state_dict(ckpt['projector'])
        self.online_predictor.load_state_dict(ckpt['projector'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler'])
        self.move_optimizer_states(self.optimizer, self.local_rank)
