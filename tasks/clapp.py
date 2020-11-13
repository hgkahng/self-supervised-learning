# -*- coding: utf-8 -*-

import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from entmax import sparsemax

from tasks.base import Task
from utils.distributed import ForMoCo
from utils.metrics import TopKAccuracy
from utils.knn import KNNEvaluator
from utils.optimization import get_optimizer
from utils.optimization import get_cosine_scheduler
from utils.logging import get_rich_pbar


class CLAPPLoss(nn.Module):
    def __init__(self,
                 temperature: float,
                 pseudo_temperature: float,
                 normalize: str = 'softmax',
                 contrast_mode: str = 'queue',
                 ):
        super(CLAPPLoss, self).__init__()

        self.temperature = temperature
        self.pseudo_temperature = pseudo_temperature
        self.normalize = normalize
        self.contrast_mode = contrast_mode

    def forward(self,
                query: torch.FloatTensor,
                pseudo: torch.FloatTensor,
                key: torch.FloatTensor,
                negatives: torch.FloatTensor,
                threshold: float = 0.95):
        """
        1.Compute logits between:
            a) query vs. {key, queue}.
            b) pseudo vs. {queue}.
        2. Compute loss as:
            c) cross entropy of deterministic positives.
            d) cross entropy of pseudo positives. Pseudo labels are learned by b).
        """

        # Clone memory queue, to avoid unintended inplace operations
        negatives = negatives.clone().detach()

        # a & c
        logits_pos = torch.einsum('bf,bf->b', [query, key]).view(-1, 1)  # (B, 1)
        logits_neg = torch.einsum('bf,fk->bk', [query, negatives])       # (B, K)
        logits     = torch.cat([logits_pos, logits_neg], dim=1)          # (B, 1+K)
        logits.div_(self.temperature)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)

        # b & d
        logits_pseudo_neg = torch.einsum('bf,fk->bk', [pseudo, negatives])  # (B, K)

        # if no pseudo-positives selected -> loss_pseudo = torch[nan]; check nan later.
        if self.contrast_mode == 'queue':
            logits_pseudo_neg.div_(self.pseudo_temperature)
            loss_pseudo, mask_pseudo = self._pseudo_loss_against_queue(logits, logits_pseudo_neg, threshold)
            return loss, logits, labels, loss_pseudo, mask_pseudo, None
        
        elif self.contrast_mode == 'queue-old':
            logits_pseudo_neg.div_(self.pseudo_temperature)
            probs_pseudo_neg = self._normalize(logits_pseudo_neg, dim=0)
            loss_pseudo = self._pseudo_loss_against_queue_v2(logits[:, 1:], probs_pseudo_neg, threshold)
            return loss, logits, labels, loss_pseudo, probs_pseudo_neg, None

        elif self.contrast_mode == 'batch':
            logits_pseudo_neg.div_(self.pseudo_temperature)
            probs_pseudo_neg = self._normalize(logits_pseudo_neg, dim=0)
            loss_pseudo = self._pseudo_loss_against_batch(logits[:, 1:], probs_pseudo_neg, threshold)
            return loss, logits, labels, loss_pseudo, probs_pseudo_neg, None

    def _normalize(self, x: torch.FloatTensor, dim: int = 0):
        if self.normalize == 'softmax':
            return F.softmax(x, dim=dim)
        elif self.normalize == 'sparsemax':
            return sparsemax(x, dim=dim)
        else:
            raise NotImplementedError

    @staticmethod
    def masked_softmax(x: torch.FloatTensor, m: torch.BoolTensor):
        x_ = x.masked_fill(~m, float('-inf'))
        return F.softmax(x_, dim=-1)

    def _pseudo_loss_against_queue(self,
                                   logits: torch.FloatTensor,
                                   logits_pseudo_neg: torch.FloatTensor,
                                   threshold: float = 0.9):
        """
        Numerator = exp{ anchor@pseudo / t }
        Denominator = exp{ anchor@pos / t } + sum[ exp{ anchor@queue } ].
        """
        b, k = logits_pseudo_neg.size()
        frac = 100 / k
        
        mask_pseudo = torch.zeros_like(logits_pseudo_neg)
        for _ in range(10):
            random_mask = torch.rand_like(logits_pseudo_neg).le(frac)
            probs_pseudo = self.masked_softmax(logits_pseudo_neg, random_mask)
            mask_pseudo += probs_pseudo.ge(threshold).float()
        
        mask_pseudo = mask_pseudo.bool()
        num_pseudo_per_anchor = mask_pseudo.sum(dim=1, keepdim=True)

        nll = -1. * F.log_softmax(logits, dim=1).div(num_pseudo_per_anchor)
        nll = nll[:, 1:].masked_select(mask_pseudo).mean()

        return nll, mask_pseudo  # (1, ) or tensor(nan)

    def _pseudo_loss_against_queue_v2(self,
                                      logits_neg: torch.FloatTensor,
                                      probs_pseudo_neg: torch.FloatTensor,
                                      threshold: float = 0.95):
        mask_pseudo_neg = probs_pseudo_neg.ge(threshold)                  # (B, K)
        num_pseudo_per_anchor = mask_pseudo_neg.sum(dim=1, keepdim=True)  # (B, 1)
        log_numerator = logits_neg.clone()                                # (B, K)
        denominator = logits_neg.exp().sum(dim=1, keepdim=True)           # (B, 1)
        log_prob = log_numerator - torch.log(denominator)                 # (B, K)
        loss_pseudo = torch.neg(log_prob)                                 # (B, K)
        loss_pseudo = loss_pseudo.div(num_pseudo_per_anchor + 1)
        loss_pseudo = loss_pseudo.masked_select(mask_pseudo_neg).mean()
        return loss_pseudo

    def _pseudo_loss_against_batch(self,
                                   logits_neg: torch.FloatTensor,
                                   probs_pseudo_neg: torch.FloatTensor,
                                   threshold: float = 0.95):
        """
        Numerator = exp{ anchor@pseudo / t }
        Denominator = sum[ exp{ batch@pseudo } ].
        """
        max_prob, max_idx = probs_pseudo_neg.max(dim=0)
        loss_pseudo = F.cross_entropy(logits_neg.T, max_idx, reduction='none')  # (K, B), (K, ) -> (K, )
        loss_pseudo = loss_pseudo.masked_select(max_prob.ge(threshold)).mean()  # (k,  ); k << K
        return loss_pseudo  # (1, ) or tensor(nan)


class CLAPP(Task):
    def __init__(self,
                 encoder: nn.Module,
                 head: nn.Module,
                 queue: nn.Module,
                 loss_function: nn.Module,
                 ):
        super(CLAPP, self).__init__()

        # Query network (backprop-trained)
        self.net_q = nn.Sequential()
        self.net_q.add_module('encoder', encoder)
        self.net_q.add_module('head', head)

        # Pseudo network (momentum-updated)
        self.net_ps = copy.deepcopy(self.net_q)
        self.freeze_params(self.net_ps)

        # Key network (momentum-updated)
        self.net_k = copy.deepcopy(self.net_ps)
        self.freeze_params(self.net_k)

        self.queue = queue
        self.loss_function = loss_function

        self.scaler = None
        self.optimizer = None
        self.scheduler = None
        self.writer = None
        self.prepared = False

    def prepare(self,
                ckpt_dir: str,
                optimizer: str,
                learning_rate: float,
                weight_decay: float,
                cosine_warmup: int = 0,
                epochs: int = 200,
                batch_size: int = 256,
                num_workers: int = 4,
                key_momentum: float = 0.999,
                pseudo_momentum: float = 0.5,
                threshold: float = 0.95,
                distributed: bool = False,
                local_rank: int = 0,
                mixed_precision: bool = True,
                resume: str = None):

         # Set attributes
        self.ckpt_dir = ckpt_dir                # pylint: disable=attribute-defined-outside-init
        self.epochs = epochs                    # pylint: disable=attribute-defined-outside-init
        self.batch_size = batch_size            # pylint: disable=attribute-defined-outside-init
        self.num_workers = num_workers          # pylint: disable=attribute-defined-outside-init
        self.key_momentum = key_momentum        # pylint: disable=attribute-defined-outside-init
        self.pseudo_momentum = pseudo_momentum  # pylint: disable=attribute-defined-outside-init
        self.threshold = threshold              # pylint: disable=attribute-defined-outside-init
        self.distributed = distributed          # pylint: disable=attribute-defined-outside-init
        self.local_rank = local_rank            # pylint: disable=attribute-defined-outside-init
        self.mixed_precision = mixed_precision  # pylint: disable=attribute-defined-outside-init
        self.resume = resume                    # pylint: disable=attribute-defined-outside-init

        # Intialize optimizer
        self.optimizer = get_optimizer(
            params=self.net_q.parameters(),
            name=optimizer,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        # LR scheduling (if cosine_warmup < 0: scheduler = None)
        self.scheduler = get_cosine_scheduler(
            self.optimizer,
            epochs=epochs,
            warmup_steps=cosine_warmup
        )

        # Resume from previous checkpoint (if `resume' is not None)
        if resume is not None:
            if not os.path.exists(resume):
                raise FileNotFoundError
            self.load_model_from_checkpoint(resume)

        # Distributed training
        if distributed:
            self.net_q = DistributedDataParallel(
                module=self.net_q.to(local_rank),
                device_ids=[local_rank]
            )
        else:
            self.net_q.to(local_rank)

        # NO DDP wrapping for {pseudo, key} encoders; no gradients
        self.net_ps.to(local_rank)
        self.net_k.to(local_rank)

        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

        # TensorBoard
        self.writer = SummaryWriter(ckpt_dir) if local_rank == 0 else None

        self.prepared = True


    def run(self,
            train_set: torch.utils.data.Dataset,
            memory_set: torch.utils.data.Dataset = None,
            query_set: torch.utils.data.Dataset = None,
            save_every: int = 100,
            **kwargs):

        if not self.prepared:
            raise RuntimeError("CLAPP training not prepared.")

        # Logging
        logger = kwargs.get('logger', None)

        # Data loader for model training
        sampler = DistributedSampler(train_set) if self.distributed else None
        shuffle = not self.distributed
        train_loader = DataLoader(train_set,
                                  batch_size=self.batch_size,
                                  sampler=sampler,
                                  shuffle=shuffle,
                                  num_workers=self.num_workers,
                                  drop_last=True,
                                  pin_memory=True)

        # Data loader for knn-based evaluation
        if (self.local_rank == 0) and (memory_set is not None) and (query_set is not None):
            memory_loader = DataLoader(memory_set, batch_size=self.batch_size * 2, num_workers=4)
            query_loader = DataLoader(query_set, batch_size=self.batch_size * 2)
            knn_eval = True
        else:
            memory_loader = None
            query_loader = None
            knn_eval = False

        # Train for several epochs
        for epoch in range(1, self.epochs + 1):
            # Shuffling index for distributed training
            if self.distributed and (sampler is not None):
                sampler.set_epoch(epoch)
            # Train
            history = self.train(train_loader)
            log = " | ".join([f"{k} : {v:.3f}" for k, v in history.items()])
            # Evaluate
            if (self.local_rank == 0) and knn_eval:
                knn = KNNEvaluator(5, num_classes=memory_loader.dataset.num_classes)
                knn_score = knn.evaluate(self.net_q, memory_loader=memory_loader, query_loader=query_loader)
                log += f" | knn@5: {knn_score*100:.2f}%"
            else:
                knn_score = None
            # Terminal
            if logger is not None:
                logger.info(f"Epoch [{epoch:>4}/{self.epochs:>4}] - " + log)
            # TensorBoard
            if self.writer is not None:
                for k, v in history.items():
                    self.writer.add_scalar(k, v, global_step=epoch)
                if knn_score is not None:
                    self.writer.add_scalar('knn@5', knn_score, global_step=epoch)
                if self.scheduler is not None:
                    lr = self.scheduler.get_last_lr()[0]
                    self.writer.add_scalar('lr', lr, global_step=epoch)
            # Save model checkpoint
            if (epoch % save_every == 0) & (self.local_rank == 0):
                ckpt = os.path.join(self.ckpt_dir, f"ckpt.{epoch}.pth.tar")
                self.save_checkpoint(ckpt, epoch=epoch, history=history)
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

    def train(self, data_loader: DataLoader):
        """CLAPP training."""

        steps = len(data_loader)
        self._set_learning_phase(train=True)
        result = {
            'loss': torch.zeros(steps, device=self.local_rank),
            'loss_pseudo': torch.zeros(steps, device=self.local_rank),
            'rank@1': torch.zeros(steps, device=self.local_rank),
            'num_correct': torch.zeros(steps, device=self.local_rank),
            'num_pseudo': torch.zeros(steps, device=self.local_rank),
            'precision': torch.zeros(steps, device=self.local_rank),
        }

        with get_rich_pbar(transient=True, auto_refresh=False) as pg:
            # Add progress bar
            if self.local_rank == 0:
                task = pg.add_task(f"[bold green] CLAPP...", total=steps)
            for i, batch in enumerate(data_loader):
                # Single batch iteration
                batch_history = self.train_step(batch)
                # Accumulate metrics
                for name in result.keys():
                    result[name][i] = batch_history[name]
                # Update progress bar
                if self.local_rank == 0:
                    desc = f"[bold green] [{i+1}/{steps}]: "
                    for name, val in result.items():
                        desc += f" {name}: {self.nanmean(val[:i+1]).item():.3f} |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

        return {k: self.nanmean(v).item() for k, v in result.items()}

    def train_step(self, batch: dict):
        """A single forward & backward pass."""
        with torch.cuda.amp.autocast(self.mixed_precision):
            # Get data (3 views)
            x_q  = batch['x1'].to(self.local_rank)
            x_k  = batch['x2'].to(self.local_rank)
            x_ps = batch['x3'].to(self.local_rank)
            # Compute strong query features; (B, f)
            z_q = F.normalize(self.net_q(x_q), dim=1)

            with torch.no_grad():
                # Update momentum {pseudo, key} networks
                self._momentum_update_key_net()
                self._momentum_update_pseudo_net()
                # Shuffle across nodes (gpus)
                x_k, idx_unshuffle_k = ForMoCo.batch_shuffle_ddp(x_k)
                x_ps, idx_unshuffle_ps = ForMoCo.batch_shuffle_ddp(x_ps)
                # Compute {key, pseudo} features; (B, f)
                z_k  = F.normalize(self.net_k(x_k), dim=1)
                z_ps = F.normalize(self.net_ps(x_ps), dim=1)
                # Restore {key, pseudo} features to their original nodes
                z_k  = ForMoCo.batch_unshuffle_ddp(z_k, idx_unshuffle_k)
                z_ps = ForMoCo.batch_unshuffle_ddp(z_ps, idx_unshuffle_ps)

            # Compute loss
            loss, logits, labels, loss_pseudo, probs_pseudo_neg, _ = \
                self.loss_function(z_q, z_ps, z_k, self.queue.buffer, threshold=self.threshold)
            # Backpropagate & update
            if loss_pseudo.isnan():
                self.backprop(loss)
            else:
                alpha = 1.0
                self.backprop(loss + alpha * loss_pseudo)
            # Compute metrics
            with torch.no_grad():
                # Accuracy of true positives against all negatives
                rank_1 = TopKAccuracy(k=1)(logits, labels)
                # Accuracy of pseudo positives with ground truth labels
                above_threshold = probs_pseudo_neg.ge(self.threshold)
                num_pseudo = above_threshold.sum()
                # No pseudo positives may have been selected
                if self.queue.is_reliable and (num_pseudo > 0):
                    labels_query = batch['y'].to(self.local_rank)                       # (B,  )
                    labels_queue = self.queue.labels                                    # (k,  )
                    is_correct = labels_query.view(-1, 1).eq(labels_queue.view(1, -1))  # (B, 1) @ (1, k) -> (B, k)
                    num_correct = is_correct.masked_select(above_threshold).sum()
                    precision = torch.true_divide(num_correct, num_pseudo)
                else:
                    num_correct = torch.zeros(1, dtype=torch.long, device=num_pseudo.device)
                    precision = torch.zeros(1, dtype=torch.float32, device=num_pseudo.device)
            # Update memory queue
            self.queue.update(keys=z_k, labels=batch['y'].to(self.local_rank))

        return {
            'loss': loss.detach(),
            'loss_pseudo': loss_pseudo.detach(),  # (1, ) or tensor(nan)
            'rank@1': rank_1,
            'num_correct': num_correct,
            'num_pseudo': num_pseudo,
            'precision': precision,
        }


    def backprop(self, loss: torch.FloatTensor):
        """SGD parameter update, optionally with mixed precision."""
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad()

    def _set_learning_phase(self, train: bool = False):
        """Set mode of networks (train or eval)."""
        if train:
            self.net_q.train()
            self.net_ps.train()
            self.net_k.train()
        else:
            self.net_q.eval()
            self.ent_ps.eval()
            self.net_k.eval()

    @torch.no_grad()
    def _momentum_update_pseudo_net(self):
        alpha = self.pseudo_momentum
        for q, ps in zip(self.net_q.parameters(), self.net_ps.parameters()):
            ps.data = ps.data * alpha + q.data * (1. - alpha)

    @torch.no_grad()
    def _momentum_update_key_net(self):
        alpha = self.key_momentum
        for q, k in zip(self.net_q.parameters(), self.net_k.parameters()):
            k.data = k.data * alpha + q.data * (1. - alpha)

    def save_checkpoint(self, path: str, **kwargs):
        """Save model to a `.tar' checkpoint file."""
        if self.distributed:
            encoder = self.net_q.module.encoder
            head = self.net_q.module.head
        else:
            encoder = self.net_q.encoder
            head = self.net_q.head

        ckpt = {
            'encoder': encoder.state_dict(),
            'head': head.state_dict(),
            'net_ps': self.net_ps.state_dict(),
            'net_k': self.net_k.state_dict(),
            'queue': self.queue.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        if kwargs:
            ckpt.update(kwargs)
        torch.save(ckpt, path)

    def load_model_from_checkpoint(self, path: str):
        """
        Loading model from a checkpoint.
        Make sure to call this function before assigning models to GPU.
        """
        ckpt = torch.load(path, map_location='cpu')
        self.net_q.encoder.load_state_dict(ckpt['encoder'])
        self.net_q.head.load_state_dict(ckpt['head'])
        self.net_ps.load_state_dict(ckpt['net_ps'])
        self.net_k.load_state_dict(ckpt['net_k'])
        self.queue.load_state_dict(ckpt['queue'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            self.scheduler.load_stae_dict(ckpt['scheduler'])
        self.move_optimizer_states(self.optimizer, self.local_rank)

    @staticmethod
    def nanmean(x: torch.FloatTensor):
        """Compute mean excluding NaNs."""
        nan_mask = torch.isnan(x)
        denominator = (~nan_mask).sum()
        if denominator.eq(0):
            return torch.full((1, ), fill_value=float('nan'), device=x.device)
        else:
            numerator = x[~nan_mask].sum()
            return torch.true_divide(numerator, denominator)

    @staticmethod
    def masked_mean(x: torch.FloatTensor, m: torch.BoolTensor):
        """Compute mean for where the values of `m` = 1."""
        if m.bool().sum() == len(m):
            return torch.full((1, ), fill_value=float('inf'), device=x.device)
        return x[m.bool()].mean()