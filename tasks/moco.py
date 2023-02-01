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
from layers.batchnorm import SplitBatchNorm2d
from models.backbone import ResNetBackbone
from models.head import MLPHead

from utils.distributed import ForMoCo, concat_all_gather
from utils.metrics import TopKAccuracy
from utils.knn import KNNEvaluator
from utils.optimization import WarmupCosineDecayLR
from utils.optimization import configure_optimizer
from utils.logging import configure_logger
from utils.progress import configure_progress_bar
from utils.decorators import timer, suppress_logging_info


class MoCoLoss(nn.Module):
    def __init__(self, temperature: float = 0.2):
        super(MoCoLoss, self).__init__()
        self.temperature = temperature

    def forward(self,
                query: torch.FloatTensor,
                key: torch.FloatTensor,
                negatives: torch.FloatTensor,
                ) -> typing.Tuple[torch.FloatTensor]:

        # Compute temperature-scaled similarities
        pos_logits = torch.einsum('nc,nc->n', [query, key]).view(-1, 1)              # (B, 1  )
        neg_logits = torch.einsum('nc,ck->nk', [query, negatives.clone().detach()])  # (B,   K)
        logits = torch.cat([pos_logits, neg_logits], dim=1)                          # (B, 1+K)
        logits.div_(self.temperature)

        # Compute instance discrimination loss
        targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = nn.functional.cross_entropy(logits, targets, reduction='mean')

        return loss, logits



class SupMoCoAttractLoss(nn.Module):
    def __init__(self, temperature: float = 0.2, n: int = 1, loss_weight: float = 1.0):
        super(SupMoCoAttractLoss, self).__init__()
        self.temperature: float = temperature
        self.n: int = n
        self.loss_weight = loss_weight
    
    def forward(self, query: torch.FloatTensor, key: torch.FloatTensor, negatives: torch.FloatTensor,
                query_labels: torch.LongTensor, negative_labels: torch.LongTensor) -> typing.Tuple[torch.Tensor]:
        """..."""
        
        # 1. Temperature-scaled similarities
        logits_pos = torch.einsum('nc,nc->n', *[query, key]).view(-1, 1)
        logits_neg = torch.einsum('nc,ck->nk', *[query, negatives.clone().detach()])
        logits = torch.cat([logits_pos, logits_neg], dim=1).div(self.temperature)
        
        # 2. Instance discrimination loss
        loss_inst = F.cross_entropy(
            input=logits,
            target=torch.zeros(logits.size(0), dtype=torch.long, device=logits.device),
            reduction='mean'
        )
        
        # 3. Attraction loss
        label_match = query_labels.view(-1, 1).eq(negative_labels.view(1, -1))
        if isinstance(self.n, int) and (self.n >= 1):
            sup_pos_idx = torch.multinomial(label_match.float(), num_samples=self.n)
            sup_pos_mask = torch.zeros_like(label_match).scatter(dim=1, index=sup_pos_idx, value=1.).bool()
        else:
            sup_pos_mask = label_match
        loss_attr = -1 * F.log_softmax(logits[:, 1:], dim=1).masked_select(sup_pos_mask).mean()

        return loss_inst + self.loss_weight * loss_attr, logits, sup_pos_mask


class SupMoCoEliminateLoss(nn.Module):
    def __init__(self, temperature: float = 0.2):
        super(SupMoCoEliminateLoss, self).__init__()
        self.temperature = temperature

    def forward(self, query: torch.FloatTensor, key: torch.FloatTensor, negatives: torch.FloatTensor,
                query_labels: torch.LongTensor, negative_labels: torch.LongTensor) -> typing.Tuple[torch.Tensor]:
        """..."""

        # 1. Temperature-scaled similarities
        logits_pos = torch.einsum('nc,nc->n', *[query, key]).view(-1, 1)
        logits_pos.div_(self.temperature)
        logits_neg = torch.einsum('nc,ck->nk', *[query, negatives.clone().detach()])
        logits_neg.div_(self.temperature)
        logits = torch.cat([logits_pos, logits_neg], dim=1)

        # 2. Identify false negatives w/ true target labels
        label_match = query_labels.view(-1, 1).eq(negative_labels.view(1, -1))

        # 3. Instance discrimination loss w/ false negatives eliminated
        loss = - (
            logits_pos - torch.log(
                torch.cat([logits_pos.exp(),
                           logits_neg.exp().masked_fill(mask=label_match, value=0.)],
                           dim=1).sum(dim=1, keepdim=True)
            )
        )
        loss = loss.mean()  # (1, ) <- (B, 1)

        return loss, logits, label_match


class SupMoCoLoss(MoCoLoss):
    def __init__(self, temperature: float = 0.2, n: int = 1, ignore_index: int = -1):
        super(SupMoCoLoss, self).__init__(temperature)
        self.n = n
        self.ignore_index = ignore_index

    def forward(self,
                query: torch.FloatTensor,
                key: torch.FloatTensor,
                negatives: torch.FloatTensor,
                query_labels: torch.LongTensor,
                negative_labels: torch.LongTensor) -> typing.Tuple[torch.FloatTensor,
                                                                   torch.FloatTensor,
                                                                   torch.FloatTensor,
                                                                   torch.BoolTensor]:
        """MoCo loss + Supervised positive selection loss (experimental)."""

        loss_moco, logits = super().forward(query, key, negatives)
        match = query_labels.view(-1, 1).eq(negative_labels.view(1, -1))                     # (B, K); 1 if class matches, 0 otherwise
        num_match_per_row = match.sum(dim=1, keepdim=True)                                   # (B, 1); number of perfect matches for each anchor
        if num_match_per_row.lt(self.n).any():
            raise ValueError(f"Insufficient matches per row (<{self.n}).")                   # TODO: a workaround

        pos_idx = torch.multinomial(match.float(), num_samples=self.n)                       # (B, n); randomly sample positives
        pos_mask = torch.zeros_like(match).scatter(dim=1, index=pos_idx, value=1.).bool()    # (B, K); mask version of `select_index`
        loss_sup = -1. * F.log_softmax(logits[:, 1:], dim=1).masked_select(pos_mask).mean()  # (1,  ); average negative log-likelihood

        return loss_moco, logits, loss_sup, pos_mask

    def attract(self, query, key, negatives, query_labels, negative_labels) -> typing.Tuple[torch.Tensor]:
        """..."""
        loss_moco, logits = super().forward(query, key, negatives)
        match = query_labels.view(-1, 1).eq(negative_labels.view(1, -1))                           # (B, K) <- (B, 1) x (1, K)
        sup_pos_idx = torch.multinomial(match.float(), num_samples=self.n)                         # (B, n)
        sup_pos_mask = torch.zeros_like(match).scatter(dim=1, index=sup_pos_idx, value=1.).bool()  # (B, K)
        loss_sup = -1.0 * F.log_softmax(logits[:, 1:], dim=1).masked_select(sup_pos_mask).mean()   # (1,  )

        return loss_moco, logits, loss_sup, sup_pos_mask


class MemoryQueue(nn.Module):
    def __init__(self, size: tuple):
        super(MemoryQueue, self).__init__()

        if len(size) != 2:
            raise ValueError(f"Invalid size for memory queue: {size}. Only supports 2D.")
        self.size = size

        self.register_buffer('buffer', F.normalize(torch.randn(*self.size), dim=0))
        self.register_buffer('ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('num_updates', torch.zeros(1, dtype=torch.long))
        self.register_buffer('indices', torch.zeros(self.size[1], dtype=torch.long))
        self.register_buffer('labels', torch.zeros(self.size[1], dtype=torch.long))
        self.register_buffer('is_reliable', torch.zeros(1, dtype=torch.bool))

    @property
    def num_negatives(self):
        return self.size[1]

    @property
    def features(self):
        return self.buffer

    @torch.no_grad()
    def update(self,
               keys: torch.FloatTensor,
               indices: torch.LongTensor = None,
               labels: torch.LongTensor = None) -> None:
        """
        Update memory queue shared along processes.
        Arguments:
            keys: torch.FloatTensor of shape (B, f)
            indices: torch.LongTensor of shape (B, )
            labels: torch.LongTensor of shape (B, )
        """

        if indices is not None:
            assert len(keys) == len(indices), print(keys.shape, indices.shape)

        if labels is not None:
            assert len(keys) == len(labels), print(keys.shape, labels.shape)

        # Gather along multiple processes.
        keys = concat_all_gather(keys)  # (B, f) -> (world_size * B, f)
        incoming, _ = keys.size()
        if self.num_negatives % incoming != 0:
            raise ValueError("Use exponentials of 2 for number of negatives.")

        # Update queue of keys
        ptr = int(self.ptr[0])
        self.buffer[:, ptr: ptr + incoming] = keys.T  # (f, B)

        # Update queue of labels
        if labels is not None:
            labels = concat_all_gather(labels)    # (B, ) -> (world_size * B, )
            self.labels[ptr: ptr + incoming] = labels

        # Update queue of indices
        if indices is not None:
            indices = concat_all_gather(indices)  # (B, ) -> (world_size * B, )
            self.indices[ptr: ptr + incoming] = indices

        # Check if the current queue is reliable
        if not self.is_reliable:
            self.is_reliable[0] = (ptr + incoming) >= self.num_negatives

        # Update pointer
        ptr = (ptr + incoming) % self.num_negatives
        self.ptr[0] = ptr
        self.num_updates[0] += 1


class MoCo(Task):
    """MoCo trainer."""
    def __init__(self, config: object, local_rank: int):
        super(MoCo, self).__init__()

        self.config:  object = config
        self.local_rank: int = local_rank

        self._init_logger()
        self._init_modules()
        self._init_cuda()
        self._init_optimization()
        self._init_criterion()
        self._resume_training_from_checkpoint()

    def _init_logger(self) -> None:
        """
        For distributed training, logging is only performed on a single process
        to avoid uninformative duplicates.
        """
        if self.local_rank == 0:
            logfile = os.path.join(self.config.checkpoint_dir, 'main.log')
            self.logger = configure_logger(logfile=logfile)
            self.logger.info(f'Checkpoint directory: {self.config.checkpoint_dir}')
        else:
            self.logger = None

    def _init_modules(self) -> None:
        """
        Initializes the following modules:
            1) query network (self.net_q)
            2) key network (self.net_k)
            3) memory queue (self.queue)
        """
        encoder = ResNetBackbone(name=self.config.backbone_type, data=self.config.data, in_channels=3)
        head    = MLPHead(encoder.out_channels, self.config.projector_dim)
        if not self.config.distributed:
            # Ghost Norm; https://arxiv.org/abs/1705.0874
            encoder = SplitBatchNorm2d.convert_split_batchnorm(encoder)

        self.net_q = nn.Sequential()
        self.net_q.add_module('encoder', encoder)
        self.net_q.add_module('head', head)
        self.net_k = copy.deepcopy(self.net_q)
        self.freeze_params(self.net_k)
        self.queue = MemoryQueue(size=(self.net_k.head.num_features, self.config.num_negatives))

        if self.logger is not None:
            self.logger.info(f"Encoder ({self.config.backbone_type}): {encoder.num_parameters:,}")
            self.logger.info(f"Head ({self.config.projector_type}): {head.num_parameters:,}")

    def _init_cuda(self) -> None:
        """
        1) Assigns cuda devices to modules.
        2) Wraps query network with `DistributedDataParallel`.
        """
        if self.config.distributed:
            self.net_q = DistributedDataParallel(
                module=self.net_q.to(self.local_rank),
                device_ids=[self.local_rank],
                bucket_cap_mb=100)
        else:
            self.net_q.to(self.local_rank)
        self.net_k.to(self.local_rank)
        self.queue.to(self.local_rank)

    def _init_criterion(self) -> None:
        self.criterion = MoCoLoss(self.config.temperature)

    def _init_optimization(self) -> None:
        """
        1) optimizer: {SGD, LARS}
        2) learning rate scheduler: linear warmup + cosine decay
        3) float16 training (optional)
        """
        self.optimizer = configure_optimizer(params=self.net_q.parameters(),
                                             name=self.config.optimizer,
                                             lr=self.config.learning_rate,
                                             weight_decay=self.config.weight_decay)
        self.scheduler = WarmupCosineDecayLR(optimizer=self.optimizer,
                                             total_epochs=self.config.epochs,
                                             warmup_epochs=self.config.lr_warmup,
                                             warmup_start_lr=1e-4,
                                             min_decay_lr=1e-4)
        self.amp_scaler = GradScaler() if self.config.mixed_precision else None

    def _resume_training_from_checkpoint(self) -> None:
        """
        Resume training from a previous checkpoint if
        a valid path is provided to the `resume` argument.
        """
        if self.config.resume is not None:
            if os.path.exists(self.config.resume):
                self.start_epoch = self.load_model_from_checkpoint(
                    self.config.resume, self.local_rank
                )
                if self.logger is not None:
                    self.logger.info("Successfully loaded model from checkpoint. "
                                     f"Resuming from epoch = {self.start_epoch}")
            else:
                if self.logger is not None:
                    self.logger.warn("Invalid checkpoint. Starting from epoch = 1")
                self.start_epoch = 1
        else:
            if self.logger is not None:
                self.logger.info(f"No checkpoint provided. Starting from epoch = 1")
            self.start_epoch = 1

    @timer
    def run(self, train_set: torch.utils.data.Dataset,
                  memory_set: torch.utils.data.Dataset,
                  test_set: torch.utils.data.Dataset,
                  **kwargs):
        """Training and evaluation."""

        if self.logger is not None:
            self.logger.info(f"Data: {train_set.__class__.__name__}")
            self.logger.info(f"Number of training examples: {len(train_set):,}")

        if self.config.distributed:
            # For distributed training, we must provide an explicit sampler to
            # avoid duplicates across devices. Each node (i.e., GPU) will be
            # trained on `len(train_set) // world_size' samples.
            sampler = DistributedSampler(train_set, shuffle=True)
        else:
            sampler = None
        shuffle: bool = sampler is None
        data_loader = DataLoader(train_set,
                                 batch_size=self.config.batch_size,
                                 sampler=sampler,
                                 shuffle=shuffle,
                                 num_workers=self.config.num_workers,
                                 drop_last=True,
                                 pin_memory=True,
                                 persistent_workers=self.config.num_workers > 0)

        if self.local_rank == 0:
            # Intermediate evaluation of representations based on nearest neighbors.
            # The frequency is controlled by the `eval_every' argument of this function.
            eval_loader_config = dict(batch_size=self.config.batch_size * self.config.world_size,
                                      num_workers=self.config.num_workers * self.config.world_size,
                                      pin_memory=True,
                                      persistent_workers=True)
            memory_loader = DataLoader(memory_set, **eval_loader_config)
            test_loader   = DataLoader(test_set, **eval_loader_config)
            knn_evaluator = KNNEvaluator(num_neighbors=[5, 200], num_classes=train_set.num_classes)
        else:
            memory_loader = None
            test_loader   = None
            knn_evaluator = None

        for epoch in range(1, self.config.epochs + 1):

            if epoch < self.start_epoch:
                if self.logger is not None:
                    self.logger.info(f"Skipping epoch {epoch} (< {self.start_epoch})")
                continue

            # The `epoch` attribute of the `sampler` is accessed when iterated.
            # Refer to the `__iter__` function of `DistributedSampler` for further information.
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)

            # A single epoch of training
            train_history = self.train(data_loader, epoch=epoch)

            # Evaluate the learned representations every `eval_every` epochs, on rank 0.
            if (knn_evaluator is not None) & (epoch % self.config.eval_every == 0):
                with torch.cuda.amp.autocast():
                    knn_scores: dict = knn_evaluator.evaluate(
                        net=self.net_q.module.encoder if self.config.distributed else self.net_q.encoder,
                        memory_loader=memory_loader,
                        test_loader=test_loader
                    )
            else:
                knn_scores = None

            # Logging; https://wandb.ai
            log = dict()
            log.update(train_history)
            if isinstance(knn_scores, dict):
                log.update({f'eval/knn@{k}': v for k, v in knn_scores.items()})
            if self.scheduler is not None:
                log.update({'misc/lr': self.scheduler.get_last_lr()[0]})
            if self.local_rank == 0:
                wandb.log(data=log, step=epoch)

            def maxlen_fmt(total: int) -> str:
                return f">0{int(math.log10(total))+1}d"

            def history_to_log_message(history: dict, step: int, total: int, fmt: str = ">04d"):
                msg: str = " | ".join([f"{k} : {v:.4f}" for k, v in history.items()])
                return f"Epoch [{step:{fmt}}/{total:{fmt}}] - " + msg

            # Logging; terminal
            if self.logger is not None:
                fmt = maxlen_fmt(self.config.epochs)
                msg = history_to_log_message(log, epoch, self.config.epochs, fmt)
                self.logger.info(msg)

            # Save intermediate model checkpoints
            if (self.local_rank == 0) & (epoch % self.config.save_every == 0):
                fmt = maxlen_fmt(self.config.epochs)
                ckpt = os.path.join(self.config.checkpoint_dir, f"ckpt.{epoch:{fmt}}.pth.tar")
                self.save_checkpoint(ckpt, epoch=epoch, history=log)

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

    @suppress_logging_info
    def train(self, data_loader: DataLoader, **kwargs) -> typing.Dict[str, float]:
        """Iterates over the `data_loader` once for MoCo training."""

        steps: int = len(data_loader)
        self._set_learning_phase(train=True)
        metrics = {
            'train/loss': torch.zeros(steps, device=self.local_rank),
            'train/rank@1': torch.zeros(steps, device=self.local_rank),
        }

        with configure_progress_bar(transient=True,
                                    auto_refresh=False,
                                    disable=self.local_rank != 0) as pbar:
            job = pbar.add_task(f":thread:", total=steps)
            for i, batch in enumerate(data_loader):
                # Single batch iteration
                loss, rank = self.train_step(batch)
                metrics['train/loss'][i]   = loss.detach()
                metrics['train/rank@1'][i] = rank.detach()
                # Update progress bar
                msg = f':thread: [{i+1}/{steps}]: ' + \
                    ' | '.join([f"{k} : {self.nanmean(v[:i+1]).item():.4f}" for k, v in metrics.items()])
                pbar.update(job, advance=1., description=msg)
                pbar.refresh()

        return {k: self.nanmean(v).item() for k, v in metrics.items()}

    def train_step(self, batch: dict) -> typing.Tuple[torch.FloatTensor]:
        """A single forward & backward pass using a batch of examples."""

        with torch.cuda.amp.autocast(self.amp_scaler is not None):
            # Fetch two positive views; {query, key}
            x_q = batch['x1'].to(self.local_rank, non_blocking=True)
            x_k = batch['x2'].to(self.local_rank, non_blocking=True)
            # Compute query features; (B, f)
            z_q = F.normalize(self.net_q(x_q), dim=1)
            with torch.no_grad():
                # An exponential moving average update of the key network
                self._momentum_update_key_net()
                # Shuffle across devices (GPUs)
                x_k, idx_unshuffle = ForMoCo.batch_shuffle_ddp(x_k)
                # Compute key features; (B, f)
                z_k = F.normalize(self.net_k(x_k), dim=1)
                # Restore key features to their original devices
                z_k = ForMoCo.batch_unshuffle_ddp(z_k, idx_unshuffle)
            # Compute loss & metrics
            loss, logits = self.criterion(z_q, z_k, self.queue.buffer)
            y = batch['y'].to(self.local_rank).detach()
            rank = TopKAccuracy(k=1)(logits.detach(), torch.zeros_like(y))
            # Backpropagate & update
            self.backprop(loss)
            # Update memory queue
            self.queue.update(keys=z_k,
                              indices=batch['idx'].to(self.local_rank),
                              labels=y)

        return loss, rank

    def backprop(self, loss: torch.FloatTensor) -> None:
        """SGD parameter update, optionally with float16 training."""
        if self.amp_scaler is not None:
            self.amp_scaler.scale(loss).backward()
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad()

    def _set_learning_phase(self, train: bool = False) -> None:
        if train:
            self.net_q.train()
            self.net_k.train()
        else:
            self.net_q.eval()
            self.net_k.eval()

    @torch.no_grad()
    def _momentum_update_key_net(self) -> None:
        for p_q, p_k in zip(self.net_q.parameters(), self.net_k.parameters()):
            p_k.data = p_k.data * self.config.key_momentum + p_q.data * (1. - self.config.key_momentum)

    def save_checkpoint(self, path: str, epoch: int, **kwargs) -> None:
        """Save model to a `.tar' checkpoint file."""
        if isinstance(self.net_q, DistributedDataParallel):
            encoder = self.net_q.module.encoder
            head = self.net_q.module.head
        else:
            encoder = self.net_q.encoder
            head = self.net_q.head

        ckpt = {
            'encoder': encoder.state_dict(),
            'head': head.state_dict(),
            'net_k': self.net_k.state_dict(),
            'queue': self.queue.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch,
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

        # Load query network
        if isinstance(self.net_q, DistributedDataParallel):
            self.net_q.module.encoder.load_state_dict(ckpt['encoder'])
            self.net_q.module.head.load_state_dict(ckpt['head'])
        else:
            self.net_q.encoder.load_state_dict(ckpt['encoder'])
            self.net_q.head.load_state_dict(ckpt['head'])

        # Load key network
        self.net_k.load_state_dict(ckpt['net_k'])

        # Load memory queue
        self.queue.load_state_dict(ckpt['queue'])

        # Load optimizer states
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.move_optimizer_states(self.optimizer, device)

        # Load scheduler states
        if 'scheduler' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler'])

        return ckpt['epoch'] + 1

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


class SupMoCoBase(MoCo):
    def __init__(self, config: object, local_rank: int):
        super(SupMoCoBase, self).__init__(config=config, local_rank=local_rank)

    def _init_criterion(self) -> None:
        raise NotImplementedError

    @suppress_logging_info
    def train(self, data_loader: DataLoader, **kwargs) -> typing.Dict[str, float]:
        """Iterates over the `data_loader` once for SupMoCo training."""
        
        steps = len(data_loader)
        self._set_learning_phase(train=True)
        metrics = {
            'loss': torch.zeros(steps, device=self.local_rank),
            'rank@1': torch.zeros(steps, device=self.local_rank),
            'precision': torch.zeros(steps, device=self.local_rank),
        }

        with configure_progress_bar(transient=True, auto_refresh=False,
                                    disable=self.local_rank != 0) as pbar:
            job = pbar.add_task(f":thread:", total=steps)
            for i, batch in enumerate(data_loader):
                # Single batch update
                loss, rank, precision = self.train_step(batch)
                metrics['loss'][i] = loss.detach()
                metrics['rank@1'][i] = rank.detach()
                metrics['precision'][i] = precision.detach()
                # Update progress bar
                msg = f":thread:[{i+1}/{steps}]: " + \
                    " | ".join([f"{k} : {self.nanmean(v[:i+1]).item():.4f}" for k, v in metrics.items()])
                pbar.update(job, advance=1., description=msg)
                pbar.refresh()

        return {k: self.nanmean(v).item() for k, v in metrics.items()}

    def train_step(self, batch: dict) -> typing.Tuple[torch.Tensor]:
        """..."""
        
        with torch.cuda.amp.autocast(enabled=self.amp_scaler is not None):
        
            # Fetch two positive views; {query, key}
            x_q = batch['x1'].to(self.local_rank, non_blocking=True)
            x_k = batch['x2'].to(self.local_rank, non_blocking=True)
        
            # Compute query features; (B, f)
            z_q = F.normalize(self.net_q(x_q), dim=1)
        
            with torch.no_grad():
                self._momentum_update_key_net()
                # Compute key features; (B, f)
                x_k, idx_unshuffle = ForMoCo.batch_shuffle_ddp(x_k)
                z_k = F.normalize(self.net_k(x_k), dim=1)
                z_k = ForMoCo.batch_unshuffle_ddp(z_k, idx_unshuffle)

            # Compute loss & metrics
            y = batch['y'].to(self.local_rank).detach()
            if self.queue.is_reliable:
                loss, logits, label_pos_mask = self.criterion(query=z_q,
                                                              key=z_k,
                                                              negatives=self.queue.features,
                                                              query_labels=y,
                                                              negative_labels=self.queue.labels)
            else:
                loss, logits = MoCoLoss(self.criterion.temperature)(query=z_q, key=z_k,
                                                                    negatives=self.queue.features)
                label_pos_mask = torch.zeros_like(logits[:, 1:], dtype=torch.bool)
            rank = TopKAccuracy(k=1)(logits.detach(), torch.zeros_like(y))
            precision = self.teacher_precision(labels_batch=y, 
                                               labels_queue=self.queue.labels,
                                               mask_teacher=label_pos_mask)
            
            # Backpropagate & update
            self.backprop(loss)

            # Update memory queue
            self.queue.update(keys=z_k, indices=batch['idx'].to(self.local_rank), labels=y)

        return loss, rank, precision

            
    @staticmethod
    def teacher_precision(labels_batch: torch.LongTensor,
                          labels_queue: torch.LongTensor,
                          mask_teacher: torch.BoolTensor) -> torch.FloatTensor:
        """Compute precision of teacher's predictions."""
        with torch.no_grad():
            labels_batch = labels_batch.view(-1, 1)           # (B,  ) -> (B, 1)
            labels_queue = labels_queue.view(1, -1)           # (K,  ) -> (1, K)
            is_true_positive = labels_batch.eq(labels_queue)  # (B, 1) @ (1, K) -> (B, K)
            num_true_positive = is_true_positive.masked_select(mask_teacher).sum()
            return torch.true_divide(num_true_positive, mask_teacher.sum())


class SupMoCoAttract(SupMoCoBase):
    def __init__(self, config: object, local_rank: int):
        super(SupMoCoAttract, self).__init__(config=config, local_rank=local_rank)

    def _init_criterion(self) -> None:
        self.criterion = SupMoCoAttractLoss(
            temperature=self.config.temperature,
            n=self.config.num_positives,
            loss_weight=self.config.loss_weight,
        )


class SupMoCoEliminate(SupMoCoBase):
    def __init__(self, config: object, local_rank: int):
        super(SupMoCoEliminate, self).__init__(config=config, local_rank=local_rank)
    
    def _init_criterion(self) -> None:
        self.criterion = SupMoCoEliminateLoss(temperature=self.config.temperature)


class SupMoCo(MoCo):  # TODO: remove as deprecated
    def __init__(self, config: object, local_rank: int):
        self.loss_weight: float = config.loss_weight    # TODO
        self.num_positives: int = config.num_positives  # FIXME
        super(SupMoCo, self).__init__(config, local_rank)

    def _init_criterion(self):
        self.criterion = SupMoCoLoss(temperature=self.config.temperature,
                                     n=self.num_positives)

    @suppress_logging_info
    def train(self, data_loader: DataLoader) -> typing.Dict[str, float]:
        """Iterates over the `data_loader` once for SupMoCo training."""

        steps = len(data_loader)
        self._set_learning_phase(train=True)
        metrics = {
            'loss':      torch.zeros(steps, device=self.local_rank),
            'loss_sup':  torch.zeros(steps, device=self.local_rank),
            'rank@1':    torch.zeros(steps, device=self.local_rank),
            'precision': torch.zeros(steps, device=self.local_rank),
        }

        with configure_progress_bar(transient=True,
                                    auto_refresh=False,
                                    disable=self.local_rank != 0) as pbar:
            job = pbar.add_task(f":thread:", total=steps)

            for i, batch in enumerate(data_loader):
                # Single batch iteration
                loss, loss_sup, rank, precision = self.train_step(batch)
                metrics['loss'][i]      = loss.detach()
                metrics['loss_sup'][i]  = loss_sup.detach()
                metrics['rank@1'][i]    = rank.detach()
                metrics['precision'][i] = precision.detach()

                # Update progress bar
                msg = f':thread:[{i+1}/{steps}]: ' + \
                    ' | '.join([f"{k} : {self.nanmean(v[:i+1]).item():.4f}" for k, v in metrics.items()])
                pbar.update(job, advance=1., description=msg)
                pbar.refresh()

        return {k: self.nanmean(v).item() for k, v in metrics.items()}

    def train_step(self, batch: dict) -> typing.Tuple[torch.FloatTensor,
                                                      torch.FloatTensor,
                                                      torch.LongTensor,
                                                      torch.FloatTensor]:
        """A single forward & backward pass using a batch of examples."""
        with torch.cuda.amp.autocast(self.amp_scaler is not None):
            # Fetch two positive view; {query, key}
            x_q = batch['x1'].to(self.local_rank, non_blocking=True)
            x_k = batch['x2'].to(self.local_rank, non_blocking=True)
            # Compute query features; (B, f)
            z_q = F.normalize(self.net_q(x_q), dim=1)
            with torch.no_grad():
                self._momentum_update_key_net()
                x_k, idx_unshuffle = ForMoCo.batch_shuffle_ddp(x_k)
                z_k = F.normalize(self.net_k(x_k), dim=1)
                z_k = ForMoCo.batch_unshuffle_ddp(z_k, idx_unshuffle)
            # Compute loss & metrics
            y = batch['y'].to(self.local_rank).detach()
            if self.queue.is_reliable:
                loss, logits, loss_sup, mask_sup = self.criterion(query=z_q,
                                                                  key=z_k,
                                                                  negatives=self.queue.features,
                                                                  query_labels=y,
                                                                  negative_labels=self.queue.labels)
            else:
                loss, logits = MoCoLoss(self.criterion.temperature)(query=z_q,
                                                                    key=z_k,
                                                                    negatives=self.queue.features)
                loss_sup = torch.zeros(1, device=loss.device)
                mask_sup = torch.zeros_like(logits[:, 1:], dtype=torch.bool)
            rank = TopKAccuracy(k=1)(logits.detach(), torch.zeros_like(y))
            precision = self.teacher_precision(labels_batch=y,
                                               labels_queue=self.queue.labels,
                                               mask_teacher=mask_sup)
            # Backpropagate & update
            self.backprop(loss + self.loss_weight * loss_sup)
            # Update memory queue
            self.queue.update(keys=z_k,
                              indices=batch['idx'].to(self.local_rank),
                              labels=y)

        return loss, loss_sup, rank, precision

    @staticmethod
    def teacher_precision(labels_batch: torch.LongTensor,
                          labels_queue: torch.LongTensor,
                          mask_teacher: torch.BoolTensor) -> torch.FloatTensor:
        """Compute precision of teacher's predictions."""
        with torch.no_grad():
            labels_batch = labels_batch.view(-1, 1)           # (B,  ) -> (B, 1)
            labels_queue = labels_queue.view(1, -1)           # (K,  ) -> (1, K)
            is_true_positive = labels_batch.eq(labels_queue)  # (B, 1) @ (1, K) -> (B, K)
            num_true_positive = is_true_positive.masked_select(mask_teacher).sum()
            return torch.true_divide(num_true_positive, mask_teacher.sum())

