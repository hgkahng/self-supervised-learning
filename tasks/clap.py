
import os
import copy
import math
import typing
import warnings

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler

from tasks.moco import MoCo, MemoryQueue
from layers.batchnorm import SplitBatchNorm2d
from models.backbone import ResNetBackbone
from models.head import MLPHead, CLAPv2PredHead

from utils.distributed import ForMoCo
from utils.metrics import TopKAccuracy
from utils.logging import get_rich_pbar
from utils.decorators import suppress_logging_info
from utils.optimization import WarmupCosineDecayLR, get_optimizer, WarmupCosineDecayLRWithRestarts


class CLAPLoss(nn.Module):
    def __init__(self,
                 temperature: float = 0.2,
                 ensemble_num_estimators: int = 128,
                 ensemble_dropout_rate: float = 0.5,
                 easy_pos_ub: float = 0.5,
                 hard_pos_lb: float = 0.5,
                 num_positives: int = 1,
                 ):
        super(CLAPLoss, self).__init__()

        self.temperature:  float = temperature
        self.ensemble_num_estimators: int = ensemble_num_estimators
        self.ensemble_dropout_rate: float = ensemble_dropout_rate
        self.easy_pos_ub: float = easy_pos_ub
        self.hard_pos_lb: float = hard_pos_lb
        self.num_positives:  int = num_positives
        self.num_pos: int = self.num_positives
        # self.subsample_size: int = subsample_size  # FIXME: remove
        # self.num_repeats:    int = num_repeats     # FIXME: remove
        # self.normalize:      str = normalize       # FIXME: remove
        # self.aggregate:      str = aggregate      # FIXME: remove

    def forward(self, query: torch.FloatTensor,
                      key: torch.FloatTensor,
                      teacher: torch.FloatTensor,
                      negatives: torch.FloatTensor) -> typing.Tuple[torch.FloatTensor,
                                                                    torch.FloatTensor,
                                                                    torch.FloatTensor,
                                                                    torch.BoolTensor,
                                                                    torch.FloatTensor,
                                                                    torch.BoolTensor]:
        # store variables for easy reference
        bsize = int(query.size(0))
        # detach & clone memory queue to avoid unintended inplace operations
        negatives = negatives.detach().clone()
        # A) compute moco loss (known positive via data augmentation)
        logits_pos = torch.einsum('bf,bf->b', *[query, key]).view(-1, 1)
        logits_neg = torch.einsum('bf,fq->bq', *[query, negatives])
        logits = torch.cat([logits_pos, logits_neg], dim=1)
        logits.div_(self.temperature)
        moco_loss = F.cross_entropy(
            input=logits,
            target=torch.zeros(logits.size(0), dtype=torch.long, device=logits.device),
            reduction='mean'
        )
        
        # compute similarities between teacher and negative representations
        teacher_sim = self.compute_teacher_similarities(teacher, negatives)  # (E, B, Q); FIXME; add temperature
        teacher_sim = teacher_sim.div(self.tau)
        # compute InstDisc scores, used to detect positives
        scores = self.compute_scores(teacher_sim, num_samples=128)           # (B, Q) <- (E, B, Q);  # FIXME
        # compute uncertainties, used to filter out false negatives
        uncertainty = self.compute_uncertainties(teacher_sim)                # (1, Q)

        # B) easy positive loss; choose easy positive
        # candidates among negatives with low uncertainty (entropy)
        ub = torch.quantile(uncertainty, q=self.easy_pos_ub, dim=1)      # (1,  )
        easy_pos_mask = uncertainty.le(ub).repeat(bsize, 1)              # (B, Q)
        easy_scores = scores.masked_fill(~easy_pos_mask, float('-inf'))  # (B, Q)
        _, easy_idx = easy_scores.topk(k=self.num_pos, dim=1)            # (B, n), (B, n)
        easy_loss = -1. * F.log_softmax(logits[:, 1:], dim=1).gather(    # FIXME: temperature-scaled  vs. raw logits? 
            dim=1, index=easy_idx).mean(dim=1).mean()
        easy_mask = torch.zeros_like(easy_scores).scatter(1, easy_idx, 1.).bool()  # (B, Q); for metric calculation

        # C) hard positive loss; choose hard positive 
        # candidates among negatives with high uncertainty (entropy)
        lb = torch.quantile(uncertainty, q=self.hard_pos_lb, dim=1)      # (1,  )
        hard_pos_mask = uncertainty.ge(lb).repeat(bsize, 1)              # (B, Q)
        hard_scores = scores.masked_fill(~hard_pos_mask, float('-inf'))  # (B, Q)
        _, hard_idx = hard_scores.topk(k=self.num_pos, dim=1)            # (B, n), (B, n)
        hard_loss = -1. * F.log_softmax(logits[:, 1:], dim=1).gather(    # FIXME: temperature-scaled vs. raw logits?
            dim=1, index=hard_idx).mean(dim=1).mean()
        hard_mask = torch.zeros_like(hard_scores).scatter(1, hard_idx, 1.).bool()  # (B, Q); for metric calculation
        
        return moco_loss, logits, easy_loss, easy_mask, hard_loss, hard_mask

    def old_forward(self,
                    query: torch.FloatTensor,
                    key: torch.FloatTensor,
                    teacher: torch.FloatTensor,
                    negatives: torch.FloatTensor) -> typing.Tuple[torch.FloatTensor,
                                                                  torch.BoolTensor]:
        """
        1. Nearest Neighbor Ensembles
            - Randomly select features (choose dropout rate)
            - L2 normalize features
            - Bootstrap samples
            - Compute against-batch similarities (multi-gpu support?)
            - Aggregate similarities (mean, uncertainty)
        2. Positive Pool Selection
            - Easy positives tend to have high similarity (and low uncertainty)
            - Hard positives tend to have high uncertainty (and low similarity)
            - Specify thresholds or quantiles.
        """

        # Clone memory queue, to avoid unintended inplace operations
        negatives = negatives.clone().detach()

        # MoCo loss; InfoNCE
        logits_pos = torch.einsum('bf,bf->b', [query, key]).view(-1, 1)    # (B, 1  )
        logits_neg = torch.einsum('bf,fk->bk', [query, negatives])         # (B,   Q)
        logits = torch.cat([logits_pos, logits_neg], dim=1).div(self.tau)  # (B, 1+Q)
        loss = F.cross_entropy(
            input=logits,
            target=torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        )

        # Teacher loss
        logits_teacher = torch.einsum('bf,fk->bk', [teacher, negatives])   # (B,  Q)
        logits_teacher = logits_teacher.div(self.tau)  # FIXME: .div() unnecessary?
        loss_teacher, mask_teacher = self._teacher_loss_queue(logits, logits_teacher)

        return loss, logits, loss_teacher, mask_teacher

    @property
    def tau(self) -> float:
        return self.temperature  # FIXME: remove

    @torch.no_grad()
    def compute_teacher_similarities(self,
                                     teacher: torch.FloatTensor,
                                     negatives: torch.FloatTensor) -> torch.FloatTensor:
        """
        Computes an ensemble of similarites between teacher and
        and negative representations.
        Arguments:
            teacher: 2D ``torch.FloatTensor`` of shape (B, F).
            negatives: 2D ``torch.FloatTensor`` of shape (F, Q).
        Returns:
            similarities: 3D ``torch.FloatTensor`` of shape (E, B, Q), where
                ``E`` is the number of ensembles.
        """
        teacher = teacher.unsqueeze(0).repeat(self.ensemble_num_estimators, 1, 1)              # (E, B, F)
        teacher = self.random_maskout(teacher, p=self.ensemble_dropout_rate, val=0.)           # (E, B, F); FIXME
        teacher = F.normalize(teacher, dim=2)                                                  # (E, B, F); FIXME
        negatives = negatives.clone().unsqueeze(0).repeat(self.ensemble_num_estimators, 1, 1)  # (E, F, Q)
        negatives = self.random_maskout(negatives, p=self.ensemble_dropout_rate, val=0.)       # (E, F, Q); FIXME
        negatives = F.normalize(negatives, dim=1)                                              # (E, F, Q); FIXME
        return torch.einsum('ebf,efq->ebq', *[teacher, negatives])                             # (E, B, Q); FIXME

    @torch.no_grad()
    def compute_scores(self, sim: torch.FloatTensor, num_samples: int) -> torch.FloatTensor:
        """
        Computes InstDisc scores, normalized across the queue dimension.
        Arguments:
            sim: 3D FloatTensor with shape (E, B, Q).
            num_samples: int.
        Returns:
            scores: 2D FloatTensor of shape (B, Q)
        """
        sample_prob = num_samples / int(sim.size(2))
        mask = torch.rand_like(sim).le(sample_prob)     # (E, B, Q)
        scores = self.masked_softmax(sim, mask, dim=2)  # (E, B, Q)
        return self.masked_mean(scores, mask, dim=0)    #    (B, Q)

    @torch.no_grad()
    def compute_uncertainties(self, sim: torch.FloatTensor) -> torch.FloatTensor:
        """Computes uncertainty scores, normalized across the batch dimension."""
        mask = [self.bootstrap(torch.ones_like(s), dim=0) for _, s in enumerate(sim)]  # FIXME
        mask = torch.stack(mask, dim=0)                                      # (E, B, Q)
        entropy = self.masked_entropy_with_logits(sim, mask, softmax_dim=1)  # (E, Q)
        return entropy.mean(dim=0, keepdim=True)                             # (1, Q)

    @staticmethod
    def entropy_with_logits(logits: torch.FloatTensor, average_across: int = 1):
        raise NotImplementedError

    @staticmethod
    def entropy_with_probs(probs: torch.FloatTensor, average_across: int = 1):
        probs = probs.add(1e-7)
        entropy = - probs.mul(torch.log(probs))
        return entropy.mean(dim=average_across, keepdim=False)

    @staticmethod
    def masked_entropy_with_logits(logits: torch.FloatTensor,
                                   mask: torch.BoolTensor,
                                   softmax_dim: int = 1):
        """
        Arguments:
            logits: 3D ``torch.FloatTensor`` of shape (E, B, Q).
            mask: 3D ``torch.FloatTensor`` of shape (E, B, Q).
            softmax_dim: ``int``, dimension to compute probabilites and
                average entropy across.
        Return:
            2D ``torch.FloatTensor`` of shape (E, Q), containing entropy values.
        """
        logits = logits.clone().masked_fill(~mask, float('-inf'))  # (E, B, Q)
        probs = F.softmax(logits, dim=softmax_dim).add(1e-7)       # (E, B, Q)
        entropy = - probs.mul(probs.log())                         # (E, B, Q)
        count = mask.sum(dim=softmax_dim).clamp(min=1.0)           # (E, Q); if `softmax_dim` = 1
        return entropy.sum(dim=softmax_dim).div(count)             # (E, Q); if `softmax_dim` = 1

    @staticmethod
    def masked_entropy_with_probs(probs: torch.FloatTensor, mask: torch.BoolTensor, average_dim: int = 1):
        """
            probs: 3D ``torch.FloatTensor`` of shape (E, B, Q).
            m: 3D ``torch.BoolTensor`` of shape (E, B, Q).
            average_dim: ``int``, dimension to compute entropy across.
        Return:
            2D ``torch.FloatTensor`` of shape (E, Q), containing entropy values.
        """
        probs = probs.add(1e-7)
        entropy = - probs.mul(probs.log())
        count = mask.sum(dim=average_dim).clamp(min=1.0)
        return entropy.sum(dim=average_dim).div(count)

    @staticmethod
    def random_maskout(x: torch.FloatTensor, p: float = 0.2, val: float = 0.0) -> torch.FloatTensor:
        """Writes ``val`` to the tensor ``x`` with random probability of ``p``."""
        m = torch.rand_like(x).le(p)
        return x.masked_fill(m, val)

    @staticmethod
    def random_sample(w: torch.FloatTensor, num_samples: int, replacement: bool = False, dim: int = 0):
        if w.ndim != 2:
            raise ValueError(f"`w` expects 2D tensors, received {w.ndim}D.")
        if dim == 0:
            index = torch.multinomial(w.T, num_samples=num_samples, replacement=replacement)
            out = torch.zeros_like(w.T).scatter(dim=1, index=index, value=1.).T
        elif dim == 1:
            index = torch.multinomial(w, num_samples=num_samples, replacement=replacement)
            out = torch.zeros_like(w).scatter_(dim=1, index=index, value=1.)
        else:
            raise NotImplementedError
        return out.bool()

    @staticmethod  # FIXME: this is a special case of random sample with `replacement`=True
    def bootstrap(w: torch.FloatTensor, num_samples: int = None, dim: int = 0) -> torch.FloatTensor:
        """Samples ``num_samples`` with replacement, using row-wise probabilities ``w``."""
        if w.ndim != 2:
            raise ValueError(f"`w` expects 2D tensors, received {w.ndim}D.")
        if dim == 0:
            num_samples = num_samples if isinstance(num_samples, int) else int(w.size(0))
            index = torch.multinomial(w.T, num_samples=num_samples, replacement=True)  # (Q, num_samples)
            out = torch.zeros_like(w.T).scatter_(dim=1, index=index, value=1.).T       # (B, Q)
        elif dim == 1:
            num_samples = num_samples if isinstance(num_samples, int) else int(w.size(1))
            index = torch.multinomial(w, num_samples=num_samples, replacement=True)    # (B, num_samples)
            out = torch.zeros_like(w).scatter_(dim=1, index=index, value=1.)           # (B, Q)
        else:
            not NotImplementedError
        return out.bool()

    def _teacher_loss_queue(self,
                            logits: torch.FloatTensor,
                            logits_teacher: torch.FloatTensor) -> typing.Tuple[torch.FloatTensor,
                                                                               torch.BoolTensor]:
        """Add function docstring."""

        device = logits.device
        b, k = logits_teacher.size()
        if self.subsample_size < 1:
            frac = self.subsample_size
        else:
            frac = self.subsample_size / k

        with torch.no_grad():
            random_mask:    torch.BoolTensor = torch.rand((self.num_repeats, b, k), device=device).le(frac)                  # (E, B, k)
            probs_teacher: torch.FloatTensor = self.masked_softmax(logits_teacher.unsqueeze(0), random_mask, dim=2)          # (E, B, k)
            if self.aggregate == 'max':
                probs_teacher: torch.FloatTensor = torch.max(probs_teacher, dim=0, keepdim=False)[0]                         # (   B, k)
            elif self.aggregate == 'mean':
                subsample_count = random_mask.sum(dim=0, keepdim=False) + 1                                                  # (   B, k)
                probs_teacher: torch.FloatTensor = torch.sum(probs_teacher, dim=0, keepdim=False).div(subsample_count)       # (   B, k)
            elif self.aggregate == 'uncertainty':
                raise NotImplementedError
            else:
                raise ValueError(f"Only supports `max`, `mean`, `uncertainty`. Received `{self.aggregate}`")
            select_idx:     torch.LongTensor = torch.argsort(probs_teacher, dim=1, descending=True)[:, :self.num_positives]  # (   B, p)
            mask_teacher = torch.zeros_like(probs_teacher).scatter(dim=1, index=select_idx, value=1.).bool()                 # (   B, k)

        nll = -1. * F.log_softmax(logits[:, 1:], dim=1)
        nll = nll.gather(dim=1, index=select_idx).mean(dim=1).mean()
        return nll, mask_teacher

    def _teacher_loss_nn(self,
                         logits: torch.FloatTensor,
                         logits_teacher: torch.FloatTensor,
                         ) -> typing.Tuple[torch.FloatTensor, torch.BoolTensor]:
        """
        Select nearest neighbor as false negative.
        """
        _, k = logits_teacher.size()
        assert (k + 1) == logits.size(1)

        with torch.no_grad():
            _, sorting_indices = torch.sort(logits_teacher, dim=1, descending=True, stable=True)
            nn_indices         = sorting_indices[:, self.nn_size].view(-1 ,self.nn_size)  # (B, self.nn_size)
        nll = -1. * F.log_softmax(logits[:, 1:], dim=1)                                   # (B, k)
        nll = nll.gather(dim=1, index=nn_indices)                                         # (B, self.nn_size)
        nll = nll.mean(dim=1).mean()

        with torch.no_grad():
            mask_teacher = torch.zeros_like(logits_teacher)                               # (B, k)
            mask_teacher.scatter_(dim=1, index=nn_indices, src=torch.ones_like(logits_teacher))

        return nll, mask_teacher

    @staticmethod
    def masked_softmax(x: torch.FloatTensor, m: torch.BoolTensor, dim: int = -1) -> torch.FloatTensor:
        x_ = x.masked_fill(~m, float('-inf'))
        return F.softmax(x_, dim=dim)

    @staticmethod
    def masked_mean(x: torch.FloatTensor, m: torch.BoolTensor, dim: int = 0) -> torch.FloatTensor:
        m_sum = torch.clamp(m.sum(dim=dim), min=1.0)
        x_sum = torch.sum(x * m.float(), dim=dim)
        return x_sum.div(m_sum)

    @staticmethod
    def masked_var(x: torch.FloatTensor, m: torch.BoolTensor, dim: int = 0) -> torch.FloatTensor:
        m_sum = torch.clamp(m.sum(dim=dim), min=1.0)
        x_sum = torch.sum(x * m.float(), dim=dim)
        sq_x_sum = torch.sum(x.pow(2) * m.float(), dim=0)
        return sq_x_sum.div(m_sum) - x_sum.div(m_sum).pow(2)

    @staticmethod
    def nan_tensor(size: typing.Union[list,tuple], dtype: torch.dtype = torch.float32, device: str = 'cuda') -> torch.FloatTensor:
        return torch.full(torch.Size(size), fill_value=float('nan'), dtype=dtype, device=device)


class CLAPv2Loss(nn.Module):
    def __init__(self, temperature: float = 0.2, tau: float = 1.):
        super(CLAPv2Loss, self).__init__()
        self.temperature: float = temperature  # scaling parameter for InstDisc
        self.tau: float = tau                  # scaling parameter for Gumbel

    def forward(self,
                query: torch.FloatTensor,
                key: torch.FloatTensor,
                teacher: torch.FloatTensor,
                negatives: torch.FloatTensor):
        """Add function docstring."""
        negatives = negatives.clone().detach()

        # InfoNCE loss (MoCo-style)
        logits_pos = torch.einsum('bf,bf->b', [query, key]).view(-1, 1)  # $\in [-1, 1]$
        logits_neg = torch.einsum('bf,fk->bk', [query, negatives])       # $\in [-1, 1]$
        logits = torch.cat([logits_pos, logits_neg], dim=1)
        loss = F.cross_entropy(
            input=logits.div(self.temperature),
            target=torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        )

        # Teacher loss
        logits_teacher = torch.einsum('bf,fk->bk', [teacher, negatives])
        onehot_teacher = F.gumbel_softmax(logits_teacher, tau=self.tau, hard=True, dim=-1)
        loss_teacher = -1. * F.log_softmax(logits_neg.div(self.temperature), dim=1)
        loss_teacher = (loss_teacher * onehot_teacher).sum(dim=1, keepdim=False).mean()

        return loss, logits, loss_teacher, onehot_teacher.bool()

    @staticmethod
    def mask_neg_inf(x: torch.FloatTensor, m: torch.BoolTensor):
        return x.masked_fill(~m, float('-inf'))

    @staticmethod
    def masked_gumbel_softmax(logits: torch.FloatTensor, mask: torch.BoolTensor, tau: float, hard: bool = True, dim: int = -1):
        logits = logits.masked_fill(~mask, float('-inf'))
        return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=dim)


class CLAP(MoCo):
    def __init__(self, config: object, local_rank: int):
        super(CLAP, self).__init__(config, local_rank)

    def _init_modules(self):
        """
        Initializes the following modules:
            1) query network
            2) key network
            3) teacher network
            4) memory queue
        """
        encoder: nn.Module = ResNetBackbone(name=self.config.backbone_type, data=self.config.data, in_channels=3)
        head:    nn.Module = MLPHead(encoder.out_channels, self.config.projector_dim)
        if not self.config.distributed:
            # Ghost Norm; https://arxiv.org/abs/1705.0874
            encoder = SplitBatchNorm2d.convert_split_batchnorm(encoder)

        # Query network
        self.net_q = nn.Sequential()
        self.net_q.add_module('encoder', encoder)
        self.net_q.add_module('head', head)

        # Key network
        self.net_k = copy.deepcopy(self.net_q)
        self.freeze_params(self.net_k)

        # Teacher network
        self.net_t = copy.deepcopy(self.net_q)
        self.freeze_params(self.net_t)

        # Create FIFO memory queue
        self.queue = MemoryQueue(size=(self.net_k.head.num_features, self.config.num_negatives))

        if self.logger is not None:
            self.logger.info(f'Encoder ({self.config.backbone_type}): {encoder.num_parameters:,}')
            self.logger.info(f'Head ({self.config.projector_type}): {head.num_parameters:,}')

    def _init_cuda(self):
        super()._init_cuda()
        self.net_t.to(self.local_rank)

    def _init_optimization(self):
        """
        1) optimizer: {SGD, LARS}
        2) learning rate scheduler: linear warmup + cosine decay with restarts
        3) float16 training (optional)
        """
        self.optimizer = get_optimizer(params=self.net_q.parameters(),
                                       name=self.config.optimizer,
                                       lr=self.config.learning_rate,
                                       weight_decay=self.config.weight_decay)
        self.scheduler = WarmupCosineDecayLR(optimizer=self.optimizer,
                                             total_epochs=self.config.epochs,
                                             warmup_epochs=self.config.lr_warmup,
                                             warmup_start_lr=1e-4,
                                             min_decay_lr=1e-4)
        self.amp_scaler = GradScaler() if self.config.mixed_precision else None

    def _init_criterion(self):
        self.criterion = CLAPLoss(temperature=self.config.temperature,
                                  ensemble_num_estimators=self.config.ensemble_num_estimators,
                                  ensemble_dropout_rate=self.config.ensemble_dropout_rate,
                                  easy_pos_ub=self.config.easy_pos_ub,
                                  hard_pos_lb=self.config.hard_pos_lb,
                                  num_positives=self.config.num_positives,
        )

    @suppress_logging_info
    def train(self, data_loader: DataLoader, epoch: int, **kwargs) -> typing.Dict[str, typing.Union[int,float]]:
        """Iterates over the `data_loader' once for CLAP training."""

        steps = len(data_loader)
        self._set_learning_phase(train=True)
        metrics = {
            'loss/moco': torch.zeros(steps, device=self.local_rank),
            'loss/easy': torch.zeros(steps, device=self.local_rank),
            'loss/hard': torch.zeros(steps, device=self.local_rank),
            'rank@1':    torch.zeros(steps, device=self.local_rank),
            'precision/easy': torch.zeros(steps, device=self.local_rank),
            'precision/hard': torch.zeros(steps, device=self.local_rank),
        }
        with get_rich_pbar(transient=True, auto_refresh=False, disable=self.local_rank != 0) as pbar:
            job = pbar.add_task(f":thread:", total=steps)

            for i, batch in enumerate(data_loader):
                # Single batch iteration
                batch_history = self.train_step(batch, epoch=epoch)
                # Accumulate metrics
                metrics['loss/moco'][i] = batch_history['moco_loss'].detach()
                metrics['loss/easy'][i] = batch_history['easy_loss'].detach()
                metrics['loss/hard'][i] = batch_history['hard_loss'].detach()
                metrics['rank@1'][i] = batch_history['rank'].detach()
                metrics['precision/easy'][i] = batch_history['easy_precision'].detach()
                metrics['precision/hard'][i] = batch_history['hard_precision'].detach()
                # Update progress bar
                msg = f':thread:[{i+1}/{steps}]: ' + \
                    ' | '.join([f"{k} : {self.nanmean(v[:i+1]).item():.4f}" for k, v in metrics.items()])
                pbar.update(job, advance=1., description=msg)
                pbar.refresh()

        train_history = {k: self.nanmean(v).item() for k, v in metrics.items()}
        
        return train_history

    def train_step(self, batch: dict, epoch: int) -> typing.Dict[str, torch.Tensor]:
        """A single forward & backward pass using a batch of examples."""
        with torch.cuda.amp.autocast(self.amp_scaler is not None):
            # Fetch three positive views; {query, key, teacher}
            x_q = batch['x1'].to(self.local_rank, non_blocking=True)
            x_k = batch['x2'].to(self.local_rank, non_blocking=True)
            x_t = batch['x3'].to(self.local_rank, non_blocking=True)
            # Compute query features; (B, f)
            z_q = F.normalize(self.net_q(x_q), dim=1)
            with torch.no_grad():
                # An exponential moving average update of the key network
                self._momentum_update_key_net()
                # An exponential moving average update of the teacher network
                self._momentum_update_teacher_net()
                # Shuffle across devices (GPUs)
                x_k, idx_unshuffle_k = ForMoCo.batch_shuffle_ddp(x_k)
                x_t, idx_unshuffle_t = ForMoCo.batch_shuffle_ddp(x_t)
                # Compute {key, teacher} features; (B, f)
                z_k = F.normalize(self.net_k(x_k), dim=1)
                z_t = F.normalize(self.net_t(x_t), dim=1)
                # Restore {key, teacher} features to their original nodes
                z_k  = ForMoCo.batch_unshuffle_ddp(z_k, idx_unshuffle_k)
                z_t = ForMoCo.batch_unshuffle_ddp(z_t, idx_unshuffle_t)

            # Compute loss & metrics
            moco_loss, logits, easy_loss, easy_mask, hard_loss, hard_mask = \
                self.criterion(query=z_q,
                               key=z_k,
                               teacher=z_t,
                               negatives=self.queue.buffer)
            y = batch['y'].to(self.local_rank).detach()
            rank = TopKAccuracy(k=1)(logits.detach(), torch.zeros_like(y))
            easy_precision = self.teacher_precision(y, self.queue.labels, easy_mask)
            hard_precision = self.teacher_precision(y, self.queue.labels, hard_mask)
            
            # Backpropagate & update
            self.backprop(
                moco_loss + \
                self.loss_weight_easy(epoch=epoch) * easy_loss + \
                self.loss_weight_hard(epoch=epoch) * hard_loss
            )
            # Update memory queue
            self.queue.update(keys=z_k,
                              indices=batch['idx'].to(self.local_rank),
                              labels=y)

            return dict(
                moco_loss=moco_loss,
                rank=rank,
                easy_loss=easy_loss,
                hard_loss=hard_loss,
                easy_precision=easy_precision,
                hard_precision=hard_precision,
            )

    def loss_weight_easy(self, epoch: int) -> float:
        if epoch >= self.config.easy_pos_start_epoch:
            return self.config.loss_weight_easy
        else:
            return 0.

    def loss_weight_hard(self, epoch: int) -> float:
        if epoch >= self.config.hard_pos_start_epoch:
            return self.config.loss_weight_hard
        else:
            return 0.

    def _set_learning_phase(self, train: bool = False):
        """Set mode of networks (train or eval)."""
        if train:
            self.net_q.train()
            self.net_k.train()
            self.net_t.train()
        else:
            self.net_q.eval()
            self.net_k.eval()
            self.net_t.eval()

    @torch.no_grad()
    def _momentum_update_key_net(self):
        m = self.config.key_momentum
        for q, k in zip(self.net_q.parameters(), self.net_k.parameters()):
            k.data = k.data * m + q.data * (1. - m)

    @torch.no_grad()
    def _momentum_update_teacher_net(self):
        m = self.config.teacher_momentum
        for q, t in zip(self.net_q.parameters(), self.net_t.parameters()):
            t.data = t.data * m + q.data * (1. - m)

    def save_checkpoint(self, path: str, epoch: int, **kwargs):
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
            'net_t': self.net_k.state_dict(),
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
        Returns the epoch of the checkpoint + 1.
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

        # Load teacher network
        self.net_t.load_state_dict(ckpt['net_t'])

        # Load optimizer
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.move_optimizer_states(self.optimizer, device)

        # Load scheduler
        if 'scheduler' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler'])
        else:
            if self.scheduler is not None:
                warnings.warn('scheduler not loaded as it was not initiated.', UserWarning)

        return ckpt['epoch'] + 1

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


class CLAPv2(CLAP):
    def __init__(self, config: object, local_rank: int):
        super(CLAPv2, self).__init__(config, local_rank)

    def _init_modules(self):
        super()._init_modules()
        self.predictor = CLAPv2PredHead(
            input_size=self.config.projector_dim,
            hidden_size=self.config.projector_dim * 4,
            output_size=self.config.projector_dim,
        )
        if self.logger is not None:
            self.logger.info(f'Predictor: {self.predictor.num_parameters:,}')

    def _init_cuda(self):
        super()._init_cuda()
        self.predictor.to(self.local_rank)

    def _init_criterion(self):
        self.criterion = CLAPv2Loss(temperature=self.config.temperature,
                                    tau=self.config.tau)

    def train_step(self, batch: dict):
        """Add function docstring."""
        with torch.cuda.amp.autocast(self.amp_scaler is not None):

            # Fetch three positive views for {query, key, teacher}
            x_q = batch['x1'].to(self.local_rank, non_blocking=True)
            x_k = batch['x2'].to(self.local_rank, non_blocking=True)
            x_t = batch['x3'].to(self.local_rank, non_blocking=True)

            # Compute query features; (B,3,H,W) -> (B,f)
            z_q = F.normalize(self.net_q(x_q), dim=1)
            # Compute key features; shuffle across devices
            with torch.no_grad():
                self._momentum_update_key_net()
                x_k, idx_unshuffle_k = ForMoCo.batch_shuffle_ddp(x_k)
                z_k = F.normalize(self.net_k(x_k), dim=1)
                z_k = ForMoCo.batch_unshuffle_ddp(z_k, idx_unshuffle_k)
            # Compute teacher features;
            z_t = F.normalize(self.predictor(self.net_t(x_t)), dim=1)
            # Compute loss
            loss, logits, loss_teacher, mask_teacher = \
                self.criterion(query=z_q, key=z_k, teacher=z_t, negatives=self.queue.buffer)
            # Compute metrics
            y = batch['y'].to(self.local_rank).detach()
            rank = TopKAccuracy(k=1)(logits.detach(), torch.zeros_like(y))
            precision = self.teacher_precision(labels_batch=y,
                                               labels_queue=self.queue.labels,
                                               mask_teacher=mask_teacher)
            # Backpropagate & update
            self.backprop(loss + self.loss_weight * loss_teacher)
            # Update memory queue
            self.queue.update(keys=z_k,
                              indices=batch['idx'].to(self.local_rank),
                              labels=y)

            return loss, loss_teacher, rank, precision

    def _set_learning_phase(self, train: bool = False):
        """Set mode of networks (train or eval)."""
        if train:
            self.net_q.train()
            self.net_k.train()
            self.net_t.train()
            self.predictor.train()
        else:
            self.net_q.eval()
            self.net_k.eval()
            self.net_t.eval()
            self.predictor.eval()

    def save_checkpoint(self, path: str, epoch: int, **kwargs):
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
            'net_t': self.net_k.state_dict(),
            'predictor': self.predictor.state_dict(),
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
        Returns the epoch of the checkpoint + 1.
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

        # Load teacher network
        self.net_t.load_state_dict(ckpt['net_t'])
        self.predictor.load_state_dict(ckpt['predictor'])

        # Load optimizer
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.move_optimizer_states(self.optimizer, device)

        # Load scheduler
        if 'scheduler' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler'])
        else:
            if self.scheduler is not None:
                warnings.warn('scheduler not loaded as it was not initiated.', UserWarning)

        return ckpt['epoch'] + 1