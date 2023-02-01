

import copy
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler

from tasks.moco import MoCo, MemoryQueue
from layers.batchnorm import SplitBatchNorm2d
from models.backbone import ResNetBackbone
from models.head import MLPHeadWithMCDropout

from utils.distributed import ForMoCo
from utils.metrics import TopKAccuracy
from utils.mixture_models import vMFMixture
from utils.optimization import WarmupCosineDecayLR
from utils.optimization import configure_optimizer
from utils.progress import configure_progress_bar
from utils.decorators import suppress_logging_info


class MoCoWithMixtureLoss(nn.Module):
    def __init__(self, temperature: float = 0.2, reduction: str = 'mean'):
        super(MoCoWithMixtureLoss, self).__init__()
        self.temperature: float = temperature
        self.reduction: str = reduction

    def forward(self,
                query: torch.FloatTensor,
                key: torch.FloatTensor,
                negatives: torch.FloatTensor,
                scores: torch.FloatTensor,
                query_assignments: torch.LongTensor,
                thresholds: typing.Tuple[float] = (0.01, 0.5),  # TODO: add to argparse
                ) -> typing.Tuple[torch.FloatTensor]:
        """
        Arguments:
            query: `torch.FloatTensor` of shape (B, F)
            key: `torch.FloatTensor` of shape (B, F)
            negatives: `torch.FloatTensor` of shape (F, Q)
            scores: `torch.FloatTensor` of shape (Q, C), where C is the number of components
            query_assignments: `torch.LongTensor` of shape (B,  )
        Returns:
            loss: torch.FloatTensor of shape (1, ) if self.reduction is not `none`.
            logits: torch.FloatTensor of shape (B, 1+Q)
            mask: torch.BoolTensor of shape (B, 1+Q)
            ...
        """

        # Compute temperature-scaled similarities
        pos_logits = torch.einsum('nf,nf->n', [query, key]).view(-1, 1)               # (B, 1  )
        neg_logits = torch.einsum('nf,fq->nq', [query, negatives.clone().detach()])   # (B,   Q)
        logits = torch.cat([pos_logits, neg_logits], dim=1)                           # (B, 1+Q)
        logits.div_(self.temperature)

        # Permute columns of scores based on the assignments of queries
        scores = scores[:, query_assignments.squeeze()]                               # (Q,  B') -> (Q,  B)
        
        # Create mask; 1 indicates true negatives, 0 otherwise.
        pos_mask = torch.ones_like(pos_logits).bool()                                 # (B, 1  )
        neg_mask = scores.gt(thresholds[0]).T & scores.lt(thresholds[1]).T            # (B,   Q)
        mask = torch.cat([pos_mask, neg_mask], dim=1)                                 # (B, 1+Q)
        
        # InfoNce loss; (B, 1) <- (B, 1) - (B, 1)
        loss = pos_logits.div(self.temperature) \
            - torch.logsumexp(logits.masked_fill(~mask, float('-inf')), dim=1, keepdim=True)
        loss = torch.neg(loss)

        if self.reduction == 'mean':
            return loss.mean(), logits, mask
        elif self.reduction == 'sum':
            return loss.sum(), logits, mask
        elif self.reduction == 'none':
            return loss.squeeze(), logits, mask
        else:
            raise NotImplementedError


class MoCoWithMixtures(MoCo):
    def __init__(self, config: object, local_rank: int):
        super(MoCoWithMixtures, self).__init__(config, local_rank)

    def _init_modules(self):
        """
        Initializes the following modules:
            1) query network (self.net_q)
            2) key network (self.net_k)
            3-1) sampling encoder (self.encoder_s)
            3-2) sampling head (self.head_s)
            4) memory queue (self.queue)
            5) von Mises-Fisher mixture module (self.vmf_mixture)
        """
        encoder: nn.Module = ResNetBackbone(name=self.config.backbone_type, data=self.config.data, in_channels=3)
        head: nn.Module = MLPHeadWithMCDropout(
            in_channels=encoder.out_channels,
            num_features=self.config.projector_dim,
            dropout_rate=self.config.dropout_rate  # TODO; add to argparse
        )
        if not self.config.distributed:
            # Replace `nn.BatchNorm2d` layers with GhostNorm2d
            encoder = SplitBatchNorm2d.convert_split_batchnorm(encoder)

        # Query network
        self.net_q = nn.Sequential()
        self.net_q.add_module('encoder', encoder)
        self.net_q.add_module('head', head)

        # Key network
        self.net_k = copy.deepcopy(self.net_q)
        self.freeze_params(self.net_k)

        # TODO: Sampling network using MC dropout
        self.encoder_s = copy.deepcopy(encoder)
        self.head_s = copy.deepcopy(head)
        self.freeze_params(self.encoder_s)
        self.freeze_params(self.head_s)

        # First-in-first-out queue to store negative examples
        self.queue = MemoryQueue(size=(self.net_k.head.num_features, self.config.num_negatives))

        if self.logger is not None:
            self.logger.info(f'Encoder ({self.config.backbone_type}): {encoder.num_parameters:,}')
            self.logger.info(f'Head ({self.config.projector_type}): {head.num_parameters:,}')

        if self.config.mixture_n_components is not None:
            order: int = self.config.mixture_n_components
        else:
            order: int = self.config.batch_size
        self.vmf_mixture = vMFMixture(x_dim=self.net_k.head.num_features, order=order)
        if self.logger is not None:
            self.logger.info(
                f'von Mises-Fisher Mixture model: {self.net_k.head.num_features} features '
                f'and {order} components.'
            )

    def _init_cuda(self) -> None:
        """
        1) Assign cuda devices to modules.
        2) Wraps query network with `torch.distributed.DistributedDataParallel`.
        """
        if self.config.distributed:
            self.net_q = DistributedDataParallel(
                module=self.net_q.to(self.local_rank),
                device_ids=[self.local_rank],
                bucket_cap_mb=100)
        else:
            self.net_q.to(self.local_rank)
        
        self.net_k.to(self.local_rank)
        self.encoder_s.to(self.local_rank)
        self.head_s.to(self.local_rank)
        self.queue.to(self.local_rank)
        self.vmf_mixture.to(self.local_rank)

    def _init_criterion(self) -> None:
        self.criterion = MoCoWithMixtureLoss(
            temperature=self.config.temperature,
            reduction='mean',
        )

    def _init_optimization(self) -> None:
        """
        1) optimizer: {SGD, LARS}
        2) learning rate scheduler: linear warmup + cosine decay
        3) float16 training (optional)
        """
        # TODO; duplicate method with super-class. remove at will.
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

    @suppress_logging_info
    def train(self, data_loader: DataLoader, epoch: int, **kwargs) -> typing.Dict[str, float]:
        """Iterates over the `data_loader` once for `MoCoWithMixtures` training."""

        steps: int = len(data_loader)
        self._set_learning_phase(train=True)
        metrics = {
            'train/loss': torch.zeros(steps, device=self.local_rank),
            'train/rank@1': torch.zeros(steps, device=self.local_rank),
        }

        with configure_progress_bar(transient=True, disable=self.local_rank != 0) as pbar:
            job = pbar.add_task(f":ocean:", total=steps)
            for i, batch in enumerate(data_loader):
                # Single batch iteration
                batch_metrics: dict = self.train_step(batch, epoch=epoch)
                metrics['train/loss'][i] = batch_metrics['loss'].detach()
                metrics['train/rank@1'][i] = batch_metrics['rank'].detach()
                # Update progress bar
                msg: str = f":ocean: [{i+1}/{steps}]: " + \
                    ' | '.join([f"{k} : {self.nanmean(v[:i+1]).item():.4f}" for k, v in metrics.items()])
                pbar.update(job, advance=1., description=msg)
                pbar.refresh()

        return {k: self.nanmean(v).item() for k, v in metrics.items()}

    def _get_score_thresholds(self, step: int, total_step: int) -> typing.Tuple[float]:  # TODO; implement
        """Anneal thresholds based on learning stage."""
        return (self.config.threshold_lower, self.config.threshold_upper)

    def train_step(self, batch: dict, epoch: int) -> typing.Dict[str, torch.Tensor]:
        """
        A single forward & backward pass using a mini-batch of samples.
        Arguments:
            batch: dict;
            epoch: int;
        Returns:
            ...
        """
        with torch.cuda.amp.autocast(enabled=self.amp_scaler is not None):
            
            # Fetch three positive views; {query, key, sampling}
            x_q = batch['x1'].to(self.local_rank, non_blocking=True)
            x_k = batch['x2'].to(self.local_rank, non_blocking=True)
            x_s = batch['x2'].to(self.local_rank, non_blocking=True)  # FIXME; use `x3`
            
            # Compute query features; (B, f)
            z_q = F.normalize(self.net_q(x_q), dim=1)
            
            # Compute key features; (B, f)
            with torch.no_grad():
                # An exponential moving average update of the key network
                self._momentum_update_key_net()
                # Shuffle across devices (GPUs)
                x_k, idx_unshuffle_k = ForMoCo.batch_shuffle_ddp(x_k)
                # Compute key features; (B, f)
                z_k = F.normalize(self.net_k(x_k), dim=1)
                # Restore key features to their original nodes
                z_k = ForMoCo.batch_unshuffle_ddp(z_k, idx_unshuffle_k)
                
            # Compute extra positive features; ? \times (B, f)
            with torch.no_grad():
                # Synchronize sampling network with key network
                self._synchronize_sampling_net()
                # Shuffle across devices (GPUs)
                x_s, idx_unshuffle_s = ForMoCo.batch_shuffle_ddp(x_s)
                # Compute encoder features
                h_s = self.encoder_s(x_s)
                # Restore encoder features to their original nodes
                h_s = ForMoCo.batch_unshuffle_ddp(h_s, idx_unshuffle_s)
                # Compute extra positive features by sampling via monte-carlo dropout
                z_s_list = list()
                for _ in range(self.config.num_extra_positives):
                    z_s = self.head_s.forward(h_s, dropout=True)
                    z_s_list += [F.normalize(z_s, dim=1)]

            """Fit von Mises-Fisher mixture models."""
            with torch.no_grad():
                # Fit vMF mixture model via expectation-maximization
                self.vmf_mixture.reset();
                self.vmf_mixture.fit(
                    x=torch.cat([z_q, z_k] + z_s_list, dim=0),
                    n_iter=self.config.mixture_n_iter,
                    tol=self.config.mixture_tol,
                    verbose=False,  # TODO: argparse
                    precise=False,  # TODO: argparse
                )  
                # Score negative examples with the vmF mixture model
                scores = self.vmf_mixture.predict_proba(self.queue.buffer.T)  # (Q, F)
                query_assignments = self.vmf_mixture.predict(z_q, probs=False)
            
            # Compute loss & metrics
            loss, logits, mask = self.criterion(
                query=z_q, key=z_k, negatives=self.queue.buffer,
                scores=scores.detach(),                        # TODO; detach not mandatory
                query_assignments=query_assignments.detach(),  # TODO; sanity check
                thresholds=self._get_score_thresholds(step=epoch, total_step=self.config.epochs),
            )
            y = batch['y'].to(self.local_rank, non_blocking=True).detach()
            rank = TopKAccuracy(k=1)(logits.detach(), torch.zeros_like(y))
            # Backpropagate & update
            self.backprop(loss)
            # Update memory queue
            self.queue.update(keys=z_k,
                              indices=batch['idx'].to(self.local_rank),
                              labels=y)

        return dict(loss=loss, rank=rank, mask=mask.clone().detach())

    def backprop(self, loss: torch.FloatTensor) -> None:
        """..."""
        if self.amp_scaler is not None:
            self.amp_scaler.scale(loss).backward()
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad()

    def _set_learning_phase(self, train: bool = False):
        """Set mode of modules (train or eval)."""
        if train:
            self.net_q.train()
            self.net_k.train()
            self.encoder_s.train()
            self.head_s.train()
        else:
            self.net_q.eval()
            self.net_k.eval()
            self.encoder_s.eval()
            self.head_s.eval()

    @torch.no_grad()
    def _momentum_update_key_net(self) -> None:
        m = self.config.key_momentum
        for q, k in zip(self.net_q.parameters(), self.net_k.parameters()):
            k.data = k.data * m + q.data * (1. - m)

    @torch.no_grad()
    def _synchronize_sampling_net(self) -> None:
        for k, s in zip(self.net_k.encoder.parameters(), self.encoder_s.parameters()):
            s.data = k.data
        for k, s in zip(self.net_k.head.parameters(), self.head_s.parameters()):
            s.data = k.data
    
    def save_checkpoint(self, path: str, epoch: int, **kwargs) -> None:
        """Save model to a `.tar` checkpoint file."""
        if isinstance(self.net_q, DistributedDataParallel):
            encoder = self.net_q.module.encoder
            head = self.net_q.module.head
        else:
            encoder = self.net_q.encoder
            head = self.net_q.head
        
        ckpt: dict = {
            'encoder': encoder.state_dict(),
            'head': head.state_dict(),
            'net_k': self.net_k.state_dict(),
            'encoder_s': self.encoder_s.state_dict(),
            'head_s': self.head_s.state_dict(),
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

        # Load sampling network
        self.encoder_s.load_state_dict(ckpt['encoder_s'])
        self.head_s.load_state_dict(ckpt['head_s'])

        # TODO; load von Mises-Fisher mixture models
        # this may not be necessary. sanity check needed.

        # Load memory queue
        self.queue.load_state_dict(ckpt['queue'])

        # Load optimizer states
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.move_optimizer_states(self.optimizer, device)

        # Load scheduler states
        if 'scheduler' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler'])
        
        return ckpt['epoch'] + 1
