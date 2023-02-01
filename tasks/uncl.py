
import os
import copy
import typing
import warnings

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler

from tasks.base import Task
from tasks.moco import MemoryQueue
from layers.batchnorm import SplitBatchNorm2d
from models.backbone import ResNetBackbone
from models.head import MLPHead

from utils.distributed import ForMoCo
from utils.metrics import TopKAccuracy
from utils.knn import KNNEvaluator
from utils.optimization import WarmupCosineDecayLR
from utils.optimization import configure_optimizer
from utils.logging import configure_logger
from utils.progress import configure_progress_bar


class UNCLLoss(nn.Module):
    def __init__(self, temperature: float = 0.2,
                       num_false_negatives: int = 1,
                       ensemble_num_estimators: int = 128,
                       ensemble_dropout_rate: float = 0.2,
                       threshold: float = 0.5,
                       teacher_temperature: float = None):
        super(UNCLLoss, self).__init__()

        self.temperature: float = temperature
        self.num_false_negatives: int = num_false_negatives
        self.ensemble_num_estimators: int = ensemble_num_estimators
        self.ensemble_dropout_rate: float = ensemble_dropout_rate
        self.threshold: float = threshold

        if teacher_temperature is not None:
            self.teacher_temperature = teacher_temperature
        else:
            self.teacher_temperature = temperature

    def forward(self, query: torch.FloatTensor,
                      key: torch.FloatTensor,
                      teacher: torch.FloatTensor,
                      negatives: torch.FloatTensor) -> typing.Tuple[torch.FloatTensor,
                                                                    torch.FloatTensor,
                                                                    torch.BoolTensor]:
        """..."""

        # detach & clone memory queue to avoid unintended inplace operations
        negatives = negatives.detach().clone()

        # similarities between teacher and negative representations
        teacher_sim = self._compute_teacher_similarities(teacher, negatives)         # (E, B, Q)
        teacher_sim = teacher_sim.div(self.teacher_temperature)
        # false-negative scores; between batch and negative examples
        scores = self._compute_scores(teacher_sim, num_samples=None)                 # (B, Q) <- (E, B, Q)
        # false-negative uncertainties; a scalar value for each negative example
        uncertainties = self._compute_uncertainties(teacher_sim)                     # (1, Q) <- (E, B, Q)

        # similarities between query and key representations (positive pairs)
        logits_pos = torch.einsum('bf,bf->b', *[query, key]).view(-1, 1)             # (B, 1)
        logits_pos.div_(self.temperature)
        # similarities between query and negative representations (negative pairs)
        logits_neg = torch.einsum('bf,fq->bq', *[query, negatives])                  # (B, Q)
        logits_neg.div_(self.temperature)
        
        # false negative selection; select false negative among
        # the negatives with low uncertainties (entropy)
        ub = torch.quantile(uncertainties, q=self.threshold, dim=1)                  # (1,  );
        false_neg_cand_mask = uncertainties.le(ub).repeat(query.size(0), 1)          # (B, Q);
        false_neg_scores = scores.masked_fill(~false_neg_cand_mask, float('-inf'))   # (B, Q);
        _, false_neg_idx = false_neg_scores.topk(k=self.num_false_negatives, dim=1)  # (B, n);
        false_neg_mask = \
            torch.zeros_like(false_neg_scores).scatter(1, false_neg_idx, 1.).bool()

        # loss (exponential of logits for false negatives <- 0)
        loss = - (
            logits_pos - torch.log(
                torch.cat([logits_pos.exp(),
                           logits_neg.exp().scatter(1, false_neg_idx, 0.)],
                           dim=1).sum(dim=1, keepdim=True)
            )
        )
        loss = loss.mean()  # (1, ) <- (B, 1)

        return loss, torch.cat([logits_pos, logits_neg], dim=1), false_neg_mask

    @torch.no_grad()
    def _compute_teacher_similarities(self, teacher: torch.FloatTensor,
                                            negatives: torch.FloatTensor) -> torch.FloatTensor:
        """..."""
        teacher = teacher.unsqueeze(0).repeat(self.ensemble_num_estimators, 1, 1)              # (E, B, F)
        teacher = self.random_maskout(teacher, p=self.ensemble_dropout_rate, val=0.)           # (E, B, F)
        teacher = F.normalize(teacher, p=2, dim=2)                                             # (E, B, F)
        negatives = negatives.clone().unsqueeze(0).repeat(self.ensemble_num_estimators, 1, 1)  # (E, F, Q)
        negatives = self.random_maskout(negatives, p=self.ensemble_dropout_rate, val=0.)       # (E, F, Q)
        negatives = F.normalize(negatives, p=2, dim=1)                                         # (E, F, Q)
        return torch.einsum('ebf,efq->ebq', *[teacher, negatives])                             # (E, B, Q)

    @staticmethod
    def random_maskout(x: torch.FloatTensor, p: float = 0.2, val: float = 0.0) -> torch.FloatTensor:
        """Writes ``val`` to the tensor ``x`` with random probability of ``p``."""
        m: torch.BoolTensor = torch.rand_like(x).le(p)
        return x.masked_fill(m, val)

    @torch.no_grad()
    def _compute_scores(self, sim: torch.FloatTensor, num_samples: int = None) -> torch.FloatTensor:
        """
        False-negative scores, normalized across the queue dimension,
        and averaged across the ensemble dimension.
        Arguments:
            sim: 3D FloatTensor with shape (E, B, Q).
            num_samples: int.
        Returns:
            scores: 2D FloatTensor of shape (B, Q)
        """
        if num_samples is not None:
            sample_prob = num_samples / int(sim.size(2))
            mask = torch.rand_like(sim).le(sample_prob)     # (E, B, Q)
            scores = self.masked_softmax(sim, mask, dim=2)  # (E, B, Q)
            return self.masked_mean(scores, mask, dim=0)    # (   B, Q)
        else:
            scores = F.softmax(sim, dim=2)                  # (E, B, Q)
            return scores.mean(dim=0, keepdim=False)        # (   B, Q); FIXME; mean? median? max? std?

    @staticmethod
    def masked_softmax(x: torch.FloatTensor, m: torch.BoolTensor, dim: int = -1) -> torch.FloatTensor:
        x_ = x.masked_fill(~m, float('-inf'))
        return F.softmax(x_, dim=dim)

    @staticmethod
    def masked_mean(x: torch.FloatTensor, m: torch.BoolTensor, dim: int = 0) -> torch.FloatTensor:
        m_sum = torch.clamp(m.sum(dim=dim), min=1.0)
        x_sum = torch.sum(x * m.float(), dim=dim)
        return x_sum.div(m_sum)

    @torch.no_grad()
    def _compute_uncertainties(self, sim: torch.FloatTensor, bootstrap: bool = False) -> torch.FloatTensor:
        """
        Compute uncertainty scores, normalized across the batch dimension
        and averaged across the ensemble dimension.
        Arguments:
            sim: 3D FloatTensor of shape (E, B, Q).
        """
        if bootstrap:
            mask = [self.bootstrap(torch.ones_like(s), dim=0) for s in sim]      #    (B, Q) x E
            mask = torch.stack(mask, dim=0)                                      # (E, B, Q)
            entropy = self.masked_entropy_with_logits(sim, mask, softmax_dim=1)  # (E, Q)
        else:
            entropy = self.entropy_with_logits(sim, softmax_dim=1)               # (E, Q)
        return entropy.mean(dim=0, keepdim=True)                                 # (1, Q)

    @staticmethod  # FIXME: this is a special case of random_sample with `replacement`=True
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

    @staticmethod
    def random_sample(w: torch.FloatTensor, num_samples: int, replacement: bool = False, dim: int = 0) -> torch.BoolTensor:
        """
        Samples `num_samples` with replacement=`replacement` along dim=`dim`.
        Arguments:
            w: 2D ``torch.FloatTensor``.
            num_samples: int.
            replacement: bool.
            dim: int.
        """
        if w.ndim != 2:
            raise ValueError(f"Expected 2D tensor for `w`, received {w.ndim}D.")
        if dim == 0:
            index = torch.multinomial(w.T, num_samples=num_samples, replacement=replacement)
            out = torch.zeros_like(w.T).scatter(dim=1, index=index, value=1.).T
        elif dim == 1:
            index = torch.multinomial(w, num_samples=num_samples, replacement=replacement)
            out = torch.zeros_like(w).scatter(dim=1, index=index, value=1.)
        else:
            raise NotImplementedError

        return out.bool()

    @staticmethod
    def entropy_with_logits(logits: torch.FloatTensor, softmax_dim: int = 1):
        probs = F.softmax(logits, dim=softmax_dim).add(1e-7)     # (E, B, Q) 
        entropy = - probs.mul(probs.log()).sum(dim=softmax_dim)  # (E, B, Q) -> (E, Q) if `softmax_dim` = 1
        return entropy

    @staticmethod
    def entropy_with_probs(probs: torch.FloatTensor, average_dim: int = 1):
        probs = probs.add(1e-7)
        entropy = - probs.mul(torch.log(probs)).sum(dim=average_dim)
        return entropy

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
        entropy = - probs.mul(probs.log()).sum(dim=softmax_dim)    # (E, Q); if `softmax_dim` = 1
        count = mask.sum(dim=softmax_dim).clamp(min=1.0)           # (E, Q); if `softmax_dim` = 1
        return entropy.div(count)

    @staticmethod
    def masked_entropy_with_probs(probs: torch.FloatTensor, mask: torch.BoolTensor, average_dim: int = 1):
        """
            probs: 3D ``torch.FloatTensor`` of shape (E, B, Q).
            m: 3D ``torch.BoolTensor`` of shape (E, B, Q).
            average_dim: ``int``, dimension to compute entropy across.
        Return:
            2D ``torch.FloatTensor`` of shape (E, Q), containing entropy values.
        """
        probs = probs.add(1e-5)
        entropy = - probs.mul(probs.log())
        count = mask.sum(dim=average_dim).clamp(min=1.0)
        return entropy.sum(dim=average_dim).div(count)


class UNCL(Task):
    def __init__(self, config: object, local_rank: int):
        super(UNCL, self).__init__()

        self.config = config
        self.local_rank = local_rank

        self._init_logger()
        self._init_modules()
        self._init_cuda()
        self._init_optimization()
        self._init_criterion()
        self._resume_training_from_checkpoint()

    def _init_logger(self) -> None:
        """..."""
        if self.local_rank == 0:
            logfile = os.path.join(self.config.checkpoint_dir, 'main.log')
            self.logger = configure_logger(logfile=logfile)
            self.logger.info(f'Checkpoint directory: {self.config.checkpoint_dir}')
        else:
            self.logger = None

    def _init_modules(self):
        """..."""
        encoder = ResNetBackbone(name=self.config.backbone_type, data=self.config.data, in_channels=3)
        head    = MLPHead(in_channels=encoder.out_channels, num_features=self.config.projector_dim)
        if not self.config.distributed:
            encoder = SplitBatchNorm2d.convert_split_batchnorm(encoder)

        # 1. Query network
        self.net_q = nn.Sequential()
        self.net_q.add_module('encoder', encoder)
        self.net_q.add_module('head', head)

        # 2. Key network
        self.net_k = copy.deepcopy(self.net_q)
        for p in self.net_k.parameters():
            p.requires_grad =  False

        # 3. Teacher network
        self.net_t = copy.deepcopy(self.net_q)
        for p in self.net_t.parameters():
            p.requires_grad = False

        # 4. Memory queue
        self.queue = MemoryQueue(size=(self.config.projector_dim, self.config.num_negatives))

    def _init_cuda(self) -> None:
        """..."""
        if self.config.distributed:
            self.net_q = DistributedDataParallel(module=self.net_q.to(self.local_rank),
                                                 device_ids=[self.local_rank])
        else:
            self.net_q.to(self.local_rank)
        self.net_k.to(self.local_rank)
        self.net_t.to(self.local_rank)
        self.queue.to(self.local_rank)

    def _init_criterion(self) -> None:
        """..."""
        self.criterion = UNCLLoss(
            temperature=self.config.temperature,
            num_false_negatives=self.config.num_false_negatives,
            ensemble_num_estimators=self.config.ensemble_num_estimators,
            ensemble_dropout_rate=self.config.ensemble_dropout_rate,
            threshold=self.config.uncertainty_threshold,
            teacher_temperature=self.config.teacher_temperature,
        )

    def _init_optimization(self) -> None:
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
            self.start_epoch = 1  # No checkpoint provided

    def run(self, train_set: torch.utils.data.Dataset,
                  memory_set: torch.utils.data.Dataset,
                  test_set: torch.utils.data.Dataset) -> None:
        """Training and evaluation."""

        if self.logger is not None:
            self.logger.info(f"Data: {train_set.__class__.__name__}")
            self.logger.info(f"Number of training examples: {len(train_set):,}")

        if self.config.distributed:
            sampler = DistributedSampler(train_set, shuffle=True)
        else:
            sampler = None
        train_loader = DataLoader(train_set,
                                  batch_size=self.config.batch_size,
                                  sampler=sampler,
                                  shuffle=sampler is None,
                                  num_workers=self.config.num_workers,
                                  drop_last=True,
                                  pin_memory=True,
                                  persistent_workers=self.config.num_workers > 0)

        if self.local_rank == 0:
            # Intermediate evaluation of representations based on nearest neighbors.
            # The frequency is controlled by the `eval_every' argument of this function.
            eval_loader_config = dict(
                batch_size=self.config.batch_size * self.config.world_size,
                num_workers=self.config.num_workers * self.config.world_size,
                pin_memory=True, persistent_workers=True
            )
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
            train_history = self.train(train_loader)

            # Evaluate the learned representations every `eval_every` epochs, on rank 0.
            if (knn_evaluator is not None) & (epoch % self.config.eval_every == 0):
                knn_scores = knn_evaluator.evaluate(
                    net=self.net_q.module.encoder if self.config.distributed else self.net_q.encoder,
                    memory_loader=memory_loader,
                    test_loader=test_loader,
                )
            else:
                knn_scores = None

            # Log; https://wandb.ai
            log = dict()
            log.update(train_history)
            if isinstance(knn_scores, dict):
                log.update({f'eval/knn@{k}': v for k, v in knn_scores.items()})
            if self.scheduler is not None:
                log.update({'misc/lr': self.scheduler.get_last_lr()[0]})
            if self.local_rank == 0:
                wandb.log(data=log, step=epoch)

            # Log; stdout
            if self.logger is not None:
                fmt = f">{len(str(self.config.epochs))}"
                msg = " | ".join([f"{k} : {v:.4f}" for k, v in log.items()])
                msg = f"Epoch [{epoch:{fmt}}/{self.config.epochs}] - " + msg
                self.logger.info(msg)

            # Save model checkpoint (every `save_every` epochs)
            if (self.local_rank == 0) & (epoch % self.config.save_every == 0):
                fmt = f">0{len(str(self.config.epochs))}"
                ckpt = os.path.join(self.config.checkpoint_dir, f"ckpt.{epoch:{fmt}}.pth.tar")
                self.save_checkpoint(ckpt, epoch=epoch, history=log)

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

    def train(self, data_loader: torch.utils.data.DataLoader) -> typing.Dict[str, float]:
        """..."""
        steps = len(data_loader)
        self._set_learning_phase(train=True)
        metrics = {
            'train/loss': torch.zeros(steps, device=self.local_rank),
            'train/rank@1': torch.zeros(steps, device=self.local_rank),
            'train/precision': torch.zeros(steps, device=self.local_rank),
        }
        with configure_progress_bar(transient=True, disable=self.local_rank != 0) as pbar:
            job = pbar.add_task(f":ocean:", total=steps)
            for i, batch in enumerate(data_loader):
                # single batch iteration
                batch_history = self.train_step(batch)
                # accumulate batch metrics
                metrics['train/loss'][i] = batch_history['loss'].detach()
                metrics['train/rank@1'][i] = batch_history['rank'].detach()
                metrics['train/precision'][i] = batch_history['precision'].detach()
                # update progress bar
                msg = f":ocean: [{i+1}/{steps}]: " + \
                    " | ".join([f"{k} : {self.nanmean(v[:i+1]).item():.4f}" for k, v in metrics.items()])
                pbar.update(job, advance=1., description=msg)
                pbar.refresh()

        train_history = {k: self.nanmean(v).item() for k, v in metrics.items()}
        
        return train_history

    def train_step(self, batch: dict) -> typing.Dict[str, torch.Tensor]:
        """..."""
        with torch.cuda.amp.autocast(self.amp_scaler is not None):
            # fetch three positive views; {query, key, teacher}
            x_q = batch['x1'].to(self.local_rank, non_blocking=True)
            x_k = batch['x2'].to(self.local_rank, non_blocking=True)
            x_t = batch['x3'].to(self.local_rank, non_blocking=True)
            # compute query features; (B, f)
            z_q = F.normalize(self.net_q(x_q), dim=1)
            with torch.no_grad():
                # update key network; exponential moving average of query network
                self._momentum_update_key_net()
                # update teacher network; exponential moving average of teacher network
                self._momentum_update_teacher_net()
                # Shuffle across gpus
                x_k, idx_unshuffle_k = ForMoCo.batch_shuffle_ddp(x_k)
                x_t, idx_unshuffle_t = ForMoCo.batch_shuffle_ddp(x_t)
                # Compute {key, teacher} features; (B, f)
                z_k = F.normalize(self.net_k(x_k), dim=1)
                z_t = F.normalize(self.net_t(x_t), dim=1)
                # Restore {key, teacher} devices to their original nodes
                z_k  = ForMoCo.batch_unshuffle_ddp(z_k, idx_unshuffle_k)
                z_t = ForMoCo.batch_unshuffle_ddp(z_t, idx_unshuffle_t)
            
            # loss & metrics
            loss, logits, false_negative_mask = self.criterion(
                query=z_q, key=z_k, teacher=z_t, negatives=self.queue.buffer
            )
            y = batch['y'].to(self.local_rank, non_blocking=True).detach()
            rank = TopKAccuracy(k=1)(logits.detach(), torch.zeros_like(y))
            precision = self.teacher_precision(y, self.queue.labels, false_negative_mask)

            # Backpropagte & update
            self.backprop(loss)
            # Update memory queue
            self.queue.update(keys=z_k,
                              indices=batch['idx'].to(self.local_rank),
                              labels=y)
        
        return dict(loss=loss, rank=rank, precision=precision)

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

    def _set_learning_phase(self, train: bool):
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



