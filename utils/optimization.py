# -*- coding: utf-8 -*-

import math
import warnings

from typing import Iterable, List

import torch
import torch.optim as optim
from torch.optim import Optimizer, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, MultiStepLR, LambdaLR, _LRScheduler


class LARS(Optimizer):
    """
    Implementation of `Large Batch Training of Convolutional Networks, You et al`.
    Code is borrowed from the following:
        https://github.com/cmpark0126/pytorch-LARS
        https://github.com/noahgolmant/pytorch-lars
    """
    def __init__(self,
                 params,
                 lr: float,
                 momentum: float = 0.9,
                 weight_decay: float = 0., 
                 trust_coef: float = 1e-3,
                 dampening: float = 0.,
                 nesterov: bool = False):
        if lr < 0.:
            raise ValueError(f"Invalid `lr` value: {lr}")
        if momentum < 0.:
            raise ValueError(f"Invalid `momentum` value: {momentum}")
        if weight_decay < 0.:
            raise ValueError(f"Invalid `weight_decay` value: {weight_decay}")
        if trust_coef < 0.:
            raise ValueError(f"Invalid `trust_coef` value: {trust_coef}")

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening.")

        defaults = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay,
            trust_coef=trust_coef, dampening=dampening, nesterov=nesterov)

        super(LARS, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LARS, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step."""

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum     = group['momentum']
            dampening    = group['dampening']
            nesterov     = group['nesterov']
            trust_coef   = group['trust_coef']
            global_lr    = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                p_norm = torch.norm(p.data, p=2)  # weight norm
                d_p_norm = torch.norm(d_p, p=2)   # gradient norm

                local_lr = torch.div(p_norm, d_p_norm + weight_decay * p_norm)
                local_lr.mul_(trust_coef)

                actual_lr = local_lr * global_lr

                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay).mul(actual_lr)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.data.add_(d_p, alpha=-1)

        return loss


class SGDW(Optimizer):
    def __init__(self,
                 params,
                 lr: float,
                 momentum: float = 0.,
                 dampening: float = 0.,
                 weight_decay: float = 0.,
                 nesterov: bool = False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov
        )

        if nesterov and (momentum <= 0. or dampening != 0.):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening factor.")

        super(SGDW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                old = torch.clone(p.data).detach()
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)  # update current gradient with updated velocity
                    else:
                        d_p = buf                     # use current updated velocity as gradient

                p.data.add_(-group['lr'], d_p)

                if weight_decay != 0.:
                    p.data.add_(-weight_decay, old)

        return loss


class LinearWarmupCosineDecay(LambdaLR):
    """Linear warmup & cosine decay.
       Implementation from `pytorch_transformers.optimization.WarmupCosineSchedule`.
       Assuming that the initial learning rate of `optimizer` is set to 1., this scheduler
       linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
       Decreases learning rate for 1. to 0. over remaining `t_total - warmup_steps` following a cosine curve.
       If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self,
                 optimizer,
                 warmup_steps: int,
                 t_total: int,
                 cycles: float = 0.5,
                 min_lr: float = 0.,
                 last_epoch: int = -1):
        
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        self.min_lr = min_lr
        self.base_lr = optimizer.defaults['lr']
        
        super(LinearWarmupCosineDecay, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step: int):
        """A lambda function used as argument for `LambdaLR`."""
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        else:
            progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
            mul = max(0., 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
            if self.expected_lr(mul) < self.min_lr:
                return self.min_lr / self.base_lr
            else:
                return mul

    def expected_lr(self, mul: float):
        return self.base_lr * mul


class LinearWarmupCosineDecayWithRestarts(LambdaLR):
    """ Linear warmup and then cosine cycles with hard restarts.
        Implementation from `pytorch_transformers.optimization.WarmupCosineWithHardRestartsSchedule`.
        Assuming that the initial learning rate of `optimizer` is set to 1., this scheduler
        linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        If `cycles` (default=1.) is different from default, learning rate follows `cycles` times a cosine decaying
        learning rate (with hard restarts).
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=1., last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(LinearWarmupCosineDecayWithRestarts, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step: int):
        """A lambda function used as argument for `LambdaLR`."""
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1. + math.cos(math.pi * ((float(self.cycles) * progress) % 1.0))))


class WarmupCosineDecayLRWithRestarts(_LRScheduler):
    def __init__(self,
                 optimizer: Optimizer,
                 total_epochs: int,
                 restart_epochs: list,
                 warmup_epochs: int = 0,
                 warmup_start_lr: float = 0.,
                 min_decay_lr: float = 0.,
                 last_epoch: int = -1,
                 verbose: bool = False):

        self._total_epochs: int = total_epochs
        self._restart_epochs: list = sorted(restart_epochs)
        self._warmup_epochs: int = warmup_epochs
        self._warmup_start_lrs = [warmup_start_lr] * len(optimizer.param_groups)
        self._min_decay_lrs = [min_decay_lr] * len(optimizer.param_groups)
        super(WarmupCosineDecayLRWithRestarts, self).__init__(optimizer,
                                                              last_epoch=last_epoch,
                                                              verbose=verbose)

    def get_lr(self):
        """Compute learning rate."""
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr().`", UserWarning)
        if self._warmup_epochs > 0 and self.last_epoch < self._warmup_epochs:
            return [
                self.linear_warmup_lr(self.last_epoch, self._warmup_epochs, base_lr, min_lr)
                for base_lr, min_lr in zip(self.base_lrs, self._warmup_start_lrs)
            ]
        else:
            if (self.last_epoch < self._restart_epochs[0]) and \
                (self.last_epoch >= self._warmup_epochs):
                remaining_epochs = self._total_epochs - self._warmup_epochs
                last_epoch = self.last_epoch
            elif (self.last_epoch >= self._restart_epochs[0]) and \
                (self.last_epoch < self._restart_epochs[1]):
                remaining_epochs = self._total_epochs - self._restart_epochs[0]
                last_epoch = self.last_epoch - self._restart_epochs[0]
            elif self.last_epoch >= self._restart_epochs[1]:
                remaining_epochs = self._total_epochs - self._restart_epochs[1]
                last_epoch = self.last_epoch - self._restart_epochs[1]
            else:
                raise NotImplementedError
            return [
                self.cosine_decay_lr(last_epoch, remaining_epochs, base_lr, min_lr)
                for base_lr, min_lr in zip(self.base_lrs, self._min_decay_lrs)
            ]        

    @staticmethod
    def linear_warmup_lr(t: int, T: int, base_lr: float, min_lr: float):
        return min_lr + (base_lr - min_lr) * (t / T)
    
    @staticmethod
    def cosine_decay_lr(t: int, T: int, base_lr: float, min_lr: float):
        return min_lr + (base_lr - min_lr) * ((1 + math.cos(math.pi * t / T)) / 2)


class WarmupCosineDecayLR(_LRScheduler):
    def __init__(self,
                 optimizer: Optimizer,
                 total_epochs: int,
                 warmup_epochs: int = 0,
                 warmup_start_lr: float = .0,
                 min_decay_lr: float = .0,
                 last_epoch: int = -1,
                 verbose: bool = False):
        """
        Args:
            optimizer: torch.optim.Optimizer, wrapped optimizer.
            total_epochs: int, total number of training epochs.
            warmup_epochs: int, (default = 0).
            warmup_start_lr: float, initial learning rate when warmup starts (default=0.0). 
            min_decay_lr: float, learning rate reached at step `total_epochs' (default=0.0).
            last_epoch: int (default=-1).
            verbose: bool (default=False).
        """
        self._total_epochs = total_epochs
        self._warmup_epochs = warmup_epochs
        self._warmup_start_lrs = [warmup_start_lr] * len(optimizer.param_groups)
        self._min_decay_lrs = [min_decay_lr] * len(optimizer.param_groups)
        super(WarmupCosineDecayLR, self).__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        """Compute learning rate."""
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        if self._warmup_epochs > 0 and self.last_epoch < self._warmup_epochs:
            return [
                self.linear_warmup_lr(self.last_epoch, self._warmup_epochs, base_lr, min_lr)
                for base_lr, min_lr in zip(self.base_lrs, self._warmup_start_lrs)
            ]
        else:
            remaining_epochs = self._total_epochs - self._warmup_epochs
            return [
                self.cosine_decay_lr(self.last_epoch, remaining_epochs, base_lr, min_lr)
                for base_lr, min_lr in zip(self.base_lrs, self._min_decay_lrs)
            ]

    @staticmethod
    def linear_warmup_lr(t: int, T: int, base_lr: float, min_lr: float):
        return min_lr + (base_lr - min_lr) * (t / T)

    @staticmethod
    def cosine_decay_lr(t: int, T: int, base_lr: float, min_lr: float):
        return min_lr + (base_lr - min_lr) * ((1 + math.cos(math.pi * t / T)) / 2)


class CyclicCosineDecayLR(_LRScheduler):
    """
    Implementation taken from:
        https://github.com/abhuse/cyclic-cosine-decay/blob/master/scheduler.py
    """
    def __init__(self,
                 optimizer,
                 init_decay_epochs,
                 min_decay_lr,
                 restart_interval=None,
                 restart_interval_multiplier=None,
                 restart_lr=None,
                 warmup_epochs=None,
                 warmup_start_lr=None,
                 last_epoch=-1,
                 verbose=False):
        """
        Initialize new CyclicCosineDecayLR object.
        :param optimizer: (Optimizer) - Wrapped optimizer.
        :param init_decay_epochs: (int) - Number of initial decay epochs.
        :param min_decay_lr: (float or iterable of floats) - Learning rate at the end of decay.
        :param restart_interval: (int) - Restart interval for fixed cycles.
            Set to None to disable cycles. Default: None.
        :param restart_interval_multiplier: (float) - Multiplication coefficient for geometrically increasing cycles.
            Default: None.
        :param restart_lr: (float or iterable of floats) - Learning rate when cycle restarts.
            If None, optimizer's learning rate will be used. Default: None.
        :param warmup_epochs: (int) - Number of warmup epochs. Set to None to disable warmup. Default: None.
        :param warmup_start_lr: (float or iterable of floats) - Learning rate at the beginning of warmup.
            Must be set if warmup_epochs is not None. Default: None.
        :param last_epoch: (int) - The index of the last epoch. This parameter is used when resuming a training job. Default: -1.
        :param verbose: (bool) - If True, prints a message to stdout for each update. Default: False.
        """

        if not isinstance(init_decay_epochs, int) or init_decay_epochs < 1:
            raise ValueError("init_decay_epochs must be positive integer, got {} instead".format(init_decay_epochs))

        if isinstance(min_decay_lr, Iterable) and len(min_decay_lr) != len(optimizer.param_groups):
            raise ValueError("Expected len(min_decay_lr) to be equal to len(optimizer.param_groups), "
                             "got {} and {} instead".format(len(min_decay_lr), len(optimizer.param_groups)))

        if restart_interval is not None and (not isinstance(restart_interval, int) or restart_interval < 1):
            raise ValueError("restart_interval must be positive integer, got {} instead".format(restart_interval))

        if restart_interval_multiplier is not None and \
                (not isinstance(restart_interval_multiplier, float) or restart_interval_multiplier <= 0):
            raise ValueError("restart_interval_multiplier must be positive float, got {} instead".format(
                restart_interval_multiplier))

        if isinstance(restart_lr, Iterable) and len(restart_lr) != len(optimizer.param_groups):
            raise ValueError("Expected len(restart_lr) to be equal to len(optimizer.param_groups), "
                             "got {} and {} instead".format(len(restart_lr), len(optimizer.param_groups)))

        if warmup_epochs is not None:
            if not isinstance(warmup_epochs, int) or warmup_epochs < 1:
                raise ValueError(
                    "Expected warmup_epochs to be positive integer, got {} instead".format(type(warmup_epochs)))

            if warmup_start_lr is None:
                raise ValueError("warmup_start_lr must be set when warmup_epochs is not None")

            if not (isinstance(warmup_start_lr, float) or isinstance(warmup_start_lr, Iterable)):
                raise ValueError("warmup_start_lr must be either float or iterable of floats, got {} instead".format(
                    warmup_start_lr))

            if isinstance(warmup_start_lr, Iterable) and len(warmup_start_lr) != len(optimizer.param_groups):
                raise ValueError("Expected len(warmup_start_lr) to be equal to len(optimizer.param_groups), "
                                 "got {} and {} instead".format(len(warmup_start_lr), len(optimizer.param_groups)))

        group_num = len(optimizer.param_groups)
        self._warmup_start_lr = [warmup_start_lr] * group_num if isinstance(warmup_start_lr, float) else warmup_start_lr
        self._warmup_epochs = 0 if warmup_epochs is None else warmup_epochs
        self._init_decay_epochs = init_decay_epochs
        self._min_decay_lr = [min_decay_lr] * group_num if isinstance(min_decay_lr, float) else min_decay_lr
        self._restart_lr = [restart_lr] * group_num if isinstance(restart_lr, float) else restart_lr
        self._restart_interval = restart_interval
        self._restart_interval_multiplier = restart_interval_multiplier
        super(CyclicCosineDecayLR, self).__init__(optimizer, last_epoch, verbose=verbose)

    def get_lr(self):

        if self._warmup_epochs > 0 and self.last_epoch < self._warmup_epochs:
            return self._calc(self.last_epoch,
                              self._warmup_epochs,
                              self._warmup_start_lr,
                              self.base_lrs)

        elif self.last_epoch < self._init_decay_epochs + self._warmup_epochs:
            return self._calc(self.last_epoch - self._warmup_epochs,
                              self._init_decay_epochs,
                              self.base_lrs,
                              self._min_decay_lr)
        else:
            if self._restart_interval is not None:
                if self._restart_interval_multiplier is None:
                    cycle_epoch = (self.last_epoch - self._init_decay_epochs - self._warmup_epochs) % self._restart_interval
                    lrs = self.base_lrs if self._restart_lr is None else self._restart_lr
                    return self._calc(cycle_epoch,
                                      self._restart_interval,
                                      lrs,
                                      self._min_decay_lr)
                else:
                    n = self._get_n(self.last_epoch - self._warmup_epochs - self._init_decay_epochs)
                    sn_prev = self._partial_sum(n)
                    cycle_epoch = self.last_epoch - sn_prev - self._warmup_epochs - self._init_decay_epochs
                    interval = self._restart_interval * self._restart_interval_multiplier ** n
                    lrs = self.base_lrs if self._restart_lr is None else self._restart_lr
                    return self._calc(cycle_epoch,
                                      interval,
                                      lrs,
                                      self._min_decay_lr)
            else:
                return self._min_decay_lr

    def _calc(self, t, T, lrs, min_lrs):
        return [min_lr + (lr - min_lr) * ((1 + math.cos(math.pi * t / T)) / 2)
                for lr, min_lr in zip(lrs, min_lrs)]

    def _get_n(self, epoch):
        _t = 1 - (1 - self._restart_interval_multiplier) * epoch / self._restart_interval
        return math.floor(math.log(_t, self._restart_interval_multiplier))

    def _partial_sum(self, n):
        return self._restart_interval * (1 - self._restart_interval_multiplier ** n) / (
                    1 - self._restart_interval_multiplier)


def configure_optimizer(params, name: str, lr: float, weight_decay: float, **kwargs):
    if name == 'sgd':
        return SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif name == 'lars':
        return LARS(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif name == 'adam':
        return AdamW(params=params, lr=lr, weight_decay=weight_decay)
    elif name == 'lamb':
        raise NotImplementedError
    elif name == 'lookahead':
        raise NotImplementedError
    else:
        raise KeyError

def get_optimizer(params, name: str, lr: float, weight_decay: float, **kwargs):
    """Returns an `Optimizer` object given proper arguments."""

    if name == 'adamw':
        return AdamW(params=params, lr=lr, weight_decay=weight_decay)
    elif name == 'sgd':
        return SGD(params=params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif name == 'lars':
        return LARS(params=params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif name == 'lookahead':
        raise NotImplementedError
    else:
        raise NotImplementedError


def get_scheduler(optimizer: optim.Optimizer, name: str, epochs: int, **kwargs):
    """Configure learning rate scheduler."""
    if name == 'step':
        step_size = kwargs.get('milestone', epochs // 10 * 9)
        gamma = kwargs.get('gamma', 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif name == 'cosine':
        warmup_steps = kwargs.get('warmup_steps', epochs // 10)
        return LinearWarmupCosineDecay(optimizer, warmup_steps=warmup_steps, t_total=epochs)
    elif name == 'restart':
        warmup_steps = kwargs.get('warmup_steps', epochs // 10)
        cycles = kwargs.get('cycles', 4)
        return LinearWarmupCosineDecayWithRestarts(optimizer, warmup_steps=warmup_steps, t_total=epochs, cycles=cycles)
    else:
        return None


def get_multi_step_scheduler(optimizer: optim.Optimizer, milestones: list, gamma: float = 0.1):
    return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)


def get_cosine_scheduler(optimizer: optim.Optimizer,
                         epochs: int,
                         warmup_steps: int = 0,
                         cycles: int = 1,
                         min_lr: float = 5e-3):
    """Configure half cosine learning rate schduler."""
    if warmup_steps < 0:
        return None
    if cycles <= 1:
        return LinearWarmupCosineDecay(optimizer, warmup_steps=warmup_steps, t_total=epochs, min_lr=min_lr)
    else:
        return LinearWarmupCosineDecayWithRestarts(optimizer, warmup_steps=warmup_steps, t_total=epochs, cycles=cycles)
