# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class Lambda(nn.Module):
    def __init__(self, lambda_func):
        super(Lambda, self).__init__()
        self.lambda_func = lambda_func
    def forward(self, x):
        return self.lambda_func(x)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1).contiguous()


class Interpolate(nn.Module):
    def __init__(self, output_size: tuple, mode: str = 'nearest'):
        super(Interpolate, self).__init__()
        self.output_size = output_size
        self.mode = mode
    def forward(self, x):
        return F.interpolate(x.float(), self.output_size, mode=self.mode)


class STGumbelSoftmax(nn.Module):
    """
    Gumbel softmax with straight-throught estimator for gradients.
    Links:
        https://arxiv.org/abs/1611.01144
        https://fabianfuchsml.github.io/gumbel/
    """
    def __init__(self):
        super(STGumbelSoftmax, self).__init__()
    
    def forward(self, x: torch.FloatTensor, tau: float = 1., dim: int = -1):
        return F.gumbel_softmax(x, tau=tau, hard=True, dim=dim)        
