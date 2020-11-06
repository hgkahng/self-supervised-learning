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
    def forward(self, x):
        return x.view(x.size(0), -1)


class Interpolate(nn.Module):
    def __init__(self, output_size: tuple, mode='nearest'):
        super(Interpolate, self).__init__()
        self.output_size = output_size
    def forward(self, x):
        return F.interpolate(x.float(), self.output_size, mode=self.mode)
