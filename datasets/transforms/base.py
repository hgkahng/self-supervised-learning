# -*- coding: utf-8 -*-

"""
    Base superclass `ImageAugment` for defining RGB image-based augmentations.
"""

import numpy as np
import albumentations as A
import torch
import torch.nn as nn
from torchvision import transforms


class ImageAugment(nn.Module):
    
    SUPPORTED_DATASETS = [
        'cifar10',
        'cifar100',
        'svhn',
        'stl10',
        'tinyimagenet',
        'imagenet'
    ]

    MEAN = {
        'cifar10':      [0.4914, 0.4822, 0.4465],
        'cifar100':     [0.5071, 0.4867, 0.4408],
        'svhn':         [0.4359, 0.4420, 0.4709],
        'stl10':        [0.485,  0.456,  0.406],
        'tinyimagenet': [0.485,  0.456,  0.406],
        'imagenet':     [0.485,  0.456,  0.406],
    }

    STD = {
        'cifar10':      [0.247, 0.243, 0.261],
        'cifar100':     [0.268, 0.257, 0.276],
        'svhn':         [0.197, 0.200, 0.196],
        'stl10':        [0.229, 0.224, 0.225],
        'tinyimagenet': [0.229, 0.224, 0.225],
        'imagenet':     [0.229, 0.224, 0.225]
    }

    def __init__(self,
                 size: int or tuple,
                 data: str,
                 impl: str ='torchvision'):
        super(ImageAugment, self).__init__()

        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple):
            self.size = size

        if data not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset: '{data}'.")
        self.data = data

        if impl not in ['torchvision', 'albumentations']:
            raise ValueError(f"Currently only supports 'torchvision'.")
        self.impl = impl
        self.transform: nn.Module = None

    @torch.no_grad()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.transform(img)
    
    @staticmethod
    def forward_albumentations(img: np.ndarray):
        raise NotImplementedError
        # return self.transform(image=img)['image']

    def with_torchvision(self, size: tuple, blur: bool = True):
        raise NotImplementedError

    def with_albumentations(self, size: tuple, blur: bool = True):  # TODO; deprecated, move to legacy files
        raise NotImplementedError

    @property
    def mean(self):
        return self.MEAN[self.data]

    @property
    def std(self):
        return self.STD[self.data]
