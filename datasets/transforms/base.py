# -*- coding: utf-8 -*-

"""
    Base superclass `ImageAugment` for defining RGB image-based augmentations.
"""

import numpy as np
import albumentations as A
from torchvision import transforms


class ImageAugment(object):
    
    SUPPORTED_DATASETS = ['cifar10', 'cifar100', 'stl10', 'tinyimagenet', 'imagenet']

    MEAN = {
        'cifar10':      [0.4914, 0.4822, 0.4465],
        'cifar100':     [0.5071, 0.4867, 0.4408],
        'stl10':        [0.485,  0.456,  0.406],
        'tinyimagenet': [0.485,  0.456,  0.406],
        'imagenet':     [0.485,  0.456,  0.406],
    }

    STD = {
        'cifar10':      [0.247, 0.243, 0.261],
        'cifar100':     [0.268, 0.257, 0.276],
        'stl10':        [0.229, 0.224, 0.225],
        'tinyimagenet': [0.229, 0.224, 0.225],
        'imagenet':     [0.229, 0.224, 0.225]
    }

    def __init__(self,
                 size: int or tuple,
                 data: str,
                 impl: str):

        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple):
            self.size = size

        if data not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset: '{data}'.")
        self.data = data

        if impl not in ['torchvision', 'albumentations']:
            raise ValueError
        self.impl = impl

    def __call__(self, img: np.ndarray):
        if self.impl == 'torchvision':
            return self.transform(img)
        elif self.impl == 'albumentations':
            return self.transform(image=img)['image']
        else:
            raise NotImplementedError

    def with_torchvision(self, size: tuple, blur: bool = True):
        raise NotImplementedError

    def with_albumentations(self, size: tuple, blur: bool = True):
        raise NotImplementedError

    @property
    def mean(self):
        return self.MEAN[self.data]

    @property
    def std(self):
        return self.STD[self.data]
