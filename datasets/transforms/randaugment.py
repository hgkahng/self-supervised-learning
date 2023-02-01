# -*- coding: utf-8 -*-

import torch.nn as nn
import torchvision.transforms as T
import albumentations as A

from datasets.transforms.base import ImageAugment
from datasets.transforms.pil_based import RandAugmentTv
from datasets.transforms.albumentations import NumpyToTensor
from datasets.transforms.albumentations import RandAugmentAlb


class RandAugment(ImageAugment):
    def __init__(self,
                 size: int or tuple = (224, 224),
                 data: str = 'imagenet',
                 impl: str = 'torchvision',
                 k: int = 5,
                 **kwargs):
        super(RandAugment, self).__init__(size, data, impl)
        
        self.k = k
        self.scale = kwargs.get('scale', (0.2, 1.0))
        
        if self.impl == 'torchvision':
            self.transform = self.with_torchvision()
        elif self.impl == 'albumentations':
            self.transform = self.with_albumentations()

    def with_torchvision_tensor_ops(self):
        raise NotImplementedError

    def with_torchvision(self):
        """RandAugment based on torchvision."""
        transform = [
            T.ToPILImage(),
            T.RandomResizedCrop(self.size, scale=self.scale),
            T.RandomHorizontalFlip(0.5),
            RandAugmentTv(k=self.k),
            T.ToTensor(),
            T.Normalize(self.mean, self.std)
        ]
        return T.Compose(transform)

    def with_albumentations(self):
        """RandAugment based on albumentations"""
        transform = [
            A.RandomResizedCrop(*self.size, scale=self.scale),
            A.HorizontalFlip(0.5),
            RandAugmentAlb(k=self.k),
            A.Normalize(self.mean, self.std, always_apply=True),
            NumpyToTensor()
        ]
        return A.Compose(transform)
