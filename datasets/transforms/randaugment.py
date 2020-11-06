# -*- coding: utf-8 -*-

import albumentations as A
from torchvision import transforms

from datasets.transforms.base import ImageAugment
from datasets.transforms.torchvision import RandAugmentTv
from datasets.transforms.albumentations import NumpyToTensor
from datasets.transforms.albumentations import RandAugmentAlb


class RandAugment(ImageAugment):
    def __init__(self,
                 size: int or tuple = (224, 224),
                 data: str = 'imagenet',
                 impl: str = 'torchvision',
                 **kwargs):
        super(RandAugment, self).__init__(size, data, impl)
        
        self.k = kwargs.get('k', 5)
        self.scale = kwargs.get('scale', (0.2, 1.0))
        
        if self.impl == 'torchvision':
            self.transform = self.with_torchvision()
        elif self.impl == 'albumentations':
            self.transform = self.with_albumentations()

    def with_torchvision(self):
        """RandAugment based on torchvision."""
        transform = [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(self.size, scale=self.scale),  # TODO: necessary?
            transforms.RandomHorizontalFlip(0.5),
            RandAugmentTv(k=self.k),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ]
        return transforms.Compose(transform)

    def with_albumentations(self):
        """RandAugment based on albumentations"""
        transform = [
            A.RandomResizedCrop(*self.size, scale=self.scale),  # TODO: necessary?
            A.HorizontalFlip(0.5),
            RandAugmentAlb(k=self.k),
            A.Normalize(self.mean, self.std, always_apply=True),
            NumpyToTensor()
        ]
        return A.Compose(transform)
