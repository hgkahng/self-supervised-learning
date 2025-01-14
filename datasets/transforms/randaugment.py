
import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
import albumentations as A

from datasets.transforms.base import ImageAugment
from datasets.transforms.pil_based import RandAugmentTv
from datasets.transforms.albumentations import NumpyToTensor
from datasets.transforms.albumentations import RandAugmentAlb


class RandAugment(ImageAugment):
    def __init__(self,
                 size: int | tuple = (224, 224),
                 data: str = 'imagenet',
                 impl: str = 'torchvision',
                 k: int = 5,
                 **kwargs):
        super().__init__(size, data, impl)
        
        self.k = k
        if 'scale' in kwargs:
            self.scale = kwargs['scale']
        else:
            self.scale = (0.2, 1.0)
        
        if self.impl == 'torchvision':
            self.transform = self.with_torchvision()
        elif self.impl == 'albumentations':
            self.transform = self.with_albumentations()

    def with_torchvision_tensor_ops(self):
        raise NotImplementedError

    def with_torchvision(self):
        """RandAugment based on `torchvision`."""
        transform = [
            v2.RandomResizedCrop(self.size, scale=self.scale),
            v2.RandomHorizontalFlip(0.5),
            v2.RandAugment(self.k),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(self.mean, self.std)
        ]
        return v2.Compose(transform)

    def with_albumentations(self):
        """RandAugment based on `albumentations`."""
        transform = [
            A.RandomResizedCrop(*self.size, scale=self.scale),
            A.HorizontalFlip(0.5),
            RandAugmentAlb(k=self.k),
            A.Normalize(self.mean, self.std, always_apply=True),
            NumpyToTensor()
        ]
        return A.Compose(transform)
