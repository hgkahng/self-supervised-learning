
"""
    Image augmentations for supervised learning.
"""

import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
import albumentations as A

from datasets.transforms.base import ImageAugment
from datasets.transforms.albumentations import NumpyToTensor


def get_evaluation_crop_torchvision(size: int | tuple, data: str) -> nn.Module:    
    
    size = (size, size) if isinstance(size, int) else size

    if data == 'imagenet':
        assert size == (224, 224), "Only supports 224 x 224 for ImageNet."
        return v2.Compose([v2.Resize(256), v2.CenterCrop(size)])
    else:  # cifar10(32), cifar100(32), svhn(32), stl10(96), tinyimagenet(64)
        return v2.RandomCrop(
            size=size,
            padding=int(size[0] * 0.125),
            padding_mode='reflect'
        )


def get_evaluation_crop_albumentations(size: tuple, data: str):
    raise NotImplementedError


class FinetuneAugment(ImageAugment):
    def __init__(self,
                 size: int | tuple = (224, 224),
                 data: str = 'imagenet',
                 impl: str = 'torchvision',
                 **kwargs):
        super().__init__(size, data, impl)

        if self.impl == 'torchvision':
            self.transform = self.with_torchvision()
        else:
            raise NotImplementedError

    def with_torchvision(self) -> nn.Module:
        transforms = [
            v2.RandomHorizontalFlip(0.5),
            get_evaluation_crop_torchvision(self.size, self.data),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(self.mean, self.std)
        ]
        return v2.Compose(transforms)

    def with_albumentations(self):
        transform = [
            A.HorizontalFlip(0.5),
            get_evaluation_crop_albumentations(self.size, self.data),
            A.Normalize(self.mean, self.std, always_apply=True),
            NumpyToTensor()
        ]
        return A.Compose(transform)


class TestAugment(ImageAugment):
    def __init__(self,
                 size: int | tuple = (224, 224),
                 data: str = 'imagenet',
                 impl: str = 'torchvision',
                 **kwargs):
        super().__init__(size, data, impl)

        if self.impl == 'torchvision':
            self.transform = self.with_torchvision()
        else:
            raise NotImplementedError

    def with_torchvision(self) -> nn.Module:
        transforms = [
            get_evaluation_crop_torchvision(self.size, self.data),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(self.mean, self.std),
        ]
        return v2.Compose(transforms)

    def with_albumentations(self):
        transform = [
            A.Normalize(self.mean, self.std, always_apply=True),
            NumpyToTensor()
        ]
        return A.Compose(transform)
