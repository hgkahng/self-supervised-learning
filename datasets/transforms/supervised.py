# -*- coding: utf-8 -*-

"""
    Image augmentations for supervised learning.
"""

import albumentations as A
from torchvision import transforms
from datasets.transforms.base import ImageAugment
from datasets.transforms.albumentations import NumpyToTensor


def get_evaluation_crop_torchvision(size: tuple, data: str):
    """..."""
    if isinstance(size, int):
        size = (size, size)
    if data == 'imagenet':
        assert size == (224, 224)
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(size),
        ])
    else:  # cifar10(32), cifar100(32), svhn(32), stl10(96), tinyimagenet(64)
        return transforms.RandomCrop(size=size,
                                     padding=int(size[0] * 0.125),
                                     padding_mode='reflect')


def get_evaluation_crop_albumentations(size: tuple, data: str):
    """..."""
    raise NotImplementedError


class FinetuneAugment(ImageAugment):
    def __init__(self,
                 size: int or tuple = (224, 224),
                 data: str = 'imagenet',
                 impl: str = 'torchvision',
                 **kwargs):
        super(FinetuneAugment, self).__init__(size, data, impl)

        if self.impl == 'torchvision':
            self.transform = self.with_torchvision()
        elif self.impl == 'albumentations':
            self.transform = self.with_albumentations()

    def with_torchvision(self):
        transform = [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(0.5),
            get_evaluation_crop_torchvision(self.size, self.data),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ]
        return transforms.Compose(transform)

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
                 size: int or tuple = (224, 224),
                 data: str = 'imagenet',
                 impl: str = 'torchvision'):
        super(TestAugment, self).__init__(size, data, impl)

        if self.impl == 'torchvision':
            self.transform = self.with_torchvision()
        elif self.impl == 'albumentations':
            self.transform = self.with_albumentations()

    def with_torchvision(self):
        transform = [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ]
        return transforms.Compose(transform)

    def with_albumentations(self):
        transform = [
            A.Normalize(self.mean, self.std, always_apply=True),
            NumpyToTensor()
        ]
        return A.Compose(transform)
