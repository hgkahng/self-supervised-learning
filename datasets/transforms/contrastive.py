# -*- coding: utf-8 -*-

"""
    Image transformations used in contrastive self-supervised learning.
    Implementations support the followings:
        1. torchvision    (PIL-based)
        2. albumentations (cv2-based)
"""

import albumentations as A
from torchvision import transforms

from datasets.transforms.base import ImageAugment
from datasets.transforms.torchvision import GaussianBlur
from datasets.transforms.albumentations import NumpyToTensor


class WeakAugment(ImageAugment):
    def __init__(self,
                 size: int or tuple = (224, 224),
                 data: str = 'imagenet',
                 impl: str = 'torchvision'):
        super(WeakAugment, self).__init__(size, data, impl)

        if self.impl == 'torchvision':
            self.transform = self.with_torchvision()
        elif self.impl == 'albumentations':
            self.transform = self.with_albumentations()

    def with_torchvision(self):
        """Weak augmentation with torchvision."""
        transform = [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomCrop(size=self.size,
                                  padding=int(self.size[0] * 0.125),
                                  padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ]
        return transforms.Compose(transform)

    def with_albumentations(self):
        """Weak augmentation with albumentations."""
        transform = [
            A.HorizontalFlip(0.5),
            A.Resize(1.125 * self.size[0], 1.125 * self.size[1], p=1.0),
            A.RandomCrop(*self.size, p=1.0),
            A.Normalize(self.mean, self.std),
            NumpyToTensor()
        ]
        return A.Compose(transform)


class MoCoAugment(ImageAugment):
    def __init__(self,
                 size: int or tuple = (224, 224),
                 data: str = 'imagenet',
                 impl: str = 'torchvision'):
        super(MoCoAugment, self).__init__(size, data, impl)

        self.blur = not self.data.startswith('cifar')  # FIXME: blur for TinyImageNet?
        if self.impl == 'torchvision':
            self.transform = self.with_torchvision()
        elif self.impl == 'albumentations':
            self.transform = self.with_albumentations()

    def with_torchvision(self):
        """MoCo_v2-style augmentation with torchvision."""
        transform = [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(self.size, scale=(0.2, 1.0)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]

        if self.blur:
            transform += [
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5)
            ]

        transform += [
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ]

        return transforms.Compose(transform)

    def with_albumentations(self):
        """MoCo_v2-style augmentation with torchvision."""
        transform = [
            A.RandomResizedCrop(*self.size, scale=(0.2, 1.0), ratio=(3/4, 4/3)),
            A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            A.ToGray(p=0.2),
        ]

        if self.blur:
            blur_limit = min(self.size[0] // 10, self.size[1] // 10)
            if blur_limit % 2 == 0:
                blur_limit += 1  # make odd number
            sigma_limit = (0.1, 2.0)
            transform += [A.GaussianBlur(blur_limit, sigma_limit, p=0.5)]

        transform += [
            A.HorizontalFlip(0.5),
            A.Normalize(self.mean, self.std, always_apply=True),
            NumpyToTensor(),
        ]

        return A.Compose(transform)


class SimCLRAugment(ImageAugment):
    def __init__(self,
                 size: int or tuple = (224, 224),
                 data: str = 'imagenet',
                 impl: str = 'torchvision'):
        super(SimCLRAugment, self).__init__(size, data, impl)

        self.blur = not self.data.startswith('cifar')  # FIXME: blur for TinyImageNet?
        if self.impl == 'torchvision':
            self.transform = self.with_torchvision()
        elif self.impl == 'albumentations':
            self.transform = self.with_albumentations()

    def with_torchvision(self):
        """SimCLR-style augmentation with torchvision."""
        transform = [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(self.size, scale=(0.08, 1.00)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]

        if self.blur:
            transform += [
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5)
            ]

        transform += [
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ]

        return transforms.Compose(transform)

    def with_albumentations(self):
        """SimCLR-style augmentation with torchvision."""
        transform = [
            A.RandomResizedCrop(*self.size, scale=(0.08, 1.00), ratio=(3/4, 4/3)),
            A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            A.ToGray(p=0.2),
        ]

        if blur:
            blur_limit = min(self.size[0] // 10, self.size[1] // 10)
            if blur_limit % 2 == 0:
                blur_limit += 1  # make odd number
            sigma_limit = (0.1, 2.0)
            transform += [A.GaussianBlur(blur_limit, sigma_limit, p=0.5)]

        transform += [
            A.HorizontalFlip(0.5),
            A.Normalize(self.mean, self.std, always_apply=True),
            NumpyToTensor(),
        ]

        return A.Compose(transform)
