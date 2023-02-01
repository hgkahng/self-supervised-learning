# -*- coding: utf-8 -*-

"""
    Image transformations used in contrastive self-supervised learning.
    Implementations support the followings:
        1. torchvision    (tensor-based + PIL-based)
        2. albumentations (cv2-based)
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import albumentations as A

from datasets.transforms.base import ImageAugment
from datasets.transforms.pil_based import GaussianBlur
from datasets.transforms.albumentations import NumpyToTensor


def apply_blur(data: str) -> bool:
    if data.startswith('cifar') or data.startswith('svhn'):
        return False
    else:
        return True


class WeakAugment(ImageAugment):
    def __init__(self,
                 size: int or tuple = (224, 224),
                 data: str = 'imagenet',
                 impl: str = 'torchvision',
                 **kwargs):
        super(WeakAugment, self).__init__(size, data, impl)

        if self.impl == 'torchvision':
            self.transform = self.with_torchvision()
        else:
            raise NotImplementedError

    def with_torchvision(self):
        """
        Weak augmentation with torchvision, expects `torch.tensor`s as input.
        Operations stay on tensors, thus can run on CUDA gpus.
        """
        transform = [
            T.RandomHorizontalFlip(0.5),
            T.RandomCrop(size=self.size,
                         padding=int(self.size[0] * 0.125),
                         padding_mode='reflect'),
            T.ConvertImageDtype(torch.float),
            T.Normalize(self.mean, self.std)
        ]
        return nn.Sequential(*transform)

    def with_torchvision_pil(self):
        """Weak augmentation with torchvision."""
        transform = [
            T.ToPILImage(),
            T.RandomHorizontalFlip(0.5),
            T.RandomCrop(size=self.size,
                         padding=int(self.size[0] * 0.125),
                         padding_mode='reflect'),
            T.ToTensor(),
            T.Normalize(self.mean, self.std)
        ]
        return T.Compose(transform)

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
                 impl: str = 'torchvision',
                 **kwargs):
        super(MoCoAugment, self).__init__(size, data, impl)
        
        self.blur = apply_blur(self.data)
        if self.impl == 'torchvision':
            self.transform = self.with_torchvision()
        elif self.impl == 'albumentations':
            self.transform = self.with_albumentations()

    def with_torchvision(self) -> nn.Module:
        """
        MoCo-v2-style augmentation with torchvision, expects `torch.tensor`s as input.
        Operations stay on tensors, thus can run on CUDA gpus.
        """
        transform = [
            T.RandomResizedCrop(self.size, scale=(0.2, 1.0)),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2)
        ]
        if self.blur:
            transform += [
                T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=.5),  # TODO: check `kernel_size'
            ]
        transform += [
            T.RandomHorizontalFlip(0.5),
            T.ConvertImageDtype(torch.float),
            T.Normalize(self.mean, self.std)
        ]
        return nn.Sequential(*transform)

    def with_torchvision_pil(self):
        """MoCo_v2-style augmentation with torchvision, expects `np.ndarray` or `torch.tensor`s."""
        transform = [
            T.ToPILImage(),
            T.RandomResizedCrop(self.size, scale=(0.2, 1.0)),
            T.RandomApply([
                T.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
        ]
        if self.blur:
            transform += [
                T.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5)
            ]
        transform += [
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(self.mean, self.std)
        ]
        return T.Compose(transform)

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
                 impl: str = 'torchvision',
                 **kwargs):
        super(SimCLRAugment, self).__init__(size, data, impl)
        
        self.blur = apply_blur(self.data)
        if self.impl == 'torchvision':
            self.transform = self.with_torchvision()
        elif self.impl == 'albumentations':
            self.transform = self.with_albumentations()

    def with_torchvision(self) -> nn.Module:
        """
        SimCLR-style augmentation with torchvision,, expects `torch.tensor`s as input.
        Operations stay on tensors, thus can run on CUDA gpus.
        """
        transform = [
            T.RandomResizedCrop(self.size, scale=(0.08, 1.00)),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
        ]
        if self.blur:
            transform += [
                T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5)
            ]
        transform += [
            T.RandomHorizontalFlip(0.5),
            T.ConvertImageDtype(torch.float),
            T.Normalize(self.mean, self.std)
        ]
        return nn.Sequential(*transform)

    def with_torchvision_pil(self):
        """SimCLR-style augmentation with torchvision, expects `np.ndarray` or `torch.tensor`s."""
        transform = [
            T.ToPILImage(),
            T.RandomResizedCrop(self.size, scale=(0.08, 1.00)),
            T.RandomApply([
                T.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
        ]
        if self.blur:
            transform += [
                T.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5)
            ]
        transform += [
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(self.mean, self.std)
        ]
        return T.Compose(transform)

    def with_albumentations(self):
        """SimCLR-style augmentation with torchvision."""
        transform = [
            A.RandomResizedCrop(*self.size, scale=(0.08, 1.00), ratio=(3/4, 4/3)),
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
