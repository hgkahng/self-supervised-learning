# -*- coding: utf-8 -*-

"""
    Image augmentations for supervised learning.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import albumentations as A
from datasets.transforms.base import ImageAugment
from datasets.transforms.albumentations import NumpyToTensor


def get_evaluation_crop_torchvision(size: tuple, data: str):
    """..."""
    if isinstance(size, int):
        size = (size, size)
    if data == 'imagenet':
        assert size == (224, 224), "Only supports 224 x 224 for ImageNet."
        return T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop(size),
        ])
    else:  # cifar10(32), cifar100(32), svhn(32), stl10(96), tinyimagenet(64)
        return T.RandomCrop(size=size,
                            padding=int(size[0] * 0.125),
                            padding_mode='reflect')


def get_evaluation_crop_torchvision_tensor_op(size: tuple, data: str):
    """..."""
    if isinstance(size, int):
        size = (size, size)
    if data == 'imagenet':
        if size != (224, 224):
            raise ValueError("Only supports 224 x 224 for ImageNet.")
        return nn.Sequential(T.Resize((256, 256)), T.CenterCrop(size))
    else:
        return T.RandomCrop(size, padding=int(size[0] * 0.125), padding_mode='reflect')


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
        else:
            raise NotImplementedError

    def with_torchvision(self) -> nn.Module:
        transform = [
            T.RandomHorizontalFlip(0.5),
            get_evaluation_crop_torchvision_tensor_op(self.size, self.data),
            T.ConvertImageDtype(torch.float),
            T.Normalize(self.mean, self.std)
        ]
        return nn.Sequential(*transform)

    def with_torchvision_pil(self):
        transform = [
            T.ToPILImage(),
            T.RandomHorizontalFlip(0.5),
            get_evaluation_crop_torchvision(self.size, self.data),
            T.ToTensor(),
            T.Normalize(self.mean, self.std)
        ]
        return T.Compose(transform)

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
                 impl: str = 'torchvision',
                 **kwargs):
        super(TestAugment, self).__init__(size, data, impl)

        if self.impl == 'torchvision':
            self.transform = self.with_torchvision()
        else:
            raise NotImplementedError

    def with_torchvision(self) -> nn.Module:
        transform = [
            get_evaluation_crop_torchvision_tensor_op(size=self.size, data=self.data),
            T.ConvertImageDtype(torch.float),
            T.Normalize(self.mean, self.std)
        ]
        return nn.Sequential(*transform)

    def with_torchvision_pil(self):
        transform = [T.ToPILImage()]
        if self.data == 'imagenet':
            transform += [get_evaluation_crop_torchvision(size=self.size, data=self.data)]
        transform += [
            T.ToTensor(),
            T.Normalize(self.mean, self.std)
        ]
        return T.Compose(transform)

    def with_albumentations(self):
        transform = [
            A.Normalize(self.mean, self.std, always_apply=True),
            NumpyToTensor()
        ]
        return A.Compose(transform)
