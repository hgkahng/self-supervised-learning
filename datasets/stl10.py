# -*- coding: utf-8 -*-

import typing
import warnings

import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import STL10 as _STL10


class STL10(_STL10):
    num_classes = 10
    def __init__(self,
                 root: str = './data/stl10',
                 split: str = 'train+unlabeled',
                 transform: nn.Module = None,
                 **kwargs):
        if split not in ['train', 'test', 'unlabeled', 'train+unlabeled']:
            raise ValueError(f"Not a valid split: `{split}`.")
        super(STL10, self).__init__(root=root,
                                    split=split,
                                    folds=None,
                                    transform=transform,
                                    target_transform=None,
                                    download=False)

        assert isinstance(self.labels, np.ndarray)
        assert isinstance(self.data, np.ndarray)
        assert self.data.shape[1] == 3, "Shape of data: BCHW"
        self.data = torch.from_numpy(self.data)
        self.labels = torch.from_numpy(self.labels)

        if 'proportion' in kwargs:
            warnings.warn("Argument 'proportion' not valid for STL-10.", UserWarning)

    def __getitem__(self, idx: int) -> typing.Dict[str,torch.Tensor]:
        img, target = self.data[idx], int(self.labels[idx])
        if self.transform is not None:
            img = self.transform(img)
        return dict(x=img, y=target, idx=idx)


class STL10Pair(STL10):
    def __init__(self,
                 root: str = './data/stl10',
                 split: str = 'train+unlabeled',
                 transform: nn.Module = None):
        super(STL10Pair, self).__init__(root=root,
                                        split=split,
                                        transform=transform)

    def __getitem__(self, idx: int) -> typing.Dict[str,torch.Tensor]:
        img, target = self.data[idx], int(self.labels[idx])
        x1 = self.transform(img) if self.transform is not None else img
        x2 = self.transform(img) if self.transform is not None else img
        return dict(x1=x1, x2=x2, y=target, idx=idx)


class STL10ForMoCo(STL10):
    def __init__(self,
                 root: str = './data/stl10',
                 split: str = 'train+unlabeled',
                 query_transform: nn.Module = None,
                 key_transform: nn.Module = None):
        super(STL10ForMoCo, self).__init__(root=root, split=split)

        self.query_transform = query_transform
        self.key_transform = key_transform

    def __getitem__(self, idx: int) -> typing.Dict[str,torch.Tensor]:
        img, target = self.data[idx], int(self.labels[idx])
        x1 = self.query_transform(img) if self.query_transform is not None else img
        x2 = self.key_transform(img) if self.key_transform is not None else img
        return dict(x1=x1, x2=x2, y=target, idx=idx)


class STL10ForCLAP(STL10):
    def __init__(self,
                 root: str = './data/stl10',
                 split: str = 'train+unlabeled',
                 query_transform: object = None,
                 key_transform: object = None,
                 teacher_transform: object = None):
        super(STL10ForCLAP, self).__init__(root=root, split=split)

        self.query_transform = query_transform
        self.key_transform = key_transform
        self.teacher_transform = teacher_transform

    def __getitem__(self, idx: int) -> typing.Dict[str,torch.Tensor]:
        img, label = self.data[idx], int(self.labels[idx])
        x1 = self.query_transform(img) if self.query_transform is not None else img
        x2 = self.key_transform(img) if self.key_transform is not None else img
        x3 = self.teacher_transform(img) if self.teacher_transform is not None else img
        return dict(x1=x1, x2=x2, x3=x3, y=label, idx=idx)


STL10ForSimCLR = STL10Pair
