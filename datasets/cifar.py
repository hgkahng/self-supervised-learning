# -*- coding: utf-8 -*-

import cv2
import numpy as np

from torchvision.datasets.cifar import CIFAR10 as _CIFAR10
from torchvision.datasets.cifar import CIFAR100 as _CIFAR100
from sklearn.model_selection import train_test_split


class CIFAR10(_CIFAR10):
    num_classes = 10
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: object = None,
                 proportion: float = 1.0,
                 **kwargs):
        super(CIFAR10, self).__init__(root=root,
                                      train=train,
                                      transform=transform,
                                      target_transform=None,
                                      download=False)

        assert isinstance(self.data, np.ndarray)
        assert isinstance(self.targets, list)

        self.proportion = proportion
        if self.proportion < 1.0:
            indices, _ = train_test_split(
                np.arange(len(self.data)),
                train_size=self.proportion,
                stratify=self.targets,
                shuffle=True,
                random_state=2020 + kwargs.get('seed', 0)
            )
            self.data = self.data[indices]
            self.targets = list(np.array(self.targets)[indices])
        else:
            pass

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        if self.transform is not None:
            img = self.transform(img)

        return dict(x=img, y=target, idx=idx)

    @staticmethod
    def load_image_cv2(path: str):
        """Load image with opencv."""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            raise NotImplementedError


class CIFAR100(_CIFAR100):
    num_classes = 100
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: object = None,
                 proportion: float = 1.0,
                 **kwargs):
        super(CIFAR100, self).__init__(root=root,
                                       train=train,
                                       transform=transform,
                                       target_transform=None,
                                       download=False)

        assert isinstance(self.data, np.ndarray)
        assert isinstance(self.targets, list)

        self.proportion = proportion
        if self.proportion < 1.0:
            indices, _ = train_test_split(
                train_size=self.proportion,
                stratify=self.targets,
                shuffle=True,
                random_state=2021 + kwargs.get('seed', 0)
            )
            self.data = self.data[indices]
            self.targets = list(np.array(self.targets)[indices])
        else:
            pass

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        if self.transform is not None:
            img = self.transform(img)

        return dict(x=img, y=target, idx=idx)


class CIFAR10Pair(_CIFAR10):
    def __init__(self, root: str , train: bool, transform: object):
        super(CIFAR10Pair, self).__init__(root=root,
                                          train=train,
                                          transform=transform,
                                          target_transform=None,
                                          download=False)

    def __getitem__(self, idx):

        img, target = self.data[idx], self.targets[idx]

        if self.transform is not None:
            x1 = self.transform(img)
            x2 = self.transform(img)

        return dict(x1=x1, x2=x2, y=target, idx=idx)


class CIFAR100Pair(_CIFAR100):
    def __init__(self, root: str , train: bool, transform: object):
        super(CIFAR100Pair, self).__init__(root=root,
                                           train=train,
                                           transform=transform,
                                           target_transform=None,
                                           download=False)

    def __getitem__(self, idx):

        img, target = self.data[idx], self.targets[idx]

        if self.transform is not None:
            x1 = self.transform(img)
            x2 = self.transform(img)

        return dict(x1=x1, x2=x2, y=target, idx=idx)


class CIFAR10ForMoCo(_CIFAR10):
    def __init__(self,
                 root: str,
                 train: bool,
                 query_transform: object = None,
                 key_transform: object = None):
        super(CIFAR10ForMoCo, self).__init__(root=root,
                                             train=train,
                                             transform=None,
                                             target_transform=None,
                                             download=False)
        self.query_transform = query_transform
        self.key_transform = key_transform

    def __getitem__(self, idx):

        img, target = self.data[idx], self.targets[idx]

        x1 = self.query_transform(img)
        x2 = self.key_transform(img)

        return dict(x1=x1, x2=x2, y=target, idx=idx)


class CIFAR100ForMoCo(_CIFAR100):
    def __init__(self,
                 root: str,
                 train: bool,
                 query_transform: object = None,
                 key_transform: object = None):
        super(CIFAR100ForMoCo, self).__init__(root=root,
                                              train=train,
                                              transform=None,
                                              target_transform=None,
                                              download=False)
        self.query_transform = query_transform
        self.key_transform = key_transform

    def __getitem__(self, idx):

        img, target = self.data[idx], self.targets[idx]

        x1 = self.query_transform(img)
        x2 = self.key_transform(img)

        return dict(x1=x1, x2=x2, y=target, idx=idx)


class CIFAR10ForCLAPP(_CIFAR10):
    def __init__(self,
                 root: str,
                 train: bool,
                 query_transform: object = None,
                 key_transform: object = None,
                 pseudo_transform: object = None):
        super(CIFAR10ForCLAPP, self).__init__(root=root,
                                              train=train,
                                              transform=None,
                                              target_transform=None,
                                              download=False)
        self.query_transform  = query_transform
        self.key_transform    = key_transform
        self.pseudo_transform = pseudo_transform

    def __getitem__(self, idx):

        img, target = self.data[idx], self.targets[idx]

        x1 = self.query_transform(img)
        x2 = self.key_transform(img)
        x3 = self.pseudo_transform(img)

        return dict(x1=x1, x2=x2, x3=x3, y=target, idx=idx)


class CIFAR100ForCLAPP(_CIFAR100):
    def __init__(self,
                 root: str,
                 train: bool,
                 query_transform: object,
                 key_transform: object,
                 pseudo_transform: object):
        super(CIFAR100ForCLAPP, self).__init__(root=root,
                                               train=train,
                                               transform=None,
                                               target_transform=None,
                                               download=False)
        self.query_transform = query_transform
        self.key_transform = key_transform
        self.pseudo_transform = pseudo_transform

    def __getitem__(self, idx):

        img, target = self.data[idx], self.targets[idx]
        x1 = self.query_transform(img)
        x2 = self.key_transform(img)
        x3 = self.pseudo_transform(img)
        return dict(x1=x1, x2=x2, x3=x3, y=target, idx=idx)


CIFAR10ForSimCLR = CIFAR10Pair
CIFAR100ForSimCLR = CIFAR100Pair
