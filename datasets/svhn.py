# -*- coding: utf-8 -*-

import cv2
import numpy as np

from torchvision.datasets.svhn import SVHN as _SVHN


class SVHN(_SVHN):
    num_classes = 10
    def __init__(self,
                 root: str,
                 split: str = 'train',
                 transform: object = None,
                 **kwargs):
        super(SVHN, self).__init__(root=root,
                                   split=split,
                                   transform=transform,
                                   download=kwargs.get('download', False))
        
        assert isinstance(self.labels, np.ndarray) and self.labels.ndim == 1
        assert isinstance(self.data, np.ndarray) and self.data.ndim == 4
        self.data = np.transpose(self.data, (0, 2, 3, 1))
    
    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        
        return dict(x=img, y=label, idx=idx)


class SVHNForMoCo(_SVHN):
    def __init__(self,
                 root: str,
                 split: str = 'train',
                 query_transform: object = None,
                 key_transform: object = None):
        super(SVHNForMoCo, self).__init__(root=root,
                                          split=split,
                                          transform=None,
                                          target_transform=None,
                                          download=False)
        self.data = np.transpose(self.data, (0, 2, 3, 1))
        self.query_transform = query_transform
        self.key_transform = key_transform

    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]

        x1 = self.query_transform(img)
        x2 = self.key_transform(img)

        return dict(x1=x1, x2=x2, y=label, idx=idx)


class SVHNForCLAPP(_SVHN):
    def __init__(self,
                 root: str,
                 split: str = 'train',
                 query_transform: object = None,
                 key_transform: object = None,
                 pseudo_transform: object = None):
        super(SVHNForCLAPP, self).__init__(root=root,
                                           split=split,
                                           transform=None,
                                           target_transform=None,
                                           download=False)
        self.data = np.transpose(self.data, (0, 2, 3, 1))
        self.query_transform = query_transform
        self.key_transform   = key_transform
        self.pseudo_transform = pseudo_transform

    def __getitem__(self, idx):

        img, label = self.data[idx], self.labels[idx]

        x1 = self.query_transform(img)
        x2 = self.key_transform(img)
        x3 = self.pseudo_transform(img)

        return dict(x1=x1, x2=x2, x3=x3, y=label, idx=idx)
