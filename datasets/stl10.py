# -*- coding: utf-8 -*-

import warnings
import numpy as np
from torchvision.datasets import STL10 as _STL10


class STL10(_STL10):
    num_classes = 10
    def __init__(self,
                 root: str = './data/stl10',
                 split: str = 'train+unlabeled',
                 transform: object = None,
                 **kwargs):
        if split not in ['train', 'test', 'unlabeled', 'train+unlabeled']:
            raise ValueError(f"Not a valid split: `{split}`.")
        super(STL10, self).__init__(root=root,
                                    split=split,
                                    folds=None,
                                    transform=transform,
                                    target_transform=None,
                                    download=False)

        assert isinstance(self.data, np.ndarray)
        assert isinstance(self.labels, np.ndarray)

        # To support both 'torchvision' & 'albumentations', make channels last (= [N, H, W, C]).
        # When composing 'torchvision' transforms, make sure to start with a `ToPILImage()`.
        # All 'albumentations' transforms assume channels to be at the last dimension.
        self.data = self.data.transpose(0, 2, 3, 1)

        if 'proportion' in kwargs:
            warnings.warn("Argument 'proportion' not valid for STL-10.")

    def __getitem__(self, idx):
        img, target = self.data[idx], int(self.labels[idx])
        if self.transform is not None:
            img = self.transform(img)

        return dict(x=img, y=target, idx=idx)


class STL10Pair(_STL10):
    def __init__(self,
                 root: str = './data/stl10',
                 split: str = 'train+unlabeled',
                 transform: object = None,
                 **kwargs):
        super(STL10Pair, self).__init__(root=root,
                                        split=split,
                                        folds=None,
                                        transform=transform,
                                        target_transform=None,
                                        download=False)
        assert isinstance(self.data, np.ndarray)
        assert isinstance(self.labels, np.ndarray)
        self.data = np.transpose(self.data, (0, 2, 3, 1))

    def __getitem__(self, idx):
        img, target = self.data[idx], int(self.labels[idx])
        x1 = self.transform(img)
        x2 = self.transform(img)

        return dict(x1=x1, x2=x2, y=target, idx=idx)


class STL10ForMoCo(_STL10):
    def __init__(self,
                 root: str = './data/stl10',
                 split: str = 'train+unlabeled',
                 query_transform: object = None,
                 key_transform: object = None,
                 **kwargs):
        super(STL10ForMoCo, self).__init__(root=root,
                                           split=split,
                                           folds=None,
                                           transform=None,
                                           target_transform=None,
                                           download=False)
        assert isinstance(self.data, np.ndarray)
        assert isinstance(self.labels, np.ndarray)
        self.data = np.transpose(self.data, (0, 2, 3, 1))
        
        self.query_transform = query_transform
        self.key_transform = key_transform

    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]

        x1 = self.query_transform(img)
        x2 = self.key_transform(img)

        return dict(x1=x1, x2=x2, y=label, idx=idx)


class STL10ForCLAPP(_STL10):
    def __init__(self,
                 root: str = './data/stl10',
                 split: str = 'train+unlabeled',
                 query_transform: object = None,
                 key_transform: object = None,
                 pseudo_transform: object = None):
        super(STL10ForCLAPP, self).__init__(root=root,
                                            split=split,
                                            folds=None,
                                            transform=None,
                                            target_transform=None,
                                            download=False)
        assert isinstance(self.data, np.ndarray)
        assert isinstance(self.labels, np.ndarray)
        self.data = np.transpose(self.data, (0, 2, 3, 1))
        
        self.query_transform = query_transform
        self.key_transform = key_transform
        self.pseudo_transform = pseudo_transform

    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]

        x1 = self.query_transform(img)
        x2 = self.key_transform(img)
        x3 = self.pseudo_transform(img)

        return dict(x1=x1, x2=x2, x3=x3, y=label, idx=idx)
        


STL10ForSimCLR = STL10Pair
