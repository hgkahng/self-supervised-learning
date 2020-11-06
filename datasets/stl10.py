# -*- coding: utf-8 -*-

import warnings
import numpy as np
from torchvision.datasets import STL10 as _STL10


class STL10(_STL10):
    num_classes = 10
    def __init__(self,
                 root: str = './data/stl10',
                 split: str = 'train',
                 transform: object = None,
                 **kwargs):
        if split not in ['train', 'test', 'unlabeled', 'train+unlabeled']:
            raise ValueError
        super(STL10, self).__init__(root=root,
                                    split=split,
                                    folds=None,
                                    transform=transform,
                                    target_transform=None,
                                    download=True)

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

    def __getitem__(self, idx):
        img, target = self.data[idx], int(self.labels[idx])
        if self.transform is not None:
            x1 = self.transform(img)
            x2 = self.transform(img)

        return dict(x1=x1, x2=x2, y=target, idx=idx)


STL10ForMoCo = STL10ForSimCLR = STL10Pair
