# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from torch.utils.data import Dataset


CORRUPTIONS = [
    'gaussian_noise',
    'shot_noise',
    'impulse_noise',
    'defocus_blur',
    'glass_blur',
    'motion_blur',
    'zoom_blur',
    'snow',
    'frost',
    'fog',
    'brightness',
    'contrast',
    'elastic_transform',
    'pixelate',
    'jpeg_compression'
]


class CIFAR10C(Dataset):
    num_classes = 10
    def __init__(self,
                 root: str,
                 corruption: str,
                 transform: object = None):
        super(CIFAR10C, self).__init__()

        self.root = root
        self.corruption = corruption
        self.transform = transform
    
        if self.corruption not in CORRUPTIONS:
            raise ValueError
        
        self.data = np.load(os.path.join(root, f'{self.corruption}.npy'))
        self.targets = np.load(os.path.join(root, 'labels.npy'))

    def __getitem__(self, idx: int):

        img, target = self.data[idx], self.targets[idx]

        if self.transform is not None:
            img = self.transform(img)

        return dict(x=img, y=target, idx=idx)

    def __len__(self):
        return len(self.data)
