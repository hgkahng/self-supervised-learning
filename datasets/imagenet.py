# -*- coding: utf-8 -*-

import os
import glob
import cv2
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.imagenet import ImageNet as _ImageNet
from sklearn.model_selection import train_test_split


def load_image_cv2(path: str):
    """Load image with OpenCV."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        raise NotImplementedError


def load_image_pil(path: str):
    """Load image with PIL."""
    return Image.open(path)


class TinyImageNet(Dataset):
    """
    Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.
    Implementation based on:
        https://github.com/leemengtaiwan/tiny-imagenet/
    """
    EXTENSION = "JPEG"
    NUM_IMAGES_PER_CLASS = 500
    CLASS_LIST_FILE = "wnids.txt"
    VAL_ANNOTATION_FILE = 'val_annotations.txt'
    num_classes = 200

    def __init__(self,
                 root: str,
                 split: str = 'train',
                 transform: object = None,
                 target_transform: object = None,
                 in_memory: bool = True,
                 proportion: float = 1.0,
                 **kwargs):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.in_memory = in_memory
        self.proportion = proportion

        self.split_dir = os.path.join(root, self.split)

        self.image_paths = glob.glob(os.path.join(self.split_dir, f"**/*.{self.EXTENSION}"), recursive=True)
        self.image_paths = sorted(self.image_paths)
        self.images = []
        self.labels = {}

        with open(os.path.join(self.root, self.CLASS_LIST_FILE), 'r') as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}

        if self.split == 'train':
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(self.NUM_IMAGES_PER_CLASS):
                    self.labels[f'{label_text}_{cnt}.{self.EXTENSION}'] = i
        elif self.split == 'val':
            with open(os.path.join(self.split_dir, self.VAL_ANNOTATION_FILE), 'r') as fp:
                for line in fp.readlines():
                    filename, label_text, *_ = line.split('\t')
                    self.labels[filename] = self.label_text_to_number[label_text]
        else:
            raise NotImplementedError

        if self.proportion < 1.:
            indices, _ = train_test_split(
                np.arange(len(self.image_paths)),
                train_size=self.proportion,
                stratify=[self.labels[os.path.basename(p)] for p in self.image_paths],
                shuffle=True,
                random_state=2021 + kwargs.get('seed', 0)
            )
            self.image_paths = [self.image_paths[i] for i in indices]

        if self.in_memory:
            print(f"Loading {self.split} data to memory...", end=' ')
            self.images = [load_image_cv2(path) for path in self.image_paths]
            print(f"Done!")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        target = self.labels[os.path.basename(path)]
        if self.in_memory:
            img = self.images[idx]
        else:
            img = load_image_cv2(path)

        if self.transform is not None:
            img = self.transform(img)

        return dict(x=img, y=target, idx=idx)


class TinyImageNetPair(TinyImageNet):
    def __init__(self,
                 root: str = './data/tiny-imagenet-200',
                 split: str = 'train',
                 transform: object = None,
                 in_memory: bool = True,
                 **kwargs):
        super(TinyImageNetPair, self).__init__(root=root,
                                               split=split,
                                               transform=transform,
                                               target_transform=None,
                                               in_memory=in_memory,
                                               **kwargs)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        target = self.labels[os.path.basename(path)]
        if self.in_memory:
            img = self.images[idx]
        else:
            img = self.load_image_cv2(path)

        if self.transform is not None:
            x1 = self.transform(img)
            x2 = self.transform(img)

        return dict(x1=x1, x2=x2, y=target, idx=idx)


class TinyImageNetForMoCo(TinyImageNet):
    def __init__(self,
                 root: str = './data/tiny-imagenet-200/',
                 split: str = 'train',
                 query_transform: object = None,
                 key_transform: object = None,
                 in_memory: bool = True,
                 **kwargs):
        super(TinyImageNetForMoCo, self).__init__(root=root,
                                                  split=split,
                                                  transform=None,
                                                  target_transform=None,
                                                  in_memory=in_memory,
                                                  **kwargs)
        self.query_transform = query_transform
        self.key_transform = key_transform

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        target = self.labels[os.path.basename(path)]
        if self.in_memory:
            img = self.images[idx]
        else:
            img = load_image_cv2(path)

        x1 = self.query_transform(img)
        x2 = self.key_transform(img)

        return dict(x1=x1, x2=x2, y=target, idx=idx)

class TinyImageNetForCLAPP(TinyImageNet):
    def __init__(self,
                 root: str = './data/tiny-imagenet-200/',
                 split: str = 'train',
                 query_transform: object = None,
                 key_transform: object = None,
                 pseudo_transform: object = None,
                 in_memory: bool = True,
                 **kwargs):
        super(TinyImageNetForCLAPP, self).__init__(root=root,
                                                   split=split,
                                                   transform=None,
                                                   target_transform=None,
                                                   in_memory=in_memory,
                                                   **kwargs)
        self.query_transform = query_transform
        self.key_transform = key_transform
        self.pseudo_transform = pseudo_transform

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        target = self.labels[os.path.basename(path)]
        if self.in_memory:
            img = self.images[idx]
        else:
            img = load_image_cv2(path)

        x1 = self.query_transform(img)
        x2 = self.key_transform(img)
        x3 = self.pseudo_transform(img)

        return dict(x1=x1, x2=x2, x3=x3, y=target, idx=idx)


class ImageNet(_ImageNet):
    num_classes = 1000
    def __init__(self,
                 root: str = './data/imagenet2012',
                 split: str = 'train',
                 transform: object = None,
                 proportion: float = 1.0,
                 **kwargs):
        super(ImageNet, self).__init__(root=root,
                                       split=split,
                                       transform=transform,
                                       target_transform=None)

        self.proportion = proportion
        if self.proportion < 1.0:
            self.samples, _ = train_test_split(
                self.samples,
                train_size=self.proportion,
                stratify=[s[-1] for s in self.samples],
                shuffle=True,
                random_state=2020 + kwargs.get('seed', 0)
            )

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = load_image_cv2(path)
        if self.transform is not None:
            img = self.transform(img)

        return dict(x=img, y=target, idx=idx)

    def __len__(self):
        return len(self.samples)


class ImageNetPair(ImageNet):
    def __init__(self,
                 root: str = '../imagenet2012',
                 split: str = 'train',
                 transform: object = None,
                 **kwargs):
        super(ImageNetPair, self).__init__(root=root,
                                           split=split,
                                           transform=transform,
                                           target_transform=None,
                                           **kwargs)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = self.load_image_cv2(path)
        if self.transform is not None:
            x1 = self.transform(img)
            x2 = self.transform(img)

        return dict(x1=x1, x2=x2, y=target, idx=idx)


ImageNetForSimCLR = ImageNetPair
TinyImageNetForSimCLR = TinyImageNetPair
