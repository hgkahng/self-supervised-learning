# -*- coding: utf-8 -*-

import os
import glob
import typing
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision.datasets.imagenet import ImageNet as _ImageNet
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms as T
from sklearn.model_selection import train_test_split


class ImageLoader(object):
    """Collection of image loading functions."""
    def __init__(self, mode: str):
        self.mode = mode.lower()
        if mode == 'pil':
            self.load_fn = self.load_image_pil
        elif mode == 'accimage':
            from torchvision import get_image_backend
            if get_image_backend() != 'accimage':
                raise AttributeError(
                    "Use `torchvision.set_image_backend()` to set backend as `accimage`."
                )
            self.load_fn = self.load_image_accimage
        elif mode == 'torch':
            self.load_fn = self.load_image_torch
        else:
            raise NotImplementedError

    def __call__(self, path: str) -> torch.ByteTensor:
        return self.load_fn(path)

    @classmethod
    def load_image_pil(cls, path: str) -> torch.ByteTensor:
        """Load image with PIL."""
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
            return T.ToTensor()(img)

    @classmethod
    def load_image_accimage(cls, path: str) -> torch.ByteTensor:
        """Load image with `accimage`."""
        import accimage
        try:
            img = accimage.Image(path)
            return T.ToTensor()(img)
        except IOError:
            # Potentially a decoding problem, fall back to PIL.Image
            return cls.load_image_pil(path)
    
    @classmethod
    def load_image_torch(cls, path: str) -> torch.ByteTensor:
        """Load image with native torchvision operations."""
        from torchvision.io import read_image, ImageReadMode
        try:
            return read_image(path, mode=ImageReadMode.RGB)
        except IOError:
            # Potentially a decoding problem, fall back to PIL.Image
            return cls.load_image_pil(path)


class TinyImageNet(Dataset):
    """
    TinyImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.
    Implementation based on:
        https://github.com/leemengtaiwan/tiny-imagenet/
    """
    EXTENSION = "JPEG"
    NUM_IMAGES_PER_CLASS = 500
    CLASS_LIST_FILE = "wnids.txt"
    VAL_ANNOTATION_FILE = 'val_annotations.txt'
    num_classes = 200

    def __init__(self,
                 root: str = './data/tinyimagenet',
                 split: str = 'train',
                 transform: nn.Module = None,
                 target_transform: nn.Module = None,
                 in_memory: bool = True,
                 proportion: float = 1.0,
                 **kwargs):
        super(TinyImageNet, self).__init__()
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
            self.images = [self.load_image_cv2(path) for path in self.image_paths]
            print(f"Done!")

        self.img_to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> typing.Dict[str,torch.Tensor]:
        path = self.image_paths[idx]
        target = self.labels[os.path.basename(path)]
        if self.in_memory:
            img = self.images[idx]
        else:
            img = self.load_image_cv2(path)
        img = self.img_to_tensor(img)
        if self.transform is not None:
            img = self.transform(img)
        return dict(x=img, y=target, idx=idx)

    @staticmethod
    def load_image_cv2(path: str):
        """Load image with OpenCV."""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            raise NotImplementedError


class TinyImageNetPair(TinyImageNet):
    def __init__(self,
                 root: str = './data/tinyimagenet',
                 split: str = 'train',
                 transform: nn.Module = None,
                 in_memory: bool = True,
                 **kwargs):
        super(TinyImageNetPair, self).__init__(root=root,
                                               split=split,
                                               transform=transform,
                                               in_memory=in_memory,
                                               **kwargs)

    def __getitem__(self, idx: int) -> typing.Dict[str,torch.Tensor]:
        path = self.image_paths[idx]
        target = self.labels[os.path.basename(path)]
        if self.in_memory:
            img = self.images[idx]
        else:
            img = self.load_image_cv2(path)
        img = self.img_to_tensor(img)
        if self.transform is not None:
            x1 = self.transform(img)
            x2 = self.transform(img)
        return dict(x1=x1, x2=x2, y=target, idx=idx)


class TinyImageNetForMoCo(TinyImageNet):
    def __init__(self,
                 root: str = './data/tinyimagenet',
                 split: str = 'train',
                 query_transform: nn.Module = None,
                 key_transform: nn.Module = None,
                 in_memory: bool = True,
                 **kwargs):
        super(TinyImageNetForMoCo, self).__init__(root=root,
                                                  split=split,
                                                  in_memory=in_memory,
                                                  **kwargs)
        self.query_transform = query_transform
        self.key_transform = key_transform

    def __getitem__(self, idx: int) -> typing.Dict[str,torch.Tensor]:
        path = self.image_paths[idx]
        target = self.labels[os.path.basename(path)]
        if self.in_memory:
            img = self.images[idx]
        else:
            img = self.load_image_cv2(path)

        img = self.img_to_tensor(img)
        x1 = self.query_transform(img) if self.query_transform is not None else img
        x2 = self.key_transform(img) if self.key_transform is not None else img

        return dict(x1=x1, x2=x2, y=target, idx=idx)


class TinyImageNetForCLAP(TinyImageNet):
    def __init__(self,
                 root: str = './data/tinyimagenet',
                 split: str = 'train',
                 query_transform: nn.Module = None,
                 key_transform: nn.Module = None,
                 teacher_transform: nn.Module = None,
                 in_memory: bool = True,
                 **kwargs):
        super(TinyImageNetForCLAP, self).__init__(root=root,
                                                  split=split,
                                                  in_memory=in_memory,
                                                  **kwargs)
        self.query_transform = query_transform
        self.key_transform = key_transform
        self.teacher_transform = teacher_transform

    def __getitem__(self, idx: int) -> typing.Dict[str,torch.Tensor]:
        path = self.image_paths[idx]
        target = self.labels[os.path.basename(path)]
        if self.in_memory:
            img = self.images[idx]
        else:
            img = self.load_image_cv2(path)
        img = self.img_to_tensor(img)
        x1 = self.query_transform(img) if self.query_transform is not None else img
        x2 = self.key_transform(img) if self.key_transform is not None else img
        x3 = self.teacher_transform(img) if self.teacher_transform is not None else img
        return dict(x1=x1, x2=x2, x3=x3, y=target, idx=idx)


class ImageNet(_ImageNet):
    num_classes = 1000
    def __init__(self,
                 root: str = './data/imagenet',
                 split: str = 'train',
                 resolution: typing.Union[int,list,tuple] = None,
                 transform: nn.Module = None,
                 proportion: float = 1.0,
                 **kwargs):
        super(ImageNet, self).__init__(root=root,
                                       split=split,
                                       transform=transform)
        
        # Resizing for fixed size outputs (experimental)
        if resolution is not None:
            warnings.warn('Specifying resolution may affect performance.', UserWarning)
            if isinstance(resolution, int):
                self.resolution = (resolution, resolution)
            elif isinstance(resolution, (list,tuple)):
                if len(resolution) != 2:
                    raise ValueError("Required type for `resolution`: `int` or `tuple` of length 2.")
                self.resolution = tuple(resolution)
            else:
                raise NotImplementedError
        else:
            self.resolution = None
        
        if self.resolution is not None:
            self.resize = T.Resize(size=self.resolution)
        else:
            self.resize = None

        # Downsampling used for semi-supervised evaluation
        self.proportion = proportion
        if self.proportion < 1.0:
            self.samples, _ = train_test_split(
                self.samples,
                train_size=self.proportion,
                stratify=[s[-1] for s in self.samples],
                shuffle=True,
                random_state=2020 + kwargs.get('seed', 0)
            )

        from torchvision import set_image_backend
        try:
            set_image_backend('accimage')
            img_load_mode = 'accimage'
        except ModuleNotFoundError:
            warnings.warn("'accimage' not installed. Falling pack to PIL", ImportWarning)
            img_load_mode = 'pil'
        finally:
            self.img_loader = ImageLoader(mode=img_load_mode)

    def __getitem__(self, idx: int) -> typing.Dict[str,torch.Tensor]:
        path, target = self.samples[idx]
        img: torch.ByteTensor =self.img_loader(path)
        if self.resize is not None:
            img = self.resize(img)
        if self.transform is not None:
            img = self.transform(img)
        return dict(x=img, y=target, idx=idx)

    def __len__(self):
        return len(self.samples)

    @classmethod
    def split_into_two_subsets(cls,
                               root: str,
                               split: str = 'val',
                               first_size: float = 0.9,
                               transforms: typing.Union[nn.Module,typing.List[nn.Module]] = None,
                               **kwargs) -> typing.Tuple[Dataset, Dataset]:
        """
        Split ImageNet data into two mutually exclusive subsets.
        """
        dataset = cls(root=root, split=split, transform=None)
        first_indices, second_indices = train_test_split(
            np.arange(len(dataset)),
            train_size=first_size,
            random_state=kwargs.get('random_state', 10010),
            shuffle=kwargs.get('shuffle', True),
            stratify=dataset.targets,
        )
        first_set = Subset(dataset, first_indices)
        second_set = Subset(dataset, second_indices)
        
        if transforms is not None:
            if len(transforms) == 1:
                first_set.dataset.transform = transforms[0]
                second_set.dataset.transform = transforms[0]
            elif len(transforms) == 2:
                first_set.dataset.transform = transforms[0]
                second_set.dataset.transform = transforms[1]
            else:
                raise NotImplementedError

        return first_set, second_set


class ImageNetPair(ImageNet):
    def __init__(self,
                 root: str = './data/imagenet',
                 split: str = 'train',
                 resolution: typing.Union[int,list,tuple] = None,
                 transform: nn.Module = None,
                 **kwargs):
        super(ImageNetPair, self).__init__(root=root,
                                           split=split,
                                           resolution=resolution,
                                           transform=transform,
                                           **kwargs)

    def __getitem__(self, idx: int) -> typing.Dict[str,torch.Tensor]:
        path, target = self.samples[idx]
        img: torch.ByteTensor = self.img_loader(path)
        if self.resize is not None:
            img = self.resize(img)
        if self.transform is not None:
            img = self.transform(img)
        return dict(x1=img, x2=img, y=target, idx=idx)


class ImageNetForMoCo(ImageNet):
    def __init__(self,
                 root: str = './data/imagenet',
                 split: str = 'train',
                 resolution: typing.Union[int,list,tuple] = None,
                 query_transform: nn.Module = None,
                 key_transform: nn.Module = None,
                 **kwargs):
        super(ImageNetForMoCo, self).__init__(root=root,
                                              split=split,
                                              resolution=resolution,
                                              **kwargs)
        self.query_transform = query_transform
        self.key_transform   = key_transform
    
    def __getitem__(self, idx: int) -> typing.Dict[str,torch.Tensor]:
        path, target = self.samples[idx]
        img: torch.ByteTensor = self.img_loader(path)
        if self.resize is not None:
            img = self.resize(img)
        x1 = self.query_transform(img) if self.query_transform is not None else img
        x2 = self.key_transform(img) if self.key_transform is not None else img
        return dict(x1=x1, x2=x2, y=target, idx=idx)


class ImageNetForCLAP(ImageNet):
    def __init__(self,
                 root: str,
                 split: str = 'train',
                 resolution: int or tuple = None,
                 query_transform: nn.Module = None,
                 key_transform: nn.Module = None,
                 teacher_transform: nn.Module = None,
                 **kwargs):
        super(ImageNetForCLAP, self).__init__(root=root,
                                              split=split,
                                              resolution=resolution,
                                              **kwargs)
        
        self.query_transform  = query_transform
        self.key_transform    = key_transform
        self.teacher_transform = teacher_transform
    
    def __getitem__(self, idx: int) -> typing.Dict[str,torch.Tensor]:
        path, target = self.samples[idx]
        img: torch.ByteTensor = self.img_loader(path)
        if self.resize is not None:
            img = self.resize(img)
        x1 = self.query_transform(img) if self.query_transform else img
        x2 = self.key_transform(img) if self.key_transform else img
        x3 = self.teacher_transform(img) if self.teacher_transform else img
        return dict(x1=x1, x2=x2, x3=x3, y=target, idx=idx)


ImageNetForSimCLR = ImageNetPair
TinyImageNetForSimCLR = TinyImageNetPair
