
import numpy as np
import torch
import torch.nn as nn

from torchvision import transforms as T
from torchvision.datasets.cifar import CIFAR10 as _CIFAR10
from torchvision.datasets.cifar import CIFAR100 as _CIFAR100
from sklearn.model_selection import train_test_split


class CIFAR10(_CIFAR10):
    num_classes = 10
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: nn.Module = None,
                 proportion: float = 1.0,
                 random_state: int = 42,
                 ):
        super().__init__(root=root, train=train, transform=transform)

        if proportion < 1.0:
            indices, _ = \
                train_test_split(
                    np.arange(len(self.data)),
                    train_size=self.proportion,
                    stratify=self.targets,
                    shuffle=True,
                    random_state=random_state,
                )
            self.data = self.data[indices]
            self.targets = list(np.array(self.targets)[indices])

        self.data = self.data.transpose((0, 3, 1, 2))  # BHWC -> BCHW
        self.data = torch.from_numpy(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img, target = self.data[idx], self.targets[idx]
        if self.transform is not None:
            img = self.transform(img)
        return dict(x=img, y=target, idx=idx)


class CIFAR100(_CIFAR100):
    num_classes = 100
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: nn.Module = None,
                 proportion: float = 1.0,
                 random_state: int = 42,
                 ):
        super().__init__(root=root, train=train, transform=transform)

        if proportion < 1.0:
            indices, _ = train_test_split(
                np.arange(len(self.data)),
                train_size=self.proportion,
                stratify=self.targets,
                shuffle=True,
                random_state=random_state,
            )
            self.data = self.data[indices]
            self.targets = list(np.array(self.targets)[indices])

        self.data = self.data.transpose((0, 3, 1, 2))  # BHWC -> BCHW
        self.data = torch.from_numpy(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img, target = self.data[idx], self.targets[idx]
        if self.transform is not None:
            img = self.transform(img)
        return dict(x=img, y=target, idx=idx)


class CIFAR10Pair(CIFAR10):
    def __init__(self, root: str , train: bool = True, transform: nn.Module = None):
        super().__init__(root=root, train=train, transform=transform)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        img, target = self.data[idx], self.targets[idx]
        x1 = self.transform(img) if self.transform is not None else img
        x2 = self.transform(img) if self.transform is not None else img
        return dict(x1=x1, x2=x2, y=target, idx=idx)


class CIFAR100Pair(CIFAR100):
    def __init__(self, root: str , train: bool = True, transform: nn.Module = None):
        super().__init__(root=root, train=train, transform=transform)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img, target = self.data[idx], self.targets[idx]
        x1 = self.transform(img) if self.transform is not None else img
        x2 = self.transform(img) if self.transform is not None else img
        return dict(x1=x1, x2=x2, y=target, idx=idx)


class CIFAR10ForMoCo(CIFAR10):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 query_transform: nn.Module = None,
                 key_transform: nn.Module = None,
                 ):
        super().__init__(root=root, train=train)
        self.query_transform = query_transform
        self.key_transform = key_transform

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img, target = self.data[idx], self.targets[idx]
        x1 = self.query_transform(img) if self.query_transform is not None else img
        x2 = self.key_transform(img) if self.key_transform is not None else img
        return dict(x1=x1, x2=x2, y=target, idx=idx)


class CIFAR100ForMoCo(CIFAR100):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 query_transform: nn.Module = None,
                 key_transform: nn.Module = None,
                 ):
        super().__init__(root=root, train=train)
        self.query_transform = query_transform
        self.key_transform = key_transform

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img, target = self.data[idx], self.targets[idx]
        x1 = self.query_transform(img) if self.query_transform is not None else img
        x2 = self.key_transform(img) if self.key_transform is not None else img
        return dict(x1=x1, x2=x2, y=target, idx=idx)


class CIFAR10ForCLAP(CIFAR10):  # TODO: deprecate
    def __init__(self,
                 root: str,
                 train: bool = True,
                 query_transform: nn.Module = None,
                 key_transform: nn.Module = None,
                 teacher_transform: nn.Module = None):
        super(CIFAR10ForCLAP, self).__init__(root=root, train=train)
        self.query_transform = query_transform
        self.key_transform = key_transform
        self.teacher_transform = teacher_transform

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img, target = self.data[idx], self.targets[idx]
        x1 = self.query_transform(img) if self.query_transform is not None else img
        x2 = self.key_transform(img) if self.key_transform is not None else img
        x3 = self.teacher_transform(img) if self.teacher_transform is not None else img
        return dict(x1=x1, x2=x2, x3=x3, y=target, idx=idx)


class CIFAR100ForCLAP(CIFAR100):  # TODO: deprecate
    def __init__(self,
                 root: str,
                 train: bool = True,
                 query_transform: nn.Module = None,
                 key_transform: nn.Module = None,
                 teacher_transform: nn.Module = None):
        super(CIFAR100ForCLAP, self).__init__(root=root, train=train)
        self.query_transform = query_transform
        self.key_transform = key_transform
        self.teacher_transform = teacher_transform

    def __getitem__(self, idx: int) -> dict[str,torch.Tensor]:
        img, target = self.data[idx], self.targets[idx]
        x1 = self.query_transform(img) if self.query_transform is not None else img
        x2 = self.key_transform(img) if self.key_transform is not None else img
        x3 = self.teacher_transform(img) if self.teacher_transform is not None else img
        return dict(x1=x1, x2=x2, x3=x3, y=target, idx=idx)


CIFAR10ForSimCLR = CIFAR10Pair
CIFAR100ForSimCLR = CIFAR100Pair
