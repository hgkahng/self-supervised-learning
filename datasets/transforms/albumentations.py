# -*- coding: utf-8 -*-

"""
    Core image augmentation functions based on albumentations.
"""

import random
import cv2
import torch
import numpy as np
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.core.composition import BaseCompose
from albumentations.core.transforms_interface import BasicTransform
from albumentations.core.transforms_interface import ImageOnlyTransform


def to_tuple(v: int or float or list or tuple, center: float = 0.):
    if isinstance(v, (int, float)):
        return (center - v , center + v)
    else:
        assert len(v) == 2
        return tuple(v)


class NumpyToTensor(BasicTransform):
    def __init__(self, always_apply: bool = True, p: float = 1.0):
        super(NumpyToTensor, self).__init__(always_apply, p)

    @property
    def targets(self):
        return {'image': self.apply}

    @property
    def targets_as_params(self):
        return ['image']

    def apply(self, img: np.ndarray, **kwargs):  # pylint: disable=unused-argument
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = img[:, :, None]  # put channels last
            if img.ndim == 3:
                # (H, W, C) -> (C, H, W)
                img = torch.from_numpy(img.transpose(2, 0, 1))
            if isinstance(img, torch.ByteTensor):
                img = img.float().div(255)
        return img

    def get_transform_init_args_names(self):
        return []

    def get_params_dependent_on_targets(self, params):
        return {}


class ManyOf(BaseCompose):
    """Select `k` transforms to apply."""
    def __init__(self, transforms, k: int = 5, p: float = 1.0):
        super(ManyOf, self).__init__(transforms, p)
        transforms_ps = [t.p for t in transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]
        self.k = k

    def __call__(self, force_apply:bool = False, **data):
        if self.replay_mode:
            raise NotImplementedError

        if self.transforms_ps and (force_apply or random.random() < self.p):
            random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
            for _ in range(self.k):
                t = random_state.choice(self.transforms.transforms, p=self.transforms_ps)
                data = t(force_apply=True, **data)
        return data


class Translate(A.ShiftScaleRotate):
    def __init__(self, shift_limit: float or tuple = 0.3, p=1.0):
        super(Translate, self).__init__(shift_limit=shift_limit,
                                        scale_limit=0.,
                                        rotate_limit=0.,
                                        interpolation=cv2.INTER_LINEAR,
                                        border_mode=cv2.BORDER_REFLECT_101,
                                        value=None,
                                        mask_value=None,
                                        always_apply=False,
                                        p=p
        )


class Rotate(A.ShiftScaleRotate):
    def __init__(self, rotate_limit: float or tuple = 30., p=1.0):
        super(Rotate, self).__init__(shift_limit=0.,
                                     scale_limit=0.,
                                     rotate_limit=rotate_limit,
                                     interpolation=cv2.INTER_LINEAR,
                                     border_mode=cv2.BORDER_REFLECT_101,
                                     value=None,
                                     mask_value=None,
                                     always_apply=False,
                                     p=p
        )


class AutoContrast(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(AutoContrast, self).__init__(always_apply, p)

    def apply(self, img: np.ndarray, **params):
        return img


class Color:  # FIXME
    def __init__(self, limit: tuple = (0.05, 0.95)):
        self.limit = limit
    def __call__(self, img: np.ndarray):
        _ = np.random.choice(np.linspace(*self.limit, num=30))
        return img


class Sharpness:  # FIXME
    def __init__(self, limit: tuple = (0.05, 0.95)):
        self.limit = limit
    def __call__(self, img: np.ndarray):
        return img


class ShearX(ImageOnlyTransform):
    def __init__(self,
                 limit: float or tuple = 0.2,
                 always_apply: bool = False,
                 p: float = 1.0):
        super(ShearX, self).__init__(always_apply, p)
        self.limit = to_tuple(limit)

    def apply(self, img: np.ndarray, **params):
        v = random.uniform(*self.limit)

        h, w = img.shape[:2]  # channels last
        if v < 0:
            img = cv2.flip(img, 1)

        M = np.array([[1, v, 0],
                      [0, 1, 0]])
        nW = int(w + (v * h))
        img = cv2.warpAffine(img, M, (nW, h))

        if v < 0:
            img = cv2.flip(img, 1)

        return cv2.resize(img, (w, h))


class ShearY(ImageOnlyTransform):
    def __init__(self,
                 limit: float or tuple = 0.2,
                 always_apply: bool = False,
                 p: float = 1.0):
        super(ShearY, self).__init__(always_apply, p)
        self.limit = to_tuple(limit)

    def apply(self, img: np.ndarray, **params):
        v = random.uniform(*self.limit)

        h, w = img.shape[:2]  # channels last
        if v < 0:
            img = cv2.flip(img, 0)
        M = np.array([[1, 0, 0],
                      [v, 1, 0]])
        nH = int(h + (v * w))
        img = cv2.warpAffine(img, M, (w, nH))

        if v < 0:
            img = cv2.flip(img, 0)

        return cv2.resize(img, (w, h))


class RandAugmentAlb(ImageOnlyTransform):
    def __init__(self,
                 k: int = 5,
                 always_apply: bool = True,
                 p: float = 1.0):
        super(RandAugmentAlb, self).__init__(always_apply, p)
        self.k = k
        self.candidates = [
            AutoContrast(p=1.0),
            A.Equalize(p=1.0),
            A.InvertImg(p=1.0),
            Rotate(30., p=1.0),
            A.Posterize([4, 8], p=1.0),
            A.Solarize([0, 256], p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=0.,
                contrast_limit=(0.05, 0.95),
                p=1.0
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(0.05, 0.95),
                contrast_limit=0.,
                p=1.0
            ),
            ShearX(0.3),
            ShearY(0.3),
            Translate(0.45),
        ]

    def apply(self, img: np.ndarray, **params):
        for _ in range(self.k):
            transform = random.choice(self.candidates)
            img = transform(image=img)['image']
        return img
