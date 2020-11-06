# -*- coding: utf-8 -*-

"""
    Core image augmentation functions based on torchvision.
"""

import random
import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import ImageFilter
from torchvision.transforms import RandomRotation


def to_tuple(v: int or float or list or tuple, center: float = 0.):
    if isinstance(v, (int, float)):
        return (center - v , center + v)
    else:
        assert len(v) == 2
        return tuple(v)


class MultipleRandomChoice(object):
    """Apply a total of `k` randomly selected transforms."""
    def __init__(self, transforms: list or tuple, k: int = 5, verbose: bool = False):
        self.transforms = transforms
        self.k = k
        self.verbose = verbose

    def __call__(self, img: PIL.Image):
        transforms = random.choices(self.transforms, k=self.k)
        for t in transforms:
            if self.verbose:
                print(str(t), end='\n')
            img = t(img)
        return img


class GaussianBlur(object):
    """Gaussian blur augmentation: https://github.com/facebookresearch/moco/"""

    def __init__(self, sigma=[.1, 2.]):  # pylint: disable=dangerous-default-value
        self.sigma = sigma

    def __call__(self, x: PIL.Image):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class ShearX(object):
    def __init__(self, limit: float or tuple = 0.3):
        self.limit = to_tuple(limit)

    def __call__(self, img: PIL.Image):
        v = np.random.choice(np.linspace(*self.limit, num=100))
        return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


class ShearY(ShearX):
    def __init__(self, limit: float or tuple = 0.3):
        super().__init__(limit)

    def __call__(self, img: PIL.Image):
        v = np.random.choice(np.linspace(*self.limit, num=100))
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


class TranslateX(object):
    def __init__(self, limit: float or tuple = 0.45):
        self.limit = to_tuple(limit)

    def __call__(self, img: PIL.Image):
        v = np.random.choice(np.linspace(*self.limit, num=100))
        v *= img.size[0]
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


class TranslateY(TranslateX):
    def __init__(self, limit: float or tuple = 0.45):
        super().__init__(limit)

    def __call__(self, img: PIL.Image):
        v = np.random.choice(np.linspace(*self.limit, num=100))
        v *= img.size[1]
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


class AutoContrast(object):
    def __call__(self, img: PIL.Image):
        return PIL.ImageOps.autocontrast(img)


class Invert(object):
    def __call__(self, img: PIL.Image):
        return PIL.ImageOps.invert(img)


class Equalize(object):
    def __call__(self, img: PIL.Image):
        return PIL.ImageOps.equalize(img)


class Solarize(object):
    def __call__(self, img: PIL.Image):
        threshold = int(np.random.choice(256))
        return PIL.ImageOps.solarize(img, threshold)


class Posterize(object):
    def __init__(self, limit: tuple = (4, 8)):
        self.limit = limit
    def __call__(self, img: PIL.Image):
        v = np.random.choice(np.arange(*self.limit))
        return PIL.ImageOps.posterize(img, int(v))


class Contrast(object):
    def __init__(self, limit: tuple = (0.05, 0.95)):
        self.limit = limit
    def __call__(self, img: PIL.Image):
        v = np.random.choice(np.linspace(*self.limit, num=30))
        return PIL.ImageEnhance.Contrast(img).enhance(v)


class Color(object):
    def __init__(self, limit: tuple = (0.05, 0.95)):
        self.limit = limit
    def __call__(self, img: PIL.Image):
        v = np.random.choice(np.linspace(*self.limit, num=30))
        return PIL.ImageEnhance.Color(img).enhance(v)


class Brightness(object):
    def __init__(self, limit: tuple = (0.05, 0.95)):
        self.limit = limit
    def __call__(self, img: PIL.Image):
        v = np.random.choice(np.linspace(*self.limit, num=30))
        return PIL.ImageEnhance.Brightness(img).enhance(v)


class Sharpness(object):
    def __init__(self, limit: tuple = (0.05, 0.95)):
        self.limit = limit
    def __call__(self, img: PIL.Image):
        v = np.random.choice(np.linspace(*self.limit, num=30))
        return PIL.ImageEnhance.Sharpness(img).enhance(v)


class Cutout(object):
    def __init__(self, scale: float or tuple = (0.2, 0.2)):
        if isinstance(scale, float):
            self.scale = (scale, scale)
        else:
            self.scale = scale
    def __call__(self, img: PIL.Image):

        w, h = img.size
        x0 = np.random.uniform(w)
        y0 = np.random.uniform(h)

        dw, dh = w * self.scale[0], h * self.scale[1]
        x0 = int(max(0, x0 - dw/2.))
        y0 = int(max(0, y0 - dh/2.))
        x1 = min(w, x0 + dw)
        y1 = min(h, y0 + dw)

        xy = (x0, y0, x1, y1)
        color = (125, 123, 114)

        img = img.copy()
        PIL.ImageDraw.Draw(img).rectangle(xy, color)

        return img


class RandAugmentTv(object):
    def __init__(self, k: int = 5):
        self.k = k
        self.transforms = MultipleRandomChoice(
            [
                AutoContrast(),
                Equalize(),
                Invert(),
                RandomRotation(30.),
                Posterize(),
                Solarize(),
                Color(),
                Contrast(),
                Brightness(),
                Sharpness(),
                ShearX(0.3),
                ShearY(0.3),
                TranslateX(0.45),
                TranslateY(0.45),
            ],
            k=self.k,
            verbose=verbose
        )

    def __call__(self, img: PIL.Image):
        return self.transforms(img)
