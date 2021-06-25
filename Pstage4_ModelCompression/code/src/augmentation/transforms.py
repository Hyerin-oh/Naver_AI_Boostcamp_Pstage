"""Image transformations for augmentation.

- Author: wlaud1001
- Email: wlaud1001@snu.ac.kr
- Reference:
    https://github.com/j-marple-dev/model_compression
"""

import random
from typing import Callable, Dict, Tuple

import PIL
from PIL.Image import Image
import PIL.ImageDraw
import PIL.ImageEnhance
import PIL.ImageOps
import cv2
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as F

from src.utils.data import get_rand_bbox_coord

FILLCOLOR = (128, 128, 128)
FILLCOLOR_RGBA = (128, 128, 128, 128)


def transforms_info() -> Dict[
    str, Tuple[Callable[[Image, float], Image], float, float]
]:
    """Return augmentation functions and their ranges."""
    transforms_list = [
        (Identity, 0.0, 0.0),
        (Invert, 0.0, 0.0),
        (Contrast, 0.0, 0.9),
        (AutoContrast, 0.0, 0.0),
        (Rotate, 0.0, 30.0),
        (TranslateX, 0.0, 150 / 331),
        (TranslateY, 0.0, 150 / 331),
        (Sharpness, 0.0, 0.9),
        (ShearX, 0.0, 0.3),
        (ShearY, 0.0, 0.3),
        (Color, 0.0, 0.9),
        (Brightness, 0.0, 0.9),
        (Equalize, 0.0, 0.0),
        (Solarize, 256.0, 0.0),
        (Posterize, 8, 4),
        (Cutout, 0, 0.5),
        (CustomClahe, 0.1, 2.0),
        (CustomGridShuffle, 0.0, 0.0),
    ]
    return {f.__name__: (f, low, high) for f, low, high in transforms_list}


def Identity(img: Image, _: float) -> Image:
    """Identity map."""
    return img


def Invert(img: Image, _: float) -> Image:
    """Invert the image."""
    return PIL.ImageOps.invert(img)


def Contrast(img: Image, magnitude: float) -> Image:
    """Put contrast effect on the image."""
    return PIL.ImageEnhance.Contrast(img).enhance(
        1 + magnitude * random.choice([-1, 1])
    )


def AutoContrast(img: Image, _: float) -> Image:
    """Put contrast effect on the image."""
    return PIL.ImageOps.autocontrast(img)


def Rotate(img: Image, magnitude: float) -> Image:
    """Rotate the image (degree)."""
    rot = img.convert("RGBA").rotate(magnitude)
    return PIL.Image.composite(
        rot, PIL.Image.new("RGBA", rot.size, FILLCOLOR_RGBA), rot
    ).convert(img.mode)


def TranslateX(img: Image, magnitude: float) -> Image:
    """Translate the image on x-axis."""
    return img.transform(
        img.size,
        PIL.Image.AFFINE,
        (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
        fillcolor=FILLCOLOR,
    )


def TranslateY(img: Image, magnitude: float) -> Image:
    """Translate the image on y-axis."""
    return img.transform(
        img.size,
        PIL.Image.AFFINE,
        (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
        fillcolor=FILLCOLOR,
    )


def Sharpness(img: Image, magnitude: float) -> Image:
    """Adjust the sharpness of the image."""
    return PIL.ImageEnhance.Sharpness(img).enhance(
        1 + magnitude * random.choice([-1, 1])
    )


def ShearX(img: Image, magnitude: float) -> Image:
    """Shear the image on x-axis."""
    return img.transform(
        img.size,
        PIL.Image.AFFINE,
        (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
        PIL.Image.BICUBIC,
        fillcolor=FILLCOLOR,
    )


def ShearY(img: Image, magnitude: float) -> Image:
    """Shear the image on y-axis."""
    return img.transform(
        img.size,
        PIL.Image.AFFINE,
        (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
        PIL.Image.BICUBIC,
        fillcolor=FILLCOLOR,
    )


def Color(img: Image, magnitude: float) -> Image:
    """Adjust the color balance of the image."""
    return PIL.ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1]))


def Brightness(img: Image, magnitude: float) -> Image:
    """Adjust brightness of the image."""
    return PIL.ImageEnhance.Brightness(img).enhance(
        1 + magnitude * random.choice([-1, 1])
    )


def Equalize(img: Image, _: float) -> Image:
    """Equalize the image."""
    return PIL.ImageOps.equalize(img)


def Solarize(img: Image, magnitude: float) -> Image:
    """Solarize the image."""
    return PIL.ImageOps.solarize(img, magnitude)


def Posterize(img: Image, magnitude: float) -> Image:
    """Posterize the image."""
    magnitude = int(magnitude)
    return PIL.ImageOps.posterize(img, magnitude)


def Cutout(img: Image, magnitude: float) -> Image:
    """Cutout some region of the image."""
    if magnitude == 0.0:
        return img
    w, h = img.size
    xy = get_rand_bbox_coord(w//2, h//2, magnitude)

    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, fill=FILLCOLOR)
    return img


def CustomClahe(img: Image, magnitude: float) -> Image:
    lab = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2LAB) 
    l, a, b = cv2.split(lab) 
    clahe = cv2.createCLAHE(clipLimit=magnitude, tileGridSize=(7, 7)) 
    cl = clahe.apply(l) 
    img = cv2.merge((cl, a, b)) 
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR) 
    return PIL.Image.fromarray(img)


def CustomGridShuffle(img: Image, magnitude: float) -> Image:
    w, h = img.size
    im = list(transforms.FiveCrop((h//2, w//2))(img))
    random.shuffle(im)
    img = PIL.Image.new('RGB', (w, h))
    img.paste(im[0], (0, 0))
    img.paste(im[1], (w//2, 0))
    img.paste(im[2], (0, w//2))
    img.paste(im[3], (w//2, h//2))
    return img


class SquarePad:
    """Square pad to make torch resize to keep aspect ratio."""
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, "constant")


class CustomRotation90Resize:
    """Rotate vertical images to horizontal and resize to average ratio."""
    def __init__(self, img_size, resize=True):
        self.img_size = img_size
        self.resize = resize

    def __call__(self, image):
        w, h = image.size
        if w < h:
            image = F.rotate(image, angle=90, expand=True)

        if self.resize:
            w, h = self.img_size, int(self.img_size*0.723)
            return F.resize(image, (h, w))
        else:
            w, h = image.size
            vp = int((w - h) / 2)
            padding = (0, vp, 0, vp)
            image = F.pad(image, padding, 0, "constant")
            return F.resize(image, (self.img_size, self.img_size))

        
