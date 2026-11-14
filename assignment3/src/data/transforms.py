"""Aspect-preserving RGB resize/padding; never alters raw image files.

Pipeline: RGB -> resize longest side to `size` (keep aspect) -> pad to size x size
-> tensor [3,size,size] -> ImageNet mean/std normalisation.
Light train-only augmentation (no horizontal flip: the text is inside the image).
"""
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
PAD_COLOR = tuple(int(round(m * 255)) for m in MEAN)  # becomes ~0 after normalisation


def resize_pad(img: Image.Image, size: int = 224) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), PAD_COLOR)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (t - mean) / std


class ImageTransform:
    def __init__(self, size: int = 224, train: bool = False):
        self.size, self.train = size, train

    def __call__(self, img: Image.Image) -> torch.Tensor:
        if self.train:  # mild brightness/contrast jitter only
            img = ImageEnhance.Brightness(img.convert("RGB")).enhance(random.uniform(0.9, 1.1))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.1))
        return to_tensor(resize_pad(img, self.size))
