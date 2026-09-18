#test_transforms.py
'''
Run: python -m pytest tests/test_transforms.py
'''

import torch
import numpy as np

from src.data.transforms import (
    get_train_transforms,
    get_val_transforms,
    get_test_transforms,
)

def test_train_transforms():
    transforms = get_train_transforms(0.5, 0.5)

    image = np.zeros((28, 28), dtype=np.uint8)
    transformed = transforms(image)

    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == (1, 28, 28)

def test_train_transforms_pixel_scaling():
    transforms = get_train_transforms()

    image = np.array(
        [
            [0, 128],
            [255, 64],
        ],
        dtype=np.uint8,
    )

    transformed = transforms(image)

    assert torch.isclose(
        transformed[0, 0, 0],
        torch.tensor(0.0),
    )
    assert torch.isclose(
        transformed[0, 0, 1],
        torch.tensor(128 / 255.0),
    )
    assert torch.isclose(
        transformed[0, 1, 0],
        torch.tensor(1.0),
    )
    assert torch.isclose(
        transformed[0, 1, 1],
        torch.tensor(64 / 255.0),
    )

def test_train_transforms_without_normalization():
    transform = get_train_transforms()

    image = np.zeros((28, 28), dtype=np.uint8)
    transformed = transform(image)

    assert transformed.min() == 0.0
    assert transformed.max() == 0.0
