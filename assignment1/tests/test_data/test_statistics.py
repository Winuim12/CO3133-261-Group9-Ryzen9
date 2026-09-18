#test_statistics.py
'''
Run: python -m pytest tests/test_data/test_statistics.py
'''
import torch
from torch.utils.data import TensorDataset

from src.data.statistics import calculate_mean_std
from src.data.dataset import get_fashion_mnist_datasets
from src.data.split import (
    create_train_val_split,
    create_train_val_datasets,
)
from src.data.transforms import get_train_transforms

def test_calculate_mean_std():
    images = torch.tensor(
        [
            [[[0.0, 1.0],
              [0.0, 1.0]]],
            [[[1.0, 0.0],
              [1.0, 0.0]]],
        ]
    )

    labels = torch.tensor([0, 1])

    dataset = TensorDataset(images, labels)

    mean, std = calculate_mean_std(dataset)

    assert mean == 0.5
    assert std == 0.5

def test_fashion_mnist_train_statistics():
    train_dataset, val_dataset, _ = get_fashion_mnist_datasets(
        "data/raw/fashion-mnist",
        train_transforms=get_train_transforms(),
        val_transforms=get_train_transforms(),
    )

    train_indices, val_indices = create_train_val_split(
        labels = train_dataset.targets,
        validation_size = 0.1,
        seed = 42,
    )

    train_subset, val_subset = create_train_val_datasets(
        train_dataset,
        val_dataset,
        train_indices,
        val_indices,
    )

    assert len(train_subset) == 54_000
    assert len(val_subset) == 6_000

    mean, std = calculate_mean_std(train_subset)

    assert 0.28 < mean < 0.29
    assert 0.35 < std < 0.36