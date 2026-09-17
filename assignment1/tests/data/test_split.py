#test_split.py
'''
Run: python -m pytest tests/test_data/test_split.py
'''
import torch

from torch.utils.data import TensorDataset
from src.data.split import (
    create_train_val_split,
    create_train_val_datasets,
)
import numpy as np

def test_train_val_split_size():
    labels = [i // 6000 for i in range(60_000)] #Simulate a dataset with 10 classes, each having 6000 samples

    train_indices, val_indices = create_train_val_split(
        labels = labels,
        validation_size = 0.1,
        seed = 42,
    )

    assert len(train_indices) == 54_000
    assert len(val_indices) == 6_000

def test_train_val_split_no_overlap():
    labels = [i // 6000 for i in range(60_000)]

    train_indices, val_indices = create_train_val_split(
        labels = labels,
        validation_size = 0.1,
        seed = 42,
    )

    assert len(set(train_indices) &set(val_indices)) == 0

def test_train_val_split_stratified():
    labels = [i // 6000 for i in range(60_000)]
    
    train_indices, val_indices = create_train_val_split(
        labels = labels,
        validation_size = 0.1,
        seed = 42,
    )

    train_labels = [labels[i] for i in train_indices]
    val_labels = [labels[i] for i in val_indices]

    for class_id in range(10):
        assert train_labels.count(class_id) == 5_400
        assert val_labels.count(class_id) == 600

def test_train_val_split_reproducibility():
    labels = [i // 6000 for i in range(60_000)]
        
    train_indices_1, val_indices_1 = create_train_val_split(
        labels = labels,
        validation_size = 0.1,
        seed = 42,
    )

    train_indices_2, val_indices_2 = create_train_val_split(
        labels = labels,
        validation_size = 0.1,
        seed = 42,
    )

    assert np.array_equal(train_indices_1, train_indices_2)
    assert np.array_equal(val_indices_1, val_indices_2)

def test_create_train_val_datasets():
    train_dataset = TensorDataset(torch.arange(8))
    val_dataset = TensorDataset(torch.arange(8, 10))

    train_indices = [0, 1, 2, 3, 4, 5, 6, 7]
    val_indices = [0, 1]

    train_dataset, val_dataset = create_train_val_datasets(
        train_dataset,
        val_dataset,
        train_indices,
        val_indices,
    )

    assert len(train_dataset) == 8
    assert len(val_dataset) == 2

    assert train_dataset[0][0].item() == 0
    assert val_dataset[0][0].item() == 8