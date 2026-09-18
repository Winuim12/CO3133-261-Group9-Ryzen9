#test_dataloader.py
'''
RUN: python -m pytest tests/test_data/test_dataloader.py
'''

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import RandomSampler, SequentialSampler

from src.data.dataloader import create_dataloaders
from src.data.pipeline import create_normalized_datasets
from src.utils.seed import set_seed



def test_create_dataloaders_returns_expected_batch_shape():
    images = torch.rand(10, 1, 28, 28)
    labels = torch.arange(10)

    train_dataset = TensorDataset(images, labels)
    val_dataset = TensorDataset(images, labels)
    test_dataset = TensorDataset(images, labels)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=4,
        num_workers=0,
    )

    for loader in (train_loader, val_loader, test_loader):
        batch_images, batch_labels = next(iter(loader))

        assert batch_images.shape == (4, 1, 28, 28)
        assert batch_labels.shape == (4,)

def test_fashion_mnist_dataloaders_return_expected_batch_shape():
    train_dataset, val_dataset, test_dataset, _, _ = (
        create_normalized_datasets(
            data_dir="data/raw/fashion-mnist",
            validation_size=0.1,
            seed=42
        )
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=128,
        num_workers=0,
    )

    for loader in (train_loader, val_loader, test_loader):
        images, labels = next(iter(loader))

        assert images.shape == (128, 1, 28, 28)
        assert labels.shape == (128,)

def test_only_train_dataloader_uses_shuffle():
    images = torch.rand(8, 1, 28, 28)
    labels = torch.arange(8)

    dataset = TensorDataset(images, labels)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset=dataset,
        val_dataset=dataset,
        test_dataset=dataset,
        batch_size=4,
        num_workers=0,
    )

    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(val_loader.sampler, SequentialSampler)
    assert isinstance(test_loader.sampler, SequentialSampler)

def test_train_dataloader_shuffle_is_reproducible_with_seed():
    images = torch.zeros(8,1,28,28)
    labels = torch.arange(8)

    dataset = TensorDataset(images, labels)

    set_seed(42)

    train_loader_1, _,_ = create_dataloaders(
        train_dataset=dataset,
        val_dataset=dataset,
        test_dataset=dataset,
        batch_size=4,
        num_workers=0,
    )

    order_1 = torch.cat([batch_labels for _, batch_labels in train_loader_1])

    set_seed(42)
    
    train_loader_2, _,_ = create_dataloaders(
        train_dataset=dataset,
        val_dataset=dataset,
        test_dataset=dataset,
        batch_size=4,
        num_workers=0,
    )

    order_2 = torch.cat([batch_labels for _, batch_labels in train_loader_2])

    assert torch.equal(order_1, order_2)