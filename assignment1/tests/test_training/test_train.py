#test_train.py
'''
RUN: python -m pytest tests/training/test_train.py -v
'''

import torch
import pytest
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.train import train_one_epoch


def test_train_one_epoch_updates_model_parameters():
    torch.manual_seed(42)

    images = torch.randn(8, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])

    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 10),
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    parameters_before_training = [parameter.detach().clone() for parameter in model.parameters()]

    train_one_epoch(
        model=model,
        dataloader=dataloader,
        loss_function=loss_function,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    parameters_after_training = list(model.parameters())

    assert any(
        not torch.equal(before, after)
        for before, after in zip(parameters_before_training, parameters_after_training)
    )

def test_train_one_epoch_returns_average_loss():
    torch.manual_seed(42)
    
    images = torch.randn(5, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 3, 4])

    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 10),
    )

    loss_function = nn.CrossEntropyLoss()
    with torch.no_grad():
        expected_loss = loss_function(model(images), labels).item()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    metrics = train_one_epoch(
        model=model,
        dataloader=dataloader,
        loss_function=loss_function,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert metrics["loss"] == pytest.approx(expected_loss, rel=1e-6)

def test_train_one_epoch_returns_accuracy():
    images = torch.zeros(4, 1, 28, 28)

    for sample_index in range(4):
        images[sample_index].view(-1)[sample_index] = 1.0

    labels = torch.tensor([0, 1, 9, 3])

    dataloader = DataLoader(
        TensorDataset(images, labels),
        batch_size=2, 
        shuffle=False,
    )

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 10, bias=False),
    )

    predicted_classes = [0, 1, 2, 3]

    with torch.no_grad():
        model[1].weight.zero_()

        for feature_index, predicted_class in enumerate(predicted_classes):
            model[1].weight[predicted_class, feature_index] = 1.0

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    metrics = train_one_epoch(
        model=model,
        dataloader=dataloader,
        loss_function=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        device=torch.device("cpu")
    )

    assert metrics["accuracy"] == pytest.approx(0.75)