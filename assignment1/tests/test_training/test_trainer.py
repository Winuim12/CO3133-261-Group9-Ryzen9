"""
RUN: python -m pytest tests/training/test_trainer.py -v
"""

import pytest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import train_model

def test_train_model_records_metrics_for_every_epoch(capsys):
    torch.manual_seed(42)

    images = torch.randn(8, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])

    dataset = TensorDataset(images, labels)

    train_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    validation_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 10),
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    history = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        device=torch.device("cpu"),
        epochs=2,
    )

    assert set(history) == {
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
    }

    assert all(len(metric_values) == 2 for metric_values in history.values())

    output = capsys.readouterr().out
    epoch_lines = [
        line for line in output.splitlines() if line.startswith("Epoch")
    ]

    assert len(epoch_lines) == 2
    assert epoch_lines[0].startswith("Epoch 1/2:")
    assert epoch_lines[1].startswith("Epoch 2/2:")
    assert all("train loss=" in line for line in epoch_lines)
    assert all("validation loss=" in line for line in epoch_lines)

def test_train_model_saves_only_best_validation_checkpoint(tmp_path, capsys):
    torch.manual_seed(42)

    images = torch.randn(8, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])

    dataloader = DataLoader(
        TensorDataset(images, labels),
        batch_size=4,
        shuffle=False,
    )

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 10),
    )

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.0,
    )

    checkpoint_path = tmp_path / "best_model.pt"

    history = train_model(
        model=model,
        train_loader=dataloader,
        validation_loader=dataloader,
        loss_function=loss_function,
        optimizer=optimizer,
        device=torch.device("cpu"),
        epochs=2,
        checkpoint_path=checkpoint_path,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint_path.exists()
    assert checkpoint["epoch"] == 1
    assert checkpoint["validation_loss"] == pytest.approx(history["validation_loss"][0])

    epoch_lines = [
        line for line in capsys.readouterr().out.splitlines() if line.startswith("Epoch")
    ]

    assert len(epoch_lines) == 2
    assert all("best epoch=1" in line for line in epoch_lines)

def test_train_model_stops_after_patience_without_improvement():
    torch.manual_seed(42)

    images = torch.randn(4, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 3])

    dataloader = DataLoader(
        TensorDataset(images, labels),
        batch_size=4,
        shuffle=False,
    )

    model = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 10))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    history = train_model(
        model=model,
        train_loader=dataloader,
        validation_loader=dataloader,
        loss_function=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        epochs=10,
        patience=2,
        min_delta=0.00001,
    )

    assert len(history["validation_loss"]) == 3