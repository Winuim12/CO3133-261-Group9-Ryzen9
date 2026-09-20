"""
RUN: python -m pytest tests/training/test_checkpoint.py -v
"""

import pytest
import torch
from torch import nn

from src.training.checkpoint import save_checkpoint, load_checkpoint

def test_save_checkpoint_writes_training_state(tmp_path):
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    checkpoint_path = tmp_path / "best_model.pt"

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=3,
        validation_loss=0.75,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint_path.exists()
    assert checkpoint["epoch"] == 3
    assert checkpoint["validation_loss"] == pytest.approx(0.75)
    assert checkpoint["optimizer_state_dict"] == optimizer.state_dict()

    for parameter_name, parameter_value in model.state_dict().items():
        assert torch.equal(checkpoint["model_state_dict"][parameter_name], parameter_value)

def test_load_checkpoint_restores_model_parameters(tmp_path):
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    original_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.state_dict().items()
    }

    checkpoint_path = tmp_path / "best_model.pt"

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=3,
        validation_loss=0.75,
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)

    metadata = load_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    for parameter_name, original_value in original_parameters.items():
        assert torch.equal(model.state_dict()[parameter_name], original_value)

    assert metadata["epoch"] == 3
    assert metadata["validation_loss"] == pytest.approx(0.75)

def test_load_checkpoint_restores_model_without_optimizer(tmp_path):
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    original_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.state_dict().items()
    }

    checkpoint_path = tmp_path / "best_model.pt"

    save_checkpoint(
        path=checkpoint_path, 
        model=model,
        optimizer=optimizer,
        epoch=3,
        validation_loss=0.75,
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)

    metadata = load_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=None,
        device=torch.device("cpu"),
    )

    for parameter_name, original_value in original_parameters.items():
        assert torch.equal(model.state_dict()[parameter_name], original_value)

    assert metadata["epoch"] == 3
    assert metadata["validation_loss"] == pytest.approx(0.75)
