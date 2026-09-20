#test_validation
"""
RUN: python -m pytest tests/training/test_validation.py -v
"""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.validate import validate_one_epoch

def test_validate_one_epoch_returns_accuracy():
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

    metrics = validate_one_epoch(
        model=model,
        dataloader=dataloader,
        loss_function=nn.CrossEntropyLoss(),
        device=torch.device("cpu")
    )

    assert metrics["accuracy"] == pytest.approx(0.75)