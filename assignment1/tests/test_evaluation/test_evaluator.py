#test_evaluator.py
"""
RUN: python -m pytest tests/test_evaluation/test_evaluator -v
"""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

from src.evaluation.evaluator import evaluate_model

def test_evaluation_model_returns_metrics_and_predictions():
    images = torch.zeros(4, 1, 28, 28)

    for sample_index in range(4):
        images[sample_index].view(-1)[sample_index] = 1.0

    targets = torch.tensor([0, 1, 9, 3])

    dataloader = DataLoader(
        TensorDataset(images, targets),
        batch_size=2, 
        shuffle=False,
    )

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 10, bias=False),
    )

    expected_predictions = torch.tensor([0, 1, 2, 3])
    expected_confusion_matrix = np.zeros((10, 10), dtype=int)

    expected_confusion_matrix[0, 0] = 1
    expected_confusion_matrix[1, 1] = 1
    expected_confusion_matrix[9, 2] = 1
    expected_confusion_matrix[3, 3] = 1

    with torch.no_grad():
        model[1].weight.zero_()

        for feature_index, predicted_class in enumerate(expected_predictions):
            model[1].weight[predicted_class, feature_index] = 1.0

        expected_loss = nn.CrossEntropyLoss()(model(images), targets).item()

    results = evaluate_model(
        model=model,
        dataloader=dataloader,
        loss_function=nn.CrossEntropyLoss(),
        device=torch.device("cpu")
    )

    assert results["loss"] == pytest.approx(expected_loss)
    assert results["accuracy"] == pytest.approx(0.75)
    assert torch.equal(results["predictions"], expected_predictions)
    assert torch.equal(results["targets"], targets)
    assert results["macro_f1"] == pytest.approx(0.6)
    assert results["trainable_parameters"] == 7_840
    np.testing.assert_array_equal(results["confusion_matrix"], expected_confusion_matrix)

    assert results["inference_time_seconds"] > 0

    assert results["inference_time_per_sample_seconds"] == pytest.approx(results["inference_time_seconds"] / len(targets))

def test_evaluate_model_warms_up_before_timing():
    class CountingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.forward_calls = 0

        def forward(self, images):
            self.forward_calls += 1

            return torch.zeros(
                images.size(0),
                10,
                device=images.device,
            )

    images = torch.zeros(4, 1, 28, 28)
    targets = torch.tensor([0, 1, 2, 3])

    dataloader = DataLoader(
        TensorDataset(images, targets),
        batch_size=2,
        shuffle=False,
    )

    model = CountingModel()

    evaluate_model(
        model=model,
        dataloader=dataloader,
        loss_function=nn.CrossEntropyLoss(),
        device=torch.device("cpu")
    )

    #One warm-up call plus two evaluated batches
    assert model.forward_calls == 3