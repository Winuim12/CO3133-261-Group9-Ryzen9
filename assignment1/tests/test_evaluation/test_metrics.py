#test_metrics.py
"""
RUN: python -m pytest tests/test_evaluation/test_metrics.py -v
"""

import pytest
import torch
from torch import nn

from src.evaluation.metrics import calculate_macro_f1, count_trainable_parameters

def test_calculate_macro_f1_average_class_f1_scores():
    targets = torch.tensor([0, 0, 1, 1]) 
    predictions = torch.tensor([0, 1, 1, 1])

    macro_f1 = calculate_macro_f1(predictions, targets)
    #Per-class F1 = 2TP / (2TP + FP + FN)
    #F10 = 2/3, F11 = 4/5, Macro F1 = (2/3 + 4/5)/2 = 11/15
    assert macro_f1 == pytest.approx(11 / 15)

def test_count_trainable_parameters_excludes_frozen_parameters():
    model = nn.Sequential(
        nn.Linear(4, 3),
        nn.Linear(3, 2, bias=False),
    )

    for parameter in model[0].parameters():
        parameter.requires_grad = False

    parameter_count = count_trainable_parameters(model)

    assert parameter_count == 6