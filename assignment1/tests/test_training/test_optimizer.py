#test_optimizer.py
"""
RUN: python -m pytest tests/training/test_optimizer.py -v
"""

import pytest
import torch
from torch import nn

from src.training.optimizer import create_optimizer

def test_create_optimizer_builds_adamw_with_requested_settings():
    model = nn.Linear(4, 2)

    optimizer = create_optimizer(
        model=model,
        optimizer_name="adamw",
        learning_rate=0.001,
        weight_decay=0.0001,
    )

    assert isinstance(optimizer, torch.optim.AdamW)

    parameter_group = optimizer.param_groups[0]

    assert parameter_group["lr"] == pytest.approx(0.001)
    assert parameter_group["weight_decay"] == pytest.approx(0.0001)
    