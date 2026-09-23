#test_factory.py
"""
RUN: python -m pytest tests/test_models/test_factory.py -v
"""

import pytest

from src.models.factory import create_model
from src.models.cnn import CNNClassifier
from src.models.mlp import MLPClassifier
from src.models.linear import LinearClassifier
from src.models.rnn import RecurrentClassifier
from src.models.transformer import VisionTransformerClassifier

@pytest.mark.parametrize(
    ("model_name", "expected_type"),
    [
        ("linear", LinearClassifier),
        ("mlp", MLPClassifier),
        ("cnn", CNNClassifier),
        ("rnn", RecurrentClassifier),
        ("transformer", VisionTransformerClassifier),
    ],
)

def test_create_model_returns_requested_model_class(model_name, expected_type):
    model = create_model(model_name=model_name, model_parameters={})
    assert isinstance(model, expected_type)