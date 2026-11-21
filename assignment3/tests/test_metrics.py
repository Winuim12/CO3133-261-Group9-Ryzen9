import json

import numpy as np
import pytest
import torch

from src.metrics import (
    accuracy, classification_summary, confusion_matrix, label_names_from_mapping, macro_f1,
    save_confusion_matrix_plot,
)

LABELS = {"very_negative": 0, "negative": 1, "neutral": 2, "positive": 3, "very_positive": 4}
NAMES = ["very_negative", "negative", "neutral", "positive", "very_positive"]


def test_label_names_follow_ids():
    shuffled = dict(reversed(list(LABELS.items())))
    assert label_names_from_mapping(shuffled) == NAMES
    with pytest.raises(ValueError):
        label_names_from_mapping({"a": 0, "b": 2})


def test_macro_f1_counts_missing_classes_as_zero():
    targets = torch.tensor([3, 3, 2, 2])
    predictions = torch.tensor([3, 3, 2, 3])
    # positive: P=2/3 R=1 F1=0.8; neutral: P=1 R=0.5 F1=2/3; three absent classes score 0.
    assert macro_f1(targets, predictions) == pytest.approx((0.8 + 2 / 3) / 5)
    assert accuracy(targets, predictions) == pytest.approx(0.75)


def test_confusion_matrix_is_always_five_by_five():
    matrix = confusion_matrix([0, 4], [0, 0])
    assert matrix.shape == (5, 5)
    assert matrix[0, 0] == 1 and matrix[4, 0] == 1 and matrix.sum() == 2


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        macro_f1([0, 1], [0])
    with pytest.raises(ValueError):
        macro_f1([0, 5], [0, 1])
    with pytest.raises(TypeError):
        macro_f1(np.array([0.0, 1.0]), np.array([0.0, 1.0]))


def test_summary_is_json_serialisable_and_plot_is_written(tmp_path):
    summary = classification_summary([0, 1, 2, 3, 4, 4], [0, 1, 2, 3, 4, 3], NAMES)
    json.dumps(summary)
    assert summary["per_class"]["very_positive"]["support"] == 2
    assert summary["per_class"]["positive"]["predicted"] == 2
    path = save_confusion_matrix_plot(summary["confusion_matrix"], NAMES, tmp_path / "cm.png")
    assert path.stat().st_size > 0
