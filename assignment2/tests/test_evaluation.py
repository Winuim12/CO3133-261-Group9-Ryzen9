import pytest
import torch
from torch import nn

from experiments.evaluate import prediction_rows
from src.evaluation.benchmark import benchmark_inference
from src.evaluation.error_analysis import errors_by_confidence, top_confused_pairs


def test_top_confused_pairs_ranks_off_diagonal_counts():
    confusion = [[5, 2, 0], [1, 3, 0], [0, 0, 4]]

    pairs = top_confused_pairs(confusion, ["a", "b", "c"])

    assert [(p["true"], p["predicted"], p["count"]) for p in pairs] == [("a", "b", 2), ("b", "a", 1)]
    assert pairs[0]["rate_of_true_class"] == pytest.approx(2 / 7)


def test_errors_by_confidence_counts_only_wrong_predictions():
    counts = errors_by_confidence([0.3, 0.6, 0.9, 1.0, 0.95], [False, False, False, False, True])

    assert counts == {"<0.5": 1, "0.5-0.8": 1, ">=0.8": 2}


def test_prediction_rows_follow_record_order():
    records = [
        {"image_id": "x", "path": "p/x.jpg", "group_id": "g1"},
        {"image_id": "y", "path": "p/y.jpg", "group_id": "g2"},
    ]
    probabilities = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])

    rows = prediction_rows(records, ["a", "b", "c"], [0, 1], probabilities)

    assert [row["pred_class"] for row in rows] == ["a", "c"]
    assert [row["correct"] for row in rows] == [True, False]
    assert rows[1]["true_class_prob"] == pytest.approx(0.3)
    assert rows[1]["top3"].startswith("c:0.6000|b:0.3000")


def test_benchmark_inference_reports_each_batch_size_on_cpu():
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2))

    results = benchmark_inference(model, "cpu", image_size=32, batch_sizes=(1, 4), warmup=1, iterations=2)

    assert [(r["batch_size"], r["precision"]) for r in results] == [(1, "fp32"), (4, "fp32")]
    assert all(r["images_per_second"] > 0 and r["peak_allocated_mb"] is None for r in results)
