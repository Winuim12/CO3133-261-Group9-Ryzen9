import pytest

from src.evaluation.metrics import compute_metrics


def test_metrics_on_known_predictions():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 2, 1, 2]

    result = compute_metrics(y_true, y_pred, num_classes=3)

    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["macro_f1"] == pytest.approx(2 / 3)

    per_class = result["per_class"]

    assert per_class["precision"] == pytest.approx([1.0, 0.5, 0.5])
    assert per_class["recall"] == pytest.approx([1.0, 0.5, 0.5])
    assert per_class["f1"] == pytest.approx([1.0, 0.5, 0.5])
    assert per_class["support"] == [2, 2, 2]

    assert result["confusion_matrix"] == [
        [2, 0, 0],
        [0, 1, 1],
        [0, 1, 1],
    ]


def test_metrics_include_all_30_classes_when_predictions_miss_classes():
    y_true = list(range(30))
    y_pred = [0] * 30

    result = compute_metrics(y_true, y_pred)

    # Only the sample belonging to class 0 is correct.
    assert result["accuracy"] == pytest.approx(1 / 30)

    # Class 0: precision = 1/30, recall = 1, F1 = 2/31.
    # The other 29 classes have F1 = 0.
    assert result["macro_f1"] == pytest.approx((2 / 31) / 30)

    per_class = result["per_class"]

    assert per_class["precision"] == pytest.approx(
        [1 / 30] + [0.0] * 29
    )
    assert per_class["recall"] == pytest.approx(
        [1.0] + [0.0] * 29
    )
    assert per_class["f1"] == pytest.approx(
        [2 / 31] + [0.0] * 29
    )
    assert per_class["support"] == [1] * 30

    expected_cm = [
        [1] + [0] * 29
        for _ in range(30)
    ]
    assert result["confusion_matrix"] == expected_cm