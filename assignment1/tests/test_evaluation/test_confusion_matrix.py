#test_confusion_matrix.py
"""
RUN: python -m pytest tests/test_evaluation/test_confusion_matrix.py -v
"""

import numpy as np
import torch

from src.evaluation.confusion_matrix import calculate_confusion_matrix, save_confusion_matrix_plot

def test_calculate_confusion_matrix_counts_actual_and_predicted_classes():
    targets = torch.tensor([0, 0, 1, 1, 2, 2])
    predictions = torch.tensor([0, 1, 1, 1, 0, 2])

    matrix = calculate_confusion_matrix(predictions, targets, num_classes=3)

    expected_matrix = np.array(
        [
            [1, 1, 0],
            [0, 2, 0],
            [1, 0, 1],
        ]
    )

    np.testing.assert_array_equal(matrix, expected_matrix)

def test_save_confusion_matrix_plot_writes_image(tmp_path):
    matrix = np.array(
        [
            [8, 2],
            [1, 9],
        ]
    )

    output_path = tmp_path / "figures" / "confusion_matrix.png"

    save_confusion_matrix_plot(
        matrix=matrix,
        class_names=["Class 0", "Class 1"],
        path=output_path,
        title="Test Confusion Matrix",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    