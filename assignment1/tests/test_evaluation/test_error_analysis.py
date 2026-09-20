#test_error_analysis.py
"""
RUN: python -m pytest tests/test_evaluation/test_error_analysis.py -v
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.error_analysis import save_prediction_examples

def test_save_prediction_examples_writes_image(tmp_path):
    images = torch.rand(6, 1, 28, 28)

    targets = torch.tensor([0, 1, 2, 0, 1, 2])
    predictions = torch.tensor([0, 2, 2, 1, 1, 0])

    dataset = TensorDataset(images, targets)

    output_path = tmp_path / "figures" / "prediction_examples.png"

    save_prediction_examples(
        dataset=dataset,
        targets=targets,
        predictions=predictions,
        class_names=["Class 0", "Class 1", "Class 2"],
        mean=0.5,
        std=0.5,
        path=output_path,
        examples_per_group=3,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0