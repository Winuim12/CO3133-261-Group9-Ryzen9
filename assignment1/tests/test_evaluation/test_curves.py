#test_curves.py
"""
RUN: python -m pytest tests/test_evaluation/test_curves.py -v
"""

from src.evaluation.curves import save_training_curves

def test_save_training_curves_writes_image(tmp_path):
    history = {
        "train_loss": [0.8, 0.5, 0.3],
        "validation_loss": [0.9, 0.6, 0.4],
        "train_accuracy": [0.7, 0.82, 0.9],
        "validation_accuracy": [0.68, 0.8, 0.87],
    }

    output_path = tmp_path / "figures" / "training_curves.png"

    save_training_curves(
        history=history,
        path=output_path,
        title="LinearClassifier",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0