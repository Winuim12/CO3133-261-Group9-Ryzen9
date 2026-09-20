#confusion_matrix.py

from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def calculate_confusion_matrix(predictions, targets, num_classes):
    predictions_array = predictions.detach().cpu().numpy()
    targets_array = targets.detach().cpu().numpy()

    return sklearn_confusion_matrix(
        y_true=targets_array,
        y_pred=predictions_array,
        labels=list(range(num_classes)),
    )

def save_confusion_matrix_plot(matrix, class_names, path, title):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(figsize=(10, 8))

    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)

    display.plot(
        ax=axes,
        cmap="Blues",
        values_format="d",
        xticks_rotation=45,
        colorbar=False,
    )

    axes.set_title(title)
    figure.tight_layout()

    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)