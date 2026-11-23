"""Accuracy, Macro-F1, per-class scores, and confusion matrices.
"""

from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix as _sk_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

NUM_CLASSES = 5


def _as_numpy(values) -> np.ndarray:
    if hasattr(values, "detach"):  # torch.Tensor without importing torch here
        values = values.detach().cpu().numpy()
    array = np.asarray(values).reshape(-1)
    if array.size and not np.issubdtype(array.dtype, np.integer):
        raise TypeError("Targets and predictions must be integer class ids.")
    return array.astype(np.int64)


def _checked(targets, predictions, num_classes):
    y_true, y_pred = _as_numpy(targets), _as_numpy(predictions)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} targets vs {y_pred.shape} predictions.")
    if y_true.size == 0:
        raise ValueError("Metrics need at least one sample.")
    for name, array in (("targets", y_true), ("predictions", y_pred)):
        if array.min() < 0 or array.max() >= num_classes:
            raise ValueError(f"{name} contain ids outside 0..{num_classes - 1}.")
    return y_true, y_pred


def accuracy(targets, predictions, num_classes: int = NUM_CLASSES) -> float:
    y_true, y_pred = _checked(targets, predictions, num_classes)
    return float((y_true == y_pred).mean())


def macro_f1(targets, predictions, num_classes: int = NUM_CLASSES) -> float:
    """Unweighted mean F1 over all classes; used for checkpoint selection."""
    y_true, y_pred = _checked(targets, predictions, num_classes)
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), average="macro", zero_division=0
    )
    return float(f1)


def confusion_matrix(targets, predictions, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Rows are true labels, columns are predictions."""
    y_true, y_pred = _checked(targets, predictions, num_classes)
    return _sk_confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def label_names_from_mapping(label_mapping: dict) -> list[str]:
    """Convert {name: id} from the config into names ordered by id."""
    ordered = sorted(label_mapping.items(), key=lambda item: item[1])
    ids = [label_id for _, label_id in ordered]
    if ids != list(range(len(ids))):
        raise ValueError("Label ids must be contiguous and start at 0.")
    return [name for name, _ in ordered]


def classification_summary(targets, predictions, label_names: list[str]) -> dict:
    """All reported scores in one JSON-serialisable dictionary."""
    num_classes = len(label_names)
    y_true, y_pred = _checked(targets, predictions, num_classes)
    labels = list(range(num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    return {
        "num_samples": int(y_true.size),
        "accuracy": accuracy(y_true, y_pred, num_classes),
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
        "weighted_f1": float(weighted_f1),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
                "predicted": int((y_pred == i).sum()),
            }
            for i, name in enumerate(label_names)
        },
        "label_names": list(label_names),
        "confusion_matrix": confusion_matrix(y_true, y_pred, num_classes).tolist(),
    }


def save_confusion_matrix_plot(matrix, label_names, path, title="Confusion matrix") -> Path:
    """Row-normalised colours with raw counts written in every cell."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray(matrix)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums > 0)

    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(normalised, cmap="Blues", vmin=0.0, vmax=1.0)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Row proportion (recall)")
    axis.set(
        xticks=range(len(label_names)), yticks=range(len(label_names)),
        xticklabels=label_names, yticklabels=label_names,
        xlabel="Predicted label", ylabel="True label", title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            colour = "white" if normalised[i, j] > 0.5 else "black"
            axis.text(j, i, f"{matrix[i, j]}\n{normalised[i, j]:.0%}", ha="center", va="center",
                      color=colour, fontsize=8)
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
