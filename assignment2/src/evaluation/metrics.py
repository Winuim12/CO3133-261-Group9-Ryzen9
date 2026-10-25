import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

def compute_metrics(y_true, y_pred, num_classes= 30):
    labels = list(range(num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "support": support.tolist()
        },
        "confusion_matrix": cm.tolist()
    }