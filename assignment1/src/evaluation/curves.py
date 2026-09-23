#curves.py

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

def save_training_curves(history, path, title):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    #Loss curve
    axes[0].plot(epochs, history["train_loss"], label="Training")
    axes[0].plot(epochs, history["validation_loss"], label="Validation")

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    #Accuracy curve
    axes[1].plot(epochs, history["train_accuracy"], label="Training")
    axes[1].plot(epochs, history["validation_accuracy"], label="Validation")

    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.suptitle(title)
    figure.tight_layout()

    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)