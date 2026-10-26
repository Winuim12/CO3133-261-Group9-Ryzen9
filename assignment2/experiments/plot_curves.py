#plot_curves.py

import json
from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def plot_curves(run_dirs, output_path):
    figure, axes = plt.subplots(
        len(run_dirs),
        3,
        figsize=(15, 4 * len(run_dirs)),
        squeeze=False,
    )

    metrics = (
        ("loss", "Cross-Entropy loss"),
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
    )

    for row, run_dir in enumerate(run_dirs):
        history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))

        model = config["model"]
        label = model["name"]
        if model.get("pretrained"):
            label += (
                " (frozen)"
                if model.get("freeze_backbone")
                else " (full fine-tune)"
            )

        epochs = [record["epoch"] for record in history]

        for column, (metric, title) in enumerate(metrics):
            axis = axes[row, column]

            for split in ("train", "val"):
                values = [
                    record[split][metric]
                    for record in history
                ]
                axis.plot(epochs, values, label=split)

            axis.axvline(
                summary["best_epoch"],
                color="gray",
                linestyle=":",
                label="Selected checkpoint",
            )

            axis.set_title(f"{label}: {title}")
            axis.set_xlabel("Epoch")
            axis.set_ylabel(title)
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)

            if metric != "loss":
                axis.set_ylim(0, 1)

    figure.suptitle("M2: training and validation histories")
    figure.tight_layout(rect=(0, 0, 1, 0.96))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    print(f"Saved curves: {output_path}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Plot saved training histories")
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument(
        "--output",
        default="results/figures/m2_training_histories.png"
    )
    args = parser.parse_args()

    plot_curves(
        [PROJECT_ROOT / path for path in args.run_dirs],
        PROJECT_ROOT / args.output,
    )
