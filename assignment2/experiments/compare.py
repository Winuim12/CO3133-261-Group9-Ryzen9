#compare.py

import csv
import json
from argparse import ArgumentParser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_comparison_row(run_dir):
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads((run_dir / "validation_reload.json").read_text(encoding="utf-8"))

    if verification["matches_saved_metrics"] is not True:
        raise ValueError(f"Checkpoint verification failed: {run_dir}")

    if verification["checkpoint_epoch"] != summary["best_epoch"]:
        raise ValueError(f"Verified epoch differs from selected epoch: {run_dir}")

    model = metadata["config"]["model"]
    parameters = metadata["parameter_count"]
    metrics = verification["metrics"]

    row = {
        "model": model["name"],
        "freeze_backbone": model.get("freeze_backbone", ""),
        "run_id": run_dir.name,
        "epochs_completed": summary["epochs_completed"],
        "best_epoch": summary["best_epoch"],
        "validation_accuracy": metrics["accuracy"],
        "validation_macro_f1": metrics["macro_f1"],
        "total_parameters": parameters["total"],
        "trainable_parameters": parameters["trainable"],
        "training_seconds": summary["training_seconds"],
        "code_commit": metadata["code_commit"],
    }

    return row, metadata["manifest_hashes"]

def main():
    parser = ArgumentParser(description="Compare verified validation runs")
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--output", default="results/comparison.csv")
    args = parser.parse_args()

    rows = []
    reference_hashes = None

    for path in args.run_dirs:
        row, hashes = load_comparison_row(PROJECT_ROOT / path)

        if reference_hashes is None:
            reference_hashes = hashes
        elif hashes != reference_hashes:
            raise ValueError("Runs use different manifests or class mappings")

        rows.append(row)

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['model']}: "
            f"val Accuracy={row['validation_accuracy']:.4f}, "
            f"val Macro-F1={row['validation_macro_f1']:.4f}, "
            f"best epoch={row['best_epoch']}"
        )

    print(f"Saved comparison: {output_path}")


if __name__ == "__main__":
    main()