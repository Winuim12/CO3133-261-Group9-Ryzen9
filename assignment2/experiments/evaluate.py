#evaluate.py
'''
Final evaluation of a run's selected checkpoint (best validation Macro-F1).

Writes into the run directory, per split:
    evaluation_<split>.json   metrics, confusion matrix, confused pairs, timing
    predictions_<split>.csv   one row per image for qualitative/error analysis
    per_class_<split>.csv     precision/recall/F1/support per class
    confusion_<split>.png     row-normalized confusion matrix
    qualitative_<split>.png   confident correct, hard correct, confident errors
and inference_benchmark.json (model-only latency/throughput, parameters).

The checkpoint is already chosen on validation; test results must not be used
to change models or hyperparameters.
'''

import csv
import json
import math
from argparse import ArgumentParser
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from experiments.context import prepare_data_loaders
from experiments.verify_checkpoint import load_best_model
from src.evaluation.benchmark import benchmark_inference
from src.evaluation.error_analysis import errors_by_confidence, top_confused_pairs
from src.evaluation.metrics import compute_metrics


def predict(model, loader, device, channels_last=False):
    """FP32 softmax probabilities in loader order, plus end-to-end seconds."""
    model.eval()
    labels, probabilities = [], []
    start = perf_counter()

    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device, non_blocking=True, memory_format=(
                torch.channels_last if channels_last else torch.preserve_format
            ))
            probabilities.append(torch.softmax(model(images).float(), dim=1).cpu())
            labels.extend(batch_labels.tolist())

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    return labels, torch.cat(probabilities), perf_counter() - start


def prediction_rows(records, classes, labels, probabilities):
    confidences, predictions = probabilities.max(dim=1)
    top3_probabilities, top3_indices = probabilities.topk(min(3, len(classes)), dim=1)
    rows = []

    for index, record in enumerate(records):
        predicted = int(predictions[index])
        rows.append({
            "image_id": record["image_id"],
            "path": record["path"],
            "group_id": record["group_id"],
            "true_idx": labels[index],
            "true_class": classes[labels[index]],
            "pred_idx": predicted,
            "pred_class": classes[predicted],
            "confidence": float(confidences[index]),
            "true_class_prob": float(probabilities[index, labels[index]]),
            "correct": predicted == labels[index],
            "top3": "|".join(
                f"{classes[int(c)]}:{float(p):.4f}"
                for c, p in zip(top3_indices[index], top3_probabilities[index])
            ),
        })

    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_confusion(confusion, classes, title, output_path):
    confusion = np.asarray(confusion, dtype=float)
    normalized = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1)

    figure, axis = plt.subplots(figsize=(13, 12))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
    axis.set_yticks(range(len(classes)), classes, fontsize=7)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Rate within true class")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_qualitative(rows, raw_dir, title, output_path, per_group=8):
    correct = [row for row in rows if row["correct"]]
    errors = [row for row in rows if not row["correct"]]
    groups = (
        ("Confident correct", sorted(correct, key=lambda r: -r["confidence"])[:per_group]),
        ("Hard but correct (lowest confidence)", sorted(correct, key=lambda r: r["confidence"])[:per_group]),
        ("Confident errors", sorted(errors, key=lambda r: -r["confidence"])[:per_group]),
    )

    figure, axes = plt.subplots(len(groups), per_group, figsize=(2.2 * per_group, 2.8 * len(groups)),
                                squeeze=False)
    for row_index, (group_title, examples) in enumerate(groups):
        for column in range(per_group):
            axis = axes[row_index, column]
            axis.set_xticks([])
            axis.set_yticks([])
            if column >= len(examples):
                axis.axis("off")
                continue
            example = examples[column]
            with Image.open(raw_dir / example["path"]) as image:
                image = image.convert("RGB")
                image.thumbnail((224, 224))
                axis.imshow(image)
            axis.set_title(
                f"T: {example['true_class']}\nP: {example['pred_class']} ({example['confidence']:.2f})",
                fontsize=7, color="green" if example["correct"] else "red",
            )
        if examples:
            axes[row_index, 0].set_ylabel(group_title, fontsize=9)

    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_path, dpi=130)
    plt.close(figure)


def evaluate_run(run_dir, splits=("val", "test"), benchmark=True):
    loaded = load_best_model(run_dir)
    run_dir, config, metadata, history = (
        loaded[name] for name in ("run_dir", "config", "metadata", "history")
    )
    context, runtime, checkpoint, model = (
        loaded[name] for name in ("context", "runtime", "checkpoint", "model")
    )
    device = context.device
    raw_dir = context.project_root / config["data"]["raw_dir"]
    _, val_loader, test_loader = prepare_data_loaders(context)
    loaders = {"val": val_loader, "test": test_loader}
    label = config["model"]["name"] + (
        "" if not config["model"].get("pretrained") else
        " (frozen)" if config["model"].get("freeze_backbone") else " (full fine tuning)"
    )

    for split in splits:
        dataset = loaders[split].dataset
        if dataset.class_to_idx != metadata["class_to_idx"]:
            raise ValueError("Class mapping differs from the saved run.")
        classes = dataset.classes
        labels, probabilities, seconds = predict(
            model, loaders[split], device, channels_last=runtime["channels_last"]
        )
        rows = prediction_rows(dataset.records, classes, labels, probabilities)
        metrics = compute_metrics(
            labels, [row["pred_idx"] for row in rows], num_classes=config["data"]["num_classes"]
        )

        if split == "val":
            # Guards against row-order or preprocessing mismatches with the trainer.
            saved = next(r for r in history if r["epoch"] == checkpoint["epoch"])["val"]
            for name in ("accuracy", "macro_f1"):
                if not math.isclose(metrics[name], saved[name], rel_tol=0, abs_tol=1e-9):
                    raise ValueError(f"Validation {name} differs from the saved history.")

        write_csv(run_dir / f"predictions_{split}.csv", rows)
        write_csv(run_dir / f"per_class_{split}.csv", [
            {"class_idx": index, "class_name": name,
             **{key: metrics["per_class"][key][index] for key in ("precision", "recall", "f1", "support")}}
            for index, name in enumerate(classes)
        ])
        (run_dir / f"evaluation_{split}.json").write_text(json.dumps({
            "split": split,
            "checkpoint_epoch": checkpoint["epoch"],
            "num_images": len(rows),
            "classes": classes,
            "metrics": metrics,
            "error_count": sum(not row["correct"] for row in rows),
            "errors_by_confidence": errors_by_confidence(
                [row["confidence"] for row in rows], [row["correct"] for row in rows]
            ),
            "top_confused_pairs": top_confused_pairs(metrics["confusion_matrix"], classes),
            "end_to_end_seconds": seconds,
            "end_to_end_images_per_second": len(rows) / seconds,
            "timing_scope": "FP32 forward pass including image decoding, transforms and transfer",
        }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        plot_confusion(metrics["confusion_matrix"], classes,
                       f"{label}: {split} confusion matrix", run_dir / f"confusion_{split}.png")
        plot_qualitative(rows, raw_dir, f"{label}: {split} examples",
                         run_dir / f"qualitative_{split}.png")
        print(f"{split}: Accuracy={metrics['accuracy']:.4f}, "
              f"Macro F1={metrics['macro_f1']:.4f}, {len(rows)} images", flush=True)

    if benchmark:
        results = benchmark_inference(
            model, device, image_size=config["data"]["image_size"],
            channels_last=runtime["channels_last"],
        )
        (run_dir / "inference_benchmark.json").write_text(json.dumps({
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "parameter_count": metadata.get("parameter_count"),
            "checkpoint_mb": (run_dir / "best_model.pt").stat().st_size / 2**20,
            "timing_scope": "Model forward pass only on synthetic inputs; excludes data loading",
            "results": results,
        }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for result in results:
            print(f"batch={result['batch_size']} {result['precision']}: "
                  f"{result['ms_per_image']:.3f} ms/image, "
                  f"{result['images_per_second']:.1f} images/s", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate a run's selected checkpoint")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--splits", nargs="+", choices=("val", "test"), default=["val", "test"])
    parser.add_argument("--no-benchmark", action="store_true")
    args = parser.parse_args()

    evaluate_run(args.run_dir, splits=args.splits, benchmark=not args.no_benchmark)
