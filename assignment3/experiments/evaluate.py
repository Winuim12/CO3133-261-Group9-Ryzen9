"""Evaluate a saved run's best checkpoint on the validation or test split.
    python -m experiments.evaluate --run <run-id or run directory> --split val
    python -m experiments.evaluate --run <run-id> --split test   # locked configs only
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import torch

from experiments.context import build_data, move_batch
from src.config import PROJECT_ROOT
from src.metrics import classification_summary, label_names_from_mapping, save_confusion_matrix_plot
from src.models.factory import build_model
from src.reproducibility import (
    data_fingerprint, find_run_dir, load_json, relative_to_root, resolve_device, save_json,
)


@torch.no_grad()
def predict(model, loader, device) -> dict:
    """Logits, predictions, and (for gated/fixed fusion) the mean gate per sample."""
    model.eval()
    names, targets, logits_list, gates = [], [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        logits = model(batch)
        logits_list.append(logits.float().cpu())
        targets.append(batch["label"].cpu())
        names.extend(batch["image_name"])
        gate = getattr(model, "last_gate", None)
        if gate is not None:
            gates.append(gate.float().mean(dim=1).cpu())
    if not targets:
        raise ValueError("Evaluation loader produced no samples.")
    logits = torch.cat(logits_list)
    return {
        "image_names": names,
        "targets": torch.cat(targets),
        "logits": logits,
        "probabilities": torch.softmax(logits, dim=1),
        "predictions": logits.argmax(dim=1),
        "gate_mean": torch.cat(gates) if gates else None,
    }


def gate_summary(outputs: dict, label_names: list[str]) -> dict | None:
    """Mean image weight overall, per true class, and for correct/incorrect predictions."""
    gate = outputs["gate_mean"]
    if gate is None:
        return None
    targets, correct = outputs["targets"], outputs["predictions"] == outputs["targets"]

    def stats(mask):
        values = gate[mask]
        if values.numel() == 0:
            return None
        return {"n": int(values.numel()), "mean": float(values.mean()), "std": float(values.std(unbiased=False)),
                "min": float(values.min()), "max": float(values.max())}

    return {
        "meaning": "mean gate weight on the image projection (1 = image only, 0 = text only)",
        "overall": stats(torch.ones_like(gate, dtype=torch.bool)),
        "correct": stats(correct),
        "incorrect": stats(~correct),
        "per_true_class": {name: stats(targets == i) for i, name in enumerate(label_names)},
    }


def save_predictions(path: Path, outputs: dict, label_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        header = ["image_name", "label_id", "label", "pred_id", "pred", "correct"]
        header += [f"prob_{name}" for name in label_names]
        if outputs["gate_mean"] is not None:
            header.append("gate_image_mean")
        writer.writerow(header)
        for i, name in enumerate(outputs["image_names"]):
            target, prediction = int(outputs["targets"][i]), int(outputs["predictions"][i])
            row = [name, target, label_names[target], prediction, label_names[prediction], int(target == prediction)]
            row += [f"{p:.6f}" for p in outputs["probabilities"][i].tolist()]
            if outputs["gate_mean"] is not None:
                row.append(f"{float(outputs['gate_mean'][i]):.6f}")
            writer.writerow(row)


def evaluate_model(model, loader, device, config: dict, split: str, run_dir: Path,
                   extra: dict | None = None) -> dict:
    """Write evaluation_<split>.json, predictions_<split>.csv, and a confusion-matrix figure."""
    label_names = label_names_from_mapping(config["labels"])
    outputs = predict(model, loader, device)
    summary = classification_summary(outputs["targets"], outputs["predictions"], label_names)
    figure = save_confusion_matrix_plot(
        summary["confusion_matrix"], label_names, run_dir / "figures" / f"confusion_matrix_{split}.png",
        title=f"{config['model']['name']} — {split} (Macro-F1 {summary['macro_f1']:.3f})",
    )
    predictions_path = run_dir / f"predictions_{split}.csv"
    save_predictions(predictions_path, outputs, label_names)
    result = {
        "model": config["model"]["name"],
        "split": split,
        "evaluated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        **summary,
        "gate": gate_summary(outputs, label_names),
        "files": {"predictions": predictions_path.name,
                  "confusion_matrix": relative_to_root(figure, run_dir)},
        **(extra or {}),
    }
    save_json(run_dir / f"evaluation_{split}.json", result)
    return result


def load_checkpoint_into(model, checkpoint_path: Path, config: dict) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("labels") != config["labels"]:
        raise ValueError("Checkpoint label mapping differs from the run configuration.")
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="Run id (results/runs/<model>/<id>) or run directory.")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--device", default=None, help="auto, cpu, or cuda (default: saved config).")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--allow-data-mismatch", action="store_true",
                        help="Evaluate even if manifest/vocabulary hashes changed since training.")
    args = parser.parse_args(argv)

    run_dir = find_run_dir(args.run)
    config = load_json(run_dir / "config.json")
    metadata = load_json(run_dir / "run_metadata.json")
    for option, key in (("device", "device"), ("batch_size", "batch_size"), ("num_workers", "num_workers")):
        if getattr(args, option) is not None:
            config["training"][key] = getattr(args, option)

    current = data_fingerprint(config)
    recorded = metadata["data"]
    changed = [key for key in ("train_manifest", "val_manifest", "test_manifest", "vocabulary")
               if current.get(key, {}).get("sha256") != recorded.get(key, {}).get("sha256")]
    if current.get("text_tokenizer") != recorded.get("text_tokenizer"):
        changed.append("text_tokenizer")
    if changed and not args.allow_data_mismatch:
        parser.exit(1, f"Data changed since training ({', '.join(changed)}). "
                       "Use the original data or pass --allow-data-mismatch.\n")
    if args.split == "test":
        print("NOTE: test split — use only for configurations already locked on validation.", flush=True)

    device = resolve_device(config["training"]["device"])
    data = build_data(config, splits=(args.split,))
    model = build_model(config, vocab_size=len(data.vocab), load_pretrained=False)
    checkpoint_path = PROJECT_ROOT / metadata["checkpoint_path"]
    checkpoint = load_checkpoint_into(model, checkpoint_path, config)
    model.to(device)

    result = evaluate_model(
        model, data.loaders[args.split], device, config, args.split, run_dir,
        extra={"run_id": metadata["run_id"], "checkpoint_epoch": checkpoint["epoch"],
               "debug_run": metadata.get("debug_run", False), "device": str(device),
               "data_changed_since_training": changed},
    )
    print(f"{config['model']['name']} | {args.split} | n={result['num_samples']} | "
          f"accuracy {result['accuracy']:.4f} | macro-F1 {result['macro_f1']:.4f}")
    print(f"Saved: {run_dir / f'evaluation_{args.split}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
