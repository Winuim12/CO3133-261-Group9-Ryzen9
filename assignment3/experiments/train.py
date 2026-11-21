"""Train one configuration with the shared trainer and evaluate it on validation.

    python -m experiments.train --config config/concat.yaml
    python -m experiments.train --config config/gated.yaml --seed 43 --device cuda

Outputs (generated, git-ignored):
    results/runs/<model>/<run-id>/   config.json, run_metadata.json, training.json,
                                     evaluation_val.json, predictions_val.csv,
                                     pip_freeze.txt, figures/
    checkpoints/<model>/<run-id>/best_model.pt   (selected by validation Macro-F1)
"""

import argparse
import math
import sys
from pathlib import Path
from time import perf_counter

import torch

from experiments.context import apply_overrides, build_data
from experiments.evaluate import evaluate_model, load_checkpoint_into
from src.config import load_config
from src.engine import fit
from src.metrics import macro_f1
from src.models.factory import build_model
from src.reproducibility import (
    build_run_metadata, count_parameters, create_run_paths, relative_to_root, resolve_device,
    save_json, save_pip_freeze, set_seed,
)

RELOAD_TOLERANCE = 1e-4


def create_optimizer(model, training: dict):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    name = str(training["optimizer"]).lower()
    lr, weight_decay = float(training["learning_rate"]), float(training["weight_decay"])
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer {name!r}.")


def save_training_curves(history: list[dict], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    figure, (loss_axis, f1_axis) = plt.subplots(1, 2, figsize=(10, 4))
    loss_axis.plot(epochs, [row["train_loss"] for row in history], marker="o", label="train")
    loss_axis.plot(epochs, [row["val_loss"] for row in history], marker="o", label="validation")
    loss_axis.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")
    loss_axis.legend()
    f1_axis.plot(epochs, [row["val_macro_f1"] for row in history], marker="o", color="tab:green")
    best = max(history, key=lambda row: row["val_macro_f1"])
    f1_axis.axvline(best["epoch"], linestyle="--", color="grey", label=f"best epoch {best['epoch']}")
    f1_axis.set(xlabel="Epoch", ylabel="Macro-F1", title="Validation Macro-F1")
    f1_axis.legend()
    for axis in (loss_axis, f1_axis):
        axis.grid(alpha=0.3)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="e.g. config/gated.yaml (merged over base.yaml)")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--deterministic", action="store_true", help="Deterministic kernels (slower).")
    parser.add_argument("--run-name", help="Optional suffix for the run id.")
    parser.add_argument("--limit-train", type=int, help="DEBUG ONLY: random training subset size.")
    parser.add_argument("--limit-val", type=int, help="DEBUG ONLY: random validation subset size.")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="DEBUG ONLY: random ResNet weights (offline smoke runs).")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = apply_overrides(load_config(args.config), args)
    debug = bool(args.limit_train or args.limit_val or args.no_pretrained)
    if args.no_pretrained:
        config["model"]["image_weights"] = None
    training = config["training"]
    seed = int(training["seed"])

    set_seed(seed, deterministic=args.deterministic)
    device = resolve_device(training["device"])
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = not args.deterministic

    data = build_data(config, splits=("train", "val"),
                      limits={"train": args.limit_train, "val": args.limit_val})
    model = build_model(config, vocab_size=len(data.vocab)).to(device)
    optimizer = create_optimizer(model, training)

    run_name = "-".join(part for part in ("debug" if debug else None, args.run_name) if part) or None
    paths = create_run_paths(config, config["model"]["name"], seed, run_name)
    save_json(paths["config_path"], config)
    metadata = build_run_metadata(config, args.config, device, paths, sys.argv, debug)
    metadata.update({
        "parameters": count_parameters(model),
        "vocab_size": len(data.vocab),
        "samples": {split: len(dataset) for split, dataset in data.datasets.items()},
        "deterministic": args.deterministic,
    })
    save_json(paths["metadata_path"], metadata)
    save_pip_freeze(paths["run_dir"] / "pip_freeze.txt")

    print(f"Run {paths['run_id']} | model {config['model']['name']} | device {device} | "
          f"train {len(data.datasets['train'])} / val {len(data.datasets['val'])} | "
          f"trainable params {metadata['parameters']['trainable']:,}"
          + (" | DEBUG RUN" if debug else ""), flush=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = perf_counter()
    history = fit(
        model, data.loaders["train"], data.loaders["val"], optimizer, device,
        epochs=int(training["epochs"]),
        patience=int(training["early_stopping_patience"]),
        score_fn=macro_f1,
        checkpoint_path=paths["checkpoint_path"],
        label_mapping=config["labels"],
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = perf_counter() - start

    compute = {
        "device": str(device),
        "training_seconds": round(seconds, 2),
        "seconds_per_epoch": round(seconds / len(history), 2),
        "peak_vram_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 1024**2, 1)
        if device.type == "cuda" else None,
        "peak_vram_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 1024**2, 1)
        if device.type == "cuda" else None,
    }
    best = max(history, key=lambda row: row["val_macro_f1"])  # first maximum = saved checkpoint
    save_training_curves(history, paths["figures_dir"] / "training_curves.png",
                         f"{config['model']['name']} (seed {seed})")

    # Reload the selected checkpoint and confirm it reproduces its validation score.
    checkpoint = load_checkpoint_into(model, paths["checkpoint_path"], config)
    model.to(device)
    evaluation = evaluate_model(model, data.loaders["val"], device, config, "val", paths["run_dir"],
                                extra={"run_id": paths["run_id"], "checkpoint_epoch": checkpoint["epoch"],
                                       "debug_run": debug, "device": str(device)})
    difference = abs(evaluation["macro_f1"] - checkpoint["val_macro_f1"])
    reload_check = {"checkpoint_val_macro_f1": checkpoint["val_macro_f1"],
                    "reloaded_val_macro_f1": evaluation["macro_f1"],
                    "abs_difference": difference, "tolerance": RELOAD_TOLERANCE,
                    "passed": difference <= RELOAD_TOLERANCE}

    save_json(paths["training_path"], {
        "run_id": paths["run_id"],
        "model": config["model"]["name"],
        "seed": seed,
        "debug_run": debug,
        "epochs_requested": int(training["epochs"]),
        "epochs_completed": len(history),
        "early_stopped": len(history) < int(training["epochs"]),
        "best_epoch": best["epoch"],
        "best_val_macro_f1": best["val_macro_f1"],
        "selection_metric": "val_macro_f1",
        "history": history,
        "compute": compute,
        "parameters": metadata["parameters"],
        "checkpoint_path": relative_to_root(paths["checkpoint_path"]),
        "reload_check": reload_check,
    })

    if not all(math.isfinite(row["train_loss"]) for row in history):
        print("WARNING: non-finite training loss recorded.", flush=True)
    if not reload_check["passed"]:
        print(f"WARNING: reloaded checkpoint differs by {difference:.2e} Macro-F1.", flush=True)
    print(f"Best epoch {best['epoch']}/{len(history)} | val macro-F1 {evaluation['macro_f1']:.4f} | "
          f"val accuracy {evaluation['accuracy']:.4f} | {compute['training_seconds']:.0f}s"
          + (f" | peak VRAM {compute['peak_vram_allocated_mib']:.0f} MiB" if device.type == "cuda" else ""))
    print(f"Run directory: {paths['run_dir']}")
    print(f"Run id: {paths['run_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
