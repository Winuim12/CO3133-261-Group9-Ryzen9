#run.py

import torch
from torch import nn

from src.models.factory import build_model
from src.training.trainer import run_epoch
from src.training.checkpoint import BestCheckpoint

import json
from argparse import ArgumentParser
from time import perf_counter

from experiments.context import (
    prepare_experiment_context,
    prepare_data_loaders,
)
from src.utils.run_info import (
    create_run_directory,
    collect_run_metadata,
)


def prepare_training_components(context):
    config = context.config
    training_config = config["training"]

    if training_config["optimizer"].lower() != "adam":
        raise ValueError("Currently supported optimizer: adam")

    model = build_model(config["model"]).to(context.device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )

    return model, criterion, optimizer

def train_epochs(
    *,
    context,
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    checkpoint_path,
    run_metadata,
):
    epochs = context.config["training"]["epochs"]
    num_classes = context.config["data"]["num_classes"]

    checkpoint = BestCheckpoint(checkpoint_path)
    history = []

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=context.device,
            optimizer=optimizer,
            num_classes=num_classes,
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=context.device,
            num_classes=num_classes,
        )

        checkpoint.update(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            val_macro_f1=val_metrics["macro_f1"],
            run_metadata=run_metadata,
        )

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        })

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss={train_metrics['loss']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} | "
            f"val Macro-F1={val_metrics['macro_f1']:.4f}",
            flush=True,
        )

    return history

def main():
    parser = ArgumentParser(description="Train a 30VNFoods classifiers")
    parser.add_argument(
        "--config",
        required=True,
        help="Model configuration path, relative to the project root",
    )
    parser.add_argument(
        "--experiment-config",
        help="Optional experiment overrider configuration",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override the configured epoch count",
    )
    parser.add_argument(
        "--code_commit",
        help="Git commit used to build the training image"
    )
    args = parser.parse_args()

    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be at least 1")

    context = prepare_experiment_context(
        args.config,
        args.experiment_config,
    )

    if args.epochs is not None:
        context.config["training"]["epochs"] = args.epochs

    if context.config["training"]["epochs"] < 1:
        parser.error("Configured epochs must be at least 1")

    model, criterion, optimizer = prepare_training_components(context)
    train_loader, val_loader, _ = prepare_data_loaders(context)

    run_dir = create_run_directory(
        context.project_root / "results" / "runs",
        context.config,
    )

    run_metadata = collect_run_metadata(
        context,
        run_dir,
        code_commit=args.code_commit,
    )

    run_metadata["parameter_count"] = {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    }

    run_metadata["split_sizes"] = {
        "train": len(train_loader.dataset),
        "val": len(val_loader.dataset),
    }

    (run_dir / "metadata.json").write_text(
        json.dumps(run_metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"Run directory: {run_dir}", flush=True)
    print(f"Device: {context.device}", flush=True)

    if context.device.type == "cuda":
        torch.cuda.synchronize(context.device)

    training_start = perf_counter()

    history = train_epochs(
        context=context,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_path=run_dir / "best_model.pt",
        run_metadata=run_metadata,
    )

    if context.device.type == "cuda":
        torch.cuda.synchronize(context.device)

    training_seconds = perf_counter() - training_start

    best_record = max(
        history,
        key=lambda record: record["val"]["macro_f1"],
    )

    summary = {
        "training_seconds": training_seconds,
        "timing_scope": (
            "Training, validation, data loading and checkpoint saving "
            "inside the epoch loop; excludes initial setup"
        ),
        "epochs_completed": len(history),
        "best_epoch": best_record["epoch"],
        "best_val_macro_f1": best_record["val"]["macro_f1"],
    }

    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"Training finished. Outputs: {run_dir}", flush=True)

if __name__ == "__main__":
    main()