#run.py

import torch
from torch import nn

from src.models.factory import build_model
from src.training.trainer import run_epoch
from src.training.checkpoint import BestCheckpoint
from src.training.runtime import configure_runtime
from src.training.resume import restore_last, save_last, write_json

import json
from argparse import ArgumentParser
from time import perf_counter
from pathlib import Path

from experiments.context import (
    prepare_experiment_context,
    prepare_data_loaders,
    ExperimentContext,
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
    runtime = configure_runtime(config, context.device)
    if runtime["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
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
    resume=False,
):
    epochs = context.config["training"]["epochs"]
    num_classes = context.config["data"]["num_classes"]
    # Stop after this many epochs without a new best validation Macro-F1.
    # Derived from best_epoch, so resume needs no extra saved state.
    patience = context.config["training"].get("early_stopping_patience")
    if patience is not None and patience < 1:
        raise ValueError("early_stopping_patience must be a positive integer")

    checkpoint = BestCheckpoint(checkpoint_path)
    run_dir = Path(checkpoint_path).parent
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime = configure_runtime(context.config, context.device)
    scaler = torch.amp.GradScaler("cuda", enabled=runtime["amp"])
    history = []
    previous_seconds = 0.0
    if resume:
        saved = restore_last(run_dir / "last.pt", model=model, optimizer=optimizer,
                             scaler=scaler, checkpoint=checkpoint, loader=train_loader)
        history = saved["history"]
        previous_seconds = saved["training_seconds"]
        print(f"Resuming after epoch {saved['epoch']}", flush=True)
    start = perf_counter()

    def patience_exhausted():
        return (patience is not None and checkpoint.best_epoch is not None
                and len(history) - checkpoint.best_epoch >= patience)

    for epoch in range(len(history) + 1, epochs + 1):
        if patience_exhausted():
            print(f"Early stopping: no validation Macro F1 improvement for {patience} epochs "
                  f"after epoch {checkpoint.best_epoch}", flush=True)
            break
        epoch_start = perf_counter()
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=context.device,
            optimizer=optimizer,
            num_classes=num_classes,
            scaler=scaler,
            channels_last=runtime["channels_last"],
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=context.device,
            num_classes=num_classes,
            channels_last=runtime["channels_last"],
        )

        checkpoint_start = perf_counter()
        # Keep the preceding best until last.pt commits this epoch. This lets
        # resume roll back an interrupted best/last checkpoint pair by rename.
        best_path = Path(checkpoint_path)
        if (context.config["training"].get("save_last", False)
                and val_metrics["macro_f1"] > checkpoint.best_score and best_path.exists()):
            best_path.replace(run_dir / "best_model.previous.pt")
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
        history[-1]["best_checkpoint_seconds"] = perf_counter() - checkpoint_start
        write_json(run_dir / "history.json", history)
        if context.config["training"].get("save_last", False):
            save_start = perf_counter()
            save_last(run_dir / "last.pt", model=model, optimizer=optimizer,
                      scaler=scaler, history=history, checkpoint=checkpoint,
                      loader=train_loader, training_seconds=previous_seconds + perf_counter() - start)
            history[-1]["last_checkpoint_seconds"] = perf_counter() - save_start
        history[-1]["epoch_seconds"] = perf_counter() - epoch_start
        write_json(run_dir / "history.json", history)
        write_json(run_dir / "progress.json", {
            "epochs_completed": len(history), "best_epoch": checkpoint.best_epoch,
            "best_val_macro_f1": checkpoint.best_score,
            "training_seconds": previous_seconds + perf_counter() - start,
            "stopped_early": False,
        })

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss={train_metrics['loss']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} | "
            f"val Macro F1={val_metrics['macro_f1']:.4f} | "
            f"epoch={history[-1]['epoch_seconds']:.1f}s | "
            f"train={train_metrics['timing']['samples_per_second']:.1f} img/s | "
            f"loader wait={train_metrics['timing']['data_wait_seconds']:.1f}s",
            flush=True,
        )

    # Also finalize progress when resuming a completed last.pt after a disconnect
    # between checkpoint serialization and the JSON/summary writes.
    write_json(run_dir / "progress.json", {
        "epochs_completed": len(history), "best_epoch": checkpoint.best_epoch,
        "best_val_macro_f1": checkpoint.best_score,
        "training_seconds": previous_seconds + perf_counter() - start,
        "stopped_early": len(history) < epochs,
    })
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
    parser.add_argument(
        "--patience",
        type=int,
        help="Early stopping patience in epochs, on validation Macro F1",
    )
    parser.add_argument("--resume", type=Path, help="Existing run directory with last.pt")
    args = parser.parse_args()

    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.patience is not None and args.patience < 1:
        parser.error("--patience must be at least 1")

    context = prepare_experiment_context(
        args.config,
        args.experiment_config,
    )

    if args.epochs is not None:
        context.config["training"]["epochs"] = args.epochs
    if args.patience is not None:
        context.config["training"]["early_stopping_patience"] = args.patience

    if context.config["training"]["epochs"] < 1:
        parser.error("Configured epochs must be at least 1")

    if args.resume:
        run_dir = args.resume.resolve()
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if config != context.config:
            parser.error("Resume configuration must match the saved run exactly")
        if not (run_dir / "last.pt").is_file() or not any(
            (run_dir / name).is_file() for name in ("best_model.pt", "best_model.previous.pt")
        ):
            parser.error("Resume requires last.pt and a saved best checkpoint")
        run_metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        current = collect_run_metadata(context, run_dir)
        if current["manifest_hashes"] != run_metadata["manifest_hashes"]:
            parser.error("Resume manifests differ from the original run")
        # Load trained weights below; don't download pretrained weights again.
        from copy import deepcopy
        model_context = ExperimentContext(context.project_root, deepcopy(config), context.device)
        model_context.config["model"]["pretrained"] = False
    else:
        model_context = context

    model, criterion, optimizer = prepare_training_components(model_context)
    train_loader, val_loader, _ = prepare_data_loaders(context)

    if not args.resume:
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

    history = train_epochs(
        context=context,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_path=run_dir / "best_model.pt",
        run_metadata=run_metadata,
        resume=bool(args.resume),
    )

    if context.device.type == "cuda":
        torch.cuda.synchronize(context.device)

    progress = json.loads((run_dir / "progress.json").read_text())
    training_seconds = progress["training_seconds"]

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
        "epochs_requested": context.config["training"]["epochs"],
        "early_stopping_patience": context.config["training"].get("early_stopping_patience"),
        "stopped_early": progress["stopped_early"],
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
