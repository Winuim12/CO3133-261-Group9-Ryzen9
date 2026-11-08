"""Atomic epoch checkpoints, including optimizer, scaler and parent RNG state."""

import json
import random
from pathlib import Path

import numpy as np
import torch


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_last(path, *, model, optimizer, scaler, history, checkpoint, loader, training_seconds):
    state = np.random.get_state()
    generator = getattr(loader, "generator", None)
    saved = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "history": history,
        "epoch": history[-1]["epoch"],
        "best_score": checkpoint.best_score,
        "best_epoch": checkpoint.best_epoch,
        "training_seconds": training_seconds,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python_rng": random.getstate(),
        "numpy_rng": [state[0], state[1].tolist(), state[2], state[3], state[4]],
        "loader_rng": generator.get_state() if generator is not None else None,
    }
    path = Path(path)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(saved, temporary)
    temporary.replace(path)


def restore_last(path, *, model, optimizer, scaler, checkpoint, loader):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    best_path = Path(path).parent / "best_model.pt"
    best = torch.load(best_path, map_location="cpu", weights_only=True) if best_path.exists() else None
    if best is None or best["epoch"] > saved["epoch"]:
        previous = best_path.with_name("best_model.previous.pt")
        if not previous.is_file():
            raise RuntimeError("Interrupted checkpoint pair has no previous best to restore")
        previous.replace(best_path)
    # Prefer completed timing fields written just after serializing last.pt.
    history_path = Path(path).parent / "history.json"
    if history_path.exists():
        disk_history = json.loads(history_path.read_text(encoding="utf-8"))
        if len(disk_history) == len(saved["history"]):
            saved["history"] = disk_history
    progress_path = Path(path).parent / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress["epochs_completed"] == saved["epoch"]:
            saved["training_seconds"] = progress["training_seconds"]
    model.load_state_dict(saved["model_state_dict"])
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    scaler.load_state_dict(saved["scaler_state_dict"])
    checkpoint.best_score = saved["best_score"]
    checkpoint.best_epoch = saved["best_epoch"]
    torch.set_rng_state(saved["torch_rng"])
    if saved["cuda_rng"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    random.setstate(saved["python_rng"])
    state = saved["numpy_rng"]
    np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
    generator = getattr(loader, "generator", None)
    if generator is not None and saved["loader_rng"] is not None:
        generator.set_state(saved["loader_rng"])
    return saved
