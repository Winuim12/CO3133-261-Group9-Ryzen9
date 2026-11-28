#engine.py
"""Shared training and validation helpers."""

import torch
import math
from torch.nn import functional as F
from pathlib import Path

def _move_batch(batch, device):
    #Packed GRU sequences use CPU lengths. Image names are metadata
    return{
        name: value.to(device)
        if isinstance(value, torch.Tensor) and name != "text_lengths"
        else value
        for name, value in batch.items()
    }

def train_one_epoch(
    model, loader, optimizer, device, *, class_weights=None, label_smoothing=0.0,
) -> float:
    model.train()
    loss_sum = torch.zeros((), device=device)
    weight_sum = torch.zeros((), device=device)
    sample_count = 0

    for batch in loader:
        batch = _move_batch(batch, device)
        labels = batch["label"]

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = F.cross_entropy(
            logits, labels, weight=class_weights, label_smoothing=label_smoothing,
        )
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        batch_weight = (
            batch_size if class_weights is None else class_weights[labels].sum()
        )
        loss_sum += loss.detach() * batch_weight
        weight_sum += batch_weight
        sample_count += batch_size

    if sample_count == 0:
        raise ValueError("Training loader produced no samples.")

    return (loss_sum / weight_sum).item()

@torch.no_grad()
def validate_one_epoch(model, loader, device):
    model.eval()
    loss_sum = torch.zeros((), device=device)
    sample_count = 0
    targets = []
    predictions = []

    for batch in loader:
        batch = _move_batch(batch, device)
        labels = batch["label"]
        logits = model(batch)

        loss_sum += F.cross_entropy(logits, labels, reduction="sum")
        sample_count += labels.size(0)
        targets.append(labels)
        predictions.append(logits.argmax(dim=1))

    if sample_count == 0:
        raise ValueError("Validation loader produced no samples.")

    return {
        "loss": loss_sum.item() / sample_count,
        "targets": torch.cat(targets).cpu(),
        "predictions": torch.cat(predictions).cpu(),
    }

def fit(
    model, train_loader, val_loader, optimizer, device, *,
    epochs, patience, score_fn, checkpoint_path, label_mapping, class_weights=None,
    label_smoothing=0.0,
):

    if epochs < 1 or patience < 1:
        raise ValueError("epochs and patience must both be positive.")

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(".pt.tmp")

    best_score = float("-inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            class_weights=class_weights, label_smoothing=label_smoothing,
        )
        val = validate_one_epoch(model, val_loader, device)
        score = float(score_fn(val["targets"], val["predictions"]))
        val_accuracy = (
            (val["predictions"] == val["targets"]).sum().item()
            / val["targets"].numel()
        )

        if not math.isfinite(score):
            raise ValueError(f"Validation Macro-F1 is not finite at epoch {epoch}.")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "val_macro_f1": score,
            "val_accuracy": val_accuracy,
        })

        status = " | new best" if score > best_score else ""
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"train loss {train_loss:.4f} | val loss {val['loss']:.4f} | "
            f"val accuracy {val_accuracy:.2%} | val Macro-F1 {score:.4f}{status}",
            flush=True,
        )

        if score > best_score:
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_macro_f1": score,
                "labels": dict(label_mapping),
            }, temporary_path)
            temporary_path.replace(checkpoint_path)
            best_score = score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    return history
