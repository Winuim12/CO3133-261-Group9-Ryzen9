from pathlib import Path

import torch

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    validation_loss,
):
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "validation_loss": validation_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(
    path,
    model,
    device,
    optimizer=None,
):
    checkpoint_path = Path(path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {
        "epoch": checkpoint["epoch"],
        "validation_loss": checkpoint["validation_loss"],
    }