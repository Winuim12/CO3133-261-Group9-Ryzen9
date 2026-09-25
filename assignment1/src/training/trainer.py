# trainer.py

from src.training.checkpoint import save_checkpoint
from src.training.train import train_one_epoch
from src.training.validate import validate_one_epoch

def train_model(
    model,
    train_loader,
    validation_loader,
    loss_function,
    optimizer,
    device,
    epochs,
    checkpoint_path=None,
    patience=None,
    min_delta=0.0001,
):
    if patience is not None and patience < 1:
        raise ValueError("patience must be a positive integer")
    if min_delta < 0:
        raise ValueError("min_delta must be non-negative")

    model.to(device)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }

    best_validation_loss = float("inf")
    best_epoch = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
        )

        validation_metrics = validate_one_epoch(
            model=model,
            dataloader=validation_loader,
            loss_function=loss_function,
            device=device,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["validation_loss"].append(validation_metrics["loss"])
        history["validation_accuracy"].append(validation_metrics["accuracy"])


        current_validation_loss = validation_metrics["loss"]

        meaningful_improvement = (
            current_validation_loss < best_validation_loss
            and best_validation_loss - current_validation_loss >= min_delta
        )

        if current_validation_loss < best_validation_loss:
            best_validation_loss = current_validation_loss
            best_epoch = epoch

            if checkpoint_path is not None:
                save_checkpoint(
                    path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    validation_loss=current_validation_loss,
                )

        if patience is not None:
            if meaningful_improvement:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        print(
            f"Epoch {epoch}/{epochs}: "
            f"train loss={train_metrics['loss']:.4f}, "
            f"train accuracy={train_metrics['accuracy']:.2%}, "
            f"validation loss={validation_metrics['loss']:.4f}, "
            f"validation accuracy={validation_metrics['accuracy']:.2%}, "
            f"best epoch={best_epoch}"
        )

        if patience is not None and epochs_without_improvement >= patience:
            print(f"Early stopping triggered after epoch {epoch}")
            break

    return history