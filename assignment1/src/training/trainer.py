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
):
    model.to(device)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }

    best_validation_loss = float("inf")

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

        if (checkpoint_path is not None and current_validation_loss < best_validation_loss):
            best_validation_loss = current_validation_loss

            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_loss=current_validation_loss,
            )

    return history