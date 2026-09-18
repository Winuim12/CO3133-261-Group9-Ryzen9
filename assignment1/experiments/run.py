#run.py

from argparse import ArgumentParser
from pathlib import Path

import torch

from torch import nn

from src.models.factory import create_model
from src.training.optimizer import create_optimizer
from src.training.trainer import train_model
from src.data.dataloader import create_dataloaders
from src.data.pipeline import create_normalized_datasets
from src.utils.config import load_config
from src.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def run_experiment(
    model_name,
    model_parameters,
    training_config,
    train_loader,
    validation_loader,
    checkpoint_path,
    device,
):
    model = create_model(
        model_name=model_name,
        model_parameters=model_parameters,
    )

    loss_function = nn.CrossEntropyLoss()

    optimizer = create_optimizer(
        model=model,
        optimizer_name=training_config["optimizer"],
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )

    history = train_model(
        model=model, 
        train_loader=train_loader,
        validation_loader=validation_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        device=device,
        epochs=training_config["epochs"],
        checkpoint_path=checkpoint_path
    )

    return model, history

def run_configured_experiment(
    shared_config,
    model_config,
    train_dataset,
    validation_dataset,
    test_dataset,
    checkpoint_path,
    device,
):

    data_config = shared_config["data"]
    training_config = shared_config["training"]
    model_settings = model_config["model"]

    train_loader, validation_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=validation_dataset,
        test_dataset=test_dataset,
        batch_size=data_config["batch_size"],
        num_workers=data_config["num_workers"],
    )

    model, history = run_experiment(
        model_name=model_settings["name"],
        model_parameters=model_settings["parameters"],
        training_config=training_config,
        train_loader=train_loader,
        validation_loader=validation_loader,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    return model, history, test_loader

def run_model_from_config(model_name, epochs_override=None):
    normalized_name = model_name.lower()

    shared_config = load_config(PROJECT_ROOT / "config" / "config.yaml")
    model_config = load_config(PROJECT_ROOT / "config" / f"{normalized_name}.yaml")

    if epochs_override is not None: 
        shared_config["training"]["epochs"] = epochs_override

    seed = shared_config["seed"]
    set_seed(seed)

    data_config = shared_config["data"]
    data_directory = (PROJECT_ROOT / "data" / "raw" / "fashion_mnist")

    print("Preparing Fashion-MNIST datasets...", flush=True)

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        mean,
        std,
    ) = create_normalized_datasets(
        data_dir = data_directory,
        validation_size = data_config["validation_size"],
        seed = seed, 
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = (PROJECT_ROOT / "results" / "checkpoints" / normalized_name / f"{normalized_name}_best.pt")

    print(
        f"Training {model_config['model']['name']} on {device} "
        f"for {shared_config['training']['epochs']} epochs...",
        flush=True,
    )

    model, history, test_loader = run_configured_experiment(
        shared_config=shared_config,
        model_config=model_config,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    print(f"Model: {model.__class__.__name__}")
    print(f"Device: {device}")
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(validation_dataset):,}")
    print(f"Test samples: {len(test_loader.dataset):,}")
    print(f"Training mean: {mean: .6f}")
    print(f"Training standard deviation: {std: .6f}")
    print(f"Epochs: {shared_config['training']['epochs']}")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    print("Final training accuracy: "f"{history['train_accuracy'][-1]:.2%}")
    print("Final validation loss: "f"{history['validation_loss'][-1]:.4f}")
    print("Final validation accuracy: "f"{history['validation_accuracy'][-1]:.2%}")
    print(f"Best checkpoint: {checkpoint_path}")

    return model, history, test_loader
    
def parse_arguments():
    parser = ArgumentParser(description="Train a Fashion-MNIST model.")

    parser.add_argument(
        "--model",
        required=True,
        help="Model configuration name, such as linear, mlp, ...",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Overide the epoch count from config/config.yaml"
    )

    return parser.parse_args()

def main():
    arguments = parse_arguments()

    run_model_from_config(
        model_name=arguments.model,
        epochs_override=arguments.epochs,
    )

if __name__ == "__main__":
    main()