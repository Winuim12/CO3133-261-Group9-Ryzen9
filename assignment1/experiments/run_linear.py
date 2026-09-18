#run_linear.py

from argparse import ArgumentParser
from pathlib import Path

import torch

from experiments.run import run_configured_experiment
from src.data.pipeline import create_normalized_datasets
from src.utils.config import load_config
from src.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).parents[1]

def parse_arguments():
    parser = ArgumentParser(description="Train the linear Fashion-MNIST classifier.")

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Overide the epoch count from config/config.yaml"
    )

    return parser.parse_args()

def main():
    arguments = parse_arguments()

    shared_config = load_config(PROJECT_ROOT / "config" / "config.yaml")
    model_config = load_config(PROJECT_ROOT / "config" / "linear.yaml")

    if arguments.epochs is not None: 
        shared_config["training"]["epochs"] = arguments.epochs

    seed = shared_config["seed"]
    set_seed(seed)

    data_config = shared_config["data"]
    data_directory = (PROJECT_ROOT / "data" / "raw" / "fashion_mnist")

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

    checkpoint_path = (PROJECT_ROOT / "results" / "checkpoints" / "linear" / "linear_best.pt")

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

if __name__ == "__main__":
    main()