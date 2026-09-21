#context.py

from dataclasses import dataclass
from typing import Any
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.dataloader import create_dataloaders
from src.data.pipeline import create_normalized_datasets
from src.utils.config import load_config
from src.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExperimentContext:
    model_name: str
    shared_config: dict[str, Any]
    model_config: dict[str, Any]

    train_dataset: Dataset
    validation_dataset: Dataset
    test_dataset: Dataset

    train_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader
    mean: float
    std: float
    device: torch.device

def prepare_experiment_context(model_name, epochs_override=None):
    normalized_name = model_name.lower()

    shared_config = load_config(PROJECT_ROOT / "config" / "config.yaml")
    model_config = load_config(PROJECT_ROOT / "config" / f"{normalized_name}.yaml")

    if epochs_override is not None: 
        shared_config["training"]["epochs"] = epochs_override

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

    train_loader, validation_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=validation_dataset,
        test_dataset=test_dataset,
        batch_size=data_config["batch_size"],
        num_workers=data_config["num_workers"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return ExperimentContext(
        model_name=normalized_name,
        shared_config=shared_config,
        model_config=model_config,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        mean=mean,
        std=std,
        device=device,
    )