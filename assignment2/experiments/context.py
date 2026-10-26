#context.py

from dataclasses import dataclass
from pathlib import Path

import torch

from src.utils.config import load_experiment_config
from src.utils.seed import set_seed
from src.data.dataset import get_food_datasets
from src.data.dataloader import create_dataloaders
from src.data.transforms import (
    get_train_transforms,
    get_val_transforms,
    get_test_transforms,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

@dataclass
class ExperimentContext:
    project_root: Path
    config: dict
    device: torch.device

def prepare_experiment_context(
    model_config_path,
    experiment_config_path=None,
    project_root=PROJECT_ROOT,
):
    project_root = Path(project_root).resolve()

    shared_path = project_root / "config" / "config.yaml"
    model_path = project_root / model_config_path

    experiment_path = (
        project_root / experiment_config_path
        if experiment_config_path is not None
        else None
    )

    config = load_experiment_config(
        shared_path,
        model_path,
        experiment_path,
    )

    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return ExperimentContext(
        project_root=project_root,
        config=config,
        device=device,
    )

def prepare_data_loaders(context):
    data_config = context.config["data"]
    image_size = data_config["image_size"]

    train_dataset, val_dataset, test_dataset = get_food_datasets(
        manifests_dir=context.project_root / data_config["manifests_dir"],
        raw_dir=context.project_root / data_config["raw_dir"],
        train_transforms=get_train_transforms(image_size),
        val_transforms=get_val_transforms(image_size),
        test_transforms=get_test_transforms(image_size),
    )

    return create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=data_config["batch_size"],
        num_workers=data_config["num_workers"],
        seed=context.config["seed"],
    )