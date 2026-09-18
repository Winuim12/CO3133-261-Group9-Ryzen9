#test_run.py

"""
RUN: python -m pytest tests/test_experiments/test_run.py -v
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.run import run_experiment, run_configured_experiment
from src.models.linear import LinearClassifier

def test_run_experiment_trains_selected_model(tmp_path):
    torch.manual_seed(42)

    images = torch.randn(8, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])

    dataloader = DataLoader(
        TensorDataset(images, labels),
        batch_size=4,
        shuffle=False,
    )

    training_config = {
        "epochs": 1,
        "optimizer": "adamw",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
    }

    model, history = run_experiment(
        model_name="linear",
        model_parameters={},
        training_config=training_config,
        train_loader=dataloader,
        validation_loader=dataloader,
        checkpoint_path=tmp_path / "best_model.pt",
        device=torch.device("cpu"),
    )

    assert isinstance(model, LinearClassifier)
    assert len(history["train_loss"]) == 1
    assert len(history["validation_loss"]) == 1
    assert (tmp_path / "best_model.pt").exists()

def test_run_configured_experiment_uses_configuration(tmp_path):
    torch.manual_seed(42)
    
    images = torch.randn(8, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])

    dataset = TensorDataset(images, labels)

    shared_config = {
        "data": {
            "batch_size": 4,
            "num_workers": 0,
        },
        "training": {
            "epochs": 1,
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
        },
    }

    model_config = {
        "model": {
            "name": "linear",
            "parameters": {
                "input_dim": 784,
                "num_classes": 10,
            },
        },
    }

    model, history, test_loader = run_configured_experiment(
        shared_config=shared_config,
        model_config=model_config,
        train_dataset=dataset,
        validation_dataset=dataset,
        test_dataset=dataset,
        checkpoint_path=tmp_path / "best_model.pt",
        device=torch.device("cpu"),
    )

    assert isinstance(model, LinearClassifier)
    assert test_loader.batch_size == 4
    assert len(history["train_loss"]) == 1
    assert (tmp_path / "best_model.pt").exists()