#test_results.py
"""
RUN: python -m pytest tests/test_evaluation/test_results.py -v
"""

import json

import numpy as np
import torch

from src.evaluation.results import save_evaluation_results, save_training_results

def test_save_evaluation_results_writes_serializable_summary(tmp_path):
    output_path = tmp_path / "metrics" / "linear" / "evaluation.json"
    checkpoint_path = tmp_path / "linear_best.pt"

    test_results = {
        "loss": 0.5,
        "accuracy": 0.75,
        "macro_f1": 0.7,
        "trainable_parameters": 7_850,
        "inference_time_seconds": 0.1,
        "inference_time_per_sample_seconds": 0.025,
        "confusion_matrix": np.array(
            [
                [2, 0],
                [1, 1],
            ]
        ),
        "predictions": torch.tensor([0, 0, 0, 1]),
        "targets": torch.tensor([0, 1, 0, 1]) 
    }

    save_evaluation_results(
        path=output_path,
        model_name="linear",
        checkpoint_path=checkpoint_path,
        test_results=test_results,
    )

    saved_results = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved_results == {
        "model_name": "linear",
        "checkpoint_path": str(checkpoint_path),
        "test_samples": 4,
        "loss": 0.5,
        "accuracy": 0.75,
        "macro_f1": 0.7,
        "trainable_parameters": 7_850,
        "inference_time_seconds": 0.1,
        "inference_time_per_sample_seconds": 0.025,
        "confusion_matrix": [
            [2, 0],
            [1, 1],
        ],
    }

def test_save_training_results_writes_configuration_and_history(tmp_path):
    output_path = tmp_path / "metrics" / "linear" / "training.json"
    checkpoint_path = tmp_path / "linear_best.pt"

    shared_config = {
        "seed": 42,
        "data": {
            "batch_size": 128,
            "validation_size": 0.1,
        },
        "training": {
            "epochs": 2,
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
        },
        "evaluation": {
            "chekcpoint_rule": "best_val_loss",
        }
    }

    model_config = {
        "model": {
            "name": "linear",
            "parameters": {
                "input_dim": 784,
                "num_classes": 10,
            }
        }
    }

    history = {
        "train_loss": [0.8, 0.6],
        "train_accuracy": [0.7, 0.8],
        "validation_loss": [0.7, 0.5],
        "validation_accuracy": [0.75, 0.85],
    }

    save_training_results(
        path=output_path,
        model_name="linear",
        shared_config=shared_config,
        model_config=model_config,
        checkpoint_path=checkpoint_path,
        history=history,
        training_time_seconds=12.5,
    )

    saved_results = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved_results == {
        "model_name": "linear",
        "seed": 42,
        "data_config": shared_config["data"],
        "training_config": shared_config["training"],
        "evaluation_config": shared_config["evaluation"],
        "model_parameters": model_config["model"]["parameters"],
        "checkpoint_path": str(checkpoint_path),
        "training_time_seconds": 12.5,
        "history": history,
        "requested_epochs": 2,
        "completed_epochs": 2,
        "best_epoch": 2,
        "best_validation_loss": 0.5,
        "stopped_early": False,
    }