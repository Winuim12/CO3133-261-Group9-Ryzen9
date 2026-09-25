#test_mlflow_logger.py
"""
RUN: python -m pytest tests/test_tracking/test_mlflow_logger.py -v
"""

import sys
import pytest

from unittest.mock import Mock, call

from src.tracking.mlflow_logger import (
    require_mlflow,
    log_epoch_metrics,
    log_run_parameters,
)


def test_require_mlflow_explains_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", None)

    with pytest.raises(RuntimeError, match="MLflow.*install"):
        require_mlflow()

def test_log_epoch_metrics_uses_epoch_numbers():
    fake_mlflow = Mock()
    history = {
        "train_loss": [0.8, 0.6],
        "train_accuracy": [0.7, 0.8],
        "validation_loss": [0.7, 0.5],
        "validation_accuracy": [0.75, 0.85],
    }

    log_epoch_metrics(fake_mlflow, history)

    assert fake_mlflow.log_metrics.call_args_list == [
        call(
            {
                "train_loss": 0.8,
                "train_accuracy": 0.7,
                "validation_loss": 0.7,
                "validation_accuracy": 0.75,
            },
            step=1,
        ),
        call(
            {
                "train_loss": 0.6,
                "train_accuracy": 0.8,
                "validation_loss": 0.5,
                "validation_accuracy": 0.85,
            },
            step=2,
        ),
    ]

def test_log_run_parameters_captures_configuration():
    fake_mlflow = Mock()
    summary = {
        "model_name": "mlp",
        "seed": 42,
        "data_config": {"batch_size": 128},
        "training_config": {"epochs": 20, "patience": 5},
        "model_parameters": {"hidden_dims": [256, 128]},
    }

    log_run_parameters(fake_mlflow, "local-run-123", summary)

    fake_mlflow.log_params.assert_called_once_with({
        "local_run_id": "local-run-123",
        "model_name": "mlp",
        "seed": 42,
        "data.batch_size": 128,
        "training.epochs": 20,
        "training.patience": 5,
        "model.hidden_dims": [256, 128],
    })

def test_log_evaluation_metrics_records_test_results():
    from unittest.mock import Mock
    from src.tracking import mlflow_logger

    client = Mock()
    results = {
        "loss": 0.5,
        "accuracy": 0.82,
        "macro_f1": 0.81,
        "inference_time_seconds": 0.02,
    }

    mlflow_logger.log_evaluation_metrics(client, results)

    client.log_metrics.assert_called_once_with({
        "test_loss": 0.5,
        "test_accuracy": 0.82,
        "test_macro_f1": 0.81,
        "inference_time_seconds": 0.02,
    })

def test_log_training_summary_records_best_results():
    from unittest.mock import Mock
    from src.tracking import mlflow_logger

    client = Mock()
    summary = {
        "best_validation_loss": 0.41,
        "best_epoch": 3,
        "completed_epochs": 5,
        "training_time_seconds": 47.7,
    }

    mlflow_logger.log_training_summary(client, summary)

    client.log_metrics.assert_called_once_with({
        "best_validation_loss": 0.41,
        "best_epoch": 3,
        "completed_epochs": 5,
        "training_time_seconds": 47.7,
    })

@pytest.mark.parametrize("with_evaluation", [False, True])
def test_log_saved_run_records_training_results(tmp_path, with_evaluation):
    import json
    from unittest.mock import MagicMock
    from src.tracking import mlflow_logger

    run_directory = tmp_path / "linear-run-1"
    run_directory.mkdir()
    summary = {
        "model_name": "linear",
        "seed": 42,
        "data_config": {"batch_size": 128},
        "training_config": {"epochs": 1},
        "evluation_config": {"checkpoint_rule": "best_val_loss"},
        "model_parameters": {"input_dim": 784},
        "checkpoint_path": str(run_directory / "best_model.pt"),
        "training_time_seconds": 2.0,
        "history": {
            "train_loss": [0.5],
            "train_accuracy": [0.8],
            "validation_loss": [0.4],
            "validation_accuracy": [0.85],
        },
        "requested_epochs": 1,
        "completed_epochs": 1,
        "best_epoch": 1,
        "best_validation_loss": 0.4,
        "stopped_early": False,
    }
    (run_directory / "training.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )

    if with_evaluation:
        evaluation = {
            "model_name": "linear",
            "checkpoint_path": str(run_directory / "best_model.pt"),
            "test_samples": 2,
            "loss": 0.3,
            "accuracy": 0.88,
            "macro_f1": 0.87,
            "trainable_parameters": 7850,
            "inference_time_seconds": 0.02,
            "inference_time_per_sample_seconds": 0.01,
            "confusion_matrix": [[1, 0], [0, 1]],
        }
        (run_directory / "evaluation.json").write_text(
            json.dumps(evaluation), encoding="utf-8"
        )

    client = MagicMock()

    mlflow_logger.log_saved_run(client, run_directory)

    client.start_run.assert_called_once_with(run_name="linear-run-1")
    client.log_metrics.assert_any_call({
        "best_validation_loss": 0.4,
        "best_epoch": 1,
        "completed_epochs": 1,
        "training_time_seconds": 2.0,
    })

    client.log_artifacts.assert_called_once_with(str(run_directory))

    if with_evaluation:
        client.log_metrics.assert_any_call({
            "test_loss": 0.3,
            "test_accuracy": 0.88,
            "test_macro_f1": 0.87,
            "inference_time_seconds": 0.02,
        })
