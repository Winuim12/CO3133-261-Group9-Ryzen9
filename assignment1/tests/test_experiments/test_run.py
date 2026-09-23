#test_run.py

"""
RUN: python -m pytest tests/test_experiments/test_run.py -v
"""

import pytest
from argparse import Namespace
import torch
from torch.utils.data import DataLoader, TensorDataset

import experiments.run as run_module
from experiments.context import ExperimentContext
from src.tracking.run_paths import create_run_paths

from experiments.run import (
    run_experiment,  
    evaluate_best_checkpoint, 
    parse_arguments,
    main,
)
from src.training.checkpoint import save_checkpoint
from src.models.linear import LinearClassifier

def test_parse_arguments_controls_evaluation_modes():
    training_arguments = parse_arguments(["--model", "linear"])
    train_and_evaluation_arguments = parse_arguments(["--model", "linear", "--evaluate-test"])
    evaluation_only_arguments = parse_arguments(["--model", "linear", "--evaluate-only", "--run-id", "existing-run"])

    assert training_arguments.evaluate_test is False
    assert training_arguments.evaluate_only is False

    assert train_and_evaluation_arguments.evaluate_test is True
    assert train_and_evaluation_arguments.evaluate_only is False

    assert evaluation_only_arguments.evaluate_test is False
    assert evaluation_only_arguments.evaluate_only is True

def test_main_routes_evaluation_only_without_training(monkeypatch):
    calls = []
    context = object()

    monkeypatch.setattr(
        run_module,
        "parse_arguments",
        lambda: Namespace(
            model="linear",
            epochs=None,
            run_name=None,
            run_id="existing-run",
            evaluate_test=False,
            evaluate_only=True,
        ),
    )

    def fake_prepare_experiment_context(
        model_name,
        epochs_override,
    ):
        calls.append(
            ("prepare", model_name, epochs_override)
        )
        return context

    def fake_training(context, run_name):
        calls.append(
            ("training", context, run_name)
        )

    def fake_evaluation(context, run_id):
        calls.append(
            ("evaluation", context, run_id)
        )

    monkeypatch.setattr(
        run_module,
        "prepare_experiment_context",
        fake_prepare_experiment_context,
        raising=False,
    )
    monkeypatch.setattr(
        run_module,
        "run_training",
        fake_training,
    )
    monkeypatch.setattr(
        run_module,
        "run_evaluation",
        fake_evaluation,
    )

    main()

    assert calls == [
        ("prepare", "linear", None),
        ("evaluation", context, "existing-run"),
    ]

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

    model, history, training_time_seconds = run_experiment(
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
    assert training_time_seconds > 0

def test_parse_arguments_accepts_run_name():
    arguments = parse_arguments([
        "--model",
        "linear",
        "--run-name",
        "linear-final",
    ])

    assert arguments.run_name == "linear-final"

def test_main_routes_training_with_prepared_context(monkeypatch):
    calls = []
    context = object()

    monkeypatch.setattr(
        run_module,
        "parse_arguments",
        lambda: Namespace(
            model="linear",
            epochs=3,
            run_name="linear-final",
            run_id=None,
            evaluate_test=False,
            evaluate_only=False,
        ),
    )

    def fake_prepare_experiment_context(
        model_name,
        epochs_override,
    ):
        calls.append(
            ("prepare", model_name, epochs_override)
        )
        return context

    def fake_training(context, run_name):
        calls.append(
            ("training", context, run_name)
        )

        return {"run_id": "new-training-run"}

    monkeypatch.setattr(
        run_module,
        "prepare_experiment_context",
        fake_prepare_experiment_context,
    )
    monkeypatch.setattr(
        run_module,
        "run_training",
        fake_training,
    )

    main()

    assert calls == [
        ("prepare", "linear", 3),
        ("training", context, "linear-final"),
    ]
    

def test_parse_arguments_accepts_evaluation_run_id():
    arguments= parse_arguments([
        "--model",
        "linear",
        "--evaluate-only",
        "--run-id",
        "20260920-120000-000000-smoke",
    ])

    assert arguments.evaluate_only is True
    assert arguments.run_id == "20260920-120000-000000-smoke"

def test_parse_arguments_requires_run_id_for_evaluation_only():
    with pytest.raises(SystemExit):
        parse_arguments([
            "--model",
            "linear",
            "--evaluate-only",
        ])

def test_parse_arguments_rejects_run_id_during_training():
    with pytest.raises(SystemExit):
        parse_arguments([
            "--model",
            "linear",
            "--run-id",
            "existing-run",
        ])

def test_run_training_uses_prepared_context_and_stores_artifacts(
    monkeypatch,
    tmp_path,
):
    torch.manual_seed(42)

    images = torch.randn(8, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])

    dataset = TensorDataset(images, labels)

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    context = ExperimentContext(
        model_name="linear",
        shared_config = {
            "seed": 42,
            "data": {
                "dataset": "fashion_mnist",
                "image_size": 28,
                "num_channels": 1,
                "num_classes": 10,
                "batch_size": 4,
                "num_workers": 0,
                "validation_size": 0.1,
            },
            "training": {
                "epochs": 1,
                "optimizer": "adamw",
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
            },
            "evaluation": {
                "checkpoint_rule": "best_val_loss",
            },
        },
        model_config = {
            "model": {
                "name": "linear",
                "parameters": {
                    "input_dim": 784,
                    "num_classes": 10,
                },
            },
        },
        train_dataset=dataset,
        validation_dataset=dataset,
        test_dataset=dataset,
        train_loader=dataloader,
        validation_loader=dataloader,
        test_loader=dataloader,
        mean=0.286209,
        std=0.353160,
        device=torch.device("cpu"),
    )

    monkeypatch.setattr(
        run_module,
        "PROJECT_ROOT",
        tmp_path,
    )

    result = run_module.run_training(
        context=context,
        run_name="smoke",
    )

    run_directory = result["run_directory"]

    assert isinstance(result["model"], LinearClassifier)
    assert len(result["history"]["train_loss"]) == 1
    assert result["training_time_seconds"] > 0
    assert result["run_id"].endswith("-smoke")

    assert (run_directory / "best_model.pt").exists()
    assert (run_directory / "training.json").exists()
    assert (run_directory / "figures" / "training_curves.png").exists()
    
def test_run_evaluation_uses_prepared_context_and_existing_run(
    monkeypatch,
    tmp_path,
): 
    torch.manual_seed(42)

    class FashionLikeDataset(TensorDataset):
        classes = [str(class_id) for class_id in range(10)]
    
    images = torch.randn(8, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])

    dataset = FashionLikeDataset(images, labels)

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    context = ExperimentContext(
        model_name="linear",
        shared_config={},
        model_config={
            "model": {
                "name": "linear",
                "parameters": {
                    "input_dim": 784,
                    "num_classes": 10,
                }
            },
        },
        train_dataset=dataset,
        validation_dataset=dataset,
        test_dataset=dataset,
        train_loader=dataloader,
        validation_loader=dataloader,
        test_loader=dataloader,
        mean=0.286209,
        std=0.353160,
        device=torch.device("cpu"),
    )

    monkeypatch.setattr(
        run_module,
        "PROJECT_ROOT",
        tmp_path,
    )

    run_id = "20260920-120000-000000-smoke"

    run_paths = create_run_paths(
        results_root=tmp_path / "results" / "runs",
        model_name="linear",
        timestamp=run_id,
    )

    checkpoint_model = LinearClassifier()
    optimizer = torch.optim.SGD(
        checkpoint_model.parameters(),
        lr=0.01,
    )

    save_checkpoint(
        path=run_paths["checkpoint_path"],
        model=checkpoint_model,
        optimizer=optimizer,
        epoch=1,
        validation_loss=0.5
    )

    results = run_module.run_evaluation(
        context=context,
        run_id=run_id,
    )

    assert isinstance(results["model"], LinearClassifier)
    assert results["run_id"] == run_id
    assert results["run_directory"] == run_paths["run_directory"]

    assert results["test_results"]["targets"].numel() == len(dataset)
    assert results["test_results"]["predictions"].numel() == len(dataset)

    assert run_paths["evaluation_results_path"].exists()
    assert run_paths["confusion_matrix_path"].exists()
    assert run_paths["prediction_examples_path"].exists()

def test_main_reuses_context_for_training_and_evaluation(monkeypatch):
    calls = []
    context = object()

    monkeypatch.setattr(
        run_module,
        "parse_arguments",
        lambda: Namespace(
            model="linear",
            epochs=3,
            run_name="combined-run",
            run_id=None,
            evaluate_test=True,
            evaluate_only=False,
        ),
    )

    def fake_prepare_experiment_context(
        model_name,
        epochs_override,
    ):
        calls.append(
            ("prepare", model_name, epochs_override)
        )
        return context

    def fake_training(context, run_name):
        calls.append(
            ("training", context, run_name)
        )

        return {"run_id": "new-training-run"}

    def fake_evaluation(context, run_id):
        calls.append(
            ("evaluation", context, run_id)
        )

    monkeypatch.setattr(
        run_module,
        "prepare_experiment_context",
        fake_prepare_experiment_context,
    )
    monkeypatch.setattr(
        run_module,
        "run_training",
        fake_training,
    )
    monkeypatch.setattr(
        run_module,
        "run_evaluation",
        fake_evaluation,
    )

    main()

    assert calls == [
        ("prepare", "linear", 3),
        ("training", context, "combined-run"),
        ("evaluation", context, "new-training-run"),
    ]
    