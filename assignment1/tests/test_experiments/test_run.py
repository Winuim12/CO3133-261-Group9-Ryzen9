#test_run.py

"""
RUN: python -m pytest tests/test_experiments/test_run.py -v
"""

from argparse import Namespace
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.run import (
    run_experiment, 
    run_configured_experiment, 
    evaluate_best_checkpoint, 
    parse_arguments,
    evaluate_test_if_requested,
    main,
)
from src.training.checkpoint import save_checkpoint
from src.models.linear import LinearClassifier

def test_parse_arguments_controls_evaluation_modes():
    training_arguments = parse_arguments(["--model", "linear"])
    train_and_evaluation_arguments = parse_arguments(["--model", "linear", "--evaluate-test"])
    evaluation_only_arguments = parse_arguments(["--model", "linear", "--evaluate-only"])

    assert training_arguments.evaluate_test is False
    assert training_arguments.evaluate_only is False

    assert train_and_evaluation_arguments.evaluate_test is True
    assert train_and_evaluation_arguments.evaluate_only is False

    assert evaluation_only_arguments.evaluate_test is False
    assert evaluation_only_arguments.evaluate_only is True

def test_main_routes_evaluation_only_without_training(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "experiments.run.parse_arguments",
        lambda: Namespace(
            model="linear",
            epochs=None,
            evaluate_test=False,
            evaluate_only=True,
        ),
    )

    def fake_training(**kwargs):
        calls.append("training")

    def fake_evaluation(model_name):
        calls.append(f"evaluation:{model_name}")

    monkeypatch.setattr("experiments.run.run_model_from_config", fake_training)
    monkeypatch.setattr("experiments.run.run_evaluation_from_config", fake_evaluation)

    main()

    assert calls == ["evaluation:linear"]

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

    model, history, test_loader, training_time_seconds = run_configured_experiment(
        shared_config=shared_config,
        model_config=model_config,
        train_dataset=dataset,
        validation_dataset=dataset,
        test_dataset=dataset,
        checkpoint_path=tmp_path / "best_model.pt",
        device=torch.device("cpu"),
    )

    assert isinstance(model, LinearClassifier)
    assert len(history["train_loss"]) == 1
    assert (tmp_path / "best_model.pt").exists()
    assert test_loader.batch_size == 4
    assert training_time_seconds > 0


def test_evaluate_test_if_requested_restores_and_evaluates_model(tmp_path):
    torch.manual_seed(42)

    images = torch.randn(4, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 3])

    test_loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=2,
        shuffle=False,
    )

    model = LinearClassifier()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01
    )

    original_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.state_dict().items()
    }

    checkpoint_path = tmp_path / "best_model.pt"

    save_checkpoint(
        path=checkpoint_path, 
        model=model,
        optimizer=optimizer,
        epoch=1,
        validation_loss=0.75,
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)

    test_results = evaluate_test_if_requested(
        evaluate_test=True,
        model=model,
        test_loader=test_loader,
        checkpoint_path=checkpoint_path,
        device=torch.device("cpu"),
    )

    for parameter_name, original_value in original_parameters.items():
        assert torch.equal(model.state_dict()[parameter_name], original_value)

    assert test_results["targets"].numel() == len(labels)
    assert test_results["predictions"].numel() == len(labels)

def test_evaluate_test_if_requested_skips_test_by_default():
    test_results = evaluate_test_if_requested(
        evaluate_test=False,
        model=None,
        test_loader=None,
        checkpoint_path=None,
        device=torch.device("cpu"),
    )

    assert test_results is None

def test_parse_arguments_accepts_run_name():
    arguments = parse_arguments([
        "--model",
        "linear",
        "--run-name",
        "linear-final",
    ])

    assert arguments.run_name == "linear-final"

def test_main_forwards_run_name_to_training(monkeypatch):
    received_arguments = {}

    monkeypatch.setattr(
        "experiments.run.parse_arguments",
        lambda: Namespace(
            model="linear",
            epochs=3,
            run_name="linear-final",
            evaluate_test=False,
            evaluate_only=False,
        )
    )

    def fake_training(**arguments):
        received_arguments.update(arguments)

    monkeypatch.setattr("experiments.run.run_model_from_config", fake_training)

    main()

    assert received_arguments == {
        "model_name": "linear",
        "epochs_override": 3,
        "evaluate_test": False,
        "run_name": "linear-final",
    }

