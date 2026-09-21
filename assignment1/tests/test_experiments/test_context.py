#test_context.py
"""
RUN: python -m pytest tests/test_experiments/test_context.py -v
"""

import torch

from experiments.context import ExperimentContext
import experiments.context as context_module

def test_experiment_context_stores_prepared_dependencies():
    train_dataset = object()
    validation_dataset = object()
    test_dataset = object()

    train_loader = object()
    validation_loader = object()
    test_loader = object()

    device = torch.device("cpu")

    context = ExperimentContext(
        model_name="linear",
        shared_config={"seed": 42},
        model_config={"model": {"name": "linear"}},
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        mean=0.286209,
        std=0.353160,
        device=device,
    )

    assert context.model_name == "linear"
    assert context.shared_config == {"seed": 42}
    assert context.train_dataset is train_dataset
    assert context.validation_dataset is validation_dataset
    assert context.test_dataset is test_dataset
    assert context.train_loader is train_loader
    assert context.validation_loader is validation_loader
    assert context.test_loader is test_loader
    assert context.mean == 0.286209
    assert context.std == 0.353160
    assert context.device == device

def test_prepare_experiment_context_prepares_dependencies_once(
    monkeypatch,
    tmp_path,
):
    train_dataset = object()
    validation_dataset = object()
    test_dataset = object()

    train_loader = object()
    validation_loader = object()
    test_loader = object()

    calls = {
        "datasets": 0,
        "dataloaders": 0,
        "seeds": [],
    }

    shared_config = {
        "seed": 42,
        "data": {
            "batch_size": 128,
            "num_workers": 2,
            "validation_size": 0.1,
        },
        "training": {
            "epochs": 20,
        },
    }

    model_config = {
        "model": {
            "name": "linear",
            "parameters": {},
        }
    }

    def fake_load_config(path):
        if path.name == "config.yaml":
            return shared_config
        return model_config

    def fake_create_normalized_datasets(
        data_dir,
        validation_size,
        seed,
    ):
        calls["datasets"] += 1

        assert data_dir == tmp_path / "data" / "raw" / "fashion_mnist"
        assert validation_size == 0.1
        assert seed == 42

        return (
            train_dataset,
            validation_dataset,
            test_dataset,
            0.286209,
            0.353160,
        )

    def fake_create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size,
        num_workers,
    ):
        calls["dataloaders"] += 1

        assert batch_size == 128
        assert num_workers == 2

        return train_loader, validation_loader, test_loader

    monkeypatch.setattr(context_module, "PROJECT_ROOT", tmp_path, raising=False)
    monkeypatch.setattr(context_module, "load_config", fake_load_config, raising=False)
    monkeypatch.setattr(context_module, "set_seed", lambda seed: calls["seeds"].append(seed), raising=False)
    monkeypatch.setattr(context_module, "create_normalized_datasets", fake_create_normalized_datasets, raising=False)
    monkeypatch.setattr(context_module, "create_dataloaders", fake_create_dataloaders, raising=False)
    monkeypatch.setattr(context_module.torch.cuda, "is_available", lambda: False)

    context = context_module.prepare_experiment_context(
        model_name="linear",
        epochs_override=3,
    )

    assert calls == {
        "datasets": 1,
        "dataloaders": 1,
        "seeds": [42],
    }

    assert context.model_name == "linear"
    assert context.shared_config["training"]["epochs"] == 3
    assert context.model_config is model_config

    assert context.train_dataset is train_dataset
    assert context.validation_dataset is validation_dataset
    assert context.test_dataset is test_dataset

    assert context.train_loader is train_loader
    assert context.validation_loader is validation_loader
    assert context.test_loader is test_loader

    assert context.mean == 0.286209
    assert context.std == 0.353160
    assert context.device == torch.device("cpu")
