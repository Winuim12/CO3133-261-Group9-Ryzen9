#test_config.py

from src.utils.config import load_config, merge_configs, load_experiment_config

def test_load_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "seed: 42\ndata:\n num_classes: 30\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config == {
        "seed": 42,
        "data": {"num_classes": 30},
    }

def test_merge_configs():
    base = {
        "seed": 42,
        "training": {"epochs": 20, "batch_size": 32},
    }
    overrides = {
        "training": {"epochs": 5},
        "model": {"name": "resnet50"},
    }

    merged = merge_configs(base, overrides)

    assert merged == {
        "seed": 42,
        "training": {"epochs": 5, "batch_size": 32},
        "model": {"name": "resnet50"},
    }

    # Editing the result must not change either input.
    merged["training"]["batch_size"] = 64
    merged["model"]["name"] = "baseline_cnn"

    assert base["training"] == {"epochs": 20, "batch_size": 32}
    assert overrides["model"]["name"] == "resnet50"

def test_load_experiment_config(tmp_path):
    shared_path = tmp_path / "shared.yaml"
    model_path = tmp_path / "model.yaml"
    experiment_path = tmp_path / "experiment.yaml"

    shared_path.write_text(
        "seed: 42\ntraining:\n  epochs: 20\n  learning_rate: 0.001\n",
        encoding="utf-8",
    )
    model_path.write_text(
        "model:\n  name: resnet50\n"
        "  freeze_backbone: false\n"
        "training:\n  learning_rate: 0.0001\n",
        encoding="utf-8",
    )
    experiment_path.write_text(
        "model:\n  freeze_backbone: true\n",
        encoding="utf-8",
    )

    config = load_experiment_config(
        shared_path,
        model_path,
        experiment_path,
    )

    assert config == {
        "seed": 42,
        "training": {
            "epochs": 20,
            "learning_rate": 0.0001,
        },
        "model": {
            "name": "resnet50",
            "freeze_backbone": True,
        },
    }