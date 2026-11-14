"""Checks for the shared handover configuration, without downloading data."""

from pathlib import Path

import pytest

from src.config import load_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["image_only", "text_only", "concat", "gated", "fixed_average"])
def test_model_configs_share_labels_and_data_paths(name):
    config = load_config(ROOT / "config" / f"{name}.yaml")
    assert config["model"]["name"] == name
    assert config["data"]["text_column"] == "text_corrected"
    assert config["data"]["train_manifest"] == "data/manifests/train.csv"
    assert config["labels"] == {
        "very_negative": 0,
        "negative": 1,
        "neutral": 2,
        "positive": 3,
        "very_positive": 4,
    }


def test_paths_are_relative_to_project_not_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config("config/gated.yaml")
    assert config["model"]["name"] == "gated"


def test_unknown_config_is_not_silently_ignored():
    with pytest.raises(FileNotFoundError):
        load_config("config/does-not-exist.yaml")


def test_model_override_preserves_other_model_settings():
    config = load_config("config/gated.yaml")
    assert config["model"]["image_encoder"] == "resnet18"
    assert config["model"]["projection_dim"] == 256
    assert config["training"]["seed"] == 42
