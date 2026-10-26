#test_run_info.py

import json

from src.utils.run_info import create_run_directory

import hashlib

import torch

from experiments.context import ExperimentContext
from src.utils.run_info import collect_run_metadata


def test_run_directories_are_unique_and_preserve_config(tmp_path):
    config = {
        "seed": 42,
        "model": {"name": "baseline_cnn"},
        "training": {"epochs": 2},
    }

    first = create_run_directory(tmp_path, config)
    second = create_run_directory(tmp_path, config)

    assert first != second
    assert first.is_dir()
    assert second.is_dir()

    for directory in (first, second):
        saved = json.loads(
            (directory / "config.json").read_text(encoding="utf-8")
        )
        assert saved == config

def test_metadata_records_manifest_identity_and_class_mapping(tmp_path):
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()

    for split in ("train", "val", "test"):
        (manifests_dir / f"{split}.csv").write_bytes(
            f"{split}-data\n".encode("utf-8")
        )

    mapping = {"pho": 0}
    (manifests_dir / "class_to_idx.json").write_text(
        json.dumps(mapping),
        encoding="utf-8",
    )

    context = ExperimentContext(
        project_root=tmp_path,
        config={
            "seed": 42,
            "data": {"manifests_dir": "manifests"},
        },
        device=torch.device("cpu"),
    )

    metadata = collect_run_metadata(
        context,
        tmp_path / "example-run",
        code_commit="test-commit",
    )

    assert metadata["run_id"] == "example-run"
    assert metadata["code_commit"] == "test-commit"
    assert metadata["class_to_idx"] == mapping
    assert metadata["manifest_hashes"]["train.csv"] == (
        hashlib.sha256(b"train-data\n").hexdigest()
    )
    assert metadata["device"] == "cpu"

    context.config["seed"] = 99
    assert metadata["config"]["seed"] == 42