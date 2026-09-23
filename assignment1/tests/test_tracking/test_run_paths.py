#test_run_paths.py
"""
RUN: python -m pytest tests/tracking/test_run_paths.py -v
"""

from src.tracking.run_paths import create_run_paths, load_run_paths
import pytest

def test_create_run_paths_creates_versioned_directory(tmp_path):
    results_root = tmp_path / "runs"

    paths = create_run_paths(
        results_root = results_root,
        model_name = "linear",
        run_name = "final",
        timestamp = "20260920-120000-000000"
    )

    expected_run_id = "20260920-120000-000000-final"
    expected_run_directory = results_root / "linear" / expected_run_id

    assert paths["run_id"] == expected_run_id
    assert paths["run_directory"] == expected_run_directory
    assert paths["checkpoint_path"] == expected_run_directory / "best_model.pt"

    assert expected_run_directory.is_dir()
    assert (expected_run_directory / "figures").is_dir()

def test_create_run_paths_normalizes_run_name(tmp_path):
    paths = create_run_paths(
        results_root = tmp_path / "runs",
        model_name = "linear",
        run_name = "My Final / Experiment!",
        timestamp = "20260920-120000-000000",
    )

    assert paths["run_id"] == (
        "20260920-120000-000000-my-final-experiment"
    )

def test_create_run_paths_rejects_empty_normalized_run_name(tmp_path):
    with pytest.raises(
        ValueError,
        match="run name must contain a letter or number",
    ):
        create_run_paths(
            results_root = tmp_path / "runs",
            model_name = "linear",
            run_name = "!!!",
            timestamp = "20260920-120000-000000",
        )

def test_create_run_paths_returns_all_artifact_paths(tmp_path):
    paths = create_run_paths(
        results_root = tmp_path / "runs",
        model_name = "linear",
        run_name = "final",
        timestamp = "20260920-120000-000000"
    )

    run_directory = paths["run_directory"]
    figures_directory = run_directory / "figures"

    assert paths["training_results_path"] == run_directory / "training.json"
    assert paths["evaluation_results_path"] == run_directory / "evaluation.json"
    assert paths["training_curves_path"] == figures_directory / "training_curves.png"
    assert paths["confusion_matrix_path"] == figures_directory / "confusion_matrix.png"
    assert paths["prediction_examples_path"] == figures_directory / "prediction_examples.png"

def test_create_run_paths_rejects_duplicate_run_id(tmp_path):
    arguments = {
        "results_root": tmp_path / "runs",
        "model_name": "linear",
        "run_name": "final",
        "timestamp": "20260920-120000-000000",
    }

    create_run_paths(**arguments)

    with pytest.raises(FileExistsError):
        create_run_paths(**arguments)

def test_load_run_paths_rejects_missing_run(tmp_path):
    with pytest.raises(
        FileNotFoundError,
        match="experiment run does not exist",
    ):
        load_run_paths(
            results_root = tmp_path / "runs",
            model_name = "linear",
            run_id = "20260920-120000-000000-final"
        )

def test_load_run_paths_returns_existing_artifact_paths(tmp_path):
    results_root = tmp_path / "runs"
    
    created_paths = create_run_paths(
        results_root = results_root,
        model_name = "linear",
        run_name = "final",
        timestamp = "20260920-120000-000000"
    )
    loaded_paths = load_run_paths(
        results_root = tmp_path / "runs",
        model_name = "linear",
        run_id = "20260920-120000-000000-final"
    )

    assert loaded_paths == created_paths