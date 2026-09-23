#run_paths.py

from datetime import datetime
from pathlib import Path
import re

def _normalize_run_name(run_name):
    normalized_name = run_name.strip().lower()
    normalized_name = re.sub(r"[^a-z0-9]+", "-", normalized_name)
    normalized_name = normalized_name.strip("-")

    if not normalized_name:
        raise ValueError("run name must contain a letter or number")

    return normalized_name

def _build_artifact_paths(run_id, run_directory):
    figures_directory = run_directory / "figures"
    return {
        "run_id": run_id,
        "run_directory": run_directory,
        "checkpoint_path": run_directory / "best_model.pt",
        "training_results_path": run_directory / "training.json",
        "evaluation_results_path": run_directory / "evaluation.json",
        "training_curves_path": figures_directory / "training_curves.png",
        "confusion_matrix_path": figures_directory / "confusion_matrix.png",
        "prediction_examples_path": figures_directory / "prediction_examples.png",
    }

def create_run_paths(
    results_root,
    model_name,
    run_name=None,
    timestamp=None,
):
    results_root = Path(results_root)
    normalized_model_name = model_name.lower()

    if timestamp is None:
        timestamp = datetime.now().strftime(
            "%Y%m%d-%H%M%S-%f"
        )

    run_id = timestamp

    if run_name is not None: 
        normalized_run_name = _normalize_run_name(run_name)
        run_id = f"{timestamp}-{normalized_run_name}"

    run_directory = results_root / normalized_model_name / run_id
    run_directory.mkdir(parents=True, exist_ok=False)

    figures_directory = run_directory / "figures"
    figures_directory.mkdir()

    return _build_artifact_paths(
        run_id = run_id,
        run_directory = run_directory,
    )

def load_run_paths(results_root, model_name, run_id):
    results_root = Path(results_root)
    run_directory = results_root / model_name.lower() / run_id

    if not run_directory.is_dir():
        raise FileNotFoundError(f"experiment run does not exist: {run_directory}")

    return _build_artifact_paths(
        run_id = run_id,
        run_directory = run_directory,
    )