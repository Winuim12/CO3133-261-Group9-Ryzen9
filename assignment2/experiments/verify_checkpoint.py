#verify_checkpoint.py

import json
import math
from argparse import ArgumentParser
from pathlib import Path

import torch

from experiments.context import (
    PROJECT_ROOT,
    ExperimentContext,
    prepare_data_loaders,
)
from src.evaluation.evaluator import evaluate
from src.models.factory import build_model
from src.utils.run_info import collect_run_metadata
from src.utils.seed import set_seed


def verify_checkpoint(run_dir):
    run_dir = PROJECT_ROOT / Path(run_dir)

    config = json.loads(
        (run_dir / "config.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (run_dir / "metadata.json").read_text(encoding="utf-8")
    )
    history = json.loads(
        (run_dir / "history.json").read_text(encoding="utf-8")
    )

    set_seed(config["seed"])
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    context = ExperimentContext(
        project_root=PROJECT_ROOT,
        config=config,
        device=device,
    )

    current_metadata = collect_run_metadata(context, run_dir)
    if current_metadata["manifest_hashes"] != metadata["manifest_hashes"]:
        raise ValueError("Manifest files differ from the saved run.")

    checkpoint = torch.load(
        run_dir / "best_model.pt",
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["run_metadata"]["config"] != config:
        raise ValueError("Checkpoint and saved configuration differ.")

    # Restore all weights from the checkpoint; no pretrained download needed.
    model_config = dict(config["model"])
    model_config["pretrained"] = False
    model = build_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)

    _, val_loader, _ = prepare_data_loaders(context)

    if val_loader.dataset.class_to_idx != metadata["class_to_idx"]:
        raise ValueError("Class mapping differs from the saved run.")

    result = evaluate(
        model,
        val_loader,
        device,
        num_classes=config["data"]["num_classes"],
    )

    saved_record = next(
        record for record in history
        if record["epoch"] == checkpoint["epoch"]
    )

    for name in ("accuracy", "macro_f1"):
        expected = saved_record["val"][name]
        actual = result["metrics"][name]

        print(f"{name}: saved={expected:.10f}, reloaded={actual:.10f}")

        if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"Reloaded validation {name} does not match.")

    output = {
        "split": "val",
        "checkpoint_epoch": checkpoint["epoch"],
        "matches_saved_metrics": True,
        "metrics": result["metrics"],
    }
    (run_dir / "validation_reload.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("Checkpoint reload verified.")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Verify saved checkpoint validation metrics"
    )
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    verify_checkpoint(args.run_dir)