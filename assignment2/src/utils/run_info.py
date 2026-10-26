#run_info.py

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import hashlib
import platform
from copy import deepcopy
from importlib.metadata import version

import torch

def create_run_directory(results_root, config):
    config_text = json.dumps(config, indent=2, allow_nan=False)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_{uuid4().hex[:8]}"

    run_dir = (
        Path(results_root)
        / config["model"]["name"]
        / run_id
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "config.json").write_text(
        config_text + "\n",
        encoding="utf-8",
    )

    return run_dir

def collect_run_metadata(context, run_dir, code_commit=None):
    manifest_dir = (
        context.project_root
        / context.config["data"]["manifests_dir"]
    )

    filenames = (
        "train.csv",
        "val.csv",
        "test.csv",
        "class_to_idx.json",
    )

    manifest_hashes = {
        filename: hashlib.sha256(
            (manifest_dir / filename).read_bytes()
        ).hexdigest()
        for filename in filenames
    }

    class_to_idx = json.loads(
        (manifest_dir / "class_to_idx.json").read_text(
            encoding="utf-8"
        )
    )

    return {
        "run_id": Path(run_dir).name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": deepcopy(context.config),
        "code_commit": code_commit,
        "class_to_idx": class_to_idx,
        "manifest_hashes": manifest_hashes,
        "python_version": platform.python_version(),
        "runtime_platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in ("torch", "torchvision", "numpy", "Pillow", "PyYAML")
        },
        "device": str(context.device),
        "gpu_name": (
            torch.cuda.get_device_name(context.device)
            if context.device.type == "cuda"
            else None
        ),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "checkpoint_selection": {
            "metric": "validation_macro_f1",
            "mode": "max",
            "tie_break": "earlier_epoch",
        },
    }