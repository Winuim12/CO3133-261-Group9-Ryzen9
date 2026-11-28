import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .config import DISTILBERT_ID, DISTILBERT_REVISION, PROJECT_ROOT


# --------------------------------------------------------------------- seeds
def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python (used by train-time image jitter), NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        # Slower; full determinism on CUDA additionally needs this variable.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: derive Python/NumPy seeds from the torch seed."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


# -------------------------------------------------------------------- device
def resolve_device(requested: str = "auto") -> torch.device:
    """`cuda` must really be available; only `auto` may fall back to CPU."""
    requested = str(requested).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was explicitly requested but is unavailable. Install requirements-gpu.txt, "
            "check the NVIDIA driver, or pass --device cpu for a CPU run."
        )
    if requested not in {"cpu", "cuda"} and not requested.startswith("cuda:"):
        raise ValueError(f"Unknown device {requested!r}; use auto, cpu, or cuda.")
    return torch.device(requested)


# -------------------------------------------------------------------- hashes
def file_sha256(path: str | Path, normalise_line_endings: bool = False) -> str:
    """SHA-256 of a file. Text manifests use LF-normalised hashes like prepare_splits.py."""
    data = Path(path).read_bytes()
    if normalise_line_endings:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def data_fingerprint(config: dict, root: Path = PROJECT_ROOT) -> dict:
    """Record the manifests and vocabulary or tokenizer that define a run."""
    data = config["data"]
    fingerprint = {}
    keys = ("train_manifest", "val_manifest", "test_manifest")
    if config.get("model", {}).get("text_encoder") == "distilbert":
        fingerprint["text_tokenizer"] = {"id": DISTILBERT_ID, "revision": DISTILBERT_REVISION}
    else:
        keys += ("vocabulary",)
    for key in keys:
        path = root / data[key]
        fingerprint[key] = {
            "path": data[key],
            "sha256": file_sha256(path, normalise_line_endings=True) if path.exists() else None,
        }
    report = root / Path(data["train_manifest"]).parent / "split_report.json"
    if report.exists():
        expected = json.loads(report.read_text(encoding="utf-8")).get("splits", {})
        fingerprint["matches_split_report"] = all(
            fingerprint[f"{name}_manifest"]["sha256"] == expected.get(name, {}).get("sha256")
            for name in ("train", "val", "test")
        )
    return fingerprint


# --------------------------------------------------------------- run folders
def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(text).strip().lower()).strip("-")
    if not slug:
        raise ValueError("Run name must contain a letter or number.")
    return slug


def create_run_paths(config: dict, model_name: str, seed: int, run_name: str | None = None,
                     root: Path = PROJECT_ROOT, timestamp: str | None = None) -> dict:
    """results/runs/<model>/<run-id>/ and checkpoints/<model>/<run-id>/best_model.pt."""
    timestamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_id = f"{timestamp}-seed{seed}"
    if run_name:
        run_id = f"{run_id}-{_slug(run_name)}"
    model_name = _slug(model_name).replace("-", "_")
    run_dir = root / config["outputs"]["runs_dir"] / model_name / run_id
    checkpoint_dir = root / config["outputs"]["checkpoints_dir"] / model_name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "figures_dir": run_dir / "figures",
        "checkpoint_path": checkpoint_dir / "best_model.pt",
        "config_path": run_dir / "config.json",
        "metadata_path": run_dir / "run_metadata.json",
        "training_path": run_dir / "training.json",
    }


def find_run_dir(run: str, config: dict | None = None, root: Path = PROJECT_ROOT) -> Path:
    """Accept a run directory path or a run id under results/runs/<model>/."""
    candidate = Path(run)
    if not candidate.is_absolute():
        candidate = root / candidate
    if candidate.is_dir():
        return candidate
    runs_dir = root / ((config or {}).get("outputs", {}).get("runs_dir", "results/runs"))
    matches = [path for path in runs_dir.glob(f"*/{run}") if path.is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one run named {run!r} under {runs_dir}, found {len(matches)}.")
    return matches[0]


# ---------------------------------------------------------------- provenance
def git_revision(root: Path = PROJECT_ROOT) -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                              timeout=20, check=True).stdout.strip()
    try:
        commit = run("rev-parse", "HEAD")
        # Ignore CRLF-only differences that Windows checkouts can report.
        changed = run("-c", "core.autocrlf=true", "status", "--porcelain", "--", ".")
        return {"commit": commit, "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(changed), "changed_files": changed.splitlines()[:50]}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "branch": None, "dirty": None, "changed_files": []}


def _version(package: str):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_info(device: torch.device | None = None) -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "in_docker": Path("/.dockerenv").exists(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "packages": {name: _version(name) for name in
                     ("torchvision", "numpy", "Pillow", "scikit-learn", "PyYAML", "matplotlib")},
    }
    if device is not None:
        info["device"] = str(device)
        if device.type == "cuda":
            index = device.index if device.index is not None else torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(index)
            info["gpu"] = {
                "name": properties.name,
                "total_memory_gib": round(properties.total_memory / 1024**3, 2),
                "capability": f"{properties.major}.{properties.minor}",
                "driver_cuda": _nvidia_smi_driver(),
            }
    return info


def _nvidia_smi_driver():
    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=20, check=True).stdout
        return output.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def save_pip_freeze(path: Path) -> None:
    try:
        output = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                text=True, timeout=120, check=True).stdout
    except (OSError, subprocess.SubprocessError) as error:
        output = f"# pip freeze failed: {error}\n"
    Path(path).write_text(output, encoding="utf-8")


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def build_run_metadata(config: dict, config_file: str, device: torch.device, run_paths: dict,
                       command: list[str], debug: bool, root: Path = PROJECT_ROOT) -> dict:
    return {
        "run_id": run_paths["run_id"],
        "model": config["model"]["name"],
        "seed": config["training"]["seed"],
        "debug_run": debug,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": command,
        "config_file": str(config_file),
        "checkpoint_path": relative_to_root(run_paths["checkpoint_path"], root),
        "labels": config["labels"],
        "data": data_fingerprint(config, root),
        "source": git_revision(root),
        "environment": environment_info(device),
    }


# --------------------------------------------------------------------- utils
def relative_to_root(path: Path, root: Path = PROJECT_ROOT) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _json_default(value):
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialise {type(value).__name__}")


def save_json(path: str | Path, payload) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
                         encoding="utf-8")
    temporary.replace(path)
    return path


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
