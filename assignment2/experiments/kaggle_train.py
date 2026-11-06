"""Run in a Kaggle GPU notebook with the source and image datasets attached; see docs/kaggle.md."""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def extract_zip(archive_path, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if not (destination / member.filename).resolve().is_relative_to(destination.resolve()):
                raise ValueError(f"Unsafe ZIP entry: {member.filename}")
        archive.extractall(destination)


def find_source(input_dir):
    """Kaggle may keep an uploaded ZIP or extract it; accept either form."""
    archives = sorted(input_dir.rglob("assignment2_source.zip"))
    if archives:
        return archives[0]
    markers = sorted(input_dir.rglob("SOURCE_COMMIT.txt"))
    if markers:
        return markers[0].parent
    raise FileNotFoundError(f"Attach the source dataset; no assignment2_source.zip under {input_dir}")


def find_image_root(input_dir, sample, work):
    """Return a folder that contains manifest paths such as archive/Images/Train/..."""
    relative = Path(sample).relative_to("archive")
    for images in sorted(input_dir.rglob("Images")):
        if images.is_dir() and (images.parent / relative).is_file():
            return images.parent
    archives = sorted(input_dir.rglob("food_raw.zip"))
    if archives:
        print(f"Extracting {archives[0]} to local disk...", flush=True)
        extract_zip(archives[0], work / "food_raw")
        return work / "food_raw/archive"
    raise FileNotFoundError(f"Attach the image dataset; {relative} was not found under {input_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/outputs"))
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--model", choices=("resnet50", "baseline_cnn"), default="resnet50")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience on validation Macro F1; 0 disables it")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 2))
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-fast-cuda", action="store_true")
    parser.add_argument("--no-channels-last", action="store_true")
    parser.add_argument("--resume-dir", type=Path,
                        help="Old session folder from an attached notebook output")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.patience < 0:
        parser.error("epochs/batch-size must be positive; workers/patience must be nonnegative")
    if args.finetune and args.model != "resnet50":
        parser.error("--finetune is only valid for resnet50")
    import torch
    import torchvision
    import yaml

    if not torch.cuda.is_available():
        raise RuntimeError("Settings → Accelerator → GPU T4 x2 before running training.")
    major, minor = torch.cuda.get_device_capability(0)
    if f"sm_{major}{minor}" not in torch.cuda.get_arch_list():
        raise RuntimeError(f"This PyTorch build does not support {torch.cuda.get_device_name(0)}; "
                           "choose GPU T4 x2 instead.")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"torch={torch.__version__}; torchvision={torchvision.__version__}", flush=True)
    output = args.output_dir.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    if args.resume_dir:
        old = args.resume_dir.resolve()
        settings = json.loads((old / "runner_settings.json").read_text())
        for name, value in settings.items():
            setattr(args, name, value)
        # Sessions saved before early stopping existed trained without it.
        args.patience = settings.get("patience", 0)
        session = output / old.name
        if not session.exists():
            # Attached notebook outputs are read-only; continue in /kaggle/working.
            shutil.copytree(old, session)
    else:
        variant = args.model if args.model == "baseline_cnn" else (
            "resnet50_finetune" if args.finetune else "resnet50_frozen")
        session = output / f"{variant}_{stamp}"
        session.mkdir(parents=True, exist_ok=False)
        settings = {name: getattr(args, name) for name in (
            "model", "finetune", "epochs", "patience", "batch_size", "workers", "no_amp",
            "no_fast_cuda", "no_channels_last",
        )}
        (session / "runner_settings.json").write_text(json.dumps(settings, indent=2))
    variant = session.name.rsplit("_", 2)[0]
    # Work outside /kaggle/working so images and source copies are not saved as output.
    work = args.work_dir / f"assignment2_{stamp}"
    work.mkdir(parents=True, exist_ok=False)
    source = session / "source_snapshot.zip" if args.resume_dir else find_source(args.input_dir)
    if source.is_dir():
        # Copy contents only; /kaggle/input permissions are read-only.
        for path in source.rglob("*"):
            if path.is_file():
                (work / path.relative_to(source)).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, work / path.relative_to(source))
    else:
        extract_zip(source, work)
    if not args.resume_dir:
        shutil.make_archive(str(session / "source_snapshot"), "zip", work)
    records = []
    for split in ("train", "val", "test"):
        with (work / "data/manifests" / f"{split}.csv").open(newline="", encoding="utf-8") as stream:
            records += [record["path"] for record in csv.DictReader(stream)]
    # Read images in place from /kaggle/input; manifests expect data/raw/archive/Images/...
    (work / "data/raw").mkdir(parents=True, exist_ok=True)
    (work / "data/raw/archive").symlink_to(
        find_image_root(args.input_dir, records[0], work), target_is_directory=True)
    # Check manifest paths themselves; resolving would follow the symlink out of data/raw.
    for record in records:
        path = work / "data/raw" / record
        if Path(record).is_absolute() or ".." in Path(record).parts or not path.is_file():
            raise FileNotFoundError(f"Manifest image is missing or invalid: {path}")
    print(f"Found all {len(records)} manifest images", flush=True)
    # The trainer writes checkpoints straight into the persistent Kaggle output.
    (session / "results").mkdir(exist_ok=True)
    (work / "results").symlink_to(session / "results", target_is_directory=True)
    override = {
        "data": {"batch_size": args.batch_size, "num_workers": args.workers,
                 "persistent_workers": True, "prefetch_factor": 2},
        "training": {"epochs": args.epochs, "amp": not args.no_amp,
                     "fast_cuda": not args.no_fast_cuda,
                     "channels_last": not args.no_channels_last, "save_last": True},
    }
    if args.patience:
        override["training"]["early_stopping_patience"] = args.patience
    if args.model == "resnet50":
        override["model"] = {"freeze_backbone": not args.finetune}
    (work / "config/kaggle.yaml").write_text(yaml.safe_dump(override), encoding="utf-8")
    suffix = f"_resume_{stamp}" if args.resume_dir else ""
    (session / f"pip_freeze{suffix}.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True
    ), encoding="utf-8")
    (session / f"gpu{suffix}.txt").write_text(subprocess.check_output(
        ["nvidia-smi"], text=True
    ), encoding="utf-8")
    commit = (work / "SOURCE_COMMIT.txt").read_text().strip()
    dirty = (work / "SOURCE_DIRTY.txt").read_text().strip()
    (session / "source_provenance.json").write_text(json.dumps({
        "base_commit": commit, "uncommitted_changes": dirty == "True",
        "exact_source": "source_snapshot.zip", "runtime_override": override,
    }, indent=2), encoding="utf-8")

    def run_module(module, *arguments):
        command = [sys.executable, "-u", "-m", module, *map(str, arguments)]
        with (session / "train.log").open("a", encoding="utf-8") as log:
            with subprocess.Popen(command, cwd=work, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                if process.wait():
                    raise RuntimeError(f"{module} failed; see {session / 'train.log'}")

    print(f"Persistent outputs: {session}", flush=True)
    resume_arguments = []
    if args.resume_dir:
        old_runs = list((session / "results/runs" / args.model).iterdir())
        if len(old_runs) != 1:
            raise RuntimeError("Expected exactly one run to resume")
        resume_arguments = ["--resume", old_runs[0]]
    run_module("experiments.run", "--config", f"config/{args.model}.yaml",
               "--experiment-config", "config/kaggle.yaml", "--code_commit", commit,
               *resume_arguments)
    runs = list((session / "results/runs" / args.model).iterdir())
    if len(runs) != 1:
        raise RuntimeError(f"Expected one run, found {len(runs)}")
    run_dir = runs[0]
    run_module("experiments.verify_checkpoint", "--run-dir", run_dir)
    # Checkpoint was selected on validation; evaluate test once, plus error analysis and cost.
    run_module("experiments.evaluate", "--run-dir", run_dir)
    run_module("experiments.plot_curves", "--run-dirs", run_dir,
               "--output", run_dir / "curves.png")
    # Repo bundle excludes model weights and does not overwrite M2 results.
    bundle = session / "repo_results.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        prefix = Path("assignment2/results/kaggle") / variant / run_dir.name
        for path in sorted(run_dir.iterdir()):
            if path.is_file() and path.suffix in (".json", ".png", ".csv"):
                archive.write(path, (prefix / path.name).as_posix())
        for path in sorted(session.iterdir()):
            if path.is_file() and path.suffix in (".txt", ".log", ".json", ".zip") and path != bundle:
                archive.write(path, (prefix / path.name).as_posix())
    print(f"\nDownload into repo: {bundle}", flush=True)
    print(f"Checkpoint in notebook output: {run_dir / 'best_model.pt'}", flush=True)


if __name__ == "__main__":
    main()
