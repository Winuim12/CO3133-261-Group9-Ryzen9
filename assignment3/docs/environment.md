# Assignment 3: local and Docker environment

Run commands from `assignment3`. The same source code, configuration files,
manifests, and vocabulary are used locally and in Docker. The environment check
exercises PyTorch tensor/autograd operations and TorchVision compiled operators;
the training, evaluation, and comparison entry points are described below.

## Version contract

- Python 3.12.10 (`.python-version`, also used by both Docker targets and CI).
- PyTorch 2.7.1 and TorchVision 0.22.1: CPU wheels or CUDA 12.8 wheels.
- Shared direct dependencies are pinned in `requirements.txt`; transitive
  dependencies are not a complete lockfile. Save `pip freeze` for reported runs.
- Select CPU OR GPU requirements, never install both into one environment.

The GPU choice follows [PyTorch's previous-version installation guide](https://pytorch.org/get-started/previous-versions/).
PyTorch 2.7 introduced [Blackwell and CUDA 12.8 support](https://pytorch.org/blog/pytorch-2-7/),
appropriate for the group's RTX 5070 Ti. Hardware detection alone does not prove
the full training pipeline works; the explicit CUDA smoke command must pass.

## Local PowerShell

Create a virtual environment with Python 3.12 (install version 3.12.10 for the
exact shared version). Do not recreate an existing working `.venv` unnecessarily.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m experiments.check_environment --device cpu --smoke-test
.\.venv\Scripts\python.exe -m pytest tests -v
```

For the RTX 5070 Ti, install the GPU requirements instead of the CPU requirements:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m experiments.check_environment --device cuda --smoke-test
```

Switching from the CPU install replaces the pinned torch/torchvision builds;
it downloads several GB. A separate `.venv-gpu` is also possible and is already
ignored by the project. The Windows NVIDIA driver is required;
the CUDA wheel supplies runtime libraries, so installing a separate CUDA toolkit
is not required for this bootstrap. Failure with `--device cuda` must not be
reported as GPU success by silently falling back to CPU.

Activation is optional. If PowerShell blocks `Activate.ps1`, the explicit
`.venv\Scripts\python.exe` commands above still work. On Linux/macOS CPU,
use `python3.12 -m venv .venv` and `.venv/bin/python` for the same entry points.
The pinned `+cpu` wheels here target Windows/Linux; macOS is not a verified
environment for this setup and may require platform-specific PyTorch wheels.

## Local training, evaluation, and comparison

Prerequisites: the dataset extracted to `data/raw/memotion/` (see README) and
the versioned manifests in `data/manifests/`. Do not rerun `prepare_splits.py`
on a training machine; it would replace the reviewed manifests. The first
training run downloads the ResNet-18 ImageNet weights (about 45 MB) into the
PyTorch cache.

```powershell
# Once per machine: training-only vocabulary -> data/processed/vocabulary.json
.\.venv\Scripts\python.exe -m src.data.tokenizer

# Debugging smoke run (labelled "debug", excluded from comparisons)
.\.venv\Scripts\python.exe -m experiments.train --config config/gated.yaml --device cpu --limit-train 200 --limit-val 200 --epochs 2

# Full runs: one per configuration; repeat with --seed 43 and --seed 44
.\.venv\Scripts\python.exe -m experiments.train --config config/image_only.yaml --device cuda
.\.venv\Scripts\python.exe -m experiments.train --config config/text_only.yaml --device cuda
.\.venv\Scripts\python.exe -m experiments.train --config config/concat.yaml --device cuda
.\.venv\Scripts\python.exe -m experiments.train --config config/gated.yaml --device cuda
.\.venv\Scripts\python.exe -m experiments.train --config config/fixed_average.yaml --device cuda

# Validation comparison (tables and figures in results/comparison/)
.\.venv\Scripts\python.exe -m experiments.compare

# Only after configurations are locked on validation: held-out test
.\.venv\Scripts\python.exe -m experiments.evaluate --run <run-id> --split test
.\.venv\Scripts\python.exe -m experiments.compare --split test
```

`--device cuda` fails if CUDA is unavailable instead of falling back to CPU;
`--device auto` uses CUDA when available. Useful training options: `--seed`,
`--epochs`, `--batch-size`, `--num-workers`, `--deterministic`, `--run-name`.
`--limit-train`, `--limit-val`, and `--no-pretrained` mark a run as debugging.
On Windows keep `num_workers: 0` (default) unless more workers are tested.

Each training run writes:

```text
results/runs/<model>/<run-id>/
  config.json            resolved configuration (base.yaml + model YAML + CLI)
  run_metadata.json      manifest/vocabulary hashes, git commit, versions, GPU
  training.json          history, best epoch, runtime, peak VRAM, parameters
  evaluation_val.json    accuracy, Macro-F1, per-class scores, confusion matrix
  predictions_val.csv    per-sample predictions (and gate weight for fusion)
  pip_freeze.txt
  figures/               training curves and confusion matrices
checkpoints/<model>/<run-id>/best_model.pt   selected by validation Macro-F1
```

The run id is `<timestamp>-seed<seed>[-name]` and is printed at the end of
training. `experiments.evaluate` rebuilds the model from the run's saved
`config.json` (not the current YAML) and refuses to run if manifests or the
vocabulary changed since training (`--allow-data-mismatch` overrides this).
After training, the selected checkpoint is reloaded and its validation
Macro-F1 is compared with the saved value (`reload_check` in `training.json`).

## Docker CPU

Start Docker Desktop before these commands. If `docker` is not on PATH, reopen
the terminal after installing Docker Desktop or use its full executable path.

```powershell
docker compose config --quiet
docker compose build
docker compose run --rm assignment3
docker compose run --rm assignment3 python -m pip check
docker compose run --rm assignment3 python -m pytest tests -v
```

The default command runs the CPU environment smoke check. A fresh clone does
not need the dataset to execute it or the bootstrap tests. Compose may create
empty mounted directories on first use.

Training, evaluation, and comparison use the same entry points; outputs land in
the host `results/` and `checkpoints/` folders through the mounts:

```powershell
docker compose run --rm assignment3 python -m src.data.tokenizer
docker compose run --rm assignment3 python -m experiments.train --config config/gated.yaml --device cpu --limit-train 200 --limit-val 200 --epochs 2
docker compose run --rm assignment3 python -m experiments.compare --include-debug
```

## Docker GPU

Windows requires Docker Desktop's WSL 2 Linux-container backend and working
NVIDIA GPU support. See [Docker's Windows GPU prerequisites](https://docs.docker.com/desktop/features/gpu/).
The `gpus: all` Compose setting needs Compose 2.30 or newer.

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml config --quiet
docker compose -f compose.yaml -f compose.gpu.yaml build
docker compose -f compose.yaml -f compose.gpu.yaml run --rm assignment3
docker compose -f compose.yaml -f compose.gpu.yaml run --rm assignment3 python -m pytest tests -v
```

Training and evaluation on the GPU (PowerShell; the backtick continues a line):

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml run --rm assignment3 `
  python -m experiments.train --config config/concat.yaml --device cuda
docker compose -f compose.yaml -f compose.gpu.yaml run --rm assignment3 `
  python -m experiments.evaluate --run <run-id> --split val --device cuda
docker compose -f compose.yaml -f compose.gpu.yaml run --rm assignment3 `
  python -m experiments.compare
```

Use the same five configuration files and seeds as the local workflow. Run
ids, checkpoints, and results are interchangeable between local and Docker
runs because both use project-relative paths and the same mounted folders.

The GPU build installs CUDA wheels; there is no separate host CUDA toolkit
requirement for the container smoke check. The default GPU command explicitly
requires CUDA. CPU CI cannot validate host GPU passthrough.

## Data, code, and output behavior

- Local project root: the directory containing `src/`, `experiments/`, and `config/`.
- Docker root: `/workspace/assignment3`; configuration paths remain project-relative.
- Raw data is mounted read-only; audit/review/manifests/processed data and
  results/checkpoints are mounted with write access.
- `torch-cache` stores downloaded weights across container runs.
- Source/config files are copied into the image, not bind-mounted. **Rebuild
  after editing source, tests, requirements, or configuration.** Output changes
  and edits to mounted datasets do not require rebuilding.
- `experiments.train`, `experiments.evaluate`, and `experiments.compare` are
  implemented and shared by local and Docker execution. They write only values
  computed from actual runs; no results are estimated or filled in.
- `.dockerignore` excludes `.git`, so runs inside a container record
  `"commit": null` in `run_metadata.json`. Record the commit of the image
  build (`git rev-parse HEAD`) with any Docker-produced result, or build from a
  clean, committed tree and note it in the report.
- `results/runs/` and `checkpoints/` are git-ignored. Copy them from the
  training machine; commit only selected tables/figures used in the report.

## CI

`.github/workflows/assignment3-ci.yaml` defines `assignment3-tests`, a separate
check from Assignment 1's `test` and Assignment 2's `assignment2-tests`.
It starts on PRs to `main` so a required check can always report. When Assignment
3 is unchanged it skips heavy steps inside the job. For Assignment 3 changes
it installs CPU requirements, checks dependencies, validates CPU/GPU Compose
configuration, runs the CPU smoke check, and executes `tests/` without raw data.
It does not build the Docker images or run GPU training. A GitHub execution
is only verified after the workflow is pushed and runs; local checks are not
evidence that remote CI has passed.

## Verification record

Executed on 14 November 2026:

| Check | Verified result |
| --- | --- |
| Local `.venv`, Windows | Python 3.12.10; torch 2.7.1+cpu; torchvision 0.22.1+cpu |
| Local dependencies | `pip check`: no broken requirements |
| Local CPU tensor/autograd/TorchVision check | Passed |
| Local bootstrap suite | 10 passed |
| CPU Docker build | `assignment3:cpu` built successfully |
| CPU Compose default command | Passed with configured mounts |
| CPU image tests, without any dataset mount | 10 passed |
| CPU image dependencies | `pip check`: no broken requirements |
| CPU and GPU Compose configuration | Both validated |
| GPU image build / CUDA execution | Not run; configured for CUDA 12.8 and needs explicit verification by Loc |
| GitHub Actions execution | Not run remotely; workflow is local and must be pushed |

The existing `.venv` was populated with the CPU requirements for these checks;
global Python packages were not changed. The local CPU wheel reporting
`cuda_available: false` does not mean the RTX 5070 Ti is absent. Install the GPU
requirements or use the GPU target before GPU smoke checks.

One static reviewer checked the bootstrap without editing files; no significant
defects were reported. Instructor approval and model experiments remain separate
from infrastructure checks. No training or final manifest creation occurred.

### Update: 21 November 2026 (fusion and experiment integration)

| Check | Verified result |
| --- | --- |
| Unit/integration tests (Linux, Python 3.12, torch 2.7.1, CPU) | 48 passed, 1 skipped (raw-data batch test without images) |
| Same suite with a real-image subset | Raw-data batch contract test passed |
| Train -> reload -> evaluate -> compare, 5 configurations, real images, CPU | All completed; reloaded checkpoints reproduced validation Macro-F1 exactly |
| Local Windows `.venv`, `gated` debug run on CPU | Completed (about 12 s/epoch on a 200/200 subset) |
| Pretrained ResNet-18 download | Not verified in the test environment (network blocked) |
| Docker rebuild with the new entry points | Not run |
| GPU (local or Docker), runtime, and peak VRAM | Not run; required before reporting GPU results |

All runs above were debugging runs (subsets and/or random image weights);
their scores are not experimental results and must not be reported.
