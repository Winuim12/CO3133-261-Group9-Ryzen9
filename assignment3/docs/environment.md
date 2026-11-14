# Assignment 3: local and Docker environment

Run commands from `assignment3`. The shared bootstrap checks configurations,
PyTorch tensor/autograd operations, and TorchVision compiled operators. It
downloads no dataset or pretrained weights and does not train a model.

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
- Training/evaluation/comparison entry points are reserved for team members.
  Their `--help` works; real invocations exit with an explicit not-implemented
  message until the assigned implementation is complete. They generate no
  fake histories, predictions, or results.

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
