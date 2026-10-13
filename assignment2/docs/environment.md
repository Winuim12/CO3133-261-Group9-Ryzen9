# Assignment 2 Environment

## Docker environment

Base image: pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

User-reported successful checks on 13 Oct 2026:
- Python 3.11.13
- PyTorch 2.7.1+cu128
- TorchVision 0.22.1+cu128
- Dependency consistency: pip check passed
- Shared package imports: passed
- GPU: NVIDIA GeForce RTX 5070 Ti
- GPU matrix calculation: returned 256.0

These checks verify environment setup, not model training or performance.

## Build

Run commands from the assignment2 directory:

```powershell
docker compose build
```

Rebuild after changing dependencies or source files copied into the image.

## CPU execution

```powershell
docker compose run --rm assignment2 python --version
```

## GPU execution

Requires compatible NVIDIA hardware/drivers and Docker GPU support.
On Windows, use Docker Desktop's WSL 2 backend.

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml run --rm assignment2 python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

## Persistent storage

- data/raw: original images, mounted read-only.
- data/manifests: writable dataset manifests.
- results: writable logs, figures and checkpoints.
- torch-cache: Docker volume for downloaded pretrained weights.

Host-mounted files and the named cache volume survive removal of
temporary containers. The --rm option removes the container after exit.

## Current limits

- Local virtual-environment setup has not been verified.
- Training and checkpoint reload have not been tested.
- Other GPUs have not been tested.
- Direct dependencies are pinned; transitive dependencies are not yet locked.