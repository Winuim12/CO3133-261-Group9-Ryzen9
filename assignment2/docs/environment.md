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

## Verification Status and Limits

- Docker CPU tests passed: 25 tests.
- GPU training completed for the scratch CNN and frozen ResNet-50.
- Both selected checkpoints reproduced their saved validation metrics.
- Comparison CSV and training-history figures were generated.
- Local virtual-environment execution and other GPUs remain unverified.
- Direct dependencies are pinned; transitive dependencies are not fully locked.
- Test-set evaluation and full fine-tuning remain pending.

## M2 Reproduction Guide

Run commands from the assignment2 directory.

### Dataset Preparation

Obtain 30VNFoods Kaggle version 11 and place the images under:

data/raw/archive/Images/

Preserve the Train, Test and Validate directories inside Images.
Experiment membership comes from the reviewed CSV manifests, not those
original directory names. Do not regenerate the splits for reproduction.

The saved manifests contain:
- Training: 17,486 images.
- Validation: 3,749 images.
- Test: 3,747 images.

### Build and Test

Commit the source before building a reported experiment. The image copies
local source files, so rebuild after source or dependency changes.

```powershell
docker compose build
docker compose run --rm assignment2 python -m pytest tests -v
```

The Compose service uses 1 GB shared memory for DataLoader workers.
GPU execution requires compatible NVIDIA hardware/drivers and container
GPU support.

### Train New Runs

Record the commit corresponding to the source copied into the image:

```powershell
$trainingCommit = git rev-parse HEAD
```

Scratch CNN:

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml run --rm assignment2 python -m experiments.run --config config/baseline_cnn.yaml --epochs 20 --code_commit $trainingCommit
```

Frozen pretrained ResNet-50:

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml run --rm assignment2 python -m experiments.run --config config/resnet50.yaml --experiment-config config/ablation_frozen.yaml --epochs 20 --code_commit $trainingCommit
```

Each command starts a new run and prints its directory. Use those new
directories when verifying or comparing reproduced runs. Numerical results
may differ across environments.

### Verify the Recorded M2 Runs

The recorded training source commit is:
5d106abab40a6720ebc5874b9a748f01356c965c

```powershell
$baselineRun = "results/runs/baseline_cnn/20261025T103059Z_a86f4102"
$resnetRun = "results/runs/resnet50/20261025T112235Z_014bbc02"

docker compose -f compose.yaml -f compose.gpu.yaml run --rm assignment2 python -m experiments.verify_checkpoint --run-dir $baselineRun
docker compose -f compose.yaml -f compose.gpu.yaml run --rm assignment2 python -m experiments.verify_checkpoint --run-dir $resnetRun
```

These commands require the corresponding run artifacts to be present
locally. Generated runs and checkpoints are excluded from Git.

### Regenerate M2 Tables and Figures

After successful checkpoint verification:

```powershell
docker compose run --rm assignment2 python -m experiments.compare --run-dirs $baselineRun $resnetRun
docker compose run --rm assignment2 python -m experiments.plot_curves --run-dirs $baselineRun $resnetRun
```

Outputs:
- results/comparison.csv
- results/figures/m2_training_histories.png