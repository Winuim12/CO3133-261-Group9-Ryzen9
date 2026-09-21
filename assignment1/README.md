# Assignment 1: Fashion-MNIST Classification

This project trains and evaluates five Fashion-MNIST classifiers: Linear,
MLP, CNN, RNN, and Transformer.

## Setup

Use Python 3.12. From the `assignment1` directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Install one PyTorch variant:

```powershell
# NVIDIA CUDA environment
python -m pip install -r requirements-gpu.txt

# Or CPU environment
python -m pip install -r requirements-ci.txt
```

The runner selects CUDA when available and otherwise uses CPU.
Fashion-MNIST is downloaded automatically on the first run.

## Train a model

```powershell
python -m experiments.run --model linear --epochs 3 --run-name m1-draft
```

Replace `linear` with `mlp`, `cnn`, `rnn`, or `transformer`. The `--epochs`
option overrides the default in `config/config.yaml`. Model-specific
parameters are in `config/<model>.yaml`.

For the M1 draft, Linear and MLP are required; CNN is optional. RNN and
Transformer are required for the M2 final submission.

Each training command creates a new directory:

```text
results/runs/<model>/<run-id>/
  best_model.pt
  training.json
  figures/training_curves.png
```

The terminal prints the run ID and directory. `best_model.pt` is selected
using the lowest validation loss. `training.json` records the configuration,
training time, and per-epoch training and validation metrics.

## Evaluate an existing run

After choosing a run, copy its printed run ID:

```powershell
python -m experiments.run --model linear --evaluate-only --run-id <run-id>
```

Evaluation adds `evaluation.json`, a confusion matrix, and prediction
examples to that run directory. To train and immediately evaluate in one
command, add `--evaluate-test` to the training command.

**Current limitation:** evaluation-only still reads the current YAML
configuration. Until saved-run configuration loading is implemented,
keep the original model and data configuration unchanged when evaluating
an older checkpoint.

## Reproduce a training run

Use the same code version, model YAML, shared `config/config.yaml`,
dependencies, and epoch count recorded in `training.json`. Run the same
training command again. This creates a **new** run directory rather than
overwriting the old checkpoint. Small differences may occur across
hardware or software environments.

Generated datasets and `results/runs/` are ignored by Git, so cloning the
repository does not download existing checkpoints or figures.

## Tests

```powershell
python -m pytest -v
```