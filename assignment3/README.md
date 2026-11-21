# Assignment 3: Multimodal Meme Sentiment Classification

Proposed task: predict one of five original sentiment labels from a meme image
and its associated text. Instructor approval is **pending**; no main model
implementation or experiment results are claimed.

- [M1 dataset proposal on the assignment page](../assignment3.html#dataset-proposal)
- [Supporting proposal working copy](docs/dataset_proposal.md)
- [Mentoring and project instructions](AGENTS.md)
- [Assignment 3 AI usage log](AI_USAGE.md)

## Team bootstrap (14 November 2026)

Shared environment, configuration, Docker, and CI scaffolding is prepared for
handover. The data pipeline, encoders, trainer, fusion models, metrics, and
experiment entry points are now implemented (see the training workflow below);
only debugging runs have been executed so far. See:

- [Local CPU/GPU and Docker setup](docs/environment.md)
- [Team ownership and batch/model interfaces](docs/task_allocation.md)
- [What each member does first](docs/handover.md)

Local CPU quick start from `assignment3` (Python 3.12.10):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
.\.venv\Scripts\python.exe -m experiments.check_environment --device cpu --smoke-test
.\.venv\Scripts\python.exe -m pytest tests -v
```

If `.venv` already exists, skip creating it. For the RTX 5070 Ti, select
`requirements-gpu.txt` instead and run the check with `--device cuda`.

Docker CPU quick start:

```powershell
docker compose build
docker compose run --rm assignment3
docker compose run --rm assignment3 python -m pytest tests -v
```

For Docker GPU, add `-f compose.yaml -f compose.gpu.yaml` to each Compose
command. Rebuild after source/configuration changes. The default GPU command
requires working CUDA rather than silently falling back to CPU.

CI workflow: `../.github/workflows/assignment3-ci.yaml`; check name:
`assignment3-tests`. CI uses CPU checks and no raw dataset. Its GitHub execution
will only be verified after the workflow is pushed and runs.

`docs/superpowers/` is local planning material excluded from future tracking.
The team-facing documents above remain versioned. Raw data, environments,
caches, generated runs, and checkpoint files are also excluded.

## Training and evaluation workflow

Run from `assignment3` after downloading the dataset to `data/raw/memotion/`.
Manifests are versioned; do not regenerate them on the training machine.

```powershell
python -m src.data.tokenizer                       # training-only vocabulary
python -m experiments.check_environment --device cuda --smoke-test
python -m pytest tests -v

# One run per configuration (repeat with --seed 43, 44 for seed variance)
python -m experiments.train --config config/image_only.yaml --device cuda
python -m experiments.train --config config/text_only.yaml --device cuda
python -m experiments.train --config config/concat.yaml --device cuda
python -m experiments.train --config config/gated.yaml --device cuda
python -m experiments.train --config config/fixed_average.yaml --device cuda

python -m experiments.compare                      # validation comparison
python -m experiments.evaluate --run <run-id> --split test   # locked configs only
python -m experiments.compare --split test
```

Each run writes `results/runs/<model>/<run-id>/` (resolved `config.json`,
`run_metadata.json` with data hashes/git commit/hardware, `training.json` with
history, runtime and peak VRAM, validation metrics, predictions, figures) and
`checkpoints/<model>/<run-id>/best_model.pt`, selected by validation Macro-F1.
`experiments.evaluate` rebuilds models from the run's saved configuration and
refuses to run if manifests or vocabulary changed since training.
Comparison tables and figures go to `results/comparison/`.

`--limit-train`, `--limit-val`, and `--no-pretrained` exist for quick smoke
runs only; such runs are labelled `debug` and excluded from comparisons.

## Downloaded data

```text
data/raw/memotion/
|-- labels.csv
`-- images/images/
```

The source is [6992 Meme Images Dataset with Labels, Kaggle version 1](https://www.kaggle.com/datasets/hammadjavaid/6992-labeled-meme-images-dataset).
The existing `scripts/download_dataset.py` downloads and extracts the archive:

```powershell
python -m pip install requests
python scripts/download_dataset.py
```

The script has no resume support and should not be rerun over an active
download. These are download instructions, not training/reproduction commands.

Local inspection on 9 November 2026 verified 6,992 rows and image files, all five
sentiment classes, and OCR/corrected text columns. One corrupt image and five
blank corrected-text rows leave 6,986 provisional pairs before duplicate and
template review. Details and limitations are in the proposal. Raw data is
excluded from Git.

Before submission, review the proposal, source license/usage terms, group
details, proposed responsibilities, and AI disclosure ownership. Record the
instructor's approval and conditions before main implementation/experiments.
