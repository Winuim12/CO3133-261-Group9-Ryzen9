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
handover. The model, data pipeline, training loop, and metric implementations
are still assigned work; no training has been run. See:

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
