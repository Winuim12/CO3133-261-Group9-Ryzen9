# Assignment 2: Vietnamese Food Recognition

## Current status - M2 Draft

This project classifies **30 Vietnamese food categories** using a CNN trained
from scratch and a pretrained ResNet-50. The data audit, EDA, Docker training
pipeline, and two 20-epoch runs are complete. Both selected checkpoints were
reloaded and reproduced their saved validation Accuracy and Macro-F1.

**M2 packaging is in progress.** The assignment webpage, submission links,
checkpoint-access instructions, and complete AI disclosures still need review.
The held-out test set has not been evaluated. Full fine-tuning and the controlled
freeze-versus-full-fine-tuning comparison remain M3 work.

- [M2 draft report](report/report.md)
- [Environment and reproduction guide](docs/environment.md)
- [Validation comparison CSV](results/comparison.csv)
- [Training and validation curves](results/figures/m2_training_histories.png)
- [EDA notebook](notebooks/01_eda.ipynb)
- [Assignment webpage source and proposal record](../assignment2.html)
- [Experiment priorities and post-M2 follow-up](docs/experiment_plan.md)
- [Assignment AI-use log](docs/ai_usage.md)

The local assignment webpage records **Approved by Instructor**. Its approval
date and any conditions still need to be recorded by the group. The original
M1 proposal remains historical; use the audited split counts below for M2.

## Dataset and evaluation protocol

| Item | Current protocol |
| --- | --- |
| Dataset | 30VNFoods, Kaggle version 11 |
| Source | [Vietnamese Foods on Kaggle](https://www.kaggle.com/datasets/quandang/vietnamese-foods) |
| Task | Single-label image classification, 30 classes |
| Original / retained samples | 25,136 / 24,982 |
| Training / validation / test | 17,486 / 3,749 / 3,747 |
| Split | Approximately 70/15/15, class-stratified with duplicate groups kept together |
| Seed | 42 |
| Input / output | RGB tensors `[B, 3, 224, 224]` / raw logits `[B, 30]` |
| Primary metric | Macro-F1 over all 30 classes |
| Additional metrics | Accuracy, per-class precision/recall/F1/support and confusion matrix |
| Checkpoint selection | Highest validation Macro-F1; earlier epoch retained on ties |
| Test policy | Reserved for final evaluation after model choices are locked |

The saved audit records no unreadable images and 154 exclusions: 25 redundant
same-label exact copies, 79 exact copies with conflicting labels, and 50
near-duplicate images excluded after review. The final manifests have no
image-ID or recorded duplicate-group overlap across splits. Duplicate detection
reduces leakage risk but cannot establish that every transformed copy was found.

The proposal records **CC BY-NC-SA 4.0** on Kaggle and an **MIT LICENSE** file in
the downloaded package. Preserve both records and document their scope; the
package license is not assumed to replace the dataset license.

### Data placement

Obtain the same Kaggle dataset version and place images so manifest paths resolve:

```text
data/raw/archive/Images/
|-- Train/
|-- Test/
`-- Validate/
```

For example, `data/raw/archive/Images/Test/Banh beo/110.jpg` must exist.
The original directory names do not determine experiment membership: loaders
read the reviewed CSVs in `data/manifests/`. Use those manifests for reproduction;
do not regenerate splits simply to train or verify an existing run.

Images are excluded from Git. Cloning the repository does not download them.
See the [report's data section](report/report.md#problem-and-data-description)
for the cleaning procedure, EDA, and limitations.

## Verified M2 validation results

Both runs used 20 epochs, batch size 64, two data-loader workers, cross-entropy,
Adam, and weight decay 0. Learning rates were 0.001 for the scratch CNN and
0.0001 for frozen ResNet-50. No scheduler, early stopping, or mixed precision
was used. Training augmentation and deterministic validation transforms are
shared; normalization uses fixed ImageNet constants.

| Model | Best epoch | Validation Accuracy | Validation Macro-F1 | Total parameters | Trainable parameters | Elapsed time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scratch CNN | 19 | 25.10% | 0.2005 | 21,249,886 | 21,249,886 | 45m 15s |
| ResNet-50, frozen backbone | 20 | 72.45% | 0.7231 | 23,569,502 | 61,470 | 45m 44s |

The scratch CNN has 33 convolutional layers. ResNet-50 uses TorchVision
`ResNet50_Weights.IMAGENET1K_V2` and a new 30-class head. In frozen mode,
backbone parameters and BatchNorm running statistics remain fixed.

Elapsed time includes training, validation, data loading, and checkpoint saving
inside the epoch loop; initial setup and pretrained-weight downloading are
excluded. Raw measurements are 2,715.1479 and 2,744.0630 seconds, respectively.

These are **single-seed validation results**, not test scores. The configurations
differ in architecture, initialization, and learning rate, so this comparison
does not isolate one factor. Similar elapsed times do not establish a bottleneck;
profiling is deferred until M2 is finished.

![M2 training and validation histories](results/figures/m2_training_histories.png)

### Recorded runs

| Model | Run ID |
| --- | --- |
| Scratch CNN | `20261025T103059Z_a86f4102` |
| Frozen ResNet-50 | `20261025T112235Z_014bbc02` |

Both runs record training source commit
`5d106abab40a6720ebc5874b9a748f01356c965c` and identical manifest hashes.
Comparison and plotting tools are subsequent packaging work; the training commit
alone is not a claim that every later reporting script is present at that revision.

## Docker setup and tests

The exercised environment uses Python 3.11.13, PyTorch 2.7.1+cu128,
TorchVision 0.22.1+cu128, and CUDA runtime 12.8 on an NVIDIA RTX 5070 Ti
with approximately 16 GB VRAM. The base image is
`pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`.

Run the following from `assignment2`:

```powershell
docker compose build
docker compose run --rm assignment2 python -m pip check
docker compose run --rm assignment2 python -m pytest tests -v
```

The last user-reported full Docker CPU test suite passed **25 tests**. Explicitly
specifying `tests` avoids discovery through the mounted image directories.

GPU commands combine `compose.yaml` with `compose.gpu.yaml`. They require
compatible NVIDIA hardware/drivers and configured Docker GPU access; on Windows,
use Docker Desktop's WSL 2 backend. The `gpus: all` setting requires Compose 2.30
or later. Other GPU models and local virtual-environment execution have not been
verified for this project.

The base Compose service mounts original images read-only and manifests/results
with write access. A named volume preserves downloaded pretrained weights.
`shm_size: "1gb"` provides shared-memory capacity for data-loader workers.
Rebuild after changing source or dependencies copied into the image; changes to
mounted images/results or runtime Compose settings do not require an image rebuild.

The dependency files pin direct dependencies, not the complete transitive
environment. `requirements-cpu.txt` is populated; `requirements-gpu.txt` and
`.python-version` are still empty placeholders and are not local setup guides.
Use the [environment guide](docs/environment.md) for details and known limits.

## Train new runs

For a reported experiment, commit source/configuration changes first and rebuild
from that source. The commit argument is a manually supplied provenance label;
it does not automatically verify that the image matches the commit.

```powershell
$trainingCommit = git rev-parse HEAD
docker compose build
```

Scratch CNN:

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml run --rm assignment2 python -m experiments.run --config config/baseline_cnn.yaml --epochs 20 --code_commit $trainingCommit
```

Frozen pretrained ResNet-50:

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml run --rm assignment2 python -m experiments.run --config config/resnet50.yaml --experiment-config config/ablation_frozen.yaml --epochs 20 --code_commit $trainingCommit
```

Configuration precedence is shared `config/config.yaml`, then the model file
passed with `--config`, then optional `--experiment-config`. The `--epochs`
override is applied before the resolved configuration is saved. The commit flag
is spelled **`--code_commit`** with an underscore.

Each invocation starts a new model and unique run directory. Resume training and
early stopping are not implemented. Epoch progress prints after the training and
validation passes finish. Training does not automatically evaluate the test set.

```text
results/runs/<model>/<run-id>/
|-- config.json       # Resolved settings
|-- metadata.json     # Commit, class mapping, split hashes, environment, parameter counts
|-- best_model.pt     # Highest validation Macro-F1 checkpoint, including optimizer state
|-- history.json      # Per-epoch training and validation metrics, saved after completion
`-- summary.json      # Selected epoch and elapsed epoch-loop time, saved after completion
```

An interrupted run may lack completed history/summary files. Do not present it as
a completed experiment. Checkpoints and generated run directories are ignored by
Git, so a fresh clone does not contain the recorded runs below. Checkpoint access
and a portable copy of run evidence still need to be finalized for submission;
the commands above describe how to reconstruct new runs. Numerical results can
vary across environments.

## Verify saved runs and regenerate M2 outputs

For the existing local M2 artifacts:

```powershell
$baselineRun = "results/runs/baseline_cnn/20261025T103059Z_a86f4102"
$resnetRun = "results/runs/resnet50/20261025T112235Z_014bbc02"

docker compose -f compose.yaml -f compose.gpu.yaml run --rm assignment2 python -m experiments.verify_checkpoint --run-dir $baselineRun
docker compose -f compose.yaml -f compose.gpu.yaml run --rm assignment2 python -m experiments.verify_checkpoint --run-dir $resnetRun
```

Verification loads the run's saved configuration, checks manifest identity,
restores the checkpoint, and compares validation Accuracy and Macro-F1 against
the selected epoch. It writes `validation_reload.json`. It does not evaluate test
predictions. Both recorded runs passed this check.

After verification:

```powershell
docker compose run --rm assignment2 python -m experiments.compare --run-dirs $baselineRun $resnetRun
docker compose run --rm assignment2 python -m experiments.plot_curves --run-dirs $baselineRun $resnetRun
```

Outputs are [comparison.csv](results/comparison.csv) and
[m2_training_histories.png](results/figures/m2_training_histories.png).
For new training runs, replace the variables with the directories printed by
those runs. The comparison command requires successful verification records and
matching manifest hashes.

There is no `--evaluate-only` or `--split test` option in the current training
runner. `experiments/prepare_data.py` remains an empty placeholder; data audit
and splitting functions are in `src/data/` and used by the EDA workflow.
Inference-cost benchmarking in `src/evaluation/benchmark.py` is also pending.

## Project layout and ownership

```text
assignment2/
|-- config/                   # Shared, model and freeze/full experiment settings
|-- data/manifests/            # Audit, duplicate review, class mapping and split CSVs
|-- data/raw/                  # Downloaded images; ignored by Git
|-- src/data/                  # Audit, splitting, dataset, transforms and loaders
|-- src/models/                # Scratch CNN, ResNet-50 wrapper and model factory
|-- src/training/              # Epoch execution and best-checkpoint saving
|-- src/evaluation/            # Metrics/evaluator; further analysis and benchmarking pending
|-- src/utils/                 # Configuration, seeding and run metadata
|-- experiments/
|   |-- context.py            # Configuration, device and loader preparation
|   |-- run.py                # Training CLI and multi-epoch orchestration
|   |-- verify_checkpoint.py  # Saved-run validation reload check
|   |-- compare.py            # Verified-run comparison CSV
|   `-- plot_curves.py        # Saved-history figure export
|-- notebooks/01_eda.ipynb     # Data audit and exploratory analysis
|-- results/                  # Comparison CSV, exported figures and ignored run directories
|-- tests/                    # Data, model, checkpoint and integration checks
|-- docs/                     # Environment, experiment priorities and disclosure log
|-- report/report.md          # M2 draft report
|-- Dockerfile                # Runtime image and dependency installation
|-- compose.yaml              # Mounts, shared memory and base execution
`-- compose.gpu.yaml          # GPU access override
```

| Owner | Working responsibility |
| --- | --- |
| You | Orchestration, configurations, environment, run records, integration and comparison |
| Teammate A | Dataset audit/review, manifests, splitting, loaders and EDA |
| Teammate B | Model implementations, training/checkpoints, metrics and error analysis |

These are working allocations, not a claim of completed individual contributions.
All members review their work, contribute to reporting/presentation, and record
actual contributions and AI use. Preserve the one-small-task-at-a-time teaching
workflow in local `AGENTS.md` when working with the coding mentor.

## Remaining milestone work

Deadlines are from handbook Section 7.5, at **23:59 GMT+7**, unless overridden by
the LMS. The LMS has not been checked here.

| Milestone | Deadline | Weight within Assignment 2 | Scope |
| --- | --- | ---: | --- |
| M1 Proposal | 07 Oct 2026 | 15% | Dataset/task/split/metrics/baseline/compute proposal and approval record |
| M2 Draft | 28 Oct 2026 | 25% | EDA, baseline, pretrained model in progress, preliminary metrics |
| M3 Final | 11 Nov 2026 | 60% | Controlled experiment, error analysis, report, slides, video, assignment page and checkpoints |

### Finish M2 packaging

- [x] Audit/EDA artifacts and reviewed manifests available.
- [x] Scratch CNN and frozen ResNet-50 trained for 20 epochs.
- [x] Selected checkpoints verified against saved validation metrics.
- [x] Comparison CSV, training curves and draft report available locally.
- [x] Docker reproduction instructions documented.
- [ ] Update the assignment webpage's methodology, setup, results and discussion.
- [ ] Publish portable run evidence and checkpoint links or finalize reconstruction instructions.
- [ ] Fill `report/deliverables.md` and verify report/EDA/source/results links.
- [ ] Complete AI disclosures in the local log, root log, report and website.
- [ ] Add actual group/member attribution and approval date/conditions.
- [ ] Commit/publish the M2 package, check CI/public links, and submit through the LMS.

### After M2

- Run the controlled frozen-versus-full-fine-tuning comparison from the same
  pretrained weights and seeded head initialization. Keep shared factors fixed.
- Add a documented early-stopping extension: validation-loss non-improvement for
  five consecutive epochs, with a configured maximum budget; checkpoint selection
  remains validation Macro-F1. Do not relabel completed fixed-budget runs.
- Profile batch waiting, transfer and GPU computation before optimizing throughput.
  The similar observed run times and low reported GPU utilization do not prove
  a particular bottleneck.
- Complete qualitative/error analysis, inference-cost measurement and final
  test-set evaluation after model choices are locked.
- Finish final report, slides, video and required submission materials.

See [experiment_plan.md](docs/experiment_plan.md) for the deferred profiling task.

## AI assistance and verification

OpenAI Codex (GPT-6) assisted with project planning, incremental implementation
examples, debugging, experiment interpretation, and documentation drafts. The
student implemented code and supplied Docker test/training outputs; the assistant
also inspected local source, saved run JSONs, CSVs and exported figures. These
roles should be attributed accurately rather than claiming the assistant ran the
reported model-training experiments.

This README was updated on **26 Oct 2026** from the local M2 artifacts and source.
The [assignment log](docs/ai_usage.md) records the subsequent implementation and
writing assistance. Trần Mạnh Thắng (2353124) confirmed his Codex use and overall
final-verification responsibility. Claude Opus 5.5 entries for Hoang Kim Cuong
(2352145) and Phan Tan Loc (2352712) are user-reported drafts; those members must
confirm exact usage dates, prompts, student edits and personal checks. Reconcile
these details with [the repository log](../AI_USAGE.md), report and website before
submission. Historical website/EDA entries are retained for group review.
