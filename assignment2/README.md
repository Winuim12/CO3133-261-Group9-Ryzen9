# Assignment 2: Vietnamese Food Recognition

## Purpose and current status

Build a reproducible image-classification pipeline for **30VNFoods**, comparing a
small CNN trained from scratch with at least one pretrained model and one
controlled experiment. Produce quantitative results, qualitative examples,
error analysis, and compute-cost measurements.

This README is the **project skeleton and creation checklist**. The paths and
commands below describe planned work. The scaffold files and directories now
exist, but Python modules, configuration and dependency files, and Docker files
are unimplemented placeholders. The two notebooks are valid empty notebooks;
`class_to_idx.json` is an empty object awaiting the audited 30-class mapping.
No experiments have been run and empty CSV files are not real manifests/results.

Sources: `handbook-ene.pdf` (especially Sections 3-5, 7.5, and 15-23) and
`../assignment2.html` (Dataset Proposal). The HTML currently records **Pending
Instructor Approval**. Record the instructor's actual decision and conditions
before starting full implementation or main experiments, as required by
handbook Sections 15.1 and 18. Preparing this outline does not establish approval.

## 1. Dataset and task contract

These details are recorded in the HTML proposal, not independently verified by
this skeleton:

| Item | Proposed protocol |
| --- | --- |
| Track | Image classification |
| Dataset | 30VNFoods, Kaggle version 11 |
| Source | https://www.kaggle.com/datasets/quandang/vietnamese-foods |
| Proposal audit date | 03 Oct 2026 |
| Original size | 25,136 images, 30 classes |
| Labels | One food label per image, derived from its class folder |
| Input | RGB image, transformed to 224 x 224 |
| Output | 30 logits; predicted class is the argmax |
| Proposed split | Approximately 70% train / 15% validation / 15% test, class-stratified with duplicate groups kept together |
| Seed | 42 |
| Primary metric | Macro-F1 over all 30 classes |
| Additional metrics | Accuracy, per-class precision/recall/F1, 30 x 30 confusion matrix |
| Checkpoint selection | Highest validation Macro-F1 |
| Baseline | Small randomly initialized CNN; cross-entropy, Adam, up to 20 epochs |
| Selected pretrained backbone | ResNet-50 with TorchVision `ResNet50_Weights.IMAGENET1K_V2`; replace the classification head with 30 outputs |
| Required controlled experiment | Frozen ResNet-50 backbone vs. full ResNet-50 fine-tuning |

ResNet-50 and the freeze/full-fine-tune comparison are the group's selected
implementation plan. The earlier HTML proposal lists multiple candidates;
preserve that proposal history and record the final selection without implying
that instructor approval or experiments have already occurred. Other backbones
are optional extensions, not required work for this plan.

The proposal lists CC BY-NC-SA 4.0 on Kaggle and an MIT LICENSE file in the
downloaded package. Preserve both records and document their scope; do not
assume the package's MIT file replaces the dataset license.

The proposed split counts are **17,523 / 3,753 / 3,749**, totaling **25,025**.
This is 111 fewer than the original count. The HTML says counts are provisional
pending duplicate review. The audit must explain every removal or exclusion and
recompute final counts; do not hard-code these numbers as verified results.

## 2. Proposed folder and file skeleton

Use a small Python project with configuration files and notebooks for analysis.
This follows the existing Assignment 1 separation of data, models, training,
evaluation, and utilities. A notebook-only approach would make reproduction
harder; copying the entire Assignment 1 project would bring unnecessary models
and assumptions into this task.

```text
assignment2/
|-- README.md                         # This plan; later add verified reproduction commands
|-- AGENTS.md                         # Advisory teaching workflow and Assignment 2 constraints
|-- handbook-ene.pdf                  # Local reference handbook
|-- .gitignore                        # Ignore images, environments, caches, generated runs
|-- .python-version                   # Selected Python version for local and container use
|-- requirements.txt                  # Shared pinned application dependencies
|-- requirements-cpu.txt              # Shared dependencies + compatible CPU PyTorch/TorchVision
|-- requirements-gpu.txt              # Shared dependencies + compatible CUDA PyTorch/TorchVision
|-- Dockerfile                        # Reproducible runtime image with a pinned base image
|-- .dockerignore                     # Exclude datasets, results, environments, caches and credentials
|-- compose.yaml                      # Container working directory, mounts and default CPU execution
|-- compose.gpu.yaml                  # Optional GPU access override for the same runtime image
|-- config/
|   |-- config.yaml                   # Data paths, seed, image size, shared run settings
|   |-- baseline_cnn.yaml             # Scratch CNN architecture and training settings
|   |-- resnet50.yaml                  # Shared ResNet-50 weights, head and training settings
|   |-- ablation_frozen.yaml          # Frozen-backbone controlled experiment
|   `-- ablation_finetune.yaml        # Full-fine-tuning controlled experiment
|-- data/
|   |-- raw/                          # Original downloaded images; keep outside Git
|   `-- manifests/
|       |-- audit.csv                 # Image inventory and cleaning decisions
|       |-- duplicate_review.csv      # Candidate groups and reviewed decisions
|       |-- class_to_idx.json         # One stable mapping for all models and splits
|       |-- train.csv                 # Final training manifest
|       |-- val.csv                   # Final validation manifest
|       `-- test.csv                  # Final held-out test manifest
|-- src/
|   |-- __init__.py
|   |-- data/
|   |   |-- __init__.py
|   |   |-- audit.py                  # Decode checks, labels, dimensions, hashes, duplicate candidates
|   |   |-- split.py                  # Group-aware, approximately stratified splitting
|   |   |-- dataset.py                # Dataset reads paths and labels from a manifest
|   |   |-- transforms.py             # Train augmentation and deterministic evaluation transforms
|   |   `-- dataloader.py             # Train/validation/test loaders
|   |-- models/
|   |   |-- __init__.py
|   |   |-- baseline_cnn.py           # Small CNN designed by the group
|   |   |-- resnet50.py                # ImageNet V2 backbone, 30-class head, freeze/full modes
|   |   `-- factory.py                # Build a model from configuration
|   |-- training/
|   |   |-- __init__.py
|   |   |-- trainer.py                # Train/validate epochs, loss, optimizer, scheduler
|   |   `-- checkpoint.py             # Save the best validation Macro-F1 checkpoint
|   |-- evaluation/
|   |   |-- __init__.py
|   |   |-- metrics.py                # Accuracy, Macro-F1, per-class scores, confusion matrix
|   |   |-- evaluator.py              # Load saved run settings and export predictions
|   |   |-- error_analysis.py         # Correct, difficult, and failed prediction examples
|   |   `-- benchmark.py              # Parameter count, training time, inference cost, memory
|   `-- utils/
|       |-- __init__.py
|       |-- config.py                 # Load and validate configuration
|       |-- seed.py                   # Python, NumPy, framework and worker seeds
|       `-- run_info.py               # Run IDs, hardware/software, commit and split identity
|-- experiments/
|   |-- __init__.py
|   |-- context.py                    # Resolve configuration, seeds, device and data dependencies
|   |-- prepare_data.py               # Run audit and create reviewed split manifests
|   |-- run.py                        # Orchestrate training or saved-run evaluation
|   `-- compare.py                    # Aggregate completed runs into comparison tables
|-- notebooks/
|   |-- 01_eda.ipynb                  # Dataset quality and class/input distributions
|   `-- 02_results_analysis.ipynb     # Curves, comparisons, ablation and failure analysis
|-- results/
|   |-- runs/                         # Per-run configs, logs, checkpoints and predictions
|   |-- figures/                      # Exported figures for report and website
|   `-- comparison.csv                # Summary of real completed experiments
|-- tests/
|   |-- test_split.py                 # No image/group overlap; valid labels and class coverage
|   |-- test_models.py                # RGB-to-30-logit contract and freeze/full behavior
|   |-- test_metrics.py               # Metrics checked on known predictions
|   |-- test_checkpoint.py            # Best checkpoint selection and reload correctness
|   `-- test_run.py                   # Small real integration: save run and reload for evaluation
|-- docs/
|   |-- dataset_proposal.md           # Proposal snapshot, approval record and conditions
|   |-- experiment_plan.md            # Hypotheses, factors and selection criteria
|   |-- environment.md                # Local/Docker setup, CPU/GPU prerequisites and verification
|   `-- ai_usage.md                   # Assignment-local log; reconcile with ../AI_USAGE.md
`-- report/
    |-- report.md                     # Draft following the handbook's three report parts
    |-- slides_outline.md             # Presentation plan and each member's speaking part
    `-- deliverables.md               # Final report/slides/video/checkpoint links
```

The tree describes the current scaffold and each file's intended role. Empty
runtime folders are preserved with `.gitkeep`. Generated
manifests should come from the actual audit, not invented placeholder samples.
Keep downloaded images and large checkpoints out of Git; publish checkpoint
access or reconstruction instructions separately.

### Local and Docker execution plan

Support both **local execution in a virtual environment** and **Docker execution**
through the same `python -m experiments...` entry points and YAML configurations.
The environment files above exist as empty placeholders; no Docker image,
working Compose configuration, dependency pins or local environment has been
implemented or verified yet.

| File | Purpose |
| --- | --- |
| `.python-version` | Record the Python version selected and verified for both execution paths |
| `requirements.txt` | Pin shared application dependencies; keep one source for local and Docker installs |
| `requirements-cpu.txt` / `requirements-gpu.txt` | Include the shared requirements and pin compatible PyTorch/TorchVision variants; use matching framework versions where supported |
| `Dockerfile` | Build a runtime using the selected Python/dependencies and an explicitly pinned base image; document its CUDA runtime if included |
| `.dockerignore` | Keep raw images, generated runs, `.venv`, Git internals, caches and secrets out of the build context |
| `compose.yaml` | Define the working directory and mounts; run the shared image on CPU by default |
| `compose.gpu.yaml` | Add GPU access for the same image when the host supports it |
| `docs/environment.md` | Explain setup for local CPU/GPU and Docker CPU/GPU, exact verified commands and known platform differences |

Environment rules for the later implementation:

- Choose actual Python, PyTorch, TorchVision and CUDA versions together after
  verifying compatibility with the RTX 5070 Ti and host. Do not invent pins now.
- Use project-relative paths resolved from the project root and `pathlib` so
  code works on local Windows and inside the container. Avoid hard-coded `C:\...`
  paths in shared experiment configurations.
- Mount original `data/raw/` read-only. Mount `data/manifests/` and `results/`
  separately with write access for preparation and output persistence. Persist
  pretrained-weight downloads in a separate cache volume.
- Keep experiment logic independent of Docker. The same saved manifests, class
  mapping, resolved configurations and checkpoints must be usable in both paths.
- Treat GPU access as an explicit environment choice. Document and verify host
  driver and container GPU prerequisites; the image does not install the host driver.
- A shared GPU-capable image may execute on CPU; local CPU installations can use
  CPU framework packages. Record which variant was actually used in each run.
- Record Python/framework versions, device/driver/runtime information, host OS,
  and container image identity for container runs. Shared versions improve
  consistency but do not guarantee identical results across operating systems,
  devices or nondeterministic operations.
- Verify a small local and container run, checkpoint reload, persistent outputs,
  and equivalent preprocessing/metrics before any long experiment. Compare
  numerical outputs with a stated tolerance rather than assuming bitwise equality.

**Ownership:** you own the environment files and execution documentation as part
of orchestration; both teammates verify that their modules work through the
shared commands. Implement the local environment first, then wrap the working
pipeline in Docker using the same incremental teaching workflow.

## 3. Create these pieces in order

### Step 1: Record the protocol and environment

- [ ] Save the proposal and actual approval decision in `docs/dataset_proposal.md`.
- [ ] Record approval date, conditions, and whether each condition is satisfied.
- [x] Select ResNet-50 with ImageNet V2 weights and freeze/full-fine-tune comparison.
- [ ] Validate the available hardware with a short trial; choose a batch size
      that fits both controlled conditions. RTX 5070 Ti is user-reported, not
      independently inspected; record the actual device and measured timings.
- [ ] Create shared and model configurations; keep all 30 classes and seed 42.
- [ ] Select and record the Python/framework versions and CPU/GPU environment.
- [ ] Define shared dependency pins and CPU/GPU variants for local and Docker use.
- [ ] Document the container base image, mounts, GPU prerequisites and persistent outputs.
- [ ] Verify the local and container execution paths with a small run before main experiments.
- [ ] Add dependency versions, local ignore rules, and the AI-use record.

PyTorch is a proposed implementation choice because Assignment 1 already uses
it. Reuse suitable utilities only after reviewing their assumptions. In
particular, replace Assignment 1's validation-loss checkpoint criterion with
validation Macro-F1 and support RGB images with 30 output classes.

### Step 2: Audit and finalize the dataset

- [ ] Record the actual downloaded version, archive identity, license records,
      storage size, image formats, and class names.
- [ ] Inventory readable/unreadable images, missing labels, image dimensions,
      class frequencies, and unusual samples.
- [ ] Detect exact duplicates; keep one same-label copy and quarantine
      conflicting-label copies for review.
- [ ] Flag near-duplicate candidates and review groups/source identities.
- [ ] Keep retained images in a duplicate/source group in one split.
- [ ] Create approximately class-stratified 70/15/15 manifests with seed 42.
      Group integrity takes priority over exact percentages; record deviations.
- [ ] Verify no path, exact duplicate, or retained duplicate group crosses splits;
      verify all 30 classes have meaningful train/validation/test representation.
- [ ] Explain the original-to-cleaned count difference and record exclusions.
- [ ] Confirm the final training set meets the handbook's 5,000-sample threshold.

Suggested audit fields: `image_id`, relative `path`, `class_name`, `class_idx`,
`width`, `height`, `format`, `readable`, `sha256`, `group_id`, `decision`, and
`reason`. Split manifests should retain stable IDs, labels, and group IDs.
Record the duplicate detection method, thresholds, review decisions, and split
manifest hashes so that the split can be reproduced.

### Step 3: Build EDA and the input pipeline

- [ ] Plot class frequencies before cleaning and for the final splits.
- [ ] Show width/height/aspect-ratio distributions and imbalance statistics.
- [ ] Show example images for every class and problematic annotations/images.
- [ ] Document cleaning, resizing/cropping, augmentation, and tensor shapes.
- [ ] Visualize a preprocessed batch with its labels.
- [ ] Use training-only augmentation and deterministic validation/test transforms.
- [ ] If computing dataset normalization, fit it on training images only. If
      using pretrained-weight normalization constants, record their source and
      the exact transforms instead of deriving statistics from held-out data.
- [ ] Explain dataset bias, limits, and remaining leakage risks.

### Step 4: Train the scratch CNN baseline

- [ ] Implement a small CNN with a 30-class head and randomly initialized weights.
- [ ] Use cross-entropy, Adam, and up to 20 epochs as proposed.
- [ ] Record architecture, total/trainable parameter counts, learning rate,
      batch size, scheduler, regularization, and stopping rules.
- [ ] Log train/validation loss, Accuracy, and Macro-F1 each epoch.
- [ ] Save the highest-validation-Macro-F1 checkpoint; define a deterministic
      tie rule, such as retaining the earlier checkpoint.
- [ ] Record training time, hardware, software, peak memory where available,
      seed, configuration, split identity, and commit.

### Step 5: Add the pretrained model and controlled experiment

Use **ResNet-50 with `ResNet50_Weights.IMAGENET1K_V2`** as the pretrained
backbone. Replace `model.fc` with a linear layer mapping its input features to
30 raw logits. Do not add Softmax before cross-entropy. Use and document the
weight-associated normalization and deterministic evaluation preprocessing;
the model input remains 224 x 224 RGB. Record training augmentation separately
and use the same transforms for both controlled conditions.

In the frozen condition, train only the new head. In the full-fine-tuning
condition, train the head and all backbone layers. Document trainable parameters,
optimizer, learning rates, and any staged training. The initial controlled
comparison uses these two direct conditions; a separately tuned or staged recipe
is an optional follow-up. Validation data guides choices; test data does not.

Recommended initial experiment matrix:

| ID | Configuration | Role |
| --- | --- | --- |
| E0 | Small CNN from scratch | Required baseline |
| E1 | ResNet-50 ImageNet V2 backbone frozen; train the new 30-class head | Pretrained comparison and ablation condition A |
| E2 | Same ResNet-50 weights and head initialization; train all layers | Full fine-tuning and ablation condition B |

For E1 versus E2:

- **Hypothesis:** unfreezing the backbone improves validation Macro-F1 for this task.
- **Changed factor:** whether backbone parameters are trainable.
- **Fixed factors:** split manifests, preprocessing/augmentation, initialization
  seed, starting weights, batch size, epoch budget, loss, optimizer settings,
  learning rate, scheduler, and checkpoint criterion.
- **Measurements:** validation Accuracy/Macro-F1, training time, parameter counts,
  inference cost, and memory. Report final test metrics only after choices are locked.
- **Decision criterion:** compare best validation Macro-F1 under the fixed budget,
  then discuss the measured compute trade-off. Report the actual outcome even if
  full fine-tuning does not improve performance.

For the frozen condition, explicitly define backbone evaluation mode and keep
BatchNorm running statistics fixed. Otherwise a supposedly frozen model can
still change its backbone state. If learning rates or schedules are tuned
differently, report that as a separate comparison rather than claiming that only
the freeze setting changed. Repeat with additional seeds when feasible and
report mean/std; a single-seed result cannot establish run-to-run stability.

### Step 6: Evaluate and analyze completed runs

- [ ] Lock configurations and checkpoint choices using validation results.
- [ ] Evaluate selected models on the same held-out test manifest.
- [ ] Export Accuracy, Macro-F1, per-class precision/recall/F1/support, and a
      confusion matrix with the saved 30-class mapping.
- [ ] Export per-image ground truth, prediction, and score/confidence for analysis.
- [ ] Plot training curves and compare performance, parameters, and compute.
- [ ] Measure inference with documented device, batch size, input size, warm-up,
      number of repeats, GPU synchronization if applicable, and preprocessing scope.
- [ ] Show ordinary correct cases, difficult correct cases, and failures.
- [ ] Group errors by observed causes, such as similar dishes, background clutter,
      occlusion, lighting, or suspected label issues; include frequency and examples.
- [ ] State limitations and improvements supported by the actual evidence.

Each run should save its resolved configuration, class mapping, split hashes,
seed, commit, dependency/hardware information, checkpoint criterion, training
history, and checkpoint. Evaluation must load the saved run configuration
rather than silently using edited current YAML files.

## 4. Planned command interface

**These commands are an interface proposal and are not runnable yet.** Once
implemented and verified, replace this section with exact working commands,
including installation and dataset placement instructions.

Run from the `assignment2` directory:

```powershell
# Audit images; review duplicate candidates before finalizing manifests
python -m experiments.prepare_data --config config/config.yaml --audit-only
python -m experiments.prepare_data --config config/config.yaml --finalize-splits

# Train the baseline and the two controlled conditions
python -m experiments.run --config config/baseline_cnn.yaml
python -m experiments.run --config config/ablation_frozen.yaml
python -m experiments.run --config config/ablation_finetune.yaml

# Review validation predictions during development
python -m experiments.run --evaluate-only --run-dir results/runs/<run-id> --split val

# Evaluate test data after model selection is complete
python -m experiments.run --evaluate-only --run-dir results/runs/<run-id> --split test
python -m experiments.compare --runs-dir results/runs
```

Split finalization should require reviewed decisions for unresolved conflicts,
and report any missing review records. Each training invocation should create a
new run directory rather than overwrite earlier results.

`config/config.yaml` contains shared defaults. `config/resnet50.yaml` contains
common backbone settings. The two ablation files inherit those same settings
and change the freeze setting only. Document the merge order as shared defaults,
model defaults, then experiment overrides, and save the fully resolved result
with each run. These inheritance and CLI behaviors still need implementation.

## 4A. Work division for three people

Names are intentionally left as **You**, **Teammate A**, and **Teammate B** until
the group assigns actual names. This is a proposed ownership split, not a record
of completed contributions. Every member writes code, reviews another member's
work, contributes to the report, and presents their part in the video.

| Owner | Main responsibility | Owned modules/artifacts | Report and presentation coverage |
| --- | --- | --- | --- |
| **You: orchestration and integration** | Connect the modules into reproducible experiments; run and compare E0/E1/E2 | `experiments/context.py`, `run.py`, `prepare_data.py`, `compare.py`; `src/utils/`; `src/evaluation/benchmark.py`; `config/`; `tests/test_run.py`; README and run provenance; `notebooks/02_results_analysis.ipynb` | Pipeline flow, configuration, reproduction, experiment comparison, compute-cost analysis, deliverable integration |
| **Teammate A: data and EDA** | Audit images, review duplicates, finalize splits, prepare loaders and transforms | `src/data/`; `data/manifests/`; `notebooks/01_eda.ipynb`; `tests/test_split.py`; `docs/dataset_proposal.md` | Problem/data description, license/version records, EDA, preprocessing, leakage prevention and data limitations |
| **Teammate B: models, training and evaluation** | Implement scratch CNN and ResNet-50 modes; train/validate correctly and compute metrics | `src/models/`; `src/training/`; `src/evaluation/metrics.py`, `evaluator.py`, `error_analysis.py`; `tests/test_models.py`, `test_metrics.py`, `test_checkpoint.py` | Architectures, fine-tuning, checkpoint selection, quantitative/qualitative evaluation and error analysis |

You coordinate `docs/experiment_plan.md`, environment/dependency recording, and
submission links with input from both teammates. Teammate A reviews data claims
and examples; Teammate B reviews model and metric claims. Each member records
their own AI usage; you consolidate the records and report/slides without taking
ownership credit for other members' work.

Your environment ownership includes `.python-version`, dependency files,
`Dockerfile`, `.dockerignore`, Compose files and `docs/environment.md`. These
exist as placeholders; documenting them does not authorize installing dependencies,
building images or running containers automatically.

### Your orchestrator modules

Follow the separation used by Assignment 1's `experiments/context.py` and
`experiments/run.py`, adapted to RGB manifests and validation Macro-F1:

1. **`context.py`:** resolve configurations and paths, set seeds before building
   loaders/models, choose the device, obtain loaders and the class mapping from
   Teammate A, and expose the resolved context to the runner. Evaluation context
   loads saved settings and manifests rather than re-splitting the dataset.
2. **`run.py`:** parse arguments, create a unique run directory, call Teammate B's
   model factory and trainer, save history/configuration/provenance, and route
   explicit evaluation requests through the saved checkpoint and evaluator.
   Keep epoch loops, model architecture, and metric formulas in their modules.
3. **`prepare_data.py`:** provide the CLI around Teammate A's audit/split functions;
   surface unresolved review decisions rather than duplicate the audit logic.
4. **`compare.py`:** collect metrics from completed run records, identify E0/E1/E2
   and their configurations, and export a comparison table without rerunning or
   inventing experiments. Keep validation and test columns clearly distinguished.
5. **`run_info.py` and `benchmark.py`:** record run identity and reproducibility
   metadata, then measure training/inference cost using a documented protocol.

### Agree on these interfaces before implementing modules

| Boundary | Shared contract |
| --- | --- |
| Data to model/trainer | Image tensors `[B, 3, 224, 224]`; integer labels `[B]` in `0..29`; one saved class mapping and reviewed manifest set |
| Model to trainer | Raw logits `[B, 30]`; model factory receives configuration; freeze/full mode is explicit |
| Trainer to orchestrator | Epoch history, best checkpoint path/epoch/validation Macro-F1, elapsed training time; no automatic test-based selection |
| Checkpoint to evaluator | Model state, resolved settings, class mapping and split identity sufficient to rebuild the exact run |
| Evaluator to analysis | Accuracy/Macro-F1, per-class scores/support, confusion matrix, per-image predictions linked to stable image IDs |
| Run record to comparison | Run ID, E0/E1/E2 condition, commit, seed, split hashes, weight version, parameter counts, checkpoint selection and compute measurements |

### Handoff and integration order

1. All three agree on manifest/configuration/run-output contracts above.
2. Teammate A provides a reviewed small image batch, class mapping and loaders;
   Teammate B checks both model output shapes and freeze/full behavior on it.
3. You connect the modules and verify one short train/validation run and checkpoint
   reload. Synthetic checks can validate interfaces without claiming dataset results.
4. After instructor approval and final split review, you run E0/E1/E2 with the
   agreed settings. Teammates review data integrity and training behavior.
5. Lock model choices using validation, evaluate the selected checkpoints on test,
   and jointly review comparisons, examples and error explanations.

**Your first learning task:** define the fields and dependencies for
`experiments/context.py` in plain language. Then implement configuration loading
as one small task before connecting device/seed setup and the data loaders.
The mentor workflow in `AGENTS.md` guides each task; this README does not
authorize automatic implementation of the entire pipeline.

## 5. Report, website, and submission skeleton

| Handbook requirement | Planned artifact |
| --- | --- |
| Dataset proposal and approval record (Section 18) | `docs/dataset_proposal.md`; proposal section in `../assignment2.html` |
| Part 1: problem and data (Section 3.1) | Motivation, task contract, sources/licenses, audit, EDA, splits, preprocessing |
| Part 2: methodology (Section 3.2) | Pipeline diagram, architectures, library vs. group contributions, training settings, experiment plan |
| Part 3: results (Section 3.3) | Curves, real metric tables, compute cost, qualitative results, ablation, error analysis, limitations |
| Reproduction (Section 4.2) | Verified install/data/train/evaluate commands, configs, seeds, dependency versions, hardware, run provenance |
| Assignment page (Section 2.2) | Fill existing `../assignment2.html` sections with actual evidence and deliverable links |
| Checkpoint access (Section 4.1) | Download links or verified reconstruction instructions |
| Report and slides (Section 4.1) | Final report and presentation, linked from website |
| Video (Section 2.3) | Public or Unlisted YouTube video; every member presents/defends part |
| AI disclosure (Section 5) | Root `../AI_USAGE.md`, landing page, assignment page, and report; reconcile assignment-local log |
| Final submission | Verified deliverable links and LMS submission |

Use the handbook's video title format:
`CO3133-Semester-261 – Group 09 – Assignment 2`.

Update the existing website after artifacts are ready. Its current statement
that Assignment 2 has used no AI for content/code/experiments/results needs to
be revised to disclose this planning assistance. Do not publish invented
metrics, unrun experiments, fabricated approvals, or placeholder links as final
deliverables.

## 6. Milestone checklist

Dates below are from handbook Section 7.5; deadlines are **23:59 GMT+7**, unless
the LMS states otherwise. No live LMS verification has been performed.

| Milestone | Handbook deadline | Weight within A2 | Minimum scope |
| --- | --- | --- | --- |
| M1 Proposal | 07 Oct 2026 | 15% | Proposal: task, dataset, split, metrics, baseline, compute estimate; record approval |
| M2 Draft | 28 Oct 2026 | 25% | EDA, simple baseline, pretrained model in progress, preliminary metrics |
| M3 Final | 11 Nov 2026 | 60% | Controlled experiment, error analysis, report, slides, video, assignment page, checkpoints |

**First implementation target:** finalize the reviewed dataset manifests, EDA,
and scratch CNN train/validation pipeline. Then add the ResNet-50 backbone,
the freeze/full-fine-tune experiment, and final analysis.

## 7. AI assistance record for this outline

- **Tool/model:** OpenAI Codex, GPT-6.
- **Date/stage:** 13 Oct 2026; project structure and implementation planning.
- **Purpose:** map the handbook and existing 30VNFoods proposal into a project skeleton;
  record the selected ResNet-50 experiment, divide module ownership among three members,
  and plan consistent local/Docker execution.
- **Prompt summary:** asked for a skeleton describing what to create based on the handbook and `assignment2.html`.
- **Affected artifacts:** `assignment2/README.md`, `assignment2/AGENTS.md`.
- **Verification performed by the assistant:** read handbook requirements and
  extracted the local HTML proposal; checked the existing Assignment 1 structure
  and the arithmetic of the proposed split counts; read Assignment 1's mentor
  instructions and orchestration modules to adapt the teaching method and roles.
  No dataset or training run
  was independently verified.
- **Student user and responsible final reviewer:** must be filled with the actual
  member names by the group; they are not inferred from the chat.
- **Student editing/verification:** pending group review against the cited local
  sources and eventual real implementation/runs.

Copy or reconcile this entry into the repository-level AI log and required
website/report disclosures when those artifacts are updated.
