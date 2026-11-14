# Assignment 3 — Task Allocation

## Project goal

Predict one of five sentiment classes from a meme image and its corrected text.

Required comparisons:
1. Image only.
2. Text only.
3. Simple concatenation fusion.
4. Improved gated fusion.
5. Fixed-average fusion as the required ablation.

The project must run locally and in Docker using the same source code,
configuration files, dataset manifests, and evaluation rules.

## Current status

Instructor approval: pending confirmation.

Shared bootstrap was prepared on 14 November 2026: environment requirements,
CPU/GPU Docker targets, shared configurations, environment checks, and CI.
Read [handover.md](handover.md) before filling assigned placeholders. Training,
data/model implementations, and final manifests are not supplied by bootstrap.

Preliminary inspection and planning may continue. Main implementation and
experiments begin only after approval and any conditions are recorded.

## Team ownership

| Member | Main responsibility | Owned files |
| --- | --- | --- |
| Member 1: Cuong | Data preparation, paired loading, and EDA | `scripts/prepare_splits.py`, `src/data/`, `data/manifests/`, `notebooks/01_eda.ipynb`, data tests |
| Member 2: You | Encoders, unimodal classifiers, and shared training loop | `src/models/image_encoder.py`, `text_encoder.py`, `unimodal.py`, `src/engine.py`, encoder/training tests |
| Member 3: Loc | Docker/local setup, fusion, and experiment integration | Environment files, `src/models/fusion.py`, `src/metrics.py`, `src/reproducibility.py`, `experiments/`, `config/`, integration tests |

Ownership means coordinating changes and reviewing contributions.
It does not prevent another member from helping.

## Shared contracts — agree before coding

### Label mapping

Store the canonical mapping in `config/base.yaml`:

- 0: very_negative
- 1: negative
- 2: neutral
- 3: positive
- 4: very_positive

Save the mapping in checkpoints and run metadata.
Do not define a different order in individual scripts.

### Dataset batch

Every loader produces:

| Key | Format |
| --- | --- |
| `image` | Float tensor `[B, 3, 224, 224]` |
| `text_ids` | Integer tensor `[B, L]`, padded |
| `text_lengths` | Integer tensor `[B]`, excluding padding |
| `label` | Integer tensor `[B]`, values 0–4 |
| `image_name` | List of filenames |

The primary text input is `text_corrected`.
Build vocabulary from training data only.

### Model interface

Every classifier supports:

```python
logits = model(batch)
```

Output: raw logits `[B, 5]`.

Unimodal classifiers ignore the unused input.
Do not apply Softmax before cross-entropy.

Encoders also expose:

```python
features = encoder.encode(...)
```

Image features: `[B, 512]` for the proposed ResNet-18.
Text features: `[B, text_feature_dim]`, recorded in configuration.

Fusion projects both representations to the configured shared dimension.

## Member 1 — Data and EDA

### Tasks

- Finish image/text validity and duplicate review.
- Preserve original data; record exclusions and reasons.
- Keep confirmed duplicate/source/template groups in one split.
- Create approximately 70/15/15 train/validation/test manifests with seed 42.
- Ensure all five classes appear in every split.
- Implement image resize/padding, text vocabulary, dataset, and batch collation.
- Produce class distribution, text-length, and pair-quality analysis.

### Deliverables

- `train.csv`, `val.csv`, and `test.csv`.
- Manifest fields:
  `image_name`, `image_path`, `text_corrected`, `label_id`, `group_id`.
- Saved vocabulary, exclusion records, and manifest hashes.
- EDA notebook and exported figures.
- Data-description report section.

### Completion checks

- No sample/group overlap across splits.
- No confirmed exact image copies cross splits.
- Vocabulary uses training data only.
- One batch matches the shared contract.
- Actual sample counts and remaining leakage limitations are documented.

## Member 2 — Encoders and Baseline Training

### Tasks

- Implement frozen pretrained ResNet-18 and record the weights identifier.
- Keep frozen encoder BatchNorm statistics fixed.
- Implement embeddings/GRU with correct padding handling.
- Implement image-only and text-only classifiers.
- Implement one shared training/validation loop used by every model.
- Support early stopping and checkpoint selection by validation Macro-F1.

### Deliverables

- Working encoders and two unimodal classifiers.
- Shared trainer with saved history and checkpoints.
- Baseline validation results from actual runs.
- Encoder/training methodology report section.

### Completion checks

- Models return finite logits `[B, 5]`.
- Loss/backpropagation works for trainable parameters.
- Frozen image encoder remains unchanged during training.
- Padding does not affect text representation incorrectly.
- Reloading a checkpoint reproduces its validation metrics within a
  documented tolerance.

## Member 3 — Environment, Fusion, and Integration

### Tasks

- Define compatible local CPU/GPU dependencies and Python version.
- Create Docker CPU/GPU targets and Compose configuration.
- Verify RTX 5070 Ti access in the chosen training environment.
- Implement concatenation, learned-gate, and fixed-average fusion.
- Implement shared metrics, experiment entry points, and comparison outputs.
- Coordinate the controlled fusion comparison and ablation.

### Deliverables

- `Dockerfile`, `compose.yaml`, `compose.gpu.yaml`, `.dockerignore`.
- `requirements.txt`, `requirements-cpu.txt`, `requirements-gpu.txt`.
- `docs/environment.md` with tested local and Docker commands.
- Configuration files and train/evaluate/compare entry points.
- Fusion results, ablation results, and comparison figures.
- Integration, experimental setup, and fusion report sections.

### Completion checks

- Local and Docker execution use the same manifests and configurations.
- Raw data is mounted read-only in Docker.
- Results/checkpoints persist outside containers.
- CPU checks and GPU smoke runs work in the documented environments.
- Actual hardware, versions, runtime, and peak VRAM are recorded.
- Training results are traceable to configuration, checkpoint, and source.

## Shared experiment rules

- Use the same cohort, split, label mapping, and preprocessing.
- Use validation data for tuning and checkpoint selection.
- Evaluate locked configurations on the held-out test set.
- Report accuracy and Macro-F1 over all five classes.
- Include per-class scores, confusion matrices, and compute costs.
- Match encoders and training settings when comparing fusion designs.
- Document remaining differences such as parameter counts.
- Never invent results or claim that fusion must improve performance.

## Integration order

1. Agree on contracts and environment.
2. Member 1 provides reviewed manifests and a working batch.
3. Members 2 and 3 connect models and trainer.
4. Finish image-only, text-only, and concatenation for M2.
5. Add learned gating and the fixed-average ablation for M3.
6. Review results and modality support/conflict examples together.

Small smoke runs may use temporary training-only subsets.
Label them as debugging runs; never report them as full experiments.

## Git collaboration

Suggested branches:
- `a3/data-pipeline`
- `a3/unimodal-training`
- `a3/fusion-environment`

- Work on a feature branch and submit focused pull requests.
- Coordinate before editing another member's owned files.
- Share an interface change before merging it.
- Update README commands when behavior changes.
- Do not commit raw data, virtual environments, credentials, or large weights.
- Include the verification command/output in each pull request.
- Merge interface foundations before dependent changes.

## Milestones

All deadlines are 23:59 Vietnam time unless changed on LMS.

- 18 November 2026 — M1 proposal.
- 25 November 2026 — M2: EDA and three baseline configurations.
- 2 December 2026 — M3: improved fusion, ablation, and complete submission.

## Responsibilities shared by everyone

- Write the report section corresponding to your work.
- Review at least one other member's contribution.
- Contribute to error and modality-interaction analysis.
- Record AI assistance and how it was verified.
- Participate in the presentation.
- Understand the full pipeline sufficiently to explain your contribution.
