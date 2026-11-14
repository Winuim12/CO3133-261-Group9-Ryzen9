# Team handover: Assignment 3 bootstrap

## Available now

- CPU/GPU dependency selections and Docker build targets using Python 3.12.10.
- Shared YAML configuration defaults and five model configuration files.
- `python -m experiments.check_environment --device cpu --smoke-test`.
- Bootstrap tests without raw images, pretrained downloads, or training.
- An independent `assignment3-tests` GitHub Actions workflow.
- Valid, unexecuted EDA/results notebook placeholders.

The environment smoke check exercises PyTorch autograd and a TorchVision
compiled operator. It does not establish model correctness or accuracy.
`train`, `evaluate`, `compare`, and `prepare_splits` are intentional stubs which
fail clearly on execution until their assigned implementation is complete.

## First step for everyone

1. Read `docs/task_allocation.md`, including labels, batch fields, and encoder interfaces.
2. Follow `docs/environment.md` and run the environment smoke check and `tests/`.
3. Compare configuration paths and Python/PyTorch versions before sharing code.
4. Record instructor approval and any conditions before main implementation.

## Cuong: data and EDA

- Start with the existing `scripts/` audit outputs and recorded review decisions.
- Complete duplicate/source/template decisions before implementing final splits.
- Implement `scripts/prepare_splits.py` and `src/data/` incrementally.
- Deliver one batch with `image`, `text_ids`, `text_lengths`, `label`, and
  `image_name`, using the label mapping from `config/base.yaml`.
- Manifest image paths are relative to the Assignment 3 project root.
- Add data tests and populate `notebooks/01_eda.ipynb` with actual evidence.

## Member 2 (the user): encoders and trainer

- Implement `src/models/image_encoder.py` first: frozen ResNet-18, no unwanted
  BatchNorm updates, output features `[B, 512]`.
- Then implement `text_encoder.py` and `unimodal.py` against the shared batch
  contract, before connecting them to `src/engine.py`.
- Return five raw logits and choose checkpoints by validation Macro-F1.
- Coordinate the text representation size and trainer hooks with Loc before
  introducing a different interface.

## Loc: environment maintenance, fusion, and experiments

- Maintain the prepared Docker/config/requirements/CI files rather than creating
  a separate training implementation for each environment.
- Implement metrics and reproducibility helpers; coordinate the encoder feature
  APIs with Member 2 and data interfaces with Cuong.
- Fill `fusion.py` and the train/evaluate/compare entry points once their
  dependencies are ready. Preserve the fixed-average ablation configuration.
- Verify explicit CUDA smoke checks on the RTX 5070 Ti before any GPU results.

## What is intentionally not included

No final manifests, tokenizer, encoders, fusion implementation, training loops,
checkpoints, model results, or executed report figures are supplied by bootstrap.
The current model settings are proposed defaults, not tuned or validated
hyperparameters. The CPU workflow does not validate GPU execution.
