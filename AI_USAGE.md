# AI Usage Disclosure

Course: Deep Learning and Its Applications (CO3133)
Semester-261  
Group 09 - Ryzen9

Members: Hoang Kim Cuong (2352145), Phan Tan Loc (2352712), Tran Manh Thang (2353124)

---

## Shared AI Disclosure

Group 09 - Ryzen9 **used generative AI tools** during this course project. The use is
disclosed here, on the shared landing page, on each assignment page, and in each report.
The tools used are listed at the start of each assignment section below.

---

## Assignment-Specific Logs

### Assignment 1

**Tools used**

| Tool | Provider | Model | Used by | How it was accessed |
|---|---|---|---|---|
| Codex | OpenAI | GPT-5.6 | Tran Manh Thang (2353124), Hoang Kim Cuong (2352145) | VS Code extension |
| Claude | Anthropic | Claude Opus | Phan Tan Loc (2352712) | Desktop app |

#### Entry A1-1 — Website and page skeleton

- **Tool:** Codex (OpenAI) — model GPT-5.6
- **Used by:** Tran Manh Thang (2353124)
- **Time / stage:** Website build, M1 Draft stage
- **Purpose:** Draft the landing page and the skeleton of the three assignment pages; convert the
  EDA notebook output into the Dataset & EDA section of the Assignment 1 page; write the Problem
  Statement and Methodology sections from the handbook and the repository code.
- **Affected sections / files:** `index.html`, `assignment1.html`, `assignment2.html`,
  `assignment3.html`, `styles.css`, `README.md`, `assets/eda/`
- **Prompt summary:** Asked for an HTML page structure covering the minimum content required by the
  course handbook; asked to turn the exported EDA notebook into a page section; asked to write the
  Problem Statement and Methodology from the handbook plus the actual code in `src/`.
- **How the output was edited:** Adapted the suggested page structure to the group's website;
  revised the draft descriptions to reflect the implemented pipeline and actual EDA outputs;
  updated figure references, captions, and deliverable links for the repository.
- **How the output was verified:** Reviewed section by section against handbook §2.1–2.2
  and §3.1–3.2; every figure and number re-checked against the outputs of
  `notebooks/01_eda.ipynb` and the JSON files in `m1_draft_results/`; pages rendered locally and
  all links opened manually.
- **Member responsible for final verification:** Tran Manh Thang
- **Sources used for verification:** Course Project Handbook; the group's own
  notebook outputs and run artifacts.
- **Representative prompt example:** "Draft the landing page and assignment-page
  structure. Use the repository code and exported EDA notebook to describe the problem, dataset,
  and methodology, and keep all figures and numbers consistent with the saved outputs."

#### Entry A1-2 — Data pipeline and training loop


- **Tool:** Codex (OpenAI) — model GPT-5.6
- **Used by:** Tran Manh Thang (2353124)
- **Time / stage:** M1 Draft implementation
- **Purpose:** Concept explanation and code review for the Dataset/DataLoader design, the stratified
  train/validation split, training-only normalization statistics, and the train/validate loop.
- **Affected sections / files:** `src/data/dataset.py`, `split.py`, `statistics.py`, `transforms.py`,
  `pipeline.py`, `dataloader.py`; `src/training/train.py`, `validate.py`, `trainer.py`,
  `checkpoint.py`
- **Prompt summary:** Asked how to compute normalization statistics without leaking validation or
  test data; asked whether a stratified split with a fixed seed is enough to keep the comparison
  fair; asked for a review of the per-epoch loss and accuracy accumulation.
- **How the output was edited:** Adapted the suggestions to the project's separate data and
  training modules, shared configuration, and function interfaces. Revised the draft workflow
  to use the same split indices throughout normalization and loading, and integrated the
  reviewed epoch loops with validation and checkpoint handling.
- **How the output was verified:** Compared with the PyTorch and scikit-learn
  documentation; validated by the unit tests in `tests/test_data/` and `tests/test_training/`;
  split overlap, coverage, and tensor shapes asserted directly in the EDA notebook.
- **Member responsible for final verification:** Tran Manh Thang
- **Sources used for verification:** PyTorch documentation (`DataLoader`, `Normalize`,
  `CrossEntropyLoss`); scikit-learn documentation (`train_test_split`); the repository's own tests.
- **Representative prompt example:** "Review the Fashion-MNIST data pipeline and
  training loop. Check that normalization uses only the training subset, the fixed-seed stratified
  split has no overlapping indices, and epoch loss and accuracy are accumulated by sample count."

#### Entry A1-3 — Linear, MLP, and CNN models

- **Tool:** Codex (OpenAI) — model GPT-5.6
- **Used by:** Hoang Kim Cuong (2352145)
- **Time / stage:** M1 Draft implementation
- **Purpose:** Explanation of activation and regularization choices; review of draft implementations
  of the Linear/Softmax classifier, the MLP, and the self-designed CNN.
- **Affected sections / files:** `src/models/linear.py`, `mlp.py`, `cnn.py`;
  `config/linear.yaml`, `mlp.yaml`, `cnn.yaml`
- **Prompt summary:** Asked why softmax must not be applied before `nn.CrossEntropyLoss`; asked
  where dropout and batch normalization belong in a small CNN; asked for a check of the channel and
  spatial dimensions through the convolution and pooling blocks.
- **How the output was edited:** Adapted the draft layers to the selected Linear, MLP, and CNN
  configurations; revised dimensions, activation and regularization placement, and forward-pass
  composition to match the model configuration files. Integrated the models with the shared
  training interface and retained logits as their outputs.
- **How the output was verified:** Layer shapes traced by hand and confirmed by
  `tests/test_models/`; architectures compared with the PyTorch documentation; all three models
  trained and evaluated, with results stored in `m1_draft_results/runs/`.
- **Member responsible for final verification:** Hoang Kim Cuong
- **Sources used for verification:** PyTorch documentation (`nn.Linear`, `nn.Conv2d`,
  `nn.BatchNorm2d`, `nn.CrossEntropyLoss`); the repository's own tests and training runs.
- **Representative prompt example:** "Review the Linear, MLP, and CNN classifiers.
  Explain why CrossEntropyLoss takes logits, check the CNN channel and spatial dimensions, and
  review the placement of dropout and batch normalization."

#### Entry A1-4 — GRU/LSTM and Transformer models

- **Tool:** Claude (Anthropic) — model Claude Opus
- **Used by:** Phan Tan Loc (2352712)
- **Time / stage:** M2 Final preparation
- **Purpose:** Explanation of timestep definitions for row, column, and patch sequences; explanation
  of patch embedding, the `[CLS]` token, and positional encoding in a Vision Transformer; review of
  draft implementations.
- **Affected sections / files:** `src/models/rnn.py`, `transformer.py`; `config/rnn.yaml`,
  `config/transformer.yaml`
- **Prompt summary:** Asked what a timestep should be when an image is fed to a GRU and what input
  size each choice implies; asked how `nn.TransformerEncoderLayer` expects its input to be shaped
  and whether pre-LN is more stable when training from scratch.
- **How the output was edited:** Adapted the sequence-model suggestions to the selected row-wise
  GRU and patch-based Transformer. Revised tensor reshaping, patch-token assembly, positional
  embeddings, and classification heads to match the configured dimensions and the shared
  image-to-logits interface.
- **How the output was verified:** Input and output shapes asserted in
  `tests/test_models/test_gru.py` and `test_transformer.py`; the behaviour of `nn.GRU` and
  `nn.TransformerEncoderLayer` checked against the PyTorch documentation before use.
- **Member responsible for final verification:** Phan Tan Loc
- **Sources used for verification:** PyTorch documentation (`nn.GRU`, `nn.LSTM`,
  `nn.TransformerEncoderLayer`, `nn.TransformerEncoder`); the repository's own tests.
- **Representative prompt example:** "Explain how to represent a 28 by 28 image as
  a sequence for a GRU and check the timestep and feature dimensions. Review the Transformer's
  patch projection, classification token, positional embeddings, and encoder input shape."

### Assignment 2

**Tools used**

| Tool | Provider | Model | Used by | How it was accessed |
|---|---|---|---|---|
| Codex | OpenAI | GPT-6 | Tran Manh Thang (2353124) | Coding-agent session |
| Claude | Anthropic | Claude Opus 5.5 | Hoang Kim Cuong (2352145) | VS Code extension |
| Claude | Anthropic | Claude Opus 5.5 | Phan Tan Loc (2352712) | Desktop app |

#### Entry A2-1 — Project scaffold, environment, and experiment runner

- **Tool:** Codex (OpenAI) — model GPT-6
- **Used by:** Tran Manh Thang (2353124)
- **Time / stage:** Project setup and M2 Draft implementation
- **Purpose:** Review of the folder and file skeleton; guidance on Docker and dependency setup,
  configuration loading and merging, seeding, experiment context, training orchestration, run
  provenance, checkpoint verification, comparison and plot export, and CI; drafting of the M2
  README, report, and page content.
- **Affected sections / files:** `assignment2/README.md`, `AGENTS.md`, `.gitignore`, `.dockerignore`,
  Docker/Compose and dependency files, `assignment2/config/`, `experiments/context.py`, `run.py`,
  `verify_checkpoint.py`, `compare.py`, `plot_curves.py`, `src/utils/`, related tests, `docs/`,
  `report/`, the M2 content of `assignment2.html`, `.github/workflows/assignment1-ci.yaml`,
  `assignment2-ci.yaml`
- **Prompt summary:** Asked for a check of the created file structure and types; asked why test
  collection was slow; asked for small code examples for configuration, seeding, and the training
  runner; asked to update the README and the assignment page against handbook §2.2.
- **How the output was edited:** Implemented the suggested code examples by hand and adapted them to
  the project modules; fixed copied-code mismatches, stale Docker source, test discovery, and
  container shared-memory issues found during debugging. Codex edited scaffold and documentation
  files directly; those drafts were reviewed before submission. No class labels, data, or results
  were generated by the tool.
- **How the output was verified:** The Docker CPU test suite passed 25 tests; both M2 runs completed
  on the GPU, and each selected checkpoint was reloaded and reproduced its saved validation Accuracy
  and Macro-F1 (CNN `20261025T103059Z_a86f4102`, frozen ResNet-50 `20261025T112235Z_014bbc02`);
  exported JSON records were compared byte-for-byte with the originals; page links and HTML structure
  were checked locally.
- **Member responsible for final verification:** Tran Manh Thang
- **Sources used for verification:** Course Project Handbook; the dataset proposal; PyTorch,
  TorchVision, Docker, GitHub Actions, and pytest documentation; the repository's own tests and run
  artifacts in `assignment2/results/m2/`.
- **Representative prompt example:** "I already created a file structure can you check the type and
  structure is it correct or not if not you can fix by yourself."

#### Entry A2-2 — Data audit, splitting, and EDA

- **Tool:** Claude (Anthropic) — model Claude Opus 5.5
- **Used by:** Phan Tan Loc (2352712)
- **Time / stage:** M2 Draft data and EDA development
- **Purpose:** Assistance with the data audit, exact and near-duplicate detection, group-aware
  splitting, the dataset class, transforms and loaders, and the EDA notebook and figures.
- **Affected sections / files:** `assignment2/src/data/`, `data/manifests/`, `notebooks/01_eda.ipynb`,
  EDA figures, `tests/test_dataset.py`, `tests/test_split.py`
- **Prompt summary:** Requests for implementation and review of the audit, duplicate grouping,
  group-aware split, and data-loading modules, and for the EDA analysis and figures.
- **How the output was edited:** Integrated into the project as commits `0f396ea` (audit,
  group-aware split, data loaders, and EDA) and `3f84900` (EDA results, preprocessed batch, and
  notebook outputs). The 54 cross-label near-duplicate pairs were reviewed by hand: 25 were removed
  as the same picture under two dishes and 29 were kept as different dishes.
- **How the output was verified:** Unit tests in `tests/test_dataset.py` and `tests/test_split.py`
  check RGB tensors and integer labels, batch shapes, reproducible manifest hashes, the 70/15/15
  ratio with full class coverage, seed reproducibility, contiguous class indices, audit handling of
  unreadable files, duplicates and label conflicts, and the absence of shared images or groups
  across splits. The saved split has no shared IDs, groups, or SHA-256 hashes across splits.
- **Member responsible for final verification:** Phan Tan Loc
- **Sources used for verification:** PyTorch and TorchVision documentation (`Dataset`,
  `DataLoader`, transforms); Pillow documentation; the repository's own tests, manifests, and audit
  outputs.
- **Representative prompt example:** "Write an audit for the 30VNFoods images that finds unreadable
  files, exact copies by SHA-256 and near copies by perceptual hash, then build a 70/15/15 split
  per class where all copies of an image stay in the same split, with a fixed seed."

#### Entry A2-3 — Models, training, and evaluation code

- **Tool:** Claude (Anthropic) — model Claude Opus 5.5
- **Used by:** Hoang Kim Cuong (2352145)
- **Time / stage:** M2 Draft model and training development
- **Purpose:** Assistance with the scratch CNN, the ResNet-50 wrapper and factory, the epoch
  trainer, the checkpoint helper, the metrics and evaluator, and the initial error-analysis module.
- **Affected sections / files:** `assignment2/src/models/`, `src/training/`,
  `src/evaluation/metrics.py`, `evaluator.py`, `error_analysis.py`; `tests/test_models.py`,
  `test_metrics.py`, `test_checkpoint.py`
- **Prompt summary:** Requests for implementation and review of the scratch CNN, the ResNet-50
  wrapper with a frozen or trainable backbone, the epoch trainer, best-checkpoint selection, and
  the metric and evaluation code.
- **How the output was edited:** Integrated into the project as commit `f029780` and connected to
  the shared configuration and the experiment runner; the models return raw logits, and checkpoint
  selection keeps the earlier epoch on ties.
- **How the output was verified:** Unit tests in `tests/test_models.py`, `test_metrics.py`, and
  `test_checkpoint.py` check the CNN output shape and backward pass, metrics on known predictions,
  per-class metrics for all 30 classes when some classes are never predicted, best-checkpoint
  selection with the earlier-epoch tie rule, and model restoration from a saved checkpoint. GPU
  training completed, and every reloaded checkpoint reproduced its saved validation metrics.
- **Member responsible for final verification:** Hoang Kim Cuong
- **Sources used for verification:** PyTorch and TorchVision documentation (`nn.Conv2d`,
  `nn.BatchNorm2d`, `nn.CrossEntropyLoss`, `torchvision.models.resnet50`); scikit-learn
  documentation (`precision_recall_fscore_support`, `confusion_matrix`); the repository's own
  tests and training runs.
- **Representative prompt example:** "Implement a plain CNN baseline and a ResNet-50 classifier with
  an option to freeze the backbone, both returning 30 logits, and a checkpoint helper that keeps the
  epoch with the best validation Macro-F1 and the earlier epoch on ties."

#### Entry A2-4 — Final results analysis and assignment page

- **Tool:** Claude (Anthropic) — model Claude Opus 5.5
- **Used by:** Hoang Kim Cuong (2352145)
- **Time / stage:** M3 Final preparation
- **Purpose:** Rewrite of the Methodology, Experimental Setup, Results, Comparison & Discussion,
  Error Analysis, and Limitations & Conclusion sections from the three final runs; objective,
  concise wording across the page; export of the final figures and comparison table; grouping of
  test errors by dish family and a visual review of 30 sampled cross-family errors; restructuring of
  this disclosure file.
- **Affected sections / files:** `assignment2.html`; `assignment2/results/final_comparison.csv`;
  `assignment2/results/figures/final_training_histories.png`, `final_qualitative_test.png`,
  `final_cross_family_errors.jpg`; `AI_USAGE.md`; AI-use links in `assignment2/README.md` and
  `assignment2/report/`
- **Prompt summary:** Asked to make the Methodology concise, objective, and free of repeated
  content; asked to remove personal pronouns and sentence fragments across the page; asked to make
  the Experimental Setup meet the handbook; asked to write the Results, Comparison & Discussion,
  Error Analysis, and Limitations sections from the final runs.
- **How the output was edited:** Requested shorter text with highlighted key points; corrected label
  word order (for example "Frozen ResNet-50" and "Trained parameters"); removed the Checkpoint
  Access and Presentation sections; reduced the Deliverables section to the Assignment 1 layout
  without links.
- **How the output was verified:** Every reported number was traced to the run artifacts
  (`summary.json`, `evaluation_*.json`, `inference_benchmark.json`, `history.json`,
  `per_class_test.csv`) in `assignment2/results/kaggle/`; figures and the CSV were produced with
  the repository scripts `experiments.plot_curves` and `experiments.compare`.
- **Member responsible for final verification:** Hoang Kim Cuong
- **Sources used for verification:** Course Project Handbook (§3, §20–21); the saved run artifacts
  of the three final runs.
- **Representative prompt example:** "Make the Assignment 2 methodology more concise, write it
  objectively without personal pronouns, and make sure no content is repeated."
