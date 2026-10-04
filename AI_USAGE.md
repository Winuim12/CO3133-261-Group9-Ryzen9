# AI Usage Disclosure

Course: Deep Learning and Its Applications (CO3133) · Semester-261  
Group 09 - Ryzen9

Members: Hoang Kim Cuong (2352145) · Phan Tan Loc (2352712) · Tran Manh Thang (2353124)

---

## Shared AI Disclosure

Group 09 - Ryzen9 **used generative AI tools** during this course project. The use is
disclosed here, on the shared landing page, on each assignment page, and in each report.

**Tools used**

| Tool | Provider | Model | Used by | How it was accessed |
|---|---|---|---|---|
| ChatGPT | OpenAI | GPT-5.6 | Tran Manh Thang (2353124), Hoang Kim Cuong (2352145) | Web |
| Claude | Anthropic | Claude Opus | Phan Tan Loc (2352712) | Web / desktop app |

**Where AI was used**

1. **Website.** The shared landing page and the skeleton of the three assignment pages —
   HTML structure, the shared stylesheet, and the section layout required by handbook §2.1–2.2 —
   were drafted with AI assistance and then edited by the group.
2. **Model and pipeline code.** AI was used as a reference while implementing the data pipeline,
   the training loop, and the five model families: explaining concepts, reviewing code, and
   suggesting draft implementations. Every suggestion was read, rewritten where needed, executed,
   and validated by the member who owns that component.

**Where AI was not used**

No experimental result, metric, figure, or table in this repository or on the website was
generated or estimated by an AI tool. Every reported number comes from running the group's own
code and is traceable to a checkpoint and a run directory.

**Responsibility**

All members understand and can explain every part of the submitted code and text. The group takes
full responsibility for the content of the repository, the website, and the reports.

**Prompt logs**

The entries below record prompt *summaries*. Each member keeps their own chat history and can
provide the verbatim prompts on request.

---

## Assignment-Specific Logs

### Assignment 1

#### Entry A1-1 — Website and page skeleton

- **Tool:** ChatGPT (OpenAI) — model GPT-5.6
- **Used by:** Tran Manh Thang (2353124)
- **Time / stage:** September 2026 — website build, M1 Draft stage
- **Purpose:** Draft the landing page and the skeleton of the three assignment pages; convert the
  EDA notebook output into the Dataset & EDA section of the Assignment 1 page; write the Problem
  Statement and Methodology sections from the handbook and the repository code.
- **Affected sections / files:** `index.html`, `assignment1.html`, `assignment2.html`,
  `assignment3.html`, `styles.css`, `README.md`, `assets/eda/`
- **Prompt summary:** Asked for an HTML page structure covering the minimum content required by the
  course handbook; asked to turn the exported EDA notebook into a page section; asked to write the
  Problem Statement and Methodology from the handbook plus the actual code in `src/`.
- **How the output was edited and verified:** Reviewed section by section against handbook §2.1–2.2
  and §3.1–3.2; every figure and number re-checked against the outputs of
  `notebooks/01_eda.ipynb` and the JSON files in `m1_draft_results/`; pages rendered locally and
  all links opened manually.
- **Member responsible for final verification:** Tran Manh Thang
- **Sources used for verification:** Course Project Handbook (rev. 14 Sep 2026); the group's own
  notebook outputs and run artifacts.

#### Entry A1-2 — Data pipeline and training loop

- **Tool:** ChatGPT (OpenAI) — model GPT-5.6
- **Used by:** Tran Manh Thang (2353124)
- **Time / stage:** September 2026 — M1 Draft implementation
- **Purpose:** Concept explanation and code review for the Dataset/DataLoader design, the stratified
  train/validation split, training-only normalization statistics, and the train/validate loop.
- **Affected sections / files:** `src/data/dataset.py`, `split.py`, `statistics.py`, `transforms.py`,
  `pipeline.py`, `dataloader.py`; `src/training/train.py`, `validate.py`, `trainer.py`,
  `checkpoint.py`
- **Prompt summary:** Asked how to compute normalization statistics without leaking validation or
  test data; asked whether a stratified split with a fixed seed is enough to keep the comparison
  fair; asked for a review of the per-epoch loss and accuracy accumulation.
- **How the output was edited and verified:** Compared with the PyTorch and scikit-learn
  documentation; validated by the unit tests in `tests/test_data/` and `tests/test_training/`;
  split overlap, coverage, and tensor shapes asserted directly in the EDA notebook.
- **Member responsible for final verification:** Tran Manh Thang
- **Sources used for verification:** PyTorch documentation (`DataLoader`, `Normalize`,
  `CrossEntropyLoss`); scikit-learn documentation (`train_test_split`); the repository's own tests.

#### Entry A1-3 — Linear, MLP, and CNN models

- **Tool:** ChatGPT (OpenAI) — model GPT-5.6
- **Used by:** Hoang Kim Cuong (2352145)
- **Time / stage:** September 2026 — M1 Draft implementation
- **Purpose:** Explanation of activation and regularization choices; review of draft implementations
  of the Linear/Softmax classifier, the MLP, and the self-designed CNN.
- **Affected sections / files:** `src/models/linear.py`, `mlp.py`, `cnn.py`;
  `config/linear.yaml`, `mlp.yaml`, `cnn.yaml`
- **Prompt summary:** Asked why softmax must not be applied before `nn.CrossEntropyLoss`; asked
  where dropout and batch normalization belong in a small CNN; asked for a check of the channel and
  spatial dimensions through the convolution and pooling blocks.
- **How the output was edited and verified:** Layer shapes traced by hand and confirmed by
  `tests/test_models/`; architectures compared with the PyTorch documentation; all three models
  trained and evaluated, with results stored in `m1_draft_results/runs/`.
- **Member responsible for final verification:** Hoang Kim Cuong
- **Sources used for verification:** PyTorch documentation (`nn.Linear`, `nn.Conv2d`,
  `nn.BatchNorm2d`, `nn.CrossEntropyLoss`); the repository's own tests and training runs.

#### Entry A1-4 — GRU/LSTM and Transformer models

- **Tool:** Claude (Anthropic) — model Claude Opus
- **Used by:** Phan Tan Loc (2352712)
- **Time / stage:** September 2026 — M2 Final preparation
- **Purpose:** Explanation of timestep definitions for row, column, and patch sequences; explanation
  of patch embedding, the `[CLS]` token, and positional encoding in a Vision Transformer; review of
  draft implementations.
- **Affected sections / files:** `src/models/rnn.py`, `transformer.py`; `config/rnn.yaml`,
  `config/transformer.yaml`
- **Prompt summary:** Asked what a timestep should be when an image is fed to a GRU and what input
  size each choice implies; asked how `nn.TransformerEncoderLayer` expects its input to be shaped
  and whether pre-LN is more stable when training from scratch.
- **How the output was edited and verified:** Input and output shapes asserted in
  `tests/test_models/test_gru.py` and `test_transformer.py`; the behaviour of `nn.GRU` and
  `nn.TransformerEncoderLayer` checked against the PyTorch documentation before use.
- **Member responsible for final verification:** Phan Tan Loc
- **Sources used for verification:** PyTorch documentation (`nn.GRU`, `nn.LSTM`,
  `nn.TransformerEncoderLayer`, `nn.TransformerEncoder`); the repository's own tests.

### Assignment 2

Work on Assignment 2 has not started, so no AI tool has been used for its content, code,
experiments, or results. One disclosure already applies: the skeleton of `assignment2.html` was
drafted with ChatGPT (OpenAI, model GPT-5.6) by Tran Manh Thang in September 2026 and then reviewed by the group
(see entry A1-1). This section will be completed as the assignment progresses.

### Assignment 3

Work on Assignment 3 has not started, so no AI tool has been used for its content, code,
experiments, or results. One disclosure already applies: the skeleton of `assignment3.html` was
drafted with ChatGPT (OpenAI, model GPT-5.6) by Tran Manh Thang in September 2026 and then reviewed by the group
(see entry A1-1). This section will be completed as the assignment progresses.
