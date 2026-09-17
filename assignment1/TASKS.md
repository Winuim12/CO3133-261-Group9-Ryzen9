# Assignment 1 — Task Checklist

> **Goal:** Build a reproducible Fashion-MNIST image classification pipeline and compare 5 mandatory architectures:
> **Linear, MLP, CNN, LSTM/GRU, Transformer**

---

# 👤 Member 1 — Team Leader

## Phase 1 — Project & Git Setup

* [x] Finalize project structure
* [x] Add `README.md`
* [x] Add `GIT_WORKFLOW.md`
* [x] Add `.gitignore`
* [x] Add `requirements.txt`
* [x] Configure GitHub branch protection
* [ ] Configure GitHub Actions CI

---

## Phase 2 — Data Pipeline

* [x] Download/load Fashion-MNIST
* [x] Define random seed
* [x] Create train/validation/test split
* [x] Verify split reproducibility
* [x] Verify no data leakage
* [x] Implement Dataset
* [x] Implement DataLoader
* [x] Implement preprocessing
* [ ] Implement augmentation
* [x] Verify image shape and labels

### Expected data interface

```text
Input image:
(B, 1, 28, 28)

Classes:
10
```

---

## Phase 3 — Training Pipeline

* [ ] Define common model interface
* [ ] Implement training loop
* [ ] Implement validation loop
* [ ] Implement test loop
* [ ] Implement loss calculation
* [ ] Implement optimizer configuration
* [ ] Implement learning-rate configuration
* [ ] Implement checkpoint saving
* [ ] Implement checkpoint loading
* [ ] Implement training/validation logging
* [ ] Implement training history

---

## Phase 4 — Evaluation

Implement **one shared evaluation framework for all 5 models**.

* [ ] Accuracy
* [ ] Macro-F1
* [ ] Number of parameters
* [ ] Training time
* [ ] Inference time
* [ ] Confusion matrix
* [ ] Correct prediction examples
* [ ] Incorrect prediction examples
* [ ] Training curves
* [ ] Validation curves
* [ ] Model comparison table

### Expected evaluation interface

```text
model
  ↓
test_loader
  ↓
Evaluator
  ├── Accuracy
  ├── Macro-F1
  ├── Parameters
  ├── Inference time
  ├── Confusion matrix
  └── Predictions
```

---

## Phase 5 — Experiment Orchestration

* [ ] Implement model factory
* [ ] Implement experiment runner
* [ ] Implement configuration system
* [ ] Save experiment configuration
* [ ] Save metrics
* [ ] Save plots
* [ ] Save predictions
* [ ] Make experiments reproducible

---

## Phase 6 — Integration

* [ ] Integrate Linear
* [ ] Integrate MLP
* [ ] Integrate CNN
* [ ] Integrate LSTM/GRU
* [ ] Integrate Transformer
* [ ] Verify all models use the same data split
* [ ] Verify all models use the same evaluation protocol
* [ ] Run end-to-end pipeline

---

# 👤 Member 2 — Linear / MLP / CNN

## Linear Classifier

* [ ] Implement Linear classifier
* [ ] Flatten image to 784 features
* [ ] Produce 10 logits
* [ ] Use CrossEntropyLoss correctly
* [ ] Do not apply Softmax before CrossEntropyLoss
* [ ] Add unit tests
* [ ] Document architecture

## MLP

* [ ] Design MLP architecture
* [ ] Implement at least one hidden layer
* [ ] Choose activation function
* [ ] Add regularization if used
* [ ] Add unit tests
* [ ] Document architecture

## CNN

* [ ] Design CNN architecture
* [ ] Implement convolution layers
* [ ] Implement pooling
* [ ] Implement feature extraction
* [ ] Implement classification head
* [ ] Do not use a pretrained model as the main CNN
* [ ] Add unit tests
* [ ] Document architecture

---

# 👤 Member 3 — LSTM/GRU / Transformer

## LSTM / GRU

* [ ] Choose LSTM or GRU
* [ ] Define image-to-sequence representation
* [ ] Define timestep
* [ ] Define input size
* [ ] Define hidden size
* [ ] Implement sequence model
* [ ] Implement classification head
* [ ] Add unit tests
* [ ] Document architecture

## Transformer

* [ ] Define image-to-sequence/patch representation
* [ ] Define token representation
* [ ] Implement token projection/embedding
* [ ] Implement positional encoding
* [ ] Implement Transformer encoder
* [ ] Implement classification head
* [ ] Add unit tests
* [ ] Document attention inputs/outputs

---

# 🧪 Shared Testing Requirements

## Unit Tests

* [ ] Dataset test
* [ ] DataLoader test
* [ ] Train/validation split test
* [ ] Linear forward-pass test
* [ ] MLP forward-pass test
* [ ] CNN forward-pass test
* [ ] LSTM/GRU forward-pass test
* [ ] Transformer forward-pass test
* [ ] Evaluation metrics test

## Smoke Test

* [ ] Load one batch
* [ ] Forward pass
* [ ] Calculate loss
* [ ] Backward pass
* [ ] Optimizer step
* [ ] Complete without errors

---

# 📊 EDA

* [ ] Class distribution
* [ ] Input image size
* [ ] Number of channels
* [ ] Dataset size
* [ ] Imbalance analysis
* [ ] Representative samples
* [ ] Train/validation/test split explanation
* [ ] Data leakage analysis

---

# 🔬 Final Experiments

For **all 5 models**:

* [ ] Train using the same split
* [ ] Use the same evaluation protocol
* [ ] Record random seed
* [ ] Record hyperparameters
* [ ] Record hardware
* [ ] Record software versions
* [ ] Record checkpoint selection rule
* [ ] Record training time
* [ ] Record inference time
* [ ] Calculate Accuracy
* [ ] Calculate Macro-F1
* [ ] Calculate parameter count
* [ ] Generate training curves
* [ ] Generate validation curves
* [ ] Generate confusion matrix
* [ ] Collect correct predictions
* [ ] Collect incorrect predictions

---

# 🔍 Analysis

* [ ] Compare Accuracy
* [ ] Compare Macro-F1
* [ ] Compare parameter count
* [ ] Compare training time
* [ ] Compare inference time
* [ ] Compare convergence
* [ ] Analyze confusion matrices
* [ ] Analyze failure cases
* [ ] Analyze data representation
* [ ] Analyze inductive bias
* [ ] Discuss limitations
* [ ] Do not rank models using accuracy alone

---

# 📝 Report & Deliverables

* [ ] Problem formulation
* [ ] EDA
* [ ] Dataset description
* [ ] Data splitting
* [ ] Preprocessing
* [ ] DataLoader
* [ ] Training methodology
* [ ] Linear explanation
* [ ] MLP explanation
* [ ] CNN explanation
* [ ] LSTM/GRU explanation
* [ ] Transformer explanation
* [ ] Experimental setup
* [ ] Results table
* [ ] Training/validation curves
* [ ] Confusion matrices
* [ ] Error analysis
* [ ] Inductive bias analysis
* [ ] Limitations
* [ ] Conclusion
* [ ] Reproducibility information
* [ ] GitHub Pages
* [ ] Presentation video
* [ ] AI disclosure

---

# 🎯 Milestones

## M1 — Draft

**Deadline: 23 Sep 2026**

### Target

* [ ] EDA completed
* [ ] Data pipeline working
* [ ] Training pipeline working
* [ ] All 5 models implemented
* [ ] All 5 models can train
* [ ] Basic evaluation working
* [ ] Initial results available

---

## M2 — Final

**Deadline: 21 Oct 2026 — 23:59 GMT+7**

### Target

* [ ] Final experiments completed
* [ ] All required metrics collected
* [ ] Confusion matrices completed
* [ ] Error analysis completed
* [ ] Inductive bias analysis completed
* [ ] Final comparison completed
* [ ] Report completed
* [ ] GitHub Pages completed
* [ ] Presentation video completed
* [ ] Reproducibility verified
* [ ] Final repository cleaned and reviewed

---

# 🔗 Git Workflow

For every task:

```text
main
 ↓
Create feature branch
 ↓
Implement
 ↓
Write tests
 ↓
Run tests locally
 ↓
Commit
 ↓
Push
 ↓
Pull Request
 ↓
CI passes
 ↓
Code review
 ↓
Merge
```

## Branch Examples

```text
feature/data-pipeline
feature/linear-model
feature/cnn-model
feature/transformer-model
feature/evaluation
fix/dataloader-split
docs/update-report
```

## Commit Examples

```text
feat: add Fashion-MNIST data pipeline
feat: implement CNN classifier
test: add CNN forward pass test
fix: correct validation split seed
feat: add macro-f1 evaluation
docs: document transformer architecture
```

---

# ✅ Definition of Done

A task is considered **Done** only when:

* [ ] Implementation completed
* [ ] Tests added/updated
* [ ] Tests pass locally
* [ ] Documentation updated if necessary
* [ ] Commit follows convention
* [ ] Pull Request created
* [ ] CI passes
* [ ] PR reviewed
* [ ] Changes merged into `main`
