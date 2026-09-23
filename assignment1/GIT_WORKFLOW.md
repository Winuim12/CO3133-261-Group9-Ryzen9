# Git Workflow & Conventions

## 1. Branching

Do not develop directly on `main`.

Use the following branch naming convention:

```text
feature/<name>       # New feature
fix/<name>           # Bug fix
refactor/<name>      # Code restructuring
test/<name>          # Tests
docs/<name>          # Documentation
experiment/<name>    # ML experiments
```

Example:

```text
feature/linear-model
feature/cnn-model
fix/dataloader-split
experiment/transformer-lr
```

---

## 2. Workflow

For each feature:

```text
main
 ↓
Create branch
 ↓
Implement
 ↓
Run tests
 ↓
Commit
 ↓
Push
 ↓
Pull Request
 ↓
CI passes
 ↓
Merge into main
```

---

## 3. Commit Convention

Use **Conventional Commits**:

```text
<type>: <description>
```

Common types:

| Type         | Purpose                   |
| ------------ | ------------------------- |
| `feat`       | Add functionality         |
| `fix`        | Fix a bug                 |
| `refactor`   | Restructure code          |
| `test`       | Add or modify tests       |
| `docs`       | Documentation             |
| `perf`       | Performance improvement   |
| `chore`      | Configuration/maintenance |
| `experiment` | ML experiment             |

Examples:

```text
feat: implement linear classifier
feat: add Fashion-MNIST data pipeline
test: add CNN forward pass test
fix: correct validation split seed
docs: document preprocessing pipeline
perf: optimize DataLoader configuration
```

Avoid vague messages:

```text
update
fix bug
change code
final
done
```

Each commit should represent **one logical change**.

---

## 4. Pull Requests

Every feature should be merged through a Pull Request.

A PR should briefly describe:

* What was changed
* What was tested
* Any important implementation notes

Example:

```text
## Changes
- Implemented CNN classifier
- Added CNN forward-pass tests

## Tests
- pytest: passed
- Forward shape test: passed
```

---

## 5. CI

GitHub Actions should automatically run on:

* Push to `main`
* Pull Requests targeting `main`

CI should verify:

```text
Install dependencies
        ↓
Run tests
        ↓
Run lint / syntax checks
        ↓
PASS / FAIL
```

CI should **not train the full models**.

Use lightweight tests such as:

* Dataset loading
* Tensor shapes
* Model forward pass
* Loss calculation
* Training smoke test
* Evaluation metrics

---

## 6. Reproducibility

Experiments must record:

* Random seed
* Dataset split
* Hyperparameters
* Model configuration
* Python/PyTorch versions
* Hardware
* Checkpoint selection rule

ML results should be traceable to the corresponding code/configuration.

---

## 7. Git Repository Rules

Do not commit:

```text
.venv/
__pycache__/
.pytest_cache/
data/
*.pt
*.pth
results/checkpoints/
```

Large datasets and trained checkpoints should be stored separately from the Git repository.
