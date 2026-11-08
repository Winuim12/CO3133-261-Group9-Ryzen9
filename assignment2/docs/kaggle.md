# Train Assignment 2 on Kaggle

Notebook: [`../notebooks/02_results_analysis.ipynb`](../notebooks/02_results_analysis.ipynb)
(training, verification, evaluation and the full results analysis in one place),
runner: [`../experiments/kaggle_train.py`](../experiments/kaggle_train.py).
The reviewed manifests are used as they are; the split is never regenerated. The checkpoint is
selected on validation Macro F1, and the test set is evaluated **once** after training
(`experiments/evaluate.py`). Test results are never used to change a model or hyperparameter.

Three runs are trained:

| Variant | Model config | Backbone | Learning rate |
| --- | --- | --- | --- |
| `baseline_cnn` | `config/baseline_cnn.yaml` | trained from scratch | 0.001 |
| `resnet50_frozen` | `config/resnet50.yaml` | frozen, only the classifier is trained | 0.0001 |
| `resnet50_finetune` | `config/resnet50.yaml` with `freeze_backbone: false` | every layer is trained | 0.0001 |

The two ResNet50 runs differ only in `freeze_backbone`: same epoch budget, early stopping rule,
batch size, seed, augmentation, learning rate and AMP option. This is the controlled freeze
versus full fine tuning experiment.

## Early stopping

Every run stops after `PATIENCE` consecutive epochs (default 5) without a new best validation
Macro F1, or at the `EPOCHS` budget (default 20). Early stopping monitors the same metric as
checkpoint selection, so `best_model.pt` is always the best epoch of the stopped run. The rule
is computed from the saved best epoch, so a resumed run continues with the same counter.
`summary.json` and `progress.json` record `stopped_early`, and `summary.json` also records
`epochs_requested` and `early_stopping_patience`. Set `PATIENCE = 0` in the notebook to train
for the full budget.

## Storage

| Item | Size | Kaggle limit |
| --- | --- | --- |
| Images (`food_raw.zip`, 24,982 images) | about 4.5 GB | input dataset, mounted read only at `/kaggle/input` |
| Source (`assignment2_source.zip`) | about 2 MB | input dataset |
| Checkpoints and logs per run | a few hundred MB | `/kaggle/working` holds about 20 GB and is saved with Save Version |

The runner reads images directly from `/kaggle/input` and works in `/tmp`. Only
`/kaggle/working/outputs` is saved, so the output contains neither images nor source copies.

## 1. Prepare on Windows

From the repository root:

```powershell
& ./assignment2/.venv/Scripts/python.exe ./assignment2/experiments/prepare_kaggle.py
```

The script uses only the standard library. It writes to `assignment2/.cache/kaggle_upload/`
(ignored by Git):

| File | Content |
| --- | --- |
| `source/assignment2_source.zip` | `src`, `experiments`, `config`, `data/manifests`, the commit and whether there are uncommitted changes |
| `data/food_raw.zip` | Contents of `assignment2/data/raw`, keeping the `archive/Images/...` paths |

If `data/raw` is empty (a fresh clone has no images), point to the downloaded dataset ZIP
without extracting it: `--data-zip "D:/Downloads/archive.zip"`. If the images live elsewhere,
use `--raw-dir "D:/dataset/raw"` (the folder that contains `archive/Images/...`).
When only the code changes later, add `--skip-data`.

The source ZIP includes uncommitted changes; results record the base commit, the dirty flag
and the exact source snapshot. Commit code and configuration before an official run.

## 2. Data on Kaggle

The account needs **phone verification** (Settings, Phone verification) before GPU and
Internet can be enabled.

**Images: use the original dataset directly, no upload needed.** In step 3, **Add Input**
the dataset [quandang/vietnamese-foods](https://www.kaggle.com/datasets/quandang/vietnamese-foods)
and select **version 11** (the version the manifests were built from). If Kaggle defaults to a
newer version, switch back to version 11 in the Input panel. The runner finds the `Images/`
folder that matches the manifests and checks every train, validation and test image before
training; a single missing image stops the run.

**Source: upload a small dataset.** **Datasets, New Dataset**, upload
`assignment2/.cache/kaggle_upload/source/assignment2_source.zip`, set it to **Private** and
name it, for example, `co3133-a2-source`. Kaggle may extract the ZIP; the runner accepts both forms.

Only when the original dataset cannot be used: also upload
`assignment2/.cache/kaggle_upload/data/food_raw.zip` (4.5 GB) as a separate dataset,
for example `co3133-a2-food-raw`. Do not attach both image sources at the same time.

Optionally use the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) instead of the web UI:
run `prepare_kaggle.py --skip-data --kaggle-user <username>` to write `dataset-metadata.json`, then

```powershell
kaggle datasets create -p assignment2/.cache/kaggle_upload/source
# After code changes:
kaggle datasets version -p assignment2/.cache/kaggle_upload/source -m "Update source"
```

## 3. Create the notebook

1. **Code, New Notebook, File, Import Notebook**, and choose `assignment2/notebooks/02_results_analysis.ipynb`.
2. In the right panel, **Add Input** the image dataset `quandang/vietnamese-foods` (version 11) and the source dataset from step 2.
3. **Settings, Accelerator, GPU T4 x2**. Do not choose P100: recent Kaggle PyTorch builds may
   no longer support it, and the runner reports a clear error. The code uses one GPU; T4 x2 is
   simply the available T4 option.
4. **Settings, Internet, On**, so the pretrained ResNet50 weights can be downloaded.

The notebook uses the PyTorch and TorchVision preinstalled on Kaggle and installs PyYAML,
scikit-learn, matplotlib and pandas only if they are missing. `pip_freeze.txt` and `gpu.txt`
record the actual environment. Do not treat Kaggle and RTX 5070 Ti timings as a controlled
speed comparison.

## 4. Smoke test

In the configuration cell:

```python
RUNS = ["baseline_cnn"]
EPOCHS = 1
```

Run the cells one by one (interactive session). Check that the log prints
`Found all ... manifest images`, the epoch time, and no CUDA out of memory error. Also try
`RUNS = ["resnet50_finetune"]` briefly, since it uses the most VRAM. Estimate the total time as
the time of epoch 2 onwards times 20 epochs times the number of runs (early stopping can only
make it shorter). The analysis sections also run after a smoke test, which checks them before
the official run.

## 5. Official training

Restore the configuration:

```python
RUNS = ["baseline_cnn", "resnet50_frozen", "resnet50_finetune"]
EPOCHS = 20
PATIENCE = 5
BATCH_SIZE = 64
```

Choose **Save Version, Save & Run All (Commit)**. Kaggle runs it in the background, so the tab
can be closed. A session lasts at most about 12 hours and the GPU quota is about 30 hours per
week (see the Settings page). If the estimate from step 4 exceeds about 11 hours, split the work
into several versions with one or two entries in `RUNS` each, then analyse them together (see
step 8). Each run creates a new session; a failed run does not block the next one, and the
training cell lists failures at the end.

If full fine tuning runs out of VRAM, lower `BATCH_SIZE` to 32 **for both ResNet50 runs** so the
comparison stays fair, and report the value used.

By default training uses AMP FP16 with GradScaler, channels last, cuDNN benchmark, TF32 (when the
GPU supports it), persistent workers and a prefetch of 2 batches per worker. Validation and the
checkpoint reload always run in FP32. Fast mode is not bit for bit identical to M2; use the same
options for every run you compare. The first epoch includes worker start up and kernel tuning,
so look at epoch 2 onwards. Each epoch prints its time, training images per second and loader
wait time; `history.json` records the timings.

## 6. Results in the notebook and the output tab

After the version finishes, the notebook itself shows every analysis section (quantitative
results, controlled experiment decision, qualitative examples, error analysis and compute cost)
with outputs saved in the version. The **Output** tab contains:

```text
outputs/<variant>_<session>/
  repo_results.zip
  source_snapshot.zip
  source_provenance.json
  runner_settings.json
  train.log
  gpu.txt
  pip_freeze.txt
  results/runs/<model>/<run_id>/
    best_model.pt
    last.pt
    config.json
    metadata.json
    history.json
    progress.json
    summary.json
    validation_reload.json
    curves.png
    evaluation_val.json, evaluation_test.json   # metrics, confusion matrix, most confused pairs
    predictions_val.csv, predictions_test.csv   # per image predictions with top 3 probabilities
    per_class_val.csv, per_class_test.csv       # per class precision, recall, F1
    confusion_val.png, confusion_test.png
    qualitative_val.png, qualitative_test.png   # confident correct, hard correct, confident wrong
    inference_benchmark.json                    # latency and images/s at batch 1 and 64, FP32 and AMP
outputs/analysis/                               # tables and figures written by the notebook
outputs/analysis_results.zip                    # the analysis folder for the repository
```

After training, the runner runs `verify_checkpoint` and then `evaluate`: the validation pass must
reproduce the saved history exactly before the test set is evaluated and inference is timed.

## 7. Bring results into the repository

Download `repo_results.zip` of each run and `analysis_results.zip`, and extract them **at the
repository root**. The ZIPs already contain the paths `assignment2/results/kaggle/<variant>/<run_id>/`
and `assignment2/results/kaggle/analysis/`, so the M2 results are not overwritten:

```powershell
Expand-Archive -LiteralPath "$env:USERPROFILE/Downloads/repo_results.zip" -DestinationPath .
Expand-Archive -LiteralPath "$env:USERPROFILE/Downloads/analysis_results.zip" -DestinationPath .
git status --short
```

Check that `validation_reload.json` contains `matches_saved_metrics: true` before using a run in
the report, then commit the results folder:

```powershell
git add assignment2/results/kaggle
git commit -m "Add Kaggle evaluation results"
```

To rebuild the comparison table (validation, test, parameters, training time, inference) and
the curves locally, from the `assignment2` folder:

```powershell
python -m experiments.compare --output results/comparison_m3.csv --run-dirs results/kaggle/baseline_cnn/<run_id> results/kaggle/resnet50_frozen/<run_id> results/kaggle/resnet50_finetune/<run_id>
python -m experiments.plot_curves --output results/figures/m3_training_histories.png --run-dirs results/kaggle/baseline_cnn/<run_id> results/kaggle/resnet50_frozen/<run_id> results/kaggle/resnet50_finetune/<run_id>
```

**Do not commit `best_model.pt` or `last.pt`** (`.gitignore` already ignores `.pt`). The
checkpoints live in the notebook version output. To give graders access, choose **New Dataset**
in the Output tab (or share the notebook), and record the link, run IDs and result paths in the
report and deliverables.

To use a checkpoint locally, download `best_model.pt` into the extracted
`results/kaggle/<variant>/<run_id>/` folder and run
`python -m experiments.verify_checkpoint --run-dir results/kaggle/<variant>/<run_id>`, or load it
directly as shown in Section 12 of the notebook.

## 8. Resume after an interruption, or analyse earlier versions

`best_model.pt` keeps the model with the best validation Macro F1; `last.pt` is saved every epoch
with the model, optimizer, AMP scaler, history, RNG state and shuffle generator state.
`history.json` and `progress.json` are also written every epoch.

Unlike Google Drive, `/kaggle/working` does not carry over to the next session. To resume:

1. The interrupted version must have an output (at least one finished epoch, so `last.pt` exists).
2. In a new version: **Add Input, Your Work, Notebook output** of the old version.
3. Set the old session folder from that input, for example:

   ```python
   RESUME_DIRS = ["/kaggle/input/<notebook>/outputs/resnet50_finetune_<stamp>"]
   ```

4. Save Version again. The runner copies the session to `/kaggle/working/outputs` and reuses the
   source snapshot and saved settings, ignoring the new `RUNS`, `EPOCHS`, `PATIENCE` and
   `BATCH_SIZE`. Sessions saved before early stopping existed resume without it.

The interrupted epoch is trained again. Never resume from `best_model.pt`. The RNG inside
persistent workers is not restored, so augmentation after a resume can differ from an
uninterrupted run; a different GPU can also change the numerics. A run is complete only when
training, verification and export all succeed.

To analyse runs from several versions together, attach their outputs, set `TRAIN = False` and
list each attached `outputs` folder in `EXTRA_OUTPUT_DIRS`. The notebook uses the most recent
complete session of each variant.

## Running the same mode on a local GPU

From `assignment2`:
`python -m experiments.run --config config/resnet50.yaml --experiment-config config/fast_gpu.yaml`
(rebuild the Docker image after code changes). `fast_gpu.yaml` enables early stopping with
patience 5; `--patience N` overrides it. To resume locally, add
`--resume results/runs/resnet50/<run_id>` and keep both configuration files unchanged.
