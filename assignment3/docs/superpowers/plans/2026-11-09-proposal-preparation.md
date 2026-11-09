# Assignment 3 Proposal Preparation Plan

**Goal:** Produce verified dataset evidence and a reviewed M1 proposal for
five-class sentiment classification using each meme's image and corrected text.

**Execution method:** Student-led implementation, one small script/stage at a
time, followed by Codex review. Creating this plan and the empty output folders
is authorized. The student will write and run the scripts below; they have not
been created or run by this plan. No agent delegation or training is planned.

**Architecture:** Read the existing CSV and images without changing raw data.
Save machine checks separately from human review decisions. Update the existing
proposal from those artifacts; retain a proposed group-aware split until the
data review is sufficient to generate final manifests.

**Tech stack:** Python standard library first; Pillow for decoding images;
ImageHash only for the later perceptual-duplicate stage. No PyTorch or GPU is
needed for proposal preparation.

**Spec:** [Dataset proposal](../../dataset_proposal.md) and
[`AGENTS.md`](../../../AGENTS.md). This plan covers preliminary data evidence
and proposal finalization, not implementation of the proposed models.

## Global constraints

- Input root: `data/raw/memotion/`; CSV: `labels.csv`; images: `images/images/`.
- Join by `image_name`, never row position or directory order.
- Target order: `very_negative`, `negative`, `neutral`, `positive`, `very_positive`.
- Primary text column: `text_corrected`; `text_ocr` is inspected separately.
- Preserve all raw files and record exclusions instead of deleting them.
- Do not infer semantic correctness from a filename match or image hash.
- Keep planned, automatically checked, and manually reviewed evidence distinct.
- Main implementation/experiments require documented instructor approval.
- M1 deadline: 18 November 2026, 23:59 Asia/Ho Chi Minh, unless changed on LMS.

## Folder structure

Paths below are relative to `assignment3`. Script/output files marked planned
will be written in later teaching steps; directories alone are created now.

```text
assignment3/
|-- AGENTS.md
|-- AI_USAGE.md
|-- README.md
|-- scripts/
|   |-- download_dataset.py               # Existing
|   |-- inspect_dataset.py                # Planned: CSV checks; console output
|   |-- audit_dataset.py                  # Planned: image/text audit + exact copies
|   |-- review_pairs.py                   # Planned: stratified manual review pack
|   `-- find_duplicate_candidates.py      # Planned: near-copy review candidates
|-- data/
|   |-- raw/memotion/                     # Existing, ignored by Git
|   |   |-- labels.csv
|   |   `-- images/images/
|   |-- audit/                            # Planned generated audit CSV/JSON
|   `-- review/                           # Planned human decisions + candidate CSV
|-- tmp/
|   `-- pair_review/                      # Planned local HTML preview; ignored
`-- docs/
    |-- dataset_proposal.md               # Existing proposal; update from evidence
    `-- superpowers/plans/
        `-- 2026-11-09-proposal-preparation.md
```

Do not create empty Python files merely to fill the tree. Add scripts as each
step is taught. Empty output folders may not appear in a Git clone; scripts
will create the folders they need.

## Review focus

1. Missing columns/unknown labels: fail clearly rather than silently remapping.
2. Blank corrected text: count whitespace-only fields as missing; do not convert
   missing values into the literal string `nan` or silently fall back to OCR.
3. `.jpe` images and corrupt files: resolve filenames from the CSV, decode actual
   pixels, and include failures in the audit instead of crashing or skipping.
4. Duplicate/template candidates: preserve human review decisions; similar
   appearance alone is not proof of an identical meme or conflicting label.
5. Human review state: unreviewed pairs must remain pending, not pass by default.

## Task 1: Inspect annotations (first teaching step)

**Create:** `scripts/inspect_dataset.py`.
**Consumes:** Original CSV using `csv.DictReader`, UTF-8 with BOM support.
**Produces:** Console-only summary: row count, exact column names, five label
counts in fixed order, missing text counts, duplicate image-name count.

- [ ] Resolve the project root from `__file__`, so the working directory does
  not determine which CSV is read.
- [ ] Check required fields `image_name`, `text_ocr`, `text_corrected`, and
  `overall_sentiment`; reject unknown sentiment strings and blank image names.
- [ ] Print raw counts without altering text or writing a cleaned dataset.
- [ ] Run from `assignment3`: `python scripts/inspect_dataset.py`.
- [ ] Compare with the current snapshot: 6,992 rows; label counts 151, 480,
  2,201, 3,127, 1,033; 161 blank OCR fields; 5 blank corrected-text fields;
  zero duplicate image names. These are evidence to compare, not assertions
  that should hide a changed download.
- [ ] Send the script/output for review before Task 2.

## Task 2: Audit image validity and exact duplicate content

**Create incrementally:** `scripts/audit_dataset.py`.
**Consumes:** Original CSV and corresponding original files.
**Produces:** `data/audit/samples.csv`, `data/audit/summary.json`.

`samples.csv` fields: `image_name`, `label`, `image_exists`, `decode_ok`,
`error`, `width`, `height`, `mode`, `corrected_text_blank`, `ocr_text_blank`,
`corrected_word_count`, `file_sha256`, `pixel_sha256`, `eligible`,
`exclusion_reason`. Hash decoded RGB dimensions plus pixel bytes for the pixel
hash; file hash alone misses identical pictures stored with different metadata.

- [ ] Install only the required package when this task starts:
  `python -m pip install pillow`.
- [ ] First stage: resolve each CSV filename, open with Pillow, call `load()`
  to decode pixels fully, capture dimensions/mode and errors per sample.
  Do not enable truncated-image loading or repair raw data automatically.
- [ ] Second stage: count blank text/word lengths and mark eligible only when
  the image exists, decodes, has a valid label, and corrected text is nonblank.
- [ ] Third stage: calculate byte/pixel hashes for duplicate review and store
  summary counts, class distributions, source CSV hash, tool versions, and
  audit timestamp. Avoid labeling all same-label copies as separate entities.
- [ ] Run: `python scripts/audit_dataset.py`.
- [ ] Confirm `image_960.jpe` is included; verify failures and blank rows are
  recorded; output rows must equal input CSV rows; summary class counts must
  agree with the sample-level file. Report new decode failures if discovered.
- [ ] Review the script/results before using its eligibility decisions.

## Task 3: Inspect image-text pairs manually

**Create:** `scripts/review_pairs.py`.
**Consumes:** Original CSV plus Task 2 sample audit.
**Produces:** `data/review/pair_review.csv` and `tmp/pair_review/index.html`.

Select 10 eligible samples per class using fixed class order, sorted filenames,
and seed 42. Fail clearly if a class cannot supply 10. The CSV records selected
filenames and supplied labels/text plus initially blank human fields:
`text_matches_image` (`yes`, `partial`, `no`, `unclear`),
`sentiment_plausible` (`yes`, `no`, `unclear`), `reviewer`, `notes`.
Preserve previously recorded decisions; do not overwrite them on rerun.

- [ ] Generate a local HTML page containing each image, corrected text,
  supplied label, and filename. Escape text before embedding it in HTML.
  Link original images rather than copying the raw dataset into tracked docs.
- [ ] Run: `python scripts/review_pairs.py`.
- [ ] Open: `Start-Process .\tmp\pair_review\index.html`.
- [ ] Review all 50 entries and fill the CSV; do not overwrite supplied labels.
  Describe doubtful annotations instead of pretending the review is ground truth.
- [ ] Summarize completed/pending reviews and each decision count. A sample
  review estimates issues; it does not prove the whole dataset is clean.

## Task 4: Review duplicate candidates and propose the split unit

**Create:** `scripts/find_duplicate_candidates.py`.
**Consumes:** Task 2 audit, original images, corrected text.
**Produces:** `data/review/duplicate_candidates.csv` with paired filenames,
candidate reason (`file_hash`, `pixel_hash`, `perceptual_hash`, `repeated_text`),
hash distance where applicable, supplied labels, and blank `decision`,
`reviewer`, `notes`. Preserve completed decisions on rerun.

- [ ] Inspect exact file/pixel hash groups first. Flag conflicting annotations
  for review rather than choosing a label by majority or test performance.
- [ ] Install ImageHash only when implementing perceptual matching:
  `python -m pip install ImageHash`.
- [ ] Generate perceptual candidates with 64-bit pHash and initial Hamming
  distance <= 6. This threshold proposes candidates; it is not a verified
  same-template detector. Also flag nonblank normalized repeated text.
- [ ] Run: `python scripts/find_duplicate_candidates.py`.
- [ ] Review candidate pairs as `same_meme`, `same_template`, `different`, or
  `uncertain`. Different captions on a common template require explicit review.
- [ ] Use confirmed relations to propose indivisible source/template groups.
  Keep uncertain relations documented and state remaining leakage risk.
  Check the planned grouping does not make five-class split coverage impossible.
- [ ] If review volume is too large before M1, report counts and pending work
  honestly. Final manifests are not a fabricated prerequisite for the proposal;
  document how grouping will be completed before training.

## Task 5: Finalize the proposal evidence

**Modify:** `docs/dataset_proposal.md`, `AI_USAGE.md` as confirmed information
becomes available; website/shared-log updates require their own authorized edit.

- [ ] Replace provisional observations only with actual audit/review outputs.
  Record exclusions, distributions, text lengths, paired-data limitations,
  candidate/review counts, and the planned split unit.
- [ ] Confirm that the source license/version record is retained and unresolved
  rights questions are identified. Do not invent license clearance.
- [ ] Confirm members' responsibilities, compute access, and AI review owner.
- [ ] Keep all four required models, explicit ablation, accuracy/Macro-F1,
  compute estimates, and support/conflict analysis in the proposal.
- [ ] Mark estimated runtimes as estimates until an approved pilot is measured.
- [ ] Review against handbook Section 27 and submit/link the M1 proposal by the
  deadline using the course's submission procedure.
- [ ] Record the actual instructor status, date, and conditions after response.

## Completion criteria

M1 evidence is reproducible, human reviews are identifiable, limitations and
pending audit work are explicit, and the proposal is ready for student review
and submission. Instructor approval is an external outcome, never generated by
a script. Model implementation and final train/validation/test manifests are
subsequent work after approval, guided by the agreed split plan.
