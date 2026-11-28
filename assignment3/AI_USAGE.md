# Assignment 3 AI Usage Log

This local log supplements the repository-wide `../AI_USAGE.md`. Before
submission, link or incorporate it into the shared log, landing page,
Assignment 3 page, and report. Those external files were not updated here.

## Claude use by Members 1 and 3

The group reported that Hoang Kim Cuong and Phan Tan Loc used Anthropic
Claude, with the model recorded as Claude Opus 5.5, only within their
responsibilities in [Assignment 3 task allocation](docs/task_allocation.md).
The workstreams below describe the reported scope. A file listed under a
workstream is an example of its assigned deliverables.

| Handbook field | Hoang Kim Cuong (2352145), Member 1 | Phan Tan Loc (2352712), Member 3 |
| --- | --- | --- |
| Tool and model | Claude (Anthropic), Claude Opus 5.5, as reported by the group | Claude (Anthropic), Claude Opus 5.5, as reported by the group |
| Time or stage | Data preparation, paired loading, and EDA during implementation | Environment, fusion, metrics, and experiment integration during implementation |
| Purpose | Assistance within the assigned data preparation, paired loading, and EDA workstream | Assistance within the assigned local/Docker setup, fusion, and integration workstream |
| Assignment sections and artifacts in scope | Problem and Data Description, including dataset checks, split manifests, preprocessing, and EDA. Assigned artifacts include `src/data/`, `data/manifests/`, and `notebooks/01_eda.ipynb`. | Methodology and Implementation Results, including fusion, shared metrics, environment, and experiment integration. Assigned artifacts include `src/models/fusion.py`, `src/metrics.py`, `config/`, and `experiments/`. |
| Representative prompt or log link | Pending an actual example or link from Cuong | Pending an actual example or link from Loc |
| Student edits and verification | Pending Cuong's record of the changes he retained and checks he performed | Pending Loc's record of the changes he retained and checks he performed |
| Final verification owner | Pending group confirmation | Pending group confirmation |
| Student verification sources | Pending Cuong's confirmed sources. Data audit, manifest hashes, and data tests are available for review. | Pending Loc's confirmed sources. Model code, fusion and metrics tests, and saved runs are available for review. |

The two members' assigned workstreams are separate from the Codex assistance
recorded below. The assignment of a task by itself does not identify the exact
files edited with Claude or count as student verification.

## Assignment 3 M3 webpage

At Member 2's request, Codex updated `../assignment3.html` against handbook
Section 2.2 and the revised Assignment 3 report. Verification compared the
webpage's metrics and compute figures against the five saved M3 runs and
checked local anchors, images, and paths for the report, slides, and
checkpoint instructions. The report and slides target the planned GitHub tag
`a3-m3-final-submission-2026-11-28`. The group confirmed that its YouTube
presentation link is pending. Student review and final verification remain
to be recorded.

The member then asked Codex to compare the webpage with `assignment2.html`.
Codex added task formulation, method and split diagrams, experimental controls,
training curves, per-class results, error analysis, and checkpoint provenance
from the Assignment 3 report. It checked the new tables against saved M3
artifacts and verified local anchors, chart paths, and the five tagged
deliverable URLs.

## Proposal preparation - 9 November 2026

| Required field | Record |
| --- | --- |
| Tool/model | Codex, OpenAI GPT-6 |
| Member who used the tool | To be completed by the requesting student |
| Stage | Dataset selection, preliminary inspection, and M1 proposal drafting |
| Purpose | Interpret handbook requirements, inspect the downloaded meme annotations, draft the proposal, adapt mentoring instructions |
| Affected artifacts | `../assignment3.html`, `docs/dataset_proposal.md`, `AGENTS.md`, local README and this disclosure |
| Representative request | "Help me create the proposal for Assignment 3 and create AGENTS.md based on the earlier assignment instructions." |
| AI contribution | Drafted proposed methods/splits/experiments; counted CSV labels and blank text; checked image references and Pillow integrity; checked Kaggle metadata |
| Verification performed by Codex | Read handbook Sections 7.6 and 24-32; read local CSV; checked local images; queried public Kaggle version/license metadata; distinguished planned experiments from actual observations |
| Student edits and verification | Pending student review; do not describe this as completed |
| Final student verification owner | To be assigned by the group |
| Verification sources | Local handbook and dataset, source Kaggle page/API, original Memotion paper linked in the proposal |

The group has also used this conversation for task explanations and download
guidance. Add the responsible member, relevant dates/stages, adopted code,
student edits, and verification details for that assistance before submission.
No training outcomes or instructor approval are claimed by this log.

## Proposal webpage update - 9 November 2026

The user requested the M1 proposal inside `../assignment3.html` and explicitly
authorized HTML edits and an AGENTS.md permission update. Codex adapted the
existing page, added the proposal and current AI disclosure, and synchronized
the supporting proposal facts. The local `data/audit/samples.csv` was read to
verify 6,986 provisional pairs and 25 exact-pixel duplicate groups (51 images,
16 conflicting-label groups). Manual duplicate decisions and semantic pair
review were not claimed as complete. Final student review and the responsible
member remain to be recorded.

The compute section was subsequently revised at the user's request for local
RTX 5070 Ti training. Codex verified the GPU name, reported memory, and driver
with `nvidia-smi`, and provided an explicit 2-4 hour planning calculation. Epoch
timings and peak training memory remain unmeasured; no benchmark was run.

At the user's request, Codex separated paired-sample/annotation coverage from
preliminary pair-quality/distribution analysis on the proposal page and added
inline SVG charts for raw sentiment counts and the proposed 70/15/15 split.
Class counts came from the local CSV. Split counts are explicitly illustrative;
no manifests were generated and no claim of completed splitting was made.

Codex subsequently checked the proposal against the requirement list supplied
by the user, added an explicit Subset rules subsection, and clarified all three
instructor decision states and the approval gate. Requirements are covered in
the document; this is not a claim that pending manual reviews or instructor
approval have been completed.

The user selected Assignment 2's proposal design as the visual reference.
Codex reused its scoped layout/CSS, preserved the nine Assignment 3 content
sections, and composed an SVG preview from unmodified original dataset sample
images `image_1.jpg` and `image_4.png`. No sentiment predictions were added to
the preview. This presentation work does not establish completed semantic review.

## Shared bootstrap - 14 November 2026

- Tool: Codex (OpenAI, GPT-6).
- Requesting member: Tran Manh Thang (2353124), Member 2.
- Purpose: shared local/Docker CPU/GPU setup, configuration and environment smoke
  checks, Assignment 3 CI, and three-member handover scaffolding.
- Affected files: requirements/Docker/Compose/config files, `src/config.py`,
  `experiments/check_environment.py`, explicit implementation placeholders,
  tests, notebooks, handover/environment documentation, ignore rules, and
  `../.github/workflows/assignment3-ci.yaml`.
- Representative request: set up code for everyone before handover, ignore
  Superpowers docs, and create Assignment 3 CI and Docker support.
- Verification: executed configuration tests; dependency installation, Compose
  validation, environment smoke checks, and final results are recorded in
  `docs/environment.md`. Official PyTorch/Docker sources are linked there.
- Student edits/final verification owner: pending student review.
- No dataset training, final split generation, or model results are claimed.
- Executed bootstrap verification: local CPU suite 10 passed; CPU Docker image
  built; Compose CPU smoke passed; image-only CPU suite 10 passed without raw
  data mounts; both environments passed `pip check`; CPU/GPU Compose validated.
  GPU target execution and remote GitHub CI are unverified. Static review found
  no significant defects. The existing tracked Superpowers plan was removed
  from Git's index under the user's ignore request, preserving the local file.

### 26 November 2026 — M3 label smoothing

- Codex (OpenAI, GPT-6.2) updated the local fixing workflow and implemented
  optional training-only label smoothing and an isolated concat configuration.
- Affected code: `src/engine.py`, `experiments/train.py`,
  `config/concat_label_smoothing.yaml`, and `tests/test_engine.py`.
- Verification: three focused engine tests passed. A one-epoch CPU debug run
  with 16 training and 16 validation pairs used smoothing 0.1 and reproduced
  its validation Macro-F1 after checkpoint reload. This is a smoke check.
- Student verification owner: pending group confirmation.

### 26 November 2026 — M3 ResNet-50 and DistilBERT comparison

- Codex (OpenAI, GPT-6.2) added a configuration for frozen ResNet-50 and
  DistilBERT with the existing concatenation head and original split manifests.
- The implementation pins the tokenizer and checkpoint to the same DistilBERT
  revision and leaves the GRU vocabulary available for existing runs.
- Verification: 20 focused model tests passed. A one-epoch CUDA debug run
  used 8 training and 8 validation pairs with pretrained weights. A fresh
  model restored from its checkpoint reproduced all 8 validation predictions
  with Hugging Face Hub access disabled. No full-run result is claimed.
- Student verification owner: pending group confirmation.

### 26 November 2026 — M3 dropout comparison

- Codex added `config/concat_dropout.yaml` and ran the seed-42 CUDA comparison
  with dropout 0.5 instead of 0.3. The run matched the baseline data fingerprints.
- Verification: run `20261126-174006-845771-seed42` retained epoch 9 and passed
  checkpoint reload. Validation accuracy was 0.3511 and Macro-F1 was 0.2118.
- Student verification owner: pending group confirmation.

### 27 November 2026 — M3 results analysis notebook

- Codex (OpenAI, GPT-6.2) implemented `notebooks/02_results_analysis.ipynb`
  with opt-in train/test cells, saved run selection, and result analysis.
- Verification: the analysis cells ran against five saved validation runs
  and one saved test run. A simulated CLI check verified the training order,
  resume behavior, and selection of missing test evaluations. No new full
  training or test evaluation was launched while authoring the notebook.
- 28 November follow-up: separated checkpoint selection from the test cell.
  Verified selection of all five new validation runs and simulated test
  dispatch for each model, including reuse on a second pass.
- Student verification owner: pending group confirmation and review of the
  qualitative examples.

### 28 November 2026 — M3 report review draft

- Codex drafted `report/Ryzen9_A3_Report.tex` from the five saved M3 runs,
  inspected selected test memes and matched them to prediction rows, compared
  results with the original Memotion paper, and prepared the report figures.
- Verification: checked run records, split counts, confusion matrices, and
  checkpoint reload outcomes. Measured forward-only CUDA time on a fixed
  validation batch and built and rendered the draft PDF for review.
- Student edits and final verification owner: pending group review.

- Report revision: Codex matched the cover and front-matter structure to the
  group's Assignment 2 report, checked missing items against handbook Sections
  3 and 28–32, and verified the parameter counts and sample batch against the
  Assignment 3 model factory and DataLoader.

- Qualitative review: Codex selected and visually inspected held-out meme
  examples for correct, difficult-but-correct, conflict and intensity cases,
  compiled four image panels, and cross-checked the true labels and predictions
  against the frozen test manifest and five saved M3 prediction CSVs.
  Student review and final verification owner remain pending.

- Checklist-based report revision: Codex used the group's reviewed checklist
  to expand problem formulation, dataset analysis, architecture, comparisons,
  qualitative analysis, reproducibility and disclosure. The requesting student
  asked to remove the decorative green text under section headings.
  Verification used the data audit and exclusion logs, frozen manifests,
  model code, saved run metrics and predictions, GPU timing record, and the
  original SemEval paper. The LaTeX report was rebuilt and rendered for layout
  review. No training was performed for the revision.

- EDA figures and split diagram: at the requesting student's direction, Codex
  computed OCR agreement and aspect ratios for retained pairs, revised the
  report charts, and drew separate validation and test branches. Checks used
  the raw annotation CSV, image audit, cleaned IDs, and frozen split manifests.

- Methodology revision: Codex redrew the training and fusion diagrams, expanded
  parameter-count equations, and tabulated the training hyperparameters.
  Verification used the installed model definitions, instantiated parameter
  tensors, saved run configurations, and rendered report pages.

- Report expansion: Codex consulted the group's Assignment 2 report for section
  depth, then expanded Assignment 3 with tokenizer coverage, a fusion example,
  cross-model learning curves, paired ablation, confidence diagnostics, and
  artifact checks. Values were checked against the frozen vocabulary, manifests,
  training histories, checkpoint files, and saved predictions. All five state
  dictionaries loaded strictly on CPU using their saved configurations.

- M3 presentation: at Member 2's request, Codex used the group's Assignment 2
  slide deck as a design reference and the revised Assignment 3 report as the
  content source. It created the Beamer source and a PDF with formulas, plots,
  and held-out examples. Verification used the report, saved run metrics and
  prediction records, and a visual review of the exported slides. Final
  student verification owner remains pending.

- Presentation navigation: Codex added five section dividers and redesigned
  the outline to follow the same order. The 39 existing titled slides and
  the cover were retained. The revised deck was rebuilt and checked visually.
