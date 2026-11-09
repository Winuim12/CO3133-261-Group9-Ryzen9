# Assignment 3 Dataset Proposal

The group presents this proposal in [`../../assignment3.html#dataset-proposal`](../../assignment3.html#dataset-proposal).
This Markdown file is the supporting working copy.

## Submission record

| Field | Value |
| --- | --- |
| Institution | Ho Chi Minh City University of Technology - VNU-HCM, Faculty of Computer Science and Engineering |
| Course | CO3133 - Deep Learning and Its Applications, Semester-261 |
| Instructor | Lê Thành Sách |
| Group | Group 09 - Ryzen9 (confirm exact registered spelling before submission) |
| Members | Hoang Kim Cuong - 2352145; Phan Tan Loc - 2352712; Tran Manh Thang - 2353124 |
| Proposed title | Five-Class Meme Sentiment Classification Using Images and Text |
| Task track | Image-text classification |
| Prepared | 9 November 2026 |
| Approval status | **Pending instructor review; no approval is claimed** |
| Approval date / conditions | Not yet recorded |

This proposal follows handbook Sections 27-31 and the milestone requirements in
Section 7.6. Main implementation and experiments will begin only after the
proposal is Approved or Approved with conditions. Current work is limited to
data inspection and planning. All architecture and training settings below are
proposals, not implemented or experimentally validated decisions.

## 1. Problem, motivation, input, and output

Memes communicate sentiment through wording, facial expressions, scenes, and
the relationship between them. A cheerful image can accompany negative or
sarcastic text. We investigate whether combining the two inputs improves
sentiment classification over using either input alone, and when fusion fails.

Each prediction unit is one meme with its associated text and one sentiment
annotation. The inputs are the RGB meme image and its supplied corrected text.
The output is five raw class scores and one predicted class:

| Proposed ID | Original label |
| ---: | --- |
| 0 | `very_negative` |
| 1 | `negative` |
| 2 | `neutral` |
| 3 | `positive` |
| 4 | `very_positive` |

For each sample `(image, text, label)`, the model estimates the label from the
image, text, or both according to the experimental configuration. The label
column is never an input feature. Predictions are evaluated against the
provided annotations; sentiment is subjective and the annotations are not
treated as an objective interpretation of every meme.

We retain the five original sentiment labels. This is **not** the three-class
SemEval Task A submission protocol, which merges the extreme sentiment labels.
It is also not Tasks B/C: this downloaded CSV has no humour, sarcasm, offense,
or motivation target columns. Results will not be described as directly
comparable to the original three-class competition leaderboard.

## 2. Dataset identity, source, version, and license

| Item | Record |
| --- | --- |
| Downloaded dataset | 6992 Meme Images Dataset with Labels |
| Uploader | Hammad Javaid (`hammadjavaid`) |
| Source | https://www.kaggle.com/datasets/hammadjavaid/6992-labeled-meme-images-dataset |
| Kaggle version | 1, verified through public API metadata on 9 November 2026 |
| Metadata last updated | 9 July 2023, 11:23:09.64 UTC |
| Uploader-declared license | `GPL 2`, as returned by Kaggle metadata |
| Related original benchmark | Memotion / SemEval-2020 Task 8; see References |
| Local location | `data/raw/memotion/` |
| Metadata file | `labels.csv` |
| Image location | `images/images/` under the local location |
| Extracted size | 738,426,163 bytes for the CSV and image files |
| CSV SHA-256 | `67ecfde3e8492001894719761e8d3cddfa1b621b6c71a2b57f80ab127f9f334b` |

The package contains internet meme images and sentiment annotations consistent
with Memotion-7K. The original benchmark paper describes human annotation and
the visuo-linguistic task. The exact transformation history of this Kaggle copy
has not been independently established. Record its own version and checksum
rather than assuming every Memotion release is identical.

The local package contains no separate license file. The uploader's GPL 2
declaration does not independently establish permission for all underlying
internet images or text. Before submission, retain a copy of the source license
metadata, review the original benchmark's conditions, and ask the instructor
about any unresolved usage questions. Link to the source rather than publishing
the raw dataset in the code repository.

## 3. Number of paired samples and annotation types

Local inspection on 9 November 2026 found 6,992 CSV rows, 6,992 image files,
unique `image_name` values, and no missing referenced image files. Include `.jpe`
when enumerating formats: `image_960.jpe` is a valid image, not a missing sample.

Each row links one meme image to OCR text (`text_ocr`), a supplied corrected
transcription (`text_corrected`), and one categorical five-class sentiment
annotation (`overall_sentiment`). The filename is the alignment key. These are
image/text classification annotations, not bounding boxes or segmentation masks.
There are 6,992 raw paired annotation records and 6,986 provisionally eligible
pairs after image/text validity checks, before duplicate decisions.

### 3.1 Preliminary pair-quality and distribution analysis

The webpage shows a horizontal class-count chart using the raw distribution
below, separately from the proposed split chart. The raw and eligible counts
are kept distinct; the chart does not imply duplicates have been removed.

| Sentiment | Raw rows | Provisional usable pairs |
| --- | ---: | ---: |
| Very negative | 151 | 151 |
| Negative | 480 | 480 |
| Neutral | 2,201 | 2,200 |
| Positive | 3,127 | 3,122 |
| Very positive | 1,033 | 1,033 |
| **Total** | **6,992** | **6,986** |

The majority class is positive, about 44.7% of raw rows. The largest-to-smallest
class ratio is about 20.7:1. Accuracy alone would obscure minority-class errors,
so Macro-F1 is the primary metric.

| Pair-quality item | Verified observation |
| --- | --- |
| CSV fields | Unnamed index, `image_name`, `text_ocr`, `text_corrected`, `overall_sentiment` |
| Blank OCR text | 161 rows, approximately 2.30% |
| Blank corrected text | 5 rows, approximately 0.072% |
| Image integrity | One file failed Pillow integrity verification: `image_5119.png`, `Truncated File Read` |
| Corrected text length | Whitespace word counts: minimum 0, median 13, 95th percentile 31, maximum 189 |
| Repeated corrected text | 59 rows beyond the first occurrence after lowercasing and collapsing whitespace, including blank repetitions |
| Image formats | 4,965 JPG, 1,679 PNG, 345 JPEG, 2 BMP, 1 JPE |

The six provisional exclusions are the corrupt image and the five blank
corrected-text rows: `image_120.jpg`, `image_4800.jpg`, `image_6782.jpg`,
`image_6785.jpg`, and `image_6787.jpg`. Raw files will remain unchanged.
For consistency, the primary cohort will require nonblank corrected text;
we will not silently fill only some rows from a different text source.

The student subsequently completed full image decoding: 6,991 images decoded
and `image_5119.png` failed. The saved audit also identifies 25 exact RGB-pixel
duplicate groups containing 51 images: 9 groups have consistent sentiment
labels and 16 have conflicting labels. Corrected-text comparison and decisions
for those groups remain pending; no duplicate exclusions are included in the
6,986 provisional count.

These counts precede completed duplicate/template and semantic pair review.
Final retained counts may change. Matching
filenames establish structural alignment, not that every supplied text is
semantically correct. Use a small class-stratified visual check for preliminary
pair-quality evidence and report its actual sample size and findings. The
earlier suggestion of 50 pairs (10 per class) is a proposed review target,
not a handbook requirement. Detailed duplicate decisions can follow M1 but
must be resolved before training and final split freezing. Do not invent
annotation corrections.

The provisional cohort exceeds the Assignment 3 default of about 5,000 paired
samples and includes five classes. Section 28.2 states a paired-sample threshold;
we do not substitute Assignment 2's separate training-size rule. If cleaning
substantially reduces coverage, seek approval for the revised cohort.

## 4. Alignment, cleaning, and leakage-aware split plan

The webpage illustrates the target 70/15/15 proportions. On 6,986 provisional
pairs, approximate counts would be 4,890 training, 1,048 validation, and 1,048
test pairs. This is a planning illustration, not a generated split. Cleaning
and indivisible source/template groups may change both counts and proportions.

1. Join images and annotations by exact `image_name`; validate every label and
   decode every image. Preserve an audit record for each exclusion.
2. Require usable images and nonblank `text_corrected` for all four models so
   the unimodal and multimodal comparisons use the same prediction units.
3. Detect byte-identical and decoded-image duplicates; flag perceptual image
   matches, repeated normalized text, and recurring meme templates for review.
   Repeated text alone is a candidate signal, not proof of the same entity.
4. Group confirmed duplicate memes and identifiable shared source/template
   families before splitting. Keep each group entirely in one partition.
   Review conflicting annotations and document exclusions; do not resolve
   conflicts using test performance. Report the limits of template detection.
5. Aim for approximately 70% training, 15% validation, and 15% test using seed
   42 and class-aware group assignment. Report actual counts and per-class
   coverage after grouping; do not claim exact ratios or independent image
   stratification. Ensure every class is represented in each split or revise
   the grouping/split plan transparently before training.
6. Save immutable manifests with sample ID, image path, text, label ID, and
   group ID, plus hashes. Check sample/group/hash disjointness. Fit vocabulary
   and any learned preprocessing statistics on training data only.

No class-balanced subsampling is proposed. Retain all eligible examples and all
five classes after documented cleaning. Data augmentation must not create new
validation/test examples. Keep test labels out of model selection and report
the actual split limitations instead of claiming perfect leakage elimination.

### 4.1 Subset rules

No deliberate subsampling is planned. Retain all five classes and all valid
paired samples after documented cleaning; do not randomly downsample or remove
majority-class examples to balance classes. Exclude missing/undecodable images,
blank corrected text, and missing/invalid sentiment labels. After text review,
retain the lexicographically smallest image filename among equivalent valid
same-label pairs, and exclude every record of equivalent pairs with conflicting
labels. Review differing transcriptions before deciding equivalence. A common
template alone is not a reason to remove a distinct valid pair; keep confirmed
source/template groups together when splitting. Preserve raw data and save the
decision/reason for each exclusion. The 6,986 count precedes duplicate decisions.

The instructor's decision states are Approved, Approved with conditions, and
Rejected. Pending instructor review is the current pre-decision state, not an
approval. Record decision date, conditions, and evidence; revise/resubmit if
rejected. Main implementation/experiments require approval and compliance with
any conditions.

## 5. Preprocessing and representations

- Images: decode to RGB, resize with aspect ratio preserved and pad to
  224 x 224, normalize with the image encoder's fixed pretrained constants.
  Avoid horizontal flips and aggressive random crops because they can reverse
  or remove embedded text and alter meme meaning.
- Text: use `text_corrected`, normalize whitespace, retain negation and
  punctuation, and construct a training-only word vocabulary. Initial maximum
  sequence length is 64 tokens with padding and an unknown token. Measure and
  report the actual truncation rate before fixing this choice.
- Image-only input is the complete meme raster, which still includes printed
  text. Thus it is an image-input baseline, not proof of text-free visual
  reasoning. The two branches contain related information; this dependence
  will be discussed explicitly.
- Corrected text is available at inference in the primary benchmark. An
  upload-only demo would require a separate OCR stage and must not claim the
  same performance without evaluating OCR errors.

## 6. Mandatory models, losses, fusion, and fine-tuning

| Model | Proposed implementation | Input/output |
| --- | --- | --- |
| Image-only baseline | ImageNet-pretrained ResNet-18, frozen encoder, trainable five-class head | RGB image to five logits |
| Text-only baseline | Trainable word embeddings and a small GRU encoder with padding-aware pooling and a five-class head | Text tokens to five logits |
| Simple fusion | Same encoders; project image/text features to equal dimensions, concatenate them, then apply a small MLP | Image and text to five logits |
| Improved fusion | Same encoders/projections; a learned sigmoid gate weights image and text features before classification | Image and text to five logits |

Use 256-dimensional projected features initially. The gated representation is
`g * image_features + (1 - g) * text_features`, where the gate is predicted from
both features. This is intermediate feature fusion. The concatenation model
is also intermediate fusion and provides the simple baseline.

Each unimodal model uses cross-entropy against the five-class label. Each
fusion model uses one cross-entropy loss on its final logits; no auxiliary
per-branch loss is proposed. The text branch receives gradients through fusion.
Initially keep the pretrained image encoder frozen, including its BatchNorm
statistics; train heads, projections, text encoder, and fusion parameters.
There is no required end-to-end image fine-tuning experiment. If later added,
report it separately because changing encoder training is another factor.

All models return raw logits `[B, 5]`; Softmax is used only to display
probabilities. Missing-modality fallback is outside the primary benchmark:
reject incomplete inputs with a clear message rather than silently presenting
them as equivalent to paired inference.

## 7. Training, metrics, and controlled experiments

Proposed common settings: AdamW, learning rate 0.001 for trainable parameters,
weight decay 0.0001, batch size 32, up to 20 epochs, dropout 0.3, seed 42, and
early stopping after five epochs without validation Macro-F1 improvement.
Select the highest validation Macro-F1 checkpoint, keeping the earlier epoch
on ties. Use unweighted cross-entropy for the initial comparison; any later
class-weighted experiment must use training-only counts and be reported as a
separate changed factor. Training batch size may be reduced if necessary,
with the actual configuration recorded.

Primary metric: Macro-F1 over the fixed five-class mapping. Secondary outputs:
accuracy, per-class precision/recall/F1/support, a 5 x 5 confusion matrix,
parameter counts, elapsed training time, inference latency/throughput, and peak
GPU memory. Lock model choices using validation data, then evaluate each locked
configuration on the same held-out test set. Do not tune on the test results.

| Experiment | Hypothesis and changed factor | Fixed factors and decision rule |
| --- | --- | --- |
| Image vs. text vs. concatenation | Combining inputs may improve minority-class sentiment recognition; change available inputs/model configuration | Same cohort, manifests, label mapping, image/text preprocessing, metrics, and checkpoint rule; compare Macro-F1 and per-class errors. This is a system comparison, not isolation of architecture alone. |
| Concatenation vs. learned gate | Adaptive modality weighting may help with ambiguous or conflicting evidence; change fusion design | Match encoders, projection size, training budget, data, and loss; report Macro-F1 difference and parameter/time differences, including possible capacity confounds. |
| Gate vs. fixed average (required ablation) | Learning the weights may matter; replace `g` with 0.5 and retrain | Same encoders, feature dimension, classifier, initialization policy, and training setup; compare Macro-F1 and support/conflict examples. |

No gain is promised. If fusion fails to beat the best unimodal model, analyze
imbalance, label ambiguity, correlated modalities, optimization, and compute.
One seed is the minimum budget; if time permits, repeat the two fusion designs
with seeds 43 and 44 and report mean and standard deviation without treating a
single-seed difference as statistically conclusive.

Qualitative analysis will include correct cases, minority-class failures,
adjacent-sentiment confusions, complementary evidence, and apparent conflicts.
Use disagreements among model predictions to find candidates, then manually
review the underlying image and text; disagreement alone does not prove
semantic conflict. Report counts and representative examples with limitations.

## 8. Compute estimate and reproducibility

Training is planned on the group's local NVIDIA GeForce RTX 5070 Ti, with
approximately 16 GB VRAM. A local `nvidia-smi` query on 9 November 2026 confirmed
the GPU name, 16,303 MiB reported memory, and driver 610.88. GPU detection does
not yet verify Assignment 3's PyTorch/CUDA training environment.

Reserve **2-4 hours of local GPU experiment time for one seed**, covering the
four main configurations and the fixed-average ablation. This is a conservative
planning allowance, not a measured benchmark or a claimed speedup over a T4.
Its explicit calculation is:

| Budget component | Planning calculation |
| --- | --- |
| Data scale | Approximately 4,890 training and 1,048 validation pairs from the provisional 70/15/15 split; final grouped sizes may differ |
| Batches per epoch | At batch size 32, about 153 training and 33 validation batches |
| Epoch budget | Five configurations x up to 20 epochs = up to 100 train/validation epochs |
| Assumed epoch allowance | 1-2 minutes per complete training-plus-validation epoch, to be measured after approval |
| Training and validation allowance | 100-200 minutes across the five runs |
| Additional allowance | 20-40 minutes for initial feature extraction where used, setup, checkpoint reloads, and evaluation |
| Total reservation | 120-240 minutes (2-4 hours); no optional extra seeds or image fine-tuning included |

Text-only and cached frozen-feature runs may take substantially less than this
uniform allowance. Early stopping can also reduce time. The local GPU alone
does not determine throughput: image decoding, storage, data-loader workers,
CPU speed, caching, and the final implementation matter. Do not interpret the
assumed epoch duration as an observed RTX 5070 Ti performance figure.

Keep the working-storage allowance at 15-20 GB for raw data, manifests, caches,
checkpoints, and logs, separate from Python/CUDA environment installation.
Actual peak GPU memory will be measured; the approximately 16 GB hardware
capacity is not a claim that the models consume that much VRAM.

After approval, time a short training/validation pilot to revise the budget.
Keep the frozen image features cached by encoder weights, preprocessing, and
image identity if useful; account for feature-extraction cost separately and
never fit a representation on held-out labels. Extra seeds are conditional on
the measured budget. If the budget is tight, preserve all required models and
the ablation before optional fine-tuning or extra runs.

Record dependency versions, seeds, hardware, resolved configuration, dataset
and manifest hashes, source commit plus dirty-tree status, run ID, selected
checkpoint, and logs. Provide installation, preparation, train, and evaluate
commands once implemented. Keep raw data/checkpoints out of Git and supply
source download and checkpoint reconstruction/access instructions.

## 9. Milestones and deliverables

All deadlines below are 23:59 Asia/Ho Chi Minh, subject to milestone-specific
LMS announcements. Assignment 3 contributes 30% of the course project.

| Milestone | Due | Weight within A3 | Planned minimum |
| --- | --- | ---: | --- |
| M1 Proposal | 18 November 2026 | 15% | This proposal, preliminary data analysis, pairing/split plan, and approval record |
| M2 Draft | 25 November 2026 | 25% | Image-only, text-only, and simple fusion with quantitative results and EDA |
| M3 Final | 2 December 2026 | 60% | Improved fusion, ablation, error/support/conflict analysis, compute analysis, complete submission |

The final package includes source and reproducibility instructions, a GitHub
Pages assignment page linked from the shared landing page, report, slides,
checkpoints or reconstruction instructions, and a Public/Unlisted YouTube
presentation in which all members participate. Submit the final PDF on the
LMS as `<GroupName>_A3_Report.pdf`, using the exact registered group name.
The report follows Problem and Data Description, Methodology, and
Implementation Results. Provide AI disclosures on the landing page,
assignment page, report, and AI usage log.

Suggested responsibilities, for group confirmation: Phan Tan Loc leads data
audit, split, and EDA; Hoang Kim Cuong leads models and evaluation; Tran Manh
Thang leads experiment setup, reproducibility, and packaging. All members
review the interpretation and present their contributions.

## 10. Items to resolve before submission and main experiments

- Confirm the registered group name, responsibilities, and AI verification owner.
- Preserve source version/license metadata and review the original usage terms.
- Report preliminary visual pair checks and the completed decoding/exact-hash
  audit in M1, with remaining duplicate/template review clearly planned.
  Finalize cleaning/grouping and manifests before training; do not claim final
  split counts in the proposal before they exist.
- Obtain and record instructor approval, date, and any conditions. Specifically
  state that the proposed task retains five original sentiment classes.
- Use the confirmed local RTX 5070 Ti; verify the Assignment 3 training software
  environment and benchmark epoch time/peak VRAM only after approval.

## References and AI disclosure

1. Course Project Handbook, revision 14 September 2026, Sections 3-5, 7.6,
   and 24-32; local file `../handbook-ene-v2.pdf`.
2. [Kaggle dataset](https://www.kaggle.com/datasets/hammadjavaid/6992-labeled-meme-images-dataset).
   Version/license metadata was checked using its public
   [dataset metadata endpoint](https://www.kaggle.com/api/v1/datasets/view/hammadjavaid/6992-labeled-meme-images-dataset).
3. Sharma et al. (2020), [SemEval-2020 Task 8: Memotion Analysis - the
   Visuo-Lingual Metaphor!](https://aclanthology.org/2020.semeval-1.99/).

Codex (OpenAI, GPT-6) assisted with handbook interpretation, local preliminary
data checks, and drafting this proposal on 9 November 2026. The local checks
were executed; the proposed models and experiments have not been run. Group
members must review the text and verify the source/usage terms and proposed
method before submission. The requesting member and final student verification
owner are not yet identified in this record; complete them in
[`../AI_USAGE.md`](../AI_USAGE.md). No model results are claimed.
