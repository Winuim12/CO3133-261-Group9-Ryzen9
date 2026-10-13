# Assignment 1 Report Additions Checklist

Review date: 13 October 2026.

Reviewed artifacts: `report/Report.pdf` (27 PDF pages, including the cover), `report/report.md`, and `handbook-ene.pdf`.

This checklist records work still needed in the final report. It does not claim that the additions have been completed. The review checked report content against handbook Section 3 and the related Assignment 1, reproducibility, and AI-disclosure requirements. Page references below use PDF viewer page numbers, including the cover.

## Review outcome

The PDF substantially covers all three main parts required by handbook Section 3. It includes dataset properties and EDA, splitting and preprocessing, the five model architectures, a pipeline diagram, implementation/library boundaries, training settings, comparison hypotheses, training curves, quantitative results, qualitative examples, error analysis, and limitations.

The main outstanding mandatory report item is AI Usage Disclosure, required separately by handbook Section 5. Reproducibility details already exist in the Markdown report but were not included in the reviewed PDF. Dataset release identification needs a small clarification.

## 1. Add AI Usage Disclosure to the final report

Priority: required. Handbook references: Sections 4.1, 5.1, and 5.2.

Finding: no AI Usage Disclosure section was found anywhere in the reviewed PDF or Markdown report. A disclosure on the website or in `AI_USAGE.md` does not replace the required disclosure in the report.

Suggested placement: a separate section or appendix before References. Preserve the report's three main academic parts.

For each AI tool actually used, include:

- [ ] Tool name and version/model, if known.
- [ ] Member who used the tool.
- [ ] Time or development stage.
- [ ] Purpose of the assistance.
- [ ] Affected assignment sections or files.
- [ ] A representative prompt example or a link to the prompt log.
- [ ] How the output was edited and verified.
- [ ] Member responsible for final verification.
- [ ] Verification sources, such as documentation, lecture material, tests, or saved experimental artifacts.

Use the existing website disclosure and `AI_USAGE.md` as starting material, checking them against the work actually performed. Include report-writing and result-analysis assistance when applicable. Do not guess a model version, invent prompts, or attribute work to a member without confirmation.

Add a short results-provenance statement explaining that metrics come from executed training/evaluation runs and that figures visualize those recorded outputs. Explain AI assistance accurately without suggesting it generated the measured numbers.

Acceptance check: the exported PDF contains the disclosure and the mandatory fields above; any prompt-log link is usable by the instructor.

## 2. Include the existing reproducibility section in the PDF

Priority: important for result traceability. Handbook reference: Section 4.2; this is outside Section 3 itself.

Finding: `report/report.md`, Section 8, already includes training/evaluation commands, the checkpoint release, archive extraction instructions, and the model-to-run-ID table. None of these appeared in the reviewed PDF. Identify the source used to build the PDF and carry the existing information into that source.

- [ ] Link the source repository and identify the baseline source tag: `a1-m2-baselines-2026-09-25`.
- [ ] Link the baseline checkpoint release:
  <https://github.com/Winuim12/CO3133-261-Group9-Ryzen9/releases/tag/a1-m2-baselines-2026-09-25>.
- [ ] Include the archive filename: `a1-m2-baseline-runs-20260925-122457.zip`.
- [ ] Include the five recorded run IDs below, preserving their spelling.
- [ ] Explain that each run stores `best_model.pt`, `training.json`, `evaluation.json`, and figures, linking the checkpoint, configuration, split, and logs to the reported results.
- [ ] Include the command working directory (`assignment1`) and at least one training and saved-checkpoint evaluation example from the existing Markdown report.
- [ ] Explain that evaluation should use the configuration and code associated with the baseline release. The current evaluation flow reads the configuration on disk.

| Model | Recorded baseline run ID |
| --- | --- |
| Linear | `20260924-124837-513213-m2-baseline` |
| MLP | `20260924-125349-995443-m2-baseline` |
| CNN | `20260924-125826-798568-m2-baseline` |
| GRU (`rnn`) | `20260924-130429-733285-m2-baseline` |
| Transformer | `20260924-130954-040349-m2-baseline` |

The source tag identifies the released code used to reproduce these configurations. Do not describe it as automatically captured training-time provenance unless the run artifacts actually establish that relationship.

Acceptance check: the final PDF provides a path from each reported model result to its saved run and released source/configuration, and its links open correctly.

## 3. Clarify dataset release identification

Priority: small clarification. Handbook reference: Section 3.1, Dataset description.

Finding: PDF Section 1.2, page 6, gives the source, license, counts, and access through Torchvision, but identifies the data only as the "standard Fashion-MNIST release."

- [ ] State the exact release/version identifier if one was recorded.
- [ ] If no dataset version identifier was recorded, say so and identify the official dataset source and downloaded file set. Do not invent a version or retrieval date.
- [ ] Keep dataset release information distinct from the installed Torchvision software version.

The existing statement that Fashion-MNIST is the sole dataset already addresses the absence of separate debugging or extension datasets.

Acceptance check: the dataset-description paragraph makes clear which dataset distribution was used and what version information was, or was not, recorded.

## 4. Optional improvements; no new experiments required for basic completion

These are opportunities to strengthen the evidence, not missing mandatory Assignment 1 experiments.

- [ ] Optional: include all five confusion matrices or link them in an appendix. The PDF currently includes the CNN matrix and an all-model recall table. The handbook does not explicitly require a separate matrix for every model.
- [ ] Optional: add prediction examples for another architecture to support qualitative comparison. The current CNN examples already include correct cases, failures, supplied labels, and discussion of success in a difficult category.
- [ ] Optional: perform a controlled ablation, such as the same CNN with and without spatial dropout. Handbook Section 3.3 recommends ablation for Assignment 1 and makes it mandatory for Assignments 2 and 3. Keep Section 3.6's honest statement if no ablation is performed.
- [ ] Optional: repeat runs with additional seeds and report mean and standard deviation. The handbook requests this when feasible. The current PDF explicitly reports one run per model and explains the resulting limits; do not fabricate variation estimates from those runs.

If any optional experiment is performed, specify its hypothesis, changed factor, fixed factors, metrics, and conclusion criterion before presenting the resulting evidence.

## 5. Existing Section 3 coverage to preserve

| Handbook requirement | Existing PDF evidence |
| --- | --- |
| Problem motivation, input/output, prediction unit, mathematical formulation, difficulties | Section 1.1, page 6 |
| Dataset source/license, counts, formats/storage, bias and limitations | Section 1.2, pages 6-7; release identifier needs clarification |
| EDA: class distribution, image size, imbalance, representative samples | Section 1.3 and Figures 1-2, pages 7-8 |
| Splits, seed, stratification, split unit, leakage prevention | Section 1.4 and Table 2, pages 7-9 |
| Preprocessing, shapes, augmentation/cleaning decisions, example batch | Section 1.5 and Figure 3, pages 9-10 |
| Pipeline diagram | Section 2.1 and Figure 4, page 11 |
| Five methods, representations, strengths/limits, parameter counts | Section 2.2 and Table 3, pages 11-15 |
| Self-implemented components versus library components | Section 2.2.7, page 15 |
| Training settings, selection/stopping rules, precision, hardware/software, time, hyperparameter selection | Section 2.3 and Tables 4-5, pages 15-16 |
| Hypotheses, changed/fixed factors, metrics and conclusion criteria | Section 2.4 and Table 6, pages 17-18 |
| Training/validation loss and accuracy curves, selected epochs, fitting behavior, issues/fixes | Section 3.1 and Figures 5-6, pages 19-21 |
| Predictive metrics, parameters, training/inference costs | Tables 5, 7, and 8, pages 16 and 21-22 |
| Correct, difficult-category correct, and failed examples | Section 3.3 and Figure 7, pages 22-23 |
| Quantified comparisons, representation discussion, resource trade-offs, hypothesis assessment | Section 3.4, pages 23-24 |
| Error groups, rates/counts, examples, possible causes and improvement ideas | Section 3.5, Figure 8, and Tables 9-10, pages 24-26 |
| Ablation status and limitations stated honestly | Sections 3.6-3.7, pages 26-27 |

## 6. Final export check

- [ ] Regenerate `Report.pdf` from the final report source after the additions.
- [ ] Confirm the AI disclosure and reproducibility section are present in the PDF itself.
- [ ] Update the table of contents and page/figure/table references as needed.
- [ ] Check figure labels and table values against the saved run artifacts.
- [ ] Check PDF links and inspect the new pages for clipping and readability.
- [ ] Ensure the website points to the intended final report.

This checklist is a content review. It does not certify visual layout, historical environment versions, or every numerical value against the original run files. It makes no changes to the report, PDF, experiments, or Git staging area.
