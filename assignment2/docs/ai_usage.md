# Assignment 2 AI Usage Log

## Scaffold review and correction - 13 Oct 2026

- Tool/model: OpenAI Codex, GPT-6.
- Member using the tool: Trần Mạnh Thắng (2353124), confirmed in this conversation.
- Stage/purpose: review the user-created folder/file skeleton against the README
  and correct file types and structural issues before implementation.
- Representative prompt: "I already created a file structure can you check the
  type and structure is it correct or not if not you can fix by yourself".
- Contribution: created the missing raw-data folder; initialized two empty
  notebooks as notebook JSON and the class mapping as an empty object; added
  runtime-folder keep files, Git/Docker ignore rules, and updated scaffold status
  in README and AGENTS.md. No class labels, data or results were invented.
- Affected paths: `README.md`, `AGENTS.md`, `.gitignore`, `.dockerignore`,
  `data/raw/.gitkeep`, `data/manifests/class_to_idx.json`, `notebooks/*.ipynb`,
  `results/runs/.gitkeep`, `results/figures/.gitkeep`, and this log.
- Assistant verification sources: local README tree, file inventory, JSON parsing,
  notebook structure checks and Git ignore/whitespace checks. No training or
  Docker build/run was performed.
- Final verification responsibility: Trần Mạnh Thắng (2353124), as confirmed by the user.
  This identifies responsibility; it does not substitute for the group reviewing
  each disclosure entry.

The original planning stage is also recorded in the README. The following
entries extend this historical scaffold record for M2.

## M2 attribution and evidence status - 26 Oct 2026

Teammate confirmation update, 26 Oct 2026: Trần Mạnh Thắng reported that
both teammates confirmed the AI-use entries. This records their confirmation of
tool use/participation; no missing private-session details were supplied.

Member names/IDs come from the assignment page. The current user confirmed that
he is Trần Mạnh Thắng (2353124) and is responsible for final verification. He also reported
that the other two members used **Claude Opus 5.5** for their work. The model label
is recorded as user-reported; private Claude sessions were not inspected.

Repository authorship is evidence of code scope, not proof of the exact prompts,
AI-generated portions, student edits, or individual review procedures. The two
Claude entries below are disclosure drafts awaiting those member-specific details.
The 25 passing tests and completed experiments were reported by the current user;
saved JSON evidence and source were inspected separately by the assistant.

## Entry A2-1 - Codex guidance, integration, experiments and M2 packaging

- **Tool/provider/model:** OpenAI Codex, GPT-6, in the current coding-agent session.
- **Used by:** Trần Mạnh Thắng (2353124), user-confirmed.
- **Dates/stage:** 13-26 Oct 2026; scaffold, Docker setup, implementation guidance,
  debugging, M2 training/reload verification and documentation packaging.
- **Purpose:** incremental teaching and assistance for configuration loading and
  merging, seeding, experiment context, training orchestration, run provenance,
  checkpoint verification, comparison/plot export, CI guidance and M2 writing.
- **Affected files/sections:** `assignment2/README.md`, `AGENTS.md`, Docker/Compose
  and dependency files, `assignment2/config/`, `experiments/context.py`, `run.py`,
  `verify_checkpoint.py`, `compare.py`, `plot_curves.py`, `src/utils/`, related
  tests, `docs/`, `report/`, and the M2 content of `assignment2.html`. CI guidance
  affected `.github/workflows/assignment1-ci.yaml` and `assignment2-ci.yaml`.
- **Representative prompts from this conversation:** "why the collecting test
  taking so long I think some thing is wrong"; "Ok let update readme"; and the
  request to update the assignment page against handbook Section 2.2.
- **AI contribution and student edits:** Codex supplied small code examples and
  explanations for the student to implement and helped diagnose copied-code
  mismatches, stale Docker source, test discovery and container shared-memory
  issues. The student integrated and ran those changes. Codex directly edited
  authorized scaffold/documentation files and the HTML page; students remain
  responsible for reviewing that text and all generated examples. This does not
  attribute teammates' model/data implementations to Codex.
- **Verification actually evidenced:** user-provided Docker output showed 25 CPU
  tests passing, successful GPU execution, and two completed 20-epoch runs. Both
  selected checkpoints reproduced validation Accuracy and Macro-F1. The assistant
  inspected source, saved run artifacts, comparison CSV and training plots; checked
  local links/HTML structure; and confirmed the 10 exported M2 JSON files match
  their originals byte-for-byte. The assistant did not execute the reported
  training runs. A connected browser preview was unavailable for the HTML update.
- **Run evidence:** CNN `20261025T103059Z_a86f4102`, selected epoch 19, validation
  Accuracy 0.2510002667 and Macro-F1 0.2004614168; frozen ResNet-50
  `20261025T112235Z_014bbc02`, selected epoch 20, Accuracy 0.7244598560 and Macro-F1
  0.7231208646. Training source commit: `5d106abab40a6720ebc5874b9a748f01356c965c`.
- **Verification sources:** local handbook and proposal, actual repository source,
  Docker/test outputs supplied by the user, `assignment2/results/m2/`,
  `results/comparison.csv`, the training-history figure, and the PyTorch,
  TorchVision, Docker, GitHub Actions and pytest documentation consulted during
  implementation. These sources support the described checks, not claims about
  unobserved teammate conversations.
- **Final verification responsibility:** Trần Mạnh Thắng (2353124), user-confirmed.
- **Status:** conversation and artifact evidence recorded; student must review the
  final submitted disclosure and preserve authorship accurately.

## Entry A2-2 - Claude assistance for data and EDA (tool use confirmed; details incomplete)

- **Tool/provider/model:** Claude, Anthropic, Opus 5.5 (reported by Trần Mạnh Thắng).
- **Used by:** Phan Tan Loc (2352712), as reported by the current user for the other
  group members. Teammate confirmation was relayed by Trần Mạnh Thắng on 26 Oct 2026.
- **Dates/stage:** Assignment 2 M2 data/EDA development. Exact AI-use dates were not
  supplied. Related repository commit `0f396ea` is dated 23 Oct 2026; the commit
  date is not presented as an AI-session date.
- **Purpose/scope for confirmation:** assistance with the member's data/EDA work.
  The related commit contains audit/duplicate detection, group-aware splitting,
  dataset/transforms/loaders, manifests, EDA notebook/figures and data tests.
  This scope is inferred from repository changes, not a claim that Claude wrote
  every listed file.
- **Affected files for member review:** `assignment2/src/data/`,
  `data/manifests/`, `notebooks/01_eda.ipynb`, EDA figures, `tests/test_dataset.py`,
  `tests/test_split.py`, and the related data configuration changes.
- **Representative prompt:** not supplied. The member must add a real prompt or
  accurate prompt summary; no example is invented here.
- **AI contribution and student edits:** exact generated/reviewed portions and
  subsequent student edits were not supplied and must be described by the member.
- **Verification evidenced at group level:** the integrated suite reported by
  Trần Mạnh Thắng passed 25 tests, including data/split checks. The assistant inspected
  saved audit counts, split IDs/groups and exported artifacts. This does not
  establish which checks Phan Tan Loc personally performed or reviewed.
- **Member verification steps/sources:** not supplied; Phan Tan Loc must record
  the actual checks, source documents, corrections and manual review performed.
- **Individual verification responsibility:** Phan Tan Loc (2352712) for the
  accuracy of this entry and his work; overall M2 final verifier: Trần Mạnh Thắng (2353124).
- **Status:** Claude use and participation confirmed by the teammates through
  Trần Mạnh Thắng on 26 Oct 2026. Exact usage dates, representative prompts,
  generated/reviewed portions, student edits and personal verification details
  remain unprovided; this entry is not yet complete against all handbook fields.

## Entry A2-3 - Claude assistance for models, training and evaluation (tool use confirmed; details incomplete)

- **Tool/provider/model:** Claude, Anthropic, Opus 5.5 (reported by Trần Mạnh Thắng).
- **Used by:** Hoang Kim Cuong (2352145), as reported by the current user for the
  other group members. Teammate confirmation was relayed by Trần Mạnh Thắng on 26 Oct 2026.
- **Dates/stage:** Assignment 2 M2 model/training development. Exact AI-use dates
  were not supplied. Related repository commit `f029780` is dated 19 Oct 2026;
  that is a code-commit date, not an asserted Claude-session date.
- **Purpose/scope for confirmation:** assistance with the member's model,
  training and evaluation work. The related commit contains the scratch CNN,
  ResNet wrapper/factory, epoch trainer, checkpoint helper, metrics/evaluator,
  initial error-analysis module and corresponding tests. The commit scope does
  not prove which parts were AI-generated.
- **Affected files for member review:** `assignment2/src/models/`,
  `src/training/`, `src/evaluation/metrics.py`, `evaluator.py`, `error_analysis.py`,
  `tests/test_models.py`, `test_metrics.py`, and `test_checkpoint.py`.
- **Representative prompt:** not supplied. Hoang Kim Cuong must provide a real
  prompt or accurate summary from his own usage.
- **AI contribution and student edits:** exact generation/review scope and student
  edits were not supplied; the member must complete this field.
- **Verification evidenced at group level:** Trần Mạnh Thắng supplied outputs showing the
  integrated tests, real GPU training and matching checkpoint-reload metrics.
  These establish group-level execution evidence, not Hoang Kim Cuong's private
  review process or a verified causal claim about Claude's contribution.
- **Member verification steps/sources:** not supplied; record the actual model,
  gradient/freeze, metric and checkpoint checks personally performed and reviewed.
- **Individual verification responsibility:** Hoang Kim Cuong (2352145) for the
  accuracy of this entry and his work; overall M2 final verifier: Trần Mạnh Thắng (2353124).
- **Status:** Claude use and participation confirmed by the teammates through
  Trần Mạnh Thắng on 26 Oct 2026. Exact usage dates, representative prompts,
  generated/reviewed portions, student edits and personal verification details
  remain unprovided; this entry is not yet complete against all handbook fields.

## Earlier website disclosures retained for reconciliation

The pre-existing assignment page attributes September 2026 page-skeleton work
with GPT-5.6 to Tran Manh Thang (2353124), and a 16-22 Oct 2026 Claude/Cowork
entry to the same member for data/EDA assistance and manual duplicate review.
Those historical records have not been independently verified in this session.
They are not erased or reassigned because a later commit has a different author.
The group must reconcile overlapping data/EDA contributions and confirm the
historical Claude model separately; the teammates' reported Opus 5.5 label is not
retroactively applied to that earlier Cowork entry.

## Before submission

- Teammate Claude use is confirmed. Obtain the remaining dates, prompts, scope,
  student edits and personal verification details from Phan Tan Loc and Hoang Kim Cuong.
- Reconcile the historical data/EDA entry with the actual contribution record.
- Keep this log, root `AI_USAGE.md`, the report and website summaries consistent.
- Trần Mạnh Thắng (2353124) reviews the final submitted disclosure.
