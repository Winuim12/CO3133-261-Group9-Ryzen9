# Assignment 3 AI Usage Log

This local log supplements the repository-wide `../AI_USAGE.md`. Before
submission, link or incorporate it into the shared log, landing page,
Assignment 3 page, and report. Those external files were not updated here.

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
