# Assignment 3: Multimodal Meme Sentiment Classification

Proposed task: predict one of five original sentiment labels from a meme image
and its associated text. Instructor approval is **pending**; no main model
implementation or experiment results are claimed.

- [M1 dataset proposal on the assignment page](../assignment3.html#dataset-proposal)
- [Supporting proposal working copy](docs/dataset_proposal.md)
- [Proposal preparation plan and folder structure](docs/superpowers/plans/2026-11-09-proposal-preparation.md)
- [Mentoring and project instructions](AGENTS.md)
- [Assignment 3 AI usage log](AI_USAGE.md)

## Downloaded data

```text
data/raw/memotion/
|-- labels.csv
`-- images/images/
```

The source is [6992 Meme Images Dataset with Labels, Kaggle version 1](https://www.kaggle.com/datasets/hammadjavaid/6992-labeled-meme-images-dataset).
The existing `scripts/download_dataset.py` downloads and extracts the archive:

```powershell
python -m pip install requests
python scripts/download_dataset.py
```

The script has no resume support and should not be rerun over an active
download. These are download instructions, not training/reproduction commands.

Local inspection on 9 November 2026 verified 6,992 rows and image files, all five
sentiment classes, and OCR/corrected text columns. One corrupt image and five
blank corrected-text rows leave 6,986 provisional pairs before duplicate and
template review. Details and limitations are in the proposal. Raw data is
excluded from Git.

Before submission, review the proposal, source license/usage terms, group
details, proposed responsibilities, and AI disclosure ownership. Record the
instructor's approval and conditions before main implementation/experiments.
