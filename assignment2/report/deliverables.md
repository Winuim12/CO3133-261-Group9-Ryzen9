# Assignment 2 — M2 Deliverables

Deadline: 28 Oct 2026, 23:59 GMT+7, unless the LMS states otherwise.

M2 integration and final verification:
Trần Mạnh Thắng (2353124).

## Artifact Index

| Artifact | Location | Status |
| --- | --- | --- |
| Source and project guide | [README](../README.md) | Available locally |
| M2 draft report | [report.md](report.md) | Draft available |
| Environment and reproduction | [environment.md](../docs/environment.md) | Documented |
| Dataset audit and EDA | [EDA notebook](../notebooks/01_eda.ipynb) | Available |
| Reviewed splits and audit records | [Manifests](../data/manifests/) | Available |
| Validation comparison | [comparison.csv](../results/comparison.csv) | Exported |
| Training curves | [Training histories](../results/figures/m2_training_histories.png) | Exported |
| CNN run evidence | [CNN JSON records](../results/m2/baseline_cnn/20261025T103059Z_a86f4102/) | Copied and checked |
| Frozen ResNet run evidence | [ResNet JSON records](../results/m2/resnet50/20261025T112235Z_014bbc02/) | Copied and checked |
| Assignment webpage | [HTML source](../../assignment2.html) | Updated locally; publication pending |
| AI-use disclosure | [Assignment log](../docs/ai_usage.md) | Completion pending |

## Checkpoints

Selected checkpoints are stored locally:

- CNN: results/runs/baseline_cnn/20261025T103059Z_a86f4102/best_model.pt
- Frozen ResNet-50: results/runs/resnet50/20261025T112235Z_014bbc02/best_model.pt

Checkpoint binaries are excluded from Git. Direct download links have not
been published. The environment guide provides reconstruction commands.

The published JSON evidence does not contain model weights. Exact
checkpoint-reload verification requires the corresponding original checkpoint;
new training runs may produce different numerical results.

## Provenance

Both recorded training runs identify source commit:
5d106abab40a6720ebc5874b9a748f01356c965c

Later reporting scripts and documentation belong to the M2 packaging commit.
Record that commit or tag after publishing the package.

## Submission Checklist

- [ ] Complete AI disclosures and final reviewer attribution.
- [ ] Commit all intended M2 source, documentation and exported evidence.
- [ ] Pass CI and merge the M2 PR.
- [ ] Verify the public assignment page, images and artifact links.
- [ ] Confirm checkpoint reproduction instructions are accessible.
- [ ] Check the LMS submission format and submit the required links/files.

## M3 Materials

Full fine-tuning, the controlled comparison, detailed qualitative error
analysis, final slides and the mandatory YouTube presentation remain pending.
No video URL is available yet.