# Experiment priorities

## Current priority: finish M2 Draft

Decision recorded on 25 Oct 2026: finish the M2 deliverables before profiling
or optimizing training throughput. Preserve the completed baseline and frozen
ResNet-50 runs and their recorded settings.

Remaining M2 work: export the verified validation comparison table and training
curves, update reproduction instructions and assignment-page results, reconcile
AI disclosures, and verify submission/checkpoint-access links.

## Follow-up after M2: investigate training throughput

- Observation: the 20-epoch CNN run took 2,715.15 seconds; the frozen ResNet-50
  run took 2,744.06 seconds. Both totals include training, validation, data loading
  and checkpoint saving. The user reported low GPU utilization during training.
- Status: a data-pipeline bottleneck is a hypothesis, not a measured conclusion.
- Profile batch waiting, CPU-to-GPU transfer and GPU computation separately using
  appropriate CUDA synchronization. Compare real batches with synthetic GPU inputs.
- Based on measurements, investigate image decoding/augmentation, worker counts,
  and Windows-to-Docker file access. Record any changes as new configurations;
  keep controlled-experiment settings consistent across conditions.
- Do not start profiling or performance changes before M2 is finished unless the
  user explicitly changes this priority.

Other post-M2 work already discussed: the frozen-versus-full-fine-tuning study
and a documented early-stopping extension with patience 5 and a maximum epoch
budget. Existing M2 runs retain their fixed 20-epoch protocol.
