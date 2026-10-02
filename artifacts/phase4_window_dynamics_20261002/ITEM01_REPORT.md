# Item 01 — frozen WSSL-STR baseline audit

Decision: **REUSE verified baseline**. No training or outer evaluation was run.

All 15 fixed development splits × seeds 42/43/44 passed: checkpoint/status hashes, explicit validation IDs and labels, class order, activity/wrist/channel order, train-only normalization, resolved balanced-CE weights, recipe and archived metric reproduction. Cache hash and dimensions passed. Class weights stored numerically in checkpoints match training-only counts; this is configuration resolution, not recipe drift.

Latest Phase-3B–E formal retained baseline is frozen HarNet10 WSSL-STR. Earlier V8/STR-only summaries are historical and do not define the current baseline. Original checkpoint/cache/prediction files were not changed.

| Metric | Seed-first mean | Split SD | SD of seed means | Mean within-split seed SD |
|---|---:|---:|---:|---:|
| accuracy | 0.745604 | 0.041218 | 0.008650 | 0.031205 |
| ba | 0.719644 | 0.034563 | 0.006714 | 0.021603 |
| auroc | 0.759553 | 0.043921 | 0.001812 | 0.021741 |
| macro_f1 | 0.706589 | 0.039561 | 0.004947 | 0.025605 |
| pd_recall | 0.782344 | 0.061379 | 0.025591 | 0.065659 |
| dd_recall | 0.656945 | 0.058007 | 0.035930 | 0.078542 |

Median selected epoch 10; median epochs executed 22. Mean selected-epoch train loss 0.291046; validation loss 0.679387. Classifier trainable parameters 143,172; frozen HarNet10 10,457,408. Threshold: DD probability >0.5, tie PD (original argmax). All metric deltas from archived WSSL baseline are zero to 1e-12.

Files: PROTOCOL.md, scripts/audit_baseline.py, analysis/baseline_lock.json, baseline_seed_split_metrics.csv, baseline_15split_seedfirst.csv and baseline_summary.csv. Public artifacts contain aggregate rows and hashes only; weights/features/predictions stay on server. New retention criteria are registered in PROTOCOL.md before candidate results.

Shape/mask/gradient/reload/one-batch tests for new branches belong to subsequent numbered items. This read-only baseline audit does not invent training tests. Fifteen overlapping splits are repeated development evidence, not 45 independent observations. Next: item 02, window preservation and numerical/logit consistency.
