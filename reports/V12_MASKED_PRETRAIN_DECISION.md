# V12 fold-train-only masked pretraining decision

Date: 2026-09-16

## Hypothesis and controlled comparison

V12 tests whether diagnosis-label-free masked sequence reconstruction on each
inner fold's training subjects improves the unchanged V8 downstream classifier.
The downstream architecture, supervised loss, optimizer settings, split and
seed are identical to V8 seed 42.

The temporary decoder receives V8's learned time and normalization-free moment
maps, and reconstructs all six normalized sensor channels only at masked points.
Thirty percent of each valid wrist sequence is hidden in contiguous 25-sample
spans. After 10 pretraining epochs, the decoder is discarded and a new supervised
optimizer fine-tunes the complete V8 model.

No diagnosis label, inner-validation subject or outer-test signal is consumed by
the pretraining stage. Each fold records its training-subject hash and leakage
audit fields in `pretraining/status.json` and `development_summary.json`.

## Preregistered gate

Seed 42 was the discovery run. Seeds 43 and 44 would be run only if all discovery
conditions held:

1. mean balanced-accuracy gain at least 0.01;
2. at least 9 of 15 paired-fold BA wins;
3. AUROC regression no worse than 0.01.

## Actual 15-fold result

| Metric | V8 seed 42 | V12 seed 42 | Paired delta | V12 wins |
|---|---:|---:|---:|---:|
| Accuracy | 0.7359 | 0.7341 | -0.0019 | 7/15 |
| Balanced accuracy | 0.6682 | 0.6736 | +0.0055 | 7/15 (1 tie) |
| Macro-F1 | 0.6710 | 0.6752 | +0.0043 | 8/15 (1 tie) |
| AUROC | 0.6649 | 0.6719 | +0.0070 | 9/15 |
| DD recall | 0.5049 | 0.5282 | +0.0233 | 8/15 (1 tie) |
| NLL | 0.6242 | 0.6140 | -0.0102 | 7/15 lower |
| Brier | 0.4196 | 0.4180 | -0.0016 | 8/15 lower |

Mean reconstruction MSE decreases from 0.9677 at epoch 1 to 0.7265 at epoch 10,
so the auxiliary task is learned. That learning does not transfer consistently:
outer-context mean BA deltas are -0.0233, +0.0030, +0.0315, -0.0026 and +0.0187.

## Decision

Reject V12 as the new candidate. It fails both the minimum BA gain and paired-fold
win gates. Do not run seeds 43/44, tune mask ratio/span length, or report isolated
0.72-0.73 folds as evidence of success. V8 remains the selected pure-deep model;
V12 is retained as a fully logged negative result.
