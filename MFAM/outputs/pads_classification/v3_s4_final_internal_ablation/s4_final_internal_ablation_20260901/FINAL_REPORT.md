# S4 final internal ablation report

> Frozen 5×3 inner-CV development evidence only. No outer-test loader, signal, feature transform, prediction, threshold, or metric was used.

## S4 development result

| BA | AUROC | Macro-F1 | Accuracy | PD recall | DD recall | NLL | DD Brier |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.6502 | 0.7239 | 0.6609 | 0.7628 | 0.9220 | 0.3784 | 0.5758 | 0.1837 |

## S4 vs S1 paired decision evidence

Mean/median ΔBA=-0.0172/-0.0094; nonnegative BA folds=3/15. Mean/median ΔAUROC=+0.0051/+0.0148; mean ΔDD recall=-0.0676.

## Preregistered success criteria

- FAIL — ba_nonnegative_folds_ge_9: observed `3`, threshold `9`.
- FAIL — mean_delta_auroc_ge_0_010: observed `0.005133139600355826`, threshold `0.01`.
- FAIL — mean_delta_ba_ge_0_015: observed `-0.01718073582748182`, threshold `0.015`.
- FAIL — mean_delta_dd_recall_ge_minus_0_02: observed `-0.0675837350142133`, threshold `-0.02`.
- PASS — zero_nonfinite_and_skipped: observed `{'nonfinite_gradient_batches': 0, 'nonfinite_loss_batches': 0, 'skipped_optimizer_steps': 0}`, threshold `0`.

Mechanical decision: **S4_rejected_S1_remains_selected**. Selected deep development specification: **S1**.

## Inference-only branch contribution

| Diagnostic | Mode | BA | AUROC | ΔBA vs D0 | ΔAUROC vs D0 | mean |ΔP(DD)| | mean logit L2 change |
|---|---|---:|---:|---:|---:|---:|---:|
| D0 | both | 0.6502 | 0.7239 | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| D1 | frequency_disabled | 0.6259 | 0.6946 | -0.0243 | -0.0292 | 0.1644 | 0.7907 |
| D2 | fullband_disabled | 0.6456 | 0.7303 | -0.0046 | +0.0064 | 0.1096 | 0.7004 |
| D3 | statistical_disabled | 0.5826 | 0.6435 | -0.0676 | -0.0804 | 0.1006 | 0.9870 |

A disabled-branch degradation shows contribution to this trained predictor; it does not establish causal synergy.

## Numerical stability and complexity

S4: 151053 parameters; total runtime=542.5s; best epoch mean/median=17.40/15.0; validation BA SD=0.0403; train-validation BA gap mean=0.1236; nonfinite loss/gradient/skipped=0/0/0.
S1: 123000 parameters; total runtime=395.2s; validation BA SD=0.0449; train-validation BA gap mean=0.1282.

## Artifacts and provenance

- Plan SHA-256: `3ae383049af664dc970d9893927d2b8543afa04c1d8d83ff98ea2a641bd1d157`.
- Frozen split SHA-256: `b3c52317cb12b73c66046bbd7c50c94a7e707a64a38d3ad4f0fd4bc0c6ba6b3e`.
- Read-only stabilization artifact-tree SHA-256: `d46afeb3b62045d427c2b455cdddb71606910a56424af5eeca5aa9d86fa8aa5c`.
- New server source files: 6; existing project source files modified: 0.
- No temporary server implementation or duplicated feature extractor was created; S4 reuses the R1/S1 modules and frozen training engine.
- Branch representation details, all 15 paired fold rows, cluster bootstrap, cross-fitted thresholds and DD subtype estimates are stored as CSV/JSON beside this report.

## Stop condition

The internal architecture search on these 390 subjects is closed after this preregistered decision. No S5, retuning, new fusion, outer-test run, cleanup, or additional architecture experiment was started.
