# V3 SubjectMFAM diagnosis and development-only baseline report

> Development inner-CV estimates are not new outer-test results. Frozen SubjectMFAM outer predictions are used only for descriptive diagnosis.

## Evidence boundary

- Frozen manifest: `af5fdda01ceed8f90944f9a25e3a630b09ad17e07ece5f1d32dfe3d147271321` (verified read-only).
- No SubjectMFAM variant was trained or modified.
- Every new baseline uses only the 15 frozen inner train/validation folds; outer-test access flag is false.

## Development leaderboard

These are model-selection/development estimates, not final outer-test estimates.

| Model | BA | Macro-F1 | AUROC | PD recall | DD recall |
|---|---:|---:|---:|---:|---:|
| H1b_handcrafted_logistic_balanced | 0.6947 | 0.6860 | 0.7508 | 0.7818 | 0.6075 |
| H1_handcrafted_logistic | 0.6918 | 0.6858 | 0.7515 | 0.7936 | 0.5900 |
| H2_handcrafted_linear_svm | 0.6875 | 0.6820 | 0.7446 | 0.7917 | 0.5833 |
| MR1_minirocket_multivariate_ridge | 0.6556 | 0.6664 | 0.7393 | 0.8859 | 0.4254 |
| M0_subject_mfam_frozen_inner_reference | 0.5897 | 0.5852 | 0.6131 | 0.9221 | 0.2573 |
| D0_demographic_logistic | 0.5247 | 0.4729 | 0.6049 | 0.9728 | 0.0766 |
| N1a_simple_cnn_lr1e4_drop0p2 | 0.5000 | 0.4144 | 0.4219 | 1.0000 | 0.0000 |
| N1b_simple_cnn_lr2e4_drop0p4 | 0.5000 | 0.4144 | 0.4183 | 1.0000 | 0.0000 |
| predict_all_PD | 0.5000 | 0.4144 | 0.5000 | 1.0000 | 0.0000 |

The top development estimate is **H1b_handcrafted_logistic_balanced** with BA=0.6947.

## Frozen outer-test diagnostic

Frozen pooled BA=0.5934, macro-F1=0.5985, AUROC=0.6016, PD recall=0.8623, DD recall=0.3246.
The frozen confusion matrix is [[238, 38], [77, 37]]; binary DD Brier=0.2043, NLL=0.6193, ECE(10 equal-width bins)=0.0872.
The five frozen fold thresholds have mean=0.2788, SD=0.0433, range=0.2408-0.3516.
A retrospective global threshold of 0.2519 gives BA=0.6048, only +0.0113; this is diagnostic only and must not be reused for model selection.
Mean p(DD) is 0.2156 for PD and 0.3011 for DD, showing substantial score overlap.

## DD subtype diagnosis

Frozen outer descriptive correct/DD-prediction rates are: Atypical 0.133 (15), ET 0.500 (28), MS 0.273 (11), Other 0.300 (60). These are exploratory subgroup estimates.
Development inner-CV DD-prediction fractions below use repeated validation rows: every subject appears in four inner validation contexts; `prediction_rows` and `unique_subjects` are retained in the CSV.

| Model | Atypical | ET | MS | Other |
|---|---:|---:|---:|---:|
| M0_subject_mfam_frozen_inner_reference | 0.067 | 0.446 | 0.091 | 0.246 |
| D0_demographic_logistic | 0.017 | 0.036 | 0.023 | 0.121 |
| H1_handcrafted_logistic | 0.550 | 0.759 | 0.455 | 0.546 |
| H1b_handcrafted_logistic_balanced | 0.600 | 0.768 | 0.455 | 0.562 |
| H2_handcrafted_linear_svm | 0.567 | 0.741 | 0.455 | 0.537 |
| MR1_minirocket_multivariate_ridge | 0.350 | 0.616 | 0.386 | 0.362 |
| N1a_simple_cnn_lr1e4_drop0p2 | 0.000 | 0.000 | 0.000 | 0.000 |
| N1b_simple_cnn_lr2e4_drop0p4 | 0.000 | 0.000 | 0.000 | 0.000 |

ET remains the easiest DD subtype for useful sensor models. The H1 family recovers many Atypical cases that SubjectMFAM misses, while MS remains near 0.455, so subtype difficulty is partly representation-dependent rather than universal.

## Activity, sensor and wrist

Top single activities: LiftHold=0.624, TouchNose=0.617, RelaxedTask=0.605, PointFinger=0.593, DrinkGlas=0.587.
Best leave-one-activity-out estimates: DrinkGlas=0.702, StretchHold=0.698, TouchIndex=0.697, Relaxed=0.697. These exploratory differences may reflect redundancy or variance and are not permission to remove activities.
Sensor ranking: acc_gyro=0.692, acc_only=0.670, gyro_only=0.668.
Wrist ranking: bilateral=0.692, left_only=0.669, right_only=0.658.

## Data-quality and FFT sensitivity audit

All 8580 processed records were audited; observed effective sampling rates span 99.2065-100.8077 Hz. Four timestamp-gap records and 12 matched no-gap controls yielded 672 channel/metric comparisons.
After strict-100-Hz interpolation, the gap group median absolute relative spectral change was 0.0357 (Q75=0.0827); controls were 0.0101 (Q75=0.0259). Both are classified `small` by the preregistered rule.
The robust warning rule flagged 2095/8580 records. It is a sensitivity warning, not a corruption label.
Incorrect-versus-correct Hedges g was -0.055 for flagged-record count and -0.022 for gap count; both bootstrap intervals include zero. Outlier-point count was lower, not higher, among errors (g=-0.230, mean difference=-104.2).

## Demographic audit

D0 uses only age, height, weight, gender and handedness. Condition, disease comments, age at diagnosis and uncertain clinical fields are excluded. Imputation, scaling and one-hot encoding are fit on each inner-training partition only.
D0: BA=0.5247, macro-F1=0.4729, AUROC=0.6049, PD recall=0.9728, DD recall=0.0766. It shows possible demographic ranking/confounding signal but weak thresholded discrimination.

## Training dynamics

Across 15 frozen inner runs, best epoch ranges 4-41 (mean=18.0, median=14.0, SD=10.85). The train-minus-validation BA gap has mean=0.1248, median=0.1370, max=0.2987.
Spearman rho between gap and validation BA is 0.282 (exploratory p=0.308); with n=15 this is no clear association.

## Answers to Q1-Q12

1. **Primary bottleneck:** predominantly representation/aggregation plus DD heterogeneity. Threshold tuning gives only +0.0113 BA, while H1 gives +0.1021 and MiniRocket +0.0660 on the same development protocol. Data-quality and demographic audits do not explain the full gap.
2. **Demographics:** D0 BA=0.5247, AUROC=0.6049, DD recall=0.0766; possible confounding/ranking signal, not a competitive classifier.
3. **Handcrafted linear baselines:** H1 BA=0.6918, macro-F1=0.6858, AUROC=0.7515; H2 BA=0.6875. Both exceed M0 BA=0.5897.
4. **MiniRocket:** BA=0.6556, AUROC=0.7393; it exceeds M0 but remains below H1.
5. **Simple CNN:** both preregistered candidates give BA=0.5000, PD recall=1.0, DD recall=0.0, with best epoch 1. This protocol collapsed to the majority class; no post-hoc candidate was added.
6. **DD subtypes:** ET is easiest. M0 DD-prediction fractions are Atypical 0.067, ET 0.446, MS 0.091, Other 0.246; H1 gives 0.550, 0.759, 0.455, 0.546. Small subgroups require exploratory wording.
7. **Activities:** LiftHold=0.624 and TouchNose=0.617 lead single activities, but full multi-activity H1=0.692 is stronger than every single activity.
8. **Sensors:** combined=0.692, Acc-only=0.670, Gyro-only=0.668; the modalities are complementary.
9. **Wrists:** bilateral=0.692, left=0.669, right=0.658.
10. **Quality association:** no meaningful evidence that timestamp gaps, Fs deviation or warning counts drive errors; the one non-zero outlier-point association is opposite the proposed bad-quality direction.
11. **Training dynamics:** fold-dependent epochs and sizable gaps indicate instability/overfitting risk, but the gap-performance correlation is inconclusive.
12. **Next experiments:** use a new preregistered development-only ablation plan below; do not alter the frozen V3 baseline.

## Prioritized next experiments (recommendations only)

### Priority 1

- Add a parallel raw/full-band or statistical-feature branch, or late-fuse SubjectMFAM with H1-style evidence. Direct support: H1 and MiniRocket both beat M0 under identical inner folds.
- Replace hard Top-K MIL with a soft/differentiable aggregation ablation while preserving a matched control. This is indirectly supported by heterogeneous, distributed subtype/activity evidence; it is a hypothesis, not a demonstrated causal fix.
- Compare the current activity attention with simple mean pooling and a low-capacity gated soft fusion. Multi-activity features outperform all single activities, while leave-one-out results show redundancy.

### Priority 2

- Test BatchNorm versus GroupNorm/LayerNorm in a strictly matched ablation. Batch size is small and training epochs/gaps are unstable, but current evidence is indirect.
- Predefine subtype-stratified reporting and uncertainty intervals. Use subtype labels for evaluation/audit, not opportunistic outer-test tuning.

### Not currently justified

- Class weighting as the main intervention: H1b improves BA by only +0.0029 over H1.
- Timestamp resampling as a primary fix: only four records have gaps, interpolation sensitivity is small, and error associations do not support it.
- Activity dropout or deleting activities based on these exploratory estimates.
- Adding new frequency bands directly to SubjectMFAM: H1 includes 12-20 Hz, but no band-specific ablation establishes which band causes its advantage.

## Guardrails

- Do not use this development leaderboard as final generalization performance.
- Do not reuse the retrospective frozen-outer diagnostic threshold for model selection.
- Outlier flags may represent pathological motion as well as acquisition artifact.
- No requested or recommended SubjectMFAM change was executed in this run.
