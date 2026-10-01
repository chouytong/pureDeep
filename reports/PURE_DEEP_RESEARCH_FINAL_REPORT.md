# PADS PD-vs-DD pure-deep research final report

Date: 2026-09-16

## Outcome

The requested pure-deep search was completed under the frozen subject-level
protocol. The aspirational 0.80 balanced accuracy was not reached. The selected
model is V8, but its one-time outer-test estimate is only 0.6116 BA; therefore it
is a reproducible research baseline, not a strong or deployment-ready backbone.

No result was estimated or manually altered. Every development result comes from
actual training on 15 fixed inner folds. The final outer test was released only
after model search stopped and V8 was frozen; each of the 390 subjects was tested
once in its assigned outer fold. No model or threshold is changed after release.

## Selected architecture: V8

V8 contains 71,026 trainable parameters and no handcrafted input feature. It uses:

1. a shared compact TCN for local temporal patterns;
2. attention/mean/std pooling over learned TCN feature maps;
3. a normalization-free learned convolution bank with kernels 1/15/63;
4. mean/std pooling only over that bank's learned maps;
5. bilateral late fusion and learned activity attention;
6. fold-train-only class-balanced cross-entropy.

Across predefined development seeds 42/43/44, V8 reaches BA
0.6572 ± 0.0123 and AUROC 0.6593 ± 0.0229. It improves matched fold-average
BA over the correspondingly seeded balanced-v3 model by 0.0144 in 11/15 folds.

## Final one-time nested outer evaluation

The final run is:

`/home/zyt/MFAM/outputs/pads_classification/pure_deep_v8_final_nested_cv_20260916`

Fold-specific thresholds and epoch counts were chosen only from the corresponding
outer fold's inner OOF predictions. Outer-test data were loaded only after final
training was fixed.

| Metric | Pooled outer test | 95% stratified subject bootstrap CI |
|---|---:|---:|
| Accuracy | 0.6615 | not primary |
| Balanced accuracy | 0.6116 | [0.5594, 0.6652] |
| Macro precision | 0.6038 | not primary |
| Macro recall | 0.6116 | not primary |
| Macro-F1 | 0.6064 | [0.5563, 0.6581] |
| AUROC | 0.6347 | [0.5716, 0.6973] |
| PD recall | 0.7319 | not primary |
| DD recall | 0.4912 | not primary |
| NLL | 0.6756 | not primary |
| two-class Brier | 0.4630 | not primary |

Confusion matrix (PD, DD): `[[202, 74], [58, 56]]`.

The five thresholded outer-fold BA values are 0.6071, 0.6538, 0.5953, 0.6545
and 0.5455 (mean 0.6112, population SD 0.0407). Default threshold 0.5 gives
pooled BA 0.6134, so inner-OOF threshold selection does not improve the final
estimate.

Against the frozen M0 SubjectMFAM outer result, V8 changes pooled BA by +0.0181,
Macro-F1 by +0.0079 and AUROC by +0.0330. Paired stratified-bootstrap 95% CIs
for these deltas are [-0.0410, 0.0788], [-0.0538, 0.0714] and
[-0.0362, 0.1025]. The intervals cross zero; the outer evidence does not establish
a statistically stable superiority over M0.

H1 remains an inner-development reference (BA 0.6918, AUROC 0.7515), not an
outer-tested result, and must not be compared as though it had passed the same
final evaluation.

## Negative results and stopping decision

- activity dropout, sensor-separated stems and wrist rotation did not pass the
  retention gate;
- an end-to-end full-spectrum branch regressed;
- moment-only, stronger weight decay and mean activity pooling did not replace V8;
- fold-train-only masked reconstruction learned its auxiliary task but improved
  seed-42 BA by only 0.0055 in 7/15 folds;
- a learnable relative-energy filter bank changed BA by only +0.0002 and worsened
  accuracy, F1 and calibration.

These are multiple independent, reasonable hypotheses without stable material
gain. Further tuning on the same 15 folds is stopped. The reliable conclusion is
not “nearly 80%”; it is that current PADS-only pure-deep performance lies around
the low-to-mid 0.60s and is sensitive to subject composition.

## Scientific interpretation

The development-to-outer drop (0.6682 to 0.6116 for seed 42) shows that inner-fold
model selection substantially overstates generalization. DD is heterogeneous and
small (114 subjects across several subtypes), with particularly small atypical-PD
and MS groups. V8 and H1 also share many persistently difficult subjects, and a
fixed diagnostic probability average reaches only 0.6987 BA on development data.
This makes an honest 0.80 claim implausible without materially new information.

## Recommended next research step

Do not continue architecture or threshold search on this cohort. A defensible next
phase requires at least one of:

1. additional unlabeled wearable data from independent subjects for pretraining;
2. a genuinely new cohort or centre for external validation;
3. better phenotype labels or enough subjects for DD-subtype-aware modelling;
4. the thesis target itself: continuous UPDRS/item-score labels, evaluated with
   subject-independent MAE/RMSE, ICC and Bland-Altman analysis.

V8, balanced-v3, M0, H1 and MiniRocket should remain frozen named baselines. V8
can seed future representation work, but the current outer evidence does not
justify calling it a generally strong PD-vs-DD backbone.

## Reproducibility evidence

- frozen split SHA-256:
  `b3c52317cb12b73c66046bbd7c50c94a7e707a64a38d3ad4f0fd4bc0c6ba6b3e`;
- final frozen-manifest SHA-256:
  `1d91164f30cf82aeabea58d413d5561edb6ef61a079a939106c7af9ff5e9a6cc`;
- final prediction SHA-256:
  `f3c8d7e8dcff5197cd7cc8aecc7ec1a77453d781b0b357291e2d0941d8a948c2`;
- final artifact hash verification: all entries passed;
- test suite after implementation: 105 passed;
- post-freeze paired analysis:
  `/home/zyt/MFAM/outputs/pads_classification/pure_deep_v8_final_analysis_20260916/final_paired_analysis.json`.
