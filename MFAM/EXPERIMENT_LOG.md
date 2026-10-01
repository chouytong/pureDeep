# PADS pure-deep development log

All reported values are actual means over the fixed 5 outer contexts x 3 inner
validation folds. Outer-test signals, loaders and predictions were not used.

| Run | Hypothesis / controlled change | Params | BA | AUROC | Macro-F1 | DD recall | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| M0 | Frozen historical MFAM reference | 121,906 | 0.5897 | 0.6131 | 0.5852 | 0.2573 | Baseline |
| R1 | Add full-band neural residual path | 181,827 | 0.6124 | 0.6316 | 0.6170 | 0.3028 | Reproduced baseline |
| PureDeepPool-v1 | Replace fixed-band/Hard Top-K path with TCN and learned-feature attention+mean+std pooling | 188,578 | 0.6188 | 0.6382 | 0.6214 | 0.3598 | Reject as new best: BA gain vs R1 0.0064 |
| PureDeepPool-mean | Remove learned-feature std pooling only | 155,682 | 0.6125 | 0.6538 | 0.6164 | 0.3437 | Reject: lower BA; std effect is metric-inconsistent |
| PureDeepPool-small-v2 | Reduce feature and activity-attention dimensions 128→64 | 53,346 | 0.6292 | 0.6388 | 0.6380 | 0.3427 | Retain: BA +0.0168 vs R1, 11/15 fold wins |
| PureDeepPool-balanced-v3 | Add fold-train-only balanced CE to small-v2 | 53,346 | 0.6464 | 0.6519 | 0.6467 | 0.4721 | Current candidate: BA/AUROC wins in 11/15 folds vs v2 |
| PureDeepPool-activity-dropout-v4 | Add train-only activity dropout p=0.15 to balanced-v3 | 53,346 | 0.6508 | 0.6492 | 0.6553 | 0.4591 | Reject as new best: BA +0.0043; AUROC and DD recall decrease |
| PureDeepPool-split-sensor-v5 | Replace shared input stem with separate Acc/Gyro shallow stems | 50,466 | 0.6494 | 0.6527 | 0.6431 | 0.5288 | Reject: BA +0.0030 in 7/15 folds; accuracy/F1/calibration worsen |
| PureDeepPool-rotation-v6 | Add train-only wrist-consistent 3-D rotation (±15°) to balanced-v3 | 53,346 | 0.6505 | 0.6491 | 0.6493 | 0.4828 | Reject: BA +0.0040; AUROC/calibration worsen and DD recall wins only 5/15 folds |
| PureDeepPool-time-spectrum-v7 | Add differentiable full-activity rFFT/log-magnitude and learnable frequency encoder | 67,299 | 0.6376 | 0.6404 | 0.6358 | 0.4718 | Reject: BA −0.0088, AUROC −0.0115; only 4/15 BA fold wins |
| PureDeepPool-normfree-moments-v8 | Add a normalization-free learnable multi-scale convolution bank and mean/std pooling of its learned feature maps | 71,026 | 0.6572±0.0123 | 0.6593±0.0229 | 0.6565±0.0132 | 0.5020±0.0333 | Retain as current candidate: three-seed matched BA +0.0144 vs v3, 11/15 fold wins |
| PureDeepPool-moment-only-v9 | Remove the BatchNorm TCN and use the learned-moment branch alone | 50,369 | 0.6557 | 0.6591 | 0.6539 | 0.5116 | Reject: BA −0.0125 and only 5/15 wins vs seed-42 v8; temporal/global branches are complementary |
| PureDeepPool-moments-wd-v10 | Increase AdamW weight decay only, 1e-4→1e-3 | 71,026 | 0.6624 | 0.6647 | 0.6685 | 0.4824 | Reject: BA −0.0058, 3/15 wins; NLL/Brier and fold variance worsen |
| PureDeepPool-mean-activity-v11 | Replace learned activity attention with parameter-free masked mean | 51,031 | 0.6727 | 0.6869 | 0.6647 | 0.5645 | Keep as ablation, not candidate: BA gain only +0.0045 and fold variance/calibration worsen |
| PureDeepPool-masked-pretrain-v12 | Fold-train-only masked sequence reconstruction before unchanged V8 fine-tuning | 71,026 downstream | 0.6736 | 0.6719 | 0.6752 | 0.5282 | Reject: BA +0.0055, only 7/15 wins; fails the preregistered replication gate |
| PureDeepPool-relative-energy-v13 | Add a shared learnable filter bank with within-channel relative energy normalization | 79,802 | 0.6684 | 0.6677 | 0.6633 | 0.5398 | Reject: BA +0.0002; accuracy/F1/calibration worsen |

## Evidence notes

- Fixed split SHA-256: `b3c52317cb12b73c66046bbd7c50c94a7e707a64a38d3ad4f0fd4bc0c6ba6b3e`.
- All model selection used default-threshold validation balanced accuracy.
- Balanced-v3 derives weights independently per inner training fold. Example fold
  0.0: PD=146, DD=59, weights=[0.7020548, 1.7372881].
- Balanced-v3 trades PD recall (0.9158→0.8208) and accuracy (0.7482→0.7187)
  for DD recall (0.3427→0.4721); it is not described as a free improvement.
- Activity-dropout-v4 improves accuracy and macro-F1 slightly, but its BA gain is
  below the predeclared 0.01 retention margin and is not coherent across AUROC and
  DD recall. It remains a robustness ablation, not the selected candidate.
- A leakage-safe cross-fitted threshold analysis was performed for balanced-v3:
  each held inner fold used a threshold fitted only on the other two validation
  folds in the same outer context. It reduced BA from 0.6464 to 0.6399 and
  macro-F1 from 0.6467 to 0.6368 while changing PD/DD recall to 0.7990/0.4808.
  The default 0.5 threshold is therefore retained; threshold tuning is not counted
  as a representation improvement.
- Balanced-v3 seed stability (42/43/44): BA 0.6464/0.6369/0.6453,
  seed mean 0.6429±0.0052; AUROC seed mean 0.6508±0.0051. After averaging
  the three seeds within each fold, BA exceeds R1 in 14/15 folds with mean
  paired delta +0.0305. DD recall rises by +0.1692 on average (14/15 wins).
  The candidate is therefore not a single-initialization result, although its
  absolute discrimination remains modest.
- Split-sensor-v5 raises DD recall by 0.0566 but lowers PD recall by 0.0507,
  accuracy by 0.0193 and macro-F1 by 0.0036 versus seed-42 balanced-v3.
  Its NLL and Brier score also worsen, so early modality separation is rejected.
- Rotation-v6 uses one rotation matrix per wrist, shared across a subject's 11
  activities and applied consistently to Acc and Gyro before normalization;
  validation data are unchanged. The augmentation is physically valid and tested
  for orthogonality/norm preservation, but does not meet the retention margin.
- Activity dropout, sensor-stem separation and small rotation are three consecutive
  reasonable but unsuccessful directions. Architecture search is paused per the
  stop rule; the next step is bottleneck/confound analysis rather than module stacking.
- Time-spectrum-v7 tests a new bottleneck hypothesis after that pause: the v3 TCN
  has an effective receptive field of about one second, so a global differentiable
  spectrum might recover long-duration periodicity. The branch uses no predefined
  bands, peaks, RMS or handcrafted inputs. Despite isolated strong folds, it lowers
  all primary aggregate metrics and worsens calibration. Frequency-bin/range tuning
  is stopped to avoid validation-set search.
- A development-only H1 feature-group diagnostic exactly reproduced H1
  (BA=0.6918, AUROC=0.7515) without accessing outer-test data. Time-domain groups
  alone reach BA=0.6786 versus 0.6570 for spectral groups. Removing absolute band
  power improves BA to 0.7002, while removing band fractions lowers it to 0.6759.
  Six measured axes alone reach 0.6952, so derived vector magnitude is not the main
  advantage. This evidence motivated v8 and argues against further fixed spectral
  expansion.
- Normfree-moments-v8 applies only learned convolutions (kernels 1/15/63) to the
  raw six-channel stream and computes mean/std only on their learned feature maps;
  no RMS, peak frequency, band power or other handcrafted feature is supplied.
  Versus seed-42 balanced-v3 it improves accuracy by +0.0172, BA by +0.0217
  (12/15 wins), macro-F1 by +0.0242 (12/15), AUROC by +0.0131, DD recall by
  +0.0328, NLL by −0.0051 and Brier by −0.0147. The predefined seeds 42/43/44
  yield BA=0.6682/0.6596/0.6440 and AUROC=0.6649/0.6789/0.6342. Their mean BA
  is 0.6572±0.0123. Against the three correspondingly seeded v3 runs, fold-averaged
  BA improves by +0.0144 in 11/15 folds, macro-F1 by +0.0132 in 10/15 folds,
  AUROC by +0.0085 in 8/15 folds and DD recall by +0.0300 in 10/15 folds. V8 is
  retained as the current pure-deep candidate, with seed sensitivity explicitly
  acknowledged; no best-seed selection is used.
- Moment-only-v9 is a structural ablation rather than another stacked module. It
  reduces parameters by 29% relative to v8 but lowers BA by 0.0125, macro-F1 by
  0.0171, AUROC by 0.0058 and worsens Brier by 0.0180. Its DD recall is only
  0.0067 higher and wins 4/15 folds. The normalization-free global branch therefore
  does not explain all v8 gains; local TCN features remain complementary.
- Stronger-weight-decay-v10 leaves the architecture fixed but lowers BA by 0.0058
  with only 3/15 fold wins. AUROC is effectively unchanged (−0.0002), DD recall
  falls by 0.0225, NLL worsens by 0.0165 and Brier by 0.0075. BA fold standard
  deviation rises from 0.0279 to 0.0303. Weight-decay tuning is stopped after this
  predeclared comparison; no validation grid is searched.
- Mean-activity-v11 reduces parameters by 28% and improves BA by 0.0045 (9/15
  wins), AUROC by 0.0220 (10/15) and DD recall by 0.0596. However accuracy falls
  by 0.0185, macro-F1 by 0.0062, BA fold standard deviation rises from 0.0279 to
  0.0417, and Brier slightly worsens. It fails the predeclared stability/calibration
  condition and remains an informative aggregation ablation rather than replacing v8.
- Internal architecture search stops after v11. V8 remains the selected pure-deep
  development candidate. Further material gains should be sought through additional
  data, self-supervised pretraining evaluated under the same folds, or a genuinely
  new validation cohort—not by continuing to tune these 15 development folds.
- **Correction on device interpretation (2026-09-16):** the paper and PhysioNet
  methods state that all participants used two Apple Watch Series 4 devices, while
  the released official observation JSON contains both `Apple Watch Series 3` and
  `Apple Watch Series 4` in `device_id`. Representative local files match the
  official PhysioNet SHA-256 checksums exactly. Therefore the field is treated only
  as an inconsistent metadata/acquisition-batch proxy, not proof of physical watch
  generation. The `device_id`-only BA/AUROC of 0.6203 remains a negative-control
  warning, but prior wording that asserted confirmed hardware-generation confounding
  is withdrawn. Domain-adversarial training based on this unverified label is not pursued.
- Masked-pretrain-v12 uses only each inner fold's training subjects and no diagnosis
  labels, validation subjects or outer-test signals during pretraining. A temporary
  decoder reconstructs the six normalized sensor channels only at 30% masked time
  points arranged in 25-sample contiguous spans; the decoder is discarded and the
  original V8 classifier is fine-tuned with a fresh optimizer. Across the 15 folds,
  masked MSE falls from 0.9677 to 0.7265 on average. Relative to the paired seed-42
  V8 run, BA rises by only 0.0055 (7 wins, 1 tie), macro-F1 by 0.0043, AUROC by
  0.0070, and DD recall by 0.0233; accuracy falls by 0.0019. Outer-context BA
  deltas are −0.0233, +0.0030, +0.0315, −0.0026 and +0.0187. This misses both the
  +0.01 BA and 9/15-fold discovery gates, so seeds 43/44 are not launched and the
  masked-reconstruction direction is stopped without hyperparameter tuning.
- Relative-energy-v13 was motivated by the H1 diagnostic: relative band fractions
  helped while absolute band powers hurt. It uses eight shared, end-to-end learned
  zero-mean/unit-norm temporal filters and normalizes their log energies within each
  sensor channel; no precomputed band power is supplied. It changes seed-42 BA by
  only +0.0002 (8/15 wins), AUROC by +0.0028, but lowers accuracy by 0.0141 and
  macro-F1 by 0.0077 while worsening NLL by 0.0174 and Brier by 0.0215. Its BA
  fold SD is 0.0344 versus V8's 0.0279. The discovery gate fails, so seeds 43/44
  and filter-count/kernel/frequency tuning are not run.
- After v13, V8 was frozen and evaluated exactly once under the full 5x3 nested
  protocol. Its pooled outer-test BA is 0.6116, macro-F1 0.6064 and AUROC 0.6347;
  95% stratified subject-bootstrap intervals are [0.5594, 0.6652], [0.5563,
  0.6581] and [0.5716, 0.6973], respectively. The default-0.5 BA is 0.6134, so
  inner-OOF fold thresholds do not improve the final estimate. The model is not
  modified after this outer-test release.
