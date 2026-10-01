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
- **Correction on device interpretation (2026-09-16):** the paper and PhysioNet
  methods state that all participants used two Apple Watch Series 4 devices, while
  the released official observation JSON contains both `Apple Watch Series 3` and
  `Apple Watch Series 4` in `device_id`. Representative local files match the
  official PhysioNet SHA-256 checksums exactly. Therefore the field is treated only
  as an inconsistent metadata/acquisition-batch proxy, not proof of physical watch
  generation. The `device_id`-only BA/AUROC of 0.6203 remains a negative-control
  warning, but prior wording that asserted confirmed hardware-generation confounding
  is withdrawn. Domain-adversarial training based on this unverified label is not pursued.
