# PADS pure-deep phase-1 summary

Date: 2026-09-16

## Protocol boundary

- Task: subject-level PD versus differential-diagnosis classification.
- Data: 390 PADS subjects, 11 activities, bilateral six-channel wrist IMU.
- Development: fixed 5 outer contexts × 3 inner validation folds.
- Outer-test loaders, signals and predictions were never accessed by the pure-deep
  development runner.
- Primary metric: default-threshold balanced accuracy. No seed or threshold was
  selected using held-out test feedback.

## Selected candidate

`PureDeepPool-normfree-moments-v8` combines:

1. a shared small TCN for local temporal patterns;
2. attention/mean/std pooling over TCN-learned feature maps;
3. a normalization-free learnable convolution bank (kernels 1/15/63);
4. mean/std pooling only over that bank's learned feature maps;
5. bilateral fusion and learned activity attention.

It contains 71,026 parameters and no predefined RMS, peak-frequency, band-energy
or other handcrafted feature input.

Across predefined seeds 42/43/44:

| Metric | Mean ± sample SD |
|---|---:|
| Accuracy | 0.7217 ± 0.0127 |
| Balanced accuracy | 0.6572 ± 0.0123 |
| Macro precision | 0.6715 ± 0.0115 |
| Macro-F1 | 0.6565 ± 0.0132 |
| AUROC | 0.6593 ± 0.0229 |
| PD recall | 0.8125 ± 0.0241 |
| DD recall | 0.5020 ± 0.0333 |
| NLL | 0.6335 ± 0.0144 |
| Brier | 0.4287 ± 0.0082 |

Against the three correspondingly seeded balanced-v3 runs, v8 improves matched
fold-average BA by 0.0144 (11/15 folds), macro-F1 by 0.0132 (10/15), AUROC by
0.0085 (8/15), and DD recall by 0.0300 (10/15).

## What the experiments established

- Fixed FFT expansion failed; the H1 advantage is not explained by adding a
  global learnable spectrum alone.
- A feature-group diagnostic showed that H1 time-domain information is stronger
  than its spectral-only subset, relative band fractions are useful, absolute
  band powers can be redundant, and derived vector magnitude is not the main gain.
- Preserving global distribution information in normalization-free learned feature
  maps produced the only stable architecture improvement in this phase.
- Removing the local TCN lowered performance: local and global learned features
  are complementary.
- Stronger weight decay did not reduce variance or improve calibration.
- Parameter-free activity averaging improved BA/AUROC/DD recall in one seed but
  increased fold variance and lowered accuracy/macro-F1; it is an ablation, not
  the selected backbone.
- Rotation, activity dropout and early Acc/Gyro separation did not meet retention
  criteria.

## Honest performance interpretation

The selected pure model is a reproducible improvement over M0/R1 and balanced-v3,
but it remains below the handcrafted H1 reference (BA 0.6918, AUROC 0.7515).
The current evidence does not support an 0.80 balanced-accuracy claim. Continuing
to tune the same 15 folds would increase development-set overfitting risk.

The published methods state that every participant wore Apple Watch Series 4,
whereas official released JSON contains Series 3 and Series 4 strings in
`device_id`. This field is therefore retained only as an inconsistent metadata or
acquisition-batch negative control, not asserted to be physical hardware generation.

## Next justified phase

1. V8, balanced-v3, H1 and MiniRocket are now frozen named baselines.
2. Fold-train-only masked reconstruction was tested in v12: it improved seed-42
   BA by only 0.0055 in 7/15 folds and failed the preregistered replication gate.
3. A learnable relative-energy filter bank was tested in v13: BA changed by only
   +0.0002 while accuracy, F1 and calibration worsened. This direction also stops.
4. V8's one-time nested outer evaluation gives pooled BA 0.6116 and AUROC 0.6347;
   no further PADS-only model tuning is permitted after this release.
5. Any next model phase requires genuinely new subjects/data, or transfer to the
   thesis target of continuous UPDRS/item-score regression with MAE/RMSE, ICC and
   Bland–Altman analysis.
