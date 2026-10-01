# V3 SubjectMFAM targeted ablation development report

> All numbers are development inner-CV estimates. No new outer-test training, loading, prediction, threshold selection, or evaluation was performed.

## Evidence boundary

- Preregistered plan SHA-256: `61dc8ef8923c7d8cadfee7615f5bd0247486a3ec619d14295bff2d29c45de448`.
- The 1560 validation rows contain repeated observations of 390 subjects; uncertainty uses subject-cluster bootstrap, never independent-row inference.
- M0, H1 and MiniRocket are read-only references from the prior audited run.
- Primary comparison uses default threshold 0.5; cross-fitted thresholds use only the other two inner validation folds in the same outer context.

## Development leaderboard

| Model | Modification | BA | Delta BA vs M0 | AUROC | Delta AUROC | Macro-F1 | PD Recall | DD Recall | Params |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H1_handcrafted_logistic | reference | 0.6918 | +0.1021 | 0.7515 | +0.1385 | 0.6858 | 0.7936 | 0.5900 | reference |
| R2 | R2 | 0.6596 | +0.0699 | 0.7152 | +0.1021 | 0.6696 | 0.8995 | 0.4198 | 123000 |
| MR1_minirocket_multivariate_ridge | reference | 0.6556 | +0.0660 | 0.7393 | +0.1262 | 0.6664 | 0.8859 | 0.4254 | reference |
| R1 | R1 | 0.6124 | +0.0227 | 0.6316 | +0.0185 | 0.6170 | 0.9220 | 0.3028 | 181827 |
| M0_subject_mfam_frozen_inner_reference | frozen M0 | 0.5897 | +0.0000 | 0.6131 | +0.0000 | 0.5852 | 0.9221 | 0.2573 | reference |
| A1 | A1 | 0.5850 | -0.0047 | 0.6313 | +0.0183 | 0.5823 | 0.9194 | 0.2505 | 121906 |
| N1 | N1 | 0.5663 | -0.0234 | 0.6205 | +0.0075 | 0.5337 | 0.9176 | 0.2151 | 121906 |
| A2 | A2 | 0.5468 | -0.0429 | 0.5703 | -0.0428 | 0.5157 | 0.9665 | 0.1271 | 49175 |

## Preregistered combination decision

Selected combinations: none. Eligibility thresholds were not relaxed.

## Answers to the 12 required questions

1. **Is representation the largest bottleneck?** Best representation delta: R1=+0.0227 BA, R2=+0.0699 BA. Interpretation must follow these observed effects rather than the prior hypothesis.
2. **Does full-band recover information?** R1 BA=0.6124, AUROC=0.6316, DD recall=0.3028; paired BA improvement in 12/15 folds.
3. **Does handcrafted residual recover information?** R2 BA=0.6596, AUROC=0.7152; H1 BA=0.6918. R2 is not called synergistic unless it exceeds H1 with supporting ranking evidence.
4. **Is Hard Top-K harmful?** A1 delta BA=-0.0047, delta AUROC=+0.0183, DD recall=0.2505.
5. **Is activity attention necessary?** A2 delta BA=-0.0429, delta AUROC=-0.0428; it removes activity-attention parameters.
6. **Does BatchNorm affect stability?** N1 delta BA=-0.0234; best epochs 1-40, validation BA SD=0.0628, train-validation gap mean=0.1912.
7. **Largest AUROC gain:** R2, delta AUROC=+0.1021.
8. **Threshold-only behavior:** compare `cross_fitted_threshold_summary.csv` with default metrics; a BA change without AUROC gain is reported as threshold behavior, not representation recovery.
9. **Largest DD-recall improvement:** R2, DD recall=0.4198 versus M0=0.2573.
10. **Subtype recovery:** Atypical best=R2 (0.317); Other best=R2 (0.392); MS best=R2 (0.182). All are exploratory cluster-aware estimates.
11. **Can a simple MFAM variant approach H1?** Best trained variant is R2, BA=0.6596; remaining gap to H1=-0.0322.
12. **Minimal next-version candidate:** R2 is the highest-BA preregistered trained candidate. Final recommendation also considers parameter count, AUROC, fold consistency and cluster interval; this report does not freeze or outer-test it.

## Stability, paired and uncertainty artifacts

- `paired_fold_differences.csv`: all 15 paired fold differences.
- `subject_cluster_bootstrap_delta_ba.csv`: 2000 subject-cluster bootstrap replicates summarized as percentile intervals.
- `training_stability.csv`: epochs, train-validation gaps, runtime, gradients, attention entropy and instance utilization.
- `dd_subtype_subject_cluster_summary.csv`: subtype N, four-context prediction summaries and subject-cluster intervals.
- `r1_branch_diagnostics.csv`: branch activation, cosine and frequency-only/full-band-only inference diagnostics.

## Stop condition

The preregistered first round and allowed combinations are complete. No third round, loss tuning, new candidate, or outer-test run was started.
