# Item 04 — A2 front/back dynamics residual

Decision: **REJECT** under the preregistered gate. All 45 development stages completed; baseline source/checkpoint/prediction/normalization hashes remain unchanged. No outer evaluation.

Files: scripts/window_hooks.py, run_inner.py, run_matrix.py, test_branch.py, analyze_variant.py and finalize_item.py; analysis/a2_delta_* aggregate CSV/JSON and this report. Historical model code stays unchanged. Checkpoints, caches and per-subject predictions are server-only.

Tests: all 45 baseline checkpoints retain exactly unchanged logits at zero initialization; real inner-train batch [8,11,2,6,2000]; invalid-window/wrist/activity isolation; finite nonzero output gradients; hidden gradient zero initially and nonzero after output update; two single-batch updates; exact checkpoint reload. Formal one-epoch/one-batch smoke passed with original train-only normalization SHA. All final checkpoint weights are finite and learned output branches are nonzero. Smoke performance did not enter retention.

Original WSSL classifier: 143,172 trainable parameters. Candidate: 151,948 (+8,776). HarNet10 stays frozen (10,457,408). Parameter-free LayerNorm, 1024→8→GELU→64, zero output initialization; branch enabled only for actual double windows and valid wrist/activity.

| Configuration | Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.745604 | 0.719644 | 0.759553 | 0.706589 | 0.782344 | 0.656945 |
| a1_mean | 0.744948 | 0.720714 | 0.761608 | 0.706817 | 0.779316 | 0.662111 |
| a2_delta | 0.741305 | 0.718099 | 0.762503 | 0.703521 | 0.774137 | 0.662062 |

| Reference | Metric | Mean delta | Improved /15 | 95% split-bootstrap CI | Cohen dz | Primary BH q |
|---|---|---:|---:|---|---:|---:|
| baseline | accuracy | -0.004299 | 6 | [-0.013884, +0.004440] | -0.227 | — |
| baseline | ba | -0.001545 | 7 | [-0.004214, +0.000836] | -0.295 | 0.6271 |
| baseline | auroc | +0.002950 | 10 | [-0.001513, +0.007195] | +0.329 | 0.3752 |
| baseline | macro_f1 | -0.003069 | 6 | [-0.009500, +0.002672] | -0.247 | — |
| baseline | pd_recall | -0.008207 | 7 | [-0.030207, +0.012621] | -0.186 | — |
| baseline | dd_recall | +0.005117 | 7 | [-0.015883, +0.026289] | +0.117 | — |
| a1_mean | accuracy | -0.003643 | 4 | [-0.010241, +0.003182] | -0.267 | — |
| a1_mean | ba | -0.002614 | 3 | [-0.005211, +0.000167] | -0.486 | 0.3752 |
| a1_mean | auroc | +0.000896 | 8 | [-0.004340, +0.005692] | +0.088 | 0.8202 |
| a1_mean | macro_f1 | -0.003296 | 4 | [-0.007776, +0.001230] | -0.356 | — |
| a1_mean | pd_recall | -0.005179 | 4 | [-0.020244, +0.010794] | -0.164 | — |
| a1_mean | dd_recall | -0.000049 | 6 | [-0.016011, +0.015674] | -0.002 | — |

| Configuration | BA seed-mean SD | AUROC seed-mean SD | BA within-split seed SD | AUROC within-split seed SD | Mean pair prediction agreement | Median best epoch |
|---|---:|---:|---:|---:|---:|---:|
| a1_mean | 0.004687 | 0.004630 | 0.018568 | 0.022587 | 0.827181 | 10 |
| a2_delta | 0.006237 | 0.000670 | 0.018435 | 0.023251 | 0.813594 | 10 |
| baseline | 0.006714 | 0.001812 | 0.021603 | 0.021741 | 0.821982 | 10 |

Gate results: ba_mean_positive=False, ba_improved_10_of_15=False, ba_ci_low_positive=False, auroc_mean_positive=True, auroc_improved_10_of_15=True, macro_f1_non_decrease=False, pd_recall_drop_at_most_001=True, dd_recall_drop_at_most_001=True, ba_within_split_seed_sd_at_most_125x=True, auroc_within_split_seed_sd_at_most_125x=True.

Dynamics-specific positive mean BA and AUROC advantage over A1: **False**. A numerical advantage alone does not establish robust dynamics-specific benefit; use paired CIs/counts above.

A1/A2 initialization matches exactly for seeds 42/43/44 (full state hashes in implementation test). Effective eligibility audit: double windows occur in Entrainment, Relaxed and RelaxedTask (780 wrist records each); remaining eight activities have single windows and zero new residual. This experiment therefore tests long-record dynamics in those three fixed activities; no activity selection occurred. See analysis/window_branch_eligibility.csv.

Three seed metrics are averaged within each split before paired analysis. The 15 splits share subjects; bootstrap intervals, Cohen dz, Wilcoxon and BH results describe repeated development robustness, not 15 independent population samples or external validation. No 45-run or subject-pair inflation. Allowed average PD/DD recall drop remains 0.01, fixed before results. No threshold/recipe/dimension search or candidate combinations.
