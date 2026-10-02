# Item 03 — A1 capacity-matched mean residual

Decision: **REJECT** under the preregistered gate. All 45 development stages completed; baseline source/checkpoint/prediction/normalization hashes remain unchanged. No outer evaluation.

Files: scripts/window_hooks.py, run_inner.py, run_matrix.py, test_branch.py, analyze_variant.py and finalize_item.py; analysis/a1_mean_* aggregate CSV/JSON and this report. Historical model code stays unchanged. Checkpoints, caches and per-subject predictions are server-only.

Tests: all 45 baseline checkpoints retain exactly unchanged logits at zero initialization; real inner-train batch [8,11,2,6,2000]; invalid-window/wrist/activity isolation; finite nonzero output gradients; hidden gradient zero initially and nonzero after output update; two single-batch updates; exact checkpoint reload. Formal one-epoch/one-batch smoke passed with original train-only normalization SHA. All final checkpoint weights are finite and learned output branches are nonzero. Smoke performance did not enter retention.

Original WSSL classifier: 143,172 trainable parameters. Candidate: 151,948 (+8,776). HarNet10 stays frozen (10,457,408). Parameter-free LayerNorm, 1024→8→GELU→64, zero output initialization; branch enabled only for actual double windows and valid wrist/activity.

| Configuration | Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.745604 | 0.719644 | 0.759553 | 0.706589 | 0.782344 | 0.656945 |
| a1_mean | 0.744948 | 0.720714 | 0.761608 | 0.706817 | 0.779316 | 0.662111 |

| Reference | Metric | Mean delta | Improved /15 | 95% split-bootstrap CI | Cohen dz | Primary BH q |
|---|---|---:|---:|---|---:|---:|
| baseline | accuracy | -0.000656 | 8 | [-0.006701, +0.004270] | -0.059 | — |
| baseline | ba | +0.001069 | 8 | [-0.001282, +0.003377] | +0.224 | 0.3305 |
| baseline | auroc | +0.002054 | 11 | [-0.000620, +0.004804] | +0.369 | 0.3111 |
| baseline | macro_f1 | +0.000228 | 8 | [-0.003974, +0.003522] | +0.030 | — |
| baseline | pd_recall | -0.003028 | 8 | [-0.016953, +0.009700] | -0.111 | — |
| baseline | dd_recall | +0.005166 | 5 | [-0.009553, +0.020089] | +0.169 | — |

| Configuration | BA seed-mean SD | AUROC seed-mean SD | BA within-split seed SD | AUROC within-split seed SD | Mean pair prediction agreement | Median best epoch |
|---|---:|---:|---:|---:|---:|---:|
| a1_mean | 0.004687 | 0.004630 | 0.018568 | 0.022587 | 0.827181 | 10 |
| baseline | 0.006714 | 0.001812 | 0.021603 | 0.021741 | 0.821982 | 10 |

Gate results: ba_mean_positive=True, ba_improved_10_of_15=False, ba_ci_low_positive=False, auroc_mean_positive=True, auroc_improved_10_of_15=True, macro_f1_non_decrease=True, pd_recall_drop_at_most_001=True, dd_recall_drop_at_most_001=True, ba_within_split_seed_sd_at_most_125x=True, auroc_within_split_seed_sd_at_most_125x=True.

Three seed metrics are averaged within each split before paired analysis. The 15 splits share subjects; bootstrap intervals, Cohen dz, Wilcoxon and BH results describe repeated development robustness, not 15 independent population samples or external validation. No 45-run or subject-pair inflation. Allowed average PD/DD recall drop remains 0.01, fixed before results. No threshold/recipe/dimension search or candidate combinations.
