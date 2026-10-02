# Item 05 — local raw Acc input control

Decision: **REJECT** under the original preregistered gate; 45/45 complete. Frozen baseline hashes remain unchanged. No outer evaluation.

Files: scripts/prepare_raw_acc.py, raw_acc_hooks.py, run_raw_inner.py, test_raw_acc.py, finalize_raw_item.py; extensions to run_matrix.py/analyze_variant.py; analysis/b1_raw_* aggregate results. Server-only raw arrays are under features/raw_acc and excluded from Git, along with checkpoints/predictions.

Input: only local Acc changes from L1-detrended to original raw post-trim48 samples; original processed Gyro stays bit-identical. All 8,580 wrist records preserve 976/2000 lengths. HarNet mean cache, classifier architecture (143,172 trainable parameters), bilateral/ordered structure, balanced CE, optimizer/scheduler, seeds and 0.5 threshold stay fixed. No A1/A2 branch is present. Raw local Acc has no added clipping/augmentation.

Tests: shape [8,11,2,6,2000], actual activity lengths, invalid activity/wrist mask isolation, finite nonzero local encoder/SSL projection gradients, one-batch training, exact checkpoint reload, frozen SSL features; train-only normalization refitted, same train-ID hashes. Gyro normalization remains exactly equal to baseline in every run; Acc normalization changes. All formal checkpoints verified finite and correctly dimensioned. Formal one-batch smoke passed before full training; smoke metrics did not enter selection.

| Configuration | Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.745604 | 0.719644 | 0.759553 | 0.706589 | 0.782344 | 0.656945 |
| b1_raw | 0.753274 | 0.717325 | 0.756623 | 0.709342 | 0.803990 | 0.630659 |

| Metric | Delta vs baseline | Improved /15 | 95% split-bootstrap CI | Cohen dz | Primary BH q |
|---|---:|---:|---|---:|---:|
| accuracy | +0.007670 | 10 | [-0.004535, +0.019126] | +0.317 | — |
| ba | -0.002320 | 7 | [-0.009392, +0.004449] | -0.164 | 0.7615 |
| auroc | -0.002930 | 7 | [-0.013261, +0.006223] | -0.146 | 0.7615 |
| macro_f1 | +0.002753 | 9 | [-0.006969, +0.011400] | +0.146 | — |
| pd_recall | +0.021646 | 9 | [-0.002756, +0.044963] | +0.445 | — |
| dd_recall | -0.026286 | 3 | [-0.049631, -0.003442] | -0.552 | — |

| Configuration | BA seed-mean SD | AUROC seed-mean SD | BA within-split seed SD | AUROC within-split seed SD | Pair prediction agreement | Median best epoch |
|---|---:|---:|---:|---:|---:|---:|
| b1_raw | 0.004872 | 0.001414 | 0.020192 | 0.020876 | 0.801521 | 11 |
| baseline | 0.006714 | 0.001812 | 0.021603 | 0.021741 | 0.821982 | 10 |

Gate: ba_mean_positive=False, ba_improved_10_of_15=False, ba_ci_low_positive=False, auroc_mean_positive=False, auroc_improved_10_of_15=False, macro_f1_non_decrease=True, pd_recall_drop_at_most_001=True, dd_recall_drop_at_most_001=False, ba_within_split_seed_sd_at_most_125x=True, auroc_within_split_seed_sd_at_most_125x=True.

Three seeds aggregate within each of 15 overlapping splits. Bootstrap intervals/BH/effect sizes are repeated-development robustness, not independent external evidence. Recall allowance 0.01 unchanged. No input/loss/hyperparameter combinations or filter search. Next numbered item is current frozen WSSL error analysis, using independent subjects and verified subtype metadata.
