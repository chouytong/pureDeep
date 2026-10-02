# Item 07 — conditional DD source-category auxiliary supervision

Decision: **REJECT** under the original preregistered gate; 45/45 complete. Original baseline hashes unchanged. No outer evaluation.

Files: C1_PROTOCOL.md, scripts/dd_aux_hooks.py, run_aux_inner.py, test_dd_aux.py, finalize_aux_item.py; run_matrix.py/analyze_variant.py extensions; analysis/c1_dd_aux_* aggregate results. Per-subject labels/predictions/checkpoints stay on server and are excluded from Git.

Eligibility: item06 Other–ET diagnosis-category consensus recall difference passed bootstrap and BH criteria; all source labels consistent and category N≥10. Categories are broad source diagnoses; no fine-grained clinical subtype labels invented. Use all four categories, one fixed auxiliary weight .1; no category, weight, sampling or head search.

Original WSSL paths/inputs/normalization and balanced CE unchanged. Training auxiliary Linear(258,4) taps original subject embedding: 1,036 added parameters, total 144,208 trainable. Loss is original main CE + .1×unweighted subtype CE on DD training samples in each batch. PD-only batch has no auxiliary loss, gradient or optimizer update. Validation and inference have no subtype targets/output/loss; the original PD/DD decision is unchanged. Frozen HarNet cache unchanged. No A1/A2/B1 combination.

Tests: exact original inference for all 45 frozen baseline checkpoints; real [8,11,2,6,2000] input and [8,4] training auxiliary logits; DD auxiliary/local gradients finite nonzero, PD-only auxiliary gradient absent and weights unchanged, invalid masks isolated; single-batch update; exact checkpoint reload with empty training-ID/label sets; zero validation auxiliary loss. Formal smoke passed before full training. All final checkpoints finite, correctly dimensioned, and every validation epoch has auxiliary loss 0.

| Configuration | Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.745604 | 0.719644 | 0.759553 | 0.706589 | 0.782344 | 0.656945 |
| c1_dd_aux | 0.742770 | 0.718603 | 0.762054 | 0.704038 | 0.777107 | 0.660099 |

| Metric | Delta vs baseline | Improved /15 | 95% split-bootstrap CI | Cohen dz | Primary BH q |
|---|---:|---:|---|---:|---:|
| accuracy | -0.002834 | 6 | [-0.012399, +0.007019] | -0.144 | — |
| ba | -0.001041 | 7 | [-0.005805, +0.004058] | -0.103 | 0.5245 |
| auroc | +0.002500 | 9 | [-0.005091, +0.009942] | +0.160 | 0.5245 |
| macro_f1 | -0.002551 | 5 | [-0.010005, +0.005202] | -0.163 | — |
| pd_recall | -0.005237 | 8 | [-0.023687, +0.014694] | -0.131 | — |
| dd_recall | +0.003154 | 9 | [-0.015579, +0.019881] | +0.087 | — |

| Configuration | BA seed-mean SD | AUROC seed-mean SD | BA within-split seed SD | AUROC within-split seed SD | Pair prediction agreement | Median best epoch |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.006714 | 0.001812 | 0.021603 | 0.021741 | 0.821982 | 10 |
| c1_dd_aux | 0.004516 | 0.004531 | 0.018365 | 0.023214 | 0.817376 | 10 |

Gate: ba_mean_positive=False, ba_improved_10_of_15=False, ba_ci_low_positive=False, auroc_mean_positive=True, auroc_improved_10_of_15=False, macro_f1_non_decrease=False, pd_recall_drop_at_most_001=True, dd_recall_drop_at_most_001=True, ba_within_split_seed_sd_at_most_125x=True, auroc_within_split_seed_sd_at_most_125x=True.

Three seeds aggregate within each of 15 overlapping splits; statistical intervals/BH/effect sizes are repeated-development evidence, not external confirmation. All permitted average recall drops stay .01. No hyperparameter/threshold adjustment after results. Item08 applies only if a candidate passes the existing retention gate.
