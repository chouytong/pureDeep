# Phase-2 STR-01 Training and Augmentation Development Study

## Boundary and stage entry

This is a new **development-only** phase. The formal STR-01 architecture (75,524 parameters), fixed PADS PD-versus-DD subject splits, bilateral 11-activity input, wrist/activity order, seeds 42/43/44, and train-only normalization were retained. The previous FOE-01 outer artifacts and results were excluded from all Phase-2 model selection, tuning, augmentation design, debugging, and validation. The original STR/V8/H1 formal archives were not overwritten. The candidate manifest, protocol, stage summary, literature screening, smoke audit, source hashes, all runs, and analysis reside in this independent directory.

The prior formal development table and stopped directions are in `PHASE2_STAGE_SUMMARY.md`. The bounded candidate list and retention gate are in `PHASE2_PROTOCOL.md`; source review for mild, label-aware wearable-IMU augmentation is in `AUGMENTATION_EVIDENCE.md`. In particular, PD motor-state studies show that transformations such as noise, scaling, and crop may change symptom-bearing patterns, so their success on other movement tasks was not assumed for PADS PD-vs-DD.

## Baseline reproduction and training adequacy

The exact formal STR-01 recipe was rerun for all 15 inner development splits and seeds 42/43/44 using only `_run_inner_fold`, never the outer-final routine. All 45 best epochs matched formal STR; 45/45 model-state tensors were bitwise identical, validation DD probabilities matched exactly by subject ID (maximum absolute difference 0), and train-only normalization hashes matched. Full checkpoint-file SHA differs only because the new run name and provenance differ. This is a successful exact reproduction.

Seed-first 15-split means: Accuracy 0.7307, BA 0.6972, AUROC 0.7176, Macro-F1 0.6861, PD Recall 0.7779, DD Recall 0.6165. Across-seed SDs: Accuracy 0.0151, BA 0.0117, AUROC 0.0120, Macro-F1 0.0107, PD Recall 0.0327, DD Recall 0.0408. Across the 15 seed-averaged splits, SDs were 0.0306/0.0316/0.0393/0.0316/0.0440/0.0622 in the same metric order. Best epoch median 12, range 6–31; no run reached the 50-epoch maximum as its best epoch. Mean train/validation loss at selected epoch was 0.3388/0.7147; at stop it was 0.0415/1.0865. In all 45 runs, train loss fell and validation loss rose between selected best epoch and stop. The widening gap supports overfitting risk, not simple undertraining. Extending max epochs or patience was therefore not prioritized; the bounded local study instead tested training regularization, loss weighting, sampling, and mild augmentation.

## Implementation and analysis

The standalone Phase-2 runner invokes frozen production inner-training code with single-factor config overrides or isolated training-only hooks. It never calls `run_nested_training` or outer-final evaluation. New hooks affect only the training fold: validation uses unchanged input, masks, and subject IDs. For each candidate, three deep seeds are aggregated within each of the 15 fixed splits; split-level paired differences, improvement counts, 10,000-draw bootstrap 95% CIs, primary BA/AUROC BH adjustment, and seed SD are computed. The 45 seed-runs, repeated subject appearances, and augmented exposures are not treated as independent evidence. Repeated development comparisons still create selection optimism; no new locked test claim is made here.

Baseline plus 16 registered single-factor candidates were scheduled: LR 1e-4, WD 1e-3, label smoothing 0.05; unweighted CE, sqrt-weighted CE, balanced focal γ=2, effective-number CE β=0.99, and balanced subject sampler; and jitter, scaling, mild temporal displacement, conservative same-class mixup each under equal and DD-only perturbation policies. Baseline train-balanced CE is the weighted CE reference. Prior V8-specific negative batch16/dropout0.2/constant-scheduler/LR3e-4 searches were not repeated as a large STR grid. Epoch cap and patience were evaluated from exact baseline trajectories. Seven implementation smoke runs verified normalization hash, identical validation IDs/labels, complete training, exposure logging, and absence of outer loader. Exact definitions and strengths are in `PHASE2_PROTOCOL.md`.

## Completed result matrix

All 17 configurations completed 45/45 inner runs (765 total). Values below are means over the 15 fixed inner splits after the three seeds are averaged within each split. “BA seed SD” and “DD seed SD” are descriptive SDs across the three seed-level means; they are not independent uncertainty estimates.

| Configuration | Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall | BA seed SD | DD seed SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.7307 | 0.6972 | 0.7176 | 0.6861 | 0.7779 | 0.6165 | 0.0117 | 0.0408 |
| lr_1e4 | 0.7191 | 0.6914 | 0.7104 | 0.6766 | 0.7583 | 0.6245 | 0.0160 | 0.0067 |
| wd_1e3 | 0.7260 | 0.6955 | 0.7189 | 0.6836 | 0.7691 | 0.6218 | 0.0123 | 0.0350 |
| smooth_005 | 0.7187 | 0.6966 | 0.7210 | 0.6800 | 0.7500 | 0.6432 | 0.0100 | 0.0398 |
| ce_unweighted | 0.7399 | 0.6676 | 0.7083 | 0.6717 | 0.8418 | 0.4934 | 0.0144 | 0.0355 |
| ce_sqrtweighted | 0.7358 | 0.6822 | 0.7166 | 0.6801 | 0.8113 | 0.5531 | 0.0153 | 0.0144 |
| focal_balanced | 0.7195 | 0.6979 | 0.7247 | 0.6809 | 0.7503 | 0.6455 | 0.0144 | 0.0215 |
| ce_effective099 | 0.7345 | 0.6875 | 0.7162 | 0.6828 | 0.8008 | 0.5743 | 0.0138 | 0.0199 |
| balanced_sampler | 0.6908 | 0.6790 | 0.7059 | 0.6547 | 0.7080 | 0.6499 | 0.0127 | 0.0095 |
| jitter_equal | 0.7267 | 0.6950 | 0.7201 | 0.6834 | 0.7713 | 0.6187 | 0.0140 | 0.0409 |
| jitter_dd_only | 0.7306 | 0.6950 | 0.7201 | 0.6847 | 0.7811 | 0.6089 | 0.0126 | 0.0201 |
| scaling_equal | 0.7247 | 0.6927 | 0.7173 | 0.6809 | 0.7700 | 0.6153 | 0.0136 | 0.0487 |
| scaling_dd_only | 0.7185 | 0.6915 | 0.7214 | 0.6772 | 0.7568 | 0.6263 | 0.0148 | 0.0468 |
| temporal_equal | 0.7251 | 0.6957 | 0.7206 | 0.6831 | 0.7666 | 0.6248 | 0.0145 | 0.0321 |
| temporal_dd_only | 0.7249 | 0.6954 | 0.7155 | 0.6824 | 0.7666 | 0.6242 | 0.0168 | 0.0281 |
| mixup_equal | 0.7205 | 0.6965 | 0.7208 | 0.6804 | 0.7543 | 0.6387 | 0.0122 | 0.0320 |
| mixup_dd_only | 0.7273 | 0.6909 | 0.7105 | 0.6806 | 0.7784 | 0.6034 | 0.0128 | 0.0197 |

The separate CSV files provide all 15 paired split values, each seed run, best/last train and validation losses, best epoch, and per-epoch augmentation exposures. Cross-loss numerical losses are not directly comparable because their objective scales differ. Across all 765 runs, train loss decreased after the best epoch; validation loss increased in 759. No baseline best epoch reached the 50-epoch cap.

## Paired comparisons with reproduced STR-01

Each entry is mean paired difference on 15 seed-averaged splits, positive-split count, and a 10,000-resample split bootstrap 95% CI. Repeated subjects across splits mean this interval is descriptive for the fixed development protocol, not an external-population CI.

| Configuration | ΔBA; positive splits; 95% CI | ΔAUROC; positive splits; 95% CI | ΔMacro-F1 | ΔPD Recall | ΔDD Recall |
|---|---:|---:|---:|---:|---:|
| lr_1e4 | -0.0057; 5/15; [-0.0129, -0.0003] | -0.0072; 3/15; [-0.0130, -0.0012] | -0.0095 | -0.0195 | +0.0080 |
| wd_1e3 | -0.0017; 7/15; [-0.0052, +0.0015] | +0.0013; 10/15; [-0.0023, +0.0050] | -0.0025 | -0.0087 | +0.0053 |
| smooth_005 | -0.0006; 6/15; [-0.0056, +0.0046] | +0.0034; 9/15; [-0.0005, +0.0076] | -0.0061 | -0.0278 | +0.0267 |
| ce_unweighted | -0.0296; 1/15; [-0.0385, -0.0203] | -0.0094; 5/15; [-0.0186, +0.0002] | -0.0143 | +0.0639 | -0.1231 |
| ce_sqrtweighted | -0.0150; 2/15; [-0.0229, -0.0078] | -0.0010; 6/15; [-0.0077, +0.0055] | -0.0059 | +0.0334 | -0.0634 |
| focal_balanced | +0.0007; 6/15; [-0.0053, +0.0075] | +0.0071; 10/15; [-0.0037, +0.0175] | -0.0052 | -0.0275 | +0.0290 |
| ce_effective099 | -0.0097; 4/15; [-0.0174, -0.0017] | -0.0014; 7/15; [-0.0051, +0.0021] | -0.0033 | +0.0229 | -0.0422 |
| balanced_sampler | -0.0182; 4/15; [-0.0286, -0.0081] | -0.0118; 4/15; [-0.0240, +0.0008] | -0.0314 | -0.0698 | +0.0334 |
| jitter_equal | -0.0022; 5/15; [-0.0059, +0.0014] | +0.0025; 9/15; [-0.0021, +0.0073] | -0.0027 | -0.0066 | +0.0022 |
| jitter_dd_only | -0.0022; 8/15; [-0.0062, +0.0014] | +0.0025; 9/15; [-0.0033, +0.0085] | -0.0014 | +0.0032 | -0.0076 |
| scaling_equal | -0.0045; 4/15; [-0.0083, -0.0010] | -0.0004; 7/15; [-0.0045, +0.0038] | -0.0051 | -0.0079 | -0.0012 |
| scaling_dd_only | -0.0057; 3/15; [-0.0095, -0.0020] | +0.0037; 9/15; [-0.0000, +0.0076] | -0.0089 | -0.0211 | +0.0098 |
| temporal_equal | -0.0015; 6/15; [-0.0059, +0.0029] | +0.0029; 10/15; [-0.0020, +0.0086] | -0.0029 | -0.0113 | +0.0083 |
| temporal_dd_only | -0.0018; 5/15; [-0.0090, +0.0062] | -0.0021; 5/15; [-0.0058, +0.0017] | -0.0037 | -0.0113 | +0.0077 |
| mixup_equal | -0.0007; 6/15; [-0.0059, +0.0045] | +0.0031; 10/15; [-0.0015, +0.0077] | -0.0056 | -0.0236 | +0.0222 |
| mixup_dd_only | -0.0063; 3/15; [-0.0117, -0.0012] | -0.0072; 3/15; [-0.0134, -0.0000] | -0.0055 | +0.0006 | -0.0131 |

No positive BA or AUROC comparison survived the primary-family BH-FDR adjustment; full Wilcoxon p and BH q values are in `analysis/variant_paired_vs_baseline.csv`. The three q<0.05 primary comparisons are BA deterioration for unweighted CE, sqrt-weighted CE, and balanced sampling. The joint retention gate in `PHASE2_PROTOCOL.md` was met by **0/16** candidates. In particular, the largest observed AUROC gain, focal-balanced (+0.0071, 10/15; CI −0.0037 to +0.0175), was accompanied by essentially unchanged BA (+0.0007, 6/15), lower Macro-F1 (−0.0052), lower PD Recall (−0.0275), and higher AUROC seed SD (0.0224 versus 0.0120). It is not selected. No two augmentation operators passed the gate, so no combination experiment was authorized by the protocol.

## Imbalance, augmentation exposure, and trade-offs

The original dataset has 276 PD and 114 DD subjects. The mean train fold contains about 147 PD and 61 DD distinct subjects. Perturbations did not add independent subjects. The equal-policy perturbation probability was 0.5 for each class; DD-only probability was 1.0 for DD and 0 for PD. The actual mean per-epoch exposures were:

| Policy | PD exposures | DD exposures | Augmented PD | Augmented DD | Augmented/original | DD exposure share |
|---|---:|---:|---:|---:|---:|---:|
| balanced_sampler | 104.1 | 103.9 | 0.0 | 0.0 | 0.000 | 0.499 |
| jitter_equal | 147.2 | 60.8 | 72.9 | 30.2 | 0.991 | 0.292 |
| jitter_dd_only | 147.2 | 60.8 | 0.0 | 60.8 | 0.413 | 0.292 |
| scaling_equal | 147.2 | 60.8 | 74.1 | 30.5 | 1.022 | 0.292 |
| scaling_dd_only | 147.2 | 60.8 | 0.0 | 60.8 | 0.413 | 0.292 |
| temporal_equal | 147.2 | 60.8 | 73.4 | 30.3 | 1.003 | 0.292 |
| temporal_dd_only | 147.2 | 60.8 | 0.0 | 60.8 | 0.413 | 0.292 |
| mixup_equal | 147.2 | 60.8 | 73.0 | 29.8 | 0.986 | 0.292 |
| mixup_dd_only | 147.3 | 60.8 | 0.0 | 59.8 | 0.403 | 0.292 |

DD-only augmentation changed the fraction of perturbed DD examples, **not** the PD/DD subject sampling ratio (still approximately 29.2% DD exposure). Only the balanced sampler made DD exposure approximately 49.9%, by resampling existing training subjects. That raised DD Recall by +0.0334, but lowered PD Recall by −0.0698, BA by −0.0182, AUROC by −0.0118, and Macro-F1 by −0.0314. The existing balanced CE therefore protects DD recognition better than no or weaker class weighting in this bounded comparison: removing weights raised Accuracy +0.0091 but reduced DD Recall −0.1231 (CI −0.1628 to −0.0843), BA −0.0296 (CI −0.0385 to −0.0203), and Macro-F1 −0.0143. This establishes sensitivity to the loss balance; it does not establish imbalance as the sole or main remaining ceiling.

Label smoothing 0.05 improved DD Recall +0.0267 (CI +0.0075 to +0.0462) but reduced PD Recall −0.0278, Macro-F1 −0.0061, and did not improve BA/AUROC jointly. DD-only scaling had a small descriptive AUROC gain (+0.0037; CI approximately 0 to +0.0076), yet BA −0.0057, Macro-F1 −0.0089, and PD Recall −0.0211. Equal conservative mixup had AUROC +0.0031 with a CI spanning zero, BA −0.0007, Macro-F1 −0.0056, and PD Recall −0.0236. DD-only mixup degraded BA and AUROC. Jitter and mild temporal displacement also failed the joint gate. No augmentation produced a stable BA/AUROC/Macro-F1 gain with DD improvement and preserved PD Recall. Accuracy-only changes were not selected.

## Final Phase-2 decision and limits

| Comparison point | Configuration | Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| Original STR-01 | frozen original STR-01 recipe | 0.7307 | 0.6972 | 0.7176 | 0.6861 | 0.7779 | 0.6165 |
| Hyperparameter-optimized STR-01 | frozen original STR-01 recipe | 0.7307 | 0.6972 | 0.7176 | 0.6861 | 0.7779 | 0.6165 |
| Augmentation STR-01 | frozen original STR-01 recipe | 0.7307 | 0.6972 | 0.7176 | 0.6861 | 0.7779 | 0.6165 |
| Final Phase-2 development candidate | frozen original STR-01 recipe | 0.7307 | 0.6972 | 0.7176 | 0.6861 | 0.7779 | 0.6165 |

No alternative earned retention, so the paired improvement of the final Phase-2 candidate over original STR-01 is exactly zero by construction; its seed SD and split SD remain those in the baseline reproduction section. The local study identifies balanced CE and the existing stopping regime as important, and gives no evidence that the tested LR, WD, smoothing, loss/sampling, or mild augmentation changes improve the joint development objective. It cannot prove a global training optimum: AdamW, scheduler, batch size, and architecture-fixed dropout were not newly varied in this Phase-2 matrix; historical V8-only negatives do not establish STR-specific effects. The baseline loss trajectory does not justify more epochs or patience. Further broad recipe or synthetic-data searching on the same 15 splits would increase selection optimism; this completed phase does not support such a continuation. No Phase-2 configuration has been checked against or chosen using FOE-01 outer information. Formal STR-01 and its old outer evaluation remain unchanged.

## Integrity and artifacts

`analysis/final_integrity_audit.json`: 765/765 complete; 765/765 identical train/validation subject lists, validation ID-label mappings, and train-only normalization hashes to baseline; 765/765 unique validation predictions from validation subjects only; 765/765 explicit `outer_test_loader_created=false`; source-file SHA-256 records match; zero audit issues. Seven smoke runs passed before full execution. `analysis/joint_gate.json` contains all 16 gate decisions (all false). Detailed source manifest and exact parameter settings are in `manifest.json` and `PHASE2_PROTOCOL.md`; all outputs are isolated in this Phase-2 directory.

## Direct answers to the Phase-2 research questions

1. **Reproducibility:** yes, exact at model-state, prediction, and best-epoch level in all 45 inner runs.
2. **Training adequacy:** the selected checkpoints occur well before the epoch cap; later train/validation loss divergence indicates overfitting risk. The tested local recipe alternatives did not improve the joint objective. Global hyperparameter optimality is not established because several factors were intentionally outside this bounded matrix.
3. **Influential factors:** train-balanced CE matters strongly for DD recall and BA; unweighted and weaker weighting favor PD/Accuracy. Label smoothing and focal loss shift DD/PD sensitivity without robust joint benefit. LR 1e-4 was unfavorable; WD 1e-3 had small inconclusive differences.
4. **Imbalance:** it is a consequential training consideration, but the available controlled comparisons do not demonstrate that it is the dominant residual limitation. Balanced resampling overshot toward DD and hurt overall BA.
5. **DD-focused augmentation:** none of jitter, scaling, mild temporal displacement, or same-class conservative mixup under DD-only policy passed the joint gate. No augmentation policy can be named a stable winner.
6. **Accuracy/BA/DD trade-off:** clear for unweighted CE (Accuracy up while BA and DD Recall fell) and balanced sampling (DD Recall up while PD Recall and BA fell).
7. **Phase-2 candidate and lift:** original STR-01 recipe, unchanged; retained paired lift 0.0000 on all six metrics. No Phase-2 candidate earns a new formal model label or outer evaluation.
8. **Next research decision:** do not keep searching this same fixed development set for training tweaks or reopen stopped deep architecture/mechanism directions on these data. The observed ceiling and unresolved hard-subject problem require independent evidence before a new design claim. This is a development-only stopping decision, not a new claim about FOE-01 outer performance.
