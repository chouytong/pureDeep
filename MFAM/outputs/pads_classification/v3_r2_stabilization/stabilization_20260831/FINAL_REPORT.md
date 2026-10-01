# V3 R2 stabilization development report

> All results are frozen-split inner-CV development estimates. No outer-test loader, signal, prediction, threshold, or evaluation was used.

## Preregistration and scope

- Plan SHA-256: `e662a059ac3a6ea19088c59d7385ba27b61ef2cba3275eca9b63340d6c224800`.
- S0 was read from the immutable targeted-ablation run; its bounded first-epoch AMP replay was diagnostic-only.
- S1/S2/S3 all used identical FP32 training settings and the same 15 folds.
- The 1560 validation rows repeat 390 subjects; uncertainty uses subject-cluster bootstrap.

## Development leaderboard

| Model | Precision | Fusion | BA | AUROC | Macro-F1 | PD Recall | DD Recall | Params | Nonfinite grad | Skipped steps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | reference | deep_only | 0.5897 | 0.6131 | 0.5852 | 0.9221 | 0.2573 | reference | not_applicable | not_applicable |
| S0 | AMP | raw_concat | 0.6596 | 0.7152 | 0.6696 | 0.8995 | 0.4198 | 123000 | 90 | 90 |
| S1 | FP32 | raw_concat | 0.6673 | 0.7188 | 0.6769 | 0.8887 | 0.4459 | 123000 | 0 | 0 |
| S2 | FP32 | dual_LayerNorm_concat | 0.5945 | 0.6698 | 0.5956 | 0.9103 | 0.2786 | 124092 | 0 | 0 |
| S3 | FP32 | 64d_vector_gated_sum | 0.6667 | 0.7099 | 0.6802 | 0.9011 | 0.4322 | 166458 | 0 | 0 |
| H1 | reference | handcrafted_logistic_reference | 0.6918 | 0.7515 | 0.6858 | 0.7936 | 0.5900 | reference | not_applicable | not_applicable |

## Candidate decision

Selected development candidate: **S1**. S2 retained=False; S3 retained=False. This is not a frozen outer model.

## Numerical and training stability

| Model | Best epoch mean [min,max] | Train-val BA gap | Runtime total (s) | Max pre-clip grad norm |
|---|---:|---:|---:|---:|
| S1 | 20.87 [3,43] | 0.1282 | 395.2 | 534.374 |
| S2 | 19.00 [6,40] | 0.1481 | 398.4 | 33.233 |
| S3 | 16.67 [2,35] | 0.1855 | 385.1 | 197.251 |

S0 historical AMP recorded 90 nonfinite-gradient/skipped batches. The bounded first-epoch replay reproduced 72 events; every event implicated `fusion_classifier`, and `r2_classifier.1.weight` was the first-listed nonfinite parameter.
S0 validation mean L2 norms were deep=13.574, statistical=44.272, classifier-input=46.934, logits=1.506. The statistical branch has the larger mean norm and a strong long tail; this supports scale mismatch but does not by itself prove causality.
S2 LayerNorm changed fusion-side mean L2 norms to deep=22.644 and statistical=5.696; aggregate L2 remains dimension-dependent (514 vs 32 dimensions).

## Statistical preprocessing and leakage controls

The frozen H1 source had frozen H1 label-independent 4928-dimensional cache; every fold used inner-train-only StandardScaler followed by 32-component, non-whitened full-SVD PCA. Cumulative explained variance across the 15 S1 folds: mean=0.7139, min=0.7016, max=0.7251. No PCA dimension search was performed.
All 45 formal candidate folds retained disjoint train/validation/test subject lists, validation predictions only, train-only preprocessing subject hashes, and false outer-test loader/signal/prediction/feature-transform flags.

## Legal cross-fitted threshold (secondary)

| Model | Cross-fit threshold | BA | AUROC | DD Recall |
|---|---:|---:|---:|---:|
| M0 | 0.2964 | 0.5722 | 0.6131 | 0.4251 |
| S0 | 0.3704 | 0.6426 | 0.7152 | 0.5503 |
| S1 | 0.3876 | 0.6542 | 0.7188 | 0.5360 |
| S2 | 0.3010 | 0.6123 | 0.6698 | 0.5147 |
| S3 | 0.3313 | 0.6437 | 0.7099 | 0.5248 |
| H1 | 0.4685 | 0.6837 | 0.7515 | 0.5920 |

Each held inner fold used thresholds fitted only from the other two inner-validation folds in the same outer context.

## Subject-cluster uncertainty

| Candidate vs reference | Observed pooled ΔBA | Subject-cluster 95% CI |
|---|---:|---:|
| S1 vs S0 | +0.0077 | [-0.0066, +0.0226] |
| S1 vs M0 | +0.0775 | [+0.0464, +0.1079] |
| S1 vs H1 | -0.0248 | [-0.0601, +0.0129] |
| S2 vs S0 | -0.0647 | [-0.0919, -0.0379] |
| S3 vs S0 | +0.0075 | [-0.0213, +0.0371] |

Intervals use 390 subject clusters, 2000 iterations, seed 20260831; the 1560 repeated validation rows were not treated as independent.

## Required questions

1. **Where did S0 AMP overflow appear first?** Dominant affected group: `fusion_classifier`; most frequently first-listed parameter: `r2_classifier.1.weight`. See `s0_amp_overflow_events.csv` for every bounded replay event.
2. **Did disabling AMP remove nonfinite gradients?** S1 nonfinite=0, skipped=0 across 15 folds.
3. **Did S1 reproduce S0?** Delta BA=+0.0077, delta AUROC=+0.0035, delta DD recall=+0.0262; BA nonnegative in 10/15 folds.
4. **Branch scale mismatch?** Mean validation raw deep/statistical L2 ratio: S1=0.317, S2=0.282, S3=0.513. Norms are evidence of scale, not proof of overflow causation.
5. **NormFusion improvement?** S2-S1 delta BA=-0.0729, delta AUROC=-0.0490; retention rule=False.
6. **GatedFusion improvement?** S3-S1 delta BA=-0.0006, delta AUROC=-0.0088; retention rule=False.
7. **Gate collapse?** Overall gate mean=0.5546, p5=0.4965, p95=0.6092; gate weights the deep branch.
8. **Independent deep contribution?** For S3, deep-disabled mean BA drop=+0.0239, AUROC drop=+0.0116; preregistered contribution criterion=True.
9. **Largest DD-recall improvement:** S1, with DD recall=0.4459.
10. **Subtype recovery:** Versus S0 predicted-DD fraction, Other best=S1 (0.438, delta=+0.046); Atypical best=S1 (0.300, delta=-0.017, therefore no candidate improved S0); MS best=S3 (0.409, delta=+0.227). These are exploratory subject-cluster estimates with only 60/15/11 unique subjects respectively.
11. **Recommended next deep candidate:** S1, following stability-first preregistered selection.
12. **Gap to H1:** BA gap=-0.0244; AUROC gap=-0.0328.
13. **Future R1+R2 advice:** There is development evidence that a deep branch remains contributory, so R1+R2 may be considered in a separately preregistered future stage; it was not run here.

## Provenance, failures, and frozen scope

- Output root: `/home/zyt/MFAM/outputs/pads_classification/v3_r2_stabilization/stabilization_20260831`.
- Frozen split SHA-256: `b3c52317cb12b73c66046bbd7c50c94a7e707a64a38d3ad4f0fd4bc0c6ba6b3e`.
- Base config SHA-256: `3bb1d8dbeea729775d300123b874752c85cce84f7760f4294e560249de472080`.
- Frozen H1 cache/schema SHA-256: `d7b465b251892abbe602539369989c133fce4c4da15e3bee3cb308d07e0f4da5` / `bdb83aae1c31b66c659a63f7a723f74b2cc82bfc99725621f6a41ba736389312`.
- Frozen V3 manifest SHA-256: `af5fdda01ceed8f90944f9a25e3a630b09ad17e07ece5f1d32dfe3d147271321`.
- Frozen S0 artifact-tree SHA-256: `528be416bfac771a9fe4477330030bfa917be411d0bc0f31fd0ee15f0224b535`.
- Candidate failures: 0/45 folds. FP32 nonfinite loss batches: 0; nonfinite gradient batches: 0; skipped optimizer steps: 0.
- Frozen object: S1 development candidate specification only (raw 514+32 concat, dropout and Linear(546,2), FP32, 123000 parameters). This is not an outer model and contains no outer-test performance claim.

## Stop condition

S0 diagnostics and S1/S2/S3 are complete. No new fusion, R1+R2, LR/loss change, outer-test run, or final clinical model freeze was performed.
