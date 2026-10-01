# Phase-3C: Minimal SSL Adaptation and Decision Optimization

Status: complete and audited, 2026-10-01 (Asia/Shanghai). PURE-DEEP; DEVELOPMENT-ONLY; NO OUTER-BASED MODEL SELECTION.

## Scope and frozen protocol

This study uses the fixed 15 subject-level inner-development splits (5 contexts × 3 inner folds), seeds 42/43/44, PD=0 and DD=1. Three seed results are aggregated inside each split before 15-split paired comparisons. All new experiments and diagnostics are development-only. The historical locked FOE-01 outer results were not used for model, learning-rate, threshold or representation selection. No H1, handcrafted features, teacher/student, new deep head, augmentation, backbone search or full HarNet fine-tuning was used. The original STR-01 and Phase-3B WSSL-STR development outputs are read-only matched references. See `PHASE3C_PROTOCOL.md` and `manifest.json` for choices fixed before validation inspection.

The official pretrained encoder is OxWearables HarNet10 at commit `150550ea5d41800229c95e36f88f5bf0d2e7cf04`, checkpoint SHA-256 `c64f9135d99e2dcdfc9ae7cc0672f2bcc438df9ceb8215665882f92cddd162a6`. Raw acceleration preprocessing and fixed wrist/activity ordering reproduce Phase-3B. Recomputed final 1024D features from the stage4 cache differed from the frozen Phase-3B cache by exactly 0. `analysis/stage4_audit.json` documents the check.

## Execution and integrity

E1 last-block adaptation uses one prespecified differential learning-rate configuration: HarNet layer1–4 frozen, layer5 trainable at `2e-5`, STR/projection at `2e-4`, with BN running statistics frozen. Trainable parameter count is 2,766,660, of which layer5 is 2,623,488. E1 smoke and checkpoint logits consistency passed: the same Phase-3B checkpoint and same input yielded maximum absolute logit difference 0; only layer5 received encoder gradients. All 45/45 E1 runs passed subject ID/label, normalization, checkpoint SHA, 1/10 LR and BN audits; no outer test loader was created. E1 training-source snapshots were retained and checked against `e1_source_sha256.txt`.

E2 uses three subject-level crossfit folds inside each target inner-training split for each of three seeds: 135 crossfit fits total, each fixed to epoch 10 without OOF-label checkpoint selection. Every inner-training subject received exactly one OOF score per seed. OOF/target-validation subject overlap was absent; no outer loader was created. Audit: `analysis/e2_oof_integrity.csv`, `e2_oof_audit.log`. The threshold is selected on grid 0.05–0.95 by OOF BA only, with fixed tie rule, and applied to archived Phase-3B validation predictions. E2 is an operating-point diagnostic and trains no final candidate.

Two post-training E2 analysis-only bugs were corrected before interpreting results: the audit expected 405 rather than 135 crossfit models; the ensemble aggregation was one indent too shallow and initially output five rather than 15 split rows. Pre-fix sources and exact correction hashes are preserved in `ANALYSIS_FIX_NOTE.md`. No training or prediction artifact changed.

E3 uses one fixed middle layer4 512D mean feature concatenated with final layer5 1024D, projected to a 64D wrist residual; HarNet stays frozen and all downstream training settings are unchanged. Smoke and checkpoint reload passed. All 45/45 E3 runs completed; validation IDs/labels, train-only normalization and checkpoint hashes matched the frozen reference, and no outer test loader was created (`analysis/e3_integrity.csv`, `e3_audit.log`).

## E1: last-block adaptation

Seed-first means across 15 splits:

| Configuration | Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall |
|---|---:|---:|---:|---:|---:|---:|
| Original STR-01 | 0.7307 | 0.6972 | 0.7176 | 0.6861 | 0.7779 | 0.6165 |
| Frozen WSSL-STR | 0.7456 | 0.7196 | 0.7596 | 0.7066 | 0.7823 | 0.6569 |
| Last-block adapted WSSL-STR | 0.7518 | 0.7244 | 0.7689 | 0.7128 | 0.7905 | 0.6583 |
| Middle+final WSSL-STR | 0.7537 | 0.7204 | 0.7667 | 0.7112 | 0.8006 | 0.6402 |

Adapted minus frozen: BA +0.0048 (10/15 split improvements; 95% split-bootstrap CI [−0.0006,+0.0102]; paired dz 0.431; four-primary-test BH q 0.2251). AUROC +0.0094 (11/15; [+0.0030,+0.0154]; dz 0.728; q 0.0603). Macro-F1 +0.0062 with CI crossing zero; DD Recall +0.0014 with CI crossing zero. Mean within-split seed SD was BA 0.0202 versus 0.0216 frozen and AUROC 0.0222 versus 0.0217. Mean seed-pair score Spearman rose 0.7734→0.7940, while 0.5-threshold disagreement was 0.1780→0.1786. The preset BA positive-CI gate and strict disagreement gate fail, so E1 is **not retained**. The modest AUROC signal is exploratory within this development comparison and does not justify a pretrained-LR search.

## E2: train-only OOF threshold

| Estimand | Rule | Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| Single WSSL | Fixed 0.5 | 0.7456 | 0.7196 | 0.7596 | 0.7066 | 0.7823 | 0.6569 |
| Single WSSL | Train-OOF BA threshold | 0.6683 | 0.6771 | 0.7596 | 0.6325 | 0.6559 | 0.6983 |
| Three-seed ensemble | Fixed 0.5 | 0.7573 | 0.7200 | 0.7816 | 0.7141 | 0.8100 | 0.6301 |
| Three-seed ensemble | Train-OOF BA threshold | 0.6764 | 0.6845 | 0.7816 | 0.6432 | 0.6650 | 0.7040 |

Single threshold minus fixed BA −0.0426, 0/15 positive splits, 95% split-bootstrap CI [−0.0682,−0.0227], dz −0.888, BH q 0.0001; Macro-F1 −0.0741. DD Recall +0.0413 had a CI crossing zero while PD Recall fell −0.1265. Ensemble BA −0.0355, 4/15 positive, CI [−0.0764,+0.0008], dz −0.451, BH q 0.0962; Macro-F1 −0.0710. Ensemble DD Recall +0.0740, CI [+0.0217,+0.1287], came with PD Recall −0.1450. AUROC is unchanged by threshold. The 15 ensemble OOF thresholds had mean 0.3927 and range 0.15–0.57; 45 single thresholds had mean 0.4071 and range 0.08–0.70. Thus a train-only OOF operating point does **not** convert the WSSL ranking gain into a better BA/Macro-F1 trade-off; keep the fixed 0.5 rule. The crossfit models are trained on two-thirds of each target inner train set and fixed at epoch 10, so OOF/full-train score-distribution shift is a plausible limitation, not a validated explanation.

## E3: one middle+final feature

Middle+final minus final-only frozen: BA +0.0008 (6/15; CI [−0.0038,+0.0054]; dz 0.081; four-primary-test BH q 0.8040); AUROC +0.0071 (10/15; CI [−0.0024,+0.0156]; dz 0.383; q 0.2251). Macro-F1 +0.0046 has a CI crossing zero; DD Recall −0.0167 has a CI crossing zero. Mean within-split seed SD was BA 0.0222 versus 0.0216 and AUROC 0.0259 versus 0.0217. The BA majority/CI and DD Recall gates fail. **Do not retain** the middle+final representation; stop layer search.

## E4: fixed subject-count learning curve

All 270 new 25/50/75% runs completed; the 90 archived 100% STR/Phase-3B WSSL runs were reused without retraining. The 360 total model/seed/split/fraction units passed the analysis assertions: matching validation IDs/labels, identical training subjects and train-only normalization across STR/WSSL and seeds at each fraction, nested 25⊂50⊂75⊂100% subject sets, and no train/validation overlap. All 270 new checkpoint hashes matched their stage-status records, and no outer test loader was created (`analysis/e4_checkpoint_integrity.csv`, `e4_checkpoint_audit.log`, `e4_curve_analysis.log`). Mean training subject count was approximately 52/104/156/208.

Seed-first 15-split development means (fixed 0.5 decision threshold):

| Train % | Model | Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall |
|---:|---|---:|---:|---:|---:|---:|---:|
| 25 | STR | 0.6991 | 0.6271 | 0.6222 | 0.6254 | 0.8005 | 0.4537 |
| 25 | frozen WSSL | 0.6932 | 0.6583 | 0.6750 | 0.6443 | 0.7424 | 0.5741 |
| 50 | STR | 0.7117 | 0.6555 | 0.6684 | 0.6495 | 0.7913 | 0.5197 |
| 50 | frozen WSSL | 0.7070 | 0.6804 | 0.7130 | 0.6630 | 0.7441 | 0.6168 |
| 75 | STR | 0.7114 | 0.6778 | 0.6947 | 0.6654 | 0.7588 | 0.5968 |
| 75 | frozen WSSL | 0.7205 | 0.6956 | 0.7253 | 0.6804 | 0.7556 | 0.6356 |
| 100 | STR | 0.7307 | 0.6972 | 0.7176 | 0.6861 | 0.7779 | 0.6165 |
| 100 | frozen WSSL | 0.7456 | 0.7196 | 0.7596 | 0.7066 | 0.7823 | 0.6569 |

WSSL−STR paired BA differences at 25/50/75/100% were +0.0311 (11/15; 95% CI [+0.0136,+0.0495]; eight-primary-test BH q=0.0086), +0.0249 (11/15; [+0.0064,+0.0455]; q=0.0345), +0.0178 (10/15; [+0.0031,+0.0335]; q=0.0479), and +0.0225 (11/15; [+0.0097,+0.0362]; q=0.0090). Corresponding AUROC differences were +0.0528 (12/15; [+0.0290,+0.0780]; q=0.0034), +0.0446 (13/15; [+0.0203,+0.0710]; q=0.0086), +0.0306 (12/15; [+0.0127,+0.0501]; q=0.0086), and +0.0419 (15/15; [+0.0300,+0.0543]; q=0.0005). Thus the frozen SSL benefit is observed at every tested training size, including small cohorts.

The direct *interaction* question is whether the low-fraction advantage exceeds the 100% advantage on the same split. At 25% versus 100%, the paired difference-in-differences was BA +0.0087 (8/15; CI [−0.0130,+0.0291]) and AUROC +0.0109 (8/15; [−0.0098,+0.0319]); both cross zero. The 50% and 75% BA/AUROC interactions also cross zero (`analysis/e4_paired.csv`). Therefore a systematically larger BA/AUROC advantage at low training counts is **not established**. WSSL−STR DD Recall was +0.1204 at 25%, +0.0971 at 50%, +0.0388 at 75%, +0.0404 at 100%. The 25% minus 100% DD Recall interaction was +0.0800 (11/15; bootstrap CI [+0.0245,+0.1372], paired dz 0.687), but three DD-interaction BH q=0.0925 (`analysis/e4_dd_interaction_bh.csv`); it is exploratory. At 25/50%, some DD improvement accompanies lower PD Recall and Accuracy. Seed variance is not uniformly lower for WSSL at every fraction: at 25% within-split BA/AUROC seed SD is 0.0238/0.0291 vs STR 0.0293/0.0421, while at 50% WSSL is 0.0282/0.0400 vs STR 0.0235/0.0344. The learning curve supports transferable SSL signal across sample sizes, but not a stable small-cohort-specific attenuation of representation instability.

## Final decision and limits

**CASE D.** No further Phase-3C adaptation passes the frozen candidate gate. Retain the pretrained-**frozen** WSSL-STR, final 1024D HarNet feature only, original balanced CE/training recipe and fixed 0.5 decision threshold as the best **single-model pure-deep development configuration**: Accuracy 0.7456, BA 0.7196, AUROC 0.7596, Macro-F1 0.7066, PD Recall 0.7823, DD Recall 0.6569. Relative to original STR-01, the archived Phase-3B paired gains remain BA +0.0225 and AUROC +0.0419; E4 exactly reproduced the 100% means. Last-block adaptation, middle+final feature fusion and OOF thresholding are rejected as defaults. The three-seed frozen WSSL probability ensemble has AUROC 0.7816 but remains an inference-level reference; its OOF-adjusted operating point is not retained. Stop further HarNet encoder-LR, layer and threshold search on these same development splits. Further SSL adaptation is not justified; proceed to paper experiment consolidation and genuinely independent validation planning, without reusing the locked FOE-01 outer set for Phase-3B/C selection or claiming it as their independent test.

Positive evidence is the robust frozen WSSL advantage over STR across tested subject-count fractions. Negative evidence is the lack of a stable added BA benefit from E1/E3, E2 operating-point harm, and absence of a supported BA/AUROC small-cohort interaction. The 15 fixed inner splits overlap in subjects and contexts, so their split-bootstrap intervals and BH values describe this repeated development design; they are not an independent external-cohort inference. Seed-runs, subjects, OOF folds, windows and fractions were not treated as independent replicates. E2 OOF fits used smaller training subsets than target models, limiting calibration transfer. E1/E3 have no independent outer evaluation, and FOE-01 outer was not used for Phase-3C development or re-evaluated for these new candidates.

## Reproduction ledger

- Frozen plan and input hashes: `PHASE3C_PROTOCOL.md`, `manifest.json`, `e1_source_sha256.txt`, `remaining_source_sha256.txt`.
- E1 feature/logit and full-run audits: `analysis/stage4_audit.json`, `analysis/e1_consistency.json`, `analysis/e1_integrity.csv`, `e1_audit.log`.
- E2 OOF audit, fixed analysis correction and tables: `analysis/e2_oof_integrity.csv`, `e2_oof_audit.log`, `ANALYSIS_FIX_NOTE.md`, `corrected_e2_analysis_sha256.txt`, `analysis/e2_threshold_paired.csv`.
- E3 complete model comparison and audit: `analysis/model_summary.csv`, `analysis/model_paired.csv`, `analysis/model_gates.json`, `analysis/e3_integrity.csv`, `e3_audit.log`.
- E4 complete learning curve and checkpoint audit: `analysis/e4_seed_split_metrics.csv`, `analysis/e4_summary.csv`, `analysis/e4_paired.csv`, `analysis/e4_dd_interaction_bh.csv`, `analysis/e4_checkpoint_integrity.csv`, `e4_curve_analysis.log`, `e4_checkpoint_audit.log`.
