# PURE-DEEP PHASE-3 DEVELOPMENT: E1–E4

## Scope and fixed boundary

All analyses use fixed 15 inner-development subject splits and seeds 42/43/44. The original STR-01 architecture, activity/wrist order, train-only normalization, balanced-CE baseline, and final score rule are the reference. Phase-2 exact-reproduction predictions and checkpoints serve as lambda=0 and natural-sampling baseline. Prior FOE-01 outer data and results are excluded; H1 and handcrafted/statistical methods are excluded from training, input, teacher, loss, representation, model and threshold selection. New checkpoints and diagnostics live only in this directory.

The fixed protocol and retention gate were written to `PHASE3A_PROTOCOL.md` before E1/E3/E4 validation results. `manifest.json` records the four new configurations, `SOURCE_SHA256SUMS` freezes their runner/hook sources, and `SMOKE_AUDIT.json` records train/validation ID and normalization agreement, no outer loader, checkpoint reload, and E4 logits consistency. The auxiliary branch has 8,354 train-time parameters; the original STR backbone has 75,524. With the same frozen checkpoint and input, E4 eval-mode final logits are bitwise identical to STR (max difference 0), and the branch is disabled at inference.

## E1 sampling/loss audit before the new control

Code review confirmed that the previous `balanced_sampler` candidate used both weighted subject resampling and the original train-balanced CE. It did not override `loss.class_weights`. The three-way comparison is therefore (A) natural sampling + balanced CE (exact STR baseline), (B) balanced sampling + balanced CE (Phase-2 archive), and (C) balanced sampling + unweighted CE (new Phase-3A run). C changes only this compensation coupling. The smoke checkpoint resolved C class weights to `None`; the fixed inner split and normalization hash matched A/B. No sampling-ratio search is part of this phase.

## E2 three-seed probability ensemble

For each of the 15 splits, 42/43/44 validation prediction files were aligned by subject ID and label, then DD probabilities were averaged before thresholding at the existing 0.5 rule. All 4,680 seed-subject records matched the recorded baseline decision rule and had no exact 0.5 ties. The ensemble analysis did not train or select a model.

| Metric | Mean of three seed metrics | Metric from averaged subject probabilities | Paired Δ | Positive splits | Bootstrap 95% CI | Paired effect dz |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy | 0.7307 | 0.7489 | +0.0182 | 11/15 | [+0.0046, +0.0319] | +0.6491 |
| BA | 0.6972 | 0.6998 | +0.0026 | 8/15 | [−0.0116, +0.0163] | +0.0916 |
| AUROC | 0.7176 | 0.7586 | +0.0410 | 15/15 | [+0.0324, +0.0493] | +2.3779 |
| Macro-F1 | 0.6861 | 0.6976 | +0.0116 | 10/15 | [−0.0029, +0.0255] | +0.4004 |
| PD Recall | 0.7779 | 0.8180 | +0.0402 | 13/15 | [+0.0237, +0.0569] | +1.1778 |
| DD Recall | 0.6165 | 0.5816 | −0.0349 | 3/15 | [−0.0564, −0.0125] | −0.7852 |

The ensemble gives a strong and consistent ranking/AUROC improvement, but its unmodified 0.5 decision rule reduces DD Recall. It is a possible ranking-oriented inference strategy, not an automatic replacement for the single-model classification baseline and not a new model architecture. No threshold tuning or outer evaluation was performed. Full per-split and subject-aligned records, Wilcoxon p and BH q values are in `analysis/e2_*`.

## E1 completed: sampling–loss compensation

All three conditions use the same 15 fixed inner splits and seeds 42/43/44. The two resampled conditions exposed approximately 104 PD and 104 DD existing training subjects per epoch, with no augmented examples or additional independent subjects. C's exposure distribution matched B's; the change was the removal of class weights from CE.

| E1 condition | Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall |
|---|---:|---:|---:|---:|---:|---:|
| A natural sampling + balanced CE | 0.7307 | 0.6972 | 0.7176 | 0.6861 | 0.7779 | 0.6165 |
| B balanced sampling + balanced CE | 0.6908 | 0.6790 | 0.7059 | 0.6547 | 0.7080 | 0.6499 |
| C balanced sampling + unweighted CE | 0.7146 | 0.6905 | 0.7167 | 0.6732 | 0.7488 | 0.6321 |

C versus B improved BA **+0.0115** (10/15 splits; 95% CI +0.0022 to +0.0208), AUROC +0.0108 (9/15; CI −0.0030 to +0.0255), and Macro-F1 +0.0185 (11/15; CI +0.0077 to +0.0317). The earlier B deficit was therefore partly caused by simultaneous balanced sampling and balanced CE. Yet C versus A still had BA **−0.0067** (3/15; CI −0.0122 to −0.0006), AUROC −0.0009 (10/15; CI −0.0118 to +0.0087), Macro-F1 −0.0128 (3/15; CI −0.0224 to −0.0021), PD Recall −0.0290, and DD Recall +0.0156 with a CI spanning zero. C did not meet the joint retention gate. **Decision: stop sampling/class-exposure search on this fixed development set.**

## E3 completed: fixed SAM rho=0.05 + AdamW

All 45 inner runs completed with the original batch, scheduler, balanced CE, architecture and early-stopping rule. The STR model uses GroupNorm and has no BatchNorm state affected by a SAM second pass. For the train/validation CE gap, training CE is taken from SAM's first (unperturbed-weight) forward pass; second-pass perturbed loss is logged separately. The median best epoch moved from 12 to 17. Mean CE gap at the selected epoch decreased from 0.3759 to 0.2542 and at stopping from 1.0450 to 0.6761; this is a training-behavior change, not sufficient classification evidence.

| Condition | Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall | BA seed SD | AUROC seed SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original STR-01 | 0.7307 | 0.6972 | 0.7176 | 0.6861 | 0.7779 | 0.6165 | 0.0117 | 0.0120 |
| STR-01 + SAM | 0.7452 | 0.6935 | 0.7213 | 0.6915 | 0.8182 | 0.5687 | 0.0209 | 0.0318 |

Paired SAM minus STR differences over 15 seed-averaged splits: BA **−0.0037** (5/15 positive; bootstrap 95% CI −0.0133 to +0.0068; dz −0.184), AUROC **+0.0036** (10/15; CI −0.0132 to +0.0166; dz +0.120), Macro-F1 +0.0054 (10/15; CI −0.0036 to +0.0147), Accuracy +0.0145, PD Recall +0.0404 (CI +0.0063 to +0.0737), and DD Recall **−0.0479** (5/15; CI −0.0929 to −0.0018). AUROC and BA seed SD rose to 0.0318 and 0.0209. The Accuracy gain mainly tracks PD favoring and fails the joint retention rule. **Decision: SAM is rejected; no rho search and no SAM+auxiliary combination.**

## E4 completed: training-only activity evidence supervision

The pre-aggregation 11×258D bilateral activity features feed a shared 258→32→2 branch. Only a masked mean of per-activity evidence receives subject-level balanced CE; no individual activity is assigned the full disease label. λ=0.1 and λ=0.3 were fixed in advance, and the original STR final logits alone supplied all validation predictions. Both variants completed 45/45 runs. The auxiliary CE and main CE are archived per epoch; the report compares main classification CE with validation CE, since total train loss includes λ×auxiliary loss and has a different scale.

| E4 condition | Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall | Best epoch median | Best CE gap | Stop CE gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.7307 | 0.6972 | 0.7176 | 0.6861 | 0.7779 | 0.6165 | 12 | 0.3759 | 1.0450 |
| e4_aux_lambda01 | 0.7239 | 0.6943 | 0.7182 | 0.6812 | 0.7655 | 0.6232 | 11 | 0.2822 | 0.9641 |
| e4_aux_lambda03 | 0.7277 | 0.6946 | 0.7154 | 0.6838 | 0.7746 | 0.6146 | 11 | 0.2872 | 0.9609 |

λ=0.1: paired BA -0.0029 (5/15 positive; 95% CI -0.0074 to +0.0022; dz -0.292); AUROC +0.0005 (7/15; CI -0.0049 to +0.0055; dz +0.051); Macro-F1 -0.0049; PD Recall -0.0123; DD Recall +0.0066.

λ=0.3: paired BA -0.0026 (6/15 positive; 95% CI -0.0071 to +0.0025; dz -0.266); AUROC -0.0023 (8/15; CI -0.0086 to +0.0038; dz -0.180); Macro-F1 -0.0022; PD Recall -0.0033; DD Recall -0.0019.

Neither variant meets the BA/AUROC/Macro-F1 retention rule.

The slightly smaller main-CE train/validation gap under E4 did not yield improved final classification. A full or minimal representation probe was **not** run: the preregistered condition for that check was final classification improvement, and neither λ met it. The result does not support activity auxiliary supervision as a retained model mechanism on this fixed development set. No λ extension or additional auxiliary objective was attempted.

## Unified seed-first statistics and retention

Three seeds were averaged inside each of 15 fixed splits. The following are split-level means; split SD is computed across those 15 units, while seed SD is descriptive across the three seed-level means. The 45 runs, repeated subjects and 11 activities were not treated as independent samples.

| Configuration | Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall | BA split SD | BA seed SD | AUROC seed SD | DD seed SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.7307 | 0.6972 | 0.7176 | 0.6861 | 0.7779 | 0.6165 | 0.0316 | 0.0117 | 0.0120 | 0.0408 |
| e1_balanced_balanced | 0.6908 | 0.6790 | 0.7059 | 0.6547 | 0.7080 | 0.6499 | 0.0249 | 0.0127 | 0.0201 | 0.0095 |
| e1_balanced_unweighted | 0.7146 | 0.6905 | 0.7167 | 0.6732 | 0.7488 | 0.6321 | 0.0279 | 0.0197 | 0.0188 | 0.0476 |
| e3_sam_rho005 | 0.7452 | 0.6935 | 0.7213 | 0.6915 | 0.8182 | 0.5687 | 0.0297 | 0.0209 | 0.0318 | 0.0425 |
| e4_aux_lambda01 | 0.7239 | 0.6943 | 0.7182 | 0.6812 | 0.7655 | 0.6232 | 0.0321 | 0.0162 | 0.0164 | 0.0309 |
| e4_aux_lambda03 | 0.7277 | 0.6946 | 0.7154 | 0.6838 | 0.7746 | 0.6146 | 0.0299 | 0.0134 | 0.0177 | 0.0228 |

Complete paired differences, improvement counts, 10,000-draw split-bootstrap 95% CIs, paired standardized effect sizes (Cohen dz), Wilcoxon p and BA/AUROC BH-FDR q values are in `analysis/phase3_paired.csv`; the fixed gate decisions are in `analysis/retention_gate.json`. No positive new-training BA/AUROC result survived BH-FDR. Across E1-C, E3, E4-0.1, and E4-0.3, **0/4** training variants passed the joint gate. E2 is a separate inference comparison: AUROC q=0.0001 and BA q=0.7197. Its DD Recall decline at the original threshold remains a material limit.

## Final decision: CASE D

The best single-model pure-deep configuration remains the **original formal STR-01** (Accuracy 0.7307, BA 0.6972, AUROC 0.7176, Macro-F1 0.6861, PD Recall 0.7779, DD Recall 0.6165). No Phase-3A training variant earns a replacement or a new outer evaluation. The observed Accuracy rise under SAM is accompanied by DD Recall and BA losses and cannot justify retention. The three-seed probability ensemble is a distinct, development-only inference finding: excellent ranking gain, but lower DD Recall at the fixed threshold; it is not selected as the default six-metric classification strategy and is not a single-model architecture claim.

E1, E3 and E4 all failed their predefined training-retention criteria. Following the user-specified decision tree, **CASE D** applies: stop further supervised STR architecture/loss/sampling/SAM/auxiliary searching on these same fixed PADS development subjects. The next research direction is to draft a separate, bounded **external wearable self-supervised pretraining** protocol using public large-scale IMU data, with pure-deep transfer and a new evidence boundary. No external pretraining is started in Phase-3A. The prior FOE-01 outer results remain frozen and cannot be reused to select this next approach.

## Integrity and limitations

Four new configurations × 45 inner runs = **180/180 complete**. `analysis/phase3_integrity_audit.json` records 180/180 matching train and validation subject IDs, validation labels, train-only normalization hashes, unique validation predictions and registered configuration identity, plus 180/180 explicit `outer_test_loader_created=false`; source SHA-256 values match, zero issues. Four smoke runs passed. `analysis/logits_consistency.json` shows original STR eval final logits remain bitwise unchanged under the E4 wrapper for an identical checkpoint and input (maximum difference 0). All experiment code and artifacts are isolated from the frozen formal archive.

The bootstrap resamples 15 fixed development splits that share subjects across outer contexts, so its CIs describe stability of this protocol, not independent external cohorts. Multiple configurations on the same development design still create selection optimism. We did not tune ensemble threshold, SAM rho, sampling ratio, auxiliary projection size, or λ beyond the two fixed values. Phase-3A results are development-only and neither revise nor reinterpret the completed locked FOE-01 outer evaluation. H1 and handcrafted statistical models played no role in this phase.
