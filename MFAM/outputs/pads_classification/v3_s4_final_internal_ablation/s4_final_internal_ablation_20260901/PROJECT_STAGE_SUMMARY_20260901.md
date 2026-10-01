# MFAM project stage summary — 2026-09-01

## Research task and evidence boundary

The project performs subject-level PD-versus-DD classification on the 390-subject PADS subset using 11 activities, bilateral wrists and accelerometer+gyroscope signals; labels are 0=PD and 1=DD. It is not healthy-control classification, UPDRS/severity regression, or clinical deployment validation.

V2 used one frozen train/validation/test split, three seeds and a probability ensemble. V3 replaced model-development evidence with a frozen 5×3 nested-CV design across five diagnosis strata, train-only normalization/PCA, subject-disjoint folds, explicit provenance and anti-leakage flags. Historical V2 test evidence and V3 inner-CV development estimates remain distinct.

## Unified evidence table

| Stage | Model | Evidence type | BA | AUROC | DD Recall | Key conclusion |
|---|---|---|---:|---:|---:|---|
| V2 | SubjectMFAM 3-seed ensemble | historical frozen test | 0.6013 | 0.6203 | 0.4348 | Historical split/threshold result; not comparable as inner-CV evidence |
| V3 | M0 SubjectMFAM | development inner-CV | 0.5897 | 0.6131 | 0.2573 | Deep representation misses substantial DD information |
| V3 | H1 handcrafted logistic | development inner-CV | 0.6918 | 0.7515 | 0.5900 | Strongest classical ranking reference by AUROC |
| V3 | H1b balanced logistic | development inner-CV | 0.6947 | 0.7508 | 0.6075 | Class balancing adds only small BA over H1 |
| V3 | MiniRocket | development inner-CV | 0.6556 | 0.7393 | 0.4254 | Alternative temporal representation beats M0 |
| V3 | Simple CNN N1a | development inner-CV | 0.5000 | 0.4219 | 0.0000 | Collapsed to all-PD |
| V3 | R1 full-band | development inner-CV | 0.6124 | 0.6316 | 0.3028 | Small, consistent full-band gain |
| V3 | R2/S0 AMP | development inner-CV | 0.6596 | 0.7152 | 0.4198 | Statistical residual helps but AMP is numerically invalid |
| V3 | S1 R2-FP32 | development inner-CV | 0.6673 | 0.7188 | 0.4459 | Stable previous deep candidate |
| V3 | S2 NormFusion | development inner-CV | 0.5945 | 0.6698 | 0.2786 | Dual LayerNorm fusion rejected |
| V3 | S3 GatedFusion | development inner-CV | 0.6667 | 0.7099 | 0.4322 | No clear improvement over raw concat |
| V3 | S4 S1+R1 | development inner-CV | 0.6502 | 0.7239 | 0.3784 | S4_rejected_S1_remains_selected |

## Hypotheses weakened or rejected

- The old split is not established as the main performance problem: V3's stricter protocol changed estimation, while matched baselines still show a large representation gap.
- Threshold is not the main bottleneck: the frozen diagnostic threshold improved BA by only +0.0113, far below the H1 and MiniRocket representation gains.
- Class weighting is not the main solution: H1b improves BA only about +0.0029 over H1.
- Timestamp gaps and robust-outlier warnings are not supported as primary error drivers: only four gap records existed and the audited associations were small/inconclusive or opposite the proposed direction.
- GroupNorm, masked-mean activity fusion and removal of Hard Top-K did not improve matched development BA (N1, A2 and A1 respectively).
- Dual LayerNorm fusion was harmful (S2); gated fusion did not clearly outperform raw concat (S3).

## Evidence that remains supported

- Handcrafted statistics remain strong (H1/H1b) and the statistical residual substantially improves SubjectMFAM ranking.
- M0 misses DD-relevant information; both MiniRocket and H1 materially outperform it.
- R1 provides a small but fold-consistent full-band improvement; S4 determines whether that signal adds enough beyond S1 under the locked rule.
- FP32 eliminated the R2 AMP nonfinite-gradient/skipped-step instability.
- Acc+Gyro (BA 0.692) exceeds Acc-only (0.670) and Gyro-only (0.668); bilateral wrists (0.692) exceed either wrist alone; full multi-activity H1 exceeds every single activity.
- DD subtype heterogeneity is pronounced, with ET generally easier and Atypical/MS estimates much less certain.

## Exploratory DD subtype recovery

| Model | ET | Other | Atypical | MS |
|---|---:|---:|---:|---:|
| M0 | 0.446 | 0.246 | 0.067 | 0.091 |
| H1 | 0.759 | 0.546 | 0.550 | 0.455 |
| S1 | 0.634 | 0.438 | 0.300 | 0.205 |
| S4 | 0.616 | 0.354 | 0.117 | 0.250 |

Values are repeated-inner-validation predicted-DD fractions with subject-cluster uncertainty in the accompanying CSV. Atypical (N=15) and MS (N=11) are especially small; all subtype conclusions are exploratory and did not select the model.

## Current code and provenance state

Maintain the primary nested-CV/provenance path, the audited handcrafted-feature baseline path, targeted ablations, R2 stabilization, and this S4 runner. Experimental entry points have accumulated and should be archived or documented in a separate future maintenance task; none were removed here because they are part of frozen experiment provenance. There is no duplicate feature extractor in S4, but its runner is intentionally experiment-specific to prevent accidental scope expansion.
This round added six source files and modified zero existing source files. No broad cleanup or refactor was performed.

## Frozen model choices

- Current historical deep baseline: V2 SubjectMFAM three-seed ensemble (historical frozen split/test evidence).
- Current strongest classical reference: H1 handcrafted logistic by AUROC; H1b has the slightly higher BA but is a balanced-sensitivity variant.
- Current selected deep development candidate: **S1**. S1 remains selected; S4 is rejected.

The selected object is a development specification, not a final, outer-tested, clinically validated model.

## Research-design limitation

The same 390 subjects have informed nested-CV development, baseline analysis, error analysis and multiple preregistered ablations. Any further result on these people cannot honestly be called a completely untouched independent final validation. Strong generalization evidence now requires an external dataset or genuinely new holdout cohort; re-randomizing these 390 subjects cannot manufacture a blind test.

## Next directions (do not execute automatically)

1. Validate the frozen selected specification on an external dataset or genuinely new cohort.
2. Prepare interpretation, uncertainty and reporting for the selected deep candidate without new architecture search.
3. In a separately authorized maintenance task, archive/document experimental entry points while preserving frozen provenance.

## Stop

Internal architecture development on the current 390 subjects stops here. No additional model, hyperparameter, fusion or outer-test experiment was started.
