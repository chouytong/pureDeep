# STR-01 Gain Mechanism Diagnosis

Date: 2026-09-23

## Executive conclusion

STR-01's improvement over V8-GN is a **distributed, complementary decision effect**, not a single-activity rescue and not a general improvement of global disease clustering or subject-invariant geometry.

The ordered residual path has strong independent ranking information (standalone AUROC 0.7209), while the jointly retrained original subject path is not stronger than V8-GN by itself (BA/AUROC 0.6299/0.6793 versus 0.6707/0.6800). Their combination produces BA/AUROC 0.6972/0.7176. Among 139 seed-aggregated `V8 wrong → STR correct` validation appearances, 59 (42.4%) require the residual path to flip the STR base-path decision; in 80 (57.6%), joint training has already moved the original path to the correct side and the residual usually strengthens it. Therefore the gain cannot be attributed to the residual head alone or to the unchanged V8 path alone.

The gain is distributed across activities. No activity's rescue-versus-harm residual contribution survives the prespecified 22-test BH correction. DrinkGlas, HoldWeight, TouchNose, RelaxedTask and Relaxed frequently provide the largest or pivotal residual contribution, but no activity dominates consistently. CrossArms, DrinkGlas and HoldWeight do not form a jointly significant high-risk subset. Standalone activity AUROC also does not improve broadly; PointFinger and Entrainment significantly decline. The positive result is therefore the ordered multi-activity decision pattern, not universally better per-activity evidence.

STR-01 partially reduces stable errors but does not solve phenotype overlap. Under the primary seed-aggregated four-split definition, stable-error subjects decrease from 85 to 74: 12 become stable-correct and 10 become unstable, but 63 remain stable-error and 11 new stable-errors appear. Under the historical 12 seed-fold appearance definition used by the frozen audit, stable errors decrease from 54 to 42: only 2 become stable-correct, 18 become unstable, 34 remain stable-error, and 8 unstable subjects become stable-error. Thus STR mainly softens some errors rather than consistently curing the original stable-error population.

The next stage should prioritize **cross-individual generalization representation learning**, not another structured readout, decision regularizer, or cross-activity architecture search. STR combined representations improve fixed disease-probe BA/AUROC, but disease silhouette falls, cross-activity subject retrieval worsens, domain AUC/mean shift do not improve, and centroid drift is larger. The remaining limitation is generalizable representation geometry rather than insufficient decision-stage capacity.

## Protocol and evidence boundary

- Models: frozen V8-GN matched reference and frozen STR-01.
- Data: the same 5×3 fixed subject-level inner-development splits.
- Seeds: 42/43/44, aggregated within matched split before decision-pattern analysis.
- Statistical units: 15 independent inner splits for paired model/activity analyses; unique subjects for persistent phenotype comparisons.
- No model training, model modification, hyperparameter search, or outer-final/test access.
- Activity margins are exact additive DD-minus-PD logit contributions. For the original path they equal activity attention multiplied by the classifier projection of the activity-context token; for STR they equal the relevant 16-D ordered-token block multiplied by the residual-head weight. Path biases are retained in final decisions but excluded from per-activity attribution.
- Maximum activity-margin/logit reconstruction error across all 45 archives: `1.91e-6`.
- Wilcoxon paired tests and rank-biserial effects use 15 split means. Subject phenotype comparisons use Mann–Whitney U/Cliff's delta. BH-FDR is applied separately within prespecified analysis families.

## 1. Which decisions changed?

Seed-aggregated validation appearances (`n=1560`):

| Decision group | PD | DD | Total |
|---|---:|---:|---:|
| Both correct | 829 | 200 | 1029 |
| V8 wrong → STR correct | 74 | 65 | 139 |
| V8 correct → STR wrong | 63 | 30 | 93 |
| Both wrong | 138 | 161 | 299 |

STR therefore produces 46 more corrected than harmed appearances. The net gain is larger for DD (`+35`) than PD (`+11`), consistent with the official DD-recall increase from 0.5631 to 0.6165 while PD recall remains essentially unchanged.

The rescue is usually near the V8 decision boundary. Mean true-direction V8 margin is -0.606 for rescued DD and -0.355 for rescued PD, compared with -1.075 and -0.707 for subjects both models misclassify. STR raises rescued final margins to +0.477 (DD) and +0.634 (PD), whereas both-wrong margins become more negative (-1.322/-1.094). STR thus separates a recoverable near-boundary subset from a persistent overlap subset; it does not uniformly improve all errors.

Only six subjects are persistently rescued in at least three of four seed-aggregated validation appearances (5 DD, 1 PD), while two are persistently harmed (1 DD, 1 PD). Most gain is distributed across split-dependent subjects rather than a large permanently repaired cohort.

## 2. Original path versus residual path

Official 15-split, three-seed means:

| Decision path | BA | AUROC | Macro-F1 | PD recall | DD recall |
|---|---:|---:|---:|---:|---:|
| V8-GN head | 0.6707 | 0.6800 | 0.6631 | 0.7782 | 0.5631 |
| STR original/base path only | 0.6299 | 0.6793 | 0.6181 | 0.7320 | 0.5279 |
| STR residual path only | 0.6561 | 0.7209 | 0.6298 | 0.6924 | 0.6199 |
| STR combined final | 0.6972 | 0.7176 | 0.6861 | 0.7779 | 0.6165 |

The original path is not an independently improved V8 classifier. Joint training reallocates evidence between the two paths: the residual path carries high-ranking disease information but has weaker thresholded BA and asymmetric recall; the base path supplies complementary decision offset/context. Combining them restores PD recall and retains most of the residual path's DD discrimination.

Within the 139 rescues:

- 59 (42.4%) are direct residual corrections: the STR base path remains wrong and the residual flips the final decision.
- 80 (57.6%) are already correct on the jointly retrained STR base path; the residual strengthens 64 and weakens 16 without changing correctness.

Within the 93 harms, the residual directly flips 42 correct base decisions to wrong; the other 51 already have a wrong STR base path, including 19 where the residual moves toward the correct class but is insufficient.

Across splits, base and residual DD margins remain positively correlated (mean Spearman 0.613), but their standalone predicted classes disagree for 25.3% of validation appearances. This is meaningful but incomplete complementarity. Net rescue rate correlates with BA gain (`rho=0.646`, raw `p=0.0092`) but does not survive the 12-test BH family (`q=0.0840`); no complementarity statistic is associated with AUROC gain after correction. Hence complementarity is supported by direct decision decomposition, while split-level strength-of-association evidence remains suggestive rather than confirmatory.

## 3. Activity-level mechanism

| Activity | Rescue residual true margin | Harm residual true margin | Positive in rescues | Largest contributor | Pivotal if removed |
|---|---:|---:|---:|---:|---:|
| Relaxed | +0.114 | -0.087 | 63.6% | 8.3% | 14.2% |
| RelaxedTask | +0.105 | -0.056 | 58.8% | 12.3% | 15.5% |
| TouchNose | +0.104 | +0.012 | 61.2% | 21.0% | 15.9% |
| HoldWeight | +0.090 | -0.123 | 64.2% | 17.0% | 21.8% |
| DrinkGlas | +0.079 | -0.036 | 64.3% | 26.5% | 23.6% |
| LiftHold | +0.032 | -0.034 | 64.7% | 3.9% | 9.1% |
| Entrainment | +0.031 | -0.047 | 57.8% | 2.2% | 7.8% |
| CrossArms | -0.015 | -0.111 | 55.5% | 8.4% | 12.7% |

The remaining activities have near-zero or negative mean rescue contribution and are rarely the largest contributor. No rescue-versus-harm activity difference survives BH (`minimum q=0.1296`). Specifically:

- HoldWeight: 12/15 split direction, `q=0.1296`.
- CrossArms: 10/15, `q=0.1752`.
- DrinkGlas: 10/15, `q=0.3153`.

No activity shows a significant standalone AUROC improvement after correction. PointFinger decreases by 0.1188 on 15/15 splits (`q=0.00134`) and Entrainment decreases by 0.0735 on 14/15 (`q=0.00201`). LiftHold has an uncorrected positive trend (+0.0359, 10/15) but `q=0.1296`.

Therefore STR's gain comes from the **ordered joint configuration of multiple weak and heterogeneous activity contributions**, not from uniformly better evidence in CrossArms, DrinkGlas, HoldWeight, or any other single activity. This directly explains why a structure-preserving joint readout helps while further per-activity architecture expansion is not justified.

## 4. Stable-error effect

### Primary seed-aggregated status

Using the frozen audit thresholds (`error rate ≤0.25`: stable-correct; `≥0.75`: stable-error) after aggregating the three seeds within each of four validation appearances:

- V8 stable-error: 85 subjects.
- Of these, STR makes 12 stable-correct, 10 unstable, and leaves 63 stable-error.
- STR also turns 4 V8 stable-correct and 7 V8 unstable subjects into stable-error.
- Net stable-error count: 85 → 74 (-11; -12.9%).

### Continuity with the historical audit

The earlier stable-error audit defined status over 12 individual seed-fold appearances. Reproducing that definition exactly gives:

- V8 stable-error: 54, exactly matching the frozen audit.
- Of these, STR makes 2 stable-correct, 18 unstable, and leaves 34 stable-error.
- Eight V8 unstable subjects become STR stable-error; no V8 stable-correct subject becomes STR stable-error.
- Net stable-error count: 54 → 42 (-12; -22.2%).

The apparent difference in counts is solely the prespecified aggregation unit. Both analyses agree that STR reduces stable-error burden, but most original stable errors remain or only become unstable. STR does not resolve the core phenotype-overlap population.

## 5. Matched representation comparison

Fixed train-only standardized logistic probes on unseen validation subjects:

| Representation | Disease probe BA | Disease probe AUROC | Centroid BA | Cosine silhouette | Domain AUC |
|---|---:|---:|---:|---:|---:|
| V8 subject | 0.6126 | 0.6599 | 0.6337 | 0.0360 | 0.6301 |
| STR subject path | 0.6127 | 0.6556 | 0.6350 | 0.0313 | 0.6390 |
| STR structured path | 0.6532 | 0.7138 | 0.6424 | 0.0321 | 0.6198 |
| STR combined decision representation | 0.6499 | 0.7064 | 0.6467 | 0.0315 | 0.6508 |

Versus V8, the STR combined representation improves disease-probe BA by 0.0374 (12/15, BH `q=0.0336`) and AUROC by 0.0465 (14/15, `q=0.0122`). The unchanged-size STR subject path alone does not improve either metric.

However, global geometry does not improve:

- Euclidean silhouette decreases by 0.0107 (14/15 worse, `q=0.0224`); cosine silhouette also trends downward.
- Domain AUC increases by 0.0207 but is not significant (`q=0.648`), so there is no evidence of greater split invariance.
- PD centroid drift increases by 0.702 (12/15, `q=0.0153`); DD drift increases by 0.990 with `q=0.0917`.
- Normalized mean shift is unchanged.

Cross-activity subject retrieval also does not explain the gain. Adding structured tokens lowers top-1, top-5 and MRR on all 15 splits (`q=0.00214`, `0.00031`, `0.00031`). Activity identity is already perfectly decodable from both V8 and STR context tokens, and remains 1.0. STR therefore adds disease-decision accessibility without improving general subject consistency or global class clustering.

## 6. Raw phenotype, frozen geometry, and H1 diagnostic

Persistent rescue is rare (6 subjects), so subject-level phenotype inference is low-powered. Among the five persistently rescued DD subjects, four are Other Movement Disorders and one is Essential Tremor; the single persistent PD rescue does not support subgroup inference.

Across 264 subject-level high-risk raw-signal comparisons for persistent rescue versus persistent both-wrong subjects, none survives BH (`minimum q=0.388`). There is no defensible stable raw Acc/Gyro phenotype for STR rescue. This is an important negative result.

The five persistent DD rescues are already somewhat closer to DD in frozen V8 geometry than persistent both-wrong DD subjects, but the V8 differences are weak (centroid-margin `q=0.0999`; kNN comparisons `q>0.13`). In STR representations the separation becomes much larger, although these comparisons remain exploratory because `n=5` rescues.

H1 agrees that rescue subjects are less deeply overlapping than both-wrong subjects: H1 is correct on 60.0% of rescued DD and 73.0% of rescued PD appearances, compared with 29.8%/38.4% for both-wrong. At the independent persistent-subject level, the five DD rescues have stronger H1 `band_fraction` contribution (`q=0.0035`) and an exploratory LiftHold contribution (`q=0.0768`). No subject-level inference is possible for the single persistent PD rescue. These results suggest that STR rescues a subset with recoverable multi-activity spectral/statistical evidence, but does not establish a new raw phenotype or causal H1 mechanism.

## Final answers

1. **Which subjects/activities drive the gain?** Near-boundary errors drive it: 139 validation appearances are rescued versus 93 harmed, with net benefit larger for DD. Only six subjects are persistently rescued. Activity support is distributed; DrinkGlas, HoldWeight, TouchNose, RelaxedTask and Relaxed frequently contribute, but no activity is independently stable after BH.
2. **Are the paths complementary?** Yes at the decision level. The residual path supplies high-AUROC information and directly flips 42.4% of rescues; joint base-path adaptation accounts for the other 57.6%. But complementarity is partial, not orthogonal, and its split-level association with gain is not BH-significant.
3. **Does STR reduce stable errors?** Partially. Counts fall under both aggregation definitions, but most original stable errors remain stable-error or merely become unstable. STR does not solve the persistent phenotype-overlap group.
4. **What should be studied next?** Cross-individual generalization representation learning. The useful cross-activity decision structure has already been demonstrated by STR, while more readout/decision refinement is not supported. The remaining negative geometry, retrieval, shift and stable-error evidence identifies subject-invariant generalization—not decision capacity—as the next bottleneck.

## Artifacts

All outputs are under `artifacts/str01_gain_mechanism_diagnosis_20260923/`:

- `representation_extract/embeddings/`: 45 paired V8/STR archives.
- `representation_extract/extraction_checks.csv`
- `analysis/validation_records_with_h1.csv`
- `analysis/subject_stability_transitions.csv`
- `analysis/decision_margin_by_group.csv`
- `analysis/internal_decision_patterns.csv`
- `analysis/activity_summary.csv`
- `analysis/activity_paired_inference.csv`
- `analysis/representation_metrics_15_splits.csv`
- `analysis/representation_paired_inference.csv`
- `analysis/token_metrics_15_splits.csv`
- `analysis/token_paired_inference.csv`
- `analysis/raw_signal_rescue_vs_both_wrong_bh.csv`
- `analysis/h1_rescue_vs_both_wrong_bh.csv`
- `analysis/geometry_rescue_vs_both_wrong_bh.csv`
- `analysis/protocol.json`
