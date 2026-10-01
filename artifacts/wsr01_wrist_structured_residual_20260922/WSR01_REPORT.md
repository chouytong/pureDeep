# WSR-01 Wrist-Structured Residual Readout Report

Date: 2026-09-22

## Decision

**Reject WSR-01. STR-01 remains the frozen strong pure-deep backbone. Structured-readout architecture expansion is stopped.**

WSR-01 markedly improved fixed-probe recovery of the two preregistered H1 feature families, but did not produce a meaningful or stable classification gain. The result therefore supports the negative mechanism conclusion: explicitly preserving additional pre-fusion wrist structure is sufficient to make wrist-specific statistical information easier to linearly recover, but that information alone is insufficient to improve subject-level PD/DD classification under the current protocol.

## Preregistered hypothesis and single change

The experiment tested whether STR-01 still loses discriminative information by preserving activity identity only after bilateral fusion. All STR-01 paths were retained unchanged. For each of 11 activities, the pre-fusion left and right 64-D wrist embeddings were passed through one shared `Linear(64, 4) + GELU` projection. The ordered `11 activities × 2 wrists × 4 dimensions` tokens were masked, flattened to 88 D, and passed through one `Linear(88, 2)` residual classifier. Its weights and bias were zero-initialized, so the initial decision function exactly matched STR-01.

No projection dimension, depth, attention, gating, fusion, loss, optimizer, preprocessing, split, or seed search was performed. The fixed protocol used the same 5×3 inner-development splits and seeds 42/43/44; no outer information was accessed.

Parameter accounting:

- STR-01: 75,524 trainable parameters.
- Shared wrist projection: `64×4 + 4 = 260`.
- Residual classifier: `88×2 + 2 = 178`.
- WSR-01: 75,962 parameters, an increment of 438 (+0.58%).

## Classification results

Three seeds were aggregated before the 15 independent split-level comparisons.

| Metric | STR-01 mean | WSR-01 mean | Delta | W/L/T | Wilcoxon p | BH q | Rank-biserial |
|---|---:|---:|---:|---:|---:|---:|---:|
| Accuracy | 0.7307 | 0.7234 | -0.0073 | 5/9/1 | 0.2092 | 0.4326 | -0.3810 |
| Balanced accuracy | 0.6972 | 0.6965 | -0.0007 | 7/8/0 | 0.7764 | 0.7764 | -0.0833 |
| Macro-F1 | 0.6861 | 0.6824 | -0.0036 | 6/9/0 | 0.3894 | 0.4673 | -0.2667 |
| AUROC | 0.7176 | 0.7216 | +0.0039 | 11/4/0 | 0.0353 | 0.2120 | +0.6167 |
| PD recall | 0.7779 | 0.7615 | -0.0163 | 5/8/2 | 0.2163 | 0.4326 | -0.4066 |
| DD recall | 0.6165 | 0.6315 | +0.0150 | 8/4/3 | 0.3101 | 0.4651 | +0.3462 |

The small AUROC rise was below the preregistered +0.015 threshold, did not survive BH correction, and co-occurred with lower accuracy, Macro-F1, and PD recall. The recall changes show a class trade-off rather than a joint improvement.

### Seed stability

| Model / seed | Accuracy | BA | Macro-F1 | AUROC | PD recall | DD recall |
|---|---:|---:|---:|---:|---:|---:|
| STR-01 / 42 | 0.7457 | 0.7084 | 0.6985 | 0.7276 | 0.7980 | 0.6187 |
| STR-01 / 43 | 0.7155 | 0.6981 | 0.6797 | 0.7209 | 0.7401 | 0.6562 |
| STR-01 / 44 | 0.7309 | 0.6851 | 0.6800 | 0.7044 | 0.7955 | 0.5746 |
| WSR-01 / 42 | 0.7322 | 0.7100 | 0.6942 | 0.7353 | 0.7638 | 0.6563 |
| WSR-01 / 43 | 0.7109 | 0.6918 | 0.6734 | 0.7204 | 0.7381 | 0.6455 |
| WSR-01 / 44 | 0.7271 | 0.6877 | 0.6797 | 0.7090 | 0.7827 | 0.5926 |

The BA and AUROC deltas were positive for seeds 42 and 44 but negative for seed 43, so the required three-seed directional agreement was absent. BA seed SD was 0.0119 versus 0.0117 for STR-01, and AUROC seed SD was 0.0132 versus 0.0120; stability did not improve.

## Fixed-probe mechanism analysis

Ridge probes with fixed `alpha=1.0` were fitted only on each inner-train partition after train-only representation and target scaling, then evaluated on unseen inner-validation subjects. Targets were the complete activity/wrist-specific `time_location_scale` and `band_fraction` vectors. Recoverability was auxiliary mechanism evidence and was not used for selection.

| Representation | Family | R² | Median Spearman | normalized MAE |
|---|---|---:|---:|---:|
| STR activity structured (176 D) | time_location_scale | -0.2096 | 0.3665 | 0.8369 |
| WSR wrist structured (88 D) | time_location_scale | -0.0648 | 0.4069 | 0.7387 |
| STR decision input (434 D) | time_location_scale | -0.2804 | 0.3536 | 0.8655 |
| WSR decision input (522 D) | time_location_scale | -0.1788 | 0.3826 | 0.8122 |
| STR activity structured (176 D) | band_fraction | -0.7356 | 0.2140 | 1.0290 |
| WSR wrist structured (88 D) | band_fraction | -0.3335 | 0.2259 | 0.8871 |
| STR decision input (434 D) | band_fraction | -1.0513 | 0.1970 | 1.1238 |
| WSR decision input (522 D) | band_fraction | -0.8073 | 0.2229 | 1.0532 |

The new wrist representation improved all three recoverability metrics for both families. Wrist-versus-STR-activity R² and normalized-MAE improvements occurred on 15/15 splits for both families; Spearman improved on 13/15 (`time_location_scale`) and 12/15 (`band_fraction`). All six comparisons remained significant after BH correction (`q ≤ 0.00641`).

Adding wrist structure to the full decision input also improved all three metrics for both families: five comparisons improved on 15/15 splits and the remaining `time_location_scale` Spearman comparison improved on 14/15. All six remained significant after BH correction (`q ≤ 0.000499`). The retrained STR activity path itself remained effectively unchanged relative to matched STR-01, providing a useful negative control.

However, split-level recoverability improvements did not track BA or AUROC improvements: none of the 24 exploratory correlations survived BH correction (minimum `q = 0.7388`). Thus the experiment provides no evidence that the small AUROC change was caused by preservation of H1-like wrist statistics.

## Preregistered decision audit

| Criterion | Required | Observed | Pass |
|---|---:|---:|---|
| BA delta | ≥ +0.010 | -0.0007 | No |
| AUROC delta | ≥ +0.015 | +0.0039 | No |
| Macro-F1 | non-decreasing | -0.0036 | No |
| No one-sided recall sacrifice | required | PD -0.0163, DD +0.0150 | No |
| Split wins | about ≥10/15 on main metrics | BA 7/15; AUROC 11/15; F1 6/15 | No |
| Three-seed direction | consistent | BA and AUROC +/−/+ | No |
| Seed stability | not materially worse | BA/AUROC SD slightly higher | No clear gain |

## Verification and evidence boundary

- Model/unit tests: 19 passed.
- Zero-initialized WSR residual exactly reproduced STR-01 logits at initialization.
- Representation archives: 45/45 seed-fold files.
- Extracted dimensions: STR activity structure 176 D, wrist structure 88 D, STR decision input 434 D, WSR decision input 522 D.
- Maximum extracted logit-combination error: 0.0.
- Statistical unit: independent split/subject, with seeds aggregated first.
- Wilcoxon signed-rank tests, paired rank-biserial effects, and BH correction were used as specified.
- No outer-final/test predictions, labels, metrics, or metadata were used.

## Final interpretation

WSR-01 confirms that compact ordered pre-fusion wrist tokens preserve substantial activity/wrist-specific H1 information that is not as linearly accessible from STR-01's decision representation. This is a representation-accessibility result, not a classification improvement. Because the extra recoverability did not translate into BA, AUROC, or Macro-F1 gains, WSR-01 is rejected and no wrist projection, gating, attention, or further structured-readout variants will be searched. STR-01 remains the frozen strong pure-deep backbone.

## Artifacts

- `analysis/classification_15_paired_splits.csv`
- `analysis/classification_paired_inference.csv`
- `analysis/classification_seed_metrics.csv`
- `analysis/classification_seed_summary.csv`
- `analysis/recoverability_45_seed_folds.csv`
- `analysis/recoverability_15_splits.csv`
- `analysis/recoverability_summary.csv`
- `analysis/recoverability_paired_inference.csv`
- `analysis/recoverability_classification_relationships.csv`
- `analysis/protocol.json`
- `representation_extract/extraction_checks.csv`
- `representation_extract/protocol.json`
