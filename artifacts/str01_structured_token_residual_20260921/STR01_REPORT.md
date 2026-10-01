# STR-01 Structured Token Residual Readout Report

Date: 2026-09-21

## Decision

**Retain STR-01 and replace V8-GN as the frozen strong pure-deep backbone for the next research stage.**

STR-01 satisfies the preregistered classification criteria: compared with matched V8-GN, balanced
accuracy increases by 0.0265, AUROC by 0.0377, and Macro-F1 by 0.0230. Balanced accuracy improves
on 14/15 independent development splits, AUROC on 15/15, and Macro-F1 on 12/15. PD recall is
essentially unchanged while DD recall increases by 0.0534. The effect is positive for BA and AUROC
in every one of the three seeds.

Mechanistically, the 176-D ordered structured residual representation makes the complete
activity/wrist-specific `time_location_scale` and `band_fraction` H1 targets substantially more
recoverable than the original V8 subject embedding on all 15 splits. However, the split-to-split
magnitude of the recoverability gain is not correlated with the classification gain after
multiple-comparison correction. The result therefore supports, but does not prove, the proposed
information-preservation mechanism.

## Preregistered architecture and fixed protocol

All V8-GN components and the original classifier path were retained. The only change was a residual
decision path before subject aggregation:

1. apply one shared `Linear(258, 16) + GELU` projection to each of the 11 bilateral activity
   representations;
2. keep the fixed 11-activity identity and ordering, masking unavailable activities;
3. flatten the ordered tokens to 176 dimensions;
4. apply one `Linear(176, 2)` residual head;
5. add its logits directly to the original V8 logits.

The residual head was zero-initialized, so the initial STR-01 decision function exactly matched the
original V8 path. No projection dimension, depth, attention, gating, or other architecture was
searched. Parameters increased from 71,026 to 75,524: +4,498, or 6.33%.

The fixed PADS PD-vs-DD task, 5×3 subject-level inner-development splits, train-only normalization,
Full bilateral fusion, original Activity Attention, balanced CE, AdamW (`lr=2e-4`, batch 8,
`weight_decay=1e-4`), cosine scheduler, and seeds 42/43/44 were unchanged. No outer-final/test
loader, signal, prediction, or metadata was accessed.

## Classification results

Three-seed means:

| Model | Params | Accuracy | BA | Macro-F1 | AUROC | PD recall | DD recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| V8-GN | 71,026 | 0.7154 | 0.6707 | 0.6631 | 0.6800 | 0.7782 | 0.5631 |
| STR-01 | 75,524 | 0.7307 | 0.6972 | 0.6861 | 0.7176 | 0.7779 | 0.6165 |
| Delta | +4,498 | +0.0154 | +0.0265 | +0.0230 | +0.0377 | -0.0004 | +0.0534 |

STR-01 seed values and sample SD:

| Metric | Seed 42 | Seed 43 | Seed 44 | Mean | SD | V8 SD |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy | 0.7457 | 0.7155 | 0.7309 | 0.7307 | 0.0151 | 0.0177 |
| BA | 0.7084 | 0.6981 | 0.6850 | 0.6972 | 0.0117 | 0.0088 |
| Macro-F1 | 0.6985 | 0.6797 | 0.6800 | 0.6861 | 0.0107 | 0.0137 |
| AUROC | 0.7276 | 0.7209 | 0.7044 | 0.7176 | 0.0120 | 0.0103 |
| PD recall | 0.7980 | 0.7401 | 0.7955 | 0.7779 | 0.0327 | 0.0303 |
| DD recall | 0.6187 | 0.6562 | 0.5746 | 0.6165 | 0.0408 | 0.0134 |

BA and AUROC dispersion increased modestly, and DD-recall dispersion increased more clearly. This
is a limitation, but it does not overturn the preregistered stability requirement because BA and
AUROC improved over their matched V8 seed in all three seeds, Macro-F1 SD decreased, and neither
class recall was sacrificed in the aggregate.

The independent unit for paired inference was the 15 fixed splits after averaging the three seeds:

| Metric | Mean delta | Wins | Wilcoxon p | Paired rank-biserial | BH q (6 metrics) |
|---|---:|---:|---:|---:|---:|
| Accuracy | +0.0154 | 9/15 | 0.0906 | +0.524 | 0.1087 |
| BA | +0.0265 | 14/15 | 0.000183 | +0.967 | 0.000549 |
| Macro-F1 | +0.0230 | 12/15 | 0.000854 | +0.900 | 0.001709 |
| AUROC | +0.0377 | 15/15 | 0.000061 | +1.000 | 0.000366 |
| PD recall | -0.0004 | 7/15 | 0.8871 | -0.042 | 0.8871 |
| DD recall | +0.0534 | 12/15 | 0.0413 | +0.600 | 0.0619 |

The primary BA, AUROC, and Macro-F1 improvements survive BH correction. Accuracy and DD-recall
improvements do not survive correction, while PD recall is a clear null result.

## H1 recoverability mechanism analysis

The fixed probe was the same train-only standardized `Ridge(alpha=1.0)` used in the preceding H1 gap
diagnosis. It predicted each complete subject-level H1 family vector on unseen inner-validation
subjects. No probe hyperparameter was selected from validation performance.

| Representation | Family | OOS R² | Median feature Spearman | Normalized MAE |
|---|---|---:|---:|---:|
| V8 subject embedding | time_location_scale | -0.689 | 0.232 | 1.007 |
| STR subject embedding | time_location_scale | -0.680 | 0.230 | 1.030 |
| STR structured residual | time_location_scale | -0.210 | 0.366 | 0.837 |
| STR decision input | time_location_scale | -0.280 | 0.354 | 0.866 |
| V8 subject embedding | band_fraction | -1.413 | 0.144 | 1.198 |
| STR subject embedding | band_fraction | -1.481 | 0.135 | 1.223 |
| STR structured residual | band_fraction | -0.736 | 0.214 | 1.029 |
| STR decision input | band_fraction | -1.051 | 0.197 | 1.124 |

The new original-path subject embedding alone does not improve either family, a useful negative
result. In contrast, the structured residual representation improves R², median feature Spearman,
and normalized MAE for both key families on 15/15 splits. For `time_location_scale`, oriented mean
deltas versus the V8 subject embedding are +0.480 R², +0.134 Spearman, and +0.170 normalized-MAE
reduction. For `band_fraction`, they are +0.678, +0.070, and +0.169. All six comparisons have
Wilcoxon p=0.000061, paired rank-biserial=+1.0, and BH q=0.000092 across the 18 key mechanism tests.

Absolute OOS R² remains negative, especially for `band_fraction`. STR-01 therefore preserves more
of the high-dimensional H1 structure, but does not make the full handcrafted feature vector
accurately reconstructable. Adding the original subject embedding to the structured representation
also does not improve the fixed Ridge probe over the structured representation alone.

Across the 15 splits, none of the 24 exploratory correlations between recoverability change and
BA/AUROC change survives BH correction (all q=0.980). Thus classification improvement and improved
recoverability co-occur globally and are each stable, but their split-wise effect magnitudes do not
track each other. A direct causal claim that H1 recovery produced the classification gain is not
supported.

## Evidence boundary and next state

STR-01 directly answers the registered question: retaining ordered pre-aggregation activity tokens
until the decision stage yields a substantive and stable development-only classification gain under
the matched protocol. The evidence is compatible with premature compression being a real V8-GN
bottleneck. It does not show that all phenotype overlap is resolved, that full wrist-specific H1
structure is recovered, or that the effect generalizes to outer-final/test data.

STR-01 now becomes the frozen strong pure-deep backbone. No additional STR dimensions, MLP depths,
attention variants, handcrafted inputs, reconstruction losses, spectral modules, Transformers,
DANN/MMD/prototype losses, fusion variants, or aggregation searches are authorized by this result.
The H1 diagnostic still has higher development AUROC (0.7515 versus 0.7176), although STR-01 now has
slightly higher BA (0.6972 versus 0.6918). The remaining AUROC gap is 0.0339 and must not be hidden by
unbounded module addition.

## Artifacts

- `classification/seed_stability.json`: original seed and paired-split summary.
- `analysis/classification_15_paired_splits.csv`: all matched 15-split results.
- `analysis/classification_paired_inference.csv`: Wilcoxon, effect size, and BH correction.
- `analysis/classification_seed_metrics.csv`: per-seed metrics.
- `analysis/recoverability_45_seed_folds.csv`: raw fixed-probe results.
- `analysis/recoverability_15_splits.csv`: seed-aggregated independent split results.
- `analysis/recoverability_summary.csv`: representation/family summaries.
- `analysis/recoverability_paired_inference.csv`: paired mechanism inference.
- `analysis/recoverability_classification_relationships.csv`: exploratory synchronization tests.
- `representation_extract/embeddings/`: 45 train/validation representation archives.
- `representation_extract/extraction_checks.csv`: exact residual-logit combination checks.

