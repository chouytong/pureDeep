# Item 06 — current WSSL independent-subject errors

Decision: **C1 source-category auxiliary test eligible**. No training or data exclusions.

Files: ERROR_ANALYSIS_PROTOCOL.md, scripts/analyze_wssl_errors.py, analysis/wssl_* aggregate CSV/JSON; private predictions/wssl_subject_error_summary.csv and features/dd_source_category_labels.json remain server-only. Shape/mask/reload/gradient tests for frozen WSSL were already verified in items01–02; this read-only diagnosis fits no model.

Primary grouping uses four context-level seed-mean decisions per independent subject. Consensus recall averages those scores before a fixed .5 threshold. Strict all-12-run grouping is sensitivity only and is never mixed with primary counts. Formal model performance remains original 15-split metrics.

## Independent diagnosis categories

| source_condition | n | consensus_recall | ci95_low | ci95_high | mean_context_recall | stable_error_n | stable_correct_n | unstable_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Atypical Parkinsonism | 15 | 0.666667 | 0.400000 | 0.866667 | 0.666667 | 2 | 7 | 6 |
| Essential Tremor | 28 | 0.821429 | 0.678571 | 0.964286 | 0.794643 | 3 | 18 | 7 |
| Multiple Sclerosis | 11 | 0.545455 | 0.272727 | 0.818182 | 0.568182 | 3 | 5 | 3 |
| Other Movement Disorders | 60 | 0.500000 | 0.366667 | 0.633333 | 0.554167 | 16 | 22 | 22 |
| Parkinson's | 276 | 0.833333 | 0.789855 | 0.876812 | 0.809783 | 14 | 176 | 86 |

## Primary error groups

| target | primary_group | n | consensus_recall | mean_context_recall |
| --- | --- | --- | --- | --- |
| 0 | stable_correct | 176 | 1.000000 | 1.000000 |
| 0 | stable_error | 14 | 0.000000 | 0.000000 |
| 0 | unstable | 86 | 0.627907 | 0.552326 |
| 1 | stable_correct | 52 | 1.000000 | 1.000000 |
| 1 | stable_error | 24 | 0.000000 | 0.000000 |
| 1 | unstable | 38 | 0.447368 | 0.519737 |

## Strict seed unanimity sensitivity

| target | sensitivity_group | n | consensus_recall | mean_seed_recall |
| --- | --- | --- | --- | --- |
| 0 | stable_correct | 111 | 1.000000 | 1.000000 |
| 0 | stable_error | 3 | 0.000000 | 0.000000 |
| 0 | unstable | 162 | 0.734568 | 0.647634 |
| 1 | stable_correct | 34 | 1.000000 | 1.000000 |
| 1 | stable_error | 9 | 0.000000 | 0.000000 |
| 1 | unstable | 71 | 0.492958 | 0.575117 |

## DD category paired exploratory comparisons

| category_a | category_b | n_a | n_b | recall_delta | ci95_low | ci95_high | fisher_p | bh_q | eligible_difference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Other Movement Disorders | Essential Tremor | 60 | 28 | -0.321429 | -0.507143 | -0.128571 | 0.004916 | 0.029499 | True |
| Other Movement Disorders | Atypical Parkinsonism | 60 | 15 | -0.166667 | -0.433333 | 0.100000 | 0.386068 | 0.579101 | False |
| Other Movement Disorders | Multiple Sclerosis | 60 | 11 | -0.045455 | -0.368182 | 0.277273 | 1.000000 | 1.000000 | False |
| Essential Tremor | Atypical Parkinsonism | 28 | 15 | 0.154762 | -0.121429 | 0.430952 | 0.280595 | 0.561191 | False |
| Essential Tremor | Multiple Sclerosis | 28 | 11 | 0.275974 | -0.048701 | 0.603896 | 0.108743 | 0.326229 | False |
| Atypical Parkinsonism | Multiple Sclerosis | 15 | 11 | 0.121212 | -0.242424 | 0.503030 | 0.689056 | 0.826868 | False |

## Quality by primary error group

| target | primary_group | n | max_offset_median | max_offset_q90 | offset_above_001_fraction | mean_raw_acc_outside_3g_fraction | subjects_any_raw_acc_above_3g |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | stable_correct | 176 | 0.147180 | 0.150389 | 1.000000 | 0.000100 | 73 |
| 0 | stable_error | 14 | 0.149561 | 0.249038 | 1.000000 | 0.000038 | 5 |
| 0 | unstable | 86 | 0.147348 | 0.230753 | 1.000000 | 0.000046 | 28 |
| 1 | stable_correct | 52 | 0.149227 | 0.248470 | 1.000000 | 0.000026 | 18 |
| 1 | stable_error | 24 | 0.147217 | 0.149921 | 1.000000 | 0.000026 | 7 |
| 1 | unstable | 38 | 0.147495 | 0.248143 | 1.000000 | 0.000018 | 12 |

Labels are consistent across all 11 activities and match PD/DD labels. Source diagnosis categories include heterogeneous Other Movement Disorders; granular adjudicated clinical subtype labels are unavailable. All sensor arrays are finite, complete and correctly length-matched, with 100Hz metadata. Wrist offset and extreme Acc are descriptive measures; they do not prove bad recordings or justify deletion. No validation-error list enters sampling.

Independent subject aggregation avoids counting 12 repeated predictions as 12 subjects. These subjects and fitted models overlap across development contexts, so subtype Fisher/BH and bootstrap results are exploratory repeated-development evidence, not external confirmation. Category N 11/15 is small. C1 eligibility follows the registered source-label/N/BH/CI gate; one weight .1, all four source categories if eligible, DD train labels only. No label or weight search.
