# Stable-error phenotype & data audit

## Protocol and evidence boundary

This audit is development-only. It uses the frozen stable-error labels, 45 inner-development embedding files, seed aggregation within each `(outer, inner)` split, subject-level raw/processed signals, and patient metadata. It does not read outer-final/test predictions, metrics, or artifacts and does not train or update a model.

## Cohort

| label_name | stable_correct | stable_error | unstable |
| --- | --- | --- | --- |
| DD | 47 | 34 | 33 |
| PD | 200 | 20 | 56 |

DD subtype counts:

| condition | stable_correct | stable_error | unstable |
| --- | --- | --- | --- |
| Atypical Parkinsonism | 7 | 5 | 3 |
| Essential Tremor | 18 | 5 | 5 |
| Multiple Sclerosis | 4 | 4 | 3 |
| Other Movement Disorders | 18 | 20 | 22 |

## Metadata

Continuous tests significant after BH-FDR: 1/24.

| label_name | feature | group_b | mean_a | mean_b | cliffs_delta | q_value |
| --- | --- | --- | --- | --- | --- | --- |
| DD | disease_duration | stable_correct | 10.62 | 21.6 | -0.4186 | 0.0327 |

Categorical omnibus tests significant after BH-FDR: 1/26.

| index | label_name | feature | cramers_v | q_value |
| --- | --- | --- | --- | --- |
| 4 | DD | appearance_in_first_grade_kinship | 0.2838 | 0.02729 |
| 17 | PD | appearance_in_first_grade_kinship | 0.1474 | 0.2259 |
| 18 | PD | effect_of_alcohol_on_tremor | 0.1532 | 0.2851 |
| 10 | DD | comment_dystonia | 0.2382 | 0.2851 |
| 23 | PD | comment_dystonia | 0.1428 | 0.3114 |
| 20 | PD | comment_tremor | 0.1245 | 0.3401 |
| 12 | DD | comment_functional | 0.2094 | 0.3401 |
| 0 | DD | condition | 0.2123 | 0.3401 |
| 22 | PD | comment_rigidity | 0.1288 | 0.3401 |
| 21 | PD | comment_hypokinesia | 0.1162 | 0.3669 |
| 25 | PD | comment_vascular | 0.1165 | 0.3669 |
| 3 | DD | appearance_in_kinship | 0.155 | 0.5203 |

Significant one-vs-rest categorical levels:

| index | label_name | feature | level | group_b | fraction_a | fraction_b | odds_ratio | q_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 22 | DD | appearance_in_first_grade_kinship | Missing | stable_correct | 0.7647 | 0.3404 | 6.297 | 0.0333 |
| 20 | DD | appearance_in_first_grade_kinship | False | stable_correct | 0.02941 | 0.3404 | 0.05871 | 0.03913 |

## Data quality

Significant quality comparisons after BH-FDR: 74/108.

| index | label_name | activity_scope | feature | group_b | mean_a | mean_b | cliffs_delta | q_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 62 | PD | all | raw_zero_difference_fraction | stable_correct | 1.721e-05 | 0.0002091 | -0.9098 | 4.685e-10 |
| 76 | PD | high_risk | raw_outlier_fraction | stable_correct | 0.001921 | 0.01817 | -0.902 | 4.685e-10 |
| 68 | PD | all | processed_maximum_constant_run | stable_correct | 1.073 | 1.326 | -0.906 | 4.685e-10 |
| 66 | PD | all | raw_maximum_constant_run | stable_correct | 1.082 | 1.352 | -0.9125 | 4.685e-10 |
| 58 | PD | all | raw_outlier_fraction | stable_correct | 0.0008674 | 0.007303 | -0.9067 | 4.685e-10 |
| 64 | PD | all | processed_zero_difference_fraction | stable_correct | 1.55e-05 | 0.0001996 | -0.9032 | 4.685e-10 |
| 60 | PD | all | processed_outlier_fraction | stable_correct | 0.000679 | 0.005341 | -0.9022 | 4.685e-10 |
| 14 | DD | all | processed_maximum_constant_run | stable_correct | 1.337 | 1.1 | 0.8586 | 5.117e-10 |
| 78 | PD | high_risk | processed_outlier_fraction | stable_correct | 0.001542 | 0.01347 | -0.8968 | 5.117e-10 |
| 98 | PD | other | raw_zero_difference_fraction | stable_correct | 2.007e-05 | 0.0002416 | -0.892 | 5.381e-10 |
| 100 | PD | other | processed_zero_difference_fraction | stable_correct | 1.886e-05 | 0.0002308 | -0.885 | 6.901e-10 |
| 50 | DD | other | processed_maximum_constant_run | stable_correct | 1.342 | 1.109 | 0.8373 | 9.201e-10 |

## Raw signal time/frequency audit

Significant signal comparisons after BH-FDR: 604/1952.

| index | label_name | activity | sensor | scope | feature | group_b | mean_difference | cliffs_delta | q_value | direction_consistency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1152 | PD | CrossArms | gyro | bilateral_mean | dominant_frequency_0p5_12 | stable_correct | 2.432 | 0.7625 | 2.422e-18 | 1 |
| 1184 | PD | CrossArms | gyro | left | dominant_frequency_0p5_12 | stable_correct | 2.346 | 0.6612 | 4.514e-17 | 1 |
| 1122 | PD | CrossArms | gyro | bilateral_asymmetry | dominant_frequency_0p5_12 | stable_correct | 1.688 | 0.6487 | 9.934e-15 | 0.9286 |
| 1214 | PD | CrossArms | gyro | right | dominant_frequency_0p5_12 | stable_correct | 2.518 | 0.572 | 1.437e-14 | 1 |
| 194 | DD | CrossArms | gyro | left | robust_outlier_fraction | stable_correct | 0.05386 | 0.8936 | 4.177e-10 | 1 |
| 162 | DD | CrossArms | gyro | bilateral_mean | robust_outlier_fraction | stable_correct | 0.05426 | 0.8867 | 8.678e-10 | 1 |
| 1396 | PD | DrinkGlas | gyro | bilateral_mean | dominant_frequency_0p5_12 | stable_correct | 1.754 | 0.5477 | 1.885e-09 | 1 |
| 1366 | PD | DrinkGlas | gyro | bilateral_asymmetry | dominant_frequency_0p5_12 | stable_correct | 2.18 | 0.533 | 5.249e-09 | 0.9286 |
| 1200 | PD | CrossArms | gyro | right | robust_outlier_fraction | stable_correct | -0.07072 | -0.9078 | 5.249e-09 | 1 |
| 1144 | PD | CrossArms | gyro | bilateral_mean | bandpower_0p5_3 | stable_correct | -0.4505 | -0.902 | 5.928e-09 | 1 |
| 1138 | PD | CrossArms | gyro | bilateral_mean | robust_outlier_fraction | stable_correct | -0.0698 | -0.895 | 7.49e-09 | 1 |
| 224 | DD | CrossArms | gyro | right | robust_outlier_fraction | stable_correct | 0.05465 | 0.8166 | 1.346e-08 | 1 |
| 180 | DD | CrossArms | gyro | bilateral_mean | tremor_peak_ratio_3_7 | stable_correct | -7.216 | -0.8486 | 1.346e-08 | 1 |
| 1156 | PD | CrossArms | gyro | bilateral_mean | tremor_peak_ratio_3_7 | stable_correct | 8.326 | 0.8705 | 1.994e-08 | 1 |
| 1036 | PD | CrossArms | acc | bilateral_relation | zero_lag_magnitude_correlation | stable_correct | -0.38 | -0.869 | 2.001e-08 | 1 |
| 1170 | PD | CrossArms | gyro | left | robust_outlier_fraction | stable_correct | -0.06888 | -0.852 | 4.072e-08 | 1 |
| 1146 | PD | CrossArms | gyro | bilateral_mean | bandpower_3_7 | stable_correct | 0.3395 | 0.8515 | 4.075e-08 | 1 |
| 1158 | PD | CrossArms | gyro | bilateral_relation | zero_lag_magnitude_correlation | stable_correct | -0.4214 | -0.85 | 4.132e-08 | 1 |
| 1176 | PD | CrossArms | gyro | left | bandpower_0p5_3 | stable_correct | -0.426 | -0.833 | 8.676e-08 | 1 |
| 170 | DD | CrossArms | gyro | bilateral_mean | bandpower_3_7 | stable_correct | -0.2761 | -0.7935 | 1.19e-07 | 1 |

Frequency-branch evidence rule required DD stable-error vs stable-correct, BH q<0.05, |Cliff's delta|>=0.33, and same-direction difference in at least 12/15 development splits.

Frequency features meeting that exploratory rule: 96.

| index | activity | sensor | scope | feature | mean_a | mean_b | cliffs_delta | q_value | direction_consistency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 180 | CrossArms | gyro | bilateral_mean | tremor_peak_ratio_3_7 | 1.446 | 8.662 | -0.8486 | 1.346e-08 | 1 |
| 170 | CrossArms | gyro | bilateral_mean | bandpower_3_7 | 0.1248 | 0.4009 | -0.7935 | 1.19e-07 | 1 |
| 242 | CrossArms | gyro | right | tremor_peak_ratio_3_7 | 1.291 | 8.6 | -0.7935 | 1.19e-07 | 1 |
| 168 | CrossArms | gyro | bilateral_mean | bandpower_0p5_3 | 0.7342 | 0.3639 | 0.791 | 1.229e-07 | 1 |
| 176 | CrossArms | gyro | bilateral_mean | dominant_frequency_0p5_12 | 0.8559 | 2.83 | -0.6902 | 4.977e-07 | 1 |
| 230 | CrossArms | gyro | right | bandpower_0p5_3 | 0.7563 | 0.3743 | 0.7547 | 5.474e-07 | 1 |
| 232 | CrossArms | gyro | right | bandpower_3_7 | 0.1175 | 0.3829 | -0.7447 | 8.039e-07 | 1 |
| 212 | CrossArms | gyro | left | tremor_peak_ratio_3_7 | 1.601 | 8.725 | -0.7322 | 1.278e-06 | 1 |
| 202 | CrossArms | gyro | left | bandpower_3_7 | 0.1321 | 0.4189 | -0.6971 | 4.793e-06 | 1 |
| 200 | CrossArms | gyro | left | bandpower_0p5_3 | 0.7121 | 0.3534 | 0.6959 | 4.932e-06 | 1 |
| 208 | CrossArms | gyro | left | dominant_frequency_0p5_12 | 0.9306 | 3.083 | -0.612 | 6.583e-06 | 1 |
| 238 | CrossArms | gyro | right | dominant_frequency_0p5_12 | 0.7812 | 2.576 | -0.5532 | 1.422e-05 | 1 |

## Frozen representation geometry

| index | label_name | feature | group_b | mean_a | mean_b | cliffs_delta | q_value |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | DD | distance_pd_centroid | stable_correct | 13.24 | 21.36 | -0.9812 | 3.028e-13 |
| 4 | DD | pd_like_centroid_margin | stable_correct | 4.175 | -3.134 | 0.9837 | 3.028e-13 |
| 6 | DD | knn5_pd_fraction | stable_correct | 0.8632 | 0.3819 | 0.9887 | 3.028e-13 |
| 8 | DD | knn5_same_label_fraction | stable_correct | 0.1368 | 0.6181 | -0.9887 | 3.028e-13 |
| 10 | DD | knn10_pd_fraction | stable_correct | 0.8635 | 0.414 | 0.9862 | 3.028e-13 |
| 12 | DD | knn10_same_label_fraction | stable_correct | 0.1365 | 0.586 | -0.9862 | 3.028e-13 |
| 14 | PD | distance_pd_centroid | stable_correct | 22.04 | 13.24 | 0.9995 | 4.458e-13 |
| 18 | PD | pd_like_centroid_margin | stable_correct | -2.514 | 4.413 | -1 | 4.458e-13 |
| 24 | PD | knn10_pd_fraction | stable_correct | 0.4367 | 0.8756 | -0.998 | 4.458e-13 |
| 22 | PD | knn5_same_label_fraction | stable_correct | 0.4183 | 0.8782 | -0.9985 | 4.458e-13 |
| 20 | PD | knn5_pd_fraction | stable_correct | 0.4183 | 0.8782 | -0.9985 | 4.458e-13 |
| 26 | PD | knn10_same_label_fraction | stable_correct | 0.4367 | 0.8756 | -0.998 | 4.458e-13 |

## DD activity-wise PD-like evidence

| index | activity | feature | group_b | mean_a | mean_b | cliffs_delta | q_value |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | CrossArms | pd_like_score | stable_correct | 2.266 | -2.078 | 0.9662 | 4.334e-12 |
| 22 | nan | pd_like_activity_mean | stable_correct | 0.6939 | -0.5335 | 0.8348 | 2.509e-09 |
| 26 | nan | pd_like_activity_fraction | stable_correct | 0.6932 | 0.4139 | 0.8066 | 6.453e-09 |
| 18 | TouchIndex | pd_like_score | stable_correct | 0.8379 | -0.4801 | 0.6033 | 2.849e-05 |
| 2 | DrinkGlas | pd_like_score | stable_correct | 0.9321 | -1.153 | 0.592 | 3.436e-05 |
| 10 | PointFinger | pd_like_score | stable_correct | 0.6566 | -0.666 | 0.5757 | 5.116e-05 |
| 4 | Entrainment | pd_like_score | stable_correct | 0.88 | 0.1126 | 0.5194 | 0.0002916 |
| 5 | Entrainment | pd_like_score | unstable | 0.88 | 0.07185 | 0.5472 | 0.0004239 |
| 24 | nan | pd_like_activity_sd | stable_correct | 1.333 | 1.823 | -0.4406 | 0.002153 |
| 6 | HoldWeight | pd_like_score | stable_correct | 0.4267 | -1.374 | 0.4418 | 0.002153 |
| 1 | CrossArms | pd_like_score | unstable | 2.266 | 1.196 | 0.4189 | 0.00833 |
| 20 | TouchNose | pd_like_score | stable_correct | 0.4389 | -0.3122 | 0.3655 | 0.01225 |

## Interpretation boundary

Associations with a frozen stable-error label are descriptive, not causal. The same subjects contribute to several development validation appearances, so inferential tests use one aggregated row per subject; split-direction counts are robustness checks, not additional independent samples. Free-text clinical keyword flags are lexical summaries only. Frequency findings are hypothesis-generating and cannot justify a frequency network unless they meet the stated stability rule and remain clinically/data-quality interpretable.
