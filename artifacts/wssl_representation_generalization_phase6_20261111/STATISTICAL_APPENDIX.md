# Statistical appendix

## Units and estimands

There are45 seed/split models,15 overlapping development splits and390 unique patients (276PD/114DD). Each appears in4 splits and3 seeds. Activities, wrists, windows and repeated appearances are not independent patients. All feature metrics start as22 matched activity/wrist values and reduce to one scalar per patient/model.

Seed stability:3 seeds within a fixed split. Split stability:15 seed-first unit estimates. Subject-cluster uncertainty:10000 class-stratified patient resamples, bootstrap seed20261111, fixed historical fits. External generalization: not estimated. Neither resampling method cures prior development exposure or estimates new training-fit uncertainty. The split-bootstrap CI and Wilcoxon p-values assume a unit structure that overlaps; they are descriptive stability summaries, not independent-cohort confirmation. BH addresses comparison multiplicity, not split dependence or repeated research reuse.

Phase6A primary = equal15-unit mean of3-seed mean within-seed Spearman correlations. Subject sensitivity = correlation of each patient's12 scalar means. Group sensitivity uses the pre-existing4-appearance correctness-difference patient groups. Do not average vectors across different learned coordinates. Phase6B split estimand =15 equal unit contrasts after seed averaging; cluster estimand = pooled exposure ratio contrasts with all appearances retained under patient resampling. These are deliberately distinct, with no opportunistic choice of interval.

## Predefined families and undefined outcomes

The original PROTOCOL and source freeze preceded representation statistics: A_primary3 (layer5×all/PD/DD margin correlations); A_exploratory18 (3variable layers×2outcomes×3classes); A_group6 (3variable layers×improved−worsened/DD−PD); B_interaction3 (error/true margin/absolute margin).

| family | planned | testable | undefined |
| --- | --- | --- | --- |
| A_exploratory | 18 | 15 | 3 |
| A_group | 6 | 3 | 3 |
| A_primary | 3 | 3 | 0 |


An implementation limitation was found during report verification: original BH adjusted only finite p-values, so exploratory m=15 and group m=3 despite planned18/6. The original outputs are retained in `representation_paired_tests.csv`. A clearly post-result conservative sensitivity in `representation_tests_planned_BH_sensitivity.csv` includes unavailable comparisons as p=1 for adjustment only; their own p/q remainNA, raw p/CI are unchanged. Reports use primaryq=.389404 and conservative planned6q=.000366 for the DD−PD layer5 contrast. This correction is disclosed, not represented as pre-result implementation. B and A_primary were complete from the start.

All3 DD flip correlations lack2 complete units because flip is constant; NA13/15. All3 appearance improved−worsened contrasts lack1 complete unit; NA14/15. No unit, negative result or class was removed to manufacture a full test. No inference is supplied for layer4's constant displacement. Patient-level cluster intervals for exploratory associations are unadjusted, descriptive sensitivities; do not read them as new family-wise discoveries.

## Full split tests with conservative sensitivity

| test | family | status | units | mean_delta | ci_low | ci_high | wilcoxon_p | bh_q | bh_q_planned_family_sensitivity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final_wrist/DD/margin_change | A_exploratory | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | -0.010535 | -0.078938 | 0.070099 | 0.524475 | 0.561938 | 0.674325 |
| final_wrist/DD/prediction_flip | A_exploratory | NA_INCOMPLETE_15_UNITS | 13 | NA | NA | NA | NA | NA | NA |
| final_wrist/PD/margin_change | A_exploratory | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.137567 | 0.081103 | 0.188126 | 0.001160 | 0.004349 | 0.005219 |
| final_wrist/PD/prediction_flip | A_exploratory | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.038689 | -0.004234 | 0.085296 | 0.187622 | 0.281433 | 0.337720 |
| final_wrist/all/margin_change | A_exploratory | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.115128 | 0.085638 | 0.143246 | 0.000122 | 0.000916 | 0.001099 |
| final_wrist/all/prediction_flip | A_exploratory | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.042957 | 0.008806 | 0.075934 | 0.041260 | 0.088414 | 0.106097 |
| layer5/DD/margin_change | A_exploratory | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.039317 | -0.044668 | 0.122806 | 0.389404 | 0.486755 | 0.584106 |
| layer5/DD/margin_change | A_primary | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.039317 | -0.044668 | 0.122806 | 0.389404 | 0.389404 | 0.389404 |
| layer5/DD/prediction_flip | A_exploratory | NA_INCOMPLETE_15_UNITS | 13 | NA | NA | NA | NA | NA | NA |
| layer5/PD/margin_change | A_exploratory | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.030796 | -0.027332 | 0.088088 | 0.276855 | 0.377530 | 0.453036 |
| layer5/PD/margin_change | A_primary | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.030796 | -0.027332 | 0.088088 | 0.276855 | 0.389404 | 0.389404 |
| layer5/PD/prediction_flip | A_exploratory | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.045992 | 0.017906 | 0.070225 | 0.005371 | 0.016113 | 0.019336 |
| layer5/all/margin_change | A_exploratory | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.021579 | -0.005612 | 0.048844 | 0.187622 | 0.281433 | 0.337720 |
| layer5/all/margin_change | A_primary | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.021579 | -0.005612 | 0.048844 | 0.187622 | 0.389404 | 0.389404 |
| layer5/all/prediction_flip | A_exploratory | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.020996 | -0.002332 | 0.044905 | 0.106995 | 0.200615 | 0.240738 |
| projected/DD/margin_change | A_exploratory | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | -0.065371 | -0.124368 | 0.004334 | 0.041260 | 0.088414 | 0.106097 |
| projected/DD/prediction_flip | A_exploratory | NA_INCOMPLETE_15_UNITS | 13 | NA | NA | NA | NA | NA | NA |
| projected/PD/margin_change | A_exploratory | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.159208 | 0.106489 | 0.210465 | 0.000183 | 0.000916 | 0.001099 |
| projected/PD/prediction_flip | A_exploratory | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.001245 | -0.025726 | 0.028493 | 1.000000 | 1.000000 | 1.000000 |
| projected/all/margin_change | A_exploratory | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.107979 | 0.074632 | 0.141704 | 0.000061 | 0.000916 | 0.001099 |
| projected/all/prediction_flip | A_exploratory | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.006835 | -0.014879 | 0.027602 | 0.454285 | 0.524175 | 0.629010 |
| final_wrist/DD_minus_PD | A_group | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.001658 | -0.001410 | 0.004710 | 0.330261 | 0.330261 | 0.660522 |
| final_wrist/improved_minus_worsened | A_group | NA_INCOMPLETE_15_UNITS | 14 | NA | NA | NA | NA | NA | NA |
| layer5/DD_minus_PD | A_group | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.008108 | 0.005299 | 0.011131 | 0.000061 | 0.000183 | 0.000366 |
| layer5/improved_minus_worsened | A_group | NA_INCOMPLETE_15_UNITS | 14 | NA | NA | NA | NA | NA | NA |
| projected/DD_minus_PD | A_group | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | -0.004291 | -0.009194 | 0.000462 | 0.151428 | 0.227142 | 0.454285 |
| projected/improved_minus_worsened | A_group | NA_INCOMPLETE_15_UNITS | 14 | NA | NA | NA | NA | NA | NA |


## Coverage interaction family

| test | family | status | units | mean_delta | median_delta | split_SD | wins | ties | losses | ci_low | ci_high | dz | rank_biserial | wilcoxon_p | bh_q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| error_interaction_PD_minus_DD | B_interaction | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.439291 | 0.455051 | 0.237837 | 15 | 0 | 0 | 0.326885 | 0.557102 | 1.847024 | 1.000000 | 0.000061 | 0.000183 |
| true_margin_interaction_DD_minus_PD | B_interaction | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.322831 | 0.350554 | 0.160293 | 14 | 0 | 1 | 0.242876 | 0.401089 | 2.013998 | 0.983333 | 0.000122 | 0.000183 |
| absolute_margin_interaction_DD_minus_PD | B_interaction | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.038530 | 0.033254 | 0.053761 | 13 | 0 | 2 | 0.011554 | 0.063949 | 0.716677 | 0.683333 | 0.018066 | 0.018066 |


Historical outer performance exposure and past whole-table/whole-array reading deviations remain permanently disclosed. This round does not restore an untouched outer test. No new outer performance was read, no outer loader was built, and no outer signal was supplied to a target model. Subjects rotate roles across development splits: the union of legitimate validation queries spans 390 people, but every model is restricted to its own validation allowlist. Full metadata rows are parsed for filtering and window-position/length metadata is used for historical numerical batch layout; hashing reads file bytes. These are application-level controls, not OS-enforced physical isolation. The 15 splits share patients and are repeated development evidence, never 15 independent cohorts.
