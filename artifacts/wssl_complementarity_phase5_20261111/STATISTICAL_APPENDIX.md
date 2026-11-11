# Phase5 完整统计附录

本轮属于同一390人、15个重叠development splits的重复开发。正式指标先同split平均三个seed的指标；三seed概率均值仅用于错误分组。split bootstrap、seed SD、subject-cluster CI均不能充当独立外部泛化证据。cluster bootstrap按PD276/DD114分层，10,000次，同一人的全部appearance/seed共享采样次数，仅固定已拟合模型条件下的不确定性；不覆盖重训练和多轮开发选择偏差。不同条件/估计对象的CI是否跨0，不构成它们之间差异的检验。

新BH族：A_primary4、A_correlations36、A_coverage3、B_mechanism4。历史六指标p/q按原族逐字复制，不用本轮新族覆盖。relative是同一确定reference评分，formal identity表中按baseline seed重复用于配对，不是3个新训练seed；其seed SD为NA。

## predeclared_tests.csv

| test | family | status | units | mean_delta | median_delta | split_delta_sd | wins | ties | losses | ci_low | ci_high | cohen_dz | rank_biserial | wilcoxon_p | bh_q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| relative_minus_WSSL_PD_correct_rate | A_primary | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | -0.106607 | -0.100457 | 0.072496 | 1 | 0 | 14 | -0.138551 | -0.067916 | -1.470507 | -0.866667 | 0.003137 | 0.004182 |
| relative_minus_WSSL_DD_correct_rate | A_primary | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.019001 | 0.043011 | 0.070945 | 9 | 0 | 6 | -0.016846 | 0.052139 | 0.267831 | 0.291667 | 0.319767 | 0.319767 |
| relative_minus_WSSL_BA | A_primary | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | -0.043803 | -0.036149 | 0.043625 | 3 | 0 | 12 | -0.066293 | -0.024382 | -1.004075 | -0.883333 | 0.001160 | 0.002319 |
| conditional_relative_AUC_on_WSSL_wrong_minus_half | A_primary | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | -0.083294 | -0.091142 | 0.061125 | 1 | 0 | 14 | -0.113138 | -0.053701 | -1.362684 | -0.950000 | 0.000305 | 0.001221 |
| DD/d_pd/dd_aligned/pearson | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | -0.769640 | -0.826152 | 0.137418 | 0 | 0 | 15 | -0.828998 | -0.697749 | -5.600718 | -1.000000 | 0.000061 | 0.000085 |
| DD/d_pd/dd_aligned/spearman | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | -0.718442 | -0.768187 | 0.169113 | 0 | 0 | 15 | -0.790921 | -0.629963 | -4.248301 | -1.000000 | 0.000061 | 0.000085 |
| DD/d_pd/relative/pearson | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.723797 | 0.731254 | 0.092791 | 15 | 0 | 0 | 0.679504 | 0.769410 | 7.800321 | 1.000000 | 0.000061 | 0.000085 |
| DD/d_pd/relative/spearman | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.690422 | 0.682759 | 0.098202 | 15 | 0 | 0 | 0.642612 | 0.738804 | 7.030625 | 1.000000 | 0.000061 | 0.000085 |
| DD/dd_aligned/relative/pearson | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | -0.141863 | -0.194364 | 0.234469 | 4 | 0 | 11 | -0.254169 | -0.026376 | -0.605041 | -0.633333 | 0.030151 | 0.032892 |
| DD/dd_aligned/relative/spearman | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | -0.105619 | -0.181758 | 0.219475 | 5 | 0 | 10 | -0.210304 | 0.004717 | -0.481237 | -0.533333 | 0.072998 | 0.072998 |
| DD/wssl_logit/d_pd/pearson | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.520842 | 0.527125 | 0.107978 | 15 | 0 | 0 | 0.467951 | 0.573846 | 4.823597 | 1.000000 | 0.000061 | 0.000085 |
| DD/wssl_logit/d_pd/spearman | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.503598 | 0.490842 | 0.112849 | 15 | 0 | 0 | 0.447383 | 0.558202 | 4.462587 | 1.000000 | 0.000061 | 0.000085 |
| DD/wssl_logit/dd_aligned/pearson | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | -0.177231 | -0.145394 | 0.170155 | 3 | 0 | 12 | -0.261003 | -0.095922 | -1.041587 | -0.883333 | 0.001160 | 0.001392 |
| DD/wssl_logit/dd_aligned/spearman | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | -0.164558 | -0.171227 | 0.157831 | 3 | 0 | 12 | -0.242738 | -0.088399 | -1.042619 | -0.866667 | 0.001526 | 0.001772 |
| DD/wssl_logit/relative/pearson | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.640964 | 0.649599 | 0.097689 | 15 | 0 | 0 | 0.591534 | 0.686037 | 6.561277 | 1.000000 | 0.000061 | 0.000085 |
| DD/wssl_logit/relative/spearman | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.611911 | 0.622312 | 0.111064 | 15 | 0 | 0 | 0.552898 | 0.661919 | 5.509537 | 1.000000 | 0.000061 | 0.000085 |
| PD/d_pd/dd_aligned/pearson | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | -0.687135 | -0.686987 | 0.092021 | 0 | 0 | 15 | -0.730044 | -0.640368 | -7.467121 | -1.000000 | 0.000061 | 0.000085 |
| PD/d_pd/dd_aligned/spearman | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | -0.660745 | -0.657102 | 0.088785 | 0 | 0 | 15 | -0.703288 | -0.618403 | -7.442084 | -1.000000 | 0.000061 | 0.000085 |
| PD/d_pd/relative/pearson | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.566135 | 0.602107 | 0.133834 | 15 | 0 | 0 | 0.491599 | 0.621587 | 4.230116 | 1.000000 | 0.000061 | 0.000085 |
| PD/d_pd/relative/spearman | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.522475 | 0.551277 | 0.148790 | 15 | 0 | 0 | 0.439899 | 0.584756 | 3.511484 | 1.000000 | 0.000061 | 0.000085 |
| PD/dd_aligned/relative/pearson | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.193734 | 0.218977 | 0.161150 | 14 | 0 | 1 | 0.117502 | 0.275319 | 1.202195 | 0.950000 | 0.000305 | 0.000392 |
| PD/dd_aligned/relative/spearman | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.209987 | 0.193971 | 0.138770 | 15 | 0 | 0 | 0.145638 | 0.282481 | 1.513205 | 1.000000 | 0.000061 | 0.000085 |
| PD/wssl_logit/d_pd/pearson | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.291036 | 0.322798 | 0.086992 | 15 | 0 | 0 | 0.246150 | 0.330489 | 3.345538 | 1.000000 | 0.000061 | 0.000085 |
| PD/wssl_logit/d_pd/spearman | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.269452 | 0.312611 | 0.095409 | 15 | 0 | 0 | 0.220565 | 0.312661 | 2.824168 | 1.000000 | 0.000061 | 0.000085 |
| PD/wssl_logit/dd_aligned/pearson | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.185166 | 0.190346 | 0.107481 | 13 | 0 | 2 | 0.131113 | 0.236044 | 1.722777 | 0.950000 | 0.000305 | 0.000392 |
| PD/wssl_logit/dd_aligned/spearman | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.191685 | 0.235612 | 0.107004 | 15 | 0 | 0 | 0.138486 | 0.242960 | 1.791379 | 1.000000 | 0.000061 | 0.000085 |
| PD/wssl_logit/relative/pearson | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.602742 | 0.608025 | 0.074207 | 15 | 0 | 0 | 0.567518 | 0.641048 | 8.122422 | 1.000000 | 0.000061 | 0.000085 |
| PD/wssl_logit/relative/spearman | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.599875 | 0.576196 | 0.076152 | 15 | 0 | 0 | 0.564246 | 0.639181 | 7.877386 | 1.000000 | 0.000061 | 0.000085 |
| all/d_pd/dd_aligned/pearson | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | -0.687652 | -0.695347 | 0.085218 | 0 | 0 | 15 | -0.728121 | -0.644972 | -8.069296 | -1.000000 | 0.000061 | 0.000085 |
| all/d_pd/dd_aligned/spearman | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | -0.660649 | -0.664093 | 0.090080 | 0 | 0 | 15 | -0.703243 | -0.616933 | -7.334026 | -1.000000 | 0.000061 | 0.000085 |
| all/d_pd/relative/pearson | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.660710 | 0.678081 | 0.085617 | 15 | 0 | 0 | 0.614378 | 0.697160 | 7.717004 | 1.000000 | 0.000061 | 0.000085 |
| all/d_pd/relative/spearman | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.611330 | 0.646499 | 0.109095 | 15 | 0 | 0 | 0.550600 | 0.655858 | 5.603628 | 1.000000 | 0.000061 | 0.000085 |
| all/dd_aligned/relative/pearson | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.079472 | 0.063498 | 0.139565 | 10 | 0 | 5 | 0.012852 | 0.148006 | 0.569426 | 0.566667 | 0.055359 | 0.056941 |
| all/dd_aligned/relative/spearman | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.110272 | 0.059219 | 0.111194 | 13 | 0 | 2 | 0.060028 | 0.168386 | 0.991708 | 0.933333 | 0.000427 | 0.000530 |
| all/wssl_logit/d_pd/pearson | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.443596 | 0.429736 | 0.070452 | 15 | 0 | 0 | 0.408397 | 0.478380 | 6.296405 | 1.000000 | 0.000061 | 0.000085 |
| all/wssl_logit/d_pd/spearman | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.405635 | 0.420053 | 0.065476 | 15 | 0 | 0 | 0.370866 | 0.435419 | 6.195205 | 1.000000 | 0.000061 | 0.000085 |
| all/wssl_logit/dd_aligned/pearson | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.062221 | 0.033453 | 0.100510 | 10 | 0 | 5 | 0.017369 | 0.113216 | 0.619049 | 0.583333 | 0.047913 | 0.050731 |
| all/wssl_logit/dd_aligned/spearman | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.084010 | 0.068370 | 0.094580 | 12 | 0 | 3 | 0.040862 | 0.131333 | 0.888240 | 0.800000 | 0.004272 | 0.004807 |
| all/wssl_logit/relative/pearson | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.675158 | 0.675474 | 0.069672 | 15 | 0 | 0 | 0.640517 | 0.708262 | 9.690575 | 1.000000 | 0.000061 | 0.000085 |
| all/wssl_logit/relative/spearman | A_correlations | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.664238 | 0.652737 | 0.073440 | 15 | 0 | 0 | 0.627177 | 0.698836 | 9.044643 | 1.000000 | 0.000061 | 0.000085 |
| coverage_high_minus_low_error_DD | A_coverage | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | -0.289173 | -0.300000 | 0.214161 | 1 | 1 | 13 | -0.393754 | -0.186329 | -1.350261 | -0.980952 | 0.001225 | 0.001838 |
| coverage_high_minus_low_error_PD | A_coverage | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.150119 | 0.126984 | 0.065933 | 15 | 0 | 0 | 0.120477 | 0.184243 | 2.276840 | 1.000000 | 0.000061 | 0.000183 |
| coverage_high_minus_low_error_all | A_coverage | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.049288 | 0.024691 | 0.064389 | 10 | 2 | 3 | 0.018706 | 0.081895 | 0.765473 | 0.758242 | 0.018351 | 0.018351 |
| W_standardized_validation_mean | B_mechanism | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | -0.088738 | -0.071055 | 0.432439 | 7 | 0 | 8 | -0.307968 | 0.123349 | -0.205203 | -0.183333 | 0.561401 | 0.748535 |
| R_standardized_validation_mean | B_mechanism | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.061573 | 0.039674 | 0.082477 | 12 | 0 | 3 | 0.022955 | 0.103785 | 0.746543 | 0.716667 | 0.012451 | 0.024902 |
| validation_mean_logit_shift | B_mechanism | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | -0.005711 | -0.019677 | 0.265751 | 7 | 0 | 8 | -0.135326 | 0.128281 | -0.021489 | -0.016667 | 0.977966 | 0.977966 |
| W_minus_R_absolute_contribution | B_mechanism | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.385954 | 0.403479 | 0.345673 | 13 | 0 | 2 | 0.218325 | 0.554594 | 1.116529 | 0.900000 | 0.000854 | 0.003418 |


## seed_split_stability.csv

| condition | metric | mean | split_SD | within_split_seed_SD | SD_of_three_seed_means | independent_external_validation |
| --- | --- | --- | --- | --- | --- | --- |
| adapted | accuracy | 0.751827 | 0.041485 | 0.034660 | 0.016949 | False |
| adapted | ba | 0.724402 | 0.037496 | 0.020238 | 0.009136 | False |
| adapted | auroc | 0.768908 | 0.043663 | 0.022211 | 0.006798 | False |
| adapted | macro_f1 | 0.712829 | 0.041144 | 0.026111 | 0.013482 | False |
| adapted | pd_recall | 0.790493 | 0.053317 | 0.074523 | 0.029909 | False |
| adapted | dd_recall | 0.658310 | 0.047875 | 0.079751 | 0.020035 | False |
| baseline | accuracy | 0.745604 | 0.041218 | 0.031205 | 0.008650 | False |
| baseline | ba | 0.719644 | 0.034563 | 0.021603 | 0.006714 | False |
| baseline | auroc | 0.759553 | 0.043921 | 0.021741 | 0.001812 | False |
| baseline | macro_f1 | 0.706589 | 0.039561 | 0.025605 | 0.004947 | False |
| baseline | pd_recall | 0.782344 | 0.061379 | 0.065659 | 0.025591 | False |
| baseline | dd_recall | 0.656945 | 0.058007 | 0.078542 | 0.035930 | False |
| fusion | accuracy | 0.752378 | 0.023076 | 0.022807 | 0.013244 | False |
| fusion | ba | 0.627638 | 0.037972 | 0.039350 | 0.011528 | False |
| fusion | auroc | 0.763640 | 0.046947 | 0.016965 | 0.008865 | False |
| fusion | macro_f1 | 0.631057 | 0.048251 | 0.049979 | 0.017857 | False |
| fusion | pd_recall | 0.927570 | 0.038237 | 0.042318 | 0.015531 | False |
| fusion | dd_recall | 0.327707 | 0.095001 | 0.099096 | 0.008804 | False |
| relative | accuracy | 0.675813 | 0.045439 | NA | NA | False |
| relative | ba | 0.675842 | 0.053166 | NA | NA | False |
| relative | auroc | 0.735639 | 0.049033 | NA | NA | False |
| relative | macro_f1 | 0.647599 | 0.047522 | NA | NA | False |
| relative | pd_recall | 0.675737 | 0.048086 | NA | NA | False |
| relative | dd_recall | 0.675946 | 0.088974 | NA | NA | False |


## per_seed_metrics.csv

| condition | seed | accuracy | ba | auroc | macro_f1 | pd_recall | dd_recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adapted | 42 | 0.741833 | 0.715268 | 0.762233 | 0.702972 | 0.779168 | 0.651368 |
| adapted | 43 | 0.742252 | 0.724397 | 0.768669 | 0.707322 | 0.767901 | 0.680892 |
| adapted | 44 | 0.771397 | 0.733540 | 0.775823 | 0.728193 | 0.824411 | 0.642670 |
| baseline | 42 | 0.752053 | 0.712322 | 0.760859 | 0.706468 | 0.808120 | 0.616525 |
| baseline | 43 | 0.735774 | 0.721099 | 0.757485 | 0.701704 | 0.756942 | 0.685255 |
| baseline | 44 | 0.748984 | 0.725512 | 0.760316 | 0.711596 | 0.781970 | 0.669055 |
| fusion | 42 | 0.763531 | 0.636209 | 0.766238 | 0.643669 | 0.942071 | 0.330347 |
| fusion | 43 | 0.737739 | 0.614533 | 0.753766 | 0.610623 | 0.911181 | 0.317884 |
| fusion | 44 | 0.755863 | 0.632173 | 0.770916 | 0.638878 | 0.929458 | 0.334888 |


## context_sensitivity.csv

| condition | context | accuracy | ba | auroc | macro_f1 | pd_recall | dd_recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adapted | 0 | 0.731515 | 0.713749 | 0.757971 | 0.694594 | 0.756140 | 0.671359 |
| adapted | 1 | 0.763894 | 0.730017 | 0.762607 | 0.722544 | 0.811407 | 0.648626 |
| adapted | 2 | 0.753027 | 0.711696 | 0.751343 | 0.706804 | 0.811325 | 0.612067 |
| adapted | 3 | 0.759716 | 0.743384 | 0.798122 | 0.727694 | 0.783064 | 0.703704 |
| adapted | 4 | 0.750984 | 0.723162 | 0.774498 | 0.712508 | 0.790530 | 0.655795 |
| baseline | 0 | 0.730364 | 0.701078 | 0.740979 | 0.687358 | 0.771196 | 0.630960 |
| baseline | 1 | 0.761726 | 0.723158 | 0.747637 | 0.717562 | 0.815850 | 0.630466 |
| baseline | 2 | 0.729481 | 0.713725 | 0.747901 | 0.694786 | 0.752540 | 0.674910 |
| baseline | 3 | 0.746956 | 0.735237 | 0.792101 | 0.716079 | 0.763544 | 0.706930 |
| baseline | 4 | 0.759490 | 0.725023 | 0.769148 | 0.717162 | 0.808589 | 0.641458 |
| fusion | 0 | 0.740914 | 0.611315 | 0.751537 | 0.618245 | 0.919701 | 0.302929 |
| fusion | 1 | 0.745720 | 0.610583 | 0.733459 | 0.611254 | 0.935024 | 0.286141 |
| fusion | 2 | 0.740379 | 0.585239 | 0.749938 | 0.570691 | 0.957814 | 0.212664 |
| fusion | 3 | 0.774429 | 0.670289 | 0.798159 | 0.684903 | 0.923135 | 0.417443 |
| fusion | 4 | 0.760446 | 0.660765 | 0.785106 | 0.670191 | 0.902176 | 0.419355 |
| relative | 0 | 0.674406 | 0.675864 | 0.731498 | 0.646501 | 0.672714 | 0.679014 |
| relative | 1 | 0.634900 | 0.613121 | 0.678881 | 0.596971 | 0.665309 | 0.560932 |
| relative | 2 | 0.679374 | 0.683297 | 0.725735 | 0.651825 | 0.673763 | 0.692832 |
| relative | 3 | 0.680875 | 0.688701 | 0.776111 | 0.656855 | 0.669875 | 0.707527 |
| relative | 4 | 0.709508 | 0.718226 | 0.765968 | 0.685841 | 0.697026 | 0.739427 |


## subject_cluster_bootstrap.csv

| mean_delta | ci_low | ci_high | valid_draws | invalid_draws | requested_draws | scope | comparison | metric | estimand |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -0.069791 | -0.098164 | -0.042072 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | relative-baseline | accuracy | seed-first 15split metric difference |
| -0.043803 | -0.073898 | -0.013282 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | relative-baseline | ba | seed-first 15split metric difference |
| -0.023914 | -0.056158 | 0.007531 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | relative-baseline | auroc | seed-first 15split metric difference |
| -0.058991 | -0.088302 | -0.030036 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | relative-baseline | macro_f1 | seed-first 15split metric difference |
| -0.106607 | -0.140544 | -0.073161 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | relative-baseline | pd_recall | seed-first 15split metric difference |
| 0.019001 | -0.030882 | 0.069487 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | relative-baseline | dd_recall | seed-first 15split metric difference |
| 0.006774 | -0.013125 | 0.026378 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | fusion-baseline | accuracy | seed-first 15split metric difference |
| -0.092006 | -0.116228 | -0.068428 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | fusion-baseline | ba | seed-first 15split metric difference |
| 0.004087 | -0.007039 | 0.015225 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | fusion-baseline | auroc | seed-first 15split metric difference |
| -0.075532 | -0.105667 | -0.049189 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | fusion-baseline | macro_f1 | seed-first 15split metric difference |
| 0.145226 | 0.124429 | 0.166568 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | fusion-baseline | pd_recall | seed-first 15split metric difference |
| -0.329238 | -0.372740 | -0.286301 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | fusion-baseline | dd_recall | seed-first 15split metric difference |
| 0.006224 | -0.004862 | 0.017339 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | adapted-baseline | accuracy | seed-first 15split metric difference |
| 0.004757 | -0.008245 | 0.017479 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | adapted-baseline | ba | seed-first 15split metric difference |
| 0.009355 | 0.001011 | 0.017972 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | adapted-baseline | auroc | seed-first 15split metric difference |
| 0.006240 | -0.006335 | 0.018721 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | adapted-baseline | macro_f1 | seed-first 15split metric difference |
| 0.008149 | -0.004697 | 0.020971 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | adapted-baseline | pd_recall | seed-first 15split metric difference |
| 0.001365 | -0.021161 | 0.023477 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | adapted-baseline | dd_recall | seed-first 15split metric difference |
| -0.083294 | -0.136756 | -0.020932 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | conditional relative ranking on WSSL strict wrong pairs | conditional_AUC_minus_half | 45unit mean, equivalent to seed-first 15split |


## coverage_subject_cluster_bootstrap.csv

| mean_delta | ci_low | ci_high | valid_draws | invalid_draws | requested_draws | scope | label | metric | estimand |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.049538 | -0.025269 | 0.124220 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | all | high_minus_low_error | pooled appearance/seed exposure rates; NOT 15split equal weight |
| 0.148441 | 0.065230 | 0.231629 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | PD | high_minus_low_error | pooled appearance/seed exposure rates; NOT 15split equal weight |
| -0.274209 | -0.439300 | -0.105957 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | DD | high_minus_low_error | pooled appearance/seed exposure rates; NOT 15split equal weight |


## error_group_subject_cluster_bootstrap.csv

| mean_delta | ci_low | ci_high | valid_draws | invalid_draws | requested_draws | scope | condition | error_group | label | subjects | gross_rescue_appearance_fraction | newly_broken_appearance_fraction | estimand |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -0.081410 | -0.111554 | -0.051282 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | relative | all | all | 390 | 0.080769 | 0.162179 | patient mean correctness change over 4 seed-probability-mean appearances |
| -0.000641 | -0.014744 | 0.014103 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | adapted | all | all | 390 | 0.043590 | 0.044231 | patient mean correctness change over 4 seed-probability-mean appearances |
| -0.134058 | -0.170290 | -0.098732 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | relative | all | PD | 276 | 0.056159 | 0.190217 | patient mean correctness change over 4 seed-probability-mean appearances |
| 0.001812 | -0.014515 | 0.019022 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | adapted | all | PD | 276 | 0.042572 | 0.040761 | patient mean correctness change over 4 seed-probability-mean appearances |
| 0.046053 | -0.008772 | 0.100877 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | relative | all | DD | 114 | 0.140351 | 0.094298 | patient mean correctness change over 4 seed-probability-mean appearances |
| -0.006579 | -0.032895 | 0.021930 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | adapted | all | DD | 114 | 0.046053 | 0.052632 | patient mean correctness change over 4 seed-probability-mean appearances |
| 0.154412 | 0.085614 | 0.228814 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | relative | stable-error | all | 68 | 0.216912 | 0.062500 | patient mean correctness change over 4 seed-probability-mean appearances |
| 0.062500 | 0.016129 | 0.111842 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | adapted | stable-error | all | 68 | 0.117647 | 0.055147 | patient mean correctness change over 4 seed-probability-mean appearances |
| 0.042857 | -0.046875 | 0.145833 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | relative | stable-error | PD | 35 | 0.150000 | 0.107143 | patient mean correctness change over 4 seed-probability-mean appearances |
| 0.078571 | 0.005682 | 0.160714 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | adapted | stable-error | PD | 35 | 0.150000 | 0.071429 | patient mean correctness change over 4 seed-probability-mean appearances |
| 0.272727 | 0.179682 | 0.371622 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | relative | stable-error | DD | 33 | 0.287879 | 0.015152 | patient mean correctness change over 4 seed-probability-mean appearances |
| 0.045455 | -0.006944 | 0.102941 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | adapted | stable-error | DD | 33 | 0.083333 | 0.037879 | patient mean correctness change over 4 seed-probability-mean appearances |
| -0.149642 | -0.182973 | -0.118326 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | relative | stable-correct | all | 279 | 0.029570 | 0.179211 | patient mean correctness change over 4 seed-probability-mean appearances |
| -0.017025 | -0.031136 | -0.002737 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | adapted | stable-correct | all | 279 | 0.019713 | 0.036738 | patient mean correctness change over 4 seed-probability-mean appearances |
| -0.169767 | -0.209906 | -0.131879 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | relative | stable-correct | PD | 215 | 0.027907 | 0.197674 | patient mean correctness change over 4 seed-probability-mean appearances |
| -0.015116 | -0.030134 | -0.001116 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | adapted | stable-correct | PD | 215 | 0.017442 | 0.032558 | patient mean correctness change over 4 seed-probability-mean appearances |
| -0.082031 | -0.134623 | -0.034479 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | relative | stable-correct | DD | 64 | 0.035156 | 0.117188 | patient mean correctness change over 4 seed-probability-mean appearances |
| -0.023438 | -0.059701 | 0.011719 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | adapted | stable-correct | DD | 64 | 0.027344 | 0.050781 | patient mean correctness change over 4 seed-probability-mean appearances |
| -0.011628 | -0.121433 | 0.101570 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | relative | unstable | all | 43 | 0.197674 | 0.209302 | patient mean correctness change over 4 seed-probability-mean appearances |
| 0.005814 | -0.046512 | 0.057692 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | adapted | unstable | all | 43 | 0.081395 | 0.075581 | patient mean correctness change over 4 seed-probability-mean appearances |
| -0.076923 | -0.204566 | 0.056461 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | relative | unstable | PD | 26 | 0.163462 | 0.240385 | patient mean correctness change over 4 seed-probability-mean appearances |
| 0.038462 | -0.035714 | 0.112500 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | adapted | unstable | PD | 26 | 0.105769 | 0.067308 | patient mean correctness change over 4 seed-probability-mean appearances |
| 0.088235 | -0.109375 | 0.281250 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | relative | unstable | DD | 17 | 0.250000 | 0.161765 | patient mean correctness change over 4 seed-probability-mean appearances |
| -0.044118 | -0.107143 | 0.017857 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | adapted | unstable | DD | 17 | 0.044118 | 0.088235 | patient mean correctness change over 4 seed-probability-mean appearances |


## reused_B_relative_vs_WSSL_paired.csv

| metric | reference_mean | candidate_mean | mean_delta | median_delta | wins | ties | losses | ci_low | ci_high | paired_dz | rank_biserial | wilcoxon_p | split_delta_sd | bh_q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| accuracy | 0.745604 | 0.675813 | -0.069791 | -0.071895 | 1 | 0 | 14 | -0.094014 | -0.045122 | -1.391446 | -0.933333 | 0.001463 | 0.050157 | 0.002927 |
| ba | 0.719644 | 0.675842 | -0.043803 | -0.036149 | 3 | 0 | 12 | -0.066293 | -0.024382 | -1.004075 | -0.883333 | 0.001160 | 0.043625 | 0.002927 |
| auroc | 0.759553 | 0.735639 | -0.023914 | -0.019326 | 3 | 0 | 12 | -0.037886 | -0.010899 | -0.868919 | -0.833333 | 0.002625 | 0.027522 | 0.003764 |
| macro_f1 | 0.706589 | 0.647599 | -0.058991 | -0.060769 | 1 | 0 | 14 | -0.082314 | -0.036875 | -1.264554 | -0.933333 | 0.000427 | 0.046649 | 0.002563 |
| pd_recall | 0.782344 | 0.675737 | -0.106607 | -0.100457 | 1 | 0 | 14 | -0.138551 | -0.067916 | -1.470507 | -0.866667 | 0.003137 | 0.072496 | 0.003764 |
| dd_recall | 0.656945 | 0.675946 | 0.019001 | 0.043011 | 9 | 0 | 6 | -0.016846 | 0.052139 | 0.267831 | 0.291667 | 0.319767 | 0.070945 | 0.319767 |


## reused_B_fusion_paired.csv

| metric | reference_mean | candidate_mean | mean_delta | median_delta | wins | ties | losses | ci_low | ci_high | paired_dz | rank_biserial | wilcoxon_p | split_delta_sd | bh_q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| accuracy | 0.745604 | 0.752378 | 0.006774 | 0.003175 | 8 | 0 | 7 | -0.008128 | 0.022273 | 0.217412 | 0.216667 | 0.460121 | 0.031157 | 0.460121 |
| ba | 0.719644 | 0.627638 | -0.092006 | -0.091441 | 0 | 0 | 15 | -0.110506 | -0.073277 | -2.377690 | -1.000000 | 0.000061 | 0.038696 | 0.000122 |
| auroc | 0.759553 | 0.763640 | 0.004087 | 0.008559 | 12 | 0 | 3 | -0.004272 | 0.010426 | 0.267336 | 0.500000 | 0.094604 | 0.015287 | 0.113525 |
| macro_f1 | 0.706589 | 0.631057 | -0.075532 | -0.068511 | 0 | 0 | 15 | -0.099924 | -0.052746 | -1.548362 | -1.000000 | 0.000061 | 0.048782 | 0.000122 |
| pd_recall | 0.782344 | 0.927570 | 0.145226 | 0.139640 | 15 | 0 | 0 | 0.116179 | 0.176635 | 2.341418 | 1.000000 | 0.000653 | 0.062025 | 0.000980 |
| dd_recall | 0.656945 | 0.327707 | -0.329238 | -0.322581 | 0 | 0 | 15 | -0.380548 | -0.275759 | -3.075123 | -1.000000 | 0.000061 | 0.107065 | 0.000122 |


## reused_A1_adapted_paired.csv

| metric | reference_mean | candidate_mean | mean_delta | median_delta | wins | ties | losses | ci_low | ci_high | paired_dz | rank_biserial | wilcoxon_p | split_delta_sd | bh_q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| accuracy | 0.745604 | 0.751827 | 0.006224 | 0.006349 | 10 | 0 | 5 | -0.002895 | 0.016014 | 0.317535 | 0.375000 | 0.201098 | 0.019600 | 0.301647 |
| ba | 0.719644 | 0.724402 | 0.004757 | 0.001598 | 10 | 0 | 5 | -0.000612 | 0.010149 | 0.430737 | 0.416667 | 0.168823 | 0.011044 | 0.301647 |
| auroc | 0.759553 | 0.768908 | 0.009355 | 0.010046 | 11 | 0 | 4 | 0.002889 | 0.015348 | 0.727474 | 0.700000 | 0.015076 | 0.012860 | 0.090454 |
| macro_f1 | 0.706589 | 0.712829 | 0.006240 | 0.005010 | 11 | 0 | 4 | -0.000119 | 0.013139 | 0.455661 | 0.483333 | 0.106995 | 0.013693 | 0.301647 |
| pd_recall | 0.782344 | 0.790493 | 0.008149 | -0.004566 | 6 | 1 | 8 | -0.012300 | 0.030976 | 0.183174 | 0.142857 | 0.637684 | 0.044489 | 0.637684 |
| dd_recall | 0.656945 | 0.658310 | 0.001365 | 0.000000 | 7 | 1 | 7 | -0.024469 | 0.024134 | 0.027494 | 0.190476 | 0.529955 | 0.049659 | 0.635946 |
