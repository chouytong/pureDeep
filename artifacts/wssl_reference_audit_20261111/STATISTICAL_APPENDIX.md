# 全部配对统计与稳定性

15 个 development splits 共享受试者和训练池，45 次已有 seed/run 不是 45 个独立样本。主统计先在同一 split 平均三个 seed 的指标，再汇总 15 splits；不是概率 ensemble。主 bootstrap seed=20261111、10,000 次；Wilcoxon 和 BH family 见各表。受试者分层 cluster bootstrap 将同一人的全部 appearance 一起重采样，仅描述固定已拟合模型下的条件性不确定性，不覆盖重训练、开发多重选择或外部泛化。5 个 context 汇总仅为 inner-development 分组，不是 outer-test。

BH family：B_primary两项AUROC；B_all探索24项；融合六项；relative-vs-WSSL六项；probe六项；adapted重分析六项；calibration三项。历史p/q不覆盖，新的seed/family不能替换原报告结论。calibration原wins计正delta=更差，以improved_count为改善。B_all保留局部bh_q列，跨24项取exploratory_bh_q；主两项取primary_bh_q。

| condition | accuracy | ba | auroc | macro_f1 | pd_recall | dd_recall |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.745604 | 0.719644 | 0.759553 | 0.706589 | 0.782344 | 0.656945 |
| centroid_relative | 0.653972 | 0.655965 | 0.722245 | 0.627172 | 0.651253 | 0.660677 |
| common_dd | 0.301213 | 0.497296 | 0.499435 | 0.247952 | 0.025349 | 0.969242 |
| common_pd | 0.711664 | 0.535583 | 0.668489 | 0.503577 | 0.959324 | 0.111843 |
| full_dd | 0.301213 | 0.497296 | 0.499435 | 0.247952 | 0.025349 | 0.969242 |
| full_pd | 0.717422 | 0.542143 | 0.651314 | 0.510392 | 0.963841 | 0.120445 |
| full_relative | 0.712842 | 0.508748 | 0.709938 | 0.432249 | 1.000000 | 0.017496 |
| fusion | 0.752378 | 0.627638 | 0.763640 | 0.631057 | 0.927570 | 0.327707 |
| lastblock_adapted | 0.751827 | 0.724402 | 0.768908 | 0.712829 | 0.790493 | 0.658310 |
| relative | 0.675813 | 0.675842 | 0.735639 | 0.647599 | 0.675737 | 0.675946 |
| score_ssl_control | 0.754067 | 0.669348 | 0.755291 | 0.678563 | 0.873442 | 0.465254 |
| score_ssl_plus_local | 0.751731 | 0.669825 | 0.755457 | 0.678378 | 0.867127 | 0.472522 |
| strict_dd | 0.287781 | 0.491631 | 0.401855 | 0.224048 | 0.000901 | 0.982361 |
| strict_pd | 0.712934 | 0.526915 | 0.624100 | 0.482359 | 0.974676 | 0.079155 |


## B_fusion_paired.csv

| metric | reference_mean | candidate_mean | mean_delta | median_delta | wins | ties | losses | ci_low | ci_high | paired_dz | rank_biserial | wilcoxon_p | split_delta_sd | bh_q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| accuracy | 0.745604 | 0.752378 | 0.006774 | 0.003175 | 8 | 0 | 7 | -0.008128 | 0.022273 | 0.217412 | 0.216667 | 0.460121 | 0.031157 | 0.460121 |
| ba | 0.719644 | 0.627638 | -0.092006 | -0.091441 | 0 | 0 | 15 | -0.110506 | -0.073277 | -2.377690 | -1.000000 | 0.000061 | 0.038696 | 0.000122 |
| auroc | 0.759553 | 0.763640 | 0.004087 | 0.008559 | 12 | 0 | 3 | -0.004272 | 0.010426 | 0.267336 | 0.500000 | 0.094604 | 0.015287 | 0.113525 |
| macro_f1 | 0.706589 | 0.631057 | -0.075532 | -0.068511 | 0 | 0 | 15 | -0.099924 | -0.052746 | -1.548362 | -1.000000 | 0.000061 | 0.048782 | 0.000122 |
| pd_recall | 0.782344 | 0.927570 | 0.145226 | 0.139640 | 15 | 0 | 0 | 0.116179 | 0.176635 | 2.341418 | 1.000000 | 0.000653 | 0.062025 | 0.000980 |
| dd_recall | 0.656945 | 0.327707 | -0.329238 | -0.322581 | 0 | 0 | 15 | -0.380548 | -0.275759 | -3.075123 | -1.000000 | 0.000061 | 0.107065 | 0.000122 |


## B_relative_vs_WSSL_paired.csv

| metric | reference_mean | candidate_mean | mean_delta | median_delta | wins | ties | losses | ci_low | ci_high | paired_dz | rank_biserial | wilcoxon_p | split_delta_sd | bh_q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| accuracy | 0.745604 | 0.675813 | -0.069791 | -0.071895 | 1 | 0 | 14 | -0.094014 | -0.045122 | -1.391446 | -0.933333 | 0.001463 | 0.050157 | 0.002927 |
| ba | 0.719644 | 0.675842 | -0.043803 | -0.036149 | 3 | 0 | 12 | -0.066293 | -0.024382 | -1.004075 | -0.883333 | 0.001160 | 0.043625 | 0.002927 |
| auroc | 0.759553 | 0.735639 | -0.023914 | -0.019326 | 3 | 0 | 12 | -0.037886 | -0.010899 | -0.868919 | -0.833333 | 0.002625 | 0.027522 | 0.003764 |
| macro_f1 | 0.706589 | 0.647599 | -0.058991 | -0.060769 | 1 | 0 | 14 | -0.082314 | -0.036875 | -1.264554 | -0.933333 | 0.000427 | 0.046649 | 0.002563 |
| pd_recall | 0.782344 | 0.675737 | -0.106607 | -0.100457 | 1 | 0 | 14 | -0.138551 | -0.067916 | -1.470507 | -0.866667 | 0.003137 | 0.072496 | 0.003764 |
| dd_recall | 0.656945 | 0.675946 | 0.019001 | 0.043011 | 9 | 0 | 6 | -0.016846 | 0.052139 | 0.267831 | 0.291667 | 0.319767 | 0.070945 | 0.319767 |


## B_all_paired.csv

| metric | reference_mean | candidate_mean | mean_delta | median_delta | wins | ties | losses | ci_low | ci_high | paired_dz | rank_biserial | wilcoxon_p | split_delta_sd | bh_q | comparison | exploratory_bh_q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| accuracy | 0.711664 | 0.675813 | -0.035852 | -0.028846 | 3 | 0 | 12 | -0.057396 | -0.017310 | -0.889458 | -0.850000 | 0.003751 | 0.040307 | 0.003751 | relative-common_pd | 0.005627 |
| ba | 0.535583 | 0.675842 | 0.140258 | 0.145815 | 15 | 0 | 0 | 0.116284 | 0.162113 | 2.979145 | 1.000000 | 0.000061 | 0.047080 | 0.000183 | relative-common_pd | 0.000163 |
| auroc | 0.668489 | 0.735639 | 0.067150 | 0.069406 | 13 | 0 | 2 | 0.042404 | 0.088420 | 1.422405 | 0.950000 | 0.000305 | 0.047209 | 0.000610 | relative-common_pd | 0.000732 |
| macro_f1 | 0.503577 | 0.647599 | 0.144022 | 0.145723 | 15 | 0 | 0 | 0.123091 | 0.164009 | 3.422700 | 1.000000 | 0.000061 | 0.042078 | 0.000183 | relative-common_pd | 0.000163 |
| pd_recall | 0.959324 | 0.675737 | -0.283586 | -0.270270 | 0 | 0 | 15 | -0.311554 | -0.258878 | -5.329888 | -1.000000 | 0.000652 | 0.053207 | 0.000782 | relative-common_pd | 0.001043 |
| dd_recall | 0.111843 | 0.675946 | 0.564103 | 0.548387 | 15 | 0 | 0 | 0.519784 | 0.609265 | 6.128576 | 1.000000 | 0.000650 | 0.092045 | 0.000782 | relative-common_pd | 0.001043 |
| accuracy | 0.301213 | 0.675813 | 0.374600 | 0.359223 | 15 | 0 | 0 | 0.349282 | 0.399523 | 7.347066 | 1.000000 | 0.000061 | 0.050986 | 0.000073 | relative-common_dd | 0.000163 |
| ba | 0.497296 | 0.675842 | 0.178546 | 0.188288 | 15 | 0 | 0 | 0.148971 | 0.206861 | 3.038396 | 1.000000 | 0.000061 | 0.058763 | 0.000073 | relative-common_dd | 0.000163 |
| auroc | 0.499435 | 0.735639 | 0.236204 | 0.247945 | 15 | 0 | 0 | 0.207170 | 0.265064 | 3.942978 | 1.000000 | 0.000061 | 0.059905 | 0.000073 | relative-common_dd | 0.000163 |
| macro_f1 | 0.247952 | 0.647599 | 0.399646 | 0.382750 | 15 | 0 | 0 | 0.372422 | 0.426439 | 7.257037 | 1.000000 | 0.000061 | 0.055070 | 0.000073 | relative-common_dd | 0.000163 |
| pd_recall | 0.025349 | 0.675737 | 0.650389 | 0.662162 | 15 | 0 | 0 | 0.624337 | 0.676971 | 12.193030 | 1.000000 | 0.000061 | 0.053341 | 0.000073 | relative-common_dd | 0.000163 |
| dd_recall | 0.969242 | 0.675946 | -0.293296 | -0.290323 | 0 | 0 | 15 | -0.342616 | -0.245053 | -2.905449 | -1.000000 | 0.000650 | 0.100947 | 0.000650 | relative-common_dd | 0.001043 |
| accuracy | 0.653972 | 0.675813 | 0.021840 | 0.019608 | 12 | 0 | 3 | 0.000457 | 0.041319 | 0.523087 | 0.591667 | 0.043687 | 0.041753 | 0.111571 | relative-centroid_relative | 0.055184 |
| ba | 0.655965 | 0.675842 | 0.019877 | 0.026484 | 11 | 0 | 4 | -0.001171 | 0.038278 | 0.492413 | 0.550000 | 0.063721 | 0.040366 | 0.111571 | relative-centroid_relative | 0.072824 |
| auroc | 0.722245 | 0.735639 | 0.013394 | 0.007411 | 11 | 0 | 4 | -0.004754 | 0.030662 | 0.371603 | 0.466667 | 0.120544 | 0.036044 | 0.144653 | relative-centroid_relative | 0.125785 |
| macro_f1 | 0.627172 | 0.647599 | 0.020427 | 0.017323 | 12 | 0 | 3 | -0.000735 | 0.039372 | 0.499643 | 0.566667 | 0.055359 | 0.040883 | 0.111571 | relative-centroid_relative | 0.066431 |
| pd_recall | 0.651253 | 0.675737 | 0.024485 | 0.027027 | 10 | 2 | 3 | -0.002913 | 0.049180 | 0.457680 | 0.560440 | 0.074381 | 0.053498 | 0.111571 | relative-centroid_relative | 0.081143 |
| dd_recall | 0.660677 | 0.675946 | 0.015269 | 0.000000 | 6 | 6 | 3 | -0.015484 | 0.043728 | 0.251460 | 0.466667 | 0.212720 | 0.060721 | 0.212720 | relative-centroid_relative | 0.212720 |
| accuracy | 0.712842 | 0.675813 | -0.037029 | -0.047619 | 4 | 0 | 11 | -0.059043 | -0.015349 | -0.830459 | -0.783333 | 0.007586 | 0.044588 | 0.007586 | relative-full_relative | 0.010115 |
| ba | 0.508748 | 0.675842 | 0.167094 | 0.169791 | 15 | 0 | 0 | 0.140425 | 0.193033 | 3.105429 | 1.000000 | 0.000061 | 0.053807 | 0.000183 | relative-full_relative | 0.000163 |
| auroc | 0.709938 | 0.735639 | 0.025701 | 0.029207 | 13 | 0 | 2 | 0.010660 | 0.039049 | 0.879233 | 0.766667 | 0.006714 | 0.029231 | 0.007586 | relative-full_relative | 0.009478 |
| macro_f1 | 0.432249 | 0.647599 | 0.215350 | 0.222296 | 15 | 0 | 0 | 0.189595 | 0.240008 | 4.112002 | 1.000000 | 0.000061 | 0.052371 | 0.000183 | relative-full_relative | 0.000163 |
| pd_recall | 1.000000 | 0.675737 | -0.324263 | -0.310811 | 0 | 0 | 15 | -0.348464 | -0.301159 | -6.743355 | -1.000000 | 0.000652 | 0.048086 | 0.000977 | relative-full_relative | 0.001043 |
| dd_recall | 0.017496 | 0.675946 | 0.658450 | 0.645161 | 15 | 0 | 0 | 0.614193 | 0.702134 | 7.294001 | 1.000000 | 0.000650 | 0.090273 | 0.000977 | relative-full_relative | 0.001043 |


## A1_probe_paired.csv

| metric | reference_mean | candidate_mean | mean_delta | median_delta | wins | ties | losses | ci_low | ci_high | paired_dz | rank_biserial | wilcoxon_p | split_delta_sd | bh_q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| accuracy | 0.754067 | 0.751731 | -0.002336 | 0.000000 | 5 | 3 | 7 | -0.007904 | 0.002980 | -0.212399 | -0.243590 | 0.455605 | 0.010998 | 0.911211 |
| ba | 0.669348 | 0.669825 | 0.000476 | 0.001294 | 8 | 0 | 7 | -0.005497 | 0.006079 | 0.040243 | 0.100000 | 0.761536 | 0.011838 | 0.977966 |
| auroc | 0.755291 | 0.755457 | 0.000166 | 0.004359 | 8 | 0 | 7 | -0.007544 | 0.007798 | 0.010368 | 0.050000 | 0.890381 | 0.016008 | 0.977966 |
| macro_f1 | 0.678563 | 0.678378 | -0.000185 | 0.003478 | 8 | 0 | 7 | -0.006685 | 0.005924 | -0.014375 | 0.016667 | 0.977966 | 0.012841 | 0.977966 |
| pd_recall | 0.873442 | 0.867127 | -0.006315 | -0.009009 | 5 | 1 | 9 | -0.013238 | 0.000634 | -0.454038 | -0.438095 | 0.148578 | 0.013908 | 0.585585 |
| dd_recall | 0.465254 | 0.472522 | 0.007267 | 0.000000 | 7 | 3 | 5 | -0.002912 | 0.017419 | 0.351526 | 0.423077 | 0.195195 | 0.020674 | 0.585585 |


## A1_adapted_paired.csv

| metric | reference_mean | candidate_mean | mean_delta | median_delta | wins | ties | losses | ci_low | ci_high | paired_dz | rank_biserial | wilcoxon_p | split_delta_sd | bh_q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| accuracy | 0.745604 | 0.751827 | 0.006224 | 0.006349 | 10 | 0 | 5 | -0.002895 | 0.016014 | 0.317535 | 0.375000 | 0.201098 | 0.019600 | 0.301647 |
| ba | 0.719644 | 0.724402 | 0.004757 | 0.001598 | 10 | 0 | 5 | -0.000612 | 0.010149 | 0.430737 | 0.416667 | 0.168823 | 0.011044 | 0.301647 |
| auroc | 0.759553 | 0.768908 | 0.009355 | 0.010046 | 11 | 0 | 4 | 0.002889 | 0.015348 | 0.727474 | 0.700000 | 0.015076 | 0.012860 | 0.090454 |
| macro_f1 | 0.706589 | 0.712829 | 0.006240 | 0.005010 | 11 | 0 | 4 | -0.000119 | 0.013139 | 0.455661 | 0.483333 | 0.106995 | 0.013693 | 0.301647 |
| pd_recall | 0.782344 | 0.790493 | 0.008149 | -0.004566 | 6 | 1 | 8 | -0.012300 | 0.030976 | 0.183174 | 0.142857 | 0.637684 | 0.044489 | 0.637684 |
| dd_recall | 0.656945 | 0.658310 | 0.001365 | 0.000000 | 7 | 1 | 7 | -0.024469 | 0.024134 | 0.027494 | 0.190476 | 0.529955 | 0.049659 | 0.635946 |


## A2_calibration_directional.csv

| metric | reference_mean | candidate_mean | mean_delta | median_delta | wins | ties | losses | ci_low | ci_high | paired_dz | rank_biserial | wilcoxon_p | split_delta_sd | bh_q | improved_count | worsened_count | direction_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nll | 0.616720 | 0.577543 | -0.039178 | -0.020002 | 3 | 0 | 12 | -0.064443 | -0.016595 | -0.808575 | -0.783333 | 0.005371 | 0.048453 | 0.016113 | 12 | 3 | lower NLL/Brier/ECE is better; original wins column counts positive delta, not improvement |
| brier | 0.386441 | 0.383837 | -0.002604 | -0.004335 | 6 | 0 | 9 | -0.009263 | 0.004177 | -0.187740 | -0.183333 | 0.561401 | 0.013870 | 0.761536 | 9 | 6 | lower NLL/Brier/ECE is better; original wins column counts positive delta, not improvement |
| ece | 0.110333 | 0.107569 | -0.002764 | -0.012458 | 7 | 0 | 8 | -0.015040 | 0.009774 | -0.109960 | -0.100000 | 0.761536 | 0.025133 | 0.761536 | 8 | 7 | lower NLL/Brier/ECE is better; original wins column counts positive delta, not improvement |


## subject_cluster_bootstrap.csv

| comparison | metric | mean_delta | ci_low | ci_high | bootstrap_draws | PD_subjects | DD_subjects | scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| relative-baseline | ba | -0.043803 | -0.073898 | -0.013282 | 10000 | 276 | 114 | conditional fixed fitted models; all appearances resampled jointly; not retraining/selection uncertainty |
| relative-baseline | auroc | -0.023914 | -0.056158 | 0.007531 | 10000 | 276 | 114 | conditional fixed fitted models; all appearances resampled jointly; not retraining/selection uncertainty |
| fusion-baseline | ba | -0.092006 | -0.116228 | -0.068428 | 10000 | 276 | 114 | conditional fixed fitted models; all appearances resampled jointly; not retraining/selection uncertainty |
| fusion-baseline | auroc | 0.004087 | -0.007039 | 0.015225 | 10000 | 276 | 114 | conditional fixed fitted models; all appearances resampled jointly; not retraining/selection uncertainty |


## SD（deterministic B不伪造3seed）

| condition | metric | mean | split_sd | seed_mean_sd | mean_within_split_seed_sd | seed_interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | accuracy | 0.745604 | 0.041218 | 0.008650 | 0.031205 | three initialized trained classifiers |
| baseline | ba | 0.719644 | 0.034563 | 0.006714 | 0.021603 | three initialized trained classifiers |
| baseline | auroc | 0.759553 | 0.043921 | 0.001812 | 0.021741 | three initialized trained classifiers |
| baseline | macro_f1 | 0.706589 | 0.039561 | 0.004947 | 0.025605 | three initialized trained classifiers |
| baseline | pd_recall | 0.782344 | 0.061379 | 0.025591 | 0.065659 | three initialized trained classifiers |
| baseline | dd_recall | 0.656945 | 0.058007 | 0.035930 | 0.078542 | three initialized trained classifiers |
| lastblock_adapted | accuracy | 0.751827 | 0.041485 | 0.016949 | 0.034660 | three initialized trained classifiers |
| lastblock_adapted | ba | 0.724402 | 0.037496 | 0.009136 | 0.020238 | three initialized trained classifiers |
| lastblock_adapted | auroc | 0.768908 | 0.043663 | 0.006798 | 0.022211 | three initialized trained classifiers |
| lastblock_adapted | macro_f1 | 0.712829 | 0.041144 | 0.013482 | 0.026111 | three initialized trained classifiers |
| lastblock_adapted | pd_recall | 0.790493 | 0.053317 | 0.029909 | 0.074523 | three initialized trained classifiers |
| lastblock_adapted | dd_recall | 0.658310 | 0.047875 | 0.020035 | 0.079751 | three initialized trained classifiers |
| fusion | accuracy | 0.752378 | 0.023076 | 0.013244 | 0.022807 | three initialized trained classifiers |
| fusion | ba | 0.627638 | 0.037972 | 0.011528 | 0.039350 | three initialized trained classifiers |
| fusion | auroc | 0.763640 | 0.046947 | 0.008865 | 0.016965 | three initialized trained classifiers |
| fusion | macro_f1 | 0.631057 | 0.048251 | 0.017857 | 0.049979 | three initialized trained classifiers |
| fusion | pd_recall | 0.927570 | 0.038237 | 0.015531 | 0.042318 | three initialized trained classifiers |
| fusion | dd_recall | 0.327707 | 0.095001 | 0.008804 | 0.099096 | three initialized trained classifiers |
| score_ssl_control | accuracy | 0.754067 | 0.029019 | 0.008402 | 0.022998 | three initialized trained classifiers |
| score_ssl_control | ba | 0.669348 | 0.039226 | 0.013973 | 0.027812 | three initialized trained classifiers |
| score_ssl_control | auroc | 0.755291 | 0.046984 | 0.003557 | 0.022800 | three initialized trained classifiers |
| score_ssl_control | macro_f1 | 0.678563 | 0.041302 | 0.014350 | 0.029322 | three initialized trained classifiers |
| score_ssl_control | pd_recall | 0.873442 | 0.023444 | 0.000626 | 0.029645 | three initialized trained classifiers |
| score_ssl_control | dd_recall | 0.465254 | 0.070003 | 0.027452 | 0.054137 | three initialized trained classifiers |
| score_ssl_plus_local | accuracy | 0.751731 | 0.031313 | 0.008124 | 0.026954 | three initialized trained classifiers |
| score_ssl_plus_local | ba | 0.669825 | 0.039658 | 0.014146 | 0.028557 | three initialized trained classifiers |
| score_ssl_plus_local | auroc | 0.755457 | 0.039340 | 0.004585 | 0.024430 | three initialized trained classifiers |
| score_ssl_plus_local | macro_f1 | 0.678378 | 0.042036 | 0.014506 | 0.030972 | three initialized trained classifiers |
| score_ssl_plus_local | pd_recall | 0.867127 | 0.028801 | 0.001285 | 0.033382 | three initialized trained classifiers |
| score_ssl_plus_local | dd_recall | 0.472522 | 0.068362 | 0.028742 | 0.050690 | three initialized trained classifiers |
| centroid_relative | accuracy | 0.653972 | 0.035904 | NA | NA | deterministic external-cache method; seed SD not applicable |
| centroid_relative | ba | 0.655965 | 0.040537 | NA | NA | deterministic external-cache method; seed SD not applicable |
| centroid_relative | auroc | 0.722245 | 0.044258 | NA | NA | deterministic external-cache method; seed SD not applicable |
| centroid_relative | macro_f1 | 0.627172 | 0.036649 | NA | NA | deterministic external-cache method; seed SD not applicable |
| centroid_relative | pd_recall | 0.651253 | 0.034700 | NA | NA | deterministic external-cache method; seed SD not applicable |
| centroid_relative | dd_recall | 0.660677 | 0.059140 | NA | NA | deterministic external-cache method; seed SD not applicable |
| common_dd | accuracy | 0.301213 | 0.011791 | NA | NA | deterministic external-cache method; seed SD not applicable |
| common_dd | ba | 0.497296 | 0.012547 | NA | NA | deterministic external-cache method; seed SD not applicable |
| common_dd | auroc | 0.499435 | 0.045913 | NA | NA | deterministic external-cache method; seed SD not applicable |
| common_dd | macro_f1 | 0.247952 | 0.016143 | NA | NA | deterministic external-cache method; seed SD not applicable |
| common_dd | pd_recall | 0.025349 | 0.016895 | NA | NA | deterministic external-cache method; seed SD not applicable |
| common_dd | dd_recall | 0.969242 | 0.029092 | NA | NA | deterministic external-cache method; seed SD not applicable |
| common_pd | accuracy | 0.711664 | 0.026256 | NA | NA | deterministic external-cache method; seed SD not applicable |
| common_pd | ba | 0.535583 | 0.027037 | NA | NA | deterministic external-cache method; seed SD not applicable |
| common_pd | auroc | 0.668489 | 0.042911 | NA | NA | deterministic external-cache method; seed SD not applicable |
| common_pd | macro_f1 | 0.503577 | 0.039666 | NA | NA | deterministic external-cache method; seed SD not applicable |
| common_pd | pd_recall | 0.959324 | 0.031093 | NA | NA | deterministic external-cache method; seed SD not applicable |
| common_pd | dd_recall | 0.111843 | 0.048719 | NA | NA | deterministic external-cache method; seed SD not applicable |
| full_dd | accuracy | 0.301213 | 0.011791 | NA | NA | deterministic external-cache method; seed SD not applicable |
| full_dd | ba | 0.497296 | 0.012547 | NA | NA | deterministic external-cache method; seed SD not applicable |
| full_dd | auroc | 0.499435 | 0.045913 | NA | NA | deterministic external-cache method; seed SD not applicable |
| full_dd | macro_f1 | 0.247952 | 0.016143 | NA | NA | deterministic external-cache method; seed SD not applicable |
| full_dd | pd_recall | 0.025349 | 0.016895 | NA | NA | deterministic external-cache method; seed SD not applicable |
| full_dd | dd_recall | 0.969242 | 0.029092 | NA | NA | deterministic external-cache method; seed SD not applicable |
| full_pd | accuracy | 0.717422 | 0.019868 | NA | NA | deterministic external-cache method; seed SD not applicable |
| full_pd | ba | 0.542143 | 0.023508 | NA | NA | deterministic external-cache method; seed SD not applicable |
| full_pd | auroc | 0.651314 | 0.039181 | NA | NA | deterministic external-cache method; seed SD not applicable |
| full_pd | macro_f1 | 0.510392 | 0.041855 | NA | NA | deterministic external-cache method; seed SD not applicable |
| full_pd | pd_recall | 0.963841 | 0.034565 | NA | NA | deterministic external-cache method; seed SD not applicable |
| full_pd | dd_recall | 0.120445 | 0.063002 | NA | NA | deterministic external-cache method; seed SD not applicable |
| full_relative | accuracy | 0.712842 | 0.008289 | NA | NA | deterministic external-cache method; seed SD not applicable |
| full_relative | ba | 0.508748 | 0.013540 | NA | NA | deterministic external-cache method; seed SD not applicable |
| full_relative | auroc | 0.709938 | 0.042822 | NA | NA | deterministic external-cache method; seed SD not applicable |
| full_relative | macro_f1 | 0.432249 | 0.026956 | NA | NA | deterministic external-cache method; seed SD not applicable |
| full_relative | pd_recall | 1.000000 | 0.000000 | NA | NA | deterministic external-cache method; seed SD not applicable |
| full_relative | dd_recall | 0.017496 | 0.027079 | NA | NA | deterministic external-cache method; seed SD not applicable |
| relative | accuracy | 0.675813 | 0.045439 | NA | NA | deterministic external-cache method; seed SD not applicable |
| relative | ba | 0.675842 | 0.053166 | NA | NA | deterministic external-cache method; seed SD not applicable |
| relative | auroc | 0.735639 | 0.049033 | NA | NA | deterministic external-cache method; seed SD not applicable |
| relative | macro_f1 | 0.647599 | 0.047522 | NA | NA | deterministic external-cache method; seed SD not applicable |
| relative | pd_recall | 0.675737 | 0.048086 | NA | NA | deterministic external-cache method; seed SD not applicable |
| relative | dd_recall | 0.675946 | 0.088974 | NA | NA | deterministic external-cache method; seed SD not applicable |
| strict_dd | accuracy | 0.287781 | 0.008112 | NA | NA | deterministic external-cache method; seed SD not applicable |
| strict_dd | ba | 0.491631 | 0.010425 | NA | NA | deterministic external-cache method; seed SD not applicable |
| strict_dd | auroc | 0.401855 | 0.044416 | NA | NA | deterministic external-cache method; seed SD not applicable |
| strict_dd | macro_f1 | 0.224048 | 0.005965 | NA | NA | deterministic external-cache method; seed SD not applicable |
| strict_dd | pd_recall | 0.000901 | 0.003489 | NA | NA | deterministic external-cache method; seed SD not applicable |
| strict_dd | dd_recall | 0.982361 | 0.021230 | NA | NA | deterministic external-cache method; seed SD not applicable |
| strict_pd | accuracy | 0.712934 | 0.017020 | NA | NA | deterministic external-cache method; seed SD not applicable |
| strict_pd | ba | 0.526915 | 0.016550 | NA | NA | deterministic external-cache method; seed SD not applicable |
| strict_pd | auroc | 0.624100 | 0.036822 | NA | NA | deterministic external-cache method; seed SD not applicable |
| strict_pd | macro_f1 | 0.482359 | 0.028260 | NA | NA | deterministic external-cache method; seed SD not applicable |
| strict_pd | pd_recall | 0.974676 | 0.020408 | NA | NA | deterministic external-cache method; seed SD not applicable |
| strict_pd | dd_recall | 0.079155 | 0.032973 | NA | NA | deterministic external-cache method; seed SD not applicable |


## 五context敏感性

| condition | context | accuracy | ba | auroc | macro_f1 | pd_recall | dd_recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0 | 0.730364 | 0.701078 | 0.740979 | 0.687358 | 0.771196 | 0.630960 |
| baseline | 1 | 0.761726 | 0.723158 | 0.747637 | 0.717562 | 0.815850 | 0.630466 |
| baseline | 2 | 0.729481 | 0.713725 | 0.747901 | 0.694786 | 0.752540 | 0.674910 |
| baseline | 3 | 0.746956 | 0.735237 | 0.792101 | 0.716079 | 0.763544 | 0.706930 |
| baseline | 4 | 0.759490 | 0.725023 | 0.769148 | 0.717162 | 0.808589 | 0.641458 |
| lastblock_adapted | 0 | 0.731515 | 0.713749 | 0.757971 | 0.694594 | 0.756140 | 0.671359 |
| lastblock_adapted | 1 | 0.763894 | 0.730017 | 0.762607 | 0.722544 | 0.811407 | 0.648626 |
| lastblock_adapted | 2 | 0.753027 | 0.711696 | 0.751343 | 0.706804 | 0.811325 | 0.612067 |
| lastblock_adapted | 3 | 0.759716 | 0.743384 | 0.798122 | 0.727694 | 0.783064 | 0.703704 |
| lastblock_adapted | 4 | 0.750984 | 0.723162 | 0.774498 | 0.712508 | 0.790530 | 0.655795 |
| fusion | 0 | 0.740914 | 0.611315 | 0.751537 | 0.618245 | 0.919701 | 0.302929 |
| fusion | 1 | 0.745720 | 0.610583 | 0.733459 | 0.611254 | 0.935024 | 0.286141 |
| fusion | 2 | 0.740379 | 0.585239 | 0.749938 | 0.570691 | 0.957814 | 0.212664 |
| fusion | 3 | 0.774429 | 0.670289 | 0.798159 | 0.684903 | 0.923135 | 0.417443 |
| fusion | 4 | 0.760446 | 0.660765 | 0.785106 | 0.670191 | 0.902176 | 0.419355 |
| score_ssl_control | 0 | 0.745208 | 0.653359 | 0.745671 | 0.662779 | 0.872681 | 0.434038 |
| score_ssl_control | 1 | 0.758460 | 0.666784 | 0.743770 | 0.678507 | 0.886853 | 0.446714 |
| score_ssl_control | 2 | 0.753036 | 0.672668 | 0.731747 | 0.681739 | 0.865646 | 0.479689 |
| score_ssl_control | 3 | 0.753141 | 0.669245 | 0.786263 | 0.676602 | 0.873257 | 0.465233 |
| score_ssl_control | 4 | 0.760487 | 0.684685 | 0.769005 | 0.693188 | 0.868773 | 0.500597 |
| score_ssl_plus_local | 0 | 0.742117 | 0.656759 | 0.744393 | 0.664685 | 0.860587 | 0.452931 |
| score_ssl_plus_local | 1 | 0.759518 | 0.670679 | 0.756750 | 0.680618 | 0.883891 | 0.457467 |
| score_ssl_plus_local | 2 | 0.744531 | 0.663389 | 0.727226 | 0.672158 | 0.858201 | 0.468578 |
| score_ssl_plus_local | 3 | 0.756234 | 0.675621 | 0.787678 | 0.683659 | 0.871673 | 0.479570 |
| score_ssl_plus_local | 4 | 0.756254 | 0.682674 | 0.761238 | 0.690773 | 0.861286 | 0.504062 |
| centroid_relative | 0 | 0.658285 | 0.657965 | 0.717018 | 0.629770 | 0.659139 | 0.656791 |
| centroid_relative | 1 | 0.637888 | 0.631525 | 0.704473 | 0.607831 | 0.646921 | 0.616129 |
| centroid_relative | 2 | 0.670067 | 0.670181 | 0.715626 | 0.642124 | 0.669752 | 0.670609 |
| centroid_relative | 3 | 0.639112 | 0.646382 | 0.742747 | 0.615077 | 0.628965 | 0.663799 |
| centroid_relative | 4 | 0.664509 | 0.673772 | 0.731360 | 0.641056 | 0.651487 | 0.696057 |
| common_dd | 0 | 0.296682 | 0.491254 | 0.486591 | 0.245879 | 0.027336 | 0.955172 |
| common_dd | 1 | 0.304475 | 0.502643 | 0.472644 | 0.251506 | 0.027150 | 0.978136 |
| common_dd | 2 | 0.304383 | 0.499488 | 0.501447 | 0.253380 | 0.031593 | 0.967384 |
| common_dd | 3 | 0.300262 | 0.494984 | 0.508726 | 0.245928 | 0.022584 | 0.967384 |
| common_dd | 4 | 0.300262 | 0.498108 | 0.527765 | 0.243069 | 0.018080 | 0.978136 |
| common_pd | 0 | 0.706773 | 0.524267 | 0.675520 | 0.487109 | 0.959213 | 0.089321 |
| common_pd | 1 | 0.717900 | 0.539308 | 0.647250 | 0.505992 | 0.968222 | 0.110394 |
| common_pd | 2 | 0.714817 | 0.540062 | 0.640824 | 0.511617 | 0.959336 | 0.120789 |
| common_pd | 3 | 0.709385 | 0.530876 | 0.702118 | 0.494298 | 0.963902 | 0.097849 |
| common_pd | 4 | 0.709447 | 0.543403 | 0.676733 | 0.518868 | 0.945946 | 0.140860 |
| full_dd | 0 | 0.296682 | 0.491254 | 0.486591 | 0.245879 | 0.027336 | 0.955172 |
| full_dd | 1 | 0.304475 | 0.502643 | 0.472644 | 0.251506 | 0.027150 | 0.978136 |
| full_dd | 2 | 0.304383 | 0.499488 | 0.501447 | 0.253380 | 0.031593 | 0.967384 |
| full_dd | 3 | 0.300262 | 0.494984 | 0.508726 | 0.245928 | 0.022584 | 0.967384 |
| full_dd | 4 | 0.300262 | 0.498108 | 0.527765 | 0.243069 | 0.018080 | 0.978136 |
| full_pd | 0 | 0.719656 | 0.536938 | 0.654423 | 0.496870 | 0.972726 | 0.101149 |
| full_pd | 1 | 0.717900 | 0.536005 | 0.641590 | 0.498854 | 0.972726 | 0.099283 |
| full_pd | 2 | 0.714787 | 0.539704 | 0.631343 | 0.508453 | 0.959336 | 0.120072 |
| full_pd | 3 | 0.715796 | 0.544783 | 0.672062 | 0.518215 | 0.959459 | 0.130108 |
| full_pd | 4 | 0.718971 | 0.553284 | 0.657153 | 0.529566 | 0.954955 | 0.151613 |
| full_relative | 0 | 0.712997 | 0.505747 | 0.699694 | 0.427018 | 1.000000 | 0.011494 |
| full_relative | 1 | 0.711582 | 0.505556 | 0.666179 | 0.426180 | 1.000000 | 0.011111 |
| full_relative | 2 | 0.708346 | 0.500000 | 0.712388 | 0.414637 | 1.000000 | 0.000000 |
| full_relative | 3 | 0.712498 | 0.510932 | 0.730895 | 0.436596 | 1.000000 | 0.021864 |
| full_relative | 4 | 0.718786 | 0.521505 | 0.740535 | 0.456814 | 1.000000 | 0.043011 |
| relative | 0 | 0.674406 | 0.675864 | 0.731498 | 0.646501 | 0.672714 | 0.679014 |
| relative | 1 | 0.634900 | 0.613121 | 0.678881 | 0.596971 | 0.665309 | 0.560932 |
| relative | 2 | 0.679374 | 0.683297 | 0.725735 | 0.651825 | 0.673763 | 0.692832 |
| relative | 3 | 0.680875 | 0.688701 | 0.776111 | 0.656855 | 0.669875 | 0.707527 |
| relative | 4 | 0.709508 | 0.718226 | 0.765968 | 0.685841 | 0.697026 | 0.739427 |
| strict_dd | 0 | 0.287003 | 0.494253 | 0.377891 | 0.222963 | 0.000000 | 0.988506 |
| strict_dd | 1 | 0.282069 | 0.483513 | 0.384108 | 0.219969 | 0.000000 | 0.967025 |
| strict_dd | 2 | 0.288479 | 0.494624 | 0.425000 | 0.223889 | 0.000000 | 0.989247 |
| strict_dd | 3 | 0.290677 | 0.494444 | 0.411804 | 0.225193 | 0.000000 | 0.988889 |
| strict_dd | 4 | 0.290677 | 0.491320 | 0.410474 | 0.228226 | 0.004505 | 0.978136 |
| strict_pd | 0 | 0.719594 | 0.533486 | 0.634462 | 0.492994 | 0.977292 | 0.089680 |
| strict_pd | 1 | 0.721137 | 0.538288 | 0.619293 | 0.500965 | 0.977292 | 0.099283 |
| strict_pd | 2 | 0.714756 | 0.527028 | 0.608317 | 0.482018 | 0.977354 | 0.076703 |
| strict_pd | 3 | 0.709385 | 0.521355 | 0.631385 | 0.471151 | 0.977477 | 0.065233 |
| strict_pd | 4 | 0.699800 | 0.514419 | 0.627041 | 0.464668 | 0.963964 | 0.064875 |


## 各seed均值

| source | seed | accuracy | ba | auroc | macro_f1 | pd_recall | dd_recall | condition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_metrics.csv | 42 | 0.752053 | 0.712322 | 0.760859 | 0.706468 | 0.808120 | 0.616525 | NA |
| baseline_metrics.csv | 43 | 0.735774 | 0.721099 | 0.757485 | 0.701704 | 0.756942 | 0.685255 | NA |
| baseline_metrics.csv | 44 | 0.748984 | 0.725512 | 0.760316 | 0.711596 | 0.781970 | 0.669055 | NA |
| B_fusion_metrics.csv | 42 | 0.763531 | 0.636209 | 0.766238 | 0.643669 | 0.942071 | 0.330347 | NA |
| B_fusion_metrics.csv | 43 | 0.737739 | 0.614533 | 0.753766 | 0.610623 | 0.911181 | 0.317884 | NA |
| B_fusion_metrics.csv | 44 | 0.755863 | 0.632173 | 0.770916 | 0.638878 | 0.929458 | 0.334888 | NA |
| A1_adapted_metrics.csv | 42 | 0.741833 | 0.715268 | 0.762233 | 0.702972 | 0.779168 | 0.651368 | NA |
| A1_adapted_metrics.csv | 43 | 0.742252 | 0.724397 | 0.768669 | 0.707322 | 0.767901 | 0.680892 | NA |
| A1_adapted_metrics.csv | 44 | 0.771397 | 0.733540 | 0.775823 | 0.728193 | 0.824411 | 0.642670 | NA |
| A1_probe_metrics.csv | 42 | 0.756454 | 0.673650 | 0.753945 | 0.682253 | 0.873170 | 0.474129 | score_ssl_control |
| A1_probe_metrics.csv | 43 | 0.744729 | 0.653730 | 0.752603 | 0.662729 | 0.872998 | 0.434463 | score_ssl_control |
| A1_probe_metrics.csv | 44 | 0.761016 | 0.680664 | 0.759325 | 0.690707 | 0.874158 | 0.487171 | score_ssl_control |
| A1_probe_metrics.csv | 42 | 0.748115 | 0.664493 | 0.750169 | 0.671997 | 0.865963 | 0.463023 | score_ssl_plus_local |
| A1_probe_metrics.csv | 43 | 0.746042 | 0.659119 | 0.757887 | 0.668157 | 0.868505 | 0.449732 | score_ssl_plus_local |
| A1_probe_metrics.csv | 44 | 0.761035 | 0.685862 | 0.758316 | 0.694981 | 0.866913 | 0.504810 | score_ssl_plus_local |
