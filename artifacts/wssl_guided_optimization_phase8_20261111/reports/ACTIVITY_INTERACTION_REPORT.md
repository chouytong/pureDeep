# Phase8B 活动token内容交互

现有softmax attention已产生权重竞争，STR保留11活动结构，但没有pooling前的content-dependent token-to-token mixing。本候选不是在已有Transformer后重复堆叠；也不声称全球方法创新。固定258→64,4head,1层,FFN128,dropout.1,零输出residual；所有旧参数冻结。smoke通过后45run完成。

| condition | accuracy | ba | auroc | macro_f1 | pd_recall | dd_recall | nll | brier | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| activity_interaction | 0.769251 | 0.725805 | 0.766245 | 0.722707 | 0.830861 | 0.620749 | 0.669741 | 0.372104 | 0.124756 |


| condition | metric | status | units | reference_mean | candidate_mean | mean_delta | median_delta | wins | ties | losses | ci_low | ci_high | split_SD | paired_dz | rank_biserial | wilcoxon_p | primary_bh_q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| activity_interaction | accuracy | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.745604 | 0.769251 | 0.023648 | 0.028571 | 14 | 1 | 0 | 0.016668 | 0.030396 | 0.013880 | 1.703740 | 1.000000 | 0.000979 | NA |
| activity_interaction | ba | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.719644 | 0.725805 | 0.006161 | 0.008258 | 11 | 0 | 4 | 0.000668 | 0.011347 | 0.010810 | 0.569887 | 0.533333 | 0.072998 | 0.072998 |
| activity_interaction | auroc | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.759553 | 0.766245 | 0.006691 | 0.006545 | 12 | 0 | 3 | 0.002925 | 0.010981 | 0.008325 | 0.803746 | 0.750000 | 0.008362 | 0.016724 |
| activity_interaction | macro_f1 | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.706589 | 0.722707 | 0.016118 | 0.016395 | 14 | 0 | 1 | 0.010853 | 0.021442 | 0.010795 | 1.493077 | 0.983333 | 0.000122 | NA |
| activity_interaction | pd_recall | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.782344 | 0.830861 | 0.048517 | 0.041096 | 13 | 0 | 2 | 0.031457 | 0.065392 | 0.034504 | 1.406139 | 0.925000 | 0.001613 | NA |
| activity_interaction | dd_recall | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.656945 | 0.620749 | -0.036196 | -0.043011 | 4 | 0 | 11 | -0.058973 | -0.014365 | 0.045677 | -0.792428 | -0.700000 | 0.017036 | NA |
| activity_interaction | nll | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.616720 | 0.669741 | 0.053020 | -0.001015 | 7 | 0 | 8 | -0.000023 | 0.118597 | 0.121662 | 0.435803 | 0.216667 | 0.488708 | NA |
| activity_interaction | brier | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.386441 | 0.372104 | -0.014337 | -0.015220 | 4 | 0 | 11 | -0.022983 | -0.005452 | 0.017955 | -0.798520 | -0.750000 | 0.008362 | NA |
| activity_interaction | ece | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.110333 | 0.124756 | 0.014423 | 0.015844 | 13 | 0 | 2 | 0.001293 | 0.026999 | 0.025705 | 0.561107 | 0.666667 | 0.021545 | NA |


**REJECT**。所有epoch/budget/梯度/冻结/provenance记录保留。inference attention-off是模型扰动敏感性，不是重训消融或另一个保留候选；attention权重不是临床重要性。原监督encoder的train表示in-sample，验证重复开发选择偏差仍在；不作临床独立泛化解释。B已完成并归档，之后才允许C。
