# Phase8C Subject-level SupCon

B已全部完成归档后才启动。单一预先固定CE+.1SupCon，tau.1；仅不同PD患者作positive，DD作negative而不强制DD跨亚型吸引。不增加projection/分类分支/augmentation/sampler；原WSSL分类侧从原seed初始化，HarNet不进入优化器；原norm和50epoch/patience12/batch8/AdamW/cosine recipe保留。不同人样本，不把活动/腕/window作为患者。此asymmetric变体区别于[原SupCon](https://arxiv.org/abs/2004.11362)，不能用结果泛化否定全部SupCon实现。

| condition | accuracy | ba | auroc | macro_f1 | pd_recall | dd_recall | nll | brier | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| subject_supcon | 0.754394 | 0.722915 | 0.767261 | 0.713233 | 0.798717 | 0.647114 | 0.592164 | 0.370729 | 0.101779 |


| condition | metric | status | units | reference_mean | candidate_mean | mean_delta | median_delta | wins | ties | losses | ci_low | ci_high | split_SD | paired_dz | rank_biserial | wilcoxon_p | primary_bh_q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| subject_supcon | accuracy | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.745604 | 0.754394 | 0.008791 | 0.009524 | 9 | 2 | 4 | -0.001853 | 0.019962 | 0.022253 | 0.395024 | 0.395604 | 0.208274 | NA |
| subject_supcon | ba | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.719644 | 0.722915 | 0.003271 | 0.003633 | 10 | 1 | 4 | -0.002036 | 0.008490 | 0.010726 | 0.304941 | 0.371429 | 0.220899 | 0.220899 |
| subject_supcon | auroc | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.759553 | 0.767261 | 0.007708 | 0.006103 | 12 | 0 | 3 | 0.001484 | 0.014181 | 0.012935 | 0.595903 | 0.633333 | 0.030151 | 0.060303 |
| subject_supcon | macro_f1 | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.706589 | 0.713233 | 0.006644 | 0.005878 | 10 | 0 | 5 | -0.001232 | 0.014840 | 0.016429 | 0.404398 | 0.366667 | 0.229309 | NA |
| subject_supcon | pd_recall | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.782344 | 0.798717 | 0.016373 | 0.013699 | 10 | 1 | 4 | -0.005973 | 0.039936 | 0.046925 | 0.348909 | 0.361905 | 0.232851 | NA |
| subject_supcon | dd_recall | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.656945 | 0.647114 | -0.009831 | -0.010753 | 5 | 1 | 9 | -0.032470 | 0.011542 | 0.045068 | -0.218132 | -0.190476 | 0.529651 | NA |
| subject_supcon | nll | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.616720 | 0.592164 | -0.024556 | -0.017472 | 4 | 0 | 11 | -0.055940 | 0.009459 | 0.066658 | -0.368388 | -0.466667 | 0.120544 | NA |
| subject_supcon | brier | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.386441 | 0.370729 | -0.015712 | -0.015907 | 2 | 0 | 13 | -0.027714 | -0.004336 | 0.023485 | -0.669028 | -0.716667 | 0.012451 | NA |
| subject_supcon | ece | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.110333 | 0.101779 | -0.008553 | 0.007580 | 8 | 0 | 7 | -0.026488 | 0.008759 | 0.035576 | -0.240425 | -0.216667 | 0.488708 | NA |


**REJECT**。DD距离/粗来源类间距离、norm/margin及anchor覆盖见CSV；粗来源类别不是可靠细粒度临床亚型，不用于新监督。即使类间距离增大也不自动意味着疾病信息增加。相关geometry的患者/seed/split依赖限制保留，末阶段统一计算cluster敏感性。
