# Phase9 Level2：数值修复与关键模型复核

**数值正确性PASS；记录独立性PASS；数据与训练来源PASS；PERFORMANCE_PROMOTION=NOT_ESTABLISHED。**

旧反例约0.008199391g已复现。新生产solver同目标float64逐轴证书和独立性检查通过，8580腕/25740轴全部重建、稀疏回读核验，15份train-only normalization完成，WSSL45及STR45正式训练按原recipe完成。HarNet冻结且输入/特征不变，任务分类侧重新初始化训练。

| condition | accuracy | ba | auroc | macro_f1 | pd_recall | dd_recall | nll | brier | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| STR-LEGACY-L1 | 0.730718 | 0.697189 | 0.717631 | 0.686068 | 0.777864 | 0.616515 | 0.649321 | 0.415655 | 0.125107 |
| WSSL-LEGACY-L1 | 0.745604 | 0.719644 | 0.759553 | 0.706589 | 0.782344 | 0.656945 | 0.61672 | 0.386441 | 0.110333 |
| STR-CERTIFIED-L1 | 0.717855 | 0.686585 | 0.712168 | 0.673253 | 0.761759 | 0.61141 | 0.685323 | 0.431873 | 0.134591 |
| WSSL-CERTIFIED-L1 | 0.741534 | 0.7189 | 0.759285 | 0.703561 | 0.773894 | 0.663905 | 0.623037 | 0.390936 | 0.113658 |


| condition | metric | status | units | reference_mean | candidate_mean | mean_delta | median_delta | wins | ties | losses | ci_low | ci_high | split_SD | paired_dz | rank_biserial | wilcoxon_p | family | BH_q | better_direction | improved_splits | worsened_splits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1_WSSL_new-old | ba | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.719644 | 0.7189 | -0.000744774 | 0 | 7 | 1 | 7 | -0.00677689 | 0.00481179 | 0.0118807 | -0.0626877 | -0.0285714 | 0.924978 | C1_primary_2 | 0.924978 | higher | 7 | 7 |
| C1_WSSL_new-old | auroc | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.759553 | 0.759285 | -0.000268235 | 0.00319675 | 9 | 0 | 6 | -0.00816897 | 0.0063136 | 0.0146731 | -0.0182807 | 0.116667 | 0.719727 | C1_primary_2 | 0.924978 | higher | 9 | 6 |
| C2_STR_new-old | ba | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.697189 | 0.686585 | -0.0106048 | -0.0103501 | 3 | 0 | 12 | -0.0183759 | -0.00258507 | 0.016159 | -0.656276 | -0.633333 | 0.0301514 | C2_primary_4 | 0.0402018 | higher | 3 | 12 |
| C2_STR_new-old | auroc | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.717631 | 0.712168 | -0.00546254 | -0.0106545 | 5 | 0 | 10 | -0.0133413 | 0.00292126 | 0.0165912 | -0.329244 | -0.333333 | 0.276855 | C2_primary_4 | 0.276855 | higher | 5 | 10 |
| C2_WSSLnew-STRnew | ba | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.686585 | 0.7189 | 0.032315 | 0.0403952 | 13 | 0 | 2 | 0.0184308 | 0.045738 | 0.0280451 | 1.15225 | 0.883333 | 0.00115967 | C2_primary_4 | 0.00231934 | higher | 13 | 2 |
| C2_WSSLnew-STRnew | auroc | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.712168 | 0.759285 | 0.0471167 | 0.0499855 | 15 | 0 | 0 | 0.0322943 | 0.0622572 | 0.0304713 | 1.54626 | 1 | 6.10352e-05 | C2_primary_4 | 0.000244141 | higher | 15 | 0 |
| DID_exploratory | ba | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0 | 0.00986002 | 0.00986002 | 0.00943683 | 9 | 0 | 6 | -0.00154628 | 0.021707 | 0.0238791 | 0.412913 | 0.4 | 0.187622 | DID_primary_2 | 0.25238 | higher | 9 | 6 |
| DID_exploratory | auroc | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0 | 0.00519431 | 0.00519431 | 0.00857309 | 9 | 0 | 6 | -0.002483 | 0.0129675 | 0.015692 | 0.331017 | 0.35 | 0.25238 | DID_primary_2 | 0.25238 | higher | 9 | 6 |


| check | passed |
| --- | --- |
| BA_positive | False |
| BA_wins_at_least10 | False |
| BA_CI_low_positive | False |
| AUROC_non_decrease | False |
| MacroF1_non_decrease | False |
| PD_recall_drop_at_most01 | True |
| DD_recall_drop_at_most01 | True |
| correctness_and_provenance | True |
| stability_at_most1_25 | True |


BA/AUC seed SD ratios: {"ba": 0.8584356927906684, "auroc": 0.8830001088191038}。未通过全部预登记gate即不能宣布优于历史WSSL；这与数值修复成功是两项判定。完整9指标、mean/median差、15split wins/ties/losses、paired CI、Wilcoxon/效应量/BH、subject-cluster CI及seed稳定性见对应CSV与SSL_INCREMENT_REVALIDATION_REPORT。

同一Certified L1下WSSL−STR：ΔBA=0.032315，paired95%CI[0.018431,0.045738]，13/15 split提高，BA的BH4 q=0.00231934；ΔAUROC=0.047117。共享患者cluster的BA95%CI[0.005118,0.059187]。结果支持这一重复development设置下的SSL增量。这不是外部或临床有效性证据。

归一化Acc变化/Gyro一致性见NORMALIZATION_PROVENANCE_REPORT，设备组recall及不确定性见DEVICE_STRATIFIED_ANALYSIS；设备差异只是描述关联。CPU预处理与缓存GPUforward分开计时，在线完整HarNet端到端未测。

DD为正类，pDD>.5，恰好tie判PD。AUROC是两概率列的macro AUROC；NLL为未加权proper log loss，新模型/WSSL历史用原logits float64 logsumexp，STR历史只有归档概率所以用−log(p_true)，不clip，有限精度差异如实保留。Brier为两类平方误差之和，ECE是10等宽top-confidence分箱。balanced CE仅训练使用，绝不把加权CE冒充NLL。沿用历史paired函数的wins/ties/losses表示差值正/零/负，损失指标越低越好，最终CSV另提供better_direction、improved_splits、worsened_splits以免误读。

Acc单位g、通道和原文件可核验；是否含完整重力向量、SDK导出前变换、左右腕物理轴/绝对同步仍UNKNOWN。本次只修复固定索引L1目标，不声称物理重力去除，不改变HarNet输入。

重复的15个development split共享390人，每split先平均3个seed的单模型指标，再对15split等权；不是概率ensemble，也不是390人的独立外部验证。历史outer接触与多轮development的选择偏差保留，本次PASS不能消除。subject-cluster条件于固定已训练模型，所有同一人的重复appearance/seed共用同一个权重。

旧模型/数据/报告/当前指针保留。研究基线与性能提升身份区分，不因legacy分数高恢复耦合，也不把分差说成旧指标虚高多少。公开代码、协议、安全聚合和正负结果；个体数据/证书/预测/checkpoint/HarNet/ID映射只在restricted_manifests。发布完成状态以根目录之外的publication receipt及远端精确tree回读为准。完成后停止。
