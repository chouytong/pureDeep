# Phase10 统计附录

主统计单位：完整矩阵先对同split三个seed等权平均，再对15split等权。final_metrics中的SD是15个split均值的sample SD(ddof=1)，另有三个seed跨split均值的sample SD。pilot的SD是同一split三个seed指标的sample SD，不能与跨split SD等同。D0只有15个确定性LR拟合，没有45个独立拟合；seed SD为NA。Recall分别给出PD与DD，主规则严格pDD>0.5、tie PD。

本轮全部新候选止于3seed pilot。mean/median delta与seed正负计数、描述seed dz保留；15split wins/ties/losses、paired split CI、Wilcoxon、rank-biserial与BH q均NA。未把seed当作患者队列作推断。预登记主族4候选×2指标=8，次族4×7=28；未运行和pilot推断项内部p=1占位，其自身公开p/q为NA，不缩减比较族或制造显著性。

subject_cluster_bootstrap.csv为10,000次PD/DD分层受试者重采样。相同subject在所有seed/appearance共用一次抽样权重，每次先计算匹配模型差，再等权平均run（完整等价于split内三seed平均后split平均）。本轮每候选仅105名validation受试者，固定既有拟合，不重新训练。CI仅为条件性敏感度，不是独立确认，不包括模型选择、训练重拟合、中心或设备迁移的不确定性。未以该CI绕过pilot闸门。weighted kernel用全1权重与正式九指标逐项核对到1e-12，包含项目两概率列macro AUROC、logsumexp NLL、two-column Brier和10等宽top-confidence ECE。

阈值仅改变硬决策及混淆矩阵，不改变固定概率下的AUROC/NLL/Brier/ECE。所有validation操作曲线是描述性；合法Certified train-only OOF缺失，部署阈值BLOCKED。PR图先逐fit计算经验曲线，再在固定recall grid插值平均，不能当成390个独立人一次预测的PR。

| condition | metric | mean_delta | subject_cluster_ci_low | subject_cluster_ci_high | bootstrap_draws | finite_draws | unique_validation_subjects | stratified_PD_DD | repeated_subjects_share_weight | training_refit | inference_scope | p_value | BH_q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FT-L1 | accuracy | 7.41594e-17 | -0.0444444 | 0.0444444 | 10000 | 10000 | 105 | True | True | False | CONDITIONAL_FIXED_FITS_DESCRIPTIVE_SENSITIVITY | NA | NA |
| FT-L1 | ba | -0.00312409 | -0.0488957 | 0.0403952 | 10000 | 10000 | 105 | True | True | False | CONDITIONAL_FIXED_FITS_DESCRIPTIVE_SENSITIVITY | NA | NA |
| FT-L1 | auroc | -0.00697472 | -0.0550712 | 0.0414124 | 10000 | 10000 | 105 | True | True | False | CONDITIONAL_FIXED_FITS_DESCRIPTIVE_SENSITIVITY | NA | NA |
| FT-L1 | macro_f1 | -0.00184557 | -0.0470561 | 0.042185 | 10000 | 10000 | 105 | True | True | False | CONDITIONAL_FIXED_FITS_DESCRIPTIVE_SENSITIVITY | NA | NA |
| FT-L1 | pd_recall | 0.0045045 | -0.0495495 | 0.0630631 | 10000 | 10000 | 105 | True | True | False | CONDITIONAL_FIXED_FITS_DESCRIPTIVE_SENSITIVITY | NA | NA |
| FT-L1 | dd_recall | -0.0107527 | -0.0860215 | 0.0537634 | 10000 | 10000 | 105 | True | True | False | CONDITIONAL_FIXED_FITS_DESCRIPTIVE_SENSITIVITY | NA | NA |
| FT-L1 | nll | -0.158344 | -0.288149 | -0.0396894 | 10000 | 10000 | 105 | True | True | False | CONDITIONAL_FIXED_FITS_DESCRIPTIVE_SENSITIVITY | NA | NA |
| FT-L1 | brier | -0.048844 | -0.109998 | 0.00889265 | 10000 | 10000 | 105 | True | True | False | CONDITIONAL_FIXED_FITS_DESCRIPTIVE_SENSITIVITY | NA | NA |
| FT-L1 | ece | -0.0917368 | -0.137105 | -0.0239584 | 10000 | 10000 | 105 | True | True | False | CONDITIONAL_FIXED_FITS_DESCRIPTIVE_SENSITIVITY | NA | NA |
| DEMO | accuracy | 0 | -0.00952381 | 0.00952381 | 10000 | 10000 | 105 | True | True | False | CONDITIONAL_FIXED_FITS_DESCRIPTIVE_SENSITIVITY | NA | NA |
| DEMO | ba | -0.00312409 | -0.016129 | 0.00675676 | 10000 | 10000 | 105 | True | True | False | CONDITIONAL_FIXED_FITS_DESCRIPTIVE_SENSITIVITY | NA | NA |
| DEMO | auroc | -0.00043592 | -0.00203429 | 0.00087184 | 10000 | 10000 | 105 | True | True | False | CONDITIONAL_FIXED_FITS_DESCRIPTIVE_SENSITIVITY | NA | NA |
| DEMO | macro_f1 | -0.00192021 | -0.0144143 | 0.00791593 | 10000 | 10000 | 105 | True | True | False | CONDITIONAL_FIXED_FITS_DESCRIPTIVE_SENSITIVITY | NA | NA |
| DEMO | pd_recall | 0.0045045 | 0 | 0.0135135 | 10000 | 10000 | 105 | True | True | False | CONDITIONAL_FIXED_FITS_DESCRIPTIVE_SENSITIVITY | NA | NA |
| DEMO | dd_recall | -0.0107527 | -0.0322581 | 0 | 10000 | 10000 | 105 | True | True | False | CONDITIONAL_FIXED_FITS_DESCRIPTIVE_SENSITIVITY | NA | NA |
| DEMO | nll | -0.00184754 | -0.00269213 | -0.00102117 | 10000 | 10000 | 105 | True | True | False | CONDITIONAL_FIXED_FITS_DESCRIPTIVE_SENSITIVITY | NA | NA |
| DEMO | brier | -0.00125844 | -0.00184823 | -0.000711406 | 10000 | 10000 | 105 | True | True | False | CONDITIONAL_FIXED_FITS_DESCRIPTIVE_SENSITIVITY | NA | NA |
| DEMO | ece | 0.00443689 | -0.0127864 | 0.0167073 | 10000 | 10000 | 105 | True | True | False | CONDITIONAL_FIXED_FITS_DESCRIPTIVE_SENSITIVITY | NA | NA |
| DD | accuracy | 0.015873 | -0.00952381 | 0.0444444 | 10000 | 10000 | 105 | True | True | False | CONDITIONAL_FIXED_FITS_DESCRIPTIVE_SENSITIVITY | NA | NA |
| DD | ba | 0.0175094 | -0.0170009 | 0.0551439 | 10000 | 10000 | 105 | True | True | False | CONDITIONAL_FIXED_FITS_DESCRIPTIVE_SENSITIVITY | NA | NA |
| DD | auroc | 0.01424 | -0.0126417 | 0.0419936 | 10000 | 10000 | 105 | True | True | False | CONDITIONAL_FIXED_FITS_DESCRIPTIVE_SENSITIVITY | NA | NA |
| DD | macro_f1 | 0.0172476 | -0.0146081 | 0.0520574 | 10000 | 10000 | 105 | True | True | False | CONDITIONAL_FIXED_FITS_DESCRIPTIVE_SENSITIVITY | NA | NA |
| DD | pd_recall | 0.0135135 | -0.0135135 | 0.0405405 | 10000 | 10000 | 105 | True | True | False | CONDITIONAL_FIXED_FITS_DESCRIPTIVE_SENSITIVITY | NA | NA |
| DD | dd_recall | 0.0215054 | -0.0430108 | 0.0860215 | 10000 | 10000 | 105 | True | True | False | CONDITIONAL_FIXED_FITS_DESCRIPTIVE_SENSITIVITY | NA | NA |
| DD | nll | 0.125395 | 0.0181904 | 0.244052 | 10000 | 10000 | 105 | True | True | False | CONDITIONAL_FIXED_FITS_DESCRIPTIVE_SENSITIVITY | NA | NA |
| DD | brier | 0.00313023 | -0.0309564 | 0.0392866 | 10000 | 10000 | 105 | True | True | False | CONDITIONAL_FIXED_FITS_DESCRIPTIVE_SENSITIVITY | NA | NA |
| DD | ece | 0.0479887 | -0.00160533 | 0.0757325 | 10000 | 10000 | 105 | True | True | False | CONDITIONAL_FIXED_FITS_DESCRIPTIVE_SENSITIVITY | NA | NA |


15个split有受试者重叠，三个seed也不是三个患者队列。新候选仅1 split×3 seed，全部未通过资源闸门，不能据此证明泛化劣于基线或发布15split显著性结论。历史数据与Certified数据分开，当前正式模型指针未变。
