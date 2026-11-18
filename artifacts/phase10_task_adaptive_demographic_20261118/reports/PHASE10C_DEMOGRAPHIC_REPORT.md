# Phase10C: DEMO

状态：**PILOT_REJECT_STOP**。

| condition | runs | splits | accuracy | ba | auroc | macro_f1 | pd_recall | dd_recall | nll | brier | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DEMO | 3 | 1 | 0.695238 | 0.668192 | 0.711421 | 0.655223 | 0.734234 | 0.602151 | 0.768642 | 0.464446 | 0.154921 |
| WSSL-CERTIFIED-L1 | 3 | 1 | 0.695238 | 0.671316 | 0.711857 | 0.657144 | 0.72973 | 0.612903 | 0.77049 | 0.465705 | 0.150485 |


| metric | mean_delta | median_delta |
| --- | --- | --- |
| accuracy | 0 | 0 |
| ba | -0.00312409 | 0 |
| auroc | -0.00043592 | -0.00043592 |
| macro_f1 | -0.00192021 | 0 |
| pd_recall | 0.0045045 | 0 |
| dd_recall | -0.0107527 | 0 |
| nll | -0.00184754 | -0.00183274 |
| brier | -0.00125844 | -0.00103946 |
| ece | 0.00443689 | 0.000400611 |


门槛逐项：

| check | passed |
| --- | --- |
| BA_positive | False |
| BA_improved_at_least2of3 | False |
| AUROC_non_decrease | False |
| DD_recall_drop_at_most01 | False |
| NLL_increase_at_most05 | True |
| correctness_and_provenance | True |
| PD_recall_drop_at_most01 | True |
| fusion_not_dominated_by_demographics_only | True |
| no_extreme_evaluable_group_recall_drop | True |
| at_least_one_evaluable_group | True |


原Certified WSSL所有参数和logits冻结，新MLP4→8→16与零初始化16→2贡献共216参数；不是用人口字段替换IMU。gender是原始记录字段，设备只分层不输入；D0单独见10A报告。人口相关性可能反映队列混杂，不是疾病因果。

仅context0/inner0三seed资源筛选；无15split CI/Wilcoxon/BH，不作独立确认。样本SD和全部九指标见metrics/paired_statistics；参数量、gradient、权重norm变化、耗时、best epoch及checkpoint/optimizer来源均保存。源数据、norm、baseline权重、split和CURRENT_MODEL保持原样。未通过不补seed、不调阈值/参数、不扩大训练；其他独立阶段按原条件继续。
