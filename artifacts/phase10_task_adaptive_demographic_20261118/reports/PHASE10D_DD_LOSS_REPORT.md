# Phase10D: DD

状态：**PILOT_REJECT_STOP**。

| condition | runs | splits | accuracy | ba | auroc | macro_f1 | pd_recall | dd_recall | nll | brier | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DD | 3 | 1 | 0.711111 | 0.688826 | 0.726097 | 0.674391 | 0.743243 | 0.634409 | 0.895885 | 0.468835 | 0.198473 |
| WSSL-CERTIFIED-L1 | 3 | 1 | 0.695238 | 0.671316 | 0.711857 | 0.657144 | 0.72973 | 0.612903 | 0.77049 | 0.465705 | 0.150485 |


| metric | mean_delta | median_delta |
| --- | --- | --- |
| accuracy | 0.015873 | 0.00952381 |
| ba | 0.0175094 | 0.016129 |
| auroc | 0.01424 | 0.00610288 |
| macro_f1 | 0.0172476 | 0.0126021 |
| pd_recall | 0.0135135 | 0 |
| dd_recall | 0.0215054 | 0.0322581 |
| nll | 0.125395 | -0.0293578 |
| brier | 0.00313023 | -0.00098923 |
| ece | 0.0479887 | 0.0032192 |


门槛逐项：

| check | passed |
| --- | --- |
| BA_positive | True |
| BA_improved_at_least2of3 | True |
| AUROC_non_decrease | True |
| DD_recall_drop_at_most01 | True |
| NLL_increase_at_most05 | False |
| correctness_and_provenance | True |
| PD_recall_drop_at_most01 | True |
| DD_recall_positive | True |
| MacroF1_non_decrease | True |


原balanced weighted-mean CE，仅DD相对cost×1.10，按train人数重标尺度；常数缩放会在实际batch权重分母中抵消，不声称保持梯度norm。未叠加focal/sampling/SupCon/AI，也未按validation错误名单加权。

仅context0/inner0三seed资源筛选；无15split CI/Wilcoxon/BH，不作独立确认。样本SD和全部九指标见metrics/paired_statistics；参数量、gradient、权重norm变化、耗时、best epoch及checkpoint/optimizer来源均保存。源数据、norm、baseline权重、split和CURRENT_MODEL保持原样。未通过不补seed、不调阈值/参数、不扩大训练；其他独立阶段按原条件继续。
