# Phase10B: FT-L1

状态：**PILOT_REJECT_STOP**。

| condition | runs | splits | accuracy | ba | auroc | macro_f1 | pd_recall | dd_recall | nll | brier | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FT-L1 | 3 | 1 | 0.695238 | 0.668192 | 0.704882 | 0.655298 | 0.734234 | 0.602151 | 0.612145 | 0.416861 | 0.0587477 |
| WSSL-CERTIFIED-L1 | 3 | 1 | 0.695238 | 0.671316 | 0.711857 | 0.657144 | 0.72973 | 0.612903 | 0.77049 | 0.465705 | 0.150485 |


| metric | mean_delta | median_delta |
| --- | --- | --- |
| accuracy | 7.40149e-17 | 0.00952381 |
| ba | -0.00312409 | -0.00152572 |
| auroc | -0.00697472 | -0.00697472 |
| macro_f1 | -0.00184557 | -0.00225506 |
| pd_recall | 0.0045045 | 0.0405405 |
| dd_recall | -0.0107527 | -0.0322581 |
| nll | -0.158344 | -0.180021 |
| brier | -0.048844 | -0.0687612 |
| ece | -0.0917368 | -0.116885 |


门槛逐项：

| check | passed |
| --- | --- |
| BA_positive | False |
| BA_improved_at_least2of3 | False |
| AUROC_non_decrease | False |
| DD_recall_drop_at_most01 | False |
| NLL_increase_at_most05 | True |
| correctness_and_provenance | True |


最后stage位置与历史Layer5相同；本轮Certified L1/norm、fresh下游状态、LR1e-5且BN affine冻结。FT-L1未通过资源闸门时不增加解冻深度：本轮没有独立证据证明瓶颈位于layer4。FT-L2为NOT_RUN，不报告最后一与最后两stage的性能差。

仅context0/inner0三seed资源筛选；无15split CI/Wilcoxon/BH，不作独立确认。样本SD和全部九指标见metrics/paired_statistics；参数量、gradient、权重norm变化、耗时、best epoch及checkpoint/optimizer来源均保存。源数据、norm、baseline权重、split和CURRENT_MODEL保持原样。未通过不补seed、不调阈值/参数、不扩大训练；其他独立阶段按原条件继续。
