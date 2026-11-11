# A2 概率校准

| condition | nll | brier | ece | classwise_ece | calibration_intercept | calibration_slope |
| --- | --- | --- | --- | --- | --- | --- |
| positive_temperature | 0.577543 | 0.383837 | 0.107569 | 0.162571 | -0.664349 | 1.311187 |
| raw | 0.616720 | 0.386441 | 0.110333 | 0.168882 | -0.664350 | 0.779466 |


| metric | reference_mean | candidate_mean | mean_delta | median_delta | wins | ties | losses | ci_low | ci_high | paired_dz | rank_biserial | wilcoxon_p | split_delta_sd | bh_q | improved_count | worsened_count | direction_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nll | 0.616720 | 0.577543 | -0.039178 | -0.020002 | 3 | 0 | 12 | -0.064443 | -0.016595 | -0.808575 | -0.783333 | 0.005371 | 0.048453 | 0.016113 | 12 | 3 | lower NLL/Brier/ECE is better; original wins column counts positive delta, not improvement |
| brier | 0.386441 | 0.383837 | -0.002604 | -0.004335 | 6 | 0 | 9 | -0.009263 | 0.004177 | -0.187740 | -0.183333 | 0.561401 | 0.013870 | 0.761536 | 9 | 6 | lower NLL/Brier/ECE is better; original wins column counts positive delta, not improvement |
| ece | 0.110333 | 0.107569 | -0.002764 | -0.012458 | 7 | 0 | 8 | -0.015040 | 0.009774 | -0.109960 | -0.100000 | 0.761536 | 0.025133 | 0.761536 | 8 | 7 | lower NLL/Brier/ECE is better; original wins column counts positive delta, not improvement |


全部45个T仅在对应target inner-train的合格deep-OOF log odds上拟合，固定正温度T∈[.05,20]，无阈值搜索；样本端点概率0行，无clip后伪造logit。validation截距/斜率90个descriptive fit只用于诊断，不部署；参数、梯度收敛、逐seed/split指标与固定10等宽reliability bins保留。proper unweighted NLL、双类sum Brier（2×二分类平方误差）、top-confidence ECE、classwise ECE均与weighted训练CE区分。

NLL .616720→.577543，Δ-.039178，12/15改善，CI[-.064443,-.016595]，BH三项q=.016113。Brier .386441→.383837、ECE .110333→.107569，两者CI跨0。预登记mean NLL<0且Brier<=0的诊断规则通过，标记RETAIN_DIAGNOSTIC_ONLY；不代表校准已外部验证，更不改变原单模型身份。正T保留每个已有模型的排序及原.5决策，AUROC/BA无提升。

OOF训练池更小、epoch10固定，deployment是full pool/best epoch；分数/参考库尺度迁移见A2_Bfusion_score_transfer.csv。该分布差不能由OOF来源合法自动消除；不使用其他见过当前validation的context模型补分数。按PD/DD/来源组的NLL/Brier/ECE及DD概率可靠性见A2_grouped_calibration、A2_DD_reliability_bins；这是描述性分组而非亚型校准器。reliability.svg为聚合图，重复appearance不当独立样本。

方法参考：[Guo等，ICML2017](https://proceedings.mlr.press/v70/guo17a.html)。

15 个 development splits 共享受试者和训练池，45 次已有 seed/run 不是 45 个独立样本。主统计先在同一 split 平均三个 seed 的指标，再汇总 15 splits；不是概率 ensemble。主 bootstrap seed=20261111、10,000 次；Wilcoxon 和 BH family 见各表。受试者分层 cluster bootstrap 将同一人的全部 appearance 一起重采样，仅描述固定已拟合模型下的条件性不确定性，不覆盖重训练、开发多重选择或外部泛化。5 个 context 汇总仅为 inner-development 分组，不是 outer-test。
