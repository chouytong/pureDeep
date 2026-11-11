# Phase8A1 三seed集成复核

历史Phase3B已报告概率集成，本轮不是首次发现。原45checkpoint和per-model prediction来源通过；无新训练/推理/温度/阈值拟合。

| condition | accuracy | ba | auroc | macro_f1 | pd_recall | dd_recall | nll | brier | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.745604 | 0.719644 | 0.759553 | 0.706589 | 0.782344 | 0.656945 | 0.616720 | 0.386441 | 0.110333 |
| logit_ensemble | 0.754069 | 0.715740 | 0.778185 | 0.709980 | 0.808170 | 0.623310 | 0.544251 | 0.350267 | 0.084375 |
| probability_ensemble | 0.757318 | 0.720025 | 0.781620 | 0.714136 | 0.809996 | 0.630053 | 0.529978 | 0.346340 | 0.073229 |


| condition | metric | mean_delta | wins | ci_low | ci_high | wilcoxon_p | primary_bh_q |
| --- | --- | --- | --- | --- | --- | --- | --- |
| probability_ensemble | accuracy | 0.011714 | 7 | -0.006064 | 0.029608 | 0.345448 | NA |
| probability_ensemble | ba | 0.000380 | 8 | -0.016742 | 0.018008 | 0.846924 | 0.846924 |
| probability_ensemble | auroc | 0.022067 | 15 | 0.017059 | 0.026521 | 0.000061 | 0.000122 |
| probability_ensemble | macro_f1 | 0.007546 | 8 | -0.010935 | 0.026512 | 0.561401 | NA |
| probability_ensemble | pd_recall | 0.027652 | 12 | 0.006968 | 0.049225 | 0.024810 | NA |
| probability_ensemble | dd_recall | -0.026892 | 4 | -0.048837 | -0.002017 | 0.044448 | NA |
| probability_ensemble | nll | -0.086742 | 0 | -0.116148 | -0.059200 | 0.000061 | NA |
| probability_ensemble | brier | -0.040102 | 0 | -0.050615 | -0.030069 | 0.000061 | NA |
| probability_ensemble | ece | -0.037104 | 3 | -0.052093 | -0.019889 | 0.002014 | NA |
| logit_ensemble | accuracy | 0.008465 | 8 | -0.007278 | 0.024616 | 0.509693 | NA |
| logit_ensemble | ba | -0.003905 | 7 | -0.019006 | 0.011877 | 0.719727 | 0.719727 |
| logit_ensemble | auroc | 0.018632 | 15 | 0.013558 | 0.023491 | 0.000061 | 0.000122 |
| logit_ensemble | macro_f1 | 0.003391 | 7 | -0.012904 | 0.020105 | 0.761536 | NA |
| logit_ensemble | pd_recall | 0.025826 | 12 | 0.005529 | 0.046757 | 0.030874 | NA |
| logit_ensemble | dd_recall | -0.033635 | 4 | -0.056231 | -0.007646 | 0.024780 | NA |
| logit_ensemble | nll | -0.072469 | 0 | -0.092943 | -0.052923 | 0.000061 | NA |
| logit_ensemble | brier | -0.036175 | 0 | -0.047584 | -0.025625 | 0.000061 | NA |
| logit_ensemble | ece | -0.025957 | 2 | -0.037196 | -0.013831 | 0.001526 | NA |


probability_ensemble: **REJECT**；失败条件：BA_wins_at_least10, BA_CI_low_positive, DD_recall_drop_at_most01, stability_at_most1_25。

logit_ensemble: **REJECT**；失败条件：BA_positive, BA_wins_at_least10, BA_CI_low_positive, DD_recall_drop_at_most01, stability_at_most1_25。

固定3模型ensemble的seedSD不可识别，不报告虚假0；between-split稳定性与seed方差不同。Logit ensemble仅敏感性，不按结果替换主要候选。所有15split共享患者，区间不是独立临床验证。患者cluster敏感性在最终统一统计阶段追加，不改变此处原始paired/gate。完成本报告后进入A2。
