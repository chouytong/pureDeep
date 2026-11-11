# Phase5B：既有融合失败的直接数值解释

45份既有系数恢复所有validation概率，最大误差2.22e-16，原决策完全相同。没有重新训练、选系数或调阈值。

```text
f = b + wW * (zDD-zPD-muW)/sigmaW + wR * (r-muR)/sigmaR
P(DD) = sigmoid(f); DD iff f>0 (tie PD)
```

| parameter | mean | min | max |
| --- | --- | --- | --- |
| W_coefficient | 0.566324 | 0.030297 | 1.021622 |
| R_coefficient | 0.387709 | -0.003749 | 1.035196 |
| intercept | -1.022500 | -1.178593 | -0.927206 |
| W_scaler_mean | -0.777134 | -1.563619 | 0.009217 |
| R_scaler_mean | -0.006932 | -0.011741 | -0.000684 |
| W_scaler_scale | 1.583680 | 1.089677 | 2.366524 |
| R_scaler_scale | 0.044801 | 0.035182 | 0.055058 |
| training_DD_prevalence | 0.292298 | 0.287805 | 0.295238 |
| training_prior_log_odds | -0.884272 | -0.906069 | -0.870078 |
| W_standardized_validation_mean | -0.088738 | -1.774235 | 0.712581 |
| R_standardized_validation_mean | 0.061573 | -0.064040 | 0.218897 |
| W_mean_contribution | -0.023490 | -1.200431 | 0.630383 |
| R_mean_contribution | 0.017779 | -0.022455 | 0.071062 |
| validation_mean_logit_shift | -0.005711 | -1.130223 | 0.633931 |
| W_mean_absolute_contribution | 0.637394 | 0.018222 | 1.818218 |
| R_mean_absolute_contribution | 0.251440 | 0.001829 | 0.639082 |


类别输出及线性分解：

| label | baseline_DD_prediction_rate | fusion_DD_prediction_rate | mean_baseline_pDD | mean_fusion_pDD | mean_W_contribution | mean_R_contribution | intercept | mean_fusion_logit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DD | 0.656945 | 0.327707 | 0.577047 | 0.420457 | 0.422764 | 0.209945 | -1.022500 | -0.389791 |
| PD | 0.217656 | 0.072430 | 0.306748 | 0.254060 | -0.207797 | -0.061763 | -1.022500 | -1.292059 |
| all | 0.345980 | 0.147096 | 0.385742 | 0.302716 | -0.023490 | 0.017779 | -1.022500 | -1.028210 |


原DD正确被融合破坏的子集：

| quantity | mean_across_nonempty_units |
| --- | --- |
| n | 10.250000 |
| mean_baseline_pDD | 0.653471 |
| mean_fusion_pDD | 0.406604 |
| mean_W_contribution | 0.497969 |
| mean_R_contribution | 0.138089 |
| intercept | -1.021457 |
| mean_fusion_logit | -0.385399 |


预登记B机制检验：

| test | family | status | units | mean_delta | median_delta | split_delta_sd | wins | ties | losses | ci_low | ci_high | cohen_dz | rank_biserial | wilcoxon_p | bh_q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W_standardized_validation_mean | B_mechanism | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | -0.088738 | -0.071055 | 0.432439 | 7 | 0 | 8 | -0.307968 | 0.123349 | -0.205203 | -0.183333 | 0.561401 | 0.748535 |
| R_standardized_validation_mean | B_mechanism | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.061573 | 0.039674 | 0.082477 | 12 | 0 | 3 | 0.022955 | 0.103785 | 0.746543 | 0.716667 | 0.012451 | 0.024902 |
| validation_mean_logit_shift | B_mechanism | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | -0.005711 | -0.019677 | 0.265751 | 7 | 0 | 8 | -0.135326 | 0.128281 | -0.021489 | -0.016667 | 0.977966 | 0.977966 |
| W_minus_R_absolute_contribution | B_mechanism | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.385954 | 0.403479 | 0.345673 | 13 | 0 | 2 | 0.218325 | 0.554594 | 1.116529 | 0.900000 | 0.000854 | 0.003418 |


直接数值事实：训练DD比例均值.292298，b均值−1.022500，45个b全部为负；W系数45个为正，R有44个正/1个微负（最小−.003749），不是整体输入方向反转。DD组两项平均贡献+.422764和+.209945，仍不足抵消负截距，平均f=−.389791。原DD被破坏者保存概率由.653471降到.406604，平均f=−.385399；上表只对该事件非空的unit平均，以保持各线性项与截距同一权重，逐行恒等式完全通过。不能用不同权重的总体均值反推精确个体f。

全体DD预测比例.345980→.147096；DD mean P从.577047→.420457，PD false positive比例.217656→.072430。固定.5规则将大量原来略偏DD的样本判为PD；同一个排序分数加截距/改变概率尺度可保留较好排序而显著移动决策。融合AUROC+.004087只是点估计，历史split CI[-.004272,.010426]、cluster[-.007039,.015225]，不能说有可靠AUROC增益。

自然类先验解释“相容”而非已证明因果：融合是未加class_weight的C1 logistic，训练DD约29.23%，原WSSL用balanced CE。b接近但不等于训练prior log odds均值−.884272，分布形状和L2也影响b。validation−train DD比例均值仅−.0000166，不能用明显新队列prior漂移解释本例。没有用重加权或去截距重算任何新性能。

OOF/full迁移证据受限：reference标准化均值迁移+.061573，split CI[.022955,.103785]，BH4项q=.024902；其方向向DD，不是整体向PD。WSSL总体标准化均值−.088738，CI跨0；最终线性均值val−OOF(=b)仅−.005711，CI[-.135326,.128281]、q=.977966。因此“整体均值迁移单独导致DD崩塌”不受支持。两输入均有尺度/类条件差异，但缺reference OOF逐人联合分布及完整状态，不能因果归因OOD/full shift或计算完整训练混淆矩阵。

保存记录中未发现编码反转、输入标准化错用或跨fold bank混用，且deployment线性重放精确；这不等于缺失reference变换链已获独立数值验证。W的mean absolute贡献.637394，大于R .251440，差的CI[.218325,.554594]；不是R全面支配。协方差完整保存，不能仅以各方差比例把相关输入的贡献分解为独立因果份额。保持原REJECT，停止融合/阈值搜索。


本轮属于同一390人、15个重叠development splits的重复开发。正式指标先同split平均三个seed的指标；三seed概率均值仅用于错误分组。split bootstrap、seed SD、subject-cluster CI均不能充当独立外部泛化证据。cluster bootstrap按PD276/DD114分层，10,000次，同一人的全部appearance/seed共享采样次数，仅固定已拟合模型条件下的不确定性；不覆盖重训练和多轮开发选择偏差。不同条件/估计对象的CI是否跨0，不构成它们之间差异的检验。
