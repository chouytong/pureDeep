# Phase5A：受试者与排序互补性

单seed、每split/seed平均四格纠错：

| label | n | gross_rescue | newly_broken | net_rescue | net_rescue_rate |
| --- | --- | --- | --- | --- | --- |
| DD | 30.400000 | 4.000000 | 3.422222 | 0.577778 | 0.019001 |
| PD | 73.600000 | 5.511111 | 13.355556 | -7.844444 | -0.106607 |
| all | 104.000000 | 9.511111 | 16.777778 | -7.266667 | -0.069791 |


预登记四项A-primary：

| test | family | status | units | mean_delta | median_delta | split_delta_sd | wins | ties | losses | ci_low | ci_high | cohen_dz | rank_biserial | wilcoxon_p | bh_q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| relative_minus_WSSL_PD_correct_rate | A_primary | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | -0.106607 | -0.100457 | 0.072496 | 1 | 0 | 14 | -0.138551 | -0.067916 | -1.470507 | -0.866667 | 0.003137 | 0.004182 |
| relative_minus_WSSL_DD_correct_rate | A_primary | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.019001 | 0.043011 | 0.070945 | 9 | 0 | 6 | -0.016846 | 0.052139 | 0.267831 | 0.291667 | 0.319767 | 0.319767 |
| relative_minus_WSSL_BA | A_primary | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | -0.043803 | -0.036149 | 0.043625 | 3 | 0 | 12 | -0.066293 | -0.024382 | -1.004075 | -0.883333 | 0.001160 | 0.002319 |
| conditional_relative_AUC_on_WSSL_wrong_minus_half | A_primary | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | -0.083294 | -0.091142 | 0.061125 | 1 | 0 | 14 | -0.113138 | -0.053701 | -1.362684 | -0.950000 | 0.000305 | 0.001221 |


四appearance患者汇总（每appearance三seed均值概率决策）：

| condition | error_group | subjects | gross_rescue_appearance_fraction | newly_broken_appearance_fraction | mean_delta | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| relative | all | 390 | 0.080769 | 0.162179 | -0.081410 | -0.111554 | -0.051282 |
| relative | stable-error | 68 | 0.216912 | 0.062500 | 0.154412 | 0.085614 | 0.228814 |
| relative | stable-correct | 279 | 0.029570 | 0.179211 | -0.149642 | -0.182973 | -0.118326 |
| relative | unstable | 43 | 0.197674 | 0.209302 | -0.011628 | -0.121433 | 0.101570 |


在WSSL严格排错的PD–DD pairs中，relative正确排序比例.416706，比.5低.083294（1/15正，split CI[-.113138,-.053701]，dz−1.362684，BH四项q=.001221）；subject-cluster中心差CI[-.136756,-.020932]。平均每unit救回223个严格错序pair，同时破坏276.47个原正确pair；pair不算独立样本。上轮固定10bin条件AUC为另一个条件问题，不能与“严格错序子集”混同，也不把前者当本轮新证据。

主stable-error68（PD35/DD33）、stable-correct279、unstable43均按既有定义逐人复核。relative对stable-error正确比例改善.154412，却对stable-correct损失.149642，整体患者四appearance正确比例−.081410。这些组按WSSL validation错误定义，天然有不同rescue/harm机会，不能解释为预先存在的临床可路由亚型。错误组中有获救者不等于全体净收益，也不能用于重加权。

H1结论：弱命题“存在与WSSL不同且部分正确的排序/预测”有直接证据；强命题“存在可稳定利用的WSSL未充分利用的独立疾病信息”仍证据不足。当前“互补不足以获得稳定分类净收益”得到支持，不能从gross rescue推导新分类器价值。主指标与三seed概率均值分组不是同一个估计对象，因此总体−.069791单seed Accuracy差与患者分组−.081410不能混写。


本轮属于同一390人、15个重叠development splits的重复开发。正式指标先同split平均三个seed的指标；三seed概率均值仅用于错误分组。split bootstrap、seed SD、subject-cluster CI均不能充当独立外部泛化证据。cluster bootstrap按PD276/DD114分层，10,000次，同一人的全部appearance/seed共享采样次数，仅固定已拟合模型条件下的不确定性；不覆盖重训练和多轮开发选择偏差。不同条件/估计对象的CI是否跨0，不构成它们之间差异的检验。
