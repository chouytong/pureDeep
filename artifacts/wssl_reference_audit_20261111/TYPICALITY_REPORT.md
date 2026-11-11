# B 冻结双参考典型性

| condition | accuracy | ba | auroc | macro_f1 | pd_recall | dd_recall |
| --- | --- | --- | --- | --- | --- | --- |
| centroid_relative | 0.653972 | 0.655965 | 0.722245 | 0.627172 | 0.651253 | 0.660677 |
| common_dd | 0.301213 | 0.497296 | 0.499435 | 0.247952 | 0.025349 | 0.969242 |
| common_pd | 0.711664 | 0.535583 | 0.668489 | 0.503577 | 0.959324 | 0.111843 |
| full_dd | 0.301213 | 0.497296 | 0.499435 | 0.247952 | 0.025349 | 0.969242 |
| full_pd | 0.717422 | 0.542143 | 0.651314 | 0.510392 | 0.963841 | 0.120445 |
| full_relative | 0.712842 | 0.508748 | 0.709938 | 0.432249 | 1.000000 | 0.017496 |
| relative | 0.675813 | 0.675842 | 0.735639 | 0.647599 | 0.675737 | 0.675946 |
| strict_dd | 0.287781 | 0.491631 | 0.401855 | 0.224048 | 0.000901 | 0.982361 |
| strict_pd | 0.712934 | 0.526915 | 0.624100 | 0.482359 | 0.974676 | 0.079155 |


预登记主比较（2项AUROC的primary BH）：

| metric | reference_mean | candidate_mean | mean_delta | median_delta | wins | ties | losses | ci_low | ci_high | paired_dz | rank_biserial | wilcoxon_p | split_delta_sd | bh_q | comparison | primary_bh_q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| auroc | 0.668489 | 0.735639 | 0.067150 | 0.069406 | 13 | 0 | 2 | 0.042404 | 0.088420 | 1.422405 | 0.950000 | 0.000305 | 0.047209 | 0.000610 | relative-common_pd | 0.000305 |
| auroc | 0.499435 | 0.735639 | 0.236204 | 0.247945 | 15 | 0 | 0 | 0.207170 | 0.265064 | 3.942978 | 1.000000 | 0.000061 | 0.059905 | 0.000073 | relative-common_dd | 0.000122 |


与WSSL全体人纠错，单位为每split/seed平均人数：

| condition | label | corrected | newly_wrong | net |
| --- | --- | --- | --- | --- |
| fusion | 0 | 10.755556 | 0.066667 | 10.688889 |
| fusion | 1 | 0.044444 | 10.022222 | -9.977778 |
| relative | 0 | 5.511111 | 13.355556 | -7.844444 |
| relative | 1 | 4.000000 | 3.422222 | 0.577778 |


全体人相关/决策一致率：

| condition | score_spearman | prediction_agreement |
| --- | --- | --- |
| fusion | 0.945960 | 0.798972 |
| relative | 0.664238 | 0.747159 |


strict PD-only/DD-only所有PADS拟合仅该类；各自5fold subject crossfit拟合整个链，95%参考类距离定阈值，不使用另一类人数。外部HarNet独立冻结无PADS标签训练。common空间则监督class-conditional，PD/DD均疾病群体，不能称无监督PD发现或生理正常异常检测。

固定22 activity/wrist各自Scaler→PCA(min16,合法rank、无whitening、randomized seed20261111)→L2，cosine k5个不同subject，双腕等权后11活动等权；零向量距离1、rank0规定、少于5不降低k。训练query排除自身。common主分析以固定hash等人数bank，完整不平衡bank为预定敏感性；阈值95%参考类训练leave-self-out（并非全链OOF），relative界0。strict阈值与common阈值来源不能混同。

5,280次固定训练变换重放hash/norm审计：实际最小rank16、拟合向量与选中bank零向量均0，无降k/退化失败。fit-ID manifests私有；同一外部encoder缓存坐标系，未拼接不同监督fold embedding。仅用保存的subject聚合cache，无伪造2秒patch或将976点虚构两个完整10秒窗。距离未映射合法概率，NLL/Brier/ECE=NA。

relative相对两个单参考AUROC稳定占优；同空间centroid/full-bank敏感性完整24项BH见B_all_paired。固定10个WSSL概率bin的条件AUC screen增量CI[.086286,.129363]只提供粗分箱互补迹象，不能证明独立临床增量。relative自身BA .675842/AUC .735639低于WSSL。

闸门通过后仅执行一次两分数C1 L2 logistic：train deep-OOF WSSL logit+完全同算法train-chain OOF relative distance，各OOF变换/参考库排除query；45个fit，部署full inner-train链。BA .627638（Δ-.092006，0/15胜）、DD recall .327707（Δ-.329238），F1与seed稳定性护栏也失败。融合REJECT，不搜索权重/阈值/类别权重补救，不取代143,172参数原生模型。独立pipeline的输入scaler/coefficients私有保存，不能把它描述为原模型结果。

普通深度kNN本身不是创新：[Sun等，ICML2022](https://proceedings.mlr.press/v162/sun22d.html)。

15 个 development splits 共享受试者和训练池，45 次已有 seed/run 不是 45 个独立样本。主统计先在同一 split 平均三个 seed 的指标，再汇总 15 splits；不是概率 ensemble。主 bootstrap seed=20261111、10,000 次；Wilcoxon 和 BH family 见各表。受试者分层 cluster bootstrap 将同一人的全部 appearance 一起重采样，仅描述固定已拟合模型下的条件性不确定性，不覆盖重训练、开发多重选择或外部泛化。5 个 context 汇总仅为 inner-development 分组，不是 outer-test。
