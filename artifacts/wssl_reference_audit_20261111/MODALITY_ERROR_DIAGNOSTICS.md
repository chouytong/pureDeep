# A1 模态和错误诊断

主错误定义每appearance先均值三seed概率，再按四appearance错误比例>=.75：68人（PD35/DD33）；全部四appearance错38（PD14/DD24），全部12次错12，是不同敏感性定义。

| target | source_category | n | mean_probability | mean_absolute_margin | mean_error_fraction | stable_error_n | stable_correct_n | split_probability_sd | within_seed_sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Parkinson's | 276 | 0.306836 | 0.235774 | 0.190217 | 35 | 215 | 0.119411 | 0.146196 |
| 1 | Atypical Parkinsonism | 15 | 0.550591 | 0.208447 | 0.333333 | 4 | 9 | 0.116913 | 0.128335 |
| 1 | Essential Tremor | 28 | 0.679337 | 0.262789 | 0.205357 | 4 | 22 | 0.113080 | 0.118208 |
| 1 | Multiple Sclerosis | 11 | 0.569467 | 0.217255 | 0.431818 | 5 | 6 | 0.106348 | 0.171321 |
| 1 | Other Movement Disorders | 60 | 0.536955 | 0.201772 | 0.445833 | 20 | 27 | 0.129595 | 0.143937 |


Adapted相对Frozen每split/seed平均纠错（对全体人，非只筛错误者）：

| label | corrected | newly_wrong | net |
| --- | --- | --- | --- |
| 0 | 4.066667 | 3.466667 | 0.600000 |
| 1 | 1.777778 | 1.733333 | 0.044444 |


固定probe完整配对：

| metric | reference_mean | candidate_mean | mean_delta | median_delta | wins | ties | losses | ci_low | ci_high | paired_dz | rank_biserial | wilcoxon_p | split_delta_sd | bh_q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| accuracy | 0.754067 | 0.751731 | -0.002336 | 0.000000 | 5 | 3 | 7 | -0.007904 | 0.002980 | -0.212399 | -0.243590 | 0.455605 | 0.010998 | 0.911211 |
| ba | 0.669348 | 0.669825 | 0.000476 | 0.001294 | 8 | 0 | 7 | -0.005497 | 0.006079 | 0.040243 | 0.100000 | 0.761536 | 0.011838 | 0.977966 |
| auroc | 0.755291 | 0.755457 | 0.000166 | 0.004359 | 8 | 0 | 7 | -0.007544 | 0.007798 | 0.010368 | 0.050000 | 0.890381 | 0.016008 | 0.977966 |
| macro_f1 | 0.678563 | 0.678378 | -0.000185 | 0.003478 | 8 | 0 | 7 | -0.006685 | 0.005924 | -0.014375 | 0.016667 | 0.977966 | 0.012841 | 0.977966 |
| pd_recall | 0.873442 | 0.867127 | -0.006315 | -0.009009 | 5 | 1 | 9 | -0.013238 | 0.000634 | -0.454038 | -0.438095 | 0.148578 | 0.013908 | 0.585585 |
| dd_recall | 0.465254 | 0.472522 | 0.007267 | 0.000000 | 7 | 3 | 5 | -0.002912 | 0.017419 | 0.351526 | 0.423077 | 0.195195 | 0.020674 | 0.585585 |


既有45个last-block adapted预测复用，normalization hash与baseline一致；未重跑37-condition contribution矩阵。各条件全部六指标和稳定性见STATISTICAL_APPENDIX.md。source_category仅从既有CrossArms manifest来源字段聚合，未新增临床个体表。分数/margin/seed与appearance稳定性见上表及私有错误表，不用于逐患者路由、采样、删人或重加权。

唯一预固定probe：原WSSL logit + flattened22×1024外部Acc cache的train StandardScaler/PCA16为control；candidate仅加ordered11×2×64 local特征经独立train Scaler/PCA16。最终StandardScaler+C=1 L2 logistic，natural class weights=None，共90个轻量fit。ΔBA=+.000476、ΔAUC=+.000166，CI均跨0，没有稳定local增量。local混合Acc+Gyro，未分离Gyro-only；不能归因Gyro，也不能证明Gyro无用。底层监督local encoder见过这些inner-train人，probe不是整个模型链OOF。

新特征含local、SSL投影完整residual、bilateral、subject、structured，来源由baseline提取脚本固定。没有将输入删除/置换解释为重新训练单模态或疾病因果；未新训练encoder/内部adapter。
