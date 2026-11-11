# Phase5C：Frozen与Adapted受试者变化

| label | n | gross_rescue | newly_broken | net_rescue | net_rescue_rate |
| --- | --- | --- | --- | --- | --- |
| DD | 30.400000 | 1.777778 | 1.733333 | 0.044444 | 0.001365 |
| PD | 73.600000 | 4.066667 | 3.466667 | 0.600000 | 0.008149 |
| all | 104.000000 | 5.844444 | 5.200000 | 0.644444 | 0.006224 |


患者四appearance模式（每appearance三seed均值概率）：

| error_group | label | subjects | improved_subjects | unchanged_subjects | worsened_subjects | subjects_any_rescue | subjects_any_new_error | subjects_both_rescue_and_harm | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | all | 390 | 43 | 301 | 46 | 60 | 62 | 17 | 4 appearances, seed probability mean; not a new deployed classifier |
| all | PD | 276 | 30 | 216 | 30 | 41 | 40 | 11 | 4 appearances, seed probability mean; not a new deployed classifier |
| all | DD | 114 | 13 | 85 | 16 | 19 | 22 | 6 | 4 appearances, seed probability mean; not a new deployed classifier |


| condition | error_group | subjects | gross_rescue_appearance_fraction | newly_broken_appearance_fraction | mean_delta | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adapted | all | 390 | 0.043590 | 0.044231 | -0.000641 | -0.014744 | 0.014103 |
| adapted | stable-error | 68 | 0.117647 | 0.055147 | 0.062500 | 0.016129 | 0.111842 |
| adapted | stable-correct | 279 | 0.019713 | 0.036738 | -0.017025 | -0.031136 | -0.002737 |
| adapted | unstable | 43 | 0.081395 | 0.075581 | 0.005814 | -0.046512 | 0.057692 |


单seed正式口径平均每unit救回5.844444、破坏5.2、净+.644444人；PD净+.6人，DD净+.044444。患者四appearance口径有43人改善、46人恶化、301不变；17人跨appearance既有rescue也有harm，不能声称所有人一致改善。后一个口径不是正式单模型性能，不能用其净数替换前者。

现有adapted vs frozen正式BA+.004757，split CI[-.000612,.010149]、历史本轮重分析6项q=.301647；subject-cluster[-.008245,.017479]。AUROC+.009355，cluster[.001011,.017972]，但历史开发联合门槛未通过、不复活候选。AUC的单一CI不抵消BA/recall/seed与既有停止规则。

stable-error中的改善伴随stable-correct上的损失，且这些组由原validation错误定义；这是异质预测变化的描述，不是可部署路由机制。按PD/DD与四事件的probability-implied log odds、真实标签signed margin变化见adaptation_prediction_drift.csv。rescued组平均true-signed log-odds变化约+1.116321、newly-broken约−1.186739，这与事件定义有关，不能拿同义数值关联当机制发现。

H3结论：预测/受试者改善异质性有直接证据；“由特定表征变化造成”无法核验，表征资产缺失，不能给出一致的疾病机制解释或据此新增适配训练。


本轮属于同一390人、15个重叠development splits的重复开发。正式指标先同split平均三个seed的指标；三seed概率均值仅用于错误分组。split bootstrap、seed SD、subject-cluster CI均不能充当独立外部泛化证据。cluster bootstrap按PD276/DD114分层，10,000次，同一人的全部appearance/seed共享采样次数，仅固定已拟合模型条件下的不确定性；不覆盖重训练和多轮开发选择偏差。不同条件/估计对象的CI是否跨0，不构成它们之间差异的检验。
