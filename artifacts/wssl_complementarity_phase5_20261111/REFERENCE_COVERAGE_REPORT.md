# Phase5A：参考覆盖度

u=min(dPD,dDD)，每validation split标签无关的q25/q75只定义描述组。

| label | low_n | high_n | low_error_rate | high_error_rate | high_minus_low_error_rate |
| --- | --- | --- | --- | --- | --- |
| DD | 5.333333 | 10.533333 | 0.519021 | 0.229848 | -0.289173 |
| PD | 21.133333 | 15.933333 | 0.150468 | 0.300587 | 0.150119 |
| all | 26.466667 | 26.466667 | 0.222634 | 0.271921 | 0.049288 |


| test | family | status | units | mean_delta | median_delta | split_delta_sd | wins | ties | losses | ci_low | ci_high | cohen_dz | rank_biserial | wilcoxon_p | bh_q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coverage_high_minus_low_error_DD | A_coverage | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | -0.289173 | -0.300000 | 0.214161 | 1 | 1 | 13 | -0.393754 | -0.186329 | -1.350261 | -0.980952 | 0.001225 | 0.001838 |
| coverage_high_minus_low_error_PD | A_coverage | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.150119 | 0.126984 | 0.065933 | 15 | 0 | 0 | 0.120477 | 0.184243 | 2.276840 | 1.000000 | 0.000061 | 0.000183 |
| coverage_high_minus_low_error_all | A_coverage | DESCRIPTIVE_REPEATED_DEVELOPMENT | 15 | 0.049288 | 0.024691 | 0.064389 | 10 | 2 | 3 | 0.018706 | 0.081895 | 0.765473 | 0.758242 | 0.018351 | 0.018351 |


患者cluster、pooled exposure估计（与上表15split等权估计不同）：

| mean_delta | ci_low | ci_high | valid_draws | invalid_draws | requested_draws | scope | label | metric | estimand |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.049538 | -0.025269 | 0.124220 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | all | high_minus_low_error | pooled appearance/seed exposure rates; NOT 15split equal weight |
| 0.148441 | 0.065230 | 0.231629 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | PD | high_minus_low_error | pooled appearance/seed exposure rates; NOT 15split equal weight |
| -0.274209 | -0.439300 | -0.105957 | 10000 | 0 | 10000 | class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only | DD | high_minus_low_error | pooled appearance/seed exposure rates; NOT 15split equal weight |


高u不是通用“低可靠性”标记：PD高u错误率.300587 vs低u .150468；DD却为.229848 vs .519021，方向相反。两类各自的条件性cluster差仍为PD+.148441[.065230,.231629]、DD−.274209[−.439300,−.105957]。总体pooled差+.049538的cluster CI跨0；不把它与15split CI不跨0解释成两方法显著不同。

u与WSSL absolute probability margin的Spearman总体−.041976，PD−.081290/DD+.107506，弱且分组方向不同；u与relative总体+.370494。完整分组相关见coverage_correlations.csv。高/低组PD/DD组成不同（平均DD人数5.33 vs10.53），整体错误差同时混合类别/几何，不能视为可靠性因果证据。这里的u也不是临床严重度、异常疾病、错标或健康偏离指数。

禁止按本表进行筛人、采样、患者路由或训练权重改变。若未来检验该现象，应在独立数据重复同一固定诊断，不能基于验证标签构建门控。


本轮属于同一390人、15个重叠development splits的重复开发。正式指标先同split平均三个seed的指标；三seed概率均值仅用于错误分组。split bootstrap、seed SD、subject-cluster CI均不能充当独立外部泛化证据。cluster bootstrap按PD276/DD114分层，10,000次，同一人的全部appearance/seed共享采样次数，仅固定已拟合模型条件下的不确定性；不覆盖重训练和多轮开发选择偏差。不同条件/估计对象的CI是否跨0，不构成它们之间差异的检验。
