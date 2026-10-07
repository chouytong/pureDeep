# Phase B：Activity-Scaled WSSL 正式报告（中文）

## A. Correctness

新增独立 `scripts/scaled_wssl.py`、`run.py`、`test_correctness.py`、`launch.py`、`analyze.py`、`write_report.py`、事后只做完整性验证的 `final_integrity.py` 与 `PROTOCOL.md`、analysis小型汇总。reference_*文件仅归档复用出处，不执行其候选入口。原 foundation/正式模型源码未改。附加11个无约束g，配置顺序建立index，同activity左右腕共享；uniform1；完整scratch训练，所有原classifier参数训练。实例绑定projection hook对完整output（含bias）缩放，随后原mask，后续bilateral/activity/structured路径完全沿用。不是zero-input，不用贡献初始化或删活动，不deepcopy。

参数：classifier-side 143,172→143,183；HarNet冻结10,457,408；系统10,600,580→10,600,591。缓存推理阶段不实例化HarNet，不生成新统计状态。g继承原AdamW同组WD，没有额外参数组。

全部45checkpoint×两个真实8subject validation batch（90批次、每checkpoint16subjects）g=1 logits与重算ordinary baseline逐位一致，fresh reload逐位一致；归档概率误差≤1e-12。正式CSV未存logits，未声称比较不存在的归档logits。Phase A已全validation复现，15train-only normalization refit和45正式资产SHA复核不变。

三seed scratch原参数和RNG、初始logits一致。两步仅inner-train正确性更新验证11g均finite/nonzero gradient（初始projection为零，第一步g梯度可为零，projection学到非零后再验证）、原STR/projection正常更新。activity/wrist/padding mask隔离、单g只影响本activity且双腕同倍数、实例storage/cache隔离、更新后logits/scalar/optimizer reload数值与dtype精确PASS。Optimizer step保存/重载的CPU/GPU位置按引擎规则恢复，比较值时转CPU。原引擎one-epoch/one-batch smoke PASS，不用smoke分数选择。

已有EMA实验**ordinary部分**45次精确matched reproduction逐项重查：model/optimizer/scheduler/norm/最佳epoch、逐epoch全部训练和验证指标、validation预测一致；日志duration_seconds为运行耗时不要求相同。正式与复现checkpoint都具有相同current source provenance。复用此baseline，不新增45baseline训练，不使用EMA结果选择。本轮45candidate runs完成，普通BA strict first-best及patience12与原规则一致，未多时点选模型。

## B. Performance

三seed的**指标**先在同split平均；15split为配对单位（不等于均值概率ensemble）。

|Metric|WSSL baseline|Activity-Scaled|Δ|正/零/负 splits|paired bootstrap 95% CI|dz|
|---|---|---|---|---|---|---|
|Accuracy|0.745604|0.747319|+0.001716|8/3/4|[-0.002998, +0.005799]|+0.192|
|BA|0.719644|0.721107|+0.001463|8/2/5|[-0.001351, +0.004273]|+0.255|
|AUROC|0.759553|0.759957|+0.000404|10/0/5|[-0.003310, +0.003671]|+0.056|
|Macro-F1|0.706589|0.707979|+0.001389|9/2/4|[-0.002203, +0.004613]|+0.198|
|PD Recall|0.782344|0.784454|+0.002110|8/3/4|[-0.008544, +0.011527]|+0.105|
|DD Recall|0.656945|0.657760|+0.000816|3/9/3|[-0.010992, +0.012545]|+0.035|


均值、中位数、split差SD、dz、Wilcoxon及BH在 `scaling_paired.csv`；BH仅exploratory，不额外进入保留闸门。不把两个CI是否跨零当成方法/类间直接差异。

## C. Stability

|Metric|within-split seed SD 原→新|3 seed-mean SD 原→新|15split SD 原→新|
|---|---|---|---|
|Accuracy|0.031205 → 0.031572|0.008650 → 0.007750|0.041218 → 0.038884|
|BA|0.021603 → 0.020449|0.006714 → 0.006975|0.034563 → 0.034361|
|AUROC|0.021741 → 0.022296|0.001812 → 0.003615|0.043921 → 0.042596|
|Macro-F1|0.025605 → 0.025180|0.004947 → 0.005065|0.039561 → 0.037563|
|PD Recall|0.065659 → 0.070207|0.025591 → 0.022413|0.061379 → 0.059023|
|DD Recall|0.078542 → 0.078828|0.035930 → 0.032682|0.058007 → 0.064101|


原/新三seed全部prediction一致比例：0.732973 / 0.731186；三seed两两prediction agreement：0.821982 / 0.820791。同split/同seed两方法prediction agreement先seed-first后split平均为 0.970878。score Spearman完整保存；agreement不是正确率。

最佳/停止epoch中位数：原10/22、新10/22；cap hits原0/45、新0/45。训练日志为dropout-active online loss，不冒充eval-mode train performance或不存在的完整EMA/intermediate checkpoints。

错误变化（每split先平均3seed，不是独立新增人数）：

|Disease|baseline errors/split|candidate errors/split|corrected|newly wrong|net|
|---|---|---|---|---|---|
|DD|10.444|10.422|0.467|0.444|0.022|
|PD|16.022|15.867|1.133|0.978|0.156|


## D. Learned activity scales

以下只在完整正式性能分析后生成。主SD/range为15个seed-first split值；另列45run范围和三个seed各15split均值。全部run/split g表完整保存。

|Activity|g整体均值|15split SD|15split范围|seed42/43/44 mean|45run范围|
|---|---|---|---|---|---|
|CrossArms|1.014659|0.006273|[1.005683,1.029784]|1.013899 / 1.014806 / 1.015273|[0.998031,1.039044]|
|DrinkGlas|1.009340|0.007540|[0.999740,1.021052]|1.013809 / 1.008168 / 1.006042|[0.992593,1.038395]|
|Entrainment|0.998922|0.002250|[0.994740,1.003335]|0.996630 / 0.998152 / 1.001985|[0.992592,1.014979]|
|HoldWeight|1.006438|0.006753|[0.998572,1.022041]|1.007490 / 1.005385 / 1.006437|[0.995271,1.024460]|
|LiftHold|1.003698|0.004435|[0.996331,1.011706]|1.004131 / 1.003748 / 1.003214|[0.993513,1.018909]|
|PointFinger|1.009949|0.003926|[1.003077,1.018286]|1.007756 / 1.009908 / 1.012182|[0.999360,1.026531]|
|Relaxed|1.007596|0.006179|[0.997508,1.018223]|1.008281 / 1.008417 / 1.006088|[0.996230,1.024380]|
|RelaxedTask|1.008995|0.005012|[0.999051,1.019832]|1.007990 / 1.009995 / 1.008999|[0.997945,1.026494]|
|StretchHold|1.002475|0.003657|[0.997152,1.009911]|0.999428 / 1.002795 / 1.005203|[0.993443,1.021004]|
|TouchIndex|1.011100|0.004889|[1.001152,1.019114]|1.005014 / 1.015439 / 1.012847|[0.996882,1.028873]|
|TouchNose|1.025430|0.008558|[1.005621,1.038657]|1.025868 / 1.020302 / 1.030120|[1.001663,1.060341]|


|Posthoc比较|Spearman rho|描述p|
|---|---|---|
|Phase A activity profile / ba|+0.8545|0.00081|
|Phase A activity profile / auroc|+0.7909|0.00375|
|TouchNose / pd_recall|-0.1776|0.52663|
|TouchNose / dd_recall|-0.0322|0.90921|
|TouchIndex / pd_recall|-0.0126|0.96458|
|TouchIndex / dd_recall|+0.3747|0.16879|


这些相关是posthoc描述。11activity不是11独立性能样本；g随projection/后续训练共同适配，不能视为临床因果importance。TouchNose/TouchIndex g与类Recall相关不能证明类特异机制，更不能据此再调g、删活动或改成22gate/MLP。

## E. Decision

**INCONCLUSIVE**。原冻结联合规则逐项：

|运行前闸门|结果|
|---|---|
|ba_mean_positive|PASS|
|ba_improved_10_of_15|FAIL|
|ba_ci_low_positive|FAIL|
|auroc_mean_non_decrease|PASS|
|macro_f1_non_decrease|PASS|
|pd_recall_drop_at_most_001|PASS|
|dd_recall_drop_at_most_001|PASS|
|ba_within_split_seed_sd_at_most_125x|PASS|
|auroc_within_split_seed_sd_at_most_125x|PASS|


INCONCLUSIVE / DO NOT RETAIN；保留原Frozen WSSL-STR。BA平均小幅收益证据不足以满足联合闸门，不启动任何参数化/LR/g范围/复杂gate补救，也不自动训练下一候选。固定within-split seedSD护栏通过，不代表所有波动指标都改善：AUROC的三个seed-mean SD由.001812升至.003615，不能只报告更有利的SD。

Phase A删除依赖异质性只支持提出对照，不能保证可学习g提高性能；性能证据优先，不因g排序符合预期而保留。

Boundary：outer data used=NO；threshold changed=NO；HarNet adapted=NO；training recipe changed=NO；additional hyperparameter search=NO。45正式新训练仅本candidate（另2step正确性/1batch smoke，非性能候选）。不使用Frequency、EMA、auxiliary、encoder适配、sampling或augmentation。不覆盖原formal资产。

所有结果 DEVELOPMENT-ONLY。15固定划分重叠、同subjects复用、反复研究选择带来乐观偏差；bootstrap仅描述repeated-development稳健性，不提供外部独立支持。seed-runs、subjects、pairs或bootstrap次数均不扩充样本量。FOE01/历史outer隔离保持；没有以其结果调设计或解释本候选。无全局recipe最优、病理因果或新外部泛化证明。代码/协议/报告/小型汇总Git提交推送并打标；数据/checkpoint/cache/逐subject预测/原始日志留server。

最终完整性：45selected checkpoints strict reload与参数/index审查、全部candidate artifact哈希、原source/cache/split/checkpoint/prediction/normalization及协议锁PASS。final_integrity.py仅事后审计，不改变冻结统计/decision，报告排版修正也不改变结果。Phase C fixed-denominator CE numerical audit本轮未启动：本次为INCONCLUSIVE，预设REJECT分支未触发；没有任何新的loss训练。
