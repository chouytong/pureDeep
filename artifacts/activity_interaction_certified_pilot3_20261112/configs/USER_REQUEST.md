# Certified L1 数据上的 Activity Token Interaction 三 Seed 小规模复核

## 一、任务与执行范围

请基于已完成的 Certified L1 数据重建与模型训练结果，
开展 Activity Token Interaction 的单一固定 split、三个 seed 复核。

本轮目标：

在保持 Phase 8 Activity Interaction 实现、结构和训练方式不变的条件下，
检验该方法在 Certified L1 数据及其对应 WSSL 表征上是否仍有收益，
尤其关注是否继续以降低 DD Recall 换取其他指标上升。

本轮不是：
- 第一次开展 Activity Interaction 三 seed。
- 全部15个split的正式复核。
- 新的Transformer结构搜索。
- 基于旧L1数据重复训练。
- 全模型端到端重新训练。
- 三模型概率集成。

正式新增训练预算：
1个固定split × seeds42/43/44 = 3个Activity Interaction训练run。

不自动扩大到45次。
不自动训练其他方法。
不因某一个seed结果较差而追加seed或更换split。

## 二、项目与来源

项目：
/home/zyt/deep_final

冻结主线：
/home/zyt/deep_final/mainline

Phase 8 Activity Interaction：
/home/zyt/deep_final/artifacts/wssl_guided_optimization_phase8_20261111

其正式来源提交：
4745549dabf34ccc89a740ad361a0410c6a93a4d

Certified L1 与新模型：
/home/zyt/deep_final/artifacts/l1_precision_rebuild_phase9_level2_20261112

其正式来源提交：
beb57167a7b8b54792f1efed672927396fe8294d

本轮独立目录：
/home/zyt/deep_final/artifacts/activity_interaction_certified_pilot3_20261112

首先核对当前服务器与GitHub版本。
如果main已有后续修改，记录差异，不覆盖用户修改，不混用不同版本。

只阅读本轮必要的：
- AGENTS.md及当前运行说明。
- Phase 8 PROTOCOL.md。
- Phase 8 ACTIVITY_INTERACTION_REPORT.md。
- Phase 8 scripts/interaction8.py及依赖文件。
- Certified L1 FINAL_REPORT.md。
- Certified L1 数据、normalization与checkpoint来源记录。
- 对应split和seed的实际配置。

不重复开展全项目历史审计或无关文献调研。

## 三、明确历史结果和本轮问题

Phase 8活动交互已完成15×3训练，旧数据结果：

Accuracy = 0.769251
BA = 0.725805
AUROC = 0.766245
Macro-F1 = 0.722707
PD Recall = 0.830861
DD Recall = 0.620749

这些是旧L1版本15个split的历史汇总，不能作为本轮单split的直接对照。

本轮主要比较：

AI-WSSL-CERTIFIED-L1
vs.
WSSL-CERTIFIED-L1

要求同：
- split。
- seed。
- Certified L1数据。
- normalization。
- 原HarNet特征。
- 标签与决策规则。

历史旧Activity Interaction只作为背景记录。
不得据本轮一个split的结果宣布超过历史15-split平均表现。

## 四、固定split与资产核验

本轮固定：
context0 / inner0
seeds = [42, 43, 44]

在新候选结果出现前，把该选择写入协议并冻结。
不得根据任何新旧性能选择split。

若真实manifest中不存在该split，标记BLOCKED并报告，
不能自动选择表现更好的其他split。

定位该split的：
- train subject IDs。
- validation subject IDs。
- excluded subject IDs。
- Certified L1信号版本。
- 已完成的train-only normalization。
- seed42/43/44对应的Certified WSSL checkpoint。
- 对应归档logits、概率及指标。
- 合法HarNet缓存。

核对所有必要hash和映射。

不重新拟合normalization。
不重新运行L1。
不重新训练三个WSSL基础模型。
不重新提取HarNet原始表示。

复用同seed的Certified WSSL基础模型：
seed42模块对应seed42基础模型；
seed43模块对应seed43基础模型；
seed44模块对应seed44基础模型。

这三个配对实例同时反映基础模型和新增模块的seed变化，
不能将其描述为仅隔离了新增模块初始化的随机性。

缺少正确来源时停止，不用旧L1 checkpoint替代。

## 五、重新导出Certified WSSL活动token

必须区分：

A. 原官方冻结HarNet raw Acc缓存：
来源核验通过后允许复用。

B. Phase 8旧WSSL监督模型产生的bilateral token：
不能作为本轮新数据版本的输入。

对三个Certified WSSL checkpoint分别进行合法特征导出：

- 只读取当前split的train和validation受试者。
- 使用已绑定的Certified信号与normalization。
- 模型eval、无梯度。
- 不更新任何参数或buffer。
- 只在真实长度下编码。
- 保持11活动、左右腕和六通道顺序。
- 导出实际bilateral representation。
- 按历史实现，预期形状为[N,11,258]，必须由源码及输出验证。
- 同时保存实际mask和必要的baseline logits。

分别保存train与validation角色和来源。
禁止将活动或腕侧当成独立患者。

用导出token恢复基础WSSL logits，
与该Certified checkpoint的归档输出进行核对。

容差依据原归档精度事先固定：
- 同一路径的零残差比较应尽可能逐位一致。
- 跨执行路径的误差明确报告。
- 不因复现失败临时放宽容差。

复现不通过，不开始新模块训练。

特征提取允许进行，但不算新增深度训练。
必须单独报告实际推理和特征导出数量。

## 六、严格复用Phase 8活动交互结构

以以下实际源码为依据：

artifacts/wssl_guided_optimization_phase8_20261111/scripts/interaction8.py

原实现的关键结构为：

- bilateral tokens：[B,11,258]。
- 258→64线性投影。
- 11项activity identity embedding。
- 单层4-head self-attention。
- 两次residual与LayerNorm。
- FFN：64→128→64。
- dropout=0.1。
- masked mean pooling。
- 64→258输出层。
- 输出层权重和bias初始化为0。
- 通过冻结的原classifier线性权重，
  将新增subject residual映射到logits。
- 原subject logits和STR logits保留作为基线输出。

最终应等价于：

new_logits = certified_baseline_logits + interaction_residual_logits

不要：
- 新增另一个独立分类头。
- 重复添加原classifier bias。
- 替换原STR路径。
- 改变attention层数、head数或维度。
- 添加gate。
- 叠加SupCon。
- 修改类别权重。
- 解冻原WSSL模块。

若源码与上述摘要有细节差异，以冻结源码为准并明确说明，
不得自行实现一个“类似的Transformer”冒充原方法。

本轮全部原Certified WSSL参数与buffer冻结。
只有新增ActivityInteraction模块接受优化器更新。

新增模块按42/43/44分别重新初始化。
不得加载旧Phase8已训练interaction权重继续微调。

## 七、训练前correctness tests

至少验证：

1. subject/label/split映射一致。
2. Certified数据与normalization匹配。
3. 三个token包分别绑定正确checkpoint。
4. token数量、维度、活动顺序正确。
5. mask语义与实际有效活动一致。
6. masked无效值不会影响有效输出。
7. 全部活动无效时明确拒绝。
8. 输出logits为[B,2]且有限。
9. 零初始化residual时，输出等于对应Certified baseline。
10. 原WSSL参数无梯度。
11. 原WSSL始终eval，参数和buffer前后hash不变。
12. 新模块能够获得有效梯度和优化器更新。
13. checkpoint恢复后预测一致。
14. 两次eval推理在规定容差内一致。
15. 不读取excluded预测或outer performance。
16. 完整网络中没有重复计算、重复相加原baseline logits。

注意：
零初始化输出层可能使第一步上游新增参数梯度为零。
不要把它直接当bug。
用预定的极少数合成smoke更新检验梯度传播，
随后重新初始化正式模型，不能把smoke状态带入正式训练。

## 八、固定训练recipe

直接复用Phase 8B实际执行recipe，特别是：

- batch size = 8。
- AdamW。
- lr = 2e-4。
- weight decay = 1e-4。
- 原betas。
- 原balanced CE。
- cosine scheduler。
- T_max = 50。
- eta_min = 1e-6。
- gradient clip = 5。
- max epochs = 50。
- patience = 12。
- validation BA严格改善时保存best。
- 原shuffle与seed设置。

实际执行前逐项核验冻结代码与配置。

注意Phase 8B使用token DataLoader，
不要直接把Phase 9全网络训练的worker等设置复制过来。
保留原方法执行设置，并记录本次精度、TF32、CUDA、
cuDNN及依赖版本。

DD为正类：
pDD > 0.5判DD；
相等时判PD。

不做阈值或温度拟合。

完整运行3个seed。
不得因为第一个seed不理想而改参数。
训练异常、失败和恢复均保留日志。

## 九、评价与统计

主表对每个seed分别列：

Certified WSSL baseline
Certified WSSL + Activity Interaction
两者差值

报告：
- Accuracy。
- BA。
- AUROC。
- Macro-F1。
- PD Recall。
- DD Recall。
- NLL。
- Brier。
- ECE。
- 混淆矩阵。
- 最佳epoch与实际训练epoch。
- 参数量。
- 训练耗时。
- 同边界的推理时间。

随后报告三个seed的均值、样本标准差及配对差值。
不挑最高seed作为最终结果。

不把三seed先平均概率后得到的ensemble指标混入主表。
本轮不开展集成候选。

所有指标沿用已完成报告的定义：
- Brier为两类平方误差之和。
- NLL为未加权proper log loss，不用训练weighted CE冒充。
- ECE沿用既定分箱。
- AUROC与原评估函数保持一致。

这是一个split，不是三个独立患者队列。
不能套用“10/15split提高”的正式保留门槛。
不能对3个seed给出貌似充分的显著性结论。

若补充患者bootstrap：
同一患者在三个seed和两种模型中使用相同重采样权重，
只能解释为固定已训练模型和本split条件下的描述性敏感性。
不能声称消除了validation选checkpoint及多轮开发偏差。

特别报告：
- BA变化是否主要来自PD Recall上升。
- DD Recall是否再次下降超过0.01。
- AUROC/F1是否同时改善。
- 不同seed的方向是否一致。

旧基础checkpoint和新增模块都在重复development框架内使用validation选epoch。
不要将本轮validation描述为未被选择过程触及的独立测试集，
也不要把这些token称为无监督或end-to-end OOF表示。

## 十、参数和推理成本

分别报告：

- 原Certified WSSL参数量。
- 新增interaction参数量。
- 本轮可训练参数量。
- 原冻结classifier-side参数量。
- HarNet冻结参数量。
- 完整部署系统的唯一总参数量。

不能只报告新增模块参数而省略基础模型。

计时至少提供公平的token→logits对照：
- 原读出。
- 原读出+新增模块。
- 相同resident token、batch、硬件、精度。
- CUDA同步、warm-up和重复次数明确。

如果只测缓存token下的读出：
必须明确排除了原始编码、L1、normalization、HarNet和IO。
不得写成完整模型单患者端到端延迟。
batch时间与单患者平均时间分别标注，不混用。

## 十一、实验判定

本轮只判定：

CORRECTNESS：
PASS / FAIL / BLOCKED

PILOT_STATUS：
COMPLETED / INCOMPLETE / BLOCKED

PERFORMANCE_PATTERN：
- 多项指标同向改善；
- PD/DD权衡仍存在；
- 结果混合；
- 未见改善。

不因为三个seed均值更高就自动RETAIN。
不修改正式模型指针。
不声称已完成Certified数据上的15×3复核。

必须回答：
1. 修复后的数据上该方法是否可正确执行？
2. 三个seed相对各自Certified baseline如何变化？
3. DD Recall下降问题是否再次出现？
4. 初步结果是否值得申请完整15×3复核？

后续正式比较如获批准，应使用原15个split。
本轮的3个run若来源、代码、配置完全一致，可作为其中一个split复用，
不必为了凑45次再次重训它们。

本轮不得自动开始其余42个run。

## 十二、产物与保护

独立生成：

PROTOCOL.md
README.md
reports/FINAL_REPORT.md
reports/CORRECTNESS_AND_PROVENANCE.md
reports/COST_AND_LATENCY.md

analysis/
    metrics_by_seed.csv
    paired_deltas_by_seed.csv
    mean_std_summary.csv
    parameter_counts.csv
    experiment_state.csv
    integrity.json

私有保存：
- checkpoint。
- bilateral token。
- subject IDs。
- 逐人logits和概率。
- 完整来源映射。
- 对应私有hash。

不覆盖：
- Phase8旧结果。
- Certified数据。
- Certified normalization。
- Certified WSSL/STR checkpoint。
- mainline。
- 当前正式模型指针。

按既有发布规范发布代码、协议和安全聚合统计，
不上传数据、权重、token或逐人结果。
执行远端回读后，才宣称发布完成。

完成3个正式run、报告与完整性检查后停止。