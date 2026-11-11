# Phase 6：Frozen WSSL-STR Representation Mechanism and Generalization Audit

你是一名具有丰富研究经验的计算机视觉、可穿戴 IMU 时序分析、深度学习、跨受试者泛化及帕金森病数字生物标志物研究人员。

当前任务是继续推进 PADS 数据集上的 PD（Parkinson's Disease）与 DD（Differential Diagnosis）受试者级二分类研究。

GitHub：
https://github.com/chouytong/pureDeep

最新正式起点：

161b904462e754a892c3d3d2e4f9ea68b33d18c0

服务器项目：

/home/zyt/deep_final

主线代码：

/home/zyt/deep_final/mainline

最新实验：

/home/zyt/deep_final/artifacts/wssl_complementarity_phase5_20261111

当前最佳模型：

Frozen WSSL-STR

正式指标：

Accuracy = 0.745604
BA = 0.719644
AUROC = 0.759553
Macro-F1 = 0.706589
PD Recall = 0.782344
DD Recall = 0.656945

本轮不是新模型搜索，而是研究当前最优模型的表征机制、适配行为和泛化证据。

## 一、必须先完整理解的历史结论

阅读：

- 最新 mainline README、AGENTS.md、CONTEXT_HANDOFF。
- Phase-3B、3C、3D、3E 正式报告。
- Phase 4 完整报告。
- WSSL contribution Phase A。
- Activity-scaled WSSL 报告。
- Patch-STR 完整报告。
- Phase 5 FINAL_REPORT.md。
- Phase 5 STATISTICAL_APPENDIX.md。
- Phase 5 CORRECTNESS_AND_PROVENANCE.md。

重点掌握：

1. Frozen WSSL-STR 是当前 retained single-model。
2. HarNet10 frozen external SSL prior 有稳定开发集增量。
3. Last-block adaptation AUROC 有正向信号，但 BA 没有通过原联合保留门槛。
4. Adapter、gated fusion、masked domain adaptation 均未保留。
5. Dual-reference relative 存在部分不同预测，但整体净救援为负。
6. WSSL 严格排错 pairs 中 relative 正确排序比例仅 0.416706。
7. Fusion 负截距造成 DD Recall 明显下降，正式 REJECT。
8. Local representation 没有表现出稳定增量。
9. 正温度校准只保留 NLL 诊断，不改变主模型。
10. CE 固定分母候选仍然 NOT_RUN_GATE_FAIL，不是性能负结果。

不得将历史失败方法重新命名后再次搜索。

## 二、本轮总体目标

只研究三个问题：

### H1：HarNet 适配是否产生了可重复的受试者级表征变化？

预测层面的异质性已经有证据：

43名受试者改善、46名恶化、301名不变。

但是尚未得到合法的 Frozen/Adapted 成对最终表征。

本轮尝试利用历史 checkpoint 完成这一缺口。

### H2：类条件参考覆盖距离是否具有不同的可靠性关系？

Phase 5 发现：

PD 高参考距离区域错误率更高，而 DD 的方向相反。

需要明确这种差异是否具有稳定的统计表现，是否受到活动、腕侧、context 或参考库规模影响。

不能把高距离直接解释为临床异常。

### H3：当前 Frozen WSSL-STR 是否具备真正的独立跨受试者泛化证据？

当前 15 splits 共享受试者。

历史 outer 信息接触及整表/整数组读取偏差已经永久披露。

本轮不得再将这些 development 结果包装成独立外部验证。

需要制定真正的独立验证协议。

---

## 三、Phase 6A：Frozen / Adapted 成对表征提取

这是本轮主要执行任务。

### 3.1 资产检查

定位全部45对：

- Frozen WSSL checkpoint。
- Last-block adapted WSSL checkpoint。
- 对应 split、seed、context。
- 相同 validation subject IDs。
- 原始 Acc 输入或合法的相同输入缓存。
- 对应预处理与 normalization。
- HarNet10 官方 pretrained checkpoint。

检查所有 SHA256。

不得根据文件名推断 checkpoint 对应关系。

必须验证：

- Frozen/Adapted 使用相同原始输入。
- 采样率、窗口长度、通道顺序一致。
- 活动和左右腕顺序一致。
- normalization 一致。
- subject ID 和 label 完全一致。
- 没有 outer-test 输入。
- 没有训练集与验证集混淆。

若任意一个条件不能满足，标记 BLOCKED。

### 3.2 成对特征提取

只有资产审计全部通过后，才允许执行。

对每一对 checkpoint，在完全相同的 validation input 上分别提取：

1. HarNet10 layer4 feature。
2. HarNet10 layer5 feature。
3. HarNet projected wrist residual。
4. 如果能够合法取得，提取最终 wrist representation。

每个受试者、活动、腕侧必须保留完整映射。

不要只保存最终聚合向量而丢弃 activity/wrist 结构。

本轮只进行 inference，不进行训练。

禁止访问 outer-test。

### 3.3 表征漂移分析

计算：

Cosine distance：

D_cos = 1 - cosine(z_frozen, z_adapted)

L2 displacement：

D_L2 = ||z_frozen - z_adapted||_2

Relative displacement：

D_rel = ||z_frozen - z_adapted||_2 / (||z_frozen||_2 + epsilon)

Representation norm：

||z_frozen||_2
||z_adapted||_2

同时报告不同层的变化。

需要注意：

不同层的维度、尺度和语义可能不同，不允许直接比较未经归一化的 L2 数值。

### 3.4 与预测变化的关系

使用同一 subject、同一 split、同一 seed、同一 activity 输入进行匹配。

计算：

- Frozen prediction。
- Adapted prediction。
- Frozen signed margin。
- Adapted signed margin。
- Margin change。
- Prediction flip。
- True-label probability change。
- Correct-to-wrong。
- Wrong-to-correct。

将受试者分为：

- improved。
- worsened。
- unchanged。

检查：

1. 三组的表征漂移是否不同。
2. PD/DD 是否有不同漂移分布。
3. layer4 和 layer5 哪一层变化更明显。
4. 漂移是否与 margin improvement 相关。
5. 漂移是否与 prediction flip 相关。

不能将相关性解释为因果。

不能根据 validation 的 improved subjects 训练新的门控模型。

### 3.5 Phase 6A 停止条件

如果：

- 输入无法严格对齐；
- checkpoint provenance 不完整；
- 特征提取位置不一致；
- 预处理不一致；
- 需要访问 outer 数据；
- 无法验证 subject ID；

则停止对应分析并标记 BLOCKED。

不得使用其他 fold 或其他 seed 的特征替代。

### 3.6 输出

- ADAPTATION_REPRESENTATION_REPORT.md
- analysis/paired_representation_integrity.csv
- analysis/representation_drift.csv
- analysis/representation_prediction_association.csv
- analysis/representation_layer_summary.csv
- analysis/phase6a_integrity.json

---

## 四、Phase 6B：参考覆盖距离的类条件可靠性

只使用 Phase 5 已有合法评分。

不重新训练模型。

不重新拟合 reference bank、Scaler 或 PCA。

### 4.1 复核原始覆盖距离

定义：

u(x) = min(d_PD(x), d_DD(x))

分别分析 PD 和 DD：

- high-u error rate。
- low-u error rate。
- high-u vs low-u prediction correctness。
- WSSL absolute margin。
- WSSL true-label signed margin。
- reference relative score。

必须使用 Phase 5 原有定义和划分规则。

不得根据本轮结果寻找新的最佳 cutoff。

### 4.2 检查可能的混杂因素

如果现有合法数据支持，分析：

- activity composition。
- wrist availability。
- context。
- reference bank subject count。
- subject appearance count。

重点检查 PD/DD 的差异是否可能由活动完成情况、参考样本覆盖范围或受试者重复造成。

对于无法控制的因素，明确列为未解决问题。

### 4.3 类条件交互

重点检验：

High-reference-distance × Disease-class

是否表现出稳定的方向差异。

使用受试者级配对统计和 cluster bootstrap。

不能只因为一类显著、另一类不显著就断言两类效应显著不同。

必须直接检验类间交互。

### 4.4 输出

- CLASS_CONDITIONAL_COVERAGE_REPORT.md
- analysis/coverage_class_interaction.csv
- analysis/coverage_subject_summary.csv
- analysis/coverage_context_sensitivity.csv
- analysis/phase6b_integrity.json

本轮不允许据此构建新的 anomaly gate 或 reject option。

---

## 五、Phase 6C：独立泛化验证协议设计

不假设已经拥有外部数据。

先调查是否有真正独立的可用数据来源。

可以考虑：

- 新医院队列。
- 不同采集时间的独立受试者队列。
- 不同设备采集的独立队列。
- 其他合法公开的 PD 手部运动 IMU 数据。

必须核查：

- 数据授权。
- 受试者重叠。
- 采集协议。
- 传感器型号。
- 活动定义。
- PD/DD 标签定义。
- DD 亚型组成。
- 可比性和缺失情况。

不能将不同任务的数据直接合并计算性能。

### 5.1 锁定当前模型

当前 Frozen WSSL-STR 作为唯一验证模型。

固定：

- checkpoint SHA256。
- 预处理。
- normalization。
- 模型权重。
- 决策阈值。
- 校准器使用规则。

校准器不能根据外部测试标签重新拟合。

### 5.2 独立验证指标

至少包括：

Accuracy
Balanced Accuracy
AUROC
Macro-F1
PD Recall
DD Recall
NLL
Brier
ECE

同时提供：

- 95% CI。
- PD/DD 分类别结果。
- 不同设备和采集条件结果。
- DD 来源亚型分析。
- 外部队列与 development 的差异。

没有实际数据就只完成协议，不伪造评估结果。

### 5.3 输出

- INDEPENDENT_VALIDATION_PROTOCOL.md
- DATASET_COMPATIBILITY_AUDIT.md
- analysis/external_data_inventory.csv
- analysis/external_validation_lock.json

不得自动读取、下载或使用未经授权的数据。

---

## 六、永久工程与统计约束

历史 outer 偏差必须永久保留。

本轮：

- 不读取 outer performance。
- 不新建 outer loader。
- 不将 outer 元数据转化为训练信息。
- 优先采用 subject ID allowlist。
- 禁止通过读取完整压缩缓存再筛选 ID 来宣称物理隔离。
- 不能宣称已经拥有系统级完全隔离能力。

对所有新提取资产记录：

- checkpoint SHA256。
- feature SHA256。
- source code SHA256。
- preprocessing hash。
- normalization hash。
- split ID。
- seed。
- context。
- subject IDs。
- fit/query/excluded IDs。

所有统计必须明确 estimand。

区分：

1. seed stability。
2. split stability。
3. subject-cluster uncertainty。
4. external generalization。

15个 development splits 不能被当成15个独立外部队列。

所有多重比较预先定义 BH family。

不能通过改变 bootstrap 口径、删除负结果或筛选子组改变保留结论。

---

## 七、本轮禁止的工作

深度训练预算：

0。

禁止：

1. 训练新的 WSSL 模型。
2. 训练新的 Adapter、LoRA、Gate。
3. 重新训练 HarNet。
4. 重新训练 Patch-STR。
5. 重新搜索 dual-reference fusion。
6. 重新搜索阈值。
7. 重新搜索 CE。
8. 使用错误患者名单训练模型。
9. 使用 validation label 选择新模型。
10. 访问 outer-test performance。
11. 修改冻结的 Frozen WSSL-STR。
12. 覆盖历史正式产物。

本轮允许：

- 只读资产审计。
- 已有 checkpoint 的合法成对推理。
- 特征导出。
- 表征漂移分析。
- 类条件统计。
- 外部数据兼容性审查。
- 独立验证协议设计。

如果执行过程中发现需要新训练，只能形成后续候选方案，不能自动启动。

---

## 八、最终研究判断

最终必须回答：

1. Last-block adaptation 是否改变了疾病相关的中间表示？
2. 表征漂移是否与改善或恶化相关？
3. 这些变化是否在 PD/DD 两类中存在不同表现？
4. 高 reference distance 为什么在 PD 和 DD 中表现出相反的错误率关系？
5. 该现象是否可能由活动分布、参考库覆盖或类别异质性解释？
6. 当前哪些结论有充分证据，哪些仍然 BLOCKED？
7. 当前是否出现值得开展新训练的研究假设？
8. 是否已经具备开展独立泛化验证的必要数据和协议？

最终给出：

- 完整实验状态表。
- 研究假设证据表。
- 新发现及其限制。
- 值得继续的研究方向。
- 应彻底停止的研究方向。
- 独立验证数据准备情况。
- 后续实验候选及停止条件。

所有文件独立保存在：

/home/zyt/deep_final/artifacts/wssl_representation_generalization_phase6_20261111/

不得修改：

/home/zyt/deep_final/mainline

不得覆盖历史正式实验。

完成后按现有 Git 发布规范发布代码、报告、协议和聚合统计。

不上传私有数据、checkpoint、cache、逐受试者预测及受保护 manifest。

本轮结束后停止，不自动启动 Phase 7。

**本轮目标是弄清楚当前 WSSL 模型为什么对某些受试者有效、对另一些受试者无效，并为独立泛化验证建立可信证据。不是继续堆叠模型模块。**