# Phase 7：Independent Cohort Validation and Class-Conditional Reliability Study

你现在是一名具有丰富研究经验的计算机视觉、可穿戴 IMU 时序分析、深度学习、跨受试者泛化及帕金森病数字生物标志物研究人员。

你需要基于当前 PURE-DEEP 项目，开展下一阶段研究。

本轮重点不是继续增加网络结构，而是解决当前研究最重要的两个问题：

1. 当前 Frozen WSSL-STR 能否在真正独立的 PD/DD 受试者队列中保持判别能力？
2. Phase 6 发现的 PD/DD 类条件参考覆盖度差异，是否具有可重复的统计规律？

同时，评估当前模型和研究协议是否已经具备开展独立临床验证的条件。

---

# 一、项目与正式实验状态

GitHub：

https://github.com/chouytong/pureDeep

最新正式提交：

6de6480fce87920a899181fa7c8586c01ce616f9

服务器项目：

/home/zyt/deep_final

主线代码：

/home/zyt/deep_final/mainline

Phase 6：

/home/zyt/deep_final/artifacts/wssl_representation_generalization_phase6_20261111

当前 retained single-model：

Frozen WSSL-STR

正式开发集历史指标：

Accuracy = 0.745604
BA = 0.719644
AUROC = 0.759553
Macro-F1 = 0.706589
PD Recall = 0.782344
DD Recall = 0.656945

当前外部验证锁定 checkpoint：

SHA256:
60b6083bcef9fcf5e7a468c03ed01a383eed4368e0fa261d6a7a3b1a72582d73

该 checkpoint 为：

seed42 / context0 / inner0

训练受试者数量：

205

注意：

该模型是一个按照预先规定的字典序选择的单一现有 checkpoint，不是45个模型的 ensemble，也不是使用全部390名受试者重新训练的模型。

外部验证必须以这个具体 checkpoint 为对象，不能将历史15-split平均指标当作该单一模型的外部性能。

---

# 二、先完整阅读历史实验

正式开始前，必须阅读：

- AGENTS.md
- README.md
- CONTEXT_HANDOFF
- 最新 mainline 源码
- Phase 3B、3C、3D、3E 报告
- Phase 4 完整报告
- Phase 5 FINAL_REPORT.md
- Phase 5 STATISTICAL_APPENDIX.md
- Phase 5 CORRECTNESS_AND_PROVENANCE.md
- Phase 6 FINAL_REPORT.md
- Phase 6 STATISTICAL_APPENDIX.md
- Phase 6 CORRECTNESS_AND_PROVENANCE.md
- Phase 6 INDEPENDENT_VALIDATION_PROTOCOL.md
- Phase 6 DATASET_COMPATIBILITY_AUDIT.md
- Phase 6 NEXT_STAGE_CANDIDATES.md

必须核对最新 GitHub main 与服务器主线是否一致。

不得根据历史对话或旧报告直接推断当前文件状态。

如果文件不存在，明确记录 MISSING。

如果来源无法证明，记录 BLOCKED。

不得伪造完整审计结果。

---

# 三、当前已经确认的研究结论

## 3.1 Frozen WSSL-STR

当前最佳 retained single-model。

不修改权重、预处理、normalization 和决策规则。

## 3.2 HarNet adaptation

45对 Frozen/Adapted checkpoint 已完成成对推理。

Layer5：

Cosine distance = 0.085248

Relative displacement = 0.396967

Frozen norm = 9.883820

Adapted norm = 7.942572

但是漂移幅度不能稳定解释预测改善。

因此，本轮不开展新的 Adapter、LoRA 或解冻训练。

## 3.3 Dual-reference

Relative AUROC = 0.735639。

低于 Frozen WSSL-STR 的0.759553。

整体平均救回9.51人、破坏16.78人，净损失7.27人。

WSSL严格排错pairs中的relative正确排序比例为0.416706。

当前融合方案正式REJECT。

不得重新搜索融合权重、阈值或门控结构。

## 3.4 Class-conditional coverage

Phase 6发现：

PD高参考距离错误率比低距离高。

DD高参考距离错误率比低距离低。

直接错误率交互：

0.439291

95% CI：

[0.326885, 0.557102]

BH q：

0.000183

患者cluster估计：

0.422650

95% CI：

[0.237459, 0.603781]

这是值得独立复核的发现，但尚未证明临床原因。

## 3.5 独立数据

当前没有核实到已经获得授权、且兼容PD/DD任务的独立队列。

已经审查的候选包括：

- PADS
- mPower
- BEAT-PD
- WATCH-PD

不能把PD/Healthy Control数据直接作为PD/DD外部测试数据。

## 3.6 永久保留的审计限制

历史outer性能文字接触、旧manifest整表读取、旧NPZ整数组读取及Phase 6公开搜索结果暴露个体元数据摘要等记录必须永久保留。

不能声称整个研究从未接触outer相关信息。

本轮不能重新访问旧outer性能，也不能将旧outer数据重新包装为独立验证。

---

# 四、Phase 7A：独立PD/DD数据集与临床队列调查

这是本轮最高优先级任务。

## 4.1 扩大数据源调查

不要局限于此前审查的四个数据集。

系统调查：

1. 公开的PD可穿戴IMU数据集。
2. PD与其他运动障碍鉴别数据集。
3. 双腕Acc/Gyro数据集。
4. 手部运动任务数据集。
5. 医院或研究机构的PD/DD临床队列。
6. 可以申请科研访问的受限数据集。
7. 具有独立受试者和长期随访信息的队列。

优先查找：

- SCI Q1/Q2相关研究。
- IEEE TBME。
- IEEE JBHI。
- Movement Disorders。
- npj Parkinson's Disease。
- Nature及其相关期刊。
- CCF B及以上相关传感器、时序表征研究。

使用官方数据门户、原始论文、DOI、PhysioNet、Synapse及机构正式发布页面。

不要仅凭二手综述判断数据可用性。

## 4.2 建立数据集兼容性表

每个候选数据集至少核查：

| 信息 | 要求 |
|---|---|
| 数据集名称 | 正式名称 |
| 数据来源 | 官方入口 |
| 论文 | 原始论文 |
| PD人数 | 可核验数据 |
| DD人数 | 可核验数据 |
| DD具体诊断 | ET、MS、非典型帕金森综合征等 |
| Healthy Control | 单独记录，不得合并到DD |
| IMU类型 | Accelerometer/Gyroscope |
| 腕侧 | 单腕或双腕 |
| 采样率 | Hz |
| 活动 | 具体任务 |
| 数据格式 | 可否映射到当前模型 |
| 数据授权 | Open/Controlled/Proposal |
| 受试者重叠 | 是否能够安全核查 |
| 当前状态 | Compatible/Partial/Blocked/Unavailable |

重点区分：

A. 真正兼容PD/DD受试者级分类的数据。

B. 只有PD/Healthy Control的数据。

C. 只有PD严重程度或连续量表的数据。

D. 只有手部运动识别或HAR数据。

E. 具有相关传感器但没有可用临床标签的数据。

不能将B、C、D、E类包装成A类。

## 4.3 评估外部数据的优先级

建立数据源评分：

- 标签兼容性：30%
- 传感器与活动兼容性：25%
- 独立受试者可信度：20%
- 数据访问可行性：15%
- 样本量及DD亚型信息：10%

分数仅用于数据准备优先级，不得用于模型性能比较。

对于缺失信息，不得自行赋予乐观分数。

## 4.4 数据访问边界

本轮只进行公开信息调查和已有授权范围内的文件审计。

禁止：

- 未经授权下载受限个体数据。
- 自动申请数据访问。
- 使用用户身份签署协议。
- 联系研究人员或发送邮件。
- 下载外部个体级传感器数据。
- 将公开的个人元数据行保存到项目中。

如果发现需要申请的数据，提供官方申请入口、申请条件和需要准备的材料，但等待用户明确授权。

---

# 五、Phase 7B：冻结模型的外部推理工程准备

当前外部模型和评价规则已经锁定。

不要重新选择 checkpoint。

## 5.1 验证锁定信息

核对：

Checkpoint SHA256：
60b6083bcef9fcf5e7a468c03ed01a383eed4368e0fa261d6a7a3b1a72582d73

Normalization SHA256：
ad8e617bce3ea20ec01eddafd6e4f47cdde76fe47c17ea938a903485e5144187

Canonical config：
95a027d075f7aa7c13d3efc49b3d73c3a3b959e3e2eed99f9da9c5bab5d62ca8

Preprocessing config：
1ced7052705192511da4d89d1a066589cede62b658bb852ff5769d20551bbe6c

Official HarNet10：
c64f9135d99e2dcdfc9ae7cc0672f2bcc438df9ceb8215665882f92cddd162a6

核对所有哈希与实际文件。

任何不匹配都停止。

## 5.2 外部数据输入适配器

当前模型需要：

- 11项活动。
- 双腕数据。
- Acc XYZ。
- Gyro XYZ。
- 对应采样率和单位。
- 原始预处理。
- 真实活动长度。

检查外部数据是否能进行确定性格式映射。

允许：

- 通道重命名。
- 单位换算。
- 采样率确定性转换。
- 已知时间戳格式转换。
- 无标签的确定性数据整理。

禁止：

- 为缺失活动伪造数据。
- 为缺失腕侧复制另一只手的数据。
- 将单腕数据包装成双腕。
- 将健康对照重新标记为DD。
- 根据外部标签修改输入。
- 使用外部测试标签重新拟合normalization。
- 为了适配而修改模型结构。

如果数据缺少必要的传感器或活动，明确标记PARTIAL或INCOMPATIBLE。

## 5.3 Synthetic correctness tests

在没有外部真实数据时，使用合成数据测试：

1. 输入维度。
2. 通道顺序。
3. 采样率转换。
4. 单位换算。
5. 活动缺失处理。
6. 腕侧缺失处理。
7. 真实长度。
8. 最后不足batch。
9. 模型eval模式。
10. BatchNorm状态不更新。
11. 确定性推理。
12. 模型输入不变性。
13. 输出logits有限性。
14. SHA与配置一致性。

合成测试通过不等于真实外部数据适配已经通过。

## 5.4 锁定模型的输出规则

固定：

- Frozen WSSL-STR。
- 原normalization。
- 原预处理。
- 原argmax决策。
- DD概率大于0.5判定DD。
- 相等时判定PD。
- Calibrator = NONE。

禁止使用外部标签调整温度、阈值或分类器。

---

# 六、Phase 7C：独立验证评价方案

这一阶段先完成协议，不假设已经拥有外部数据。

未来获得合格数据后，执行以下指标：

### 主要分类指标

- Accuracy
- Balanced Accuracy
- AUROC
- Macro-F1
- PD Recall
- DD Recall

### 概率可靠性

- NLL
- Brier Score
- ECE
- Classwise ECE
- PD/DD reliability curve

### 泛化分析

- 不同设备。
- 不同采集环境。
- 不同活动。
- 不同DD来源亚型。
- 不同年龄与性别分层（数据允许时）。
- 不同临床诊断确定性。
- 缺失数据情况。

### 统计

- Subject-level stratified bootstrap。
- 95% confidence interval。
- 预先固定的主要检验。
- 多重比较校正。
- 样本量不足时明确NA。

不得将同一个受试者的多次记录当成多个独立患者。

不得根据测试结果选择表现最好的亚组。

---

# 七、Phase 7D：类条件参考覆盖方法的独立复核准备

Phase 5和Phase 6的结果提示：

高参考距离对PD与DD的错误率影响方向相反。

但Phase 5缺少完整的逐人reference OOF分数及Scaler/PCA/bank状态。

因此，本轮先进行算法来源恢复审计。

## 7.1 原始reference算法审计

寻找：

- Reference bank构建代码。
- PD/DD bank来源。
- Scaler。
- PCA。
- 特征归一化。
- kNN或其他距离计算。
- 距离聚合。
- q25/q75 cutoff。
- 每个split的fit/query/excluded IDs。
- OOF逐人分数。
- 创建时源码及状态hash。

将结果分为：

COMPLETE：
可以完整恢复。

PARTIAL：
有部分代码和参数，但缺少关键状态。

MISSING：
不存在可验证资产。

BLOCKED：
来源或受试者隔离无法证明。

## 7.2 禁止伪造原方法复现

如果原Scaler、PCA或reference bank向量缺失：

不得声称能够精确复现Phase 5 reference pipeline。

可以提出新的、独立的reference protocol，但必须作为新研究方法重新登记。

不能将新方法结果与旧方法结果直接混用。

## 7.3 未来独立验证的固定假设

主要假设：

H1：PD与DD的参考覆盖距离和分类错误率存在不同的类条件关系。

H2：这种差异在独立临床队列中仍然存在。

H3：这种差异可能与DD亚型异质性、运动任务质量或参考空间几何有关。

主要检验：

Disease class × Reference distance interaction。

不允许先观察新队列结果再寻找最佳cutoff。

本轮不训练 anomaly gate，不训练拒绝分类器，不训练新的融合网络。

---

# 八、Phase 7 的工程要求

独立目录：

/home/zyt/deep_final/artifacts/independent_validation_phase7_20261111/

不修改：

/home/zyt/deep_final/mainline

不覆盖：

- Frozen WSSL-STR正式checkpoint。
- Phase 3历史结果。
- Phase 4历史结果。
- Phase 5历史结果。
- Phase 6历史结果。

每项新产物记录：

- Source SHA256。
- Config SHA256。
- Checkpoint SHA256。
- Normalization SHA256。
- Preprocessing SHA256。
- 数据来源。
- 数据授权状态。
- 受试者身份核查状态。
- 运行日志。
- Correctness tests。
- Fit/query/excluded IDs（私有保存）。

公开发布：

- 代码。
- 协议。
- 数据集兼容性表。
- 聚合统计。
- 审计报告。

不得上传：

- 原始传感器数据。
- 受试者ID。
- 逐患者预测。
- checkpoint。
- 私有特征。
- 受保护manifest。
- 未获授权的个体数据。

永久保留历史outer接触和数据访问边界披露。

---

# 九、本轮实验预算

新增深度训练：

0

新增模型优化器更新：

0

新增分类器训练：

0

新增温度或阈值拟合：

0

新增reference Scaler/PCA拟合：

0

外部真实数据下载：

0

未经授权的数据访问：

0

允许：

- 官方论文和数据门户调查。
- 公开元数据审查。
- 已有模型资产核对。
- 输入适配器开发。
- 合成数据correctness测试。
- 外部验证协议编制。
- Reference算法来源审计。
- 已有聚合结果的必要复核。

如果确实需要下载、申请或访问数据，必须先报告具体来源、授权要求和预期用途，等待用户明确批准。

如果需要新的训练或拟合才能完成某项工作，标记NOT_AUTHORIZED，不自行启动。

---

# 十、最终报告必须回答的问题

1. 是否发现真正兼容PD/DD的独立队列？
2. 哪个数据集最有可能成为外部验证来源？
3. 它与PADS在标签、活动、传感器和受试者方面有哪些差异？
4. 当前Frozen WSSL-STR能否直接应用？
5. 是否需要格式适配器？哪些适配会改变模型输入分布？
6. 当前锁定checkpoint和normalization是否完整可复现？
7. 原reference算法是否能够完整恢复？
8. Phase 6的PD/DD类条件交互是否有资格进入独立验证？
9. 哪些研究结论已经有充分development证据？
10. 哪些结论仍然缺少独立临床证据？
11. 当前是否有任何充分理由重新开展深度训练？

最终生成：

- FINAL_REPORT.md
- DATASET_DISCOVERY_REPORT.md
- DATASET_COMPATIBILITY_MATRIX.csv
- MODEL_DEPLOYMENT_AUDIT.md
- EXTERNAL_INPUT_ADAPTER_REPORT.md
- INDEPENDENT_VALIDATION_PROTOCOL.md
- REFERENCE_PIPELINE_PROVENANCE.md
- NEXT_STAGE_RESEARCH_PLAN.md
- CORRECTNESS_AND_PROVENANCE.md
- analysis/phase7_integrity.json

最后给出：

1. 数据集候选优先级。
2. 独立验证准备程度。
3. 当前模型适用范围。
4. 类条件覆盖研究的证据等级。
5. 后续研究方向优先级。
6. 每个方向的研究假设、实验成本和停止条件。

不要为了让报告看起来完整而虚构数据集、样本量、授权、性能或论文结论。

本轮完成后，发布代码、协议、报告和聚合统计至GitHub，保留私有资产。

完成后停止，不自动启动Phase 8。

**本轮核心目标：为当前 Frozen WSSL-STR 建立真正独立的 PD–DD 泛化验证条件，并将 Phase 6 的类条件覆盖发现转化为可独立证伪的研究假设，而不是继续进行同队列网络结构搜索。**