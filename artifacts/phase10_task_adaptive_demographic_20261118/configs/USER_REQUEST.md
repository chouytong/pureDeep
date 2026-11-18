# Phase 10：Task-Adaptive SSL Fine-tuning and Demographic-Aware PD/DD Classification

## 一、研究角色与核心任务

你是一名具有丰富研究经验的计算机视觉、可穿戴 IMU 时序分析、Self-Supervised Learning、迁移学习、医学机器学习和帕金森病数字生物标志物研究人员。

请基于 PURE-DEEP 项目开展下一阶段研究。

研究任务：

基于 PADS 数据集的双腕 IMU 传感器数据，完成受试者级 PD（Parkinson's Disease）与 DD（Differential Diagnosis）二分类。

本轮研究重点：

1. 研究 Frozen HarNet10 是否存在进一步的下游任务适配空间。
2. 研究人口统计学信息能否为 WSSL-STR 提供独立的辅助判别信息。
3. 分析当前 PD/DD 判别边界和 DD 亚型错误分布。
4. 研究 DD-aware cost-sensitive optimization 是否能够改善类别判别平衡。
5. 分析固定0.5阈值是否适合当前模型的实际概率分布。

本轮不能直接将多个改动组合成一个模型。

必须采用单变量、阶段化、可复现的实验设计。

最终目标：

**在 Certified L1 数据版本上，进一步改善 Frozen WSSL-STR 的 BA、AUROC、Macro-F1 和 DD Recall，同时不明显牺牲 PD Recall，并保持受试者级泛化稳定性。在服务器上面执行实验后，更新实验汇总的csv文件，并及时同步至GitHub**

---

# 二、项目位置与当前正式基线

GitHub：

[https://github.com/chouytong/pureDeep](https://github.com/chouytong/pureDeep)

项目：

/home/zyt/deep_final

当前模型由：

- STR Wrist Encoder
- Frozen HarNet10
- Wrist Residual Fusion
- Bilateral Fusion
- Activity Fusion
- Subject Path
- Structured Residual Path

组成。

当前正式数据：

Certified L1。

当前正式 normalization：

15份inner-train-only normalization。

当前正式训练协议：

15 subject-level splits × 3 seeds。

Seeds：

42、43、44。

必须从最新GitHub main和服务器实际读取当前状态。

如果main已经出现新的提交：

- 记录SHA。
- 检查与上述依据版本的差异。
- 不覆盖用户的后续修改。
- 不混用不同版本的checkpoint和数据。

---

# 三、历史实验结论与本轮约束

开始之前，必须阅读：

- AGENTS.md
- README.md
- CONTEXT_HANDOFF
- Phase 3B/3C/3D/3E
- Phase 4
- Phase 5
- Phase 6
- Phase 7
- Phase 8
- Phase 9 Level 2 FINAL_REPORT.md
- Phase 9 WSSL_RETRAIN_REPORT.md
- Phase 9 STR_RETRAIN_REPORT.md
- Phase 9 SSL_INCREMENT_REVALIDATION_REPORT.md
- Phase 9 NORMALIZATION_PROVENANCE_REPORT.md
- Phase 9 CORRECTNESS_AND_PROVENANCE.md

重点恢复：

1. Frozen WSSL-STR真实架构。
2. HarNet10真实网络层级和block定义。
3. Phase 3C历史Layer5解冻的实际配置。
4. Phase 3C历史训练日志和指标。
5. 当前Certified WSSL checkpoint。
6. 当前15个subject-level splits。
7. 每个split对应的三个seed。
8. Balanced CE实际实现。
9. 当前类别权重计算方式。
10. PADS人口统计学原始字段及其来源。
11. 当前DD subtype metadata。
12. 现有validation预测和OOF预测的来源。

禁止只根据历史对话推断。

所有重要参数必须从实际源码、配置和日志恢复。

## 3.1 已有历史实验必须纳入判断

### HarNet adaptation

历史Layer5 Adaptation：

BA = 0.7244
AUROC = 0.7689

高于旧Frozen WSSL的点估计，但没有通过联合保留标准。

因此：

本轮部分解冻不是首次尝试HarNet适配。

必须说明新实验与旧Layer5 Adaptation的区别：

- 新实验使用Certified L1。
- 使用新版本normalization。
- 重新建立WSSL下游训练状态。
- HarNet输入保持原始Raw Acc路径。
- 使用独立的HarNet低学习率。

不能把旧Layer5结果当作新实验的性能结果。

### Class imbalance

历史已经尝试：

- Balanced CE。
- Sampling compensation。
- Focal Loss。
- DD auxiliary classification。
- Activity Interaction。
- Subject-level SupCon。

这些方法均没有建立稳定、全面的分类收益。

因此，本轮DD-aware Loss必须与历史方法有明确区别。

### Activity Interaction

Certified L1下的三seed pilot：

平均：

ΔBA = +0.004795
ΔAUROC = -0.004795
ΔPD Recall = +0.117117
ΔDD Recall = -0.107527

三个seed的DD Recall均下降。

当前不继续扩大相同Activity Interaction方案。

它仅作为历史候选，不作为本轮初始化模型。

---

# 四、本轮实验总目录与阶段顺序

独立目录：

/home/zyt/deep_final/artifacts/phase10_task_adaptive_demographic_20261118/

结构：

phase10/
audit/
phase10a_diagnostics/
phase10b_partial_unfreezing/
phase10c_demographic_fusion/
phase10d_dd_optimization/
reports/
configs/
scripts/
tests/
logs/
analysis/
private/

严格执行：

Phase 10A
↓
Phase 10B
↓
Phase 10C
↓
Phase 10D
↓
Phase 10 Final Comparison

每一阶段先完成报告，再进入下一阶段。

允许按下文规定的pilot门槛自动决定是否扩展完整15×3。

如果某阶段没有通过门槛：

- 保留负结果。
- 不扩大该候选训练。
- 继续执行其他独立且符合条件的研究方向。
- 不通过修改参数来挽救单次结果。

本轮不得访问outer-test。

---

# 五、Phase 10A：PD/DD判别边界、DD亚型与人口统计学诊断

本阶段不进行新的深度模型训练。

目的是确定当前模型的性能瓶颈，并为后续实验提供证据。

## 5.1 Logit Margin Analysis

对每个subject计算：

s(x) = z_DD - z_PD

定义真实标签方向signed margin：

m(x) = (2y - 1) \* s(x)

其中：
y=1表示DD；
y=0表示PD。

正margin代表预测方向与真实标签一致。

分别分析：

- PD margin分布。
- DD margin分布。
- 正确预测与错误预测。
- 高置信错误。
- 接近决策边界的样本。
- 不同seed的margin稳定性。

报告：

- Mean。
- Median。
- SD。
- IQR。
- Quantiles。
- Margin histogram。
- Class-conditional density。

不能只展示平均值。

## 5.2 Error Transition Analysis

使用：

- WSSL-CERTIFIED-L1。
- STR-CERTIFIED-L1。
- Phase 8 Activity Interaction历史结果（仅作历史对照，不混入Certified主结果）。

分析：

- PD错误。
- DD错误。
- 不同模型之间的预测翻转。
- 正确转错误。
- 错误转正确。
- 高置信错误。
- 不同seed的一致性。

如果历史逐人预测来源不完整，不重新读取未经授权的整队列压缩缓存。

## 5.3 DD subtype analysis

读取实际合法的DD来源标签。

分别检查：

- Other Movement Disorders。
- Essential Tremor。
- Atypical Parkinsonism。
- Multiple Sclerosis。

核对：

- 每个类别实际人数。
- 各split内训练与validation数量。
- 预测错误率。
- Recall。
- Margin分布。
- 置信区间。

样本太少时：

- 明确标记NA。
- 不强行做显著性检验。
- 不把宽置信区间解释成没有差异。
- 不将粗来源标签称为精确临床亚型。

特别研究：

DD错误是否主要集中在某一个来源类别？

如果无法可靠映射标签，停止亚型分析。

## 5.4 Threshold diagnostic

当前主规则：

pDD > 0.5 判定DD。

本阶段先进行描述性诊断：

- DD Recall vs threshold。
- PD Recall vs threshold。
- BA vs threshold。
- Macro-F1 vs threshold。
- Precision-Recall curve。
- Confusion matrix at different thresholds。

注意：

Validation threshold curve只能用于展示操作点变化。

禁止根据validation结果选择最优阈值后，再将同一个validation上的指标称为最终测试成绩。

如果希望研究一个可部署的新阈值：

必须从对应inner-train中的OOF预测确定。
然后锁定阈值，应用于validation。
所有OOF预测必须具有完整来源和严格subject-level separation。

如果没有合法OOF预测，标记BLOCKED。

本阶段不修改正式0.5决策规则。

## 5.5 Demographic metadata audit

核查PADS真实可用字段。

优先考虑：

- Age at assessment。
- Sex。
- Height。
- Weight。

但必须从实际数据字典确认字段定义。

禁止从subject ID推断人口统计学信息。

禁止使用：

- Diagnosis comment。
- Medication。
- Disease duration。
- Clinical diagnosis text。
- 与标签直接关联的问卷答案。
- 未来信息。
- 任何可能直接泄漏PD/DD标签的字段。

检查：

- 缺失比例。
- 数据来源。
- 单位。
- 时间点。
- 取值范围。
- PD/DD分布。
- DD subtype分布。
- Device与demographics的关联。

尤其关注：
年龄、性别和设备型号是否存在类别混杂。

## 5.6 Demographic-only baseline

在数据字段有效且有授权的情况下，进行一个轻量Logistic Regression诊断。

输入仅为经过确认的人口统计学特征。

所有：

- Imputation。
- Encoding。
- Standardization。
- Feature selection。

只能在inner-train拟合。

Validation只做transform和predict。

目的：

判断人口统计学本身具有多大的分类能力。

这个模型只是诊断baseline，不是本轮正式候选主模型。

如果只有少量有效记录或字段缺失严重，明确报告限制。

## 5.7 Phase 10A输出

生成：

PHASE10A_DIAGNOSTIC_REPORT.md
MARGIN_ANALYSIS.md
DD_SUBTYPE_ERROR_ANALYSIS.md
THRESHOLD_DIAGNOSTIC.md
DEMOGRAPHIC_METADATA_AUDIT.md
DEMOGRAPHIC_ONLY_REPORT.md

以及：

analysis/margin_metrics.csv
analysis/error_transition.csv
analysis/dd_subtype_metrics.csv
analysis/threshold_curves.csv
analysis/demographic_distribution.csv
analysis/demographic_only_metrics.csv
analysis/phase10a_integrity.json

完成后形成明确结论：

1. 当前DD错误主要集中在哪里？
2. 是否存在明显高置信DD错误？
3. 是否存在可用的OOF阈值诊断？
4. 人口统计学字段是否完整且可信？
5. 人口统计学与PD/DD是否存在明显队列混杂？
6. 是否有充分依据开展10B、10C和10D？

---

# 六、Phase 10B：HarNet10部分解冻与分层微调

这是本轮第一个重点模型实验。

## 6.1 研究假设

H1：

Frozen HarNet10仍然保留了部分未被下游PD/DD任务充分利用的运动表示。

H2：

仅解冻最后一个实际网络block，使用较低学习率，可能比完全冻结获得更好的PD/DD判别能力。

H3：

适度解冻能够提高AUROC和DD Recall，同时避免全网络微调的过拟合。

这些是假设，不是预先成立的结论。

## 6.2 首先审计HarNet10真实结构

检查：

- HarNet10所有stage/block。
- 每个block的输入输出维度。
- 卷积层。
- Normalization层。
- Residual connection。
- 最终输出层。
- BatchNorm running statistics。
- 原预训练权重来源。
- 原预训练配置。

不能直接根据Layer5名称假设最后两个block的结构。

明确：

- Last block。
- Last two blocks。
- 冻结部分。
- 可训练部分。

并绘制真实网络结构图。

## 6.3 候选实验

只预先登记两个配置：

FT-L1：
仅解冻HarNet10最后一个实际block。

FT-L2：
解冻HarNet10最后两个实际block。

两者使用相同backbone learning rate：

1e-5。

分类器侧：

2e-4。

如果Phase 3C历史实际使用的Layer5解冻范围与FT-L1相同，必须明确说明两者的区别主要是Certified L1数据版本和重新训练的下游模型状态，而不是宣称网络结构创新。

## 6.4 训练设置

保持：

- Certified L1。
- 原15个subject-level splits。
- Seeds 42、43、44。
- 原WSSL架构。
- 原STR wrist encoder。
- 原bilateral fusion。
- 原activity fusion。
- 原subject-level readout。
- 原balanced CE。
- 原数据增强。
- 原batch size。
- 原训练epoch和checkpoint selection。

只改变HarNet的解冻范围和backbone学习率。

HarNet初始化必须来自原始官方预训练checkpoint。

禁止从旧Adapted checkpoint继续训练。

禁止将旧Layer5 checkpoint当作新实验初始化。

## 6.5 BatchNorm与归一化

必须检查HarNet内部BatchNorm。

小样本和batch size=8条件下，不应默认允许running statistics随训练变化。

第一版固定：

- BatchNorm running mean/variance冻结。
- BatchNorm affine参数冻结。
- 只允许指定卷积和其他明确指定的block参数更新。

如果HarNet实际使用其他Normalization，按实际结构处理并记录。

不能因为冻结BN就改变其余预训练状态。

## 6.6 Pilot训练

先在context0/inner0上运行：

FT-L1：
seeds 42、43、44。

共3次训练。

然后判断是否值得扩大。

Pilot只用于筛选资源投入，不是完整性能验证。

预先登记资源门槛：

- 平均ΔBA > 0。
- 至少2/3 seeds的BA改善。
- 平均AUROC不能明显下降。
- 平均DD Recall下降不超过0.01。
- 没有明显的训练不稳定或过拟合。
- correctness全部通过。

不能因为单个seed最高就扩展。

如果FT-L1通过pilot门槛：

执行完整15 splits × 3 seeds。

如果FT-L1未通过：

不自动继续FT-L2。

先完成FT-L1报告，再检查是否有明确证据支持增加解冻深度。

如果有证据，再对FT-L2开展独立3-seed pilot。

不得同时启动两种完整训练。

## 6.7 完整实验

通过pilot后，执行15×3。

每个split/seed：

- 原HarNet官方权重重新初始化。
- 不继承其他seed或split的微调权重。
- 对应inner-train normalization。
- backbone LR=1e-5。
- classifier LR=2e-4。
- 不改变其他训练策略。

记录每个参数组的：

- 参数量。
- 梯度。
- 更新前后norm。
- 学习率。
- 训练耗时。
- 最佳epoch。
- validation曲线。

## 6.8 评价

与同split、同seed的WSSL-CERTIFIED-L1比较：

Accuracy
BA
AUROC
Macro-F1
PD Recall
DD Recall
NLL
Brier
ECE

重点研究：

- 是否提高DD Recall。
- 是否改善PD/DD margin。
- 是否提高AUROC。
- 是否发生过拟合。
- 不同解冻block的差异。

输出：

PHASE10B_UNFREEZING_REPORT.md
analysis/unfreezing_metrics.csv
analysis/unfreezing_parameter_groups.csv
analysis/unfreezing_gradient_summary.csv
analysis/unfreezing_paired_statistics.csv
analysis/unfreezing_integrity.json

---

# 七、Phase 10C：Demographic Fusion

这是本轮第二个重点方向。

只有Phase 10A确认存在可信、合法、可用的人口统计学字段后，才开展。

## 7.1 研究假设

H1：

年龄、性别、身高和体重包含纯IMU表征未充分利用的辅助信息。

H2：

在保留Frozen WSSL-STR的基础上，独立的人口统计学分支可能改善PD/DD边界样本的判别。

H3：

人口统计学增量应在控制设备与数据缺失差异后仍具有一定稳定性。

## 7.2 固定输入字段

首轮只允许：

- Age at assessment。
- Sex。
- Height。
- Weight。

实际字段必须从数据字典核实。

如果缺少某个字段：

- 记录缺失。
- 不自行推断。
- 不从ID推导。
- 不伪造。

如果无法获得可信人口统计学数据：
标记BLOCKED，不启动模型训练。

## 7.3 预处理

所有处理只能在inner-train拟合。

连续变量：

- Median imputation。
- Train-only standardization。

分类变量：

- 固定类别映射。
- Train-only imputation。
- 不使用validation频率确定类别。

缺失情况：

- 先分析缺失比例及类别分布。
- 首轮不自动把缺失模式作为预测特征。
- 不用DD/PD标签决定缺失值填补。

人口统计学变量必须与当前subject IDs严格匹配。

同一个subject的所有活动共用一份人口统计学记录。

## 7.4 模型设计

建立：

WSSL-DEMO-FUSION

保持原Frozen WSSL所有encoder和原始logit不变。

从原WSSL获取：

Subject embedding：

e ∈ R^258

Demographic branch：

输入：
Age、Sex、Height、Weight。

结构：

4 → 8 → 16

输出：

d ∈ R^16

融合：

[z_subject + W_demo d] + z_STR

其中：

- 原subject classifier权重W保持冻结。
- 原subject classifier bias保持冻结。
- W_demo为新增16→2线性映射。
- W_demo初始化为0。
- STR residual logits保持原样。
- 新增demographic MLP可训练。

该设计等价于将16维人口统计学表示拼接到原258维subject embedding后，保持原258维分类权重不变，只学习新增16维的判别贡献。

这样在初始化时：

z_final = z_baseline

避免一开始就改变原模型预测。

只训练：

- Demographic MLP。
- 新增W_demo。

冻结：

- HarNet10。
- STR wrist encoder。
- 原始WSSL投影。
- Bilateral fusion。
- Activity fusion。
- 原subject classifier。
- 原STR classifier。

## 7.5 对照组

D0：
Demographics-only Logistic Regression。

D1：
WSSL-CERTIFIED-L1。

D2：
WSSL-DEMO-FUSION。

三个结果必须分开报告。

不能只报告D2相对于WSSL的结果而不报告D0。

## 7.6 Pilot

固定context0/inner0。

seeds 42、43、44。

共3个新增训练run。

固定：

- 新分支结构。
- CE训练目标。
- 原WSSL classifier logits。
- 原normalization原则。
- 不使用DD额外加权。
- threshold=0.5。

Pilot门槛：

- 平均ΔBA > 0。
- 至少2/3 seeds的BA改善。
- AUROC平均不下降。
- DD Recall平均下降不超过0.01。
- PD Recall平均下降不超过0.01。
- demographics-only不能表现出明显的单独主导效应而fusion没有额外增量。
- 无明显设备组或人口统计学组的极端不稳定。
- correctness全部通过。

如果不满足：
不执行完整15×3。

如果满足：
执行完整15×3。

注意：
pilot使用的split也属于开发数据，完整结果只能称为development replication，不能称为独立确认。

## 7.7 混杂和稳健性分析

至少分析：

- Age distribution。
- Sex distribution。
- Height/Weight distribution。
- Device Series 3/4分布。
- 不同年龄段的性能。
- 不同性别的性能。
- 设备标记分层表现。

小组样本不足时标记NA。

不得因为某一人口群体性能差而删除样本。

不得将年龄或性别相关性解释为疾病因果。

## 7.8 输出

PHASE10C_DEMOGRAPHIC_REPORT.md
analysis/demographic_only_metrics.csv
analysis/demographic_fusion_metrics.csv
analysis/demographic_missingness.csv
analysis/demographic_group_analysis.csv
analysis/demographic_parameter_count.csv
analysis/demographic_paired_statistics.csv
analysis/demographic_integrity.json

---

# 八、Phase 10D：DD-aware Cost-sensitive Optimization

本阶段有条件启动。

只有在Phase 10A发现：

- DD错误在多个split中具有稳定的方向；
- 类别margin分析支持存在决策代价不对称；
- 不能仅由少数极小DD亚型样本解释；

才开展。

如果DD错误主要集中在单一小亚型，或者数据质量问题尚未排除，则先不做统一DD权重优化。

## 8.1 历史实验约束

过去已经测试：

- Balanced CE。
- Sampling compensation。
- Focal Loss。
- DD auxiliary classification。

这些方法没有稳定收益。

因此，本轮不重新搜索上述方法。

首先审计当前Balanced CE实际代码。

确认：

- 当前PD权重。
- 当前DD权重。
- reduction方式。
- loss归一化。
- batch内权重实现。

## 8.2 唯一候选

保持原Balanced CE。

仅在现有DD权重基础上增加固定的1.10倍相对系数。

将权重重新归一化到与原实现相同的平均尺度，避免同时改变总体梯度量级。

不搜索：
1.2、1.3、1.5等一系列系数。

不叠加：

- Focal。
- Sampling。
- SupCon。
- Margin loss。
- Activity Interaction。

唯一改动：
DD相对于PD的loss cost增加。

## 8.3 Pilot

固定context0/inner0。

seeds42、43、44。

共3次训练。

比较：

- Certified WSSL baseline。
- DD-aware CE。

保留条件：

- 平均DD Recall提高。
- 平均BA提高。
- PD Recall平均下降不超过0.01。
- AUROC不下降。
- Macro-F1不下降。
- NLL不明显恶化。
- 训练稳定。
- correctness通过。

如果不满足：
正式REJECT，不继续搜索权重。

如果满足：
才执行完整15×3。

注意：
这不是证明DD样本数量不足，而是测试不同误分类代价的影响。

---

# 九、Threshold Optimization：作为独立诊断，不与Loss训练混合

本轮同时完成阈值分析，但不自动改变正式模型阈值。

## 9.1 分析内容

在Certified WSSL上研究：

- PD Recall。
- DD Recall。
- BA。
- Macro-F1。
- Precision。
- NLL。
- Brier。
- ECE。

随threshold变化的关系。

固定主报告threshold：

0.5。

## 9.2 阈值选择规则

禁止直接使用validation标签挑选最高BA阈值，然后在同一validation报告最优结果。

如果具备合法train-only OOF预测：

可以在inner-train OOF上预先确定一个候选阈值。

建议采用约束规则：

在PD Recall不低于train-OOF baseline所规定的预设下限时，最大化BA。

然后将阈值锁定，应用于对应validation。

如果没有合法OOF预测：
仅绘制validation曲线作为描述性诊断，不发布新阈值性能。

## 9.3 输出

THRESHOLD_DIAGNOSTIC_REPORT.md
analysis/threshold_metrics.csv
analysis/threshold_operating_points.csv
analysis/threshold_integrity.json

不把阈值调整称为新的深度学习模型。

不把AUROC变化归因于阈值。

---

# 十、统一实验统计标准

所有完整实验使用：

15 subject-level splits。

每个split：

3 seeds。

使用同一Certified L1数据版本。

同一split内三个seed共享对应train-only normalization。

每个seed独立初始化模型。

## 10.1 主要指标

Accuracy
Balanced Accuracy
AUROC
Macro-F1
PD Recall
DD Recall

辅助指标：

NLL
Brier
ECE

模型复杂度：

- Total Params。
- Trainable Params。
- Frozen Params。
- GPU Memory。
- Training Time。
- Inference Time。

## 10.2 配对统计

所有新候选与同split、同seed的WSSL-CERTIFIED-L1配对。

报告：

- Mean delta。
- Median delta。
- 15 split wins/ties/losses。
- 95% paired CI。
- Wilcoxon signed-rank。
- Effect size。
- BH q。
- Seed stability。
- Subject-cluster bootstrap。

不将15个重叠split称为15个独立临床队列。

不把三个seed视为三个独立患者队列。

## 10.3 统一保留标准

完整15×3实验使用：

1. ΔBA > 0。
2. 至少10/15 split的BA提升。
3. BA配对95% CI下界 > 0。
4. AUROC不下降。
5. Macro-F1不下降。
6. PD Recall平均下降不超过0.01。
7. DD Recall平均下降不超过0.01。
8. Seed stability不超过baseline的1.25倍。
9. Correctness和provenance全部通过。

这是开发阶段保留标准，不代表独立临床验证。

Pilot只有资源筛选意义，不能使用完整统计显著性结论。

---

# 十一、执行预算与顺序

按以下顺序：

Phase 10A：
不进行新的深度训练。
允许轻量Demographics-only诊断拟合。

Phase 10B：
先3个seed pilot。
通过门槛后，最多增加42个正式run，完成15×3。

Phase 10C：
先3个seed pilot。
通过门槛后，最多增加42个正式run。

Phase 10D：
先完成margin和错误诊断。
满足启动条件后3个seed pilot。
通过门槛后，最多增加42个正式run。

所有方向严格串行执行。

不得同时训练多个候选。

任何候选没有通过pilot：
停止该候选，不扩大训练。

如果数据字段不存在、来源不明或授权不足：
相应实验标记BLOCKED。

不得为了让实验完整而伪造人口统计学数据。

---

# 十二、工程正确性与数据保护

本轮只使用Certified L1数据。

禁止使用：

- Legacy L1。
- 旧normalization。
- 旧WSSL checkpoint。
- 旧监督embedding。
- 来源不完整的OOF预测。

不得修改：

- 原始PADS数据。
- mainline正式模型。
- Certified L1正式数据。
- 正式normalization。
- WSSL-CERTIFIED-L1正式checkpoint。
- STR-CERTIFIED-L1历史checkpoint。
- Phase 8 Activity Interaction历史结果。
- 当前正式模型指针。

每个实验必须记录：

- Code SHA256。
- Config SHA256。
- Checkpoint SHA256。
- Data version。
- Normalization SHA256。
- Train subject provenance。
- Validation subject provenance。
- Optimizer state。
- Seed。
- Best epoch。
- Training log。
- Correctness tests。

禁止：

- 使用validation标签拟合normalization。
- 使用validation错误名单调整训练样本权重。
- 读取outer-test performance。
- 将PD/Healthy Control替代PD/DD。
- 将窗口、活动和腕侧当作独立受试者。
- 将demographic字段从subject ID推断。
- 将设备型号作为新输入特征。
- 删除不利样本或亚型。

私有数据、checkpoint、逐人预测、完整ID映射不上传GitHub。

---

# 十三、最终报告

最终生成：

reports/
FINAL_REPORT.md
PHASE10A_DIAGNOSTIC_REPORT.md
PHASE10B_UNFREEZING_REPORT.md
PHASE10C_DEMOGRAPHIC_REPORT.md
PHASE10D_DD_LOSS_REPORT.md
THRESHOLD_DIAGNOSTIC_REPORT.md
STATISTICAL_APPENDIX.md
CORRECTNESS_AND_PROVENANCE.md
COST_AND_DEPLOYMENT_REPORT.md
NEXT_STAGE_RESEARCH_PLAN.md

analysis/
final_metrics.csv
paired_statistics.csv
parameter_counts.csv
inference_benchmark.csv
experiment_state.csv
demographic_integrity.json
unfreezing_integrity.json
dd_loss_integrity.json
phase10_integrity.json

必须回答：

1. HarNet10部分解冻是否优于完全冻结？
2. 最后一个block与最后两个block的适配是否有差异？
3. 部分解冻是否提高DD Recall？
4. 人口统计学特征是否单独具有较强分类能力？
5. 人口统计学融合是否提供超出WSSL的额外增量？
6. 人口统计学增量是否受到年龄、性别或设备混杂影响？
7. DD错误是否主要来自某个来源亚型？
8. DD-aware Loss是否改善了PD/DD平衡？
9. 阈值变化能否改善当前操作点？
10. 哪个候选通过联合保留条件？
11. 哪些方向应该停止？
12. 当前最佳模型是否需要更新？

如果没有任何新候选通过标准：

保留WSSL-CERTIFIED-L1，不强行选择指标最高的候选。

如果Demographic Fusion通过标准：

仍然将它登记为独立多模态模型，不替代纯IMU模型的身份。

如果部分解冻通过标准：

登记新的HarNet适配模型，不覆盖Frozen WSSL-STR。

---

# 十四、最终执行要求

先完成Phase 10A，再自动按照门槛进入10B、10C和10D。

每个阶段完成后生成报告。

不跳过阶段。
不重复历史失败实验。
不修改正式模型。
不自动开展Phase 11。

最终将代码、协议、报告和安全聚合统计发布到GitHub，并执行远端回读核验。

完成后停止。

**最终目标：在Certified L1可靠数据处理基础上，研究外部自监督表示的任务适配能力、人口统计学辅助信息的独立增量，以及PD/DD类别判别代价是否存在可改善空间，从而进一步提升模型的稳定分类能力。**
