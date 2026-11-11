# Phase 8：Frozen WSSL-STR Guided Discriminative Optimization

## 角色与研究背景

你现在是一名具有丰富研究经验的计算机视觉、可穿戴 IMU 时序分析、深度学习、时序表征学习和帕金森病数字生物标志物研究人员。

请基于当前 PURE-DEEP 项目开展下一阶段模型性能优化。

研究任务：

基于 PADS 数据集的双腕 IMU 传感器数据，进行受试者级 PD（Parkinson's Disease）与 DD（Differential Diagnosis）二分类。

GitHub：

https://github.com/chouytong/pureDeep

当前正式起点：

35b0a7cb78b28f29385724cf98f4f1dbdde8b2b2

服务器项目：

/home/zyt/deep_final

主线代码：

/home/zyt/deep_final/mainline

当前最佳模型：

Frozen WSSL-STR

当前正式 development 指标：

Accuracy = 0.745604
BA = 0.719644
AUROC = 0.759553
Macro-F1 = 0.706589
PD Recall = 0.782344
DD Recall = 0.656945

当前模型结构：

- PADS Acc + Gyro：STR wrist encoder
- Raw Acc：Frozen HarNet10
- HarNet10 external representation：1024D
- LN + Linear：1024D → 64D
- Wrist residual fusion：64D
- Bilateral fusion：左右腕融合
- Activity fusion：11项活动融合
- Subject path + STR path：最终分类

当前模型是所有历史实验中唯一正式保留的最佳 single-model。

本轮研究目标：

**不推翻 Frozen WSSL-STR，而是通过集成、读出优化和跨活动交互学习，寻找当前模型尚未充分利用的 PD/DD 判别信息。**

---

# 一、必须首先完成历史审计

不得直接开始训练。

首先完整阅读：

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
- Phase 7 FINAL_REPORT.md
- Phase 7 INDEPENDENT_VALIDATION_PROTOCOL.md
- Phase 7 NEXT_STAGE_RESEARCH_PLAN.md

必须从服务器和 GitHub 最新提交实际读取，不得只根据历史对话推断代码状态。

重点检查：

1. Frozen WSSL-STR 的真实 forward。
2. Activity Fusion 的真实实现。
3. Subject Path 和 STR Path 的输入与输出。
4. Wrist representation 的生成位置。
5. Activity token 的实际维度和聚合方式。
6. 当前 classifier 的输入。
7. Dataset、normalization、mask 和真实长度处理。
8. 现有 45 个 Frozen checkpoint。
9. 历史 deep-OOF 和其他特征缓存。
10. 现有训练和推理脚本。

必须形成一张实验历史状态表。

特别防止重复以下实验：

- V8-GN、STR-01、WSR-01。
- 原始训练 recipe 搜索。
- Sampling、SAM、Focal、Label Smoothing。
- HarNet Layer5 解冻。
- Output Adapter。
- Gated Fusion。
- Masked Domain Adaptation。
- HarNet5、BioPM。
- Window mean/delta。
- Raw Acc。
- DD auxiliary classification。
- WSSL EMA。
- Frequency-Prior STR。
- Activity Scalar Scaling。
- Patch-STR。
- Dual-reference fusion。
- 普通 OOF threshold optimization。

以上历史负结果必须作为本轮候选设计的约束。

---

# 二、全局实验原则

## 2.1 主模型保护

Frozen WSSL-STR 始终保留为 baseline。

禁止：

- 修改原始正式模型文件。
- 覆盖历史 checkpoint。
- 修改原始 normalization。
- 修改原始 preprocessing。
- 修改原始 decision threshold。
- 覆盖历史正式实验结果。

所有新实验放入独立目录：

/home/zyt/deep_final/artifacts/wssl_guided_optimization_phase8_20261111/

建议结构：

phase8/
    audit/
    phase8a1_ensemble/
    phase8a2_readout/
    phase8b_activity_interaction/
    phase8c_subject_supcon/
    phase8d_comparison/
    reports/
    configs/
    logs/
    tests/

## 2.2 数据划分

沿用项目正式的 subject-level development splits。

优先复用历史15个 development splits 和每个 split 的三个 seed。

所有候选必须在相同受试者划分上进行配对比较。

每个 split 的训练受试者、验证受试者和 excluded subjects 必须严格核验。

Normalization 只能在当前 inner-train 受试者上拟合。

严禁：

- 使用 outer-test 选择模型。
- 根据 validation 错误名单训练模型。
- 使用 validation 标签调整训练权重。
- 将窗口、活动或重复 appearance 当成独立患者。
- 将15个重叠 split 当成15个独立外部队列。

历史 outer 接触和数据访问边界披露必须永久保留。

本轮所有新模型只属于 PADS development evidence。

## 2.3 统一评价指标

主要指标：

- Balanced Accuracy
- AUROC
- Macro-F1
- PD Recall
- DD Recall

同时报告：

- Accuracy
- NLL
- Brier Score
- ECE
- Seed stability
- Paired split improvement
- 95% confidence interval
- Wilcoxon paired test
- BH multiple-testing correction
- Subject-cluster bootstrap

不得根据单个 seed、单个 split 或最高一次结果选择模型。

---

# 三、Phase 8A-1：三 Seed Probability Ensemble

这是本轮第一个任务。

不需要训练新的深度模型。

## 3.1 研究目的

当前 Frozen WSSL-STR 有三个 seed。

历史正式报告使用：

先计算每个 seed 的指标，再在 split 内平均。

这与实际概率集成不同。

本轮检验：

**不同 seed 的预测是否具有互补性，概率平均是否能提高受试者级分类性能？**

## 3.2 资产复核

对每个 split 找到三个 seed 的 Frozen WSSL checkpoint。

核对：

- checkpoint SHA256。
- split ID。
- seed ID。
- 训练 subject IDs。
- validation subject IDs。
- normalization。
- preprocessing。
- 原始预测。
- label mapping。

不得将不同 split、不同 context 或不同模型版本的预测混合。

如果预测缓存存在，先复核来源。

只有来源不完整时才重新进行 validation inference。

不重复训练。

## 3.3 实验方法

主要方法：

Probability averaging：

p_ensemble = (p_seed1 + p_seed2 + p_seed3) / 3

固定：

- 正类 DD。
- 原始 probability 定义。
- threshold = 0.5。
- tie = PD。

辅助敏感性分析：

Logit averaging：

z_ensemble = (z_seed1 + z_seed2 + z_seed3) / 3

不得为两种方法分别搜索最佳阈值。

不得对 ensemble 单独拟合 temperature。

不得搜索 seed 子集或挑选表现最好的两个 seed。

## 3.4 评价

与以下结果配对比较：

1. Frozen WSSL-STR 的正式 seed-first 平均指标。
2. 三个 seed 各自指标。
3. Probability ensemble。
4. Logit ensemble。

重点检查：

- BA 是否提升。
- AUROC 是否提升。
- Macro-F1 是否提升。
- DD Recall 是否提升。
- PD Recall 是否下降。
- 三个 seed 的错误互补性。
- Ensemble 是否降低 seed 方差。

注意：概率集成是多模型系统，不得仍称为单模型。

## 3.5 保留标准

Probability ensemble 为预先指定的主要候选。

只有满足以下条件才 RETAIN：

- BA 有稳定提升。
- AUROC 和 Macro-F1 不下降。
- PD/DD Recall 均不出现超过0.01的平均绝对下降。
- 至少10/15个 split 的 BA 提升。
- 配对 CI 下界大于0。
- 不出现明显稳定性恶化。

Logit ensemble 作为敏感性分析，不得根据结果事后替换主要候选。

## 3.6 输出

- ENSEMBLE_REPORT.md
- analysis/ensemble_metrics.csv
- analysis/ensemble_error_complementarity.csv
- analysis/ensemble_paired_statistics.csv
- analysis/ensemble_integrity.json

如果概率集成没有通过门槛，明确 REJECT 或 INCONCLUSIVE。

完成后进入 Phase 8A-2，不重复训练。

---

# 四、Phase 8A-2：Frozen Representation + Traditional Classifiers

这是第二个任务。

只使用已有 Frozen WSSL-STR checkpoint 和合法特征。

不训练新的深度 encoder。

## 4.1 研究目的

检验当前 WSSL 表征是否已经包含较充分的 PD/DD 判别信息，而现有 subject-level readout 是否存在改进空间。

核心假设：

H1：Frozen WSSL-STR 的表征具有比当前原始分类读出更好的可分性。

H2：不同分类器的决策边界可能更适合当前表征空间。

## 4.2 首先审计 representation extraction point

不能直接选择一个看起来最有利的特征位置。

通过实际 forward 代码确认并固定：

1. 每个 activity 的 bilateral representation。
2. 每个 activity 的 structured representation。
3. Subject Path 聚合前表示。
4. STR Path 聚合前表示。
5. 原始 subject-level embedding。

只选择一个主要特征表示。

如果多个表示都要比较，必须预先登记为独立候选，并进行多重比较校正。

禁止把同一个受试者的11项活动拆成独立样本。

所有输入最终必须形成一行一个 subject 的特征矩阵。

## 4.3 重要的训练和验证约束

对每个 split/seed：

使用该模型对应的 inner-train 和 validation subjects。

Encoder 必须保持冻结。

训练新的传统分类器时：

- 只允许 inner-train embedding 参与拟合。
- 所有 scaler、PCA、feature selection 和 classifier 均只能在 inner-train 上 fit。
- Validation 只能 transform 和 predict。
- 不得将 validation label 用于任何拟合。
- 不得将不同 split 的特征拼接。
- 不得将不同 checkpoint 的特征直接混合。
- 不得使用历史全队列 feature cache 进行无约束筛选。

如果已有 train embedding 是通过当前 checkpoint 提取的，可以使用，但必须记录该 checkpoint 的训练受试者及特征来源。

如果需要真正评估 representation 的独立性，单独标明其是 supervised frozen representation，而不是无监督表征。

## 4.4 分类器候选

只比较以下固定候选：

A. 原始 WSSL classifier head（baseline）。

B. Logistic Regression。

C. Linear SVM。

D. RBF-SVM。

E. Random Forest。

F. XGBoost，仅在环境中已有兼容依赖时使用，不安装或升级依赖。

不要开展大规模超参数搜索。

建议固定低复杂度参数：

- Logistic Regression：L2 regularization。
- Linear SVM：固定C。
- RBF-SVM：固定C与gamma规则。
- Random Forest：固定树数、深度和叶节点约束。
- XGBoost：固定浅树、树数和正则化。

如必须调整参数，只能在 inner-train 内使用严格嵌套的 subject-level CV。

本轮优先研究 readout 类型，不研究超参数最优值。

## 4.5 特征处理

必须核对特征是否存在：

- 标准化。
- L2 normalization。
- 降维。
- 结构化活动拼接。

只允许预先固定的处理。

不得根据 validation 结果选择最好的 feature subset。

特别注意：

如果使用11活动×双腕特征拼接，必须保证所有受试者的 activity/wrist 顺序完全一致。

缺失活动或腕侧不能通过复制特征补齐。

## 4.6 输出

- TRADITIONAL_READOUT_REPORT.md
- analysis/readout_metrics.csv
- analysis/readout_parameters.csv
- analysis/readout_feature_integrity.csv
- analysis/readout_paired_statistics.csv
- analysis/readout_integrity.json

回答：

1. 原始 WSSL head 是否已经接近线性可分读出的上限？
2. RBF-SVM 是否表现出稳定非线性收益？
3. 树模型是否能够利用活动间非线性关系？
4. 读出变化是否真正改善 DD Recall？
5. 是否值得将传统分类器作为新的正式 baseline？

如果没有任何候选通过联合门槛，停止该方向。

完成后再评估 Phase 8B。

---

# 五、Phase 8B：Activity Token Interaction Learning

这是本轮第一个新的深度模型结构候选，也是重点实验。

## 5.1 研究动机

当前 WSSL-STR 已经融合了：

- 外部 HarNet10 表征。
- 本地 STR wrist representation。
- 左右腕信息。
- 11项活动结构。

但是，仍有一个可以研究的问题：

**不同活动之间是否存在尚未被现有 Activity Fusion 充分利用的疾病判别交互信息？**

例如：

一个受试者可能在 TouchNose 活动中表现正常，但在 Entrainment 活动中出现明显差异。

另一个受试者可能在两项活动中表现一致。

活动间响应关系可能具有疾病判别价值。

## 5.2 必须先审查现有 Activity Fusion

这一条非常重要。

在写新模块之前，完整检查：

- Activity Fusion 是否已经使用 self-attention。
- 是否已经使用 cross-activity attention。
- 是否已经有活动间 token interaction。
- 是否仅使用 pooling、activity ID embedding 或 learned attention。
- 是否已有等价的非线性跨活动交互。

如果当前结构已经实现了等价机制：

不要重复添加一个 Transformer 后就称为创新。

先分析原有机制的表达能力，再提出真正不同的候选。

## 5.3 新模型的唯一主要改动

如果审计确认存在明确的跨活动交互缺口，建立：

Activity Interaction WSSL-STR（AI-WSSL-STR）

保留原有：

- Frozen HarNet10。
- Local STR wrist encoder。
- Wrist residual fusion。
- Bilateral fusion。
- 原始数据处理。
- 原始 subject-level label。
- 原始 baseline loss。

新增一个轻量 Activity Token Interaction 模块。

输入：

H = [h1, h2, ..., h11]

其中每个 h 对应一个 activity-level subject token，维度必须从真实模型中读取，不得假定。

主要模块：

1. Linear projection（仅在必要时）。
2. Activity positional/identity embedding。
3. Single-layer multi-head self-attention。
4. Residual connection。
5. LayerNorm。
6. Feed-forward layer。
7. Residual aggregation。

结构：

Activity tokens
      ↓
Activity identity embedding
      ↓
Single-layer self-attention
      ↓
Residual + LayerNorm
      ↓
Feed-forward
      ↓
Residual + LayerNorm
      ↓
Subject-level pooling
      ↓
Original classifier-compatible representation
      ↓
PD/DD logits

必须保证输出仍然对应原 subject-level classification。

## 5.4 关键设计约束

不要一次搜索多种结构。

第一版固定：

- 1层 attention。
- 4个 heads；如果实际 token 维度无法整除，先通过明确 projection 解决。
- 一个固定 FFN expansion。
- 一个固定 dropout。
- 一个固定 residual fusion 方式。

只允许一个主要结构改动。

不添加：

- 新的 Gyro 分支。
- 新的 HarNet 解冻。
- 新的 Gate。
- 新的频域模块。
- 新的 Patch 模块。
- 新的 DD subtype classifier。
- 新的异常检测分支。

如果现有 Activity Fusion 已经具有完整 token interaction，则将本候选标记为 REDUNDANT，不启动训练。

## 5.5 训练策略

第一版采用：

Frozen WSSL-STR representation + Activity Interaction module。

优先冻结：

- HarNet10。
- 原始 STR wrist encoder。
- 原始 wrist residual projection。

只训练：

- 新增 Activity Interaction module。
- 必要的 subject-level readout。

原始 baseline classifier 保留为对照。

如果原模型中的某个部分无法冻结而不破坏 forward，先做 correctness 诊断，不要随意改变原始网络。

训练采用：

- 原始 subject-level splits。
- 原始 normalization。
- 原始 batch 和 optimizer recipe。
- 原始 seed 列表。
- 相同训练预算。
- 相同 checkpoint selection 规则。

先进行小规模 correctness smoke test，不使用 outer-test。

只有 smoke test 通过后，才允许启动完整15×3配对训练。

## 5.6 正确性测试

至少测试：

1. 输入 token 数量为11。
2. activity ID 顺序正确。
3. 左右腕顺序正确。
4. 真实活动 mask 正确。
5. 不存在 validation subject 进入 train。
6. 输出 logits 为 [B,2]。
7. 无 NaN/Inf。
8. 原始 Frozen baseline 不变。
9. HarNet10 冻结。
10. 原始预处理不变。
11. 新模块参数可训练。
12. checkpoint 可以独立恢复。
13. inference 输出可重复。
14. padding 不影响真实 token。
15. 训练和评估 subject-level separation 正确。

## 5.7 主要评价

与 Frozen WSSL-STR 配对比较：

- Accuracy
- BA
- AUROC
- Macro-F1
- PD Recall
- DD Recall
- NLL
- Brier
- ECE
- 参数量
- 推理时间

特别关注：

活动交互是否提高 DD Recall，同时保持 PD Recall。

不要只报告总体 BA。

## 5.8 保留条件

只有满足预先固定的联合条件才 RETAIN：

- ΔBA 为正。
- 至少10/15 splits BA 提升。
- BA 配对95% CI 下界大于0。
- AUROC、Macro-F1 不下降。
- PD/DD Recall 各自平均下降不超过0.01。
- Seed stability 不明显恶化。
- correctness 和 provenance 全部通过。

如果只提高 AUROC，不提高 BA/Macro-F1，不得自动保留。

如果仅 DD Recall 上升但 PD Recall 明显下降，也不得保留。

## 5.9 输出

- ACTIVITY_INTERACTION_REPORT.md
- analysis/activity_interaction_metrics.csv
- analysis/activity_interaction_ablation.csv
- analysis/activity_attention_summary.csv
- analysis/activity_interaction_parameters.csv
- analysis/activity_interaction_integrity.json

Attention 权重只能作为模型内部计算结果，不能直接解释为临床重要性。

---

# 六、Phase 8C：Subject-level Supervised Contrastive Learning

只有 Phase 8B 完成、结果归档后，才进入本阶段。

不得同时开展 B 和 C 的完整训练。

## 6.1 研究假设

当前 WSSL-STR 主要通过交叉熵优化分类结果。

研究问题：

是否可以通过 subject-level supervised contrastive learning 改善 PD/DD 的类间判别表示？

目标：

- 增加有效的类别间分离。
- 保持同类患者具有合理的表征一致性。
- 避免 DD 内部异质性被过度压缩。

## 6.2 设计原则

使用 subject-level embedding。

不得将：

- activity 当成独立受试者。
- wrist 当成独立受试者。
- window 当成独立受试者。
- 同一个人的不同记录当成独立正样本。

DD 包含多个诊断亚型，不能简单强制所有 DD 特征紧密聚集。

因此需要先检查现有训练标签和 DD subtype metadata 的可用性。

如果亚型标签在 inner-train 不完整或无法可靠对应：

不进行基于亚型的额外监督。

## 6.3 损失函数

主要候选：

L = L_CE + lambda * L_SupCon

其中：

L_CE：
原始二分类交叉熵。

L_SupCon：
subject-level supervised contrastive loss。

第一轮固定一种实现。

不同时搜索：

- Triplet loss。
- ArcFace。
- Center loss。
- Focal contrastive。
- 多种 prototype loss。

lambda 必须在 inner-development 训练流程中确定，不得使用 outer-test。

为了控制实验规模，第一轮只使用一个预先固定的辅助损失权重。

如果 smoke test 显示对比样本构成不合法、batch 内正样本不足或 loss 退化，先修正 correctness；不能通过改变受试者身份或复制样本凑正样本。

## 6.4 DD 内部异质性检查

训练前后分析：

- PD 类内距离。
- DD 类内距离。
- PD-DD 类间距离。
- 不同 DD 亚型之间的距离（数据允许时）。
- Embedding norm。
- Classifier margin。
- 每类 Recall。

特别检查：

是否发生 DD 类内部过度压缩。

如果整体类间距离增加，但 DD 亚型结构被破坏，不能直接解释为更好的疾病表征。

## 6.5 训练协议

保持：

- Frozen WSSL-STR 原始输入。
- 原始 subject-level splits。
- 原始 normalization。
- 原始训练预算。
- HarNet10 冻结。
- 不改变活动结构。
- 不添加新的分类分支。

仅改变训练目标：

CE → CE + SupCon。

这是本阶段唯一主要改动。

## 6.6 输出

- SUBJECT_SUPCON_REPORT.md
- analysis/supcon_metrics.csv
- analysis/supcon_embedding_geometry.csv
- analysis/supcon_class_margin.csv
- analysis/supcon_subtype_analysis.csv
- analysis/supcon_integrity.json

如果没有通过联合保留条件，正式 REJECT。

---

# 七、Phase 8D：统一模型比较

前三个阶段全部完成后，统一整理：

1. Frozen WSSL-STR。
2. Probability ensemble。
3. Logit ensemble。
4. Logistic Regression。
5. Linear SVM。
6. RBF-SVM。
7. Random Forest。
8. XGBoost（如成功运行）。
9. AI-WSSL-STR。
10. WSSL-STR + SupCon。

不允许将未通过 correctness 的模型纳入性能排名。

不允许把不同验证协议下的结果混在同一张主表。

如果某个阶段没有通过门槛，仍保留其负结果，不强行加入正式模型。

统一比较：

| 指标 | 目标 |
|---|---|
| BA | 主要判别指标 |
| AUROC | 排序能力 |
| Macro-F1 | 两类综合分类 |
| PD Recall | PD 识别能力 |
| DD Recall | DD 识别能力 |
| Accuracy | 总体正确率 |
| NLL | 概率质量 |
| Brier | 概率误差 |
| ECE | 校准误差 |
| Seed SD | 稳定性 |
| Parameter count | 模型复杂度 |
| Inference time | 部署成本 |

最终选择必须依据联合条件，而不是单一最高指标。

---

# 八、统计与保留规则

## 8.1 配对统计

所有新候选使用相同 subject-level splits。

先在每个 split 内按正式口径聚合三个 seed，再对15个 development units 进行配对比较。

报告：

- Mean improvement。
- Median improvement。
- Split wins/ties/losses。
- Paired 95% CI。
- Wilcoxon signed-rank p。
- BH q。
- Effect size。
- Seed stability。

另外报告 subject-cluster bootstrap。

不要将15个重叠 splits 当作15个独立临床队列。

## 8.2 多重比较

预先定义统计检验 family。

不得在实验完成后根据结果更改 family 大小。

对不可检验或缺失项明确标记 NA。

不得通过删除负结果降低多重比较惩罚。

## 8.3 统一保留标准

本轮所有新模型使用以下开发保留标准：

1. BA 平均提升。
2. 至少10/15 splits 提升。
3. 配对95% CI 下界大于0。
4. AUROC 不下降。
5. Macro-F1 不下降。
6. PD Recall 平均下降不超过0.01。
7. DD Recall 平均下降不超过0.01。
8. Seed stability 不超过 baseline 的1.25倍。
9. Correctness 和 provenance 全部通过。

这是本轮预先登记的 development 保留规则。

通过不等于临床独立验证成功。

---

# 九、严格禁止的操作

1. 不访问 outer-test performance。
2. 不修改 Frozen WSSL-STR 正式源码。
3. 不覆盖历史 checkpoint。
4. 不重新搜索历史失败的 Adapter、Gate、Patch。
5. 不重新搜索阈值。
6. 不重新搜索 CE/focal/sampler。
7. 不重新解冻 HarNet10。
8. 不根据 validation 错误名单训练模型。
9. 不把 PD/Healthy Control 作为 PD/DD。
10. 不把窗口、活动或腕侧当独立患者。
11. 不删除历史负结果。
12. 不上传私有数据、checkpoint、cache、逐人预测和受保护 manifest。

所有新训练必须具有独立目录、配置、源码 hash、日志、checkpoint provenance 和 correctness tests。

---

# 十、最终研究报告要求

完成所有阶段后，必须给出：

## A. 实验状态表

每个候选：

- 实验目的。
- 研究假设。
- 实验方法。
- 主要模型。
- 参数量。
- 训练成本。
- 主要结果。
- 六项分类指标。
- 统计显著性。
- Seed stability。
- 最终状态。

## B. 关键科学问题

明确回答：

1. 三 seed 集成是否具有实际收益？
2. 当前 Frozen WSSL 表征是否存在 readout bottleneck？
3. 活动间交互是否提供新判别信息？
4. SupCon 是否改善 PD/DD 表征？
5. DD 内部异质性是否受到破坏？
6. 哪个方向最值得继续？
7. 是否有新模型通过联合保留门槛？

## C. 下一阶段建议

对每个后续方向给出：

- 研究假设。
- 现有证据。
- 与历史失败实验的区别。
- 预期贡献。
- 训练成本。
- 保留条件。
- 停止条件。

不要为了得到新模型而强行保留候选。

---

# 十一、执行顺序和阶段门槛

严格按以下顺序执行：

Phase 8A-1：
先做三 Seed 集成。

完成报告后：

Phase 8A-2：
再做 Frozen Representation + Traditional Classifiers。

完成报告后：

Phase 8B：
检查现有 Activity Fusion，再决定是否开展 Activity Token Interaction。

如果发现已有等价机制，则标记 REDUNDANT，不重复训练。

Phase 8B 完成后：

Phase 8C：
再开展 Subject-level SupCon。

最后：

Phase 8D：
统一模型比较。

每个阶段必须先通过 correctness、provenance 和数据划分检查，才进入后续阶段。

如果某阶段发现数据泄漏、来源缺失或模型结构重复，立即停止对应候选并记录。

不要因为某阶段被拒绝就跳过审计直接开始其他训练。

本轮允许按照上述顺序执行完整实验，但不得越过任何阶段门槛。

---

# 十二、最终目录与发布

所有产物：

/home/zyt/deep_final/artifacts/wssl_guided_optimization_phase8_20261111/

不得修改：

/home/zyt/deep_final/mainline

所有模型和历史产物只读保护。

最终按现有 Git 发布规范上传：

- 新增实验源码。
- 独立配置。
- correctness tests。
- 训练协议。
- 汇总统计。
- 最终报告。
- 负结果。
- 实验状态表。

不上传任何私有数据、checkpoint、缓存、逐受试者预测及受保护 manifest。

最终核对 GitHub 发布内容与服务器快照一致。

完成 Phase 8D 后停止，不自动启动 Phase 9。

**最终目标：在严格 subject-level evaluation 和统一配对协议下，判断多 Seed 集成、读出优化、跨活动交互和监督对比学习是否能够真正提升 Frozen WSSL-STR 的 PD/DD 判别能力。不是单纯追求某一次实验的最高 BA。**