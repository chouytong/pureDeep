# V8-GN Layer-wise Representation Diagnosis

日期：2026-09-20  
实验编号：LRD-01  
范围：冻结 V8-GN 的 inner-development representation diagnosis

## 1. 研究问题与协议

本阶段不训练模型、不修改 backbone、不增加损失或训练模块。唯一目标是定位疾病信息、个体信息、activity 信息和 train→unseen-validation shift 在网络中的演化位置。

使用冻结配置：V8-GN + AdamW + LR=2e-4 + batch=8 + WD=1e-4 + cosine scheduler。读取 seed 42/43/44 在完全相同 subject splits 上的已有 inner-development checkpoint。所有指标先在每个 `(outer, inner)` 内对三个 seed 求均值，再以15个独立 inner split 为统计单位；45个 seed-fold 结果只作为原始记录，不作为独立推断样本。outer-test 未被访问或用于分析。

## 2. 层级定义

| 层级 | 维度 | 精确定义 |
|---|---:|---|
| Wrist encoder | 64 | 每个 activity、每只腕经过冻结时序编码器及 learned feature pooling 后的 embedding |
| Bilateral activity | 258 | `[left, right, masked mean, |left-right|, wrist mask]` 双腕融合结果 |
| Activity context | 258 | bilateral activity 加显式 activity identity embedding，再经 LayerNorm |
| Subject aggregation | 258 | Activity Attention 对11个 activity context 加权聚合后的 subject embedding |
| Classifier logits | 2 | 冻结线性分类头输出的 PD/DD logits |

为公平比较疾病可读出性和 shift，前三层均先对有效 token 做无参数 mean pooling，形成每位受试者一个向量，再拟合 train-only StandardScaler 与 balanced logistic linear probe。该 mean pooling 是诊断读出，不是实际网络前向路径，因此结果回答“该层已有多少可线性恢复的疾病信息”，不能等同于替换网络聚合器后的真实性能。

跨 activity subject retrieval 与 activity probe 只对仍保留 activity 轴的前三层定义。subject aggregation 和 logits 已不存在 activity 轴，因此记为 N/A，而不是复制 subject 向量制造伪观测。

自定义提取路径与冻结模型原始 forward 在45个 fold 的 train/validation 首批数据上逐元素一致，activity、subject、logits 的最大绝对误差均为0。

## 3. 疾病信息逐层演化

下表为15个独立 split 的均值±split SD。

| 层级 | Linear probe BA | Linear probe AUROC | Centroid BA | Cosine silhouette | Disease between variance | Within-disease subject variance |
|---|---:|---:|---:|---:|---:|---:|
| Wrist encoder | 0.6048±0.0199 | 0.6474±0.0229 | 0.5965±0.0307 | 0.0247±0.0150 | 0.0281±0.0140 | 0.9719±0.0140 |
| Bilateral activity | 0.6070±0.0234 | 0.6549±0.0254 | 0.6020±0.0313 | 0.0189±0.0115 | 0.0232±0.0100 | 0.9768±0.0100 |
| Activity context | 0.6078±0.0270 | 0.6587±0.0242 | 0.6029±0.0296 | 0.0189±0.0112 | 0.0229±0.0096 | 0.9771±0.0096 |
| Subject aggregation | 0.6126±0.0375 | 0.6599±0.0420 | 0.6337±0.0280 | 0.0360±0.0137 | 0.0303±0.0103 | 0.9697±0.0103 |
| Classifier logits | 0.6539±0.0275 | 0.6813±0.0433 | 0.6521±0.0323 | 0.1296±0.0558 | 0.1014±0.0441 | 0.8986±0.0441 |

### 相邻层 matched-split 变化

- **Wrist→bilateral：** probe AUROC +0.0074，但未达到显著性（10/15上升，p=0.135）；cosine silhouette -0.0058（14/15下降，p=0.00018），disease-between variance ratio -0.00493（13/15下降，p=0.00061）。双腕拼接增加少量线性信息，但疾病几何相对总变异反而变弱。
- **Bilateral→activity context：** probe AUROC +0.00381（12/15上升，p=0.0302），其他疾病几何基本不变。显式 activity identity 主要编码任务身份，只带来很小的疾病读出增量。
- **Activity context→subject aggregation：** probe BA +0.00476、AUROC +0.00126，均无显著变化；但 centroid BA +0.0308（12/15上升，p=0.00451）、cosine silhouette +0.0171（14/15上升，p=0.00012）、disease-between variance +0.00742（p=0.00836）。因此 Activity Attention 没有削弱疾病信息，反而改善了疾病几何集中性。
- **Subject→logits：** probe BA +0.0413（14/15上升，p=0.00043），AUROC +0.0214（p=0.0413），disease-between variance +0.0711（15/15上升，p=0.000061）。这是监督分类头将已有表示压缩到疾病决策轴的预期结果，不能解释为分类头独立创造了新病理信息。

冻结模型实际 head 的 BA/AUROC 为0.6707/0.6800；logits 上另拟合的 linear probe 只用于统一几何比较，不替代原分类指标。

## 4. Activity 与 subject-specific information

| 层级 | Activity probe accuracy | Subject retrieval Top-1 | Top-5 | Subject similarity gap |
|---|---:|---:|---:|---:|
| Wrist encoder | 0.6672±0.0200 | 0.0501±0.0055 | 0.1590±0.0125 | 0.1853±0.0226 |
| Bilateral activity | 0.6950±0.0231 | 0.0632±0.0072 | 0.1792±0.0161 | 0.1509±0.0186 |
| Activity context | 1.0000±0.0000 | 0.0542±0.0063 | 0.1632±0.0144 | 0.1699±0.0226 |

平均 Top-1 随机机会约为0.0096，三层均明显高于机会水平，说明跨 activity 可恢复同一受试者身份。

- Wrist→bilateral 后 activity probe、Top-1 和 Top-5 在15/15 split 上上升；但 similarity gap 在15/15 split 上下降。不同 identity 指标反映的几何侧面不同，不能把某一指标单独解释为“全部个体信息”。
- Activity context 的 activity probe 为100%，这是模型显式加入 activity embedding 的直接架构结果，不是数据泄漏。它同时使 Top-1/Top-5 相对 bilateral 层下降，但 similarity gap 回升。
- 现有结果确认 subject information 从 wrist 层起就存在，并贯穿 activity representation；没有证据表明某一中间层会将其单调清除。

## 5. Train→unseen-validation shift

| 层级 | Disease-controlled domain AUC | Normalized mean shift | CORAL covariance shift |
|---|---:|---:|---:|
| Wrist encoder | 0.5543±0.0334 | 0.1173±0.0233 | 0.3751±0.0713 |
| Bilateral activity | 0.5839±0.0373 | 0.1197±0.0173 | 0.4652±0.0515 |
| Activity context | 0.5833±0.0442 | 0.1206±0.0160 | 0.4699±0.0573 |
| Subject aggregation | 0.6388±0.0646 | 0.1143±0.0104 | 0.4727±0.0622 |
| Classifier logits | 0.6472±0.0843 | 0.0700±0.0287 | 0.3583±0.1812 |

- Wrist→bilateral 的 domain AUC +0.0295（12/15上升，p=0.00336），CORAL +0.0900（15/15上升，p=0.000061）。双腕融合是第一个稳定放大分布差异的位置。
- Activity context→subject aggregation 的 domain AUC +0.0555（14/15上升，p=0.00043），同时 mean shift 略降且 CORAL 基本不变。说明聚合后 shift 更容易被分类器识别，但不是简单的均值平移。
- logits 的 mean/CORAL shift 下降，但 domain AUC 仍高。二维监督投影压缩了总体协方差差异，却保留了与训练/验证域可分的决策分布。

这些 shift 指标与最终 BA 在15个 split 上均未形成稳定关系；例如 subject 层 domain AUC 与 BA 的 ρ=0.0214、p=0.9396。因此“shift 增加”是已确认的表示现象，但“它导致 BA 下降”仍未得到验证。

## 6. 各层指标与最终 BA

- Wrist probe AUROC：ρ=0.4000，p=0.1396。
- Bilateral activity probe AUROC：ρ=0.7214，p=0.00240。
- Activity context probe AUROC：ρ=0.6750，p=0.00576。
- Subject probe AUROC：ρ=0.5929，p=0.01985。
- Subject disease silhouette：ρ=0.6071，p=0.01638。
- Subject disease-between variance ratio：ρ=0.5607，p=0.02968。
- Subject retrieval Top-1 在 wrist、bilateral、context 层与 BA 均无显著关系。

因此，真正与 split-level 泛化表现稳定相关的是 activity/subject 表征中的疾病可读出性，而不是总体 subject retrieval 强度或 shift 强度。

logits 指标与最终 BA 高度相关是同一决策输出的直接结果，属于近端/部分循环证据，不用于定位上游因果瓶颈。

## 7. DD 亚型探索

DD subtype cosine silhouette 在 wrist、bilateral、context、subject、logits 层分别为 -0.0851、-0.0625、-0.0616、-0.0821、-0.2905。最近训练亚型中心的验证准确率分别为0.4168、0.4349、0.4328、0.3927、0.2450，均低于验证 DD 的多数亚型基准0.5265。

因此，当前 embedding 中没有发现与已知 DD 临床亚型一致的清晰多模态结构。该结果不能排除 DD 异质性：Atypical Parkinsonism 和 Multiple Sclerosis 样本量很小，且疾病亚型不一定对应运动信号中的紧凑簇。现阶段没有证据支持立即加入 subtype-aware auxiliary supervision。

## 8. 结论分级

### 实验确认

1. 疾病信息在 wrist encoder 已经可线性恢复，但仅为中等水平（AUROC约0.647）。
2. 双腕融合只带来小幅 probe AUROC 增益，却稳定降低疾病相对几何分离并增加 domain/CORAL shift。
3. 显式 activity context 几乎完整保留任务身份；该层没有显著削弱疾病信息。
4. Activity Attention 聚合没有造成 activity→subject 疾病信息下降，反而改善 centroid、silhouette 和 disease-between variance。
5. subject-specific information 在所有保留 activity 轴的层均可恢复，且显著高于随机检索机会。
6. disease-between variance 在 classifier 之前仅约2.3%–3.0%，within-disease subject variance 约97%，说明疾病信号处于强个体差异背景中。
7. DD subtype 没有形成稳定、清晰的已知亚型簇。

### 合理推测

1. 主要疾病表示瓶颈更接近 wrist encoder / bilateral activity representation，而不是 subject aggregation。
2. 双腕融合的原始拼接、绝对差和 mask 可能放大与疾病无关的跨受试者尺度或左右差异；这是后续值得做受控诊断的具体位置。
3. subject aggregation 同时增强疾病几何和 domain 可辨识性，可能是一种“有用判别放大与域特异放大并存”的权衡。

### 尚未验证

1. 尚未证明修改双腕融合能够提高 unseen-subject BA。
2. 尚未识别 subject-specific variation 的具体来源，也未证明消除全部个体信息有益。
3. 尚未证明 domain shift 是性能下降的因果因素。
4. 尚未证明 subtype-aware supervision 无效，只能说当前表征和样本规模不支持优先采用它。

## 9. 方法决策

本诊断不支持“activity→subject 信息丢失”假设，因此下一步不应优先重做多活动聚合。也不支持立即使用 subtype-aware auxiliary supervision。

如果继续进行方法实验，最合理的单因素起点应是冻结整体框架下针对 **wrist→bilateral activity** 的表示问题提出严格对照：先区分左右腕独立疾病可读出、mean/difference 各分量贡献及其 shift，再决定是否需要可靠性加权或规范化双腕融合。任何新方法仍需满足15个 matched split、多 seed 和疾病性能/不变性双重标准，且不得依据 outer-test 选择。

