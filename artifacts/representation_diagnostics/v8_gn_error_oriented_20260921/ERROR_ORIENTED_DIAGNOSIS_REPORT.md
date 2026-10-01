# V8-GN Error-oriented Activity/Wrist Evidence Diagnosis

日期：2026-09-21  
模型：冻结 V8-GN + frozen training recipe + Full bilateral fusion  
范围：inner-development train/validation only；未训练模型，未修改参数，未访问outer信息。

## 研究问题

稳定错分是否与activity-level或wrist-level疾病证据的冲突、离散度和可靠性变化有关？

稳定性分组沿用Stage 2已有validation appearances：247名stable-correct、54名
stable-error、89名unstable；每位受试者平均出现12次。分组中PD/DD分别为：stable-correct
200/47，stable-error 20/34，unstable 56/33。

## 方法

对每个seed×inner split，从冻结checkpoint只做前向提取。针对11个activity及Left、Right、
bilateral Mean三类64维表示，分别在inner-train拟合StandardScaler + balanced logistic
probe，并仅在unseen inner-validation计算DD logit margin。正margin支持DD，负margin支持PD；
乘以真实标签方向后得到true-class margin。

主要量：activity true margin、错误方向activity比例、activity两两符号冲突率、activity margin
SD/IQR、左右腕符号冲突率、左右腕margin绝对差，以及它们与冻结subject head margin/error的
关系。三个seed先在同一split聚合，15个inner split为统计单位。受试者级和activity级检验分别
进行BH-FDR。冻结前向与原forward逐元素最大误差为0。

## 总体分组结果

| 指标 | Stable-correct | Stable-error | Unstable | Error−Correct | split方向 | BH-FDR q |
|---|---:|---:|---:|---:|---:|---:|
| Activity true margin均值 | 0.902 | -0.447 | 0.251 | -1.349 | 0/15更高 | 0.000106 |
| 错误方向activity比例 | 0.327 | 0.583 | 0.449 | +0.256 | 15/15更高 | 0.000106 |
| Activity冲突率 | 0.423 | 0.459 | 0.468 | +0.035 | 15/15更高 | 0.000106 |
| Activity margin SD | 2.021 | 2.233 | 2.043 | +0.212 | 13/15更高 | 0.00555 |
| Activity margin IQR | 2.173 | 2.382 | 2.142 | +0.208 | 13/15更高 | 0.00159 |
| Wrist冲突率 | 0.407 | 0.447 | 0.446 | +0.039 | 14/15更高 | 0.000187 |
| Wrist disagreement均值 | 2.146 | 2.451 | 2.212 | +0.305 | 15/15更高 | 0.000106 |
| Wrist disagreement最大值 | 6.288 | 7.128 | 6.519 | +0.839 | 15/15更高 | 0.000106 |

稳定错分相对稳定正确确实具有更多activity/wrist冲突。但与unstable比较时：activity冲突率
差异为-0.0088（q=0.0902），wrist冲突率差异为+0.0006（q=1.0）；两种符号冲突率都不能
稳定区分“持续错误”和“一般困难”。相反，错误方向activity比例、activity dispersion和
wrist disagreement幅值仍能区分stable-error与unstable。

## 疾病类别异质性

这是本次最重要的限制性发现。

### PD稳定错分

- activity冲突率：+0.0697，13/14 split更高，q=0.000721；
- activity margin SD：+0.9005，14/14更高，q=0.000159；
- wrist冲突率：+0.0559，11/14更高，q=0.00193；
- wrist disagreement均值：+1.0545，14/14更高，q=0.000159。

PD稳定错分符合“多activity/双腕证据冲突与高离散度”的模式。

### DD稳定错分

- activity冲突率：-0.0274，只有5/15更高，q=0.0429；
- activity margin SD：-0.6147，0/15更高，q=0.000159；
- wrist冲突率：+0.0002，8/15更高，q=0.890；
- wrist disagreement均值：-0.5777，0/15更高，q=0.000159；
- 错误方向activity比例仍增加+0.2057，15/15更高，q=0.000159。

DD稳定错分不是“证据互相矛盾”，而更接近多个activity和双腕一致地提供PD方向的错误证据。
因此，通用的低可靠性降权或冲突抑制未必能修复DD错误，甚至可能只帮助PD而不帮助DD。

## Activity贡献方向

基于bilateral Mean probe，stable-error相对stable-correct的true margin差：

| Activity | 差值 | split方向 | BH-FDR q |
|---|---:|---:|---:|
| CrossArms | -4.392 | 15/15更低 | 0.000279 |
| DrinkGlas | -2.057 | 15/15更低 | 0.000279 |
| HoldWeight | -1.624 | 15/15更低 | 0.000279 |
| LiftHold | -1.406 | 15/15更低 | 0.000279 |
| TouchIndex | -1.162 | 15/15更低 | 0.000279 |
| StretchHold | -1.149 | 15/15更低 | 0.000279 |
| PointFinger | -1.146 | 15/15更低 | 0.000279 |
| Entrainment | -1.054 | 14/15更低 | 0.000555 |
| TouchNose | -0.736 | 14/15更低 | 0.000555 |
| RelaxedTask | -0.093 | 10/15更低 | 0.373 |
| Relaxed | -0.021 | 9/15更低 | 0.639 |

CrossArms是最强、最一致的系统性错误方向来源；DrinkGlas、HoldWeight和LiftHold次之。
Relaxed/RelaxedTask没有稳定组间方向差。这里说明的是train-only probe的可读出方向，不证明
冻结attention head实际赋予了相同因果权重。

## 与最终预测的关系

45个seed-fold内相关系数再按split聚合后：

- 错误方向activity比例与最终错误：平均Spearman rho=0.377；
- activity冲突率与最终错误：rho=0.123；
- wrist冲突率与最终错误：rho=0.089；
- wrist disagreement均值与最终错误：rho=0.064；
- bilateral true margin与最终错误：rho=-0.432。

最终错误更直接关联于“多数activity共同朝错误方向”，而不是单纯的冲突大小。

## 结论分级

**实验确认：** stable-error相对stable-correct具有更高的activity/wrist冲突、更大的activity
dispersion及wrist disagreement；但符号冲突率不能区分stable-error与unstable。PD错误呈高冲突
模式，DD错误呈低冲突但一致错误方向模式。CrossArms等九个活动存在稳定的错误方向差。

**合理推测：** 当前困难不是单一“可靠性估计失败”。PD可能受跨activity/双腕不一致影响，
DD更可能受共享表征偏置、DD异质性、标签/亚型结构或所有活动共同的混杂因素影响。

**尚未验证：** CrossArms等活动是错误原因还是仅反映同一潜在表型；可靠性加权能否改善PD
同时不损害DD；DD一致错误证据是否与具体subtype、疾病严重程度、用药状态或采集质量有关。

## 决策

当前证据不足以启动通用reliability-aware multi-activity或bilateral fusion训练：冲突模式不但
不能稳定区分stable-error与unstable，而且在PD和DD中方向相反。按照停止规则，不继续当前
PADS分类结构搜索。下一步应优先回到数据与标签层面，检查DD subtype、临床严重度、任务执行
质量以及CrossArms等高风险活动的原始信号/元数据；如未来提出可靠性方法，必须预先规定
class-balanced安全条件并验证DD recall不下降。

## Provenance

历史seed-42 outer runner误启动记录继续保留。本分析只glob读取`outer_*/inner_*` checkpoint、
inner split和train-only normalization；没有读取outer-final目录、outer predictions或outer metrics。

