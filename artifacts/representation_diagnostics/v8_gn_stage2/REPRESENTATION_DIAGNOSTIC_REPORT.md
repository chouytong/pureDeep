# V8-GN Stage 2 冻结表征诊断报告

日期：2026-09-19  
对象：V8-GN（71,026 参数）+ 已冻结 training recipe  
范围：seed 42/43/44，5 outer development partitions × 3 inner folds，共 45 个
checkpoints；相关检验以三个seed先聚合后的15个独立inner splits为统计单位

## 1. 协议与证据边界

- 本轮没有训练或修改 backbone，也没有加入 DANN、MMD loss、原型对齐等模块。
- 每个 fold 只加载该 fold 的 `best.pt`、train-only normalization、inner-train 和
  inner-validation subject 列表；分析代码没有构造 outer-test dataset 或 loader。
- StandardScaler、线性 probe、activity centroid 和疾病类中心均只在 inner-train
  拟合，再在 unseen inner-validation subjects 上计算结果。
- 独立训练的 checkpoint 之间可能存在表示空间旋转/置换，因此只在同一 checkpoint
  内比较 train 与 validation；跨 fold 仅汇总不依赖坐标对齐的标量统计。
- PCA 只导出代表性 fold 的二维坐标，未把降维图作为结论依据。
- “subject-specific information”通过同一 validation subject 的跨 activity 检索诊断；
  它表示身份相关信息可恢复，不等同于已识别其临床或人口学来源。
- 三个seed共享相同subject splits。45行只用于描述seed稳定性；shift、disease
  separability与BA的关系均先按`(outer, inner)`对三个seed求均值，再以n=15检验。

协议完整性复核：重新提取后的冻结分类 head 在 45 folds 上得到 BA 0.6707、AUROC
0.6800、Macro-F1 0.6631、Accuracy 0.7154，与 Stage 1 冻结结果一致。

## 2. 核心结果

| 诊断问题 | 主要定量结果 | 结论 |
|---|---|---|
| 258维 subject embedding 是否含疾病信息 | train-only 线性 probe：BA 0.6126，AUROC 0.6599；最近类中心 BA 0.6337；cosine silhouette 0.0360 | 有可泛化疾病信号，但 PD/DD 全局簇分离较弱，信息不是简单的紧凑双簇结构 |
| 是否含受试者特异信息 | 跨 activity Top-1 检索 6.32%，随机 0.96%，45/45 folds 均高于随机；Top-5 17.92%；相同受试者相似度 0.1752，同病种其他受试者 0.0243 | 明显保留跨活动稳定的个体信息 |
| activity 信息是否明显 | activity probe accuracy 69.50%，随机 9.09% | activity 仍是表示中的强结构因素 |
| unseen-subject shift 是否存在 | disease-controlled train/validation domain probe AUC 0.6334；41/45 folds >0.5；normalized mean shift 0.1143；CORAL shift 0.4727 | train 与 unseen validation 表示可区分，存在稳定 shift 迹象 |
| shift 是否影响泛化 | 15-split统计：mean shift与head BA ρ=-0.3893，p=0.1515 | 保留负向趋势，但不再宣称稳定显著关系 |
| 错分是否稳定 | 390 人每人 12 次 unseen-validation appearance；54 人稳定错分，247 人稳定正确，89 人不稳定 | 不是只有随机 fold 噪声，存在稳定困难受试者群体 |

## 3. RD-01：疾病可分性

冻结 head 的 BA/AUROC 为 0.6707/0.6800；在同一 258维 embedding 上重新拟合的
train-only balanced logistic probe 为 0.6126/0.6599，最近类中心 BA 为 0.6337。
validation cosine silhouette 均值只有 0.0360，Euclidean silhouette 为 0.0501，说明
PD 与 DD 不是两个清晰紧凑、线性远离的簇。与此同时，cosine silhouette 与 head BA
在15个seed-aggregated splits上仍显著正相关（ρ=0.6071，p=0.01638）；disease-probe
BA与head BA为ρ=0.6524、p=0.00839。因此当疾病几何分离更清楚时，分类通常更好。

**实验确认：** embedding 含疾病信息，但可分信号中等且 fold 依赖明显。  
**不能据此断言：** 线性 probe 低于冻结 head 并不证明需要更复杂 classifier；probe
重拟合、class weighting 和有限 train subjects 都会影响该差值。

## 4. RD-02：受试者特异信息

在去除 inner-train activity centroid 后，用某一 activity embedding 检索该受试者其余
activities 的平均表示：Top-1 为 6.32%，随机期望为 0.96%，相当于随机的 6.57 倍，且
45/45 folds 均高于随机；Top-5 为 17.92%，MRR 为 0.1385。相同受试者跨活动 cosine
similarity 为 0.1752，同疾病其他受试者只有 0.0243，差值为 0.1509。

辅助方差统计中，疾病组间项占 0.44%，疾病内 subject 项占 16.16%，后者约为前者
36.9 倍。该统计不是严格正交 ANOVA，不能把百分比解释成因果贡献，但与跨活动检索
共同支持“个体身份/个体状态信息明显多于紧凑疾病组间结构”。

**实验确认：** raw activity embeddings 保留可跨活动恢复的个体信息。  
**仍需验证：** 这些信息具体来自运动表型、年龄/性别、执行质量还是其他个体因素。

## 5. RD-03：activity 结构与聚合

activity linear probe 的 train→validation accuracy 为 69.50%，显著高于 11 类随机
水平 9.09%。单 activity disease probes 均弱于 subject head；平均 AUROC 从
CrossArms 0.6135 到 Relaxed 0.5375。Activity Attention 高度集中于 CrossArms：
其平均权重 0.5029，随后是 DrinkGlas 0.1040、HoldWeight 0.0922；其余单项均低于
0.06。11项活动的平均 attention 与其单 activity AUROC 排名相关（ρ=0.8364，
p=0.00133），说明 attention 至少与开发数据上的疾病信息强度一致。

**实验确认：** activity identity 是强表示因素，聚合器长期偏重 CrossArms。  
**证据边界：** attention 不是因果特征重要性；不同 seed/fold 中 CrossArms 权重范围
0.0177–0.9864，说明具体权重不稳定，不能据此删减活动。

## 6. RD-04：unseen-subject representation shift

控制疾病类别后，train/validation domain probe 的平均 AUC 为 0.6334（seed 均值：
0.5911/0.6614/0.6477），37/45 folds >0.55，27/45 folds >0.60。normalized mean
shift 为 0.1143±0.0143，CORAL covariance shift 为 0.4727±0.0873。DD centroid
drift 6.436 明显大于 PD 的 2.949，但 DD 样本更少、亚型更异质，当前不能把差值完全
归因于模型。

以15个独立split为单位，normalized mean shift与BA为ρ=-0.3893、p=0.1515；domain
probe AUC为ρ=0.0179、p=0.9496；CORAL shift为ρ=-0.3571、p=0.1913。此前把45个
相关seed-fold rows当独立观测得到的mean-shift显著性已撤回。当前仅支持“train与
validation表示存在可检测差异，以及mean shift有负向趋势”，不支持断言shift已稳定
影响BA，也不支持全局covariance alignment或adversarial confusion必然有效。

## 7. RD-05：稳定错分受试者

每名受试者在 3 seeds 下总计有 12 次 unseen inner-validation appearance。以错误率
≥0.75 定义稳定错分、≤0.25 定义稳定正确：54/390 稳定错分，247/390 稳定正确，
89/390 不稳定。DD 平均错误率 43.71%，PD 为 22.19%；稳定错分中 DD 34/114，PD
20/276。

DD 亚型的稳定错分率为：MS 36.36%（n=11）、Atypical Parkinsonism 33.33%
（n=15）、Other Movement Disorders 33.33%（n=60）、Essential Tremor 17.86%
（n=28）。小亚型样本量有限，只能视为探索性结果。

稳定错分最强的共同特征是：

- validation embedding 更靠近错误疾病类中心（centroid margin Cohen's d=-4.18）；
- 最近 train neighbour 与真实标签一致率仅 24.54%，稳定正确组为 81.48%
  （d=-3.78）；
- activity dispersion 略高（13.324 vs 12.613，d=0.43，p=0.0173）；
- activity consistency 略高（0.188 vs 0.162，d=0.36，p=0.0257）。

embedding norm、分类置信度、attention entropy 和 attention maximum 无显著组间差异。
前两项强效应部分接近错误定义本身，应视为几何描述而非独立病因证据；activity
dispersion 的中小效应更值得后续做可靠性建模验证。

## 8. 综合判断：疾病信息与个体信息存在何种关系

当前可以确认两类信息共存：疾病 probe AUROC 0.6599，说明表征不是纯身份编码；
跨活动 subject retrieval 又稳定高于随机，说明表征并未消除个体特异因素。疾病
silhouette 很小、subject-within-disease 方差远大于疾病组间项，共同构成“疾病信息
嵌在更强个体变异中”的混杂迹象。mean shift与BA只有负向趋势，不能再作为已确认的
泛化损害证据。

这里的“混杂迹象”是表示几何意义，而非已证明的因果混杂。没有 demographics、
疾病时长、药物状态或动作执行质量的配对 probe，就不能把 subject-specific component
归因于某一具体来源。

## 9. 下一步方法选择

当前不建议直接采用单一全局 MMD/CORAL alignment：covariance shift 本身与 BA 无关，
强行全局对齐还可能抹除疾病信号。也不建议立即把 subject ID 作为 DANN domain 标签：
每个 subject 是一个域且每域只有11个相关活动，adversary 很容易不稳定或学到 activity
捷径。

建议下一阶段先做一个**不训练 backbone 的受控验证层**：

1. 在现有 embeddings 上做按疾病分层的 train-centering / shrinkage whitening，检验
   “均值 shift 可修正”是否能跨15个seed-aggregated splits改善线性probe；所有变换仅用train拟合。
2. 加入可获得的 nuisance probes（年龄、性别、执行质量/信号质量），定位
   subject-specific information 的来源，并检查其是否同时预测错误。
3. 对 activity dispersion 建立 validation-only 风险分析和 calibration/abstention 对照，
   判断它是困难样本标记还是可学习的不变性目标。
4. 只有当某个明确 nuisance 在多 fold 下可预测且与错误相关，再选择条件化 DANN、
   nuisance-adversarial 或 class-conditional alignment；若只有均值修正有效，则优先采用
   更低风险的 train-fitted centering，而不是复杂对抗模块。

这一路线遵守“先定位 shift 来源，再选择不变学习机制”，不会把本轮诊断结果误用为
预设某一种方法必然有效。

## 10. 产物

- `diagnostic_summary.json`：45-fold 汇总与 seed-level 均值。
- `fold_diagnostics.csv`：疾病、身份、activity、shift 的逐 fold 指标。
- `activity_diagnostics.csv`：逐 activity、逐 fold probe 和 attention。
- `validation_appearances.csv`：每次 unseen validation appearance 的错误与几何特征。
- `subject_error_stability.csv`：按 subject 聚合的稳定错误率。
- `stable_error_feature_comparison.csv`：稳定错分与稳定正确效应量。
- `condition_error_summary.csv`：PD/DD 亚型探索性错误统计。
- `representative_subject_pca.csv`：代表性 fold 的 PCA 坐标，仅供可视化。
- `embeddings/*.npz`：45 folds 的冻结 embedding 与必要标签，不含 outer-test。
- `run_representation_diagnostics.py`：可复核的只读诊断脚本。
- `seed_aggregated_methodology/seed_aggregated_15_splits.csv`：主要统计单位数据。
- `seed_aggregated_methodology/seed_aggregated_relationships.csv`：n=15相关检验。
