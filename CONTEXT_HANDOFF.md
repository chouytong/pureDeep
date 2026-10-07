# PADS / STR-01 项目交接摘要

> **最新状态（2026-10-01）：Phase-3E 已完成并归档（180 new development runs）。HarNet5 与纯神经 BioPM acc-only 均未替代 pretrained-frozen HarNet10 WSSL-STR；E3 random control 已完成，E4/E5 未满足条件而未运行。最佳 retained single-model 仍为 WSSL-STR（BA0.7196/AUROC0.7596）。最高固定 inference AUROC 为六模型等权0.7870，但BA/DDRecall更低。以文末 Phase-3E update、最新README/正式报告为准；旧FOE01不得用于本阶段选择或独立验证。**

更新时间：2026-09-23  
服务器项目：`/home/zyt/MFAM`  
当前研究与归档根目录：`/home/zyt/deep_final`  
当前主模型：**STR-01（Structured Token Residual Readout）**  
当前状态：STR-01为strong pure-deep backbone；V8-GN仅作matched reference；WSR-01已拒绝；本次交接不包含任何新训练或模型修改。

## 1. 证据优先级与历史差异

恢复上下文时按以下优先级取证：

1. `README.md`中日期最新的正式章节及对应artifact报告；
2. artifact内的protocol、paired statistics、CSV/JSON和提取校验；
3. 本交接摘要；
4. 早期阶段报告与聊天记录。

`README.md`顶部仍保留“V8最终归档”的历史说明，因为它记录原始BatchNorm V8的一次性nested-CV归档；这不再代表当前development主模型。README的`STR-01 Structured Token Residual Readout`、`WSR-01`及`STR-01 Gain Mechanism Diagnosis`章节和相应正式artifact supersede旧状态：**当前主模型是STR-01，V8-GN是matched reference。**

## 2. 当前研究任务、数据与输入

当前实际任务是PADS数据集上的**受试者级PD vs DD二分类**，不是UPDRS/MDS-UPDRS连续严重程度回归。

- 390名受试者：PD 276、DD 114；Healthy不进入当前任务。
- DD组成：Other Movement Disorders 60、Essential Tremor 28、Atypical Parkinsonism 15、Multiple Sclerosis 11；亚型样本较小，只能探索性解释。
- 每名受试者包含11项固定活动：CrossArms、DrinkGlas、Entrainment、HoldWeight、LiftHold、PointFinger、Relaxed、RelaxedTask、StretchHold、TouchIndex、TouchNose。
- 每项活动包含左/右腕、Acc XYZ与Gyro XYZ，共6通道，标称采样率100 Hz。
- 输入张量为`[B, 11, 2, 6, T_a]`，活动真实长度为976或2000；不同活动不拼成长序列。
- Acc执行L1 trend removal并删除前48个采样点；Gyro保留原信号。
- normalization及所有需要拟合的统计量只使用当前训练fold受试者。

该任务可作为纯深度分类和跨个体表征研究平台；没有独立中心、新设备或新队列验证，也不能直接外推为连续临床严重程度评估。

## 3. 固定development与统计协议

- 固定subject-level `5 outer contexts × 3 inner folds`，共15个inner-development validation splits。
- split SHA-256：`b3c52317cb12b73c66046bbd7c50c94a7e707a64a38d3ad4f0fd4bc0c6ba6b3e`。
- seed固定为42/43/44；三个seed共享相同subject splits。
- 主要统计流程：先在相同`(outer, inner)`内聚合三个seed，再以15个split为独立推断单位；45个seed-fold只用于原始记录和seed稳定性，不能当作45个独立样本。
- 主指标为Balanced Accuracy；同时报告Accuracy、Macro-Precision、Macro-F1、AUROC、PD/DD recall、seed SD、paired split wins及必要的Wilcoxon、paired effect size和BH-FDR。
- probe、scaler、prototype、centroid或校正统计量只能在当前inner-train拟合，unseen inner-validation只评估。
- 不根据单seed、单fold或偶然最高值选择模型；负结果必须保留。

## 4. 当前主模型：STR-01（正式模型介绍）

STR-01是面向双腕、多活动IMU的端到端纯深度受试者级分类网络，共有**75,524个可训练参数**。模型不接收H1 handcrafted features、预定义bandpower、dominant frequency或外部临床信息。

### 4.1 共享wrist encoder

每项活动的左右腕信号分别进入权重共享的wrist encoder；同一编码器也跨11项活动共享。

**Temporal stream**

1. `Conv1D(6→64, kernel=15, stride=2, padding=7, bias=False)`；
2. `GroupNorm(8) + GELU`；
3. 三个64通道depthwise-separable residual blocks，`kernel=7`、dilation依次为`[1,2,4]`；每个block为depthwise Conv → GroupNorm → GELU → pointwise `1×1` Conv → GroupNorm → GELU → Dropout(0.1)，并加入残差；
4. feature-statistics pooling同时计算learned attention mean、global mean与global std，拼接为192D；
5. `Linear(192→64) + LayerNorm + GELU`，得到64D temporal embedding。

**Learned-moment stream**

1. 三个并行可学习Conv1D分支，kernel为`[1,15,63]`，每支路16通道并接GELU；
2. 拼接为48通道，经`Conv1D(48→32, kernel=1) + GELU`；
3. 在未做预池化归一化的learned feature maps上计算global mean与std，拼接为64D；
4. `Linear(64→32) + LayerNorm + GELU`，得到32D moment embedding。

两个分支拼接为96D，经`Linear(96→64) + LayerNorm + GELU`形成每腕64D embedding。

### 4.2 Full bilateral activity representation

对每项活动，将`Left 64D`、`Right 64D`、masked bilateral mean 64D、双腕均有效时的`|Left−Right| 64D`及2D wrist-valid mask拼接，形成**258D bilateral activity representation**。

### 4.3 Activity-attention subject path

11个258D activity representations分别加上learned activity-ID embedding并经LayerNorm。attention scorer为`Linear(258→64) → Tanh → Dropout(0.1) → Linear(64→1)`；masked softmax产生11项活动权重，加权求和得到258D subject embedding。`Dropout(0.2) + Linear(258→2)`输出base logits。

### 4.4 Structured activity residual path

聚合前的11个258D bilateral activity representations各自经过共享`Linear(258→16) + GELU`；activity mask清零缺失活动，固定活动身份和顺序保持为`11×16D`，随后展平为176D。`Linear(176→2)`输出structured residual logits。

### 4.5 输出

最终logits为`base logits + structured residual logits`，Softmax输出PD/DD概率。模型同时可返回base/residual logits、activity attention、subject embedding、structured activity tokens和176D structured representation。STR residual readout包含4,498个参数；总参数量75,524。

## 5. 当前训练配置

```text
loss: train-fold balanced cross entropy
optimizer: AdamW
learning_rate: 2.0e-4
batch_size: 8
weight_decay: 1.0e-4
adam_betas: [0.9, 0.999]
temporal_encoder_dropout: 0.1
activity_attention_dropout: 0.1
classifier_dropout: 0.2
scheduler: cosine
minimum_learning_rate: 1.0e-6
epochs: 50
early_stopping: balanced_accuracy, patience=12
gradient_clip_norm: 5.0
mixed_precision: false
seeds: [42, 43, 44]
```

## 6. 当前development性能

以下均为相同15个fixed inner-development splits、三个seed先聚合的development证据，不是outer结果。

| Model | Params | Accuracy | BA | Macro-F1 | AUROC | PD Recall | DD Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| V8-GN reference | 71,026 | 0.7154 | 0.6707 | 0.6631 | 0.6800 | 0.7782 | 0.5631 |
| STR-01 | 75,524 | 0.7307 | 0.6972 | 0.6861 | 0.7176 | 0.7779 | 0.6165 |
| STR−V8 | +4,498 | +0.0154 | +0.0265 | +0.0230 | +0.0377 | -0.0004 | +0.0534 |

- BA：14/15 splits改善，Wilcoxon `p=0.000183`，BH `q=0.000549`。
- AUROC：15/15改善，`p=0.000061`，`q=0.000366`。
- Macro-F1：12/15改善，`p=0.000854`，`q=0.001709`。
- STR seed BA为0.7084/0.6981/0.6850，SD 0.0117；AUROC为0.7276/0.7209/0.7044，SD 0.0120。
- DD recall SD为0.0408，高于V8的0.0134，是必须保留的稳定性限制。
- development-only H1 diagnostic为BA 0.6918、AUROC 0.7515、Macro-F1 0.6858、PD/DD recall 0.7936/0.5900。STR的BA略高，但AUROC仍低0.0339。

## 7. 影响当前决策的模型演化

| 节点 | 关键变化 | 结论 |
|---|---|---|
| M0 | 固定频带、MS-CNN、Hard Top-K MIL | 初始参照；开发BA/AUROC 0.5897/0.6131 |
| Pure-deep V8 | TCN、learned feature-statistics pooling、normalization-free learned-moment stream、Full bilateral fusion、Activity Attention | 三seed BA 0.6572；奠定纯深度主干 |
| V8-GN / NR-01 | temporal BatchNorm改为GroupNorm(8) | BA 0.6707，较V8 +0.0134；成为后续matched reference |
| Strong-backbone benchmark | V8-GN对比ResNet1D与InceptionTime | 标准更大CNN无实质增益；V8-GN最强且最稳定 |
| H1–Deep gap diagnosis | 分解H1并探测各层recoverability | 关键gap定位为activity/wrist-specific联合结构在单一subject vector中的可访问性不足 |
| STR-01 | 将有序11项activity representations直接保持到decision stage | BA/AUROC +0.0265/+0.0377；正式成为当前主模型 |
| WSR-01 | 额外引入pre-fusion wrist-structured readout | recoverability提高但分类未提高；拒绝并停止structured-readout扩展 |

## 8. 主要已完成实验与状态

- **LR-01 long-range dilation：拒绝。** BA 0.6406，较V8低0.0167；不继续dilation/depth搜索。
- **AGG-01 masked mean / AGG-02 mean-attention residual：不替换/拒绝。** 没有一致改善BA、AUROC和F1；activity aggregation搜索停止。
- **NR-01 GroupNorm：保留为V8-GN reference的关键组成。** 是早期基础架构中唯一达到预登记收益的单因素修改。
- **Training-recipe audit：原recipe保留。** LR 1e-4、batch16、WD1e-3、dropout0.2、constant LR均未满足联合门槛；停止在同一development数据上调参。
- **PA-01/PA-02 prototype alignment：拒绝。** 未同时保持分类并稳定降低subject information/shift；不追加lambda，不由此恢复DANN/MMD。
- **MF-01 Mean-only fusion：拒绝。** 参数减少21.58%，但BA/AUROC/F1下降0.0213/0.0129/0.0264；Full fusion具有线性单分量诊断未揭示的互补信息。
- **ResNet1D：拒绝。** BA/AUROC较V8-GN低0.0237/0.0260。
- **InceptionTime：拒绝替换。** 约6.8倍参数，BA -0.0014、AUROC +0.0019、F1 -0.0052，且recall波动更大。
- **Gyro-only learnable rFFT branch：拒绝。** BA/AUROC/F1分别下降0.0146/0.0163/0.0151，seed方差恶化；FFT/STFT/wavelet/spectral Transformer路线停止。
- **STR-01：保留。** 满足预登记分类、paired-split和类别recall条件。
- **WSR-01：拒绝。** BA -0.0007、AUROC +0.0039、F1 -0.0036，所有分类比较BH后不显著并出现PD/DD trade-off；不搜索wrist projection/gating/attention。

## 9. 关键representation与机制证据

### 9.1 V8-GN基础表征诊断

- V8 subject embedding的train-fitted disease probe BA/AUROC为0.6126/0.6599，cosine silhouette仅0.0360：有可泛化疾病信息，但PD/DD不是紧凑双簇。
- 跨activity subject retrieval Top-1为6.32%，随机期望0.96%，45/45 seed-fold高于随机；疾病信号嵌在更强个体变异中。
- activity probe为69.50%；CrossArms平均attention 0.5029，但attention跨fold极不稳定，不能当作因果重要性。
- disease-controlled train/validation domain AUC为0.6334。按15个独立split收口后，domain AUC、mean shift和CORAL与BA均不显著；不能宣称shift已被证明导致错误。
- 历史定义下V8-GN有54/390 stable-error subjects（DD 34、PD 20）；局部错误类邻域和activity dispersion是主要描述性特征。

### 9.2 Layer-wise与bilateral证据

- disease probe AUROC从wrist 0.6474、bilateral 0.6549、activity context 0.6587、subject 0.6599到logits 0.6813；activity→subject聚合提高疾病几何，不是一般疾病信息丢失点。
- 但bilateral fusion使silhouette下降、domain AUC和CORAL上升；Mean分量疾病信息最强，AbsDiff disease information较弱且CORAL最高。
- 45个V8-GN seed-fold中左右腕均完整，wrist mask恒为`[1,1]`；mask在当前数据中不提供疾病信息，只保留接口意义。
- Mean-only端到端分类明确下降，说明Left/Right/AbsDiff仍含非线性互补信息；不能仅凭probe删除分量。
- DD subtype silhouette在各层为负，现有样本不支持subtype-aware auxiliary supervision。

### 9.3 Stable-error phenotype与数据审计

- 8580条双腕记录均finite、时间戳单调；采样率99.2065–100.8077 Hz，长度固定976/2000，无系统性硬件损坏、冻结或丢包证据。
- DD subtype整体与stable-error无FDR显著关联；年龄、性别、BMI、handedness、device及现有症状词项也未形成稳定关联。当前metadata缺少UPDRS、用药/ON-OFF、严重度和逐活动执行质量。
- DD stable-error在CrossArms、DrinkGlas呈低频化、3–7 Hz相对功率下降和更强双腕同步；HoldWeight/LiftHold幅值也更接近PD stable-correct。方向跨15 splits稳定，但这没有转化为frequency branch分类收益。
- 原始time/frequency空间、V8 representation、centroid/kNN与多activity evidence均显示错误受试者具有对侧类别表型；unstable处于中间。当前最强解释是PD/DD运动表型或标签空间重叠，而非坏数据。
- 该证据仍无法区分真实临床重叠、诊断边界、共病、病程/用药或动作执行差异。

### 9.4 H1–Deep representation gap

- H1真实4928D schema为`11 activities × 2 wrists × 8 signals × 28 statistics`，分为六个真实feature families。
- H1优势不能由单一family解释；主要稳定贡献来自`time_location_scale`与`band_fraction`的跨family、跨activity、双腕联合结构。
- V8 wrist/activity-context已能恢复部分H1信息：activity-context OOS R²对time/location/scale为0.394、band fraction为0.212；因此不是底层完全未编码统计或频率信息。
- V8最终subject embedding对完整family结构的R²均为负：time/location/scale -0.689，band fraction -1.413。明确gap是细粒度activity/wrist structure在单一subject vector中的可访问性不足。
- H1只纠正44.3%的V8 errors，对stable-error appearances仅纠正30.0%；大多数核心overlap受试者二者共同失败。

### 9.5 STR-01机制与剩余问题

- 1560个seed-aggregated validation appearances中：both-correct 1029、V8 wrong→STR correct 139、V8 correct→STR wrong 93、both-wrong 299；净修复46次，主要来自DD净+35，PD净+11。
- rescue主要是near-boundary errors；只有6名subject在四个validation appearances中至少3次持续被rescue。STR没有形成大型永久修复亚群。
- STR original/base-only BA/AUROC为0.6299/0.6793；residual-only为0.6561/0.7209；combined为0.6972/0.7176。训练将疾病证据重新分配到两条path，二者互补但不正交。
- 在139次rescue中，59次由residual直接翻转，80次由联合训练后的base已纠正；residual也直接造成42次harm。不能把STR描述为简单附加的独立纠错器。
- rescue贡献分布于多个activity；没有单一activity的正贡献通过BH。PointFinger和Entrainment的单activity evidence反而下降。STR收益来自distributed ordered cross-activity decision pattern。
- seed-first定义下stable-error由85降至74；历史12 seed-fold定义由54降至42。多数原stable-error仍持续错误或仅变unstable，核心phenotype overlap未解决。
- STR combined representation的disease-probe BA/AUROC相对V8提高0.0374/0.0465，但Euclidean silhouette下降0.0107、subject retrieval下降、domain AUC/mean shift无改善，centroid drift恶化。
- 264项rescue-vs-both-wrong raw-signal比较无一通过BH；没有可辩护的稳定raw rescue phenotype。

## 10. 当前结论分级

### 实验确认

1. STR-01是当前fixed inner-development协议下最强且最稳定的pure-deep模型，BA/AUROC显著高于matched V8-GN。
2. 保留有序activity-conditioned representations到decision stage能提供稳定分类增益；额外保留pre-fusion wrist structure只提高recoverability，不能提高分类。
3. 疾病、activity和subject-specific information在表示中并存；当前疾病几何分离弱，个体差异强。
4. STR增益主要修复边界受试者并改善DD识别，真正持续修复的受试者很少。
5. STR没有改善全局silhouette、cross-activity subject consistency或train/validation invariance；多数stable-error仍存在。
6. ResNet1D、InceptionTime、Gyro FFT、prototype alignment、Mean-only fusion及WSR-01均不满足预登记替换条件。

### 合理推测

1. 当前主要瓶颈已从“activity信息过早压缩”转向**跨个体泛化表示不稳定**：疾病判别信息可读，但仍受强subject variability和split-dependent geometry影响。
2. stable-error更可能代表动作/临床表型与PD/DD标签空间重叠，而不是系统采集失败。
3. disease-preserving、class/activity-conditional的跨个体表示学习可能比继续扩展decision readout更值得研究；但必须先找到明确训练目标，不能把移除全部subject information当作默认正确目标。

### 尚未验证

1. STR-01的outer-test、独立队列、跨中心、跨设备泛化；不得用原V8 outer结果代替。
2. 哪种具体subject-invariant或split-stable learning objective能提高STR分类。
3. subject variability的临床来源：UPDRS、用药/ON-OFF、严重度、共病或动作执行质量。
4. phenotype overlap究竟来自真实疾病重叠、诊断标签边界还是协议执行差异。
5. STR是否能改善连续严重程度回归；当前任务没有相应标签或证据。

## 11. 当前核心性能与representation gap

- STR development AUROC 0.7176仍低于H1 0.7515，gap为0.0339；不能宣称纯深度模型已超过handcrafted/statistical diagnostic。
- STR提高decision-accessible disease information，却没有形成更紧凑、更subject-consistent或更domain-invariant的表示。
- 多数V8 stable-error未被STR真正转为stable-correct，both-wrong群体仍表现为对侧类别表型。
- 当前最重要研究问题是：**如何在保留activity-conditioned疾病信息和Full bilateral互补性的同时，降低与疾病标签无关、跨subject/split不稳定的表示变化，并把这一变化转化为unseen-subject分类收益？**

## 12. 已停止或不得自动重新开启的方向

- STR-02、更多structured-readout维度/深度、decision gating、additional activity attention或aggregation搜索。
- WSR projection、wrist gating/attention或新的bilateral fusion搜索。
- Mean-only、AbsDiff normalization、weighted mean及由MF-01派生的fusion变体。
- ResNet1D/InceptionTime专属调参、大型Transformer或无边界backbone搜索。
- 通用FFT/STFT/wavelet/spectral Transformer、频带或FFT range/bins搜索。
- 当前prototype alignment lambda搜索；不得将PA失败后自动升级为DANN/MMD/CORAL loss。
- H1 handcrafted input、H1 reconstruction auxiliary loss或把probe recoverability当作分类替代指标。
- 围绕LR、batch、WD、dropout、scheduler继续调参。
- 仅凭attention、单activity或未校正raw phenotype结果设计新模块。

若未来出现新的、跨split稳定且与疾病标签不等价的机制证据，可重新预注册单一实验；否则不要因为已有路线而强行恢复上述方向。

## 13. Development状态与outer/provenance边界

1. 原始BatchNorm V8在模型冻结后合法执行过一次outer nested-CV：Accuracy 0.6615、BA 0.6116、Macro-F1 0.6064、AUROC 0.6347、PD/DD recall 0.7319/0.4912。该结果只属于原始V8历史归档。
2. outer-test随后永久封存。V8-GN、STR-01、WSR-01和所有后续诊断均没有可用于开发或选择的新outer结果。
3. **STR-01尚未做outer evaluation。** 不得把原始V8 outer结果、MF-01误生成结果或任何旧模型结果当作STR证据。
4. MF-01 seed42曾误用通用nested runner：15个inner splits后自动生成outer-final/test产物。该执行偏差必须保留；结果未进入MF-01选择或后续分析。不得读取、汇总、比较或引用该误启动outer产物，也不得声称它从未产生。
5. seed43/44 MF-01及Strong Backbone、H1 gap、STR、WSR和gain diagnosis均使用development-only协议；protocol记录outer loader/signal/predictions为false。
6. 后续脚本必须显式限制在`outer_*/inner_*` development路径，禁止进入`outer_final`、根级outer predictions或`artifacts/final_nested_cv`进行方法选择。

## 14. 下一步唯一推荐任务

下一会话首先开展**STR-01 Cross-Individual Generalization Target Audit（只读、development-only）**，暂不训练新模型。

目标不是再验证H1 recoverability，而是利用现有STR/V8 inner-development checkpoints与archives，定位一个可被后续单因素表示学习明确针对的跨个体泛化目标：

1. 在wrist 64D、bilateral 258D、activity-context 258D、subject 258D、structured 176D、combined decision representation及logits层，统一比较class-conditional subject retrieval、domain/split predictability、centroid drift与stable-error关系；
2. 所有probe只在inner-train拟合，三个seed先聚合，以15个split或独立subject为推断单位，并做effect size和BH-FDR；
3. activity和disease必须作为条件变量，避免把有效activity结构或疾病差异误当成nuisance删除；
4. 只在发现“跨split稳定、与错误相关、非疾病标签同义且推理时不依赖outer”的具体目标后，预注册一个单因素representation-learning实验；
5. 若没有明确目标，不训练泛化模块，停止PADS架构扩展，优先新队列/外部验证或获取连续严重度标签。

本任务不等于恢复DANN、MMD或prototype alignment；这些方法不能在目标变量未明确前自动启用。

## 15. 后续真正需要的路径

### 状态与主代码

- 当前README：`/home/zyt/deep_final/README.md`
- 本交接文件：`/home/zyt/deep_final/CONTEXT_HANDOFF.md`
- 当前开发工程：`/home/zyt/deep_final/foundation_validation/`
- STR/V8/WSR模型实现：`/home/zyt/deep_final/foundation_validation/src/models/pure_deep.py`
- Activity Attention：`/home/zyt/deep_final/foundation_validation/src/models/activity_fusion.py`
- Full bilateral fusion：`/home/zyt/deep_final/foundation_validation/src/models/bilateral.py`
- 固定splits：`/home/zyt/deep_final/splits/`
- 原始项目：`/home/zyt/MFAM/`

### STR配置与checkpoint输出

- configs：`/home/zyt/deep_final/foundation_validation/configs/str01_seed42.yaml`、`str01_seed43.yaml`、`str01_seed44.yaml`
- seed42输出：`/home/zyt/deep_final/foundation_validation/outputs/structured_token_residual/str01_seed42_20260921/`
- seed43输出：`/home/zyt/deep_final/foundation_validation/outputs/structured_token_residual/str01_seed43_20260921/`
- seed44输出：`/home/zyt/deep_final/foundation_validation/outputs/structured_token_residual/str01_seed44_20260921/`
- 每个split的best checkpoint位于对应`outer_*/inner_*/checkpoints/best.pt`。

### 当前最关键报告与artifacts

- STR正式报告：`/home/zyt/deep_final/artifacts/str01_structured_token_residual_20260921/STR01_REPORT.md`
- STR gain机制：`/home/zyt/deep_final/artifacts/str01_gain_mechanism_diagnosis_20260923/STR01_GAIN_MECHANISM_DIAGNOSIS_REPORT.md`
- WSR负结果：`/home/zyt/deep_final/artifacts/wsr01_wrist_structured_residual_20260922/WSR01_REPORT.md`
- H1–Deep gap：`/home/zyt/deep_final/artifacts/h1_deep_gap_diagnosis_20260921/H1_DEEP_REPRESENTATION_GAP_DIAGNOSIS_REPORT.md`
- Strong Backbone Benchmark：`/home/zyt/deep_final/artifacts/strong_backbone_benchmark_20260921/STRONG_BACKBONE_BENCHMARK_REPORT.md`
- Stage 2 representation diagnosis：`/home/zyt/deep_final/artifacts/representation_diagnostics/v8_gn_stage2/REPRESENTATION_DIAGNOSTIC_REPORT.md`
- Layer-wise diagnosis：`/home/zyt/deep_final/artifacts/representation_diagnostics/v8_gn_layerwise_20260920/LAYERWISE_REPRESENTATION_DIAGNOSIS_REPORT.md`
- Bilateral diagnosis：`/home/zyt/deep_final/artifacts/representation_diagnostics/v8_gn_bilateral_components_20260920/BILATERAL_FUSION_COMPONENT_DIAGNOSIS_REPORT.md`
- Stable-error phenotype/data audit：`/home/zyt/deep_final/artifacts/representation_diagnostics/v8_gn_stable_error_phenotype_data_audit_20260921_r4/STABLE_ERROR_PHENOTYPE_DATA_AUDIT_REPORT.md`
- Error-oriented diagnosis：`/home/zyt/deep_final/artifacts/representation_diagnostics/v8_gn_error_oriented_20260921/ERROR_ORIENTED_DIAGNOSIS_REPORT.md`
- Prototype alignment负结果：`/home/zyt/deep_final/artifacts/invariant_representation/pa_stage3_20260919/PROTOTYPE_ALIGNMENT_REPORT.md`
- Mean-only负结果：`/home/zyt/deep_final/artifacts/ablation/mf01_mean_only_20260920/MF01_FINAL_REPORT.md`
- 原始V8 outer provenance：`/home/zyt/deep_final/artifacts/final_nested_cv/`

### 复现与分析脚本

- development runner：`/home/zyt/deep_final/foundation_validation/scripts/run_pure_deep_development.py`
- STR representations：`extract_str01_representations.py`、`analyze_str01.py`、`finalize_str01_analysis.py`
- STR gain diagnosis：`extract_str01_gain_mechanism.py`、`analyze_str01_gain_mechanism.py`
- H1 gap：`run_h1_deep_representation_gap_diagnosis.py`及相关`extract/analyze_h1_*`脚本
- stable-error phenotype：`run_stable_error_phenotype_audit.py`
- layer-wise/representation diagnosis：`run_layerwise_representation_diagnosis.py`、`run_representation_diagnostics.py`

## Next Session Start Here

1. 当前主模型是**STR-01**，75,524参数；V8-GN仅是matched reference，WSR-01已拒绝。
2. STR在fixed development上达到BA 0.6972、AUROC 0.7176、Macro-F1 0.6861；相对V8-GN为+0.0265/+0.0377/+0.0230，主要净改善DD。
3. 已确认STR通过distributed ordered cross-activity evidence和base/residual互补获得收益，但没有改善global clustering、subject consistency或representation shift；多数stable-error仍存在。
4. 当前核心问题是如何获得保留疾病/activity信息、同时对跨subject/split变化更稳定的表示，而不是继续扩展decision readout。
5. 下一步首先做**STR-01 Cross-Individual Generalization Target Audit**：只读分析现有inner-development representations，定位明确的class/activity-conditional泛化目标；本任务开始时不要训练模型。
6. 不自动重启STR-02、WSR/gating、fusion/activity aggregation、ResNet/Inception、FFT/STFT/wavelet、Transformer、prototype、DANN、MMD、H1输入/重建或超参数搜索。
7. outer information永久禁止用于开发、方法选择和统计推断；STR没有outer结果；保留原始V8合法outer与MF-01 seed42误生成outer的完整provenance。
8. 所有统计继续遵循train-only fitting、seed先聚合、15 splits/独立subject推断及BH-FDR；negative/null结果必须报告。
9. 每完成一个分析或实验，立即把hypothesis、唯一改动、配置、参数量、三seed结果、15-split paired statistics、正负结果和retain/reject理由追加到`README.md`，并同步更新本handoff的状态与下一步。

## 2026-09-28 Phase-3B status update — supersedes the earlier “Next Session Start Here”

The preceding historical start list predates DSG/RGD/PRR/PAG, FOE-01, Phase-2 and Phase-3A/B. In particular, its statements that STR has no outer result and that the next task is a cross-individual target audit are **obsolete**. Read the later formal reports and README appended sections for current evidence.

- **Frozen formal Phase-1 model:** STR-01, 75,524 parameters; original 15-split × 3-seed development means Accuracy 0.7307, BA 0.6972, AUROC 0.7176, Macro-F1 0.6861, PD Recall 0.7779, DD Recall 0.6165. V8-GN is the matched reference. FOE-01 locked outer evaluation has been completed and remains sealed for all later development and model selection; do not use it to select Phase-3B/C methods or thresholds.
- **Stopped routes:** DSG-01 consistency/invariance, RGD/PRR readout/head, PAG information-preservation, Phase-2 standard hyperparameter/loss/augmentation/sampling, Phase-3A SAM and activity auxiliary. Do not reopen backbone, attention, gating, structured-readout, frequency, DANN/MMD/prototype/contrastive or broad external-model searches.
- **Phase-3A:** exact original STR remains pre-Phase-3B single-model best; three-seed STR probability ensemble AUROC 0.7586, with DD Recall trade-off. Source: `artifacts/phase3a_puredeep_20260928/PHASE3A_REPORT.md`.
- **Phase-3B latest development result:** official pretrained OxWearables HarNet10 frozen SSL at STR wrist level gave Accuracy **0.7456**, BA **0.7196**, AUROC **0.7596**, Macro-F1 **0.7066**, PD Recall **0.7823**, DD Recall **0.6569**. Versus original STR, BA +0.0225 (11/15; 95% CI +0.0094 to +0.0362; BH q=0.0134) and AUROC +0.0419 (15/15; CI +0.0298 to +0.0541; q=0.0002). Versus matched random-frozen control, BA +0.0334 and AUROC +0.0506, both stable. BA/AUROC seed variability and prediction disagreement fell. The preregistered frozen-transfer gate **PASS**; this is the new **development-only** best pure-deep configuration. No Phase-3B outer evaluation and no partial/full external encoder fine-tuning were performed. Source: `artifacts/phase3b_external_ssl_20260928/PHASE3B_REPORT.md` and README.
- **Next task, if authorized in a new phase:** preregister exactly one Phase-3C minimal adaptation experiment (last HarNet block unfreezing **or** lightweight adapter, not both) with a smaller encoder LR, fixed inner splits/seeds and random-frozen/B controls as appropriate. Do not start it merely from reading this handoff. The original FOE-01 outer set cannot be reused as an independent evaluation of Phase-3B/C.

## 2026-10-01 Phase-3C final update — supersedes Phase-3B next task

Phase-3C has completed and is archived at `artifacts/phase3c_ssl_adaptation_20260930/PHASE3C_REPORT.md`; the newest README section is the concise source, with protocol, hashes and CSVs in that isolated artifact directory. This is **PURE-DEEP DEVELOPMENT-ONLY** evidence on fixed 15 inner splits × seeds 42/43/44. The original FOE-01 locked outer was not used for Phase-3C model, LR, representation or threshold decisions; it cannot be reused as an independent evaluation of Phase-3B/C.

- **Current best single-model pure-deep development configuration:** Phase-3B WSSL-STR with frozen pretrained OxWearables HarNet10 final 1024D wrist representation, original STR path, balanced CE/recipe and 0.5 threshold. Seed-first means: Accuracy 0.7456, BA 0.7196, AUROC 0.7596, Macro-F1 0.7066, PD Recall 0.7823, DD Recall 0.6569. Original formal STR-01 remains a frozen Phase-1 reference (BA 0.6972, AUROC 0.7176); Phase-3B WSSL has no new independent outer test. The WSSL three-seed ensemble AUROC 0.7816 is an inference-only reference.
- **E1 last-block adaptation:** 45/45 complete, smoke/logits consistency and integrity passed. Versus frozen WSSL BA +0.0048 (10/15; 95% CI −0.0006 to +0.0102; primary BH q=0.2251), AUROC +0.0094 (11/15; CI +0.0030 to +0.0154; q=0.0603). BA gate and threshold-disagreement gate failed. Do not retain or search pretrained LR.
- **E2 train-only OOF threshold:** 135/135 crossfit fits audited; no target-validation/outer use for threshold choice. Single BA 0.7196→0.6771 (0/15 positive), Macro-F1 0.7066→0.6325; ensemble BA 0.7200→0.6845 and Macro-F1 0.7141→0.6432. DD Recall improved only with a substantial PD Recall decline; AUROC unchanged. Keep 0.5. Two analysis-only post-training bugs and fixes are documented in `ANALYSIS_FIX_NOTE.md`; trained outputs unchanged.
- **E3 middle+final SSL:** 45/45 complete and audited. BA +0.0008 (6/15; CI −0.0038 to +0.0054, q=0.8040), AUROC +0.0071 (10/15; CI crosses zero, q=0.2251), DD Recall −0.0167. Do not retain or search layers.
- **E4 fixed subject-count curve:** 270 new runs plus 90 archived 100% references passed ID/label, nested-subset, normalization and checkpoint audits. Frozen WSSL−STR BA and AUROC were positive at 25/50/75/100%, including BH-positive paired primary differences. But the direct 25%−100% interaction was BA +0.0087 (CI −0.0130 to +0.0291) and AUROC +0.0109 (CI −0.0098 to +0.0319); no stable low-sample-specific BA/AUROC amplification. DD Recall interaction at 25% was exploratory (three-contrast BH q=0.0925), with PD Recall trade-off at low fractions. E4 is mechanism-only, not selection.
- **Final Phase-3C decision: CASE D.** No new adaptation/threshold candidate passes the prespecified gate. Stop HarNet encoder fine-tuning, middle-layer and threshold search on these development splits. Keep frozen WSSL-STR as current development best. Next work is paper experiment consolidation and planning genuinely independent validation, with FOE-01 outer excluded from new model selection and not relabeled as Phase-3B/C independent evidence. Do not reopen previously rejected supervised STR readout/head, attention/gating, augmentation/loss, frequency, DANN/MMD/prototype or H1 fusion directions.

## 2026-10-01 Phase-3D final update — supersedes Phase-3C next task

The user explicitly authorized Phase-3D despite the earlier Phase-3C CASE D. It is now complete, with all positive and negative results in `artifacts/phase3d_ssl_adaptation_20261001/PHASE3D_REPORT.md` and the newest README section. This is **PURE-DEEP, DEVELOPMENT-ONLY** evidence using the same fixed 15 subject-level inner splits × seeds 42/43/44. FOE-01 outer information was not used to select, tune or validate the new candidates; no independent outer evidence exists for Phase-3B/D.

- **Current retained development best:** Phase-3B STR-01 + pretrained-**frozen** HarNet10 final1024 wrist residual, original balanced CE recipe, fixed 0.5 threshold. Accuracy 0.7456; BA 0.7196; AUROC 0.7596; Macro-F1 0.7066; PD Recall 0.7823; DD Recall 0.6569. Full system 10,600,580 parameters including frozen HarNet, 143,172 classifier-side trainable. STR-01 remains formal Phase-1 reference; Phase-3C last-block model remains rejected by its preregistered gate despite higher numerical means.
- **E1 adapter:** 45/45 complete, smoke/logits and full audit pass. One 1024→64→1024 residual adapter adds 132,160 trainable parameters. Versus frozen WSSL BA −0.0001 (9/15; CI −0.0053 to +0.0044; final BH q=0.9780), AUROC +0.0048 (10/15; CI −0.0011 to +0.0100; q=0.3616). AUROC seed SD worsened. Reject. Versus Phase-3C *last-block* adaptation, BA −0.0049 and AUROC −0.0046 with CIs crossing zero. Full encoder fine-tuning was not run and cannot be compared.
- **E2 gated wrist fusion:** 45/45 complete, nonzero gate learning audited. Versus frozen BA −0.0019 (4/15; CI −0.0055 to +0.0024; q=0.5048), AUROC −0.0015 (7/15; CI −0.0077 to +0.0039; q=0.9780), DD Recall −0.0218. AUROC seed SD and disagreement worsened. Reject. The conditional **E3 activity-specific gate was not run**, because E2 failed its pre-registered prerequisite; it has no measured effect.
- **E4 train-only masked domain adaptation:** 45/45 complete; no outer raw subjects accessed, no validation windows in self-supervised loss, BN running statistics fixed, feature/checkpoint hashes and subject IDs audited. Masked reconstruction loss improved in all 45 units, but WSSL classification did not: BA −0.0070 (5/15; CI −0.0149 to +0.0003; q=0.3616), AUROC −0.0023 (8/15; CI −0.0080 to +0.0028; q=0.9780), Macro-F1 −0.0095, AUROC seed SD and threshold disagreement worse. Reject. A label-independent smoke equality tolerance was adjusted only after quantifying rare fold-batch floating differences against archived layer4 features, before formal E4; details in `DOMAIN_SMOKE_NOTE.md`.
- **Final Phase-3D decision:** no new candidate met joint BA/AUROC/recall/seed-stability criteria. Keep frozen WSSL-STR, and stop further adapter-width, gated fusion, activity-specific gate and masked-domain-SSL searches on this repeated development cohort. This does not prove frozen SSL is globally optimal. The next research task is paper evidence consolidation and genuinely independent validation planning, not more model selection on the locked FOE-01 outer set or the same 15 development splits. Previously stopped STR readout/head, H1 fusion, attention/backbone, frequency, ordinary loss/sampling/augmentation and invariance routes remain closed.


## Phase-3E update (2026-10-01)

1. **5s prior 没有优于 10s prior。** HarNet5 BA 0.6967、AUROC 0.7238；相对 HarNet10 分别 −0.0229 / −0.0358，只有 4/15 / 1/15 splits 改善，seed 波动增大。停止本次 temporal-scale 路线。官方 family 的维度和参数量也不同，不能将差异仅归因于时间尺度。
2. **本轮 Bio-PM acc-only 未建立有用的增量迁移收益。** BioPM-only BA 0.6438 / AUROC 0.6659；STR+BioPM 为 0.6924 / 0.7181。相对原 STR 的 BA −0.0048，AUROC +0.0005（95% CI [−0.0064,+0.0072]，BH q=1.0）；相对 HarNet10 的两项指标均明显较低。本轮只使用官方 384D 神经 acc-token pooling，不包含默认输出中的未编码 gravity；不能据此否定完整 Bio-PM HAR pipeline。
3. **未证实 Bio-PM pretrained 优于 random-frozen control。** BA 差 −0.0008，7/15 改善，CI [−0.0062,+0.0053]，dz −0.068；AUROC 差 −0.0024，8/15 改善，CI [−0.0109,+0.0062]，dz −0.135；两项 BH q=1.0。DD Recall 的正向点估计伴随 PD Recall 下降，相关 CI 均跨零，不能归因于稳定的预训练收益。
4. **HarNet/BioPM disease complementarity 尚未检验。** E4 的 Bio-PM 价值门槛失败，因此按条件未执行；未测量不等于证明不存在互补信息。
5. **Dual-prior 未训练。** Bio-PM 未稳定接近 HarNet10，也没有满足 E4 证据要求；不存在可报告的 dual-prior 增益。
6. **最佳 retained single-model 仍是 frozen WSSL-STR。** STR-01 + pretrained-frozen HarNet10 final1024 wrist residual，原 recipe 和 0.5 decision rule。Accuracy 0.7456、BA 0.7196、AUROC 0.7596、Macro-F1 0.7066、PD Recall 0.7823、DD Recall 0.6569。本轮没有保留新 single-model candidate。
7. **最高预注册 inference AUROC 为六模型等权 ensemble：0.7870。** Accuracy 0.7541、BA 0.7150、Macro-F1 0.7099、PD Recall 0.8090、DD Recall 0.6210。相对 frozen 三 seed ensemble（AUROC 0.7816 / BA 0.7200），AUROC +0.0054，14/15 改善，CI [+0.0028,+0.0074]，dz 1.127，E0 BH q=0.0427；BA 与 DD Recall 点估计更低。只作为 inference reference，不替代 single-model。
8. **当前不支持新机制或 dual-prior training。** 保留现有模型，停止本次 5s / Bio-PM acc-only integration 变体。External-prior 路线整体未被否定，后续只考虑有官方资产和明确依据的单一 prior 验证，并规划真正独立的验证数据。不得重开已停止的 HarNet adapter/gate/domain/layer 或 STR architecture/loss/sampling/augmentation 搜索；不会自动启动另一个 encoder。

完整记录：artifacts/phase3e_external_priors_20261001/PHASE3E_REPORT.md。以上 supersede 历史 next-step 列表；旧 FOE-01 不得用于本阶段选择或独立验证。


## 2026-10-02 latest window experiment progress: items 01–04

Authoritative new independent study: `artifacts/phase4_window_dynamics_20261002/PROTOCOL.md`, `ITEM01_REPORT.md` through `ITEM04_REPORT.md`. Item01 audited/reused 45 frozen WSSL stages. Item02 extracted true front/back features and masks: mean cache/logits exact. Item03 A1 mean capacity control and item04 A2 back−front control both completed 45 runs and were rejected by preregistered gates. Retained best remains original Phase3B frozen WSSL-STR (BA .719644/AUROC .759553). No outer results accessed. Next numbered task: B1 local raw Acc input control, independently from original WSSL baseline; then updated WSSL errors, conditional subtype auxiliary, and candidate sensitivity only if eligible. No candidate stacking or additional encoder search. Each completed item is Git committed, pushed, tagged; items01–03 tags already verified, item04 publishing follows this record.


## 2026-10-02 final ordered WSSL study — all items resolved

Latest formal record: `artifacts/phase4_window_dynamics_20261002/FINAL_REPORT.md`, reports ITEM01–ITEM08. All180 new training runs complete. A1 mean, A2 delta, B1 raw Acc and fixed-weight C1 DD source-category auxiliary all rejected. Retain original frozen WSSL-STR: Acc .745604, BA .719644, AUROC .759553, F1 .706589, PD .782344, DD .656945. Item08 skipped due no retained candidate; no stopping-rule change or extra refit. Updated WSSL primary stable-error groups PD14/DD24; strict all-seed unanimity remains separate sensitivity. No outer information used, no active experiment and no automatically recommended additional model search. All numbered items committed/pushed/tagged on chouytong/pureDeep; final item08 push/tag follows this closure record.


## 2026-10-02 WSSL single EMA control in progress

User authorized exactly one candidate (EMA) and copy fix/trajectory diagnosis. Item 01 PASS all 45 old checkpoints exact logits; new independent wrapper avoids captured closure. Best remains original frozen WSSL. Protocol artifacts/wssl_ema_20261002/PROTOCOL.md; no outer access, no other candidates.


## 2026-10-03 Latest verified record: WSSL single EMA COMPLETE / REJECT

All5 requested items complete. Isolated copy-independent WSSL interface passes45 old-checkpoint exact logits/caches/reload/RNG; architecture unchanged.45 same-trajectory EMA controls with fixed one-epoch half-life alpha2**(-1/S),S26/27. Ordinary BA chooses common checkpoint epoch; all ordinary weights/epoch metrics/predictions exact archive. EMA BA .683824 vs .719644,AUROC .754912 vs .759553,DDRecall .555377 vs .656945;BA/DD lower15/15;BA seed SD worsens.**REJECT; STOP EMA coefficient/start/combination and stop-rule search.** No further training running or recommended automatically. Retained best remains original frozen WSSL-STR (Acc.745604,BA.719644,AUC.759553,F1.706589,PD.782344,DD.656945). No outer use and no new mechanism/module search. Correctness interface is available at artifacts/wssl_ema_20261002/scripts/independent_wssl.py; report FINAL_REPORT.md. Prior in-progress sections are superseded by this complete record.


## 2026-10-04 current-model audit PASS; single fixed-recipe test protocol

Latest record artifacts/wssl_review_20261004/AUDIT_REPORT.md:15train-only normalization refits/45ckpts exact; full390metadata/8580wrists/SSL-input/cache contract PASS;45frozen-checkpoint sampled forward/ID/reload tests PASS. No newly demonstrated model/data-path bug. Original frozen WSSL remains retained(BA.719644,AUC.759553,DD.656945). Original45ordinary trajectories already exactly reproduced in completed EMA. Late overfitting, no50epoch cap hits. Uniform-index sampling/time gap/clock limitations documented; no data modifications.

Current active user goal is review existing model then controlled tests, not new models. Only justified new contrast: same frozen WSSL classifierLR1e-4 vs inherited2e-4,15splits×3seeds, all other rules unchanged; protocol LR_CONTROL_PROTOCOL.md fixed before results. Prior STR-only LR negative is not a WSSL-specific result. No candidate trained/result yet in this record. All rejected EMA/architecture/loss/augmentation/etc routes remain stopped;FOE01outer excluded. Prior no-active-experiment note superseded only by this authorized review/control.


## 2026-10-04 LR single control entry PASS

Original frozen WSSL audit passed; initialization exact and original-engine LR1e-4 smoke passed. Pre-formal full source/protocol/runner/analysis/launcher hash lock set; only LR1e-4 is allowed for45development stages, frozen2e-4 reference reused. Full results pending; no retention decision. User-authorized current-model review/testing, no newarchitecture/outer use/failed-route reopening. SMOKE_REPORT.md documents pre-formal audit-only修正.


## 2026-10-04 LATEST VERIFIED CLOSURE: current-model audit + single LR COMPLETE / REJECT

Supersedes earlier20261004 pending/running records. Allfour audit areas complete: preprocessing/contracts, train-only normalization/recipe, actual architecture and ID-safe fusion/copy/reload.15normrefits exact45ckpts;390metadata/8580wrists/10920windows PASS;45sampled logits/reload PASS. Prior45ordinary full exact reproduction reused. Best/last actual eval supports late overfit, not maxepoch undertraining. No new current implementation bug warrants formal-model changes.

Single frozen-WSSL classifierLR1e-4 contrast complete45/45; frozen2e-4reference reused; initialization/engine smoke/allstage/finalsource-artifact integrity PASS. BA.725133(+.005489),DD.685352(+.028408),butAUROC.758559(-.000994),F1.706159(-.000431),PD.764914(-.017430);BA9/15,AUC6/15,jointgateREJECT. Positives/uncertainty retained;BA bootstrap-positive≠primaryBHsignificance;DDnot6-metricFDR-confirmed. No global/near-global recipe optimality claim.

Retained official best remains Phase3B frozen WSSL-STR LR2e-4:Acc.745604,BA.719644,AUC.759553,F1.706589,PD.782344,DD.656945;143172trainable/10457408frozen. No newmodel,encoder,loss,augmentation,EMA,threshold or stop-rule. Originalformal/checkpoint/cache/split/prediction/norm hashes unchanged, GPU no training process at closure.

Stop this LR extension and compensatoryWD/epoch/patience/optimizer/scheduler/combination searches;other prior rejected routes remain closed. Next legitimate work: consolidate manuscript and plan genuinely independent validation, without revisitingFOE01outer or claiming it as WSSL confirmation. Allthisstage evidence development-only,15seed-firstoverlapping splits;no45seed-runs/subjects/pairs pseudoreplication. Latestformal report artifacts/wssl_review_20261004/FINAL_REPORT.md,pairedCSV/lr_decision.json/final_integrity_audit.json. READMEincludes allpositive/negative/limitations. Gitfinalclosurepush/tag follows;datasets/weights/individualoutputs excluded.


## 2026-10-07 Frequency-Prior STR — F0 audit PASS / controlled protocol

User explicitly authorized one bounded five-band Frequency-Prior path against original STR-01, with F1 near-capacity control and F0 archived reference. This does not reopen generic gyro-rFFT or WSSL/other searches. Same-seed random initialization training was confirmed; trained STR checkpoints are consistency references only. F0 all45 full validation probabilities/decisions and existing Phase2 reproduced states match (max probability diff1.11e-16);15train-only normalization refits match45formal checkpoints. Reuse F0, no retraining. Current STR BA.697189/AUROC.717631; overall retained WSSL best is not the baseline for this experiment. No outer outcomes/data accessed.

[F0 audit](artifacts/frequency_prior_str_20261007/F0_AUDIT_REPORT.md); [pre-result protocol](artifacts/frequency_prior_str_20261007/PROTOCOL.md). New F1/F2 software/numeric/smoke gates and full development90runs have not yet completed. No new performance conclusion.


### 2026-10-07 Frequency-Prior STR — filter and model tests PASS

Fixed five-band synthetic numeric/edge/autograd tests PASS; F1/F2 additional2,481/2,474params, total78,005/77,998. All45STRcheckpoints×two candidates90sampled cases show exact zero-init logits and exact reload. Fresh3seed initial base/RNG match; two-step training-only scratch gradient, branch update, independent copy and padding/wrist/activity isolation PASS. Production source/STRweights unchanged. Full even reflection is a boundary assumption; ideal masks noncausal/ringing; no physiological guarantee. No performance selection or outer access. [Implementation tests](artifacts/frequency_prior_str_20261007/IMPLEMENTATION_TEST_REPORT.md). Engine smoke/formal90runs not complete yet.


### 2026-10-07 Frequency-Prior STR — original-engine smoke PASS / pre-result freeze

F1/F2 original-engine one-epoch/batch smoke PASS, same fixed recipe and normalization; no scores used for design. Protocol/source/analysis/correctness gates SHA-frozen before formal results; full F1/F2 15splits×3seeds each planned, no other candidate. Frozen retention requires paired robust BA benefit versus both F0/F1 and preset AUROC/F1/recall/seed-variance guard; no post-hoc changes. [Smoke report](artifacts/frequency_prior_str_20261007/SMOKE_REPORT.md); [protocol](artifacts/frequency_prior_str_20261007/PROTOCOL.md). Outer outcomes not accessed; datasets/checkpoints/predictions excluded from publication.
