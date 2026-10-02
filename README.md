# PADS Pure-Deep V8 最终归档

本目录是本轮纯深度模型探索结束后的只读归档。最终冻结模型为
`pads_pure_deep_normfree_moments_v8`（简称 V8），参数量 71,026。归档只复制
已有源码和实验产物，没有重训、续训或重新选择超参数。

## 为什么最终选择 V8

V8 不是按单个 seed、单个 fold 或最高一次结果选出的。选择依据是：

1. 所有候选先在同一固定 subject-level 5×3 inner-development 协议上比较；
2. V8 在预先指定的 seed 42/43/44 上 BA 为 0.6682/0.6596/0.6440，均值
   0.6572±0.0123；相对对应三种子的 balanced-v3，15 个配对 fold 中 11 个
   获胜，平均 BA 增加 0.0144；
3. 后续 V9–V13 均未通过预先定义的提升幅度、配对 fold 胜数、稳定性或校准
   门槛，因此没有用偶然较高的单种子均值替代 V8；
4. V8 冻结后才执行一次完整 5×3 nested-CV 外层测试，每名受试者恰好测试一次。

现有协议没有在全部 390 名受试者上重训一个最终 checkpoint。因此本归档保留
5 个 `outer_k` 最终 checkpoint；它们共同构成对最终 nested-CV 结果的可复现模型
集合。不得把其中某一折的 checkpoint 宣称为“全数据最终模型”，也不得把五折
概率平均的结果当作已经评估过的新 ensemble。

## 模型结构与输入输出

输入张量为 `[B, 11, 2, 6, T]`：11 个活动、左右腕、Acc XYZ + Gyro XYZ，
真实长度为 976 或 2000。每个 outer fold 的 normalization 仅由该 fold 的训练
受试者拟合。

V8 的主要流程为：共享紧凑 TCN → 对网络学习 feature map 做
attention/mean/std pooling → 无归一化的可学习多尺度卷积支路（卷积核 1/15/63）
→ 对该支路学习表征做 mean/std pooling → 双腕晚期融合 → activity attention →
二分类头。模型没有把人工 RMS、峰值频率或预定义频带能量作为额外输入。

输出为 `PD`/`DD` logits、概率、activity attention 及中间表征。正式结果使用每个
outer fold 的 inner-OOF 数据选出的独立 DD 阈值，而不是统一挑选一个测试后阈值。

## 最终性能

一次性 pooled outer test（390 人）的正式 thresholded 结果：

| 指标 | 数值 |
|---|---:|
| Accuracy | 0.6615 |
| Balanced accuracy | 0.6116 |
| Macro precision | 0.6038 |
| Macro recall | 0.6116 |
| Macro-F1 | 0.6064 |
| AUROC | 0.6347 |
| PD recall | 0.7319 |
| DD recall | 0.4912 |
| NLL | 0.6756 |
| two-class Brier | 0.4630 |

混淆矩阵（PD、DD）为 `[[202,74],[58,56]]`。BA、Macro-F1、AUROC 的 95%
分层受试者 bootstrap CI 分别为 `[0.5594,0.6652]`、`[0.5563,0.6581]`、
`[0.5716,0.6973]`。

相对相同 outer-test 受试者上的 M0，V8 的 BA、Macro-F1、AUROC 分别变化
`+0.0181`、`+0.0079`、`+0.0330`；三项配对 bootstrap 差值 CI 都跨 0，不能
据此声称 V8 已稳定优于 M0。V8 的 Accuracy 下降 0.0436，但 DD recall 增加
0.1667，体现了更均衡而非全面占优的分类取舍。

## 文件结构

```text
deep_final/
├── README.md                         使用、结构、性能与模型选择说明
├── FINAL_REPORT.md                   完整阶段报告和实验演化表
├── BUNDLE_MANIFEST.json              归档版本、路径、模型与证据索引
├── SHA256SUMS                        归档内文件哈希（不包含自身）
├── configs/                          V8 原始配置、nested-CV 基础配置和最终解析配置
├── models/outer_0...outer_4/          五个最终 checkpoint 及对应配置/归一化/划分/结果
├── source/                           模型、数据、训练、构建、推理和评估源码
├── frozen_project/                   与 checkpoint source hash 一致的严格加载源码快照
├── splits/                           冻结 subject-level nested-CV 划分
├── artifacts/final_nested_cv/        完整最终运行产物（inner、outer、日志、预测、hash）
├── artifacts/final_analysis/         与 M0 的冻结后配对分析
├── artifacts/development/            V8 三种子和各主要候选的开发结果摘要
├── reports/                          原始实验日志和阶段报告
├── environment/                      依赖声明和实际运行环境记录
└── tools/                            归档校验与 checkpoint smoke test
```

PADS 原始/处理后信号没有复制进本归档；配置保留服务器上的绝对数据路径，确保
归档在当前服务器能复现，也避免复制受控数据。若迁移到其他主机，须单独迁移数据
并以可审计方式更新路径；修改冻结配置会改变 provenance hash。

## 只读验证与加载

在服务器上执行：

```bash
cd /home/zyt/deep_final
/home/zyt/envs/mfam_analysis/bin/python tools/verify_bundle.py
/home/zyt/envs/mfam_analysis/bin/python tools/smoke_load.py
```

`verify_bundle.py` 校验归档哈希、5 个 final checkpoint、配置、归一化和预测产物；
`smoke_load.py` 在 CPU 上依次构建 5 个模型、严格加载权重，并用合成张量执行一次
无梯度前向传播。它不会读取训练数据或启动训练。

对一个真实受试者做单折推理（以下仅示例 outer 0；输出默认 argmax）：

```bash
cd /home/zyt/deep_final/frozen_project
/home/zyt/envs/mfam_analysis/bin/python inference.py \
  --config runtime_configs/outer_0.yaml \
  --checkpoint /home/zyt/deep_final/models/outer_0/final.pt \
  --subject-id SUBJECT_ID \
  --device cpu
```

`frozen_project/` 保留了训练时严格 provenance 所需的精确源码集合、预处理审计
文件和 frozen split；`source/` 则额外保留了冻结后配对分析脚本，便于审阅但不用于
checkpoint 的 source-tree 身份校验。正式 nested-CV 判定必须读取
`artifacts/final_nested_cv/nested_cv_summary.json`
中对应 outer fold 的 threshold。通用 `inference.py` 使用 argmax，不应将其输出与
正式 thresholded 指标混为一谈。

## 解释边界

该模型解决 PADS 上的 PD-vs-DD subject-level 二分类，不是 UPDRS 连续严重程度
回归模型。外层结果波动较大，DD 亚型样本少，尚无独立队列或外部中心验证。因此 V8 可作为可复现的纯深度 baseline
和后续表征研究起点，但尚不能称为临床可靠或跨域泛化已证实的 backbone。

## 冻结后基础架构验证日志

本节记录 2026-09-16 新启动的受限基础架构验证。已查看的 outer-test 被永久封存，
后续所有模型选择只允许使用固定 5×3 inner-development folds。每个候选必须与同
seed V8 配对，并在 seed 42/43/44 下判断；不得根据单 fold 或单 seed 选择模型。

### 持续更新的实验总表

| 实验 | 日期 | 单一主改动 | 参数量 | BA | AUROC | Macro-F1 | PD/DD recall | 相对 V8 | 状态 |
|---|---|---|---:|---:|---:|---:|---|---|---|
| LR-01 | 2026-09-16 | TCN dilation `[1,2,4]` → `[1,4,16,64,128]` | 80,626 | 0.6406±0.0130 | 0.6383±0.0164 | 0.6390±0.0117 | 0.8008/0.4804 | BA -0.0167，3/15 胜 | 拒绝 |
| AGG-01 | 2026-09-16 | Activity Attention → Masked Mean | 51,031 | 0.6590±0.0125 | 0.6719±0.0134 | 0.6502±0.0129 | 0.7659/0.5521 | BA +0.0018，10/15 胜 | 消融保留，不替换 V8 |
| AGG-02 | 2026-09-16 | Mean + gated Attention Residual | 71,027 | 0.6562±0.0060 | 0.6529±0.0026 | 0.6544±0.0053 | 0.8114/0.5010 | BA -0.0010，7/15 胜 | 拒绝 |
| NR-01 | 2026-09-17 | Temporal BatchNorm → GroupNorm(8) | 71,026 | 0.6707±0.0088 | 0.6800±0.0103 | 0.6631±0.0137 | 0.7782/0.5631 | BA +0.0134，10/15 胜 | 保留为新开发 backbone |

### LR-01：Long-range temporal modeling

- **实验编号和日期：** LR-01，2026-09-16。
- **假设与动机：** V8 的局部 TCN 理论感受野约 99 个原始采样点（约 0.99 秒），
  可能无法表征 10–20 秒完整活动的慢变化与跨阶段动态。
- **相比 baseline 的改动：** 只把时序残差块 dilation 从 `[1,2,4]` 改为
  `[1,4,16,64,128]`，理论感受野约 2,571 点（约 25.7 秒）。V8 的数据、预处理、
  learned-moment 分支、双腕融合、activity attention、损失和训练协议保持不变。
- **关键配置：** kernel size 7，feature dimension 64，dropout 0.1，参数量
  80,626，seed 42/43/44；每个 seed 使用固定 15 个 inner folds。模型构建与
  dilation 检查通过，相关单元测试 11 项全部通过。
- **主要结果与 baseline 差值：** seed 42/43/44 的 BA 分别为 0.6256、0.6489、
  0.6473；三种子均值 BA 0.6406±0.0130、AUROC 0.6383±0.0164、Macro-F1
  0.6390±0.0117、PD recall 0.8008±0.0182、DD recall 0.4804±0.0399。
  相对同 seed V8，BA -0.0167（仅 3/15 seed-averaged 配对 fold 获胜）、AUROC
  -0.0210（2/15）、Macro-F1 -0.0175（4/15）、DD recall -0.0216（6/15）。
- **保留规则：** 三种子平均 BA 至少提高 0.01、seed-averaged 15 个配对 fold 至少
  9 个获胜，并且 AUROC 与 Macro-F1 不发生实质回退；否则拒绝。
- **是否保留及原因：** 拒绝。三种子平均 BA、AUROC、Macro-F1 和 DD recall 均
  下降，且 fold 胜数远低于预设 9/15；增加 9,600 个参数并未换来稳定收益。
- **当前结论与下一步：** 当前约 1 秒局部 TCN 感受野不能据此认定为主要瓶颈；
  直接用大 dilation/depth 覆盖完整活动反而更易过拟合。不做 dilation 网格搜索，按
  预定顺序进入 activity aggregation 验证。

### AGG-01：Masked Mean

- **实验编号和日期：** AGG-01，2026-09-16。
- **假设与动机：** 390 名受试者下，学习式 activity attention 可能具有较高方差；
  参数无关的 masked mean 可能以更稳定的方式整合 11 项活动。
- **相比 baseline 的改动：** 仅把 V8 的 activity attention 替换为 masked mean；
  TCN 恢复 `[1,2,4]`，其余数据、moment branch、双腕融合、损失和训练协议不变。
- **关键配置：** 参数量 51,031；复用已有 seed-42 V11 正式开发产物，并新增
  seed 43/44 的固定 15-fold 复现实验。
- **主要结果：** seed 42/43/44 BA 为 0.6727/0.6482/0.6562；三种子均值
  BA 0.6590±0.0125、AUROC 0.6719±0.0134、Macro-F1 0.6502±0.0129、
  PD recall 0.7659±0.0164、DD recall 0.5521±0.0210、NLL 0.6253、Brier
  0.4327。
- **相对 V8：** BA +0.0018（10/15 配对 folds 胜）、AUROC +0.0125（9/15）、
  DD recall +0.0501（9 胜、1 平）；但 Accuracy -0.0185、Macro-F1 -0.0063
  （仅 6/15 胜）、PD recall -0.0466，Brier +0.0040，BA seed SD 也从 0.0123
  略升到 0.0125。
- **是否保留及原因：** 作为聚合消融保留，但不替换 V8。它提供了更高 AUROC 和
  DD recall、较好 NLL，却没有实质 BA 增益，也没有降低 BA seed 方差，并牺牲
  Accuracy、Macro-F1、PD recall 与 Brier。
- **当前结论与下一步：** 简单平均可改变敏感度/特异度取舍，但证据不足以说明其
  是更稳定的主聚合器；继续验证预登记的 mean+attention residual，不调 mean 权重。

### AGG-02：Mean + Attention Residual

- **实验编号和日期：** AGG-02，2026-09-16。
- **假设与动机：** 纯 mean 稳定但可能忽略任务差异；纯 attention 灵活但方差较大。
  以 masked mean 为锚点、只学习 attention 相对 mean 的残差，可能兼顾稳定性与
  activity-specific weighting。
- **相比 baseline 的改动：** V8 编码器完全不变；聚合改为
  `mean + sigmoid(g) × (attention - mean)`，全局标量门控初始为 0.25。
- **关键配置：** activity scorer 与 V8 相同，只新增一个可学习标量，参数量
  71,027；seed 42/43/44、固定 15 folds。初始 gate=0.25 的前向 smoke test 通过。
- **保留规则：** 三种子平均 BA 至少 +0.01；或 BA 与 V8 相差不超过 0.005 且
  seed/fold 方差、AUROC、Macro-F1、NLL/Brier整体改善；BA 配对胜数至少 9/15。
- **主要结果：** seed 42/43/44 BA 为 0.6613/0.6495/0.6579；三种子均值
  BA 0.6562±0.0060、AUROC 0.6529±0.0026、Macro-F1 0.6544±0.0053、
  Accuracy 0.7205±0.0046、PD recall 0.8114±0.0142、DD recall
  0.5010±0.0232、NLL 0.6218、Brier 0.4278。
- **相对 V8：** BA -0.0010（7/15 配对 folds 胜）、AUROC -0.0064（4/15）、
  Macro-F1 -0.0021（6/15）、DD recall -0.0010（5 胜、2 平）。seed 间 BA
  SD 从 0.0123 降到 0.0060，NLL -0.0117、Brier -0.0009，但判别指标没有
  同步改善。
- **是否保留及原因：** 拒绝。尽管门控残差降低了 seed 间方差和 NLL，并保持
  BA 接近 V8，但未达到预登记的 9/15 BA 配对胜数；AUROC、Macro-F1 也发生回退，
  因而不能把“更平滑的种子均值”解释为稳定泛化提升。
- **当前结论与下一步：** mean 锚定能调整方差和类别召回取舍，但三种聚合方式中
  没有一种在 BA、AUROC、Macro-F1 上一致优于 V8 attention。结束聚合搜索，进入
  BatchNorm 与 batch-statistics-independent normalization 的单因素审计。

### NR-01：BatchNorm vs GroupNorm

- **实验编号和日期：** NR-01，2026-09-17。
- **假设与动机：** V8 batch size 为 8，且同一套 temporal encoder 接收不同
  activity、腕侧和序列长度；BatchNorm 的 batch/running statistics 可能对受试者
  组成敏感。GroupNorm 不依赖 batch statistics，可能改善跨受试者开发折稳定性。
- **相比 baseline 的改动：** 仅把时间分支 stem 与三个 separable residual blocks
  中的 BatchNorm1d 替换为 GroupNorm；组数为 8。TCN `[1,2,4]`、moment branch、
  V8 activity attention、双腕融合、损失、数据与训练协议均保持不变。
- **关键配置：** seed 42/43/44、固定 15 folds；GroupNorm affine 参数与 BatchNorm
  affine 参数量相同，总参数量 71,026。构建检查确认模型含 7 个 GroupNorm、0 个
  BatchNorm；纯深度单元测试 11 项全部通过。
- **保留规则：** 三种子平均 BA 至少 +0.01；或 BA 与 V8 相差不超过 0.005，且
  seed/fold 方差、AUROC、Macro-F1、NLL/Brier 整体改善；BA 配对胜数至少 9/15。
- **主要结果：** seed 42/43/44 BA 为 0.6778/0.6609/0.6732；三种子均值
  BA 0.6707±0.0088、AUROC 0.6800±0.0103、Macro-F1 0.6631±0.0137、
  Accuracy 0.7154±0.0177、Macro-Precision 0.6746±0.0157、PD recall
  0.7782±0.0303、DD recall 0.5631±0.0134、NLL 0.6524、Brier 0.4308。
- **相对 V8：** BA +0.0134（10/15 配对 folds 胜）、AUROC +0.0206（8/15）、
  Macro-F1 +0.0066（9/15）、DD recall +0.0611（10 胜、1 平）；BA seed SD
  从 0.0123 降到 0.0088。代价为 Accuracy -0.0063、PD recall -0.0343、
  NLL +0.0188、Brier +0.0021。
- **是否保留及原因：** 保留为新的开发协议 backbone 候选。它满足预登记的平均
  BA 至少 +0.01 和 BA 配对胜数至少 9/15，同时 AUROC、Macro-F1 和 DD recall
  同向改善、BA seed 方差下降。该决定只基于 inner-development，不代表 outer-test
  或外部泛化已经改善。
- **当前结论与下一步：** 小 batch、跨 activity/腕侧共享 encoder 的场景中，
  batch-statistics-independent GroupNorm 是本轮唯一获得稳定判别收益的基础改动。
  基础架构验证至此停止，不继续搜索 normalization 变体；冻结 NR-01，并转入跨个体
  泛化、不变表征与校准研究。由于校准指标和 PD recall 回退，后续需把阈值/校准与
  表征泛化分开验证，不能仅以 BA 宣称全面优于 V8。

### 本轮冻结结论

在固定 5×3 inner-development、seed 42/43/44 且不访问 outer-test 的条件下：

1. 扩大 TCN 感受野失败，说明直接增加 dilation/depth 不是当前主要突破口；
2. Masked Mean 改善 DD recall/AUROC，但未稳定改善 BA/F1；门控残差也未超过 V8；
3. GroupNorm 是唯一满足预登记保留门槛的改动，因此冻结为新的开发阶段 backbone。

冻结不等于最终泛化结论。NR-01 尚未在未触碰外层测试、新队列或跨中心数据上
验证；现有 outer-test 已被查看，禁止再用于本轮模型选择。

## V8-GN 训练 recipe 确认日志

本节记录 2026-09-19 启动的受控训练超参数确认。V8-GN 网络结构、输入、损失定义、
subject splits、inner-development folds、阈值选择和指标实现完全冻结。outer-test 不得
访问。小规模筛选只用于决定候选是否值得进入三 seed 确认，不构成最终模型选择证据；
最终 recipe 必须在 seed 42/43/44 下验证。

当前 recipe：AdamW，LR 2e-4，batch size 8，weight decay 1e-4，encoder/activity/
classifier dropout 为 0.1/0.1/0.2，cosine scheduler（minimum LR 1e-6）。

### 持续更新的训练实验总表

| 实验 | 日期 | 阶段 | 相比当前 recipe 的单一变化 | BA | AUROC | Macro-F1 | PD/DD recall | 配对 fold | 状态 |
|---|---|---|---|---:|---:|---:|---|---|---|
| TR-BASE | 2026-09-17 | 三 seed baseline | V8-GN 当前 recipe | 0.6707±0.0088 | 0.6800±0.0103 | 0.6631±0.0137 | 0.7782/0.5631 | — | 冻结起点 |
| TR-LR01 | 2026-09-19 | seed-42 筛选 | LR 2e-4 → 1e-4 | 0.6840 | 0.6937 | 0.6836 | 0.8134/0.5546 | BA +0.0061，9/15 胜 | 暂定进入后续筛选 |
| TR-LR02 | 2026-09-19 | seed-42 筛选 | LR 2e-4 → 3e-4 | 0.6790 | 0.6935 | 0.6753 | 0.8035/0.5545 | BA +0.0012，5/15 胜 | 拒绝 |
| TR-BS01 | 2026-09-19 | seed-42 筛选 | 在 LR=1e-4 下 batch 8 → 16 | 0.6821 | 0.6905 | 0.6833 | 0.8225/0.5417 | BA -0.0018，6/15 胜 | 拒绝，保留 batch 8 |
| TR-WD01 | 2026-09-19 | seed-42 筛选 | 在 LR=1e-4 下 WD 1e-4 → 1e-3 | 0.6863 | 0.6967 | 0.6838 | 0.8043/0.5683 | BA +0.0023，3 胜/10 平 | 拒绝，保留 WD 1e-4 |
| TR-DO01 | 2026-09-19 | seed-42 筛选 | temporal dropout 0.1 → 0.2 | 0.6831 | 0.6962 | 0.6811 | 0.8088/0.5574 | BA -0.0009，6/15 胜 | 拒绝，保留 0.1 |
| TR-SC01 | 2026-09-19 | seed-42 筛选 | cosine → constant LR | 0.6814 | 0.6943 | 0.6774 | 0.8050/0.5579 | BA -0.0025，3 胜/4 平 | 拒绝，保留 cosine |
| TR-FINAL | 2026-09-19 | seed 42/43/44 确认 | LR=1e-4，其余保持原 recipe | 0.6749±0.0133 | 0.6821±0.0100 | 0.6718±0.0128 | 0.8028/0.5470 | BA +0.0043，11/15 胜 | 拒绝；冻结原 recipe |

### TR-LR01 / TR-LR02：Learning-rate 局部确认

- **实验假设：** GroupNorm 版本的部分 folds 训练损失快速接近零而验证指标波动，
  当前 2e-4 可能偏高；但过低 LR 也可能在 early-stopping 窗口内欠拟合。因此只检查
  相邻的 1e-4 与 3e-4，不进行对数网格搜索。
- **单一变化：** 仅修改 AdamW learning rate；架构、batch size、weight decay、
  dropout、cosine scheduler 和全部评估协议保持不变。
- **筛选规则：** seed 42 仅用于方向筛选。候选需相对 seed-42 V8-GN 的 BA 至少
  +0.005、至少 9/15 folds 获胜，且 AUROC/Macro-F1 无明显回退，才可作为后续 recipe
  的暂定起点；最终是否保留必须由 seed 42/43/44 决定。
- **TR-LR01 结果：** Accuracy 0.7379、BA 0.6840、Macro-Precision 0.6965、
  Macro-F1 0.6836、AUROC 0.6937、PD recall 0.8134、DD recall 0.5546、
  NLL 0.6237、Brier 0.4302。相对 seed-42 当前 recipe：BA +0.0061（9/15 胜）、
  AUROC +0.0040（8/15）、Macro-F1 +0.0085（10/15）、DD recall +0.0059
  （8 胜、3 平）。达到筛选门槛，暂定 1e-4 作为后续单因素筛选起点，但尚未完成
  三 seed 确认，不能作为最终 recipe。
- **TR-LR02 结果：** Accuracy 0.7310、BA 0.6790、Macro-Precision 0.6828、
  Macro-F1 0.6753、AUROC 0.6935、PD recall 0.8035、DD recall 0.5545、
  NLL 0.6756、Brier 0.4251。相对 seed-42 当前 recipe：BA +0.0012（5/15 胜）、
  AUROC +0.0038（10/15）、Macro-F1 +0.0001（6 胜、1 平）。拒绝：未达到 BA
  幅度和 fold 胜数门槛，且 NLL 明显恶化。LR 搜索停止，暂定 1e-4 进入下一筛选。

### TR-BS01：Batch-size 确认

- **实验假设：** GroupNorm 已消除 batch-statistics 依赖，在暂定 LR=1e-4 下把 batch
  从 8 增至 16 可能降低梯度噪声，提高 fold 稳定性。
- **单一变化：** 仅把 training batch size 从 8 改为 16；evaluation batch、架构、
  weight decay、dropout 和 cosine scheduler 不变。
- **筛选规则：** 相对 TR-LR01，BA 至少 +0.005、至少 9/15 folds 胜且 AUROC/F1
  不回退才暂定保留；否则回退 batch 8。最终仍须三 seed 确认。
- **结果：** Accuracy 0.7405、BA 0.6821、Macro-Precision 0.6913、Macro-F1
  0.6833、AUROC 0.6905、PD recall 0.8225、DD recall 0.5417、NLL 0.6204、
  Brier 0.4279。相对 TR-LR01：BA -0.0018（6 胜、1 平）、AUROC -0.0032、
  Macro-F1 -0.0003、DD recall -0.0128。
- **是否保留：** 拒绝，回退 batch 8。没有稳定收益，且 DD recall 和 AUROC 回退。

### TR-WD01：Weight-decay 确认

- **实验假设：** 即使 LR 降为 1e-4，若干 folds 仍出现训练损失持续下降而验证回落；
  将 AdamW decoupled weight decay 从 1e-4 提至 1e-3 可能减少过拟合。
- **单一变化：** 在 TR-LR01 的 LR=1e-4、batch=8 基础上，仅修改 weight decay。
- **筛选规则：** 相对 TR-LR01，BA 至少 +0.005、至少 9/15 folds 胜且 AUROC/F1
  不回退才暂定保留；否则回退 1e-4。
- **结果：** Accuracy 0.7353、BA 0.6863、Macro-Precision 0.6943、Macro-F1
  0.6838、AUROC 0.6967、PD recall 0.8043、DD recall 0.5683、NLL 0.6257、
  Brier 0.4312。相对 TR-LR01：BA +0.0023（3 胜、10 平）、AUROC +0.0031
  （8 胜、5 平）、Macro-F1 +0.0001（2 胜、10 平）。
- **是否保留：** 拒绝，保留 weight decay 1e-4。变化主要是少数临界样本翻转，
  BA 幅度和获胜 folds 均未达到筛选门槛，不能解释为稳定正则化收益。

### TR-DO01：Dropout 确认

- **实验假设：** 提高 temporal encoder 内部 dropout 可减少时序特征共适应，同时
  不改变 activity aggregation 和 classifier 的正则化强度。
- **单一变化：** 在 LR=1e-4、batch=8、WD=1e-4 下，仅把 temporal encoder
  dropout 从 0.1 提至 0.2；activity dropout=0.1、classifier dropout=0.2 不变。
- **筛选规则：** 相对 TR-LR01，BA 至少 +0.005、至少 9/15 folds 胜且 AUROC/F1
  不回退才暂定保留；否则回退 temporal dropout 0.1。
- **结果：** Accuracy 0.7353、BA 0.6831、Macro-Precision 0.6919、Macro-F1
  0.6811、AUROC 0.6962、PD recall 0.8088、DD recall 0.5574、NLL 0.6151、
  Brier 0.4222。相对 TR-LR01：BA -0.0009（6/15 胜）、AUROC +0.0026、
  Macro-F1 -0.0025、DD recall +0.0028（4 胜、5 平）。
- **是否保留：** 拒绝，保留 temporal dropout 0.1。虽然校准指标改善，但 BA 与
  Macro-F1 回退，未形成稳定判别收益。

### TR-SC01：Learning-rate scheduler 确认

- **实验假设：** LR=1e-4 时 cosine 在 50 epoch 内可能过早降低有效步长；恒定 LR
  配合 early stopping 可能更好地达到部分 folds 的较晚验证峰值。
- **单一变化：** 在 TR-LR01 基础上仅把 cosine scheduler 改为 `none`，即恒定
  1e-4；AdamW、batch、WD、dropout、epochs 和 early stopping 不变。
- **筛选规则：** 相对 TR-LR01，BA 至少 +0.005、至少 9/15 folds 胜且 AUROC/F1
  不回退才暂定保留；否则保留 cosine。
- **结果：** Accuracy 0.7325、BA 0.6814、Macro-Precision 0.6952、Macro-F1
  0.6774、AUROC 0.6943、PD recall 0.8050、DD recall 0.5579、NLL 0.6301、
  Brier 0.4360。相对 TR-LR01：BA -0.0025（3 胜、4 平）、AUROC +0.0006、
  Macro-F1 -0.0062、DD recall +0.0033（4 胜、6 平）。
- **是否保留：** 拒绝，保留 cosine scheduler。恒定 LR 没有产生稳定收益，且
  Macro-F1、BA 与校准指标回退。scheduler 搜索停止。

### TR-FINAL：三 seed 最终 recipe 确认

- **候选来源：** 五项筛选中，只有 LR=1e-4 达到预登记的 seed-42 方向筛选门槛；
  batch 16、WD 1e-3、temporal dropout 0.2、constant LR 均被拒绝。因此最终候选
  只修改 learning rate，避免组合多个未经支持的变化。
- **最终候选 recipe：** AdamW；LR 1e-4；batch 8；WD 1e-4；encoder/activity/
  classifier dropout 0.1/0.1/0.2；cosine scheduler，minimum LR 1e-6；其余训练和
  评估协议与 V8-GN 完全一致。
- **最终保留规则：** seed 42/43/44 平均 BA 至少 +0.005、15 个 seed-averaged
  配对 folds 至少 9 个获胜，并且 AUROC、Macro-F1 不发生实质回退；同时报告
  PD/DD recall、NLL/Brier 和 seed SD。若不满足，则冻结原 LR=2e-4 recipe。
- **三 seed 结果：** seed 42/43/44 BA 为 0.6840/0.6597/0.6812；均值
  Accuracy 0.7281±0.0086、BA 0.6749±0.0133、Macro-Precision 0.6831±0.0121、
  Macro-F1 0.6718±0.0128、AUROC 0.6821±0.0100、PD recall 0.8028±0.0152、
  DD recall 0.5470±0.0343、NLL 0.6293±0.0049、Brier 0.4329±0.0023。
- **相对原 V8-GN recipe：** BA +0.0043（11/15 配对 folds 胜）、AUROC +0.0022
  （7/15）、Macro-F1 +0.0087（11/15）、Accuracy +0.0127、PD recall +0.0246、
  DD recall -0.0160（仅 4 胜、3 平）、NLL -0.0231、Brier +0.0020。BA seed SD
  从 0.0088 增至 0.0133。
- **冻结结论：** 拒绝 LR=1e-4，不替换原 recipe。虽然多数 folds 的 BA 和 F1
  改善、Accuracy/NLL 更好，但平均 BA 未达到预登记的 +0.005，seed 稳定性恶化，
  DD recall 下降，AUROC 也仅 7/15 folds 获胜。依照“差异不稳定则保留当前配置”
  的原则，冻结原 LR=2e-4 recipe。

### 最终冻结的 V8-GN training recipe

```text
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
```

本轮没有增加优化器对照：AdamW 的核心候选均未形成足够稳定的 recipe 改善，继续
加入优化器会扩大搜索空间，且没有明确的失败机制要求更换 AdamW。训练 recipe 确认
至此停止；不得继续围绕同一开发数据调参，正式进入跨个体泛化与不变表征研究。

## Stage 2：冻结表征诊断日志

本阶段从 2026-09-19 开始。V8-GN 架构和最终 training recipe 完全冻结，不训练新
backbone，不加入 DANN、MMD loss、原型对齐或其他不变学习模块。所有诊断只使用
固定 inner-development train/validation subjects 和对应 best checkpoints；outer-test
信号、预测和指标均不得读取。

### 持续更新的诊断总表

| 编号 | 研究问题 | 主要方法 | 证据范围 | 状态 | 当前结论 |
|---|---|---|---|---|---|
| RD-01 | 258维 subject embedding 是否包含疾病可分信息 | 冻结 embedding、线性 probe、最近类中心、距离与 silhouette | 3 seeds × 15 folds | 完成 | probe BA/AUROC 0.6126/0.6599；有疾病信号但全局簇分离弱 |
| RD-02 | 是否保留 subject-specific information | 跨 activity 受试者检索、方差分解 | validation subjects，3 seeds × 15 folds | 完成 | Top-1 6.32% vs 随机0.96%，45/45 folds 高于随机；个体信息明显 |
| RD-03 | activity 信息及活动间分布结构 | activity linear probe、activity centroid/dispersion、attention | train→validation | 完成 | activity probe 69.50%；attention 平均50.29%集中于 CrossArms |
| RD-04 | unseen-subject representation shift | train-vs-validation domain probe、均值/CORAL shift、类中心漂移 | fold 内比较，3 seeds × 15 folds | 完成 | domain AUC 0.6334；mean shift 与 BA 负相关 ρ=-0.397，p=.0069 |
| RD-05 | 错分受试者是否具有稳定共同特征 | 跨 fold/seed 错误频率、margin、邻域、attention、activity dispersion | 每受试者12次未见验证 | 完成 | 54/390稳定错分；DD和高activity dispersion更易稳定错误 |

### 统一方法与证据边界

- **表示提取：** 每个 inner fold 只加载该 fold 的冻结 best checkpoint 和 train-only
  normalization；提取 train 与 unseen validation 的 258维 subject embedding、11个
  activity embeddings 和 attention。不得构造 outer-test dataset/loader。
- **疾病信息：** StandardScaler 和线性 probe 只在该 fold 的 train subjects 拟合，
  再评估 validation subjects；同时报告非参数距离、最近类中心和 silhouette。
- **个体信息：** 由于 subject-level embedding 每人只有一个样本，不能直接用它做
  subject-ID 分类。改用同一受试者的11个 activity embeddings，在去除训练集 activity
  centroid 后做跨 activity 检索和方差分解；这是“个体信息可恢复性”的诊断，不是
  身份识别部署实验。
- **representation shift：** 只在同一个 checkpoint 的坐标系内比较 train 与 validation；
  不把不同 fold 独立训练模型的坐标直接拼接后解释，因为其表示空间存在旋转/置换
  不可辨识性。
- **可视化：** PCA 仅作辅助；所有结论必须由多 fold/seed 定量统计支持，不根据单张
  t-SNE/UMAP 图下结论。
- **下一步决策：** 只有诊断显示稳定、与错误相关的域/个体混杂证据后，才决定是否
  尝试 adversarial、distribution 或 prototype alignment；不预设具体方法。

### RD-01：258维 subject embedding 的疾病可分性

- **研究问题：** 冻结 V8-GN 的最终表示是否真的包含能泛化到 unseen subjects 的
  PD/DD 信息，还是分类 head 主要依赖偶然边界。
- **方法：** 每 fold 用 inner-train 拟合 StandardScaler 与 balanced logistic probe，
  在 inner-validation 评估；同时计算最近疾病类中心、Euclidean/cosine silhouette 和
  Fisher ratio。模型和 embedding 均不更新。
- **结果：** 45-fold 平均 probe Accuracy 0.6869、BA 0.6126、Macro-F1 0.6138、
  AUROC 0.6599；最近类中心 BA 0.6337；Euclidean/cosine silhouette 为
  0.0501/0.0360。原冻结 head 为 BA 0.6707、AUROC 0.6800。cosine silhouette 与
  head BA 呈正相关（Spearman ρ=0.6290，p=3.70e-6）。
- **结论：** 已确认 embedding 含中等、可迁移的疾病信息，但 PD/DD 并未形成清晰
  紧凑双簇；疾病几何分离程度是 fold 泛化表现的重要伴随因素。
- **证据边界：** probe 与 head 的拟合目标、class weighting 不同，二者差值不能直接
  解释为 classifier 复杂度需求。
- **下一步：** 保留冻结 head；后续对齐必须检查是否损伤 disease probe/silhouette。

### RD-02：跨 activity 的 subject-specific information

- **研究问题：** activity embeddings 是否仍携带同一受试者跨动作稳定的身份信息。
- **方法：** 用 inner-train activity centroids 去除活动均值后，以 validation subject
  某一 activity 检索其余10项 activity 的平均表示；同时比较同受试者和同病种其他
  受试者 cosine similarity，并做辅助方差统计。
- **结果：** Top-1/Top-5/MRR 为 6.32%/17.92%/0.1385，Top-1 随机期望仅
  0.96%，45/45 folds 均高于随机，平均为随机的6.57倍。相同受试者相似度0.1752，
  同疾病其他受试者0.0243，差值0.1509。辅助统计中 disease between-group 项0.44%，
  subject-within-disease 项16.16%，约为前者36.9倍。
- **结论：** 已确认 activity embeddings 明显保留跨动作稳定的个体信息；疾病信号
  是嵌在更强个体变异中的，而非纯疾病簇表示。
- **证据边界：** 方差项不是严格正交因果分解；当前不能判断个体信息来自运动表型、
  人口学、药物状态还是动作质量。
- **下一步：** 增加可获得 nuisance metadata 的只读 probes，定位个体信息来源。

### RD-03：activity 分布与聚合行为

- **研究问题：** activity identity 在表示中是否仍占主导，以及 attention 是否与各活动
  的疾病信息一致。
- **方法：** inner-train→validation 的11类 activity linear probe；逐活动 disease
  probe；汇总冻结 attention，不将 attention 当作因果解释。
- **结果：** activity probe Accuracy 69.50%，远高于随机9.09%。单活动 AUROC：
  CrossArms 0.6135、DrinkGlas 0.6028、HoldWeight 0.5944，最低 Relaxed 0.5375；
  均低于完整 subject head。平均 attention 为 CrossArms 0.5029、DrinkGlas 0.1040、
  HoldWeight 0.0922，其余均低于0.06；活动平均 attention 与单活动 AUROC 排名相关
  （ρ=0.8364，p=0.00133）。
- **结论：** activity 是强表示结构，聚合器总体偏重较有疾病信息的 CrossArms，但
  11项联合仍优于任一单活动。
- **证据边界：** CrossArms 在各 fold 的平均权重范围0.0177–0.9864，attention 数值
  不稳定且不等于因果重要性，不能据此删活动。
- **下一步：** 把 activity 保留为条件变量；后续若做对齐，应避免把 activity 结构误当
  成 subject/domain shift 一并抹除。

### RD-04：inner-train 到 unseen validation 的 representation shift

- **研究问题：** 控制 PD/DD 后，训练和 unseen-subject 表示能否被区分，以及 shift
  是否与分类下降有关。
- **方法：** 分别在 PD、DD 内做交叉验证 domain probe 后取均值；计算 train-scaled
  mean shift、CORAL covariance shift 和疾病类中心漂移。所有比较均在同一 checkpoint
  坐标系内完成。
- **结果：** disease-controlled domain AUC 0.6334，41/45 seed-fold observations >0.5；
  normalized mean shift 0.1143±0.0143，CORAL shift 0.4727±0.0873。DD/PD centroid
  drift 为6.436/2.949。方法学收口后，以15个 seed-aggregated inner splits 为单位：
  mean shift 与 head BA 为ρ=-0.3893、p=0.1515；domain AUC 为ρ=0.0179、p=0.9496；
  CORAL shift 为ρ=-0.3571、p=0.1913，均未达到显著水平。
- **结论：** 已确认 train/validation 表示在描述性统计上可区分，但“shift 大小与 BA
  下降相关”目前仅是方向性迹象，不能作为已经确认的因果或稳定相关结论；也没有证据
  支持全局 covariance alignment 必然改善性能。
- **证据边界：** DD 更少且亚型异质，较大 centroid drift 不能完全归因于模型。
- **下一步：** 先做不训练 backbone 的 train-fitted centering/shrinkage 受控验证；不
  直接进入 MMD/CORAL loss 或 DANN。

### RD-05：稳定错分群体

- **研究问题：** 错误是否跨 seed/fold 重复，以及稳定错误是否有共同表示特征。
- **方法：** 每名受试者汇总12次 unseen inner-validation appearance；错误率≥0.75
  定义稳定错分，≤0.25定义稳定正确；比较 margin、近邻、embedding norm、attention
  和 activity dispersion。
- **结果：** 54/390稳定错分、247稳定正确、89不稳定。DD/PD 平均错误率为
  43.71%/22.19%；稳定错分中 DD 34/114、PD 20/276。稳定错分的 nearest-train-label
  agreement 为24.54%，稳定正确为81.48%（Cohen's d=-3.78）；activity dispersion
  为13.324 vs 12.613（d=0.43，p=0.0173）。embedding norm、置信度和 attention
  entropy/max 无显著差异。MS、Atypical、Other Movement Disorders 的稳定错分率
  较高，但各亚型样本小，只作探索性记录。
- **结论：** 存在重复出现的困难受试者；局部错误类邻域是最强几何特征，跨活动不一致
  有中小效应，可能适合作为可靠性/拒识信号。
- **证据边界：** 类中心 margin 和近邻一致性接近错误定义本身，是描述性证据而非
  独立病因；亚型结果不能支持临床亚组结论。
- **下一步：** 先验证 activity dispersion 的 cross-fitted calibration/abstention 价值，
  再决定是否把它用于不变学习目标。

### Stage 2 阶段判断与方法选择

**实验确认：** 疾病信息、activity 信息和 subject-specific information 在冻结表示中
同时存在；稳定错误尤其集中于 DD 和错误训练邻域。以15个独立 split 收口后，疾病
separability 与 BA 的相关仍稳定，而 mean shift 与 BA 只保留负向趋势。因此当前可以
确认疾病判别信息嵌在较强的个体/活动变异中，但尚不能确认 shift 本身必然导致性能下降。

**仍是推测：** 尚未证明某个明确 nuisance（年龄、性别、药物或动作执行质量）
是这种个体变异的来源，也未证明消除全部 subject information 会改善疾病分类。

因此，本轮不直接加入 subject-ID DANN、MMD 或多个对齐模块。后续按低风险顺序为：

1. 在冻结 embeddings 上做 train-only centering/shrinkage 对照，验证 mean-shift 假设；
2. 对可获得 nuisance metadata 做 probe，并检查其与稳定错误的关联；
3. 对 activity dispersion 做 cross-fitted 风险/拒识验证；
4. 只有找到“可预测、与错误相关、且不等同疾病标签”的 nuisance 后，才选择条件化
   adversarial 或 class-conditional alignment；否则优先采用简单 train-fitted 校正。

详细结果与逐 fold 产物见
`artifacts/representation_diagnostics/v8_gn_stage2/REPRESENTATION_DIAGNOSTIC_REPORT.md`。

### Stage 2 方法学收口：以15个独立 inner split 为统计单位

- **修正原因：** seed 42/43/44 使用相同 subject splits，原45个 seed-fold observations
  不是45个完全独立样本。此前45行相关检验会低估不确定性。
- **修正方法：** 对每个 `(outer, inner)` split 先在三个 seed 上求均值，再以得到的
  15个 split 为主要统计单位做 Spearman 相关；45行仅保留为 seed稳定性和描述性汇总。
- **疾病可分性与BA：** disease-probe BA ρ=0.6524、p=0.00839；probe AUROC
  ρ=0.5929、p=0.01985；centroid BA ρ=0.6393、p=0.01029；cosine silhouette
  ρ=0.6071、p=0.01638。疾病几何可分性与泛化表现的关系在收口后仍成立。
- **shift与BA：** mean shift ρ=-0.3893、p=0.1515；domain-probe AUC
  ρ=0.0179、p=0.9496；CORAL shift ρ=-0.3571、p=0.1913。原先基于45行得到的
  mean-shift显著性被撤回，现仅报告负向趋势。
- **个体信息与BA：** subject retrieval Top-1 ρ=-0.0964、p=0.7325；subject
  similarity gap ρ=-0.1964、p=0.4829。个体信息明确可恢复，但其总体强度尚未显示与
  split-level BA 的稳定单调关系。
## Stage 3：activity-conditioned disease prototype alignment

V8-GN 架构与 frozen recipe 保持不变。首个不变表征方向只增加一个训练期损失，不增加
推理模块，不使用 subject ID，不加入 DANN、MMD 或其他联合对齐。

### 预登记实验表

| 实验 | 唯一变化 | λ | seed / split | 状态 | 保留规则 |
|---|---|---:|---|---|---|
| PA-BASE | V8-GN + classification loss | 0 | 42/43/44，15 matched splits | 已冻结 | 唯一baseline |
| PA-01 | classification + activity-conditioned disease prototype loss | 0.01 | 42/43/44，15 matched splits | 拒绝 | BA/AUROC/F1下降，subject信息未下降，shift变化不稳定 |
| PA-02 | 同PA-01，仅提高alignment权重 | 0.05 | 42/43/44，15 matched splits | 拒绝 | 分类基本持平，但subject信息未下降，shift仅单项趋势 |

### 实现与无泄漏约束

- 每个 epoch 开始时，用**当前 inner-train fold 的全部训练受试者**计算11×2个
  activity×disease prototype；使用独立、非shuffle prototype loader，不读取 validation。
- prototype 建立在 Activity Attention 聚合前的258维 enriched activity embeddings 上。
- alignment loss 使用归一化 cosine distance，包括同 activity/同疾病 compactness 与
  正负疾病 prototype 的 margin separation；总损失为
  `classification_loss + λ × prototype_alignment_loss`，margin固定0.2。
- prototype 为训练期统计量，不是推理参数；validation 只计算 classification loss。
- λ 在运行前固定为0.01和0.05，两个候选都必须完成三个 seed；不做追加网格搜索。
- 主要判断单位仍是15个 seed-aggregated matched splits。除BA、AUROC、Macro-F1、
  PD/DD recall外，必须复算 disease silhouette、subject retrieval、same-subject similarity、
  domain probe AUC和mean shift。

### PA-01：λ=0.01

- **实验假设：** 较弱的activity-conditioned prototype约束可以先降低同activity、同疾病
  的个体离散，而不会明显改变冻结baseline的疾病决策边界。
- **唯一改动：** `classification + 0.01 × prototype_alignment`；V8-GN架构、AdamW、
  LR=2e-4、batch=8、WD=1e-4、dropout、cosine和early stopping均不变。推理参数量
  仍为71,026；prototype仅为训练期fold内统计量。
- **分类结果：** 三seed均值 Accuracy 0.7057±0.0174、BA 0.6692±0.0092、
  Macro-Precision 0.6667±0.0162、Macro-F1 0.6574±0.0137、AUROC 0.6783±0.0071、
  PD recall 0.7570±0.0288、DD recall 0.5814±0.0104、NLL 0.6540±0.0275、
  Brier 0.4363±0.0123。
- **相对baseline：** BA -0.0015（15 matched splits中9胜5负1平）、AUROC -0.0017
  （9胜6负）、Macro-F1 -0.0057、Accuracy -0.0097；PD recall -0.0213，DD recall
  +0.0183，表现为决策平衡移动而不是整体判别改善。
- **表示诊断：** disease probe BA/AUROC +0.0007/+0.0022，cosine silhouette
  -0.0015；subject retrieval Top-1 +0.0003，same-subject similarity +0.0046（10/15
  上升），subject similarity gap +0.0036；domain probe AUC -0.0140（9/15下降，
  Wilcoxon p=0.252），mean shift -0.0008（9/15下降，p=0.421），CORAL -0.0089。
- **是否保留：** 拒绝。分类F1/Accuracy下降，subject-specific information没有减少；
  三项shift虽均值下降，但matched-split一致性和统计证据均不足。
- **结论：** λ=0.01太弱，未形成可确认的不变表征收益；DD recall提升不能单独作为
  保留依据。

### PA-02：λ=0.05

- **实验假设：** 在不改变方法形式的情况下，提高单一alignment权重可能使prototype
  compactness产生可测量的不变性，同时classification loss维持PD/DD分离。
- **唯一改动：** 相对PA-01仅将λ从0.01改为0.05；其余架构、recipe、margin=0.2、
  数据split和评估完全相同。
- **分类结果：** 三seed均值 Accuracy 0.7170±0.0151、BA 0.6715±0.0040、
  Macro-Precision 0.6734±0.0119、Macro-F1 0.6641±0.0088、AUROC 0.6789±0.0072、
  PD recall 0.7811±0.0352、DD recall 0.5618±0.0356、NLL 0.6529±0.0281、
  Brier 0.4334±0.0152。
- **相对baseline：** BA +0.0008（8/15胜，Wilcoxon p=0.679）、AUROC -0.0011
  （8/15胜，p=0.978）、Macro-F1 +0.0010（7/15胜），Accuracy +0.0016；PD recall
  +0.0028，DD recall -0.0013。分类能力整体可视为持平，BA seed SD由0.0088降至0.0040。
- **疾病表示：** disease probe BA +0.0082；probe AUROC +0.0093，在12/15 splits
  提升（Wilcoxon p=0.0125）；cosine silhouette -0.00003，基本不变。原型损失增强了
  线性疾病可读出性，但没有改善最终head AUROC。
- **个体信息：** subject retrieval Top-1 +0.0013；Top-5 -0.0006；same-subject
  similarity +0.0030且12/15 splits上升；subject similarity gap +0.0018。没有证据表明
  subject-specific information减少，反而出现轻微增强迹象。
- **shift诊断：** domain probe AUC -0.0079且12/15 splits下降，但Wilcoxon p=0.073；
  mean shift -0.00265，仅9/15下降（p=0.421）；CORAL +0.00047。只有domain-probe
  方向较一致，其他shift指标不支持稳定下降。
- **是否保留：** 拒绝。虽然分类持平、disease probe改善且domain AUC有下降趋势，但
  预登记目标要求subject-specific information或unseen shift稳定下降；当前subject指标
  未下降，shift证据也未跨指标成立，不能据此替换baseline。
- **结论：** activity-conditioned disease prototype loss更像是增强疾病类中心可读出性，
  而不是消除个体信息。V8-GN + frozen recipe继续作为唯一baseline，本方向不追加λ搜索。

### Stage 3 本轮决策

PA-01与PA-02均已按三个seed和15个matched splits完成，不访问outer-test。没有候选同时
满足“疾病判别不降”和“subject-specific information或unseen shift稳定下降”。因此：

1. 不保留prototype alignment为新backbone；
2. 不继续增加λ，也不叠加DANN、MMD或第二个模块补救；
3. 冻结baseline保持为V8-GN + AdamW + LR=2e-4 + batch=8 + WD=1e-4 + cosine；
4. 本轮得到的积极但不足证据仅限于PA-02 disease probe AUROC改善及domain AUC下降趋势，
   后续若重新研究prototype方法，应先改变“压缩疾病类中心即等于个体不变性”的假设，
   而不是扩大超参数搜索。

## LRD-01：冻结 V8-GN 的 layer-wise representation diagnosis（2026-09-20）

### 研究问题与约束

- **目的：** 定位疾病、subject 和 activity 信息在 wrist encoder、双腕 activity、activity
  context、subject aggregation 与 classifier logits 中的逐层演化。
- **模型状态：** V8-GN、71,026参数和冻结 recipe 完全不变；不训练、不调参、不增加
  DANN、MMD、prototype 或其他训练模块。
- **统计单位：** seed 42/43/44 先在相同 `(outer, inner)` split 内求均值，再以15个
  inner splits 做推断；45个 seed-fold 仅作为原始记录。
- **数据边界：** 只读取 inner-development checkpoint、train 与 unseen-validation；不访问
  outer-test。
- **提取验证：** 自定义逐层提取与冻结模型 forward 的 activity、subject、logits 最大误差
  均为0。

### 实验总表

| 层级 | 维度 | Probe BA | Probe AUROC | Disease silhouette | Disease variance | Activity probe | Subject Top-1 | Domain AUC | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Wrist encoder | 64 | 0.6048 | 0.6474 | 0.0247 | 0.0281 | 0.6672 | 0.0501 | 0.5543 | 已有中等疾病信息，同时保留明显subject/activity信息 |
| Bilateral activity | 258 | 0.6070 | 0.6549 | 0.0189 | 0.0232 | 0.6950 | 0.0632 | 0.5839 | 疾病线性读出略升，但几何分离下降、shift上升 |
| Activity context | 258 | 0.6078 | 0.6587 | 0.0189 | 0.0229 | 1.0000 | 0.0542 | 0.5833 | 显式activity embedding使任务身份完全可读，疾病信息基本保留 |
| Subject aggregation | 258 | 0.6126 | 0.6599 | 0.0360 | 0.0303 | N/A | N/A | 0.6388 | 疾病几何改善，不是activity→subject信息丢失瓶颈；domain可辨识性增加 |
| Classifier logits | 2 | 0.6539 | 0.6813 | 0.1296 | 0.1014 | N/A | N/A | 0.6472 | 监督分类头将已有表征集中到疾病决策轴 |

表中为15个独立 split 均值。Subject Top-1 的平均随机机会为0.0096；三种 token-level
表示均显著高于机会水平。Disease variance 是标准化表示中 disease-between variance占比，
其余约97%在分类头之前仍属于 disease 内受试者变异。

### 相邻层关键变化

- **Wrist→bilateral：** probe AUROC +0.0074（p=0.135），但 cosine silhouette
  -0.0058（14/15下降，p=0.00018）、disease variance -0.00493（p=0.00061）；
  domain AUC +0.0295（12/15上升，p=0.00336），CORAL +0.0900（15/15上升，
  p=0.000061）。双腕融合是第一个稳定放大shift并削弱相对疾病几何的位置。
- **Bilateral→activity context：** probe AUROC +0.00381（12/15上升，p=0.0302），
  activity probe由0.695升至1.0；其他疾病几何基本不变。
- **Activity context→subject：** centroid BA +0.0308（12/15上升，p=0.00451）、
  silhouette +0.0171（14/15上升，p=0.00012）、disease variance +0.00742
  （p=0.00836）。因此多活动聚合没有削弱疾病表征。
- **Subject→logits：** disease variance从0.0303升至0.1014，属于监督 readout 对疾病轴
  的压缩；不能解释为分类头凭空产生新疾病信息。

### 与最终BA的关系

- Wrist、bilateral、activity-context、subject层 probe AUROC 与BA的Spearman ρ依次为
  0.4000（p=0.1396）、0.7214（p=0.00240）、0.6750（p=0.00576）、0.5929
  （p=0.01985）。
- subject层 silhouette ρ=0.6071（p=0.01638），disease variance ratio ρ=0.5607
  （p=0.02968）。
- subject retrieval、domain AUC 和 mean shift 与BA未显示稳定关系。疾病可读出性比总体
  identity/shift强度更能解释split间泛化差异。

### DD亚型与决策

DD subtype silhouette 在所有层均为负；最近训练亚型中心准确率最高仅0.4349，低于验证
DD多数亚型基准0.5265。当前没有证据支持优先加入 subtype-aware auxiliary supervision，
但小亚型样本不足意味着这不是“亚型异质性不存在”的证明。

**实验确认：** 疾病信息没有在activity→subject阶段下降；subject aggregation反而改善疾病
几何。主要薄弱环节位于更早的wrist/activity表示，双腕融合尤其表现为疾病相对几何下降和
shift增加。

**合理推测：** 原始left/right/mean/absolute-difference拼接可能同时放大有用左右差异与
个体尺度差异；subject aggregation可能同时放大疾病判别和域特异性。

**尚未验证：** 尚未证明修改双腕融合可提高BA，也未证明shift是错误的因果来源。

**下一步：** 不优先重做activity aggregation，不启动adversarial/disentanglement。若开展
下一个单因素实验，应先在冻结诊断协议下分解left、right、mean、difference各分量的疾病
读出和shift贡献，再决定是否需要可靠性加权或规范化双腕融合。

完整报告与逐split产物见
`artifacts/representation_diagnostics/v8_gn_layerwise_20260920/`。

## BFC-01：bilateral fusion component diagnosis（2026-09-20）

### 目标与协议

冻结 V8-GN、checkpoint 和全部 training recipe，仅对 wrist encoder 输出构成的 Left、
Right、Mean、AbsDiff 四个同为64维的表示做离线诊断。所有 scaler/probe 只在当前
inner-train 拟合；三个seed先在相同split内聚合，以15个inner splits为统计单位。未训练
新模型、未访问outer-test。

### Wrist mask 审计

全部45个seed-fold的train/validation中，left/right缺失数均为0，bilateral activity不完整
数为0，每个split只有一种mask pattern `[1,1]`。因此wrist mask是常量，只保留接口意义，
不赋予疾病或个体信息解释。

### 四分量总表

| 分量 | Probe BA | Probe AUROC | Disease silhouette | Disease variance | Subject Top-1 | Same-subject similarity | Domain AUC | CORAL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Left | 0.5937 | 0.6309 | 0.0217 | 0.0243 | 0.0427 | 0.1905 | 0.5527 | 0.4052 |
| Right | 0.5933 | 0.6364 | 0.0189 | 0.0233 | 0.0432 | 0.1859 | 0.5335 | 0.4315 |
| Mean | **0.6048** | **0.6474** | **0.0247** | **0.0281** | **0.0501** | **0.2167** | 0.5555 | **0.3751** |
| AbsDiff | 0.5828 | 0.6226 | 0.0121 | 0.0166 | 0.0375 | 0.1191 | **0.5693** | **0.5238** |

表中为15个独立split均值。Mean是疾病读出、疾病几何及跨activity稳定subject signature
最强的等维分量；AbsDiff的疾病信息和稳定subject信息均最弱，但CORAL shift最高。

### Mean 与 AbsDiff 的matched-split证据

- Mean BA高0.0220，12/15 split更高，p=0.0125；
- Mean AUROC高0.0248，13/15更高，p=0.00671；
- Mean silhouette高0.0126，14/15更高，p=0.00043；
- Mean disease variance高0.0115，14/15更高，p=0.00043；
- Mean Subject Top-1高0.0126，15/15更高，p=0.000061；
- Mean same-subject similarity和similarity gap均约高0.0977，15/15更高；
- AbsDiff CORAL比Mean高0.1486，15/15更高，p=0.000061。

AbsDiff AUROC仍为0.6226，说明它并非没有疾病相关信息；但现有结果不支持“raw absolute
difference同时是强疾病不对称通道和强稳定个体通道”。它更突出的是分布敏感性，而非稳定
subject identity。

### Full fusion 相对 Mean

完整258维fusion相对Mean的probe BA/AUROC仅增加+0.0022/+0.0074，均不显著；同时
silhouette下降0.0058、disease variance下降0.0049、domain AUC增加0.0283、CORAL增加
0.0900。Subject Top-1增加0.0131。

因此完整拼接主要增加了容量、identity可恢复性和shift，疾病读出增益较小。由于维度和实际
分类头不同，这仍是描述性证据，不能替代因果ablation。

### 指标与最终BA

四分量的大部分指标与BA未形成稳定关系。Left domain AUC与BA为ρ=-0.7786、
p=0.000627，在本报告全部相关检验BH-FDR后q=0.0351；这是值得复核的线索，但不是
shift导致错误的因果证明。

### 结论分级与下一步

**实验确认：** Mean承载最多共享疾病信息和稳定个体信息；Left/Right没有稳定单侧优势；
AbsDiff疾病信息较弱、subject similarity最低、CORAL shift最高；mask恒定。

**合理推测：** 当前bilateral fusion的主要有效信息来自Mean，raw AbsDiff更接近不稳定的
跨域差异来源；重复拼接Left/Right/Mean提高了可读出容量但没有等比例增强疾病几何。

**尚未验证：** Mean-only是否能保持最终分类性能；AbsDiff shift的具体来源；删除AbsDiff
是否会损失非线性互补信息。

**单一后续决策：** 若进入因果实验，只优先开展一个Mean-only bilateral fusion ablation，
保持其余架构、recipe、split和seed不变。在该结果出来前，不并行搜索scale-normalized
AbsDiff、reliability gating或多种新融合结构。

完整报告与逐split产物见
`artifacts/representation_diagnostics/v8_gn_bilateral_components_20260920/`。

## MF-01：Mean-only bilateral fusion ablation（2026-09-20，已完成）

- **实验假设：** 64维双腕Mean可能保留主要共享疾病信号，同时移除使shift增加的
  Left/Right/AbsDiff冗余维度。
- **唯一结构变化：** activity fusion由
  `[Left, Right, Mean, AbsDiff, wrist mask]`（258维）变为Mean（64维）；仅同步调整
  activity embedding、attention和classifier的输入维度。Wrist encoder、activity
  aggregation形式、loss、AdamW、LR=2e-4、batch=8、WD=1e-4、cosine、数据处理、
  splits和评估协议均保持不变。
- **预登记保留规则：** 三seed BA、AUROC、Macro-F1相对V8-GN均不低于-0.005；
  PD/DD recall任一不下降超过0.02；15个matched splits至少7个BA改善；BA seed SD
  增量不超过0.005；参数至少减少15%；表示shift不得一致恶化。
- **复杂度：** 71,026→55,700参数，减少15,326（21.58%）。
- **实现验证：** 新旧fusion模式向后兼容；Mean为严格masked wrist mean；单腕缺失路径、
  无效mode和参数缩减均有测试。相关测试18 passed。
- **最小smoke：** CPU单batch、单epoch smoke完成，checkpoint/normalization/split/log
  产物正常，outer-test loader/evaluation均为false。Smoke指标仅验证流程，不作为实验结果。
- **开发协议：** 固定5个outer context × 3个inner split；seed 42/43/44先在相同split上
  分别训练，再以15个seed-aggregated split为主要配对统计单位。模型选择未使用outer-test。

### 分类结果

| 模型 | Accuracy | BA | Macro-Precision | Macro-F1 | AUROC | PD recall | DD recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| V8-GN Full fusion | 0.7154 | 0.6707 | 0.6746 | 0.6631 | 0.6800 | 0.7782 | 0.5631 |
| MF-01 Mean-only | 0.6939 | 0.6494 | 0.6587 | 0.6367 | 0.6671 | 0.7571 | 0.5417 |
| 差值（MF-01−V8-GN） | -0.0214 | -0.0213 | -0.0159 | -0.0264 | -0.0129 | -0.0212 | -0.0214 |

MF-01三seed BA为0.6514/0.6557/0.6411，seed SD=0.0075；V8-GN的BA seed
SD=0.0088。15个matched split上，MF-01的BA为4胜11负（平均差-0.0213，
Wilcoxon p=0.0554），Macro-F1为3胜12负（差-0.0264，p=0.0353），AUROC为
6胜9负（差-0.0129，p=0.359）。因此下降不是单个seed或单个fold造成。

### 表示诊断（15个seed-aggregated split）

| 指标 | V8-GN | MF-01 | 差值 |
|---|---:|---:|---:|
| Disease probe BA | 0.6126 | 0.6129 | +0.0004 |
| Disease probe AUROC | 0.6599 | 0.6623 | +0.0024 |
| Disease centroid BA | 0.6337 | 0.6221 | -0.0115 |
| Disease silhouette（Euclidean） | 0.0501 | 0.0530 | +0.0030 |
| Subject retrieval Top-1 | 0.0632 | 0.0503 | -0.0129 |
| Subject similarity gap | 0.1509 | 0.2065 | +0.0556 |
| Disease-controlled domain AUC | 0.6334 | 0.6096 | -0.0238 |
| Normalized mean shift | 0.1143 | 0.1189 | +0.0046 |
| CORAL shift | 0.4727 | 0.3899 | -0.0828 |

Mean-only显著降低subject retrieval Top-1（p=0.000061）和CORAL shift
（p=0.0020），但subject similarity gap反而显著增加（p=0.000061），mean shift轻微
恶化；因此不能表述为shift或subject-specific information“一致下降”。疾病linear probe
基本不变，但实际端到端分类头明显下降，提示Full fusion中的Left/Right/AbsDiff存在仅靠
单分量线性诊断未能揭示的非线性互补信息。

### 决策

**拒绝MF-01，不替换冻结backbone。** 它只满足参数缩减（21.58%）和BA seed SD未恶化，
未满足BA/AUROC/Macro-F1容忍阈值、两类recall阈值、至少7/15 BA胜出以及shift不得一致
恶化等条件。继续保留V8-GN Full fusion（71,026参数）作为唯一冻结backbone；按照预登记，
不由本结果派生AbsDiff normalization、gating、weighted mean或其他fusion搜索。

**协议说明：** seed 42首次误用通用nested runner，runner在15个inner split完成后自动生成了
outer-final/test产物。这是一次执行偏差；相关outer结果未被读取作为模型选择证据、未进入本节
任何数值，并被明确排除。seed 42的开发汇总由已有15个inner checkpoint独立重建；seed
43/44均使用development-only runner，协议文件记录outer-test loader/signal/prediction均未访问。
该偏差不改变MF-01的开发集拒绝结论，但必须在审计中保留，不能声称seed 42运行从未产生过
outer-test产物。

完整结果、配置、三seed开发汇总、配对比较与表示诊断见
`artifacts/ablation/mf01_mean_only_20260920/`及
`foundation_validation/outputs/mean_only_fusion/`。

## ED-01：稳定错分的activity/wrist evidence诊断（2026-09-21）

### 研究问题与协议

在继续冻结V8-GN、training recipe和Full bilateral fusion的前提下，检验稳定错分是否与
activity-level或wrist-level疾病证据冲突和可靠性变化有关。没有训练模型或更新参数。

每个activity及Left/Right/Mean表示均只在当前inner-train拟合StandardScaler + balanced
logistic probe，再应用于unseen inner-validation。三个seed先在相同split内聚合，以15个inner
split为主要统计单位；受试者级和activity级检验分别进行BH-FDR。247名stable-correct、54名
stable-error、89名unstable沿用Stage 2冻结定义。

### 主要结果

| 指标 | Stable-correct | Stable-error | Unstable | Error−Correct | BH-FDR q |
|---|---:|---:|---:|---:|---:|
| 错误方向activity比例 | 0.327 | 0.583 | 0.449 | +0.256 | 0.000106 |
| Activity冲突率 | 0.423 | 0.459 | 0.468 | +0.035 | 0.000106 |
| Activity margin SD | 2.021 | 2.233 | 2.043 | +0.212 | 0.00555 |
| Wrist冲突率 | 0.407 | 0.447 | 0.446 | +0.039 | 0.000187 |
| Wrist disagreement均值 | 2.146 | 2.451 | 2.212 | +0.305 | 0.000106 |

相对stable-correct，activity冲突15/15 split更高，wrist冲突14/15更高。但相对unstable，
activity冲突差异不显著（q=0.0902），wrist冲突完全无差异（q=1.0）。因此冲突能描述
“错误/困难”，但不能单独识别“持续错误”。

### 类别异质性

- **PD stable-error：** activity冲突+0.0697（13/14，q=0.000721），activity SD
  +0.9005（14/14，q=0.000159），wrist disagreement均值+1.0545（14/14，q=0.000159）。
- **DD stable-error：** activity冲突-0.0274（仅5/15更高，q=0.0429），activity SD
  -0.6147（0/15更高，q=0.000159），wrist disagreement均值-0.5777（0/15更高，
  q=0.000159）；但错误方向activity比例增加+0.2057（15/15，q=0.000159）。

PD错误符合“多activity/双腕冲突”模式；DD错误则更像多个activity和双腕一致地提供错误的PD
方向证据。通用可靠性降权可能只帮助PD，不能被假定会修复DD。

### Activity系统性模式

bilateral Mean true-margin中，stable-error相对stable-correct下降最大的是CrossArms
（-4.392）、DrinkGlas（-2.057）、HoldWeight（-1.624）和LiftHold（-1.406），均15/15
split更低且q=0.000279。Relaxed和RelaxedTask无稳定差异。最终错误与错误方向activity比例
的平均rho为0.377，而与activity/wrist冲突率仅为0.123/0.089；多数活动共同朝错误方向比
单纯冲突更接近当前错误机制。

### 结论与停止决策

**实验确认：** 总体stable-error具有更高activity/wrist冲突及离散度，但该模式主要由PD
驱动；DD呈低冲突、方向一致的系统性错误。

**合理推测：** DD错误更可能涉及DD内部异质性、共享表征偏置、标签/临床变量或活动执行
质量，而非单一fusion可靠性问题。

**尚未验证：** 高风险activity是否为因果来源；错误是否与DD subtype、严重程度、用药状态
或采集质量对应；class-conditional可靠性方法是否安全。

**决策：** 当前不启动reliability-aware multi-activity/bilateral fusion训练，也不恢复结构搜索。
优先回到数据、标签、DD subtype和高风险activity的原始信号/元数据审计。

完整报告和逐split产物见
`artifacts/representation_diagnostics/v8_gn_error_oriented_20260921/`。历史outer runner误启动
记录继续保留；本分析只读取inner checkpoint/split/normalization，未读取任何outer信息。

## SEPA-01：stable-error phenotype & data audit（2026-09-21）

### 研究问题与协议

本阶段继续冻结V8-GN、Full bilateral fusion与training recipe，不训练或更新模型。目标是解释
为什么部分DD在多个activity和双腕上稳定呈PD-like表征。分析覆盖390名受试者（DD：47
stable-correct、33 unstable、34 stable-error；PD：200/56/20），只读取45个
inner-development embedding文件、ED-01 inner产物、patient metadata、原始质量记录和处理后
信号；未读取任何outer-final/test prediction、metric或artifact。

三个seed先在同一`(outer, inner)`内聚合。表示分析以15个独立split及subject聚合为单位；
metadata、质量和原始信号检验以每名受试者一行为推断单位。连续变量使用Mann–Whitney U和
Cliff's delta，分类变量使用列联检验/Cramér's V与Fisher exact，并在对应检验族内执行
BH-FDR。split方向一致性只作稳健性检查，不把同一受试者的重复validation appearance当成
独立样本。

### Clinical phenotype与DD subtype

DD各subtype的stable-error计数/总数为：Atypical Parkinsonism 5/15、Essential Tremor
5/28、Multiple Sclerosis 4/11、Other Movement Disorders 20/60。整体subtype分布未通过
多重校正（Cramér's V=0.212，q=0.340），因此没有证据把稳定错误归因于某个特定DD subtype。

DD stable-error的记录疾病时长为10.62年，stable-correct为21.60年（Cliff's delta=-0.419，
q=0.0327），但与unstable的12.55年没有差异（q=0.922）；分subtype后样本很小，方向主要由
ET、MS和Other Movement Disorders中的较短病程贡献，不能视为稳定错误特异的临床机制。
年龄、性别、BMI、handedness、device、subtype以及disease-comment中的tremor、
hypokinesia、rigidity、dystonia、gait、functional、vascular等词项均未形成FDR显著关联。
`appearance_in_first_grade_kinship`的缺失模式达到q=0.0273，但这是metadata completeness差异，
不能解释为临床表型。

现有patient JSON没有UPDRS/MDS-UPDRS、用药/ON-OFF状态、临床严重度或逐activity执行质量
标注；因此这些因素仍未验证，不能把缺失字段当作阴性证据。

### 数据完整性与动作质量代理

8580条双腕记录均为finite且时间戳单调；有效采样率99.2065–100.8077 Hz，所有活动长度固定
为976或2000，最大常值连续段仅3个采样点。全库4条大时间间隙记录均不属于本阶段四个高风险
activity。DD stable-error与其他组的记录时长、时间戳间隙和采样率没有指向系统采集失败。

DD stable-error相对stable-correct具有更高的MAD-based outlier fraction、更高但绝对值极小的
zero-difference fraction（0.000215 vs 0.000051）和平均maximum constant run（1.36 vs
1.12）。其中outlier fraction与unstable无差异（raw q=0.899），而这些“outlier”是相对每条
信号自身MAD定义的运动突发点，并非传感器越界、NaN或饱和。因此它们更像运动形态/动作执行
代理，而不是硬件损坏证据；微小的zero-difference/constant-run差异也不足以构成冻结或丢包。

### 高风险activity的原始时域/频域模式

最强模式出现在CrossArms。DD stable-error双腕Gyro的0.5–3 Hz相对功率为0.734，3–7 Hz为
0.125，主频0.856 Hz，3–7 Hz peak ratio为1.45，左右腕幅值相关为0.855；DD
stable-correct对应0.364/0.401/2.830 Hz/8.66/0.371。DD stable-error几乎重现PD
stable-correct的0.727/0.125/0.889 Hz/1.54/0.873。上述方向均在15/15 split保持一致，
主要比较的|Cliff's delta|约0.69–0.85，BH q最低至1.35e-8。

DrinkGlas出现相同低频化：DD stable-error的0.5–3/3–7 Hz功率、主频和peak ratio为
0.645/0.186/0.994 Hz/2.29，DD stable-correct为0.424/0.399/2.597 Hz/7.67，而PD
stable-correct为0.653/0.190/0.893 Hz/2.37。HoldWeight的Gyro motion RMS为0.0487 vs
DD stable-correct 0.2121，接近PD stable-correct 0.0422；LiftHold为0.1732 vs 0.2734，
接近PD stable-correct 0.1765。左右腕相关和幅值不对称结果同样表明这不是单腕随机异常。

在四项活动的全部原始特征空间中，DD stable-error到PD/DD stable-correct centroid的相对
PD-like margin为+6.27，DD stable-correct为-3.59；其5/10-NN中的PD比例为0.982/0.979，
DD stable-correct为0.404/0.487。仅频域时相对margin仍为+4.24 vs -3.07，5/10-NN PD比例
为0.965/0.962 vs 0.430/0.496。反方向上，PD stable-error的全特征margin为-3.49、5-NN
PD比例0.35，而PD stable-correct为+7.85和0.99。该近似对称的类别反转比“DD特有坏数据”
更支持动作/运动表型与诊断标签空间重叠。

但DD unstable本身已经较PD-like（全特征margin +4.11、5/10-NN PD比例0.909/0.897）；
DD stable-error相对unstable的大部分原始特征空间差异较弱。因此这是从stable-correct到
unstable再到stable-error的连续谱，而不是只存在于34名stable-error中的孤立异常。

### 冻结representation geometry与activity一致性

在train-only StandardScaler及疾病centroid下，DD stable-error到PD centroid距离为13.24，
stable-correct为21.36；PD-like centroid margin为+4.175 vs -3.134；5-NN中的PD比例为
0.863 vs 0.382，DD比例为0.137 vs 0.618（以上q=3.03e-13）。其到DD centroid的距离与
stable-correct没有显著差异（q=0.192），说明它们不是简单远离DD，而是同时明显靠近PD区域。
相对unstable，DD stable-error仍更靠近PD且PD邻域比例更高。

DD stable-error的平均PD-like activity score为+0.694，stable-correct为-0.534；PD-like
activity比例为0.693 vs 0.414，activity间SD反而更低（1.33 vs 1.82）。CrossArms差异最大
（+2.266 vs -2.078），随后为DrinkGlas（+0.932 vs -1.153）、HoldWeight（+0.427 vs
-1.374）、LiftHold（+0.397 vs -0.448）；Relaxed和RelaxedTask无显著差异。该结果确认
DD stable-error是多activity方向一致的PD-like证据，CrossArms最强，但并非11项活动全部一致。

### 四类解释的当前判断

1. **数据质量或动作执行问题：** 没有硬性采集损坏证据。动作幅值、突发性、双腕同步和
   频谱差异明确存在，但缺少视频/执行质量标注，无法区分协议执行方式与真实运动表型。
2. **特定DD/临床表型：** subtype未确认；较短记录病程是有限关联，但不能区分stable-error
   与unstable，且缺少严重度、用药和ON/OFF metadata。
3. **稳定时域/频域模式：** 已确认。尤其CrossArms/DrinkGlas的低频化、3–7 Hz功率下降和
   双腕同步增强跨split稳定，并在PD stable-error中反向出现。
4. **PD/DD标签空间重叠：** 当前最强解释。原始time/frequency空间、冻结representation、
   centroid/kNN和多activity证据均显示错误受试者具有对侧类别的运动表型，unstable位于中间。
   这仍不能区分真实临床重叠、诊断边界、共病、病程/用药或动作执行差异。

### 决策与证据边界

本阶段不改变backbone、fusion或training recipe，也不直接启动训练。预设的频域探索门槛
（DD stable-error vs stable-correct：BH q<0.05、|Cliff's delta|>=0.33、至少12/15 split
同方向）有96个冗余的activity×sensor×wrist频域特征满足，因此一个针对低频/3–7 Hz结构的
**learnable frequency branch**可以列为后续单因素预登记候选。它不能被视为已证明会提高BA：
原始V7通用rFFT支路已经失败，当前模式同时存在于时域并延伸到unstable，且可能反映真实标签
重叠。若临床/执行审阅不能进一步解释该模式，再单独制定与V7明确不同的频域假设和保留门槛；
在此之前不实现或训练该分支。

完整报告、全部显著和不显著检验、subject/split表及可复现脚本见
`artifacts/representation_diagnostics/v8_gn_stable_error_phenotype_data_audit_20260921_r4/`
与`foundation_validation/scripts/run_stable_error_phenotype_audit.py`。

## Strong Pure-Deep Baseline Benchmark — Phase A（2026-09-21）

### 预注册问题与固定协议

本阶段把V8-GN重新定义为reference backbone，并在不改变PADS PD-vs-DD任务、双腕输入、11项
activity、subject-level aggregation和评估协议的前提下，与规范的小型ResNet1D及
InceptionTime/multi-scale temporal CNN比较。三者均使用相同的subject-level fixed
inner-development splits（5个development context × 3 inner folds）、train-only
normalization、seed 42/43/44、Full bilateral fusion、balanced CE、AdamW 2e-4、batch 8、
weight decay 1e-4、cosine schedule、50 epochs及BA early stopping；没有为候选单独搜索
超参数，也没有创建、读取或预测任何outer-final/test loader或信号。

参数量分别为V8-GN 71,026、ResNet1D 279,137、InceptionTime 484,449。新增模型契约和
fusion测试18/18通过；每个候选的三个seed均完整产生15个development fold结果。

### 三seed结果与15-split配对结果

| Backbone | Params | Accuracy | BA | Macro-F1 | AUROC | PD Recall | DD Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| V8-GN reference | 71,026 | 0.7154 | 0.6707 | 0.6631 | 0.6800 | 0.7782 | 0.5631 |
| ResNet1D | 279,137 | 0.7093 | 0.6470 | 0.6380 | 0.6540 | 0.7968 | 0.4972 |
| InceptionTime | 484,449 | 0.7078 | 0.6692 | 0.6579 | 0.6819 | 0.7625 | 0.5760 |

ResNet1D的seed BA为0.6426/0.6407/0.6578（sample SD 0.0094），AUROC为
0.6577/0.6485/0.6558（SD 0.0049）。相对matched V8-GN，15个独立split三seed聚合后的
平均差为BA -0.0237、AUROC -0.0260、Macro-F1 -0.0251、DD recall -0.0659；对应改善
split仅4/15、3/15、4/15和4/15。该候选明确拒绝。

InceptionTime的seed BA为0.6688/0.6788/0.6601（SD 0.0093），AUROC为
0.6770/0.6913/0.6773（SD 0.0082）。相对matched V8-GN，平均差为BA -0.0014、AUROC
+0.0019、Macro-F1 -0.0052、DD recall +0.0129；对应改善split为8/15、9/15、6/15和
9/15。其PD recall跨seed波动较大（0.7208/0.7581/0.8087，SD 0.0441），DD recall也为
0.6168/0.5995/0.5116（SD 0.0564）。该候选仅与V8-GN持平，且参数量约为6.8倍，不满足
BA约+0.01、AUROC约+0.015、Macro-F1不下降、多数split稳定改善及seed稳定性不恶化的联合
替换条件，因此拒绝替换。

### Phase A决定

V8-GN继续作为当前最强且最稳定的pure-deep backbone，并作为Phase B唯一基座。现有证据不
支持V8-GN存在可由标准ResNet1D或InceptionTime修复的明显结构能力不足；更准确的边界是：
在固定recipe和有限候选下，增加标准CNN容量没有产生实质性跨受试者泛化收益，不能据此证明
任务已达到性能上限。ResNet1D和InceptionTime均保留为负结果，不继续做backbone专属搜索。

Phase A统计产物位于
`artifacts/strong_backbone_benchmark_20260921/phase_a/`。下一步严格只在V8-GN上执行一个
预注册的Gyro-only lightweight learnable FFT branch，不比较FFT/STFT/wavelet等多个方案；
若未满足既定联合门槛，则停止frequency branch路线。

## Frequency-aware 单因素实验 — Phase B（2026-09-21）

### 假设与唯一改动

基于SEPA-01在CrossArms、DrinkGlas等activity中确认的Gyro低频及3–7 Hz稳定结构，只在
Phase A保留的V8-GN上增加一个Gyro-only learnable rFFT branch。支路对Gyro XYZ执行可微
log-magnitude rFFT，截取0–12 Hz并插值到128 bins，再由轻量Conv1D编码器（hidden 24）学习
16维spectral embedding，与原64维temporal embedding简单拼接投影回64维。没有输入
bandpower、dominant frequency、tremor ratio或其他handcrafted统计量，没有测试STFT、
wavelet或其他频域网络。模型参数由71,026增至78,450（+7,424，约+10.5%）；其余数据、
Full bilateral fusion、activity aggregation、loss、optimizer、normalization、splits、seeds和
评估协议全部冻结。

### 三seed与配对结果

| Model | Params | Accuracy | BA | Macro-F1 | AUROC | PD Recall | DD Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| V8-GN | 71,026 | 0.7154 | 0.6707 | 0.6631 | 0.6800 | 0.7782 | 0.5631 |
| V8-GN + Gyro FFT | 78,450 | 0.7041 | 0.6561 | 0.6480 | 0.6637 | 0.7718 | 0.5404 |

频域模型seed BA为0.6753/0.6553/0.6377（sample SD 0.0188；V8-GN为0.0088），AUROC为
0.6827/0.6653/0.6429（SD 0.0199；V8-GN为0.0103）。15个独立split内先聚合三个seed后，
相对matched V8-GN的平均差为BA -0.0146（5/15 wins）、AUROC -0.0163（5/15）、Macro-F1
-0.0151（4/15）和DD recall -0.0227（4/15）。两侧paired Wilcoxon的未校正p值依次为
0.0413、0.1070、0.0479和0.1980；paired rank-biserial effect size为-0.600、-0.483、
-0.583和-0.390。对Phase A/B共12项预设比较做Benjamini–Hochberg校正后q值依次为
0.1329、0.1834、0.1329和0.2970，均不显著。这里“不显著”不等于等效：点估计、负向
效应量、低split胜率和明显变大的seed方差一致支持该分支不稳定且无收益。

### Phase B决定与停止规则

Gyro learnable FFT branch没有提供独立、稳定增益，反而降低全部四项关键指标并恶化seed
稳定性，明确拒绝。按预注册停止规则，不继续搜索FFT bins/range、STFT、wavelet、spectral
Transformer或其他频域结构。SEPA-01的频谱差异仍是有效的表型/诊断证据，但本实验说明它
不能自动转化为V8-GN之外的可泛化预测增益，可能已被时域分支部分利用，或主要反映当前
PD/DD标签空间重叠。

所有9个新增候选seed目录均各含15个development folds，且`outer_test_loader_created`、
`outer_test_signal_accessed`、`outer_test_predictions_accessed`全部为false。配对统计以15个
独立split为单位，seed先聚合，不把45次训练当作独立样本。完整结果与统计位于
`artifacts/strong_backbone_benchmark_20260921/`。

## Strong Backbone Benchmark 最终冻结决定

正式冻结V8-GN（71,026参数，Full bilateral fusion与原training recipe）为后续博士研究的
strong pure-deep backbone。ResNet1D明显更差；InceptionTime用约6.8倍参数仅达到统计持平；
Gyro FFT分支明显退化。因此没有证据表明V8-GN存在可由这些标准backbone或该频域分支修复的
明显结构能力不足。

同时必须保留一个不利事实：现有development-only H1 handcrafted/statistical diagnostic为
BA 0.6918、AUROC 0.7515，高于本轮最强pure-deep V8-GN的0.6707/0.6800，尤其AUROC差距
0.0715。H1结果不是本轮模型选择信息，也没有使用outer数据；但它说明当前纯深度模型尚未
超过已有handcrafted/statistical参照，不能通过继续堆叠模块来掩盖。后续创新应针对跨个体
泛化或可解释表征提出新的、预注册的假设，而不是恢复已停止的backbone/frequency搜索。

## H1–Deep Representation Gap Diagnosis（2026-09-21）

本阶段未训练或修改backbone，未增加FFT、Transformer、DANN、attention或其他模块。全部分析
使用相同固定15个subject-level inner-development splits；V8-GN seeds 42/43/44先在相同
split内聚合；H1与所有recoverability probes只在inner-train拟合并应用于unseen
inner-validation subjects；没有访问outer-final/test，也没有H1/probe超参数搜索。

### H1 feature-family decomposition

H1真实4928维schema为11 activities × 2 wrists × 8 signals × 28 statistics。按真实实现分为
time/location/scale、time shape、derivative、global spectral、absolute band power和relative
band fraction六族。Full H1精确复现BA 0.6918、AUROC 0.7515、Macro-F1 0.6858、PD/DD
recall 0.7936/0.5900。

没有任何family-only模型接近full H1：time-domain all为BA/AUROC 0.6786/0.7341，spectral
all为0.6570/0.7161，time/location/scale-only为0.6523/0.7034，band-fraction-only为
0.6505/0.6985。因此H1优势不能由单一族完整解释。

LOFO显示两个稳定主贡献。移除time/location/scale令BA/AUROC下降0.0174/0.0181，15个split
仅4/1个改善，paired rank-biserial为-0.667/-0.950，60项BH校正q=0.0404/0.0020。移除band
fraction令BA/AUROC下降0.0159/0.0130，仅4/2个split改善；BA q=0.1338，AUROC q=0.0023。
time shape仅有弱趋势；derivative与global spectral没有稳定独立贡献；移除absolute band
power反而令BA/AUROC提高0.0084/0.0047，但校正后不显著。故主要优势是time/location/scale与
relative band fractions的跨family、跨activity、双腕联合结构，而不是absolute频带能量。

### Deep recoverability

冻结前向提取了45个seed-fold的wrist、bilateral activity、activity-context和subject表示。
固定Ridge α=1使用train-only X/target scaling。Activity-context对H1 family的OOS R²为：
derivative 0.581、time/location/scale 0.394、time shape 0.219、band fraction 0.212、global
spectral 0.193、absolute band 0.130；wrist层结果相近。上述24个local representation ×
family组合全部15/15 split R²>0，BH q=6.1e-5。

最终subject embedding对六族完整结构的R²全部为负；关键time/location/scale为-0.689
（Spearman 0.232，normalized MAE 1.007），band fraction为-1.413（0.144，1.198）。将完整
family targets用inner-train PCA降到16维后，learned subject及简单mean-pooled subject
representations的R²仍全为负。因此V8不是完全没有编码H1信息，而是局部activity/wrist信息
没有以可恢复形式保留到单一subject vector；结果不能归因于某个attention权重单独失败。

“高疾病分类价值、低subject recoverability”的交集首先是time/location/scale，其次是band
fraction。相对频谱信息的局部recoverability也弱于time/location/scale。

### H1–V8 error complementarity

1560个unseen validation appearances中，both-correct 951、H1-only 194、V8-only 171、
both-wrong 244。H1纠正194/438（44.3%）个V8错误，但对stable-error appearances仅纠正
60/200（30.0%）：DD 41/130、PD 19/70。54名V8 stable-error subjects中，persistent H1
rescue仅14名，persistent both-wrong为32名，8名mixed；只有5名四个H1 contexts全对，
27名全错。因此H1没有真正解决大多数V8稳定错误。

H1 rescue subjects更靠近V8疾病边界。DD rescue与both-wrong的PD-like centroid margin为
+3.00 vs +4.66，PD为-1.31 vs -2.45；DD rescue在CrossArms和DrinkGlas的PD-like score
也较低。H1 correction由多activity联合贡献而非CrossArms单项驱动。对persistent rescue与
both-wrong的高风险raw signals做528项比较后无一通过BH q<0.10，不能强行归因于单一原始
时域/频域表型。

### 最终决定与证据边界

H1的约0.0715 AUROC gap不是“V8完全没看到频率”的证据。V8局部表示已编码部分H1统计，
真正明确的gap是：activity-conditioned、wrist-structured的幅值/离散度和relative spectral
allocation没有在subject representation中保持可访问。下一版pure-deep研究应首先以这一
具体representation gap为目标；本阶段不设计或训练修改，也不重启通用FFT/backbone搜索。

完整报告与全部正/负结果位于
`artifacts/h1_deep_gap_diagnosis_20260921/H1_DEEP_REPRESENTATION_GAP_DIAGNOSIS_REPORT.md`。

## STR-01 Structured Token Residual Readout（2026-09-21）

### 预注册改动与协议

本轮只在V8-GN的subject aggregation之前增加一条structure-preserving residual decision path：
对11个258D bilateral activity representations使用共享`Linear(258,16)+GELU`，保留固定
activity身份与顺序后展平为176D，经单层`Linear(176,2)`输出residual logits并与原V8 logits
直接相加。Residual head零初始化，因此初始decision function与原V8主路径严格一致。没有搜索
projection dimension、MLP depth、attention、gating或其他结构。参数由71,026增至75,524，
增加4,498（6.33%）。新增及既有模型测试17/17通过。

V8-GN wrist encoder、Full bilateral fusion、activity representations、原Activity Attention、
subject embedding与classifier主路径全部保留；preprocessing、train-only normalization、balanced
CE、AdamW 2e-4、batch 8、WD 1e-4、cosine scheduler、固定5×3 inner-development splits和
seeds 42/43/44均未改变。没有创建、读取或预测任何outer-final/test loader、信号或metadata。

### 分类结果与paired inference

| Model | Params | Accuracy | BA | Macro-F1 | AUROC | PD Recall | DD Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| V8-GN | 71,026 | 0.7154 | 0.6707 | 0.6631 | 0.6800 | 0.7782 | 0.5631 |
| STR-01 | 75,524 | 0.7307 | 0.6972 | 0.6861 | 0.7176 | 0.7779 | 0.6165 |
| Delta | +4,498 | +0.0154 | +0.0265 | +0.0230 | +0.0377 | -0.0004 | +0.0534 |

STR-01的seed BA为0.7084/0.6981/0.6850（sample SD 0.0117；V8为0.0088），AUROC为
0.7276/0.7209/0.7044（SD 0.0120；V8为0.0103）。主指标方差略增，但每个seed的BA与
AUROC都高于matched V8，Macro-F1 SD由0.0137降至0.0107。DD recall SD由0.0134增至
0.0408，是必须保留的稳定性限制；不过三个seed的DD recall均未低于matched V8，平均PD
recall基本不变，未出现以一类recall换另一类的总体牺牲。

三个seed先在相同split内聚合。相对matched V8，BA平均+0.0265、14/15 splits改善，paired
Wilcoxon p=0.000183、rank-biserial=+0.967、6项BH q=0.000549；AUROC +0.0377、15/15，
p=0.000061、effect=+1.000、q=0.000366；Macro-F1 +0.0230、12/15，p=0.000854、
effect=+0.900、q=0.001709。Accuracy为+0.0154、9/15、q=0.1087；PD recall为-0.0004、
7/15、q=0.8871；DD recall为+0.0534、12/15，但q=0.0619。后三者是校正后不显著或明确
null的结果，不以主指标显著性代替报告。

### H1 recoverability机制证据

固定`Ridge(alpha=1.0)`与上一阶段相同，只使用inner-train拟合及标准化。原V8 subject
embedding对`time_location_scale`完整结构的OOS R²/Spearman/nMAE为-0.689/0.232/1.007，
STR structured residual representation为-0.210/0.366/0.837；对`band_fraction`分别由
-1.413/0.144/1.198改善为-0.736/0.214/1.029。两个family的三项指标均15/15 splits改善；
六项比较的p均为0.000061、rank-biserial均为+1.0，18项BH q均为0.000092。

STR原主路径subject embedding本身没有改善：time/location/scale R²为-0.680，band fraction
为-1.481。说明额外可恢复信息确实位于structured residual path，而不是原subject vector自动
变好。与此同时，完整target的绝对R²仍为负，尤其band fraction仍难以准确恢复；不能表述为
STR已经重建完整H1结构。

Classification与recoverability在总体方向上同步改善，但15个split间的改善幅度没有稳定对应：
24个recoverability-delta与BA/AUROC-delta的探索性Spearman相关均未通过BH（全部q=0.980）。
因此结果支持“过早压缩是V8瓶颈”的机制解释，但不足以证明分类提升由H1结构恢复直接导致。

### Retain决定与新冻结状态

STR-01满足预注册的BA、AUROC、Macro-F1、双类recall和split一致性要求，参数增量受控，故
**保留STR-01并将其冻结为下一阶段strong pure-deep backbone**；V8-GN降为matched reference。
不据此搜索STR维度/深度、attention、handcrafted输入、reconstruction loss、FFT/STFT/wavelet、
Transformer、DANN/MMD/prototype、新fusion或activity aggregation。

本结果只证明固定development协议内的收益，不证明outer泛化，也没有解决全部phenotype
overlap。H1 development AUROC仍为0.7515，高于STR-01的0.7176，剩余gap为0.0339；STR-01
BA 0.6972则略高于H1的0.6918。完整报告、matched 15-split统计、45个representation archives、
全部positive/negative mechanism结果位于
`artifacts/str01_structured_token_residual_20260921/STR01_REPORT.md`。

## WSR-01 Wrist-Structured Residual Readout（2026-09-22）

### 预注册改动与验证

本轮以STR-01为唯一matched baseline，只从Full bilateral fusion之前的left/right 64D wrist
embeddings增加一条轻量残差决策路径。每个activity的左右腕使用同一个
`Linear(64,4)+GELU` projection，保留固定`11 activities × 2 wrists × 4 dimensions`结构，
展平为88D后由单层`Linear(88,2)`输出residual logits并加到完整STR-01 logits。Residual
classifier零初始化，初始decision function与STR-01完全一致。Full fusion、Activity Attention、
V8 classifier path及STR-01的11×16 ordered activity path均完整保留，没有搜索维度、深度、
attention、gating或融合形式。

STR-01为75,524参数；shared projection增加260参数，residual classifier增加178参数，WSR-01
共75,962参数，净增438（0.58%）。模型与单元测试19/19通过；零初始化logit等价测试通过。
固定preprocessing、train-only normalization、balanced CE、AdamW 2e-4、batch 8、WD 1e-4、
cosine scheduler、5×3 inner-development splits和seeds 42/43/44均未改变。没有访问任何
outer-final/test信息。

### 分类结果与paired inference

| Model | Params | Accuracy | BA | Macro-F1 | AUROC | PD Recall | DD Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| STR-01 | 75,524 | 0.7307 | 0.6972 | 0.6861 | 0.7176 | 0.7779 | 0.6165 |
| WSR-01 | 75,962 | 0.7234 | 0.6965 | 0.6824 | 0.7216 | 0.7615 | 0.6315 |
| Delta | +438 | -0.0073 | -0.0007 | -0.0036 | +0.0039 | -0.0163 | +0.0150 |

三个seed先在相同split聚合。BA为7/15 splits改善，Wilcoxon p=0.7764、rank-biserial=-0.083、
6项BH q=0.7764；AUROC为11/15改善，但均值仅+0.0039，p=0.0353、effect=+0.617，BH
q=0.2120；Macro-F1为6/15改善、q=0.4673。Accuracy下降0.0073；PD recall下降0.0163而
DD recall提高0.0150，表现为类别trade-off，不是双类共同改善。所有六项classification
comparison均未通过BH校正。

WSR-01三个seed的BA为0.7100/0.6918/0.6877，AUROC为0.7353/0.7204/0.7090；两项相对
matched STR-01均为`+/−/+`，未满足三seed方向一致。BA seed SD为0.0119（STR 0.0117），
AUROC SD为0.0132（STR 0.0120），稳定性未改善。

### Wrist-structured recoverability机制结果

固定`Ridge(alpha=1.0)`仅在inner-train拟合，并在unseen inner-validation subjects评估完整
activity/wrist-specific targets。新增88D wrist representation使`time_location_scale`的
R²/Spearman/nMAE由STR activity-structured representation的-0.210/0.366/0.837改善为
-0.065/0.407/0.739；`band_fraction`由-0.736/0.214/1.029改善为
-0.334/0.226/0.887。两族R²和nMAE均15/15 splits改善，Spearman分别13/15和12/15；六项
比较全部通过18项BH校正（最大q=0.00641）。

公平比较完整decision input时，加入wrist structure后`time_location_scale`由
-0.280/0.354/0.866改善为-0.179/0.383/0.812，`band_fraction`由
-1.051/0.197/1.124改善为-0.807/0.223/1.053。五项为15/15改善，剩余一项为14/15；六项
BH q≤0.000499。匹配训练下的STR activity path基本不变，支持新增可恢复信息来自wrist path。

但是recoverability改善没有转化为分类改善：24项split-level recoverability delta与BA/AUROC
delta的探索性相关均未通过BH（最小q=0.7388）。因此不能将微小AUROC变化归因于H1-like
wrist信息保留，也不能以机制probe代替分类标准。

### Reject决定与停止边界

WSR-01未达到BA +0.01、AUROC +0.015、Macro-F1不下降、无单侧recall牺牲、主要指标约
10/15 split改善、三seed方向一致及稳定性不恶化等预注册要求，故**正式拒绝WSR-01**。
结论为：显式保留pre-fusion wrist structure能显著提高H1关键统计结构的线性可恢复性，但
preserving wrist-structured handcrafted-statistical information alone is insufficient to improve
subject-level classification。

按停止规则，不继续搜索wrist projection、gating、attention或其他structured-readout扩展，
**STR-01继续冻结为当前strong pure-deep backbone**，V8-GN继续仅作matched reference。
完整报告、45个representation archives、15-split paired statistics、effect sizes、BH校正和全部
positive/negative mechanism结果位于
`artifacts/wsr01_wrist_structured_residual_20260922/WSR01_REPORT.md`。

## STR-01 Gain Mechanism Diagnosis（2026-09-23）

本阶段没有训练或修改模型，只对冻结V8-GN与STR-01进行matched、development-only诊断。使用
完全相同的固定5×3 inner splits和seeds 42/43/44；decision-pattern分析先在相同split聚合三个
seed，paired inference以15个split为独立单位，persistent phenotype以独立subject为单位。
没有访问outer-final/test。45个paired archives的activity contribution可精确重建base、
residual及final DD-minus-PD logits，最大数值误差`1.91e-6`。

### 哪些decision被修复

1560个seed-aggregated validation appearances中，both-correct 1029、V8 wrong→STR correct 139、
V8 correct→STR wrong 93、both-wrong 299，净修复46次。Rescue按标签为DD 65、PD 74，harm为
DD 30、PD 63，因此净收益主要来自DD（+35 vs PD +11），与正式指标中DD recall由0.5631提高
至0.6165、PD recall基本不变一致。

Rescue主要是原V8 near-boundary errors：DD/PD的V8 true-direction margin为-0.606/-0.355，
而both-wrong为-1.075/-0.707；STR把rescue最终margin提高到+0.477/+0.634，但both-wrong反而
保持在-1.322/-1.094。只有6名subject在四个validation appearances中至少3次被稳定rescue
（DD 5、PD 1），2名被稳定harm；增益主要分布于split-dependent的边界受试者，而不是一个
大的永久修复亚群。

### Original path与residual path的互补

| Path | BA | AUROC | Macro-F1 | PD Recall | DD Recall |
|---|---:|---:|---:|---:|---:|
| V8-GN | 0.6707 | 0.6800 | 0.6631 | 0.7782 | 0.5631 |
| STR original/base only | 0.6299 | 0.6793 | 0.6181 | 0.7320 | 0.5279 |
| STR residual only | 0.6561 | 0.7209 | 0.6298 | 0.6924 | 0.6199 |
| STR combined final | 0.6972 | 0.7176 | 0.6861 | 0.7779 | 0.6165 |

STR原subject path单独并未超过V8；residual path具有很强排序信息但threshold BA与双类recall不平衡，
二者组合才产生最终收益。这说明训练把疾病证据重新分配到两个path，而不是在原V8 logits上简单
追加一个独立修正器。

在139次rescue中，59次（42.4%）是STR base仍错、由residual直接翻转；80次（57.6%）在联合
训练后的base path已经正确，其中residual增强64次、减弱16次。93次harm中，42次由residual
直接把正确base翻错，其余51次base已经错误。Base与residual margins的split平均Spearman为
0.613，但预测类别有25.3%不一致，故为稳定但不完全正交的互补证据。Net rescue rate与BA gain
有原始相关`rho=0.646, p=0.0092`，但12项BH后`q=0.0840`；所有与AUROC gain的关系均未通过
校正，不能将split间增益幅度强行归因于单一互补统计量。

### Activity contribution

Rescue residual true-margin较大的activity为Relaxed +0.114、RelaxedTask +0.105、TouchNose
+0.104、HoldWeight +0.090、DrinkGlas +0.079；DrinkGlas、TouchNose、HoldWeight最常成为最大
贡献者（26.5%、21.0%、17.0%），且移除后成为pivotal的比例为23.6%、15.9%、21.8%。但是
11个activity的rescue-vs-harm差异均未通过22项BH（最小q=0.1296）。CrossArms、DrinkGlas、
HoldWeight分别仅10/15、10/15、12/15方向一致，q=0.175/0.315/0.130，不能声称高风险activity
形成稳定单独机制。

单activity disease AUROC也没有普遍提高；PointFinger下降0.1188（15/15，q=0.00134），
Entrainment下降0.0735（14/15，q=0.00201），没有任何activity取得校正后显著提升。因此STR
的正机制是**多个异质activity的ordered joint decision pattern**，而不是某一activity证据普遍
变强。这支持保留STR，不支持继续扩展per-activity/structured readout架构。

### Stable-error是否真正减少

按本阶段主要的seed-first aggregation，在四个validation appearances上使用冻结audit阈值
（error rate≤0.25为stable-correct，≥0.75为stable-error）：V8 stable-error为85，STR为74。
原85名中12名变stable-correct、10名变unstable、63名仍stable-error；同时出现11名新的
stable-error，净减少11（12.9%）。

为与既有audit连续，复现其12个individual seed-fold appearances定义：V8 stable-error精确为
54，STR为42；原54名中仅2名变stable-correct、18名变unstable、34名仍stable-error，另有8名
V8 unstable变为STR stable-error，且没有V8 stable-correct变stable-error。两种单位均说明STR
确实减少stable-error burden，但多数原stable-error仍持续错误或仅被软化为unstable，并未解决
核心phenotype-overlap群体。

### Matched representation结果

固定train-only standardized disease probe显示，STR combined representation相对V8 subject
embedding的BA提高0.0374（12/15，BH q=0.0336），AUROC提高0.0465（14/15，q=0.0122）；
STR subject path单独没有提升。另一方面，combined Euclidean silhouette下降0.0107（14/15，
q=0.0224），domain AUC与normalized mean shift没有改善，PD centroid drift增加0.702（12/15，
q=0.0153），DD drift也呈恶化趋势。

Cross-activity subject retrieval同样没有改善：combined tokens的top-1、top-5、MRR均15/15下降，
BH q=0.00214/0.00031/0.00031。Activity identity在V8/STR context均已达到1.0。由此可见STR
提高的是decision-accessible disease evidence，而不是全局class clustering、subject consistency或
split/domain invariance。

### Raw phenotype、geometry与H1边界

Persistent rescue只有6名，subject-level raw phenotype统计功效有限。264项高风险raw-signal比较
无一通过BH（最小q=0.388），没有可辩护的稳定Acc/Gyro rescue phenotype。5名persistent DD
rescue中4名为Other Movement Disorders、1名Essential Tremor；单名PD不支持亚群推断。

H1在appearance-level rescue中的correct rate为DD 60.0%、PD 73.0%，高于both-wrong的
29.8%/38.4%，说明STR rescue通常不是最深的共同overlap。以独立persistent subjects比较时，
5名DD rescue仅`band_fraction`稳定高于both-wrong（q=0.0035），LiftHold为探索性趋势
（q=0.0768）；PD仅1名，不能推断。Raw信号没有同步稳定差异，不能建立H1或频域因果解释。

### 最终方向决定

STR-01 gain来自distributed cross-activity evidence与base/residual decision complementarity，但该
机制已经由当前STR充分验证；继续cross-activity readout或decision regularization的收益证据不足。
更关键的负结果是silhouette、subject retrieval、domain/shift与centroid drift未改善，以及多数
stable-error仍存在。因此下一阶段应优先进入**跨个体泛化表示学习**，目标是subject-invariant、
split-stable disease representation，而不是STR-02、gating、attention或新的decision-stage模块。

完整报告与全部positive/negative结果位于
`artifacts/str01_gain_mechanism_diagnosis_20260923/STR01_GAIN_MECHANISM_DIAGNOSIS_REPORT.md`。

## DSG-01：Cross-subject Disease Subspace Stability Diagnosis（2026-09-26）

本阶段只读使用固定 5×3 inner-development splits、seeds 42/43/44 及现有 V8-GN/STR-01 checkpoint；不训练模型，不修改正式 forward、recipe、split、checkpoint 或历史结果。90 个 checkpoint 均 strict load；normalization 的拟合 subject IDs 与各自 inner-train 完全一致，train/validation 不重叠。split SHA-256 仍为 `b3c52317cb12b73c66046bbd7c50c94a7e707a64a38d3ad4f0fd4bc0c6ba6b3e`。11 个 activity 和 left/right 腕顺序按冻结代码核对。只读 hook 前后两模型同输入 logits 最大差为 0；180 个新 train/validation archive 与旧正式归档按 ID/label 对齐，final logits 最大差为 0。没有建立 outer loader 或读取 outer 信号、标签、预测、指标；inner-stage split 文件中的 test-subject 字段未用于本轮分析。

代码核查明确了一个命名差异：`forward()` 返回的 `activity_embeddings` 是加 activity-ID 并经 LayerNorm 后的 context 258D；STR residual 输入是此前的 bilateral activity 258D。DSG-01 同时提取两者，以及 STR original subject 258D、`11×16D` projected tokens、176D ordered representation、base/structured/final 2D logits；V8 提取对应 activity、subject pre-logit 与 logits。所有向量都有 subject ID、label、split、seed、train/validation role、activity ID 与 representation type 元数据。

预设方法：在共同 train subjects 上计算 Linear CKA 并做 unbiased-HSIC CKA 核查；各模型实例仅用 inner-train 拟合 disease centroid/probe，跨实例先在无标签共同 train subjects 上使用固定 PCA-16 + Orthogonal Procrustes 对齐，再比较 disease/probe directions 与 principal angles。原始 latent cosine 只作描述。A→B probe transfer 严格排除所有 A.train 与 B.validation 重叠者；至少 20 人且每类至少 5 人才评分。STR decision stage 的 630 个同 seed 跨 split 有向比较中 540 个有效，90 个因合格人数为 0 标记 NA。三个 seed 先在 split 内聚合，以 15 split 为主统计单位，activity 和相关分析做 BH-FDR。

**正结果：**STR combined decision 的 train-only disease probe BA/AUROC 为 `0.6582/0.7057`，V8 subject 为 `0.6164/0.6590`；配对增量 `+0.0418/+0.0468`，BH `q=0.0298/0.0028`。STR combined 相对 V8 subject 的 unbiased CKA 为 `0.7013` vs `0.5102`，对齐后 centroid/probe direction cosine 为 `0.9334/0.7215` vs `0.9166/0.5245`；方向一致性在该固定对齐量度下改善。原 subject path 的 CKA 却只有 `0.5456`；从 activity context 到单一 subject embedding 的稳定性降幅仍是最明显 stage。STR 的 ordered path 提高 decision-accessible disease information，但不能据 CKA 单独宣称 disease-specific invariance。

**负结果与限制：**STR combined 的 cross-split native→transfer BA drop 为 `0.0294`（14/15 split 为正），V8 subject 为 `0.0307`，两者配对差 `−0.0013`、BH `q=0.9642`；STR 未消除 probe transfer loss。STR combined 的同 split 跨 seed/跨 split centroid cosine 为 `0.9647/0.9334`，有可检测的 split 组成相关方向变化，但 probe direction 没有相应跨 split 特异性；同 split 跨 seed CKA 还低于跨 split 同 seed，说明训练随机性/量度差异也明显。不同对照的共同 train 人数不同，不能将其视为纯因果效应。

11 个 activity 的 raw/context centroid direction cosine，STR 平均比 V8 低约 `0.024`（8 项 aggregate-family BH `q=0.00073`），但单 activity 有升有降；平均 probe direction、BA、AUROC 无系统改善，任何单 activity 的 probe BA/AUROC 改善均未通过 BH。STR combined 的稳定性指标与正式 BA、AUROC、Macro-F1、PD/DD recall 的 15-split 相关均未通过 BH；例如 unbiased CKA 与 DD recall `ρ=0.373`、bootstrap 95% CI `[−0.129,0.753]`、`q=0.397`，transfer BA drop 与 DD recall `ρ=0.164`、CI `[−0.431,0.675]`、`q=0.729`。不能把非显著解释为等效，15 个 split 的 CI 也不代表外部队列。

Stable-error primary 固定为**先聚合三个 seed，再按四次 validation appearance 判定**：V8 stable-error 85、STR 74。历史 12 seed-fold 定义的 54/42 仅作 sensitivity，不与 primary 混用。STR stable-error 在四次 appearance 上的 disease-axis/probe true margin 通常持续指向错误类，probe-margin SD 还低于 unstable（2.143 vs 2.918，BH `q=0.0032`）；这更符合持续对侧类别表示或 phenotype/label overlap，而不是仅由剧烈 split 波动造成。final logit 组差异部分由 stable-error 定义直接导致，不能作为独立机制证据；缺少临床 metadata 与外部队列，真实重叠原因仍未证实。

**最终决定：不启动 activity-conditioned cross-subject disease consistency experiment。** 虽可检测跨实例变化，尚未同时证明它是明确的 disease-specific instability、并且与分类退化有稳定关系。停止基于 DSG-01 的 consistency regularization、DANN/MMD/prototype、supervised contrastive 和新模块训练；保留 STR-01 为冻结主模型。完整报告、正负结果、NA、pairwise matrices、效应量、BH、bootstrap CI、smoke 与 logits consistency 位于 `artifacts/dsg01_cross_subject_disease_subspace_20260926/DSG01_REPORT.md` 及其 `analysis/`、`extraction/` 子目录。

## RGD-01：H1–STR Residual Ranking Gap & Complementarity Diagnosis（2026-09-26）

本阶段只使用 fixed inner-development 15 splits、冻结的 STR-01/V8-GN 表示及正式 development predictions、H1 train-only 特征；没有访问 outer information，也没有训练新 deep model、修改正式模型或覆盖既有结果。资产审查发现 H1 是**每 split 一份确定性正式预测**，而 STR/V8 是 seeds 42/43/44 三份；H1 不能视作 45 个独立结果。15 split 的 H1/STR/V8 validation subject ID、label 完全对齐。按正式 H1 pipeline 在 inner-train 重拟合后，validation probability 与正式 H1 的最大差为 `1.11×10⁻¹⁶`。Stable-error primary 沿用 DSG-01 的 seed-first、四次 validation appearance 定义：STR stable-error 74、stable-correct 284、unstable 32；历史 12 seed-fold 定义不混入主分析。

**Ranking gap：**每 split 先平均三个 STR seeds，再以 15 split 为主要单位，H1/STR AUROC 为 `0.7515/0.7176`，差 `+0.0339`（13/15 split；受试者分层配对 bootstrap 95% CI `[+0.0098,+0.0567]`）。每 split 平均 2,237.7 个 PD–DD pairs 中，H1 对/STR 错 326.6（14.59%），STR 对/H1 错 249.7（11.17%），净 +77.0。100,695 个 seed-pair 实例不作独立样本。涉及 STR stable-error subjects 的 pairs 有 H1/STR rescue `10,460/6,827`，净 +3,633；其余 pairs 净 −170。移除 74 名 stable-error 后 H1−STR AUROC gap 由 +0.0339 到 −0.0031；移除 STR pair burden top 5%/10% 后到 +0.0198/+0.0134。这些 validation outcome 定义的排除仅为敏感性，不能宣称新的正式性能或独立临床 phenotype 证明。**H1 的净排序优势主要集中在 persistent hard subjects 参与的 pairs，非广泛均匀优势。**

**互补性正结果：**同一 validation subjects 的 STR–H1 score Spearman/Kendall 为 `0.541/0.386`，全部 subject-pair 排序分歧 `25.83%`；双向 rescue 均存在。只在 inner-train 拟合的简单双分数 logistic 在 validation AUROC 为 `0.7621`，相对 STR-only `+0.0445`（14/15，BH q=0.0017），相对 H1-only `+0.0103`（12/15，q=0.0067）；BA 相对两者分别 `+0.0433/+0.0146`（BH q=0.0090/0.0353）。这只是机制诊断，不是正式新模型。基础 train scores 是 in-sample，validation scores 为 held-out，fusion 的外推仍有限制。

**Frozen representation：**控制 STR final score 后，用 inner-train 固定 PCA-16 + Ridge 预测 H1−STR 标准化 logit residual。Activity raw 的 validation R²/MAE/Spearman 从 score-only `0.083/0.560/0.419` 到 `0.171/0.525/0.496`，三项 BH q≤0.0058；combined decision input 到 `0.153/0.528/0.456`，R²/MAE 的 q=0.0154/0.0028，Spearman 增益未通过 BH。未降维的高维 Ridge 在 decision input 上反而恶化，提示结果依赖维数控制。可以说一部分 H1 residual score 信息已存在于冻结 STR 表示；**尚不能说具体 H1-rescued PD–DD pairs 可由现有 decision representation 稳定救回，亦不能断言 readout 是唯一瓶颈**。

按 H1 feature schema 对 rescue pairs 的条件性归因显示左右腕、六个 feature families 和多数 activity 同向贡献；由于按 H1 是否排对分组，这些差异受到 outcome conditioning，**不能当作单个 activity、腕或频率特征的独立机制证据**，不重开 activity/frequency 搜索。综合判断为 persistent hard-subject burden 主导、两种分数确有互补、部分 residual 信息已进入 STR 表示；临床 phenotype overlap 与真正的 decision utilization 因果机制仍证据不足。**本轮不启动新的网络结构或 consistency/invariance training**。若继续研究，下一步只能先在 development 内做预注册的 frozen decision-representation 对 H1-rescued pairs 的可恢复性诊断，再决定是否设计新模型实验。完整正负结果、统计限制、bootstrap CI、BH、smoke、H1 精确复现和全部明细见 `artifacts/rgd01_h1_str_residual_ranking_20260926/RGD01_REPORT.md`。

## PRR-01：Frozen STR Pair-Rescue Recoverability Diagnosis（2026-09-26）

在 RGD-01 后，本阶段仅使用固定 15 个 inner-development splits、STR seeds 42/43/44、每 split 唯一的正式 H1 prediction、DSG-01 冻结表示和 RGD-01 train-only H1 artifact；没有访问 outer information，也没有训练新 deep model、修改 STR/V8/H1 正式模型、split、normalization 或既有结果。Stable-error primary 保持 DSG-01/RGD-01 的 seed-first、四次 validation appearance 定义（STR stable-error 74、stable-correct 284、unstable 32）。资产审查与单实例 smoke 通过；使用 H1 **正式原始概率**定义排名后，100,695 个原始 PD–DD pair categories 与 RGD-01 逐条完全一致。全量 verify 通过，pair net gain/pair 数与 AUROC Δ 最大误差 `3.23×10⁻¹⁶`。

所有 residual predictor 只在 inner-train 拟合，沿用 RGD-01 的正类 DD、train-only H1/STR logit 标准化、固定 PCA-16 和 Ridge α=1。比较 A 原 STR、B score-only residual、C activity raw、D original subject、E structured ordered、F combined decision；validation 用 STR score 加预测 residual，枚举 pair recovery/harm，三个 seed 先在 split 内平均，以 15 split 统计。B 是单调分数变换，AUROC/配对排序与 A 一致。

**有限正结果：**C activity raw 的 corrected AUROC 为 `0.7299`，比 STR/B `0.7176` 高 `+0.0123`（11/15 split；paired bootstrap 95% CI `[+0.0056,+0.0188]`；12 项主要比较 BH q=0.0107）。每 split 约 326.6 个 H1-rescue pairs 中 gross 恢复 74.3（22.87%），同时纠正一部分 H1/STR 都排错的 pairs；全部原错恢复 100.5，原本 STR 排对被新破坏 73.0，整体 net `+27.5` pairs/split（q=0.0107）。训练集 PD/DD 类内 residual target 置换的 10 次对照中，C 的真实 AUROC Δ/总净 pairs/gross rescue 均超过置换均值：`+0.0123` vs `+0.0005`、`+27.5` vs `+1.1`、`22.87%` vs `8.79%`，对应 BH q=`0.0161/0.0167/0.0002`。因此**前聚合 frozen activity representation 确有当前 STR final score 未用尽的一部分宽泛 ranking information**。

**关键负结果：**C 的 74.3 个目标 H1-rescue 恢复几乎被 73.0 个原对 pairs 的损害抵消；targeted net 仅 `+1.3` pairs/split，CI `[−12.5,+14.7]`、BH q=0.720。涉及 STR primary stable-error subjects 的目标配对，C 恢复率仅 `18.85%`、harm `10.26%`、targeted net `−8.0`；不涉及者分别 `33.46%/2.01%/+9.29`。F decision 虽 gross 恢复 `25.73%`，新破坏约 106 个原对 pairs，targeted net `−21.2`；AUROC Δ 仅 `+0.0064`（9/15；CI 跨 0；主要 BH q=0.203）。D 的净 AUROC 未过 BH，E 为边界未过主要 BH（q=0.0619）。C corrected BA 为 `0.6682`，低于原 STR 固定阈值 `0.6972`；所有方法均不是正式性能替代品。Pair-level 条件性描述显示，未恢复 H1-rescue pairs 原 STR margin 更负且 stable-error 参与更多，activity 表示到 train mean 的 L2 magnitude 未明显区分 recovered/unrecovered/newly broken；不支持单 activity 搜索或因果归因。

**最终机制与停止决定：**预设的宽泛五项判据在 activity raw stage 成立，但 H1 原有 AUROC gap 的 persistent hard-subject pairs 没有稳定净救回，combined decision stage 也没有稳定正的净 AUROC。结果与从 pre-aggregation activity 到 final decision 的部分压缩/利用损失相容，然而 C 与 F 的直接 AUROC 差未显著，不能证明压缩是因果原因；临床 phenotype overlap 仍无独立确证。**不启动 lightweight readout、decision-head 或任何新 deep model training；不重开 attention、gating、structured-readout、activity/frequency、contrastive 或 invariance 搜索。** 完整正负结果、15-split SD/median/paired CI、BH、subject/pair records、negative controls、smoke/verify 和统计限制见 `artifacts/prr01_frozen_str_pair_rescue_20260926/PRR01_REPORT.md`。

## PAG-01：Pre-Aggregation Disease Information Preservation & Ambiguity Diagnosis（2026-09-26）

仅使用 fixed 15 个 inner-development splits、STR seeds 42/43/44 的冻结 train/validation representations 与正式 final logits；没有访问 outer information、H1 residual 或 H1-rescue pairs，没有训练新深度模型或修改正式产物。核验 bilateral activity raw/context `11×258D`、original subject `258D`、structured ordered `176D`、combined decision `434D`，subject ID/label/split 对齐。Stable-error primary 沿用 DSG-01 seed-first 口径（STR 74/284/32），**只作 sensitivity，不用于 ambiguity 阈值**。单实例 smoke、全量 verify 均 PASS；score-only validation AUROC 与冻结 STR 正式分数 45/45 完全一致，100,695 个 pair 的净改变与 AUROC Δ 恒等式误差 `2.31×10⁻¹⁶`。

固定 final DD−PD score covariate，inner-train StandardScaler + PCA-16 + C=1 linear disease probe。Activity raw 的 validation AUROC 从 score-only `0.7176` 增至 `0.7350`，Δ `+0.0174`（12/15 split；bootstrap 95% CI `[+0.0078,+0.0261]`；5-stage primary BH q=0.0171）；activity context Δ `+0.0145`（13/15，q=0.0084）。Subject embedding Δ `−0.0212`（14/15 恶化，q=0.0084）；structured 与 combined decision 分别 `+0.0026/−0.0039`，不稳定。PCA-8/16/32 的 raw ΔAUROC 为 `+0.0098/+0.0174/+0.0223`，三档均为正。训练集 STR-score deciles 内 conditional permutation 10 次对照：raw 真实 Δ `+0.0174`、置换均值 `+0.0064`，配对差 `+0.0110`（11/15，CI `[+0.0048,+0.0176]`，BH q=0.0311）。**Raw/context 在 final score 之外含有可在 validation 泛化的部分增量 PD/DD ranking information。** 但 raw/context probe 的 log loss `0.798/0.801`，均差于 score-only `0.669`，不能作为正式模型或概率校准改进。

Inner-train 5-fold subject-level cross-fitting 定义 activity-raw incremental disease logit；固定 Ridge α=1 预测其 validation 对应量。PCA-16 raw/context/subject/structured/combined 的 signal R² 为 `0.834/0.794/0.456/0.555/0.544`；预测 signal 修正 score 后 AUROC Δ 为 `+0.0195/+0.0176/+0.0042/+0.0084/+0.0071`。**直接配对** raw−combined 的 disease AUROC 差 `+0.0213`（13/15，CI `[+0.0137,+0.0288]`，BH q=0.0005），signal R² 差 `+0.290`（15/15，q≈0.0001）；PCA-8/32 同方向。不过 raw→context AUROC 降幅不稳定，context→subject 明显下降 `0.0357`，subject→combined 又回升 `0.0173`；严格单调 AUROC 衰减仅 2/15 split，signal R² 为 0/15。故证据支持 activity/context→subject 的**局部可读性下降**，不支持全链路逐级单调 information-loss/compression 机制。

Ambiguity 仅用 inner-train score-only probe 的 OOF 绝对 disease margin 三分位阈值，再应用到 validation。High/medium/low 的 STR error rate 为 `37.5%/22.2%/14.6%`，high−medium/low 均 15/15 为正且 BH q<0.001；high 能识别更高错误负担。Raw probe 的组内 AUROC 增益为 `+0.0236/+0.0431/+0.0362`；high CI 跨零、q=0.107，medium/low 为正，但 high 与 medium/low 的**增益差未过 BH**。High-pair 原错恢复率 `19.8%`，低于 medium `24.0%` 和 low `34.6%`；high 的 net gain/pair CI 触零。不能声称 high 完全不可改善，也不能证明 high ambiguity 与 persistent stable-error 是独立机制；两者有部分重叠。

**重要边界：**5-fold probe cross-fitting 不能去掉冻结 STR 模型曾用 inner-train subjects 训练这一事实。Train OOF score-only AUROC `0.9541`，held-out validation 只有 `0.7176`；raw 对应 `0.9475/0.7350`。因此 OOF incremental target 和 train-derived ambiguity threshold 的 validation 尺度可能偏移，不能将它们等同于深度模型层面的真正 OOF disease evidence。15 splits 也共享同一开发队列，subjects/pairs/seeds/permutations 均非独立样本。

**最终决定：ALLOW_NEW_TRAINING = NO。** 预设闸门虽获得 raw 增量、conditional control 和 raw-vs-combined 直接差异，却没有稳定的逐级 attenuation；同时 OOF/validation 分布错位显著。停止把 PAG-01 直接转成 activity-to-subject information-preservation、auxiliary loss、decision head 或其他新模型训练；STR-01 保持冻结。完整正负结果、PCA-8/16/32、paired difference、effect size、bootstrap CI、BH、subject/pair records、stable-error sensitivity、smoke/verify 与全部统计限制见 `artifacts/pag01_preaggregation_disease_info_20260926/PAG01_REPORT.md`。

## FOE-01：Final Locked Outer Evaluation（2026-09-26）

**最终 outer protocol 在首次读取 FOE outer 结果之前锁定。**只评估 STR-01（主模型）、V8-GN/NR01（matched deep reference）与 H1（handcrafted reference）。锁文件 `artifacts/foe01_final_locked_outer_20260926/LOCKED_BEFORE_OUTER.json` 创建于 2026-09-26 11:49:33 UTC，SHA-256 为 `124d091122153ddceced6fb90d59b51611fa9b32ac2c90f5d2087501415eb64b`。STR 75,524 参数、V8 71,026 参数；full bilateral 11 activity、train-only normalization、固定 recipe/seeds 42/43/44。每个 outer fold/seed/model 的 final-refit epoch 为既有 3 个 inner best epochs 的中位数，DD 阈值仅由相应 inner OOF BA 选定，所有 30 个 epoch 数和阈值均记录在首次 outer access 之前。H1 使用既定 train-only handcrafted logistic pipeline 与 0.5 阈值。历史误生成 outer artifact 保持隔离，未用于本轮规则制定或选择。**outer information was not used for development/model selection and was accessed only after the final protocol was frozen.**

实际完成五个 subject-disjoint outer folds（390 unique subjects，PD 276、DD 114）、STR/V8 各 5 folds×3 seeds 共 30 次固定 final refit 与一次性 outer evaluation，以及 H1 的五次 train-only 拟合/evaluation。Postrun audit `PASS`：30 deep units、5 H1 folds 齐全，ID/label 对齐，train-only normalization 与 outer train IDs 相符，train/test 无重叠，每个单元只评估一次，lock SHA 未变。分析独立重算 metrics 和 confusion matrix；新增代码仅在独立 FOE artifact 的 `scripts/`，冻结模型 forward、正式 development 结果未改动；没有新的 diagnostic candidate 或 hyperparameter search。

**Primary outer 结果**（每 fold 内先平均三个 deep seeds，再等权平均五 fold；deep 使用预锁定 inner-OOF DD 阈值）：

| Model | Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall |
|---|---:|---:|---:|---:|---:|---:|
| STR-01 | 0.7317 | 0.6498 | 0.7218 | 0.6482 | 0.8456 | 0.4539 |
| V8-GN | 0.6905 | 0.6267 | 0.6801 | 0.6163 | 0.7792 | 0.4741 |
| H1 | 0.7281 | 0.6841 | 0.7659 | 0.6804 | 0.7899 | 0.5784 |

STR−V8 的 five-fold paired difference：Accuracy `+0.0412`（5/5）；BA `+0.0231`（4/5；fold-bootstrap 95% CI `[+0.0014,+0.0447]`）；AUROC `+0.0417`（3/5；`[+0.0077,+0.0757]`）；Macro-F1 `+0.0319`（5/5）；PD Recall `+0.0664`（3/5）；**DD Recall `−0.0202`**（2/5；`[−0.1187,+0.0819]`）。STR−H1 AUROC `−0.0441`（仅 1/5 STR 更高；CI `[−0.0912,−0.0077]`）；STR−H1 BA `−0.0344`（CI 跨零）。五 fold 的 STR−V8 AUROC 差为 `−0.0027/+0.0582/−0.0032/+0.0501/+0.1061`，有明显 context 异质性；不能将 mean gain 理解为所有 fold 均改善。Confusion matrix 按 PD=0/DD=1、`[[PD→PD,PD→DD],[DD→PD,DD→DD]]`，三个 deep seed 的 pooled count 平均为 STR `[[233.33,42.67],[62.00,52.00]]`、V8 `[[215.00,61.00],[59.67,54.33]]`；H1 为 `[[218,58],[48,66]]`。这些平均计数不是 seed ensemble，也不扩大 subject 样本量。

**Development/outer boundary：**正式 development 用 0.5 decision rule，因此同阈值可比时 STR BA `0.6972→0.6527`、DD Recall `0.6165→0.4597`、AUROC `0.7176→0.7218`；V8 BA `0.6707→0.6305`、DD Recall `0.5631→0.4831`、AUROC `0.6800→0.6801`；H1 BA `0.6918→0.6841`、AUROC `0.7515→0.7659`。STR−V8 在 development 的 DD Recall 增益 `+0.0534` 未在 outer 保持，outer 同阈值为 `−0.0234`。预锁定 outer 阈值下 STR BA `0.6498`，与 outer 0.5 的 `0.6527` 接近；下降不能只归因于 threshold。H1−STR AUROC gap 由 development `+0.0339` 到 outer `+0.0441`。

**最终结论：**STR 相对 matched V8 的平均 BA/AUROC、Accuracy/Macro-F1 提升获得独立于 development model selection 的 outer 支持，但 DD recognition 提升没有保持；outer BA 优势主要伴随 PD Recall 增加，不能继续声称“不明显牺牲 PD Recall 地改善 DD recognition”。H1 仍有较高 AUROC。STR thresholded class balance 出现 generalization limitation，尤其 DD Recall；AUROC 大致保持。只有五个同源队列 outer folds、训练集在 folds 间重叠，fold-bootstrap CI 有限；outer final refit 与 inner development 训练样本量/停训规则不同，development→outer 差异不能单独归因于 dataset shift。这不是独立外部队列验证。DSG/RGD/PRR/PAG 的 representation、persistent-error、residual ranking、ambiguity 与局部 stage attenuation 均为 **development-only mechanism evidence**，没有在 outer 上重新进行机制搜索。STR-01 冻结和 PAG 的 `ALLOW_NEW_TRAINING = NO` 不变；不得据同一 outer set 修改模型、recipe、阈值后再次声称独立验证。

逐 fold 所有六项指标、paired CI、完整 confusion、预测、锁文件、审计和正负结论见 `artifacts/foe01_final_locked_outer_20260926/FOE01_REPORT.md` 及其 `analysis/`、`model_runs/`。

## Phase-2 development: STR-01 training, imbalance, and mild augmentation study (2026-09-26–27)

**Scope and evidence boundary.** This is a new inner-development study, using the fixed 15 subject-level development splits and seeds 42/43/44. The original FOE-01 outer artifacts/results were frozen and were not used for Phase-2 selection, tuning, augmentation design, debugging, or validation. The formal STR-01 architecture, data/activity/wrist order, and train-only normalization were retained. New outputs are isolated under `artifacts/phase2_str_training_20260926/`; the prior formal STR-01, V8-GN, H1, and outer archives were not overwritten. The full [Phase-2 report](artifacts/phase2_str_training_20260926/PHASE2_REPORT.md), [stage-entry model summary](artifacts/phase2_str_training_20260926/PHASE2_STAGE_SUMMARY.md), [bounded protocol](artifacts/phase2_str_training_20260926/PHASE2_PROTOCOL.md), [augmentation evidence](artifacts/phase2_str_training_20260926/AUGMENTATION_EVIDENCE.md), [manifest](artifacts/phase2_str_training_20260926/manifest.json), and [analysis tables](artifacts/phase2_str_training_20260926/analysis/) record all positive and negative results.

**Baseline reproducibility and training adequacy.** Reproduced formal STR-01 on all 45 inner runs. Best epoch matched 45/45; model-state tensors were bitwise identical 45/45; validation DD probabilities matched by subject ID exactly (maximum absolute difference 0); normalization hashes matched. Seed-first 15-split means: Accuracy **0.7307**, BA **0.6972**, AUROC **0.7176**, Macro-F1 **0.6861**, PD Recall **0.7779**, DD Recall **0.6165**. Across-seed SDs in that order: 0.0151/0.0117/0.0120/0.0107/0.0327/0.0408; across 15 seed-averaged splits: 0.0306/0.0316/0.0393/0.0316/0.0440/0.0622. Best epoch median 12 (range 6–31), with none at the epoch-50 cap. At the selected epoch, mean train/validation loss was 0.3388/0.7147; at stop it was 0.0415/1.0865. In all 45 baseline runs, training loss fell and validation loss rose after the selected epoch. This does not support a simple undertraining or too-short patience explanation; later training shows overfitting risk.

**Controlled study.** Baseline plus 16 single-factor candidates each completed 15 splits × 3 seeds (765 runs total): LR 1e-4, WD 1e-3, label smoothing 0.05; unweighted CE, sqrt-weighted CE, balanced focal loss (gamma 2), effective-number CE (beta 0.99), and balanced train-subject sampler; jitter, scaling, mild temporal displacement, and same-class conservative mixup, each with equal-class or DD-only perturbation. All augmentation stayed within inner-train and did not add independent DD subjects. Seven smoke runs passed. The final integrity audit found 765/765 complete, with identical train/validation IDs and labels, train-only normalization hashes, unique validation predictions, and explicit `outer_test_loader_created=false`; zero audit issues. Each variant's six metrics, seed SD, per-epoch class/augmentation exposure, paired split differences, bootstrap 95% CIs, and primary-family BH-FDR values are archived in the report and analysis CSVs. The 15 seed-averaged splits, not 45 seed-runs or augmented exposures, are the paired comparison units.

| Phase-2 configuration | Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Reproduced STR-01 | 0.7307 | 0.6972 | 0.7176 | 0.6861 | 0.7779 | 0.6165 | Retain |
| Best observed AUROC candidate: balanced focal | 0.7195 | 0.6979 | 0.7247 | 0.6809 | 0.7503 | 0.6455 | Reject: joint metrics/PD Recall/seed stability |
| Unweighted CE | 0.7399 | 0.6676 | 0.7083 | 0.6717 | 0.8418 | 0.4934 | Reject: Accuracy-only trade-off |
| Balanced sampler | 0.6908 | 0.6790 | 0.7059 | 0.6547 | 0.7080 | 0.6499 | Reject: DD gain with larger PD and BA losses |
| Label smoothing 0.05 | 0.7187 | 0.6966 | 0.7210 | 0.6800 | 0.7500 | 0.6432 | Reject: PD and Macro-F1 loss |

**Findings.** Unweighted CE increased Accuracy by +0.0091 but decreased BA by −0.0296 (95% CI −0.0385 to −0.0203), DD Recall by −0.1231 (−0.1628 to −0.0843), and Macro-F1 by −0.0143. Thus the current train-balanced CE materially protects DD recognition. Balanced sampling increased DD Recall +0.0334 but lowered PD Recall −0.0698 and BA −0.0182; it resampled existing DD subjects, rather than creating independent subjects. Focal loss gave the largest observed AUROC difference (+0.0071, 10/15 positive splits) but its CI crossed zero (−0.0037 to +0.0175), BA difference was only +0.0007 (6/15), Macro-F1 fell −0.0052, PD Recall fell −0.0275, and AUROC seed SD rose from 0.0120 to 0.0224. Label smoothing improved DD Recall +0.0267 but lowered PD Recall −0.0278 and Macro-F1 −0.0061. None of the eight mild augmentation policies produced a stable joint BA/AUROC/Macro-F1 improvement with DD gain and preserved PD Recall. DD-only perturbation left the DD *subject exposure share* near 29.2%; only the separate sampler changed it to approximately 49.9%. No augmentation combination was run because no single policy met the recorded joint gate (set after baseline reproduction and an interim check of two already-complete negative candidates, before the remaining matrix was inspected). No positive BA/AUROC comparison survived BH-FDR; 0/16 candidates met the retention gate.

**Phase-2 decision.** The development-only best configuration remains the exact original STR-01; optimized, augmented, and final Phase-2 candidate therefore coincide with it, with **zero retained paired lift** over original STR-01. The present results show sensitivity to class weighting, but do not establish imbalance as the sole or main residual performance bottleneck. The bounded study does not prove that the recipe is globally optimal: optimizer, scheduler, batch size, and fixed-architecture dropout were not newly varied for STR-01 in this matrix; baseline trajectories did not justify extending epoch cap or patience. Broad recipe or synthetic-data searching on these same development splits is not supported. These are development-only training findings and do not update or reinterpret the locked FOE-01 outer evaluation.

Phase-2 next-step decision: the tested training strategies do not justify continuing hyperparameter or augmentation search on these same 15 splits. Earlier mechanism studies likewise do not justify reviving stopped architecture directions. Keep original STR-01 as the formal pure-deep baseline; seek independent evidence before proposing another model change. This decision uses development results only and does not reuse FOE-01 outer data.

## PURE-DEEP PHASE-3 DEVELOPMENT — E1–E4 (2026-09-28)

**Evidence boundary.** Phase-3A used only the fixed 15 subject-level inner-development splits, seeds 42/43/44, existing exact STR-01 inner predictions/checkpoints and train-only normalization. FOE-01 outer results remained frozen and were not used for training, debugging, selection or threshold choice. H1 and all handcrafted/statistical models were excluded from model input, fusion, teacher/student, loss, representation construction, selection and threshold choice. Formal STR-01 and prior archives were not overwritten. Full [Phase-3A report](artifacts/phase3a_puredeep_20260928/PHASE3A_REPORT.md), [fixed protocol](artifacts/phase3a_puredeep_20260928/PHASE3A_PROTOCOL.md), [registered configurations](artifacts/phase3a_puredeep_20260928/manifest.json), [smoke/logits audit](artifacts/phase3a_puredeep_20260928/SMOKE_AUDIT.json), [paired analysis](artifacts/phase3a_puredeep_20260928/analysis/phase3_paired.csv), and [integrity audit](artifacts/phase3a_puredeep_20260928/analysis/phase3_integrity_audit.json) are in a new independent directory.

**E1 — sampling/loss coupling.** Phase-2 `balanced_sampler` indeed combined balanced subject sampling with the original train-balanced CE. The missing control, balanced sampling + unweighted CE, improved BA relative to that double-compensated condition by **+0.0115** (10/15 paired splits; bootstrap 95% CI +0.0022 to +0.0208), showing that double compensation caused part of the earlier deficit. It still fell below natural sampling + balanced CE: BA **0.6905 vs 0.6972** (Δ−0.0067; 3/15 positive; CI −0.0122 to −0.0006), AUROC **0.7167 vs 0.7176**, Macro-F1 **0.6732 vs 0.6861**. Existing DD subjects were resampled, not multiplied as independent subjects. **Stop sampling/class-exposure search.**

**E2 — three-seed probability ensemble.** Validation subjects and labels were aligned by subject ID; DD probabilities were averaged before metrics, using the original 0.5 rule. This differs materially from averaging three per-seed metrics. AUROC rose from **0.7176 to 0.7586** (paired Δ+0.0410, 15/15 positive; CI +0.0324 to +0.0493; BH q=0.0001). Accuracy rose 0.7307→0.7489 and Macro-F1 0.6861→0.6976; BA changed only 0.6972→0.6998 (8/15; CI spans zero). PD Recall rose 0.7779→0.8180, while **DD Recall fell 0.6165→0.5816** (CI for Δ −0.0564 to −0.0125). This is a strong development-only ranking/inference finding, not a new single-model architecture; its fixed-threshold DD trade-off prevents automatic adoption as the default classification strategy. No threshold was tuned.

**E3 — fixed SAM rho=0.05 with AdamW.** Train/validation CE gap at best epoch decreased 0.3759→0.2542, and median best epoch moved 12→17, but classification did not jointly improve: Accuracy **0.7452**, BA **0.6935** (Δ−0.0037; 5/15; CI −0.0133 to +0.0068), AUROC **0.7213** (Δ+0.0036; 10/15; CI spans zero), Macro-F1 **0.6915**, PD Recall **0.8182**, DD Recall **0.5687** (Δ−0.0479; CI −0.0929 to −0.0018). BA and AUROC seed SD increased from 0.0117/0.0120 to 0.0209/0.0318. The Accuracy gain favors PD and harms DD recognition. **Reject SAM; no rho search or SAM+auxiliary combination.**

**E4 — pre-aggregation activity auxiliary supervision.** A training-only shared 258→32→2 branch consumed 11 bilateral activity representations; its masked subject-level auxiliary prediction alone received balanced CE. The original STR final logits remained the sole validation and inference output. With the same frozen checkpoint/input, wrapped and original final logits were bitwise identical (maximum difference 0). Both fixed λ completed 45/45 runs:

| Single-model configuration | Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall |
|---|---:|---:|---:|---:|---:|---:|
| Original STR-01 / λ=0 | 0.7307 | 0.6972 | 0.7176 | 0.6861 | 0.7779 | 0.6165 |
| Activity auxiliary λ=0.1 | 0.7239 | 0.6943 | 0.7182 | 0.6812 | 0.7655 | 0.6232 |
| Activity auxiliary λ=0.3 | 0.7277 | 0.6946 | 0.7154 | 0.6838 | 0.7746 | 0.6146 |

λ=0.1 had paired BA Δ−0.0029 (5/15; CI −0.0074 to +0.0022), AUROC Δ+0.0005 (7/15; CI −0.0049 to +0.0055), and Macro-F1 Δ−0.0049. λ=0.3 had BA Δ−0.0026 (6/15; CI −0.0071 to +0.0025), AUROC Δ−0.0023 (8/15; CI −0.0086 to +0.0038), and Macro-F1 Δ−0.0022. Neither improved final classification; per the fixed protocol, no follow-up representation probe or further λ/auxiliary-objective search was run. **Reject E4 as a retained training mechanism.**

**Statistics, integrity and decision.** The three seeds were aggregated before comparing the 15 fixed splits. Full six-metric means/SDs, paired differences, positive-split counts, 10,000-draw split-bootstrap 95% CIs, paired Cohen dz and BA/AUROC BH-FDR are archived in `artifacts/phase3a_puredeep_20260928/analysis/`. Subjects, activities and 45 seed-runs were not treated as independent statistical units. The new training matrix completed **180/180** runs; 180/180 matched baseline train/validation IDs, labels and train-only normalization, and none created an outer test loader. Source hashes matched; zero audit issues. No positive new-training BA/AUROC comparison survived BH-FDR, and **0/4** new training variants passed the fixed retention gate. Split bootstrap intervals describe the repeated development design, not an external cohort.

**CASE D:** The best single-model pure-deep configuration remains the original formal STR-01, unchanged. Stop further supervised STR architecture/loss/sampling/SAM/activity-auxiliary searching on these same PADS development subjects. The next direction is a **separately planned external wearable self-supervised pretraining** study with public large-scale IMU data and pure-deep transfer; no such pretraining was started here. This Phase-3A conclusion is development-only and does not update or reinterpret the locked FOE-01 outer evaluation.

## PURE-DEEP PHASE-3B DEVELOPMENT — frozen external wearable SSL transfer (2026-09-28)

**Evidence boundary and assets.** This study used only the fixed 15 inner-development subject splits and seeds 42/43/44. FOE-01 outer predictions/results remained frozen and were not used for training, debugging, model/threshold choice or validation. H1, handcrafted statistics, teacher/student distillation and statistical-model fusion were excluded. The original STR-01, V8-GN, Phase-2/3A and FOE-01 formal artifacts were not overwritten. The sole external model was the official OxWearables `ssl-wearables` HarNet10, Git commit `150550ea5d41800229c95e36f88f5bf0d2e7cf04`, checkpoint SHA-256 `c64f9135d99e2dcdfc9ae7cc0672f2bcc438df9ceb8215665882f92cddd162a6`; all 131 extractor weights matched. Its 10,457,408-parameter feature extractor remained frozen. PADS raw Acc XYZ in g, after fixed 48-sample trim, was clipped to ±3g and anti-alias resampled 100→30 Hz. The fixed first/last 10-second window rule for long records and 12-sample edge padding for shorter records produced 10,920 windows, cached by subject/activity/left-right wrist as 1024D features. The STR path retained its original processed six-channel input, train-only normalization and 0.5 DD threshold. Full [protocol](artifacts/phase3b_external_ssl_20260928/PHASE3B_PROTOCOL.md), [report](artifacts/phase3b_external_ssl_20260928/PHASE3B_REPORT.md), [manifest](artifacts/phase3b_external_ssl_20260928/manifest.json), [analysis](artifacts/phase3b_external_ssl_20260928/analysis/) and [official encoder source](https://github.com/OxWearables/ssl-wearables) are linked here.

**Controlled candidates and verification.** A was pretrained SSL-only with a light ordered bilateral subject classifier; B added a zero-initialized projected pretrained SSL residual at the STR wrist embedding; C used the identical frozen HarNet10 architecture with fixed random weights as the capacity control. B/C each had 143,172 trainable parameters versus original STR 75,524; A had 72,146. All used the original balanced CE, optimizer, scheduler and early-stopping recipe. A/B/C smoke tests and checkpoint reload passed; the original formal STR checkpoint gave exactly identical logits before and after insertion of the zero residual (maximum absolute difference 0). Frozen caches had no gradients. All **135/135** candidate runs completed. Validation subject IDs/labels and train-only normalization hashes matched original STR for every run; checkpoints were present, no outer test loader was created, and source hashes passed. Original Phase-3A E2 ensemble values were exactly reproduced by subject-ID probability alignment.

**Seed-first single-model development means** (three seeds averaged within each split; 15 splits averaged equally):

| Model | Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall |
|---|---:|---:|---:|---:|---:|---:|
| Original STR-01 | 0.7307 | 0.6972 | 0.7176 | 0.6861 | 0.7779 | 0.6165 |
| A: pretrained SSL-only | 0.7348 | 0.6949 | 0.7300 | 0.6865 | 0.7911 | 0.5987 |
| B: STR + pretrained-frozen SSL | **0.7456** | **0.7196** | **0.7596** | **0.7066** | **0.7823** | **0.6569** |
| C: STR + random-frozen SSL | 0.7272 | 0.6862 | 0.7089 | 0.6781 | 0.7850 | 0.5874 |

**Positive result and control.** B−STR paired BA **+0.0225** (11/15 improving splits; split-bootstrap 95% CI [+0.0094,+0.0362], paired dz 0.815, eight-primary-test BH q=0.0134) and AUROC **+0.0419** (15/15; [+0.0298,+0.0541], dz 1.693, q=0.0002). Macro-F1 rose +0.0205, DD Recall +0.0404 (11/15; CI [+0.0064,+0.0752]), while mean PD Recall changed +0.0045 (CI spans zero). B−C BA **+0.0334** (14/15; CI [+0.0222,+0.0449], q=0.0005) and AUROC **+0.0506** (15/15; [+0.0366,+0.0652], q=0.0002); C was below STR on both primary metrics. Thus the frozen pretrained representation provided development gain beyond the extra residual/projection capacity control. A had AUROC >0.5 in every split (mean 0.7300), supporting transferable PD/DD signal, but its BA −0.0023 and AUROC +0.0123 relative to STR had CIs crossing zero, and DD Recall decreased −0.0179; SSL-only is not retained as the main classifier. Secondary metrics were descriptive, not included in the primary BH family.

**Seed stability and inference reference.** B versus STR mean within-split seed SD decreased for BA 0.0300→0.0216 (paired CI for difference [−0.0136,−0.0033]) and AUROC 0.0323→0.0217 ([−0.0177,−0.0039]); DD Recall seed SD 0.1024→0.0785 had a CI crossing zero. Across-seed DD-score Spearman increased 0.6000→0.7734 (15/15 splits; paired CI for +0.1734 [+0.1358,+0.2093]); 0.5-threshold prediction disagreement fell 23.72%→17.80%. Three seed probabilities averaged by subject ID gave ensemble AUROC A/B/C **0.7414/0.7816/0.7389** versus reproduced original E2 STR **0.7586**. B ensemble−E2 AUROC was +0.0230 (8/15; bootstrap CI [+0.0044,+0.0420]); its BA +0.0202 CI crossed zero. B single-model AUROC 0.7596 approximately matched the earlier STR ensemble 0.7586, and B ensemble DD Recall 0.6301 exceeded original E2 0.5816. Ensemble and single-model figures are distinct estimands; no validation threshold was searched.

**Decision and limits.** B passed every preregistered frozen-transfer retention condition: positive, majority-split and bootstrap-supported BA/AUROC gains over both STR and C; no mean Macro-F1, DD Recall or PD Recall harm; reduced BA/AUROC seed variation. **B is the new best pure-deep development configuration.** Formal STR-01 and locked FOE-01 outer conclusions remain unchanged; B has no independent outer evaluation. The 15 splits reuse the same PADS cohort, and three seeds give limited variance precision. Windows, subjects, seed-runs and subject pairs were not used as independent statistical units. This result justifies planning one separately registered **Phase-3C minimal external-SSL adaptation** test (last-block unfreezing *or* lightweight adapter), with a smaller encoder learning rate; no adaptation, full fine-tuning, new external-model search or complex fusion was run in Phase-3B. FOE-01 outer information must remain excluded from Phase-3C selection.

## PURE-DEEP PHASE-3C DEVELOPMENT — minimal SSL adaptation and decision optimization (2026-09-30–10-01)

**Evidence boundary and fixed plan.** Phase-3C used only the fixed 15 subject-level inner-development splits and seeds 42/43/44. Three seeds were averaged within split before paired inference. The historical locked FOE-01 outer results were not used to train, debug, choose a threshold or model, or validate Phase-3B/C. H1/handcrafted features, teacher/student, augmentation, new head/backbone, broad layer/LR search and full encoder fine-tuning were excluded. The [frozen protocol](artifacts/phase3c_ssl_adaptation_20260930/PHASE3C_PROTOCOL.md), [complete report](artifacts/phase3c_ssl_adaptation_20260930/PHASE3C_REPORT.md), [manifest](artifacts/phase3c_ssl_adaptation_20260930/manifest.json), [analysis tables](artifacts/phase3c_ssl_adaptation_20260930/analysis/) and [E2 analysis correction note](artifacts/phase3c_ssl_adaptation_20260930/ANALYSIS_FIX_NOTE.md) are archived separately; prior formal STR, Phase-3B and FOE-01 artifacts were not overwritten.

**E1/E3 model comparison, seed-first 15-split means:**

| Single model | Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Original STR-01 | 0.7307 | 0.6972 | 0.7176 | 0.6861 | 0.7779 | 0.6165 | Phase-1 formal reference |
| Phase-3B pretrained-frozen WSSL-STR | 0.7456 | 0.7196 | 0.7596 | 0.7066 | 0.7823 | 0.6569 | **Retain** |
| E1 last-block adapted WSSL | 0.7518 | 0.7244 | 0.7689 | 0.7128 | 0.7905 | 0.6583 | Reject preset BA gate |
| E3 middle+final frozen WSSL | 0.7537 | 0.7204 | 0.7667 | 0.7112 | 0.8006 | 0.6402 | Reject BA and DD gates |

E1 was a single registered HarNet layer5 configuration (encoder LR `2e-5`, STR/projection `2e-4`; layer1–4 and BN running statistics frozen). The same Phase-3B checkpoint/input gave identical logits before/after the isolated interface change (maximum difference 0). All 45 E1 checkpoints, subject IDs, normalization hashes, LR ratios and frozen BN statistics passed audit. Relative to frozen WSSL, E1 BA was +0.0048 (10/15; 15-split bootstrap 95% CI [−0.0006,+0.0102], paired dz 0.431, four-primary-test BH q=0.2251), while AUROC was +0.0094 (11/15; [+0.0030,+0.0154], dz 0.728, q=0.0603). Within-split seed BA/AUROC SD was 0.0202/0.0222 versus 0.0216/0.0217 frozen; score Spearman increased 0.7734→0.7940, but 0.5-threshold disagreement was 0.1780→0.1786. Thus partial fine-tuning did not establish a stable joint BA/AUROC/seed advantage and was not retained. E3 tested one prespecified layer4 512D + final 1024D feature concatenation. All 45 runs and checkpoint/ID/normalization checks passed; versus final-only BA was +0.0008 (6/15; CI [−0.0038,+0.0054], q=0.8040), AUROC +0.0071 (10/15; [−0.0024,+0.0156], q=0.2251), and DD Recall −0.0167 (CI crosses zero). Middle+final was rejected; no layer search follows.

**E2 train-only operating point.** Exactly 135 subject-level OOF crossfit fits (15 target splits × 3 seeds × 3 folds) used only target inner-train subjects and a fixed epoch-10 rule; OOF labels were never used for checkpoint selection. The fixed OOF BA grid and tie rule produced mean thresholds 0.4071 for single models and 0.3927 for the three-seed ensemble. Applied to the archived WSSL validation probabilities, single-model BA fell 0.7196→0.6771 (0/15 improving splits; CI for −0.0426 [−0.0682,−0.0227], BH q=0.0001) and Macro-F1 0.7066→0.6325; DD Recall rose 0.6569→0.6983 while PD Recall fell 0.7823→0.6559. Ensemble BA fell 0.7200→0.6845 (4/15; CI for −0.0355 [−0.0764,+0.0008], BH q=0.0962) and Macro-F1 0.7141→0.6432; DD Recall rose 0.6301→0.7040 while PD Recall fell 0.8100→0.6650. AUROC stayed 0.7596 single / 0.7816 ensemble because thresholding cannot change ranking. Keep fixed 0.5. A post-training audit-count typo and ensemble aggregation indentation error were fixed in analysis-only scripts before interpreting results; original sources/hashes and correction ledger are preserved. OOF/full-train score-distribution shift is a plausible, untested reason for poor threshold transfer.

**E4 fixed subject-count curve, mechanism-only.** All 270 new 25/50/75% training runs completed; archived 100% STR/WSSL runs were reused. All 360 total units passed validation ID/label, matching train-subject and train-only-normalization, nested subset, train/validation separation and no-outer-loader checks; all 270 new checkpoint hashes matched. WSSL−STR BA differences at 25/50/75/100% were respectively +0.0311/+0.0249/+0.0178/+0.0225 (11/11/10/11 of 15 improving splits; bootstrap CIs all positive; eight-primary-test BH q=0.0086/0.0345/0.0479/0.0090). AUROC differences were +0.0528/+0.0446/+0.0306/+0.0419 (12/13/12/15 improving; CIs all positive; q=0.0034/0.0086/0.0086/0.0005). Thus frozen SSL helps at every tested training size. However the **direct 25%−100% advantage interaction** was only BA +0.0087 (CI [−0.0130,+0.0291]) and AUROC +0.0109 ([−0.0098,+0.0319]); 50%/75% interactions also crossed zero. The claim that SSL's BA/AUROC benefit is specifically larger in smaller clinical cohorts is not supported. At 25%, WSSL−STR DD Recall was +0.1204, versus +0.0404 at 100%; the interaction +0.0800 had a positive unadjusted bootstrap CI but BH q=0.0925 across three DD interactions, so is exploratory. At 25/50%, stronger DD recognition partly traded off PD Recall and Accuracy; seed variance was not uniformly lower for WSSL at every fraction. Complete six-metric curves, CIs and seed SDs are in the report and CSVs.

**Phase-3C decision: CASE D.** The best single-model **pure-deep development-only** configuration remains Phase-3B **pretrained-frozen WSSL-STR**, using final HarNet10 1024D wrist features, original balanced CE/optimizer/early stopping and fixed 0.5 threshold: Accuracy **0.7456**, BA **0.7196**, AUROC **0.7596**, Macro-F1 **0.7066**, PD Recall **0.7823**, DD Recall **0.6569**. E1, E3 and E2 thresholding are not retained; E4 does not select a model. Stop further SSL encoder adaptation, layer and threshold search on the same development cohort. The three-seed WSSL ensemble AUROC 0.7816 remains an inference-only reference. No Phase-3B/C method has a new independent outer evaluation; FOE-01 remains locked and cannot be reused for their selection or claimed as their independent test. Next work is paper experiment consolidation and planning genuinely independent validation. Split-bootstrap and BH results use 15 overlapping fixed development splits, not 45 seed-runs, subjects, OOF folds, windows or fractions as independent samples.

## PURE-DEEP PHASE-3D DEVELOPMENT — bounded advanced SSL adaptation (2026-10-01)

**Scope and evidence boundary.** The user explicitly authorized a new Phase-3D after Phase-3C CASE D. Its [fixed protocol](artifacts/phase3d_ssl_adaptation_20261001/PHASE3D_PROTOCOL.md) was recorded before Phase-3D validation inspection. The same 15 fixed subject-level inner-development splits and seeds 42/43/44, original STR architecture, bilateral/activity/structured path, balanced CE, train-only normalization, fixed 0.5 rule and Phase-3B official pretrained HarNet10 were retained. FOE-01 outer results were not used for Phase-3D model, threshold or method selection. H1, handcrafted features, teacher/student, new backbone, ordinary augmentation/loss/sampling and full HarNet fine-tuning were excluded. Outputs are isolated under `artifacts/phase3d_ssl_adaptation_20261001/`; formal Phase-1–3C and FOE artifacts were not overwritten. The full [Phase-3D report](artifacts/phase3d_ssl_adaptation_20261001/PHASE3D_REPORT.md), [manifest](artifacts/phase3d_ssl_adaptation_20261001/manifest.json), [analysis tables](artifacts/phase3d_ssl_adaptation_20261001/analysis/) and [E4 numerical smoke note](artifacts/phase3d_ssl_adaptation_20261001/DOMAIN_SMOKE_NOTE.md) preserve methods, audits, exact statistics and negative findings.

**Seed-first 15-split development means** (three seeds averaged within each split):

| Configuration | Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Original STR-01 | 0.7307 | 0.6972 | 0.7176 | 0.6861 | 0.7779 | 0.6165 | Frozen reference |
| Phase-3B pretrained-frozen WSSL-STR | **0.7456** | **0.7196** | **0.7596** | **0.7066** | **0.7823** | **0.6569** | **Retain** |
| Phase-3C last-block adapted WSSL | 0.7518 | 0.7244 | 0.7689 | 0.7128 | 0.7905 | 0.6583 | Historical rejected candidate |
| E1 64D bottleneck adapter | 0.7452 | 0.7195 | 0.7643 | 0.7064 | 0.7815 | 0.6576 | Reject |
| E2 128→32→64 gated wrist fusion | 0.7520 | 0.7178 | 0.7581 | 0.7092 | 0.8004 | 0.6351 | Reject |
| E4 train-only masked-domain SSL | 0.7362 | 0.7127 | 0.7572 | 0.6971 | 0.7696 | 0.6557 | Reject |

**E1 adapter.** One `1024→64→GELU→1024` zero-initialized residual adapter was trained before the unchanged SSL wrist projection; HarNet stayed frozen. Initial logits and logits from the loaded Phase-3B checkpoint were exactly consistent (max difference 0). All 45 runs passed checkpoint, subject-ID/label, train-only normalization and no-outer-loader audits; adapter weights changed in every run. Adapter−frozen BA was −0.0001 (9/15 positive; 15-split bootstrap CI [−0.0053,+0.0044]; six-primary-test BH q=0.9780), AUROC +0.0048 (10/15; [−0.0011,+0.0100]; q=0.3616), Macro-F1 −0.0001, DD Recall +0.0007. BA within-split seed SD decreased 0.0216→0.0164, but AUROC SD increased 0.0217→0.0268; probability agreement was slightly worse. Versus Phase-3C **last-block** adaptation, adapter BA −0.0049 and AUROC −0.0046 both had CIs crossing zero. Full encoder fine-tuning was not run; no adapter-versus-full-fine-tuning claim is possible.

**E2 gate and conditional E3.** The single fixed gate used the 64D STR wrist feature and 64D projected SSL feature; no attention or gate-width search. Its new weights changed in all 45 runs. Gate−frozen BA −0.0019 (4/15; CI [−0.0055,+0.0024], q=0.5048), AUROC −0.0015 (7/15; [−0.0077,+0.0039], q=0.9780), while DD Recall fell −0.0218 (CI [−0.0408,−0.0020]). Accuracy alone rose +0.0064; AUROC seed SD and prediction disagreement increased. Thus learnable wrist gating did not establish better SSL use. E2 failed its preregistered gate, so the proposed 11-activity-specific lightweight E3 gate was **not run**. Its effect remains unmeasured, not negative.

**E4 train-only masked PADS-domain adaptation.** Because neither E1 nor E2 was retained, the one pre-registered fallback E4 was run. Each of 45 split-seed units used only its inner-training raw Acc windows for fixed five-epoch 15%-masked reconstruction; HarNet layers1–4 and BN running statistics were frozen, layer5 LR was `1e-5`, and the temporary reconstruction decoder was discarded before training the unchanged WSSL-STR classifier. An initial smoke exact-cache assertion stopped before formal training; rare floating convolution differences from fold-local batching were quantified against Phase-3C layer4 features and bounded with max/mean/p99 tolerance without changing raw preprocessing, subjects or model selection. The corrected smoke passed. The completed audit found 45/45 feature/checkpoint hashes matching, no outer raw subject access, **zero validation windows in self-supervised adaptation**, and only the fixed inner-train subjects in its loss. Masked loss fell in 45/45 units (mean 0.02661→0.01882); layer5 weights changed. Yet E4−frozen BA was −0.0070 (5/15; CI [−0.0149,+0.0003], q=0.3616), AUROC −0.0023 (8/15; [−0.0080,+0.0028], q=0.9780), Macro-F1 −0.0095 (CI [−0.0204,−0.0007]), and AUROC seed SD rose 0.0217→0.0271. Prediction disagreement rose 0.1780→0.2205. A better reconstruction objective did not yield better PD/DD transfer.

**Parameter counts and decision.** Frozen WSSL-STR has 10,600,580 full-system parameters including the 10,457,408-parameter frozen HarNet extractor, with 143,172 classifier-side trainable parameters. E1 adds 132,160 trainable adapter parameters; E2 adds 6,240 trainable gate parameters. E4 temporarily trains 2,623,488 layer5 + 922,500 decoder parameters for its self-supervised stage, then trains only the original 143,172 classifier-side parameters. **No Phase-3D candidate passed the joint BA/AUROC/recall/seed-stability retention gate.** The current retained best single-model pure-deep development configuration remains pretrained-**frozen** WSSL-STR with the original recipe and 0.5 threshold. Stop adapter-width, gate, activity-gate and masked-domain-SSL searches on these same development subjects. This bounded result does not prove global optimality. Repeated development phases on overlapping subjects limit confirmatory inference, and no Phase-3B/D method has a new independent outer evaluation. Continue with paper evidence consolidation and planning genuinely independent validation; do not reuse FOE-01 outer information to select or claim validation for these methods.


## Phase-3E External Wearable SSL Prior Diversification (2026-10-01)

**已完成并审计：DEVELOPMENT-ONLY / PURE-DEEP。** 新增 180 runs（HarNet5、BioPM-only、BioPM residual、random BioPM residual，各 15 splits × 3 seeds）。[固定 protocol](artifacts/phase3e_external_priors_20261001/PHASE3E_PROTOCOL.md)、[官方资产核验](artifacts/phase3e_external_priors_20261001/OFFICIAL_ASSET_AUDIT.md)、[完整报告](artifacts/phase3e_external_priors_20261001/PHASE3E_REPORT.md)、[最终审计](artifacts/phase3e_external_priors_20261001/FINAL_AUDIT.json)、[统计表](artifacts/phase3e_external_priors_20261001/analysis/)。

| Configuration | Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall |
|---|---:|---:|---:|---:|---:|---:|
| str | 0.7307 | 0.6972 | 0.7176 | 0.6861 | 0.7779 | 0.6165 |
| frozen | 0.7456 | 0.7196 | 0.7596 | 0.7066 | 0.7823 | 0.6569 |
| harnet5 | 0.7236 | 0.6967 | 0.7238 | 0.6811 | 0.7612 | 0.6323 |
| biopm_only | 0.6963 | 0.6438 | 0.6659 | 0.6347 | 0.7705 | 0.5171 |
| biopm | 0.7238 | 0.6924 | 0.7181 | 0.6806 | 0.7680 | 0.6168 |
| biopm_random | 0.7316 | 0.6932 | 0.7205 | 0.6846 | 0.7855 | 0.6009 |

1. **5s prior 没有优于 10s prior。** HarNet5 BA 0.6967、AUROC 0.7238；相对 HarNet10 分别 −0.0229 / −0.0358，只有 4/15 / 1/15 splits 改善，seed 波动增大。停止本次 temporal-scale 路线。官方 family 的维度和参数量也不同，不能将差异仅归因于时间尺度。
2. **本轮 Bio-PM acc-only 未建立有用的增量迁移收益。** BioPM-only BA 0.6438 / AUROC 0.6659；STR+BioPM 为 0.6924 / 0.7181。相对原 STR 的 BA −0.0048，AUROC +0.0005（95% CI [−0.0064,+0.0072]，BH q=1.0）；相对 HarNet10 的两项指标均明显较低。本轮只使用官方 384D 神经 acc-token pooling，不包含默认输出中的未编码 gravity；不能据此否定完整 Bio-PM HAR pipeline。
3. **未证实 Bio-PM pretrained 优于 random-frozen control。** BA 差 −0.0008，7/15 改善，CI [−0.0062,+0.0053]，dz −0.068；AUROC 差 −0.0024，8/15 改善，CI [−0.0109,+0.0062]，dz −0.135；两项 BH q=1.0。DD Recall 的正向点估计伴随 PD Recall 下降，相关 CI 均跨零，不能归因于稳定的预训练收益。
4. **HarNet/BioPM disease complementarity 尚未检验。** E4 的 Bio-PM 价值门槛失败，因此按条件未执行；未测量不等于证明不存在互补信息。
5. **Dual-prior 未训练。** Bio-PM 未稳定接近 HarNet10，也没有满足 E4 证据要求；不存在可报告的 dual-prior 增益。
6. **最佳 retained single-model 仍是 frozen WSSL-STR。** STR-01 + pretrained-frozen HarNet10 final1024 wrist residual，原 recipe 和 0.5 decision rule。Accuracy 0.7456、BA 0.7196、AUROC 0.7596、Macro-F1 0.7066、PD Recall 0.7823、DD Recall 0.6569。本轮没有保留新 single-model candidate。
7. **最高预注册 inference AUROC 为六模型等权 ensemble：0.7870。** Accuracy 0.7541、BA 0.7150、Macro-F1 0.7099、PD Recall 0.8090、DD Recall 0.6210。相对 frozen 三 seed ensemble（AUROC 0.7816 / BA 0.7200），AUROC +0.0054，14/15 改善，CI [+0.0028,+0.0074]，dz 1.127，E0 BH q=0.0427；BA 与 DD Recall 点估计更低。只作为 inference reference，不替代 single-model。
8. **当前不支持新机制或 dual-prior training。** 保留现有模型，停止本次 5s / Bio-PM acc-only integration 变体。External-prior 路线整体未被否定，后续只考虑有官方资产和明确依据的单一 prior 验证，并规划真正独立的验证数据。不得重开已停止的 HarNet adapter/gate/domain/layer 或 STR architecture/loss/sampling/augmentation 搜索；不会自动启动另一个 encoder。

**统计与实现边界。** 三个 seed 先在 split 内聚合，15 fixed splits 为主要比较单位；subjects、windows、activities、pairs、seed-runs 不作为独立样本。报告 paired difference、split improvement count、10000 次 bootstrap CI、dz 和 BH-FDR。重复且重叠的 development splits 与连续阶段选择限制确认性；没有新的独立 outer validation。

STR architecture、双腕/activity order、structured path、balanced CE、原 training recipe、train-only normalization 与 0.5 rule 保持不变。HarNet5 官方为 512D / 4.23M，HarNet10 为 1024D / 10.46M，不能隔离 duration 因果效应。Bio-PM 只保留纯神经 384D acc 分支，使用官方固定 30Hz、filter/tokenization 与 192-token cap；10920 contexts、0 empty，8.14% 触及 cap，未据此改 preprocessing。Zero-residual 与 loaded-WSSL 接口 logits consistency 最大差 0；smoke、全部 checkpoint、subject/label 对齐、train-only normalization、no-outer-loader 和 source SHA 审计均 PASS。h5py 仅装入本阶段 vendor/deps，正式代码和旧结果未修改。E3 的 any-positive-primary control 澄清在完整 BioPM residual 统计前记录，未放宽保留标准。E4/E5 是未执行，不是测量后的失败。

**Development-only conclusions：旧 FOE-01 继续冻结，未用于本阶段模型选择、调参或独立验证。**


## GitHub archive and ongoing version control (2026-10-01)

Code, reports and aggregate results are published to [pureDeep](https://github.com/chouytong/pureDeep). See [GIT_VERSION_CONTROL.md](GIT_VERSION_CONTROL.md) for repository mapping, exclusions and the required commit/push workflow after every code change. Data, weights, fitted arrays, per-subject predictions and credentials remain outside Git. Publication does not change any frozen model, result or development/outer evidence boundary.


GitHub publication target updated on 2026-10-02 to [chouytong/pureDeep](https://github.com/chouytong/pureDeep), at the owner's request. Existing code, reports and aggregate results are preserved with Git history.


## 2026-10-02 — WSSL window dynamics, item 01

Decision: **REUSE verified baseline**. No training or outer evaluation was run.

All 15 fixed development splits × seeds 42/43/44 passed: checkpoint/status hashes, explicit validation IDs and labels, class order, activity/wrist/channel order, train-only normalization, resolved balanced-CE weights, recipe and archived metric reproduction. Cache hash and dimensions passed. Class weights stored numerically in checkpoints match training-only counts; this is configuration resolution, not recipe drift.

Latest Phase-3B–E formal retained baseline is frozen HarNet10 WSSL-STR. Earlier V8/STR-only summaries are historical and do not define the current baseline. Original checkpoint/cache/prediction files were not changed.

| Metric | Seed-first mean | Split SD | SD of seed means | Mean within-split seed SD |
|---|---:|---:|---:|---:|
| accuracy | 0.745604 | 0.041218 | 0.008650 | 0.031205 |
| ba | 0.719644 | 0.034563 | 0.006714 | 0.021603 |
| auroc | 0.759553 | 0.043921 | 0.001812 | 0.021741 |
| macro_f1 | 0.706589 | 0.039561 | 0.004947 | 0.025605 |
| pd_recall | 0.782344 | 0.061379 | 0.025591 | 0.065659 |
| dd_recall | 0.656945 | 0.058007 | 0.035930 | 0.078542 |

Median selected epoch 10; median epochs executed 22. Mean selected-epoch train loss 0.291046; validation loss 0.679387. Classifier trainable parameters 143,172; frozen HarNet10 10,457,408. Threshold: DD probability >0.5, tie PD (original argmax). All metric deltas from archived WSSL baseline are zero to 1e-12.

Files: PROTOCOL.md, scripts/audit_baseline.py, analysis/baseline_lock.json, baseline_seed_split_metrics.csv, baseline_15split_seedfirst.csv and baseline_summary.csv. Public artifacts contain aggregate rows and hashes only; weights/features/predictions stay on server. New retention criteria are registered in PROTOCOL.md before candidate results.

Shape/mask/gradient/reload/one-batch tests for new branches belong to subsequent numbered items. This read-only baseline audit does not invent training tests. Fifteen overlapping splits are repeated development evidence, not 45 independent observations. Next: item 02, window preservation and numerical/logit consistency.

Full protocol/report: `artifacts/phase4_window_dynamics_20261002/PROTOCOL.md`, `ITEM01_REPORT.md`.
