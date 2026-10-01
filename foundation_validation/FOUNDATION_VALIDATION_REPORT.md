# V8 基础架构验证阶段报告

日期：2026-09-17  
范围：只使用固定 5×3 inner-development folds；seed 42/43/44；未访问 outer-test。

## 1. 目的与模型选择规则

本轮从 V8（71,026 参数）出发，依次验证长程时序建模、活动聚合稳定性和归一化。
每个实验只改变一个主要因素，保持受试者划分、训练流程、class-balanced loss、
early stopping、OOF threshold 与指标实现一致。模型选择以三个 seed 的均值和离散度
为主，并把三个 seed 在同一 fold 的指标先平均后与 V8 配对；45 次训练不被当作 45
个相互独立的样本。

预登记保留门槛：平均 BA 至少提高 0.01；或 BA 与 V8 相差不超过 0.005 且方差、
AUROC、Macro-F1、NLL/Brier 整体改善；同时 BA 至少在 9/15 个配对 folds 获胜。

## 2. 实验演化表

| Experiment | 核心改动 | 参数量 | BA（mean±SD） | AUROC | Macro-F1 | 相比 V8 BA | BA fold 胜数 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| V8 | `[1,2,4]` TCN + learned moments + activity attention + BatchNorm | 71,026 | 0.6572±0.0123 | 0.6593±0.0229 | 0.6565±0.0132 | — | — | 起点 |
| LR-01 | dilation 改为 `[1,4,16,64,128]` | 80,626 | 0.6406±0.0130 | 0.6383±0.0164 | 0.6390±0.0117 | -0.0167 | 3/15 | 拒绝 |
| AGG-01 | Activity Attention 改为 Masked Mean | 51,031 | 0.6590±0.0125 | 0.6719±0.0134 | 0.6502±0.0129 | +0.0018 | 10/15 | 消融保留，不替换 |
| AGG-02 | Mean + gated Attention Residual | 71,027 | 0.6562±0.0060 | 0.6529±0.0026 | 0.6544±0.0053 | -0.0010 | 7/15 | 拒绝 |
| NR-01 | Temporal BatchNorm 改为 GroupNorm(8) | 71,026 | **0.6707±0.0088** | **0.6800±0.0103** | **0.6631±0.0137** | **+0.0134** | **10/15** | **保留并冻结** |

## 3. 各方向的假设、结果与结论

### 3.1 LR-01：Long-range temporal modeling

- 假设：V8 约 0.99 秒的局部感受野限制了对 10–20 秒完整活动动态的学习。
- 单一改动：dilation `[1,2,4]` → `[1,4,16,64,128]`，理论感受野约 25.7 秒。
- 结果：BA -0.0167、AUROC -0.0210、Macro-F1 -0.0175、DD recall -0.0216；
  BA 仅 3/15 folds 获胜。
- 已确认结论：直接通过更深、更稀疏 dilation 覆盖整段活动会降低当前协议下的表现。
- 推测：额外容量和稀疏采样可能放大过拟合或削弱局部运动模式，但本实验不能区分
  两种机制。

### 3.2 AGG-01：Masked Mean

- 假设：无参数平均可能比小样本下的 activity attention 更稳定。
- 单一改动：只把 activity attention 改为 masked mean。
- 结果：BA +0.0018、AUROC +0.0125、DD recall +0.0501；但 Macro-F1 -0.0063、
  PD recall -0.0466、Accuracy -0.0185，BA seed SD 未下降。
- 已确认结论：均值聚合改变了类别召回取舍并提高 AUROC，但不能稳定替换 V8。
- 推测：所有活动等权有利于保留 DD 的分散证据，同时稀释对 PD 更有辨识力的活动。

### 3.3 AGG-02：Mean + Attention Residual

- 假设：以 mean 为锚点、门控 attention 残差可兼顾稳定性和活动选择能力。
- 单一改动：`mean + sigmoid(g) × (attention - mean)`，初始权重 0.25。
- 结果：BA -0.0010（7/15 胜）、AUROC -0.0064、Macro-F1 -0.0021；BA seed SD
  降至 0.0060，NLL 改善 0.0117。
- 已确认结论：种子间均值更平滑不等于判别性能稳定提高；该方案不满足保留门槛。
- 推测：单一全局门控不足以针对不同受试者或活动学习可靠性。

### 3.4 NR-01：Normalization audit

- 假设：batch size 8 且 encoder 跨活动、腕侧、长度共享时，BatchNorm 的 batch 和
  running statistics 会引入受试者组成敏感性；GroupNorm 可降低这种依赖。
- 单一改动：stem 和三个 separable residual blocks 中共 7 个 BatchNorm1d 替换为
  GroupNorm(8)。参数量仍为 71,026，其他结构和训练协议不变。
- 三 seed 结果：

| Seed | Accuracy | BA | Macro-Precision | Macro-F1 | AUROC | PD recall | DD recall | NLL | Brier |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.7315 | 0.6778 | 0.6892 | 0.6751 | 0.6897 | 0.8070 | 0.5486 | 0.6233 | 0.4166 |
| 43 | 0.6964 | 0.6609 | 0.6581 | 0.6482 | 0.6691 | 0.7467 | 0.5751 | 0.6796 | 0.4501 |
| 44 | 0.7182 | 0.6732 | 0.6766 | 0.6659 | 0.6811 | 0.7810 | 0.5655 | 0.6543 | 0.4258 |
| Mean±SD | 0.7154±0.0177 | **0.6707±0.0088** | 0.6746±0.0157 | **0.6631±0.0137** | **0.6800±0.0103** | 0.7782±0.0303 | 0.5631±0.0134 | 0.6524±0.0282 | 0.4308±0.0173 |

- 相对 V8：BA +0.0134（10/15 胜）、AUROC +0.0206（8/15）、Macro-F1 +0.0066
  （9/15）、DD recall +0.0611（10 胜、1 平），BA seed SD 0.0123 → 0.0088。
  Accuracy -0.0063、PD recall -0.0343、NLL +0.0188、Brier +0.0021。
- 已确认结论：NR-01 满足预登记的 BA 幅度与 fold 胜数，并获得 AUROC、Macro-F1、
  DD recall 的同向改善，因此是本轮唯一应保留的架构改动。
- 推测：收益可能来自消除 batch statistics，而非 GroupNorm 特定分组本身；没有继续
  做组数或其他 normalization 网格，因此不能声称 GroupNorm(8) 是全局最优。

## 4. 最终模型选择

冻结对象为 **NR-01 / V8-GroupNorm development backbone**。选择依据不是单个 seed、
单个 fold 或偶然最高值，而是：三 seed 平均 BA +0.0134；10/15 配对 folds 获胜；
AUROC、Macro-F1、DD recall 同向改善；BA seed 方差下降；参数量不增加。

该结论严格限定为开发协议下的模型选择。现有 outer-test 已被查看，本轮没有读取或
运行 outer-test，不能以本报告推断 outer-test 或外部泛化必然改善。

## 5. 为什么停止基础架构迭代

用户预定的三个方向已经逐项完成。长程 TCN 和两个聚合替代均未产生一致收益；
GroupNorm 达到保留门槛。继续搜索 dilation、聚合门控或 normalization 组数会增加
对同一 390 人开发数据的适配风险，且不能解决跨个体或跨中心证据缺口。
因此停止基础网络搜索，冻结 NR-01，下一阶段应转向跨个体泛化、不变表征和校准。

## 6. 限制与下一步

1. NR-01 的 NLL/Brier 和 PD recall 相比 V8 回退，不能称为所有指标全面改进。
2. 结果仍来自同一 PADS 队列，没有独立外部验证。
3. DD 亚型样本少，类别召回改善未证明在各亚型上一致。
4. 下一阶段优先研究 subject/domain-invariant representation、跨中心评估、
   calibration 与临床代价驱动阈值；不再继续基础模块堆叠。

## 7. 产物位置

- 工作副本：`/home/zyt/deep_final/foundation_validation`
- NR-01 三 seed 完整运行：
  - `outputs/foundation_validation/groupnorm_nr01_seed42_20260917`
  - `outputs/foundation_validation/groupnorm_nr01_seed43_20260917`
  - `outputs/foundation_validation/groupnorm_nr01_seed44_20260917`
- 汇总：`outputs/foundation_validation/groupnorm_nr01_three_seed_analysis.json`
- 冻结索引：`/home/zyt/deep_final/artifacts/foundation_validation/nr01_groupnorm_backbone`

## 8. 只读验证

- 纯深度相关测试：11/11 passed。
- 构建检查：71,026 参数，7 个 GroupNorm，0 个 BatchNorm。
- checkpoint smoke test：使用 seed 42 / outer 1 / inner 0 的 `best.pt`，按 checkpoint
  实际 `model_state` 格式执行 `strict=True` 加载；missing/unexpected keys 均为 0。
- 冻结索引完整性：`SHA256SUMS` 全部校验通过；`BEST_CHECKPOINTS.sha256` 记录三 seed
  共 45 个开发折最佳 checkpoint 的路径与哈希，checkpoint 本体保留在完整运行目录。
