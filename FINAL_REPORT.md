# PADS 纯深度模型探索阶段性收尾报告

日期：2026-09-16

## 1. 收尾结论

本轮纯深度模型搜索正式结束。最终冻结模型为 V8
`pads_pure_deep_normfree_moments_v8`，而不是开发期某个单 seed 数值最高的 V11、
V12 或 V13。V8 通过了预先设定的多种子和配对 fold 稳定性标准；冻结后的一次性
5×3 nested-CV outer test 给出 BA 0.6116、Macro-F1 0.6064、AUROC 0.6347。

V8 是当前可复现、复杂度可控的纯深度 baseline，但不是已经证明泛化很强的模型。
其相对 M0 的 outer-test 增益较小且置信区间跨 0。本轮停止继续迭代的原因是：多个
合理且彼此独立的方向均未产生稳定、可复制的实质提升；继续围绕同一批 390 人调整
结构，会增加对固定开发折的适配，而不是增加可信证据。

## 2. 任务和一致评估协议

任务是 PADS 数据集上的 subject-level PD-vs-DD 二分类，共 390 名受试者（PD 276，
DD 114），输入包含 11 个活动、左右腕、Acc/Gyro 六轴信号。全部实验沿用冻结的
subject-level 5 outer × 3 inner 划分，split SHA-256 为
`b3c52317cb12b73c66046bbd7c50c94a7e707a64a38d3ad4f0fd4bc0c6ba6b3e`。

开发期只使用 15 个 inner validation fold；所有 normalization、类别权重和 SSL
预训练均仅使用相应 fold 的训练受试者。模型冻结后，outer test 才被统一释放一次。
每个 outer fold 的决策阈值只由该 outer context 的 inner-OOF 预测选择，最终训练
轮数取三个 inner best epoch 的中位数。最终每名受试者恰好测试一次。

主指标为 balanced accuracy，同时报告 Accuracy、Macro precision/recall/F1、
AUROC、PD/DD recall、NLL 和 two-class Brier。开发均值与最终 outer pooled 结果
属于不同证据层级，报告中不将二者直接当作同一次测试比较。

## 3. 最终模型 V8

### 3.1 结构

V8 有 71,026 个可训练参数。输入为 `[B,11,2,6,T]`。每个活动的有效长度先被恢复，
再经共享腕侧编码器处理：

1. 紧凑 TCN 学习局部时序表征；
2. 对 TCN feature map 做 attention/mean/std pooling；
3. 并行的 normalization-free 可学习卷积组使用 1/15/63 三种核；
4. 只对该支路学习出的 feature map 计算 mean/std；
5. 两支路投影融合后进行左右腕晚期融合；
6. 通过 activity attention 得到 subject embedding；
7. 二分类头输出 PD/DD logits。

模型不接收人工预先定义的 RMS、峰值、频带功率或 handcrafted statistical feature。
训练使用 fold-train-only balanced cross-entropy、AdamW，FP32；每个 fold 的标准化
参数也只由训练受试者拟合。

### 3.2 选择依据

V8 的三个预定义开发 seed 结果如下：

| Seed | BA | AUROC |
|---:|---:|---:|
| 42 | 0.6682 | 0.6649 |
| 43 | 0.6596 | 0.6789 |
| 44 | 0.6440 | 0.6342 |
| 均值±样本 SD | 0.6572±0.0123 | 0.6593±0.0229 |

三种子 V8 还得到 Accuracy 0.7217±0.0127、Macro-F1 0.6565±0.0132、PD recall
0.8125±0.0241、DD recall 0.5020±0.0333、NLL 0.6335±0.0144、Brier
0.4287±0.0082。相对相同三种子的 balanced-v3，平均 BA +0.0144（11/15 fold
获胜）、Macro-F1 +0.0132（10/15）、AUROC +0.0085（8/15）、DD recall
+0.0300（10/15）。这是 V8 被选中的主要证据。

V11、V12、V13 的 seed-42 BA 数值略高于 V8 三种子均值，但这些是单种子候选，且
未通过预注册的提升幅度、fold 胜数、稳定性或校准门槛。用它们替换 V8 会构成根据
偶然开发结果追逐最高值。

## 4. 实验演化表

除明确标注的 V8 三种子行外，表中为固定 15 个 inner fold 的 seed-42 开发均值；
它们不是 outer-test 指标。

| Experiment | 核心改动 | BA | AUROC | 相比上一版本/参照 | 结论 |
|---|---|---:|---:|---|---|
| M0 | 固定频带 + MS-CNN + Hard Top-K MIL | 0.5897 | 0.6131 | 原始 baseline | 基线 |
| R1 | 增加全频神经 residual | 0.6124 | 0.6316 | BA +0.0227 | 已有改进参照 |
| PureDeep-v1 | TCN + 学习表征 attention/mean/std pooling | 0.6188 | 0.6382 | vs R1 BA +0.0064 | 未达保留幅度 |
| Pool-mean | 去掉学习表征 std pooling | 0.6125 | 0.6538 | vs v1 BA -0.0063 | 拒绝，指标不一致 |
| small-v2 | 特征维度与 activity attention 128→64 | 0.6292 | 0.6388 | vs R1 BA +0.0168，11/15 胜 | 保留 |
| balanced-v3 | 仅用 fold-train 标签计算 balanced CE | 0.6464 | 0.6519 | vs v2 BA +0.0172 | 保留 |
| activity-dropout-v4 | 训练期 activity dropout 0.15 | 0.6508 | 0.6492 | vs v3 BA +0.0043 | 拒绝，AUROC/DD recall 降 |
| split-sensor-v5 | Acc/Gyro 分离浅层 stem | 0.6494 | 0.6527 | vs v3 BA +0.0030 | 拒绝，仅 7/15 胜且校准差 |
| rotation-v6 | 腕内一致三维旋转增强 ±15° | 0.6505 | 0.6491 | vs v3 BA +0.0040 | 拒绝，不稳定 |
| time-spectrum-v7 | 可微全活动 rFFT/频谱编码支路 | 0.6376 | 0.6404 | vs v3 BA -0.0088 | 拒绝，整体回退 |
| V8（三种子） | 无归一化学习卷积矩支路 | 0.6572±0.0123 | 0.6593±0.0229 | vs 三种子 v3 BA +0.0144，11/15 胜 | **最终冻结** |
| moment-only-v9 | 删除 TCN，仅保留学习矩支路 | 0.6557 | 0.6591 | vs V8 seed42 BA -0.0125 | 拒绝，两支路互补 |
| weight-decay-v10 | weight decay 1e-4→1e-3 | 0.6624 | 0.6647 | vs V8 seed42 BA -0.0058 | 拒绝，校准/方差变差 |
| mean-activity-v11 | activity attention 改 masked mean | 0.6727 | 0.6869 | vs V8 seed42 BA +0.0045 | 仅作消融，未达门槛 |
| masked-SSL-v12 | fold-train-only masked reconstruction | 0.6736 | 0.6719 | vs V8 seed42 BA +0.0055，7/15 胜 | 拒绝，未通过复制门槛 |
| relative-energy-v13 | 可学习相对能量滤波器组 | 0.6684 | 0.6677 | vs V8 seed42 BA +0.0002 | 拒绝，F1/校准回退 |

## 5. 各方向的实验结论

### 已由实验确认

- 用紧凑 TCN 与学习表征统计池化替代固定频带/Hard Top-K 后，纯深度路线可以超过
  M0 的 inner-development BA，但初始 v1 的增益不足以作为稳定新模型。
- 降低容量到 small-v2、再加入 train-only balanced CE，是两次清晰且可复现的
  改进；balanced CE 主要通过提高 DD recall 换取部分 PD recall/Accuracy。
- V8 的 normalization-free learned-moment 支路是本轮最重要的有效结构改动。它在
  三个预定义 seed 上相对对应 v3 平均 BA +0.0144，并在 11/15 配对 fold 获胜。
- moment-only-v9 低于 V8，证明局部 TCN 与全局学习矩信息在当前实现中互补。
- activity dropout、传感器分 stem、旋转增强和全频谱支路均没有稳定通过保留门槛。
- masked SSL 的重建损失确实下降，但下游 BA 只增加 0.0055 且只有 7/15 fold 获胜；
  “预训练任务学会了”不等于“分类泛化稳定改善”。
- V8 最终 outer BA 只有 0.6116，显著低于其 seed-42 inner 均值 0.6682，说明固定
  inner folds 上的开发结果高估了未见 outer subjects 的表现。

### 有证据支持但仍属推测

- H1 时间域统计组强于频域组，以及 V8 学习矩支路的成功，支持“深度编码器先前
  未充分保留跨受试者幅度/分布信息”的解释；但现有实验不能证明这是唯一原因。
- V8 和 H1 共享较多困难受试者，暗示上限可能受标签异质性、DD 亚型混合或信号
  本身判别信息不足影响；需要新队列或更细临床标签验证。

## 6. 最终 outer-test 结果及 baseline 比较

| 指标 | V8 pooled outer | M0 pooled outer | V8−M0 |
|---|---:|---:|---:|
| Accuracy | 0.6615 | 0.7051 | -0.0436 |
| Balanced accuracy | 0.6116 | 0.5934 | +0.0181 |
| Macro precision | 0.6038 | 0.6244 | -0.0206 |
| Macro recall | 0.6116 | 0.5934 | +0.0181 |
| Macro-F1 | 0.6064 | 0.5985 | +0.0079 |
| AUROC | 0.6347 | 0.6016 | +0.0330 |
| PD recall | 0.7319 | 0.8623 | -0.1304 |
| DD recall | 0.4912 | 0.3246 | +0.1667 |
| NLL | 0.6756 | 0.6193 | +0.0563（更差） |
| two-class Brier | 0.4630 | 0.4086 | +0.0544（更差） |

V8 的混淆矩阵为 `[[202,74],[58,56]]`。五个 outer fold 的 BA 为 0.6071、
0.6538、0.5953、0.6545、0.5455，均值 0.6112、总体 SD 0.0407。默认 0.5
阈值的 pooled BA 为 0.6134，说明 inner-OOF 阈值选择没有改善最终 pooled BA。

V8−M0 的配对分层 bootstrap 95% CI：BA `[-0.0410,0.0788]`、Macro-F1
`[-0.0538,0.0714]`、AUROC `[-0.0362,0.1025]`，均跨 0。因此可以确认 V8
提高了本次样本中的 DD recall 和点估计 BA/AUROC，但不能确认其总体性能稳定优于 M0。

H1 的 BA 0.6918、AUROC 0.7515 是 inner-development 结果，并未经过同一次最终
outer release，不能伪装成协议完全相同的最终比较。

## 7. 为什么停止继续迭代

停止并非因为已经达到高性能，而是因为研究收益/过拟合风险已经逆转：

1. V4–V7 连续四类合理结构或增强均未通过门槛；
2. V9–V13 的消融、正则化、融合、SSL 和相对能量方向也没有稳定实质增益；
3. 单种子更高的候选没有多种子复制，继续挑选会产生 winner's curse；
4. 最终 outer 结果表明 inner-development 增益不能充分迁移；
5. 继续使用同一 390 人和同一 15 个开发 fold 调参，会弱化下一阶段泛化结论。

因此，内部模型搜索到此冻结。以后若重启研究，应视为新阶段，并引入新数据、预先
登记的外部验证或新的临床目标，而不是回看 outer test 后继续改 V8。

## 8. 归档与验证

归档路径：`/home/zyt/deep_final`。归档包含源码、配置、五折 checkpoint、各 fold
normalization、冻结 split、outer 与开发预测/指标、训练日志、provenance、环境记录
和 SHA-256 清单。原项目 `/home/zyt/MFAM` 及原实验产物未修改、未删除。

其中 `frozen_project/` 是与 checkpoint 记录的 source-tree SHA-256 完全一致的加载
快照；冻结后新增的配对分析脚本单独保存在 `source/`，没有混入严格身份快照。使用
outer-0 checkpoint 对既有受试者 `002` 完成了一次 CPU 只读推理，严格 provenance、
数据清单、normalization、权重加载和前向传播均通过。

最终冻结 manifest SHA-256：
`1d91164f30cf82aeabea58d413d5561edb6ef61a079a939106c7af9ff5e9a6cc`；
pooled prediction SHA-256：
`f3c8d7e8dcff5197cd7cc8aecc7ec1a77453d781b0b357291e2d0941d8a948c2`。
最终运行前后的测试记录为 105 passed；完整 final artifact hash 校验通过。

## 9. 当前限制与后续 backbone 研究问题

- 当前任务是 PD-vs-DD 分类，不是论文主目标所需的 UPDRS/手部运动严重程度回归。
- 全部 390 人仍来自 PADS，缺少独立中心或新队列外部验证。
- DD 只有 114 人且亚型高度不均衡，亚型泛化结论不足。
- 外层 BA 的 95% CI 较宽且 fold 波动明显；V8 对 M0 的增益未达到统计稳定。
- NLL/Brier 比 M0 差，当前概率校准不足；inner 阈值选择也未改善最终 BA。

下一阶段若将 V8 作为跨个体表征 backbone，最值得研究的不是继续加模块，而是：
独立受试者/中心外部验证；更多无标签独立数据上的自监督预训练；DD 亚型或
临床表型感知评价；校准与不确定性；以及获得 MDS-UPDRS item/total score 后转向
subject-independent 连续回归，并报告 MAE、RMSE、ICC 与 Bland–Altman。

## 10. 证据定位

- 最终结果：`artifacts/final_nested_cv/nested_cv_summary.json`
- 全体预测：`artifacts/final_nested_cv/outer_test_predictions_all_folds.csv`
- 选择协议：`artifacts/final_nested_cv/protocol.json`
- M0 配对比较：`artifacts/final_analysis/final_paired_analysis.json`
- 三种子稳定性：`artifacts/development/pure_deep_normfree_moments_v8_seed_stability_20260916.json`
- 完整开发日志：`reports/EXPERIMENT_LOG.md`
- checkpoint 来源与 hash：各 `models/outer_k/provenance.json`、
  `final.pt.provenance.json` 及根目录 `SHA256SUMS`
