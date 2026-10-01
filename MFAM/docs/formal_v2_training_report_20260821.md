# PADS PD-vs-DD v2 全时长多活动正式训练报告

日期：2026-08-21  
服务器项目：`/home/zyt/MFAM`  
实验范围：验证 v2 数据预处理与 variable-length activity 输入；不修改 MFAM、Activity Attention、loss、frequency bands 或其他主体网络结构。

> Latest-v2 cleanup 说明：v2 三 seed 与正式 ensemble 仍保留在 active 项目；本文引用的 v1、调参、single-activity 对照产物已移至 `/home/zyt/MFAM_cleanup_archive_20260821-224905`。

## 1. 结论摘要

本轮完成 seed 42、43、44 三次正式训练。训练阶段关闭测试集评估；三个最佳 checkpoint 和三模型集成阈值均只由冻结验证集确定，随后才执行一次最终测试。

最终三种子概率平均集成在测试集上得到：

- Accuracy：`0.6709`
- Balanced accuracy：`0.6013`
- Macro-F1：`0.6013`
- AUROC：`0.6203`
- DD recall：`0.4348`
- NLL：`0.5888`
- Brier score：`0.1993`
- 混淆矩阵（行=真实 PD/DD，列=预测 PD/DD）：`[[43, 13], [13, 10]]`

结果证明 v2 全时长、变长 activity 数据能稳定完成端到端正式训练和推理，但尚不能证明获得可靠诊断性能：测试集只有 79 人，95% CI 较宽，且三种子验证表现差异明显。

## 2. 固定实验协议

- 数据版本：`/home/zyt/MFAM/data/processed/pads_multi_activity/v2_l1_full_length`
- 11 个 activity 全部保留。
- `Relaxed`、`RelaxedTask`、`Entrainment`：2,000 点；其余 activity：976 点。
- 预处理：每段删除开始 48 点；Acc 三轴 L1 trend filtering，lambda=50；Gyroscope 保持原始信号；不增加额外滤波。
- 输入：`[B, A, 2, 6, T_max]`，由 `activity_lengths` 标记每个 activity 的真实长度；各 activity 独立经过 MFAM 后在 embedding 层融合。
- 标准化：仅使用训练 subjects，按 `activity × wrist × channel` 拟合，统计量形状 `[11, 2, 6, 1]`。
- subject-level 冻结划分：train 264（PD 187/DD 77），validation 47（PD 33/DD 14），test 79（PD 56/DD 23）。
- split SHA256：`62fe2451c9b4071ec7b563e064142e4815d8fd7fee5a130bc19aa836cfdeb207`。
- v2 audit SHA256：`bb0b012a46944e32e093914172565c31e31d583286d1be5e559b05f35c44738c`。
- 网络参数量：121,906，全部可训练；未改变主体架构。
- 用户指定训练参数：最大 50 epochs，train/eval batch size 8。
- 其余关键参数：AdamW，lr `2e-4`，weight decay `1e-4`，无 label smoothing、无 class weight，activity fusion dropout `0.1`，classifier dropout `0.2`。
- early stopping：验证 macro-F1，patience 12。
- 最终预测：三个 seed 的 DD 概率算术平均。
- DD 阈值：`0.2778912236293157`，仅在三种子验证集平均概率上按 macro-F1 选择；并以 balanced accuracy、距 0.5 的距离依次打破并列。

配置 SHA256：`dca6c1adf6441ab33b91cf13651ea5711dd97e3c257166510485006fe4e46457`。

## 3. 训练与验证结果

| Seed | 最佳 epoch | 最佳验证 Macro-F1 | 验证 AUROC | 实际停止 epoch | 现象 |
|---:|---:|---:|---:|---:|---|
| 42 | 2 | 0.4125 | 0.4827 | 14 | 0.5 阈值下全部预测为 PD |
| 43 | 13 | 0.4503 | 0.5606 | 25 | DD recall 0.0714 |
| 44 | 23 | 0.5712 | 0.4762 | 35 | DD recall 0.2143 |

三模型平均概率在验证集上的阈值选择结果：

| 阈值 | Accuracy | Balanced Acc. | Macro-F1 | AUROC | Confusion matrix |
|---:|---:|---:|---:|---:|---|
| 固定候选 0.5 | 0.6596 | 0.4697 | 0.3974 | 0.5260 | `[[31,2],[14,0]]` |
| 验证选定 0.2779 | 0.7021 | 0.6028 | 0.6083 | 0.5260 | `[[28,5],[9,5]]` |

训练 loss 持续下降，而最佳验证指标通常较早出现，尤其 seed 42/43；seed 44 后期训练 macro-F1 已达 0.94 左右，但验证明显较低。当前主要限制仍是小样本过拟合和类别不平衡，而不是训练轮数不足。

## 4. 最终测试结果

所有行统一使用验证集预先确定的 DD 阈值 `0.2778912236293157`：

| 模型 | Accuracy | Balanced Acc. | Macro-F1 | DD Recall | AUROC | NLL | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed 42 | 0.7089 | 0.5512 | 0.5385 | 0.1739 | 0.5866 | 0.5960 | 0.2033 |
| seed 43 | 0.7342 | 0.6588 | 0.6645 | 0.4783 | 0.6467 | 0.6170 | 0.1956 |
| seed 44 | 0.6709 | 0.5885 | 0.5905 | 0.3913 | 0.5644 | 0.7806 | 0.2447 |
| 三种子概率融合 | **0.6709** | **0.6013** | **0.6013** | **0.4348** | **0.6203** | **0.5888** | **0.1993** |

融合模型分层 subject bootstrap 95% CI（5,000 次，模型和阈值固定）：

| 指标 | 点估计 | 95% CI |
|---|---:|---:|
| Accuracy | 0.6709 | [0.5696, 0.7722] |
| Balanced accuracy | 0.6013 | [0.4876, 0.7240] |
| Macro-F1 | 0.6013 | [0.4840, 0.7190] |
| AUROC | 0.6203 | [0.4720, 0.7655] |
| Brier score | 0.1993 | [0.1656, 0.2320] |

默认 0.5 阈值的测试 sensitivity analysis 为 Accuracy 0.7089、balanced accuracy 0.5512、macro-F1 0.5385，低于预先固定阈值的 balanced accuracy 和 macro-F1。该敏感性分析没有参与阈值选择。

## 5. Activity Attention 诊断

测试集三种子平均 attention 排名前六：HoldWeight `0.3426`、DrinkGlas `0.1534`、CrossArms `0.1140`、Entrainment `0.1068`、StretchHold `0.1022`、LiftHold `0.0729`。

单种子的归一化 attention entropy 分别为 0.9050、0.6861、0.5100。与旧 v1 中两个种子接近单 activity 塌缩相比，v2 的 attention 覆盖更广，但仍存在明显种子差异，且 HoldWeight 权重长期占优。Attention 权重只能用于模型行为诊断，不能直接解释为某 activity 的临床重要性；需要消融实验验证。

保留的 `PointFinger`、`TouchIndex`、`LiftHold` 在三种子平均权重中分别约为 0.0319、0.0229、0.0729。低 attention 不等于无价值，本轮不据此删除 activity。

## 6. 与旧 v1 结果的边界化比较

旧 v1（统一 1,024 点、batch size 2、最大 80 epochs）三种子集成测试结果为 Accuracy 0.608、balanced accuracy 0.569、macro-F1 0.560、AUROC 0.651、DD recall 0.478、NLL 0.605、Brier 0.205。

本轮 v2 点估计相对变化：Accuracy `+0.063`、balanced accuracy `+0.032`、macro-F1 `+0.041`、NLL `-0.016`、Brier `-0.006`，但 AUROC `-0.031`、DD recall `-0.043`。由于本轮同时将 batch size 从 2 改为 8、最大 epochs 从 80 改为 50，这不是严格的单变量预处理对照；加之置信区间宽，不能把差异归因于 v2 预处理。

## 7. 建议

### 优先级 1：先做严格对照，不改主体网络

1. 在 v1 与 v2 上使用完全相同的 batch size、epoch、seed、split 和阈值选择协议，形成真正 paired preprocessing comparison。
2. 做单 activity baseline 和 11 个 leave-one-activity-out 消融，特别关注 HoldWeight、DrinkGlas、CrossArms，同时保留 LiftHold、PointFinger、TouchIndex直到消融完成。
3. 增加简单 late-fusion baseline，例如各 activity logits/probability 平均，判断 learned attention 是否真正优于无参数融合。
4. 用 repeated stratified subject-level CV 或 nested CV 验证稳定性；每个 outer fold 内独立拟合 normalization 和阈值。

### 优先级 2：数据预处理验证

1. 对审计中 4 个大时间戳间隙和 2,095 条 robust outlier flags 做按 activity/wrist/channel 的人工或统计分层复核；当前只是警告，未删除数据。
2. 以消融方式验证 L1 trend filtering，而不是默认认为它必然改善分类；至少比较 raw Acc、L1 detrended Acc、Acc dynamic+trend 双分支。
3. 保持所有处理版本化，训练始终直接读取固定预处理结果；任何增强只在 train loader 中生效并记录随机种子。

### 后续若允许调整训练策略

可优先测试更强正则化、activity dropout、balanced sampler/focal loss、temperature calibration；但应与本轮结果分开，避免同时改变预处理和主体训练方法。模型规模不是当前首要问题。

## 8. 可复现产物

训练目录：

- `/home/zyt/MFAM/outputs/pads_classification/formal_v2_full_length_seed42_20260821`
- `/home/zyt/MFAM/outputs/pads_classification/formal_v2_full_length_seed43_20260821`
- `/home/zyt/MFAM/outputs/pads_classification/formal_v2_full_length_seed44_20260821`

集成目录：

- `/home/zyt/MFAM/outputs/pads_classification/formal_v2_full_length_ensemble_20260821/validation_threshold`
- `/home/zyt/MFAM/outputs/pads_classification/formal_v2_full_length_ensemble_20260821/test_final`

最佳 checkpoint SHA256：

- seed 42：`99c1b7d08c49f513576d1ca41aec6bf28baf42bd2c7170e405b988dc06f47dc0`
- seed 43：`0b223cdc9f477d128ef26f72decab8738a4379e0be994edc4fa551726f57d82c`
- seed 44：`8f3a9dc3ad1ac4574d4011681829dac08782d47c4fd218e84eb2e69b1c6e5fdd`

环境：Python 3.10.16，PyTorch 2.8.0+cu128，CUDA 12.8，NVIDIA GeForce RTX 5090 D v2。正式训练单次峰值 GPU memory 约 647 MiB。项目不是 Git repository，因此无法记录 commit hash。

验证：`PYTHONDONTWRITEBYTECODE=1 python -m pytest -q`，结果 `42 passed in 2.13s`。直接调用 `pytest` 时因入口脚本未把项目根目录加入 `sys.path` 而收集失败；改用 Python 模块入口后全部通过。
