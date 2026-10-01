# MFAM v2：PADS subject-level 多活动 PD-vs-DD 分类

当前项目只保留 v2 主线：使用 PADS 的 11 个 activity、左右腕 Acc+Gyro，输出一个 subject-level PD/DD 预测。Healthy、严重程度回归、序数分级、旧 v1 固定长度流程和 CNN1D baseline 不属于当前活动代码。

## 当前数据与输入

- 390 subjects：PD 276、DD 114；
- activity 顺序：CrossArms、DrinkGlas、Entrainment、HoldWeight、LiftHold、PointFinger、Relaxed、RelaxedTask、StretchHold、TouchIndex、TouchNose；
- wrist 顺序：left、right；
- channel 顺序：AccX、AccY、AccZ、GyroX、GyroY、GyroZ；
- Entrainment、Relaxed、RelaxedTask 为 2000 点，其余为 976 点；
- batch 输入 `[B,11,2,6,Tmax]`，并携带 `activity_mask` 和 `activity_lengths`。

预处理只执行 Acc XYZ 的 L1 trend removal（lambda=50）、Gyro 原样保留、删除前 48 点。训练只读取：

`data/processed/pads_multi_activity/v2_l1_full_length`

## 环境

```bash
cd /home/zyt/MFAM
/home/zyt/envs/mfam/bin/python -V
```

## 数据准备与审计

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
/home/zyt/envs/mfam/bin/python scripts/prepare_multi_activity_data_v2.py \
  --source-root data/raw/pads \
  --processed-output data/processed/pads_multi_activity/v2_l1_full_length

/home/zyt/envs/mfam/bin/python scripts/check_pads_classification.py \
  --config configs/pads_multi_activity_v2.yaml
```

已有数据目录不可覆盖。重新预处理时必须指定新的独立输出路径。

## 训练

```bash
/home/zyt/envs/mfam/bin/python train.py \
  --config configs/pads_multi_activity_v2.yaml
```

当前 canonical 配置为最大 50 epochs、batch size 8、AdamW、Cosine scheduler，并按 validation macro-F1 保存 `best.pt`。训练阶段默认不自动运行 test；test 与多 seed threshold 分析应在训练全部完成后单独执行。

## 评估与推理

```bash
/home/zyt/envs/mfam/bin/python evaluate.py \
  --config RUN_DIR/config.yaml \
  --checkpoint RUN_DIR/checkpoints/best.pt \
  --split validation

/home/zyt/envs/mfam/bin/python inference.py \
  --config RUN_DIR/config.yaml \
  --checkpoint RUN_DIR/checkpoints/best.pt \
  --subject-id 006
```

`evaluate.py` 和 `inference.py` 是单 checkpoint、softmax argmax 接口。正式三 seed 概率融合及 validation 阈值选择由 `scripts/analyze_probability_ensemble.py` 执行，不能在 test 上重新选择阈值。

## 模型

```text
[B,A,2,6,T]
  -> 每个有效 activity 按真实 T 切片
  -> 左右腕共享 WristMFAMEncoder
  -> 三频带分解
  -> multi-scale Conv1D
  -> Channel Attention
  -> Attention-MIL + Top-K
  -> bilateral fusion（514 维）
  -> activity embedding + masked Activity Attention
  -> subject embedding（514 维）
  -> Linear(514,2)
```

当前参数量为 121,906。详细结构和证据边界见 `docs/current_project_technical_overview_20260821.md`。

## 验证

```bash
PYTHONDONTWRITEBYTECODE=1 bash scripts/static_check.sh
PYTHONDONTWRITEBYTECODE=1 /home/zyt/envs/mfam/bin/python -m pytest -q -p no:cacheprovider
```

小规模 smoke test 只证明流程可运行。结果解释必须联合报告 macro-F1、balanced accuracy、AUROC、per-class recall 和 confusion matrix。

## 当前冻结研究进展索引（2026-09-26）

当前正式主模型、固定 inner-development 协议及 DSG-01/RGD-01 的完整追加记录，以 `/home/zyt/deep_final/README.md` 和相应正式报告为准。本仓库顶部的旧模型结构说明属于历史阶段。RGD-01 已完成 H1–STR AUROC 排序差、hard-subject burden、train-only score fusion 与 frozen-representation residual 诊断；主要结论是 H1 的净 AUROC 优势集中于 STR primary stable-error subjects 参与的 pairs，两个分数存在互补，但尚不足以启动新的 deep model。报告：`/home/zyt/deep_final/artifacts/rgd01_h1_str_residual_ranking_20260926/RGD01_REPORT.md`。本轮没有使用 outer information，没有修改冻结 STR-01/checkpoint/正式预测。
