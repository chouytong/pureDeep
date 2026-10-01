# PADS 双腕分类实现说明

## 数据流

```text
patient JSON -> PD/DD 标签
observation JSON 的同一 session -> 左右腕 pair
pair manifest -> subject-level train/validation/test
训练受试者完整记录 -> [2,C,1] 或 [A,2,C,1] mean/std
paired dataset -> x=[B,2,C,T], wrist_mask=[B,2]
共享腕侧编码器 -> 每腕表示 [B,2,H]
左右/均值/绝对差异/mask 融合 -> PD/DD
```

`left`、`right` 和 `bilateral` 模式只改变 `wrist_mask`，不改变 manifest 或受试者
划分。标准化参数按腕侧和通道分别计算，但只能来自训练受试者。
v2 subject 配置进一步按 activity、腕侧和通道分别统计，禁止 per-subject
normalization。

## 模型

MFAM 腕侧编码器为频率分解、MultiScaleChannelAttentionEncoder 和
AttentionMIL。左右腕调用同一个编码器对象。MIL 放在融合前，使模型不依赖两块
手表逐点同步，并保持现有单腕时序编码含义。

融合向量为：

```text
[left, right, masked_mean, abs(left-right), wrist_mask]
```

单腕模式的无效腕不会进入编码器；差异项只在双腕都有效时启用。CNN 基线遵循相同
共享编码和融合接口。

## 泄漏控制

- manifest 一行对应完整 pair，不能把腕侧拆成样本。
- 先按 subject 划分，再裁剪或切窗。
- 三个 split 的 subject 集合强制两两不相交。
- mean/std 只遍历 train records，validation/test 复用 checkpoint 数值。
- 三种腕侧对照共用冻结 split 文件。
- checkpoint 校验腕侧顺序、腕侧模式、传感器模式和通道名称。

## 时间轴

PADS 左右腕时间列从 0 开始且标称 100 Hz，但独立时钟存在漂移。manifest 生成时
检查时间单调性、有效采样率和长度，并保存最大时间偏差。当前模型只做腕侧晚期
融合，不声称逐采样点同步。

## 证据边界

单元测试、单批次前向反向和小规模训练只验证工程流程。只有按冻结协议完成正式
训练、多随机种子评估并综合 macro-F1、balanced accuracy、AUROC 和混淆矩阵后，
才能讨论分类性能。

## Subject-level 多活动扩展

多活动配置按固定 activity 顺序把一个受试者组织为 `[A,2,C,T]`。每个有效活动
使用同一套 WristMFAMEncoder、频率分解、MS-CAE、Attention-MIL 和 bilateral
fusion，得到 514 维 activity feature。可学习 activity embedding 和 masked
Activity Attention 将其聚合为一个 subject embedding；缺失 activity 不进入 MFAM，
attention 权重严格为零。

受试者划分发生在 activity 聚合之前，归一化只遍历训练受试者的全部有效活动。
原始数据副本与 processed NPY 分开保存，网络训练不再重复读取和解析原 TXT。

## v2 完整时长与变长输入

`pads_multi_activity_v2.yaml` 不再配置固定 `sequence_length`。预处理后，CrossArms
等约 10 秒活动保留 976 点；Relaxed、RelaxedTask、Entrainment 保留 2000 点。
Dataset 只为组成 mini-batch 在时间维右侧补零，同时返回 `activity_lengths`。
SubjectMFAM 按真实长度分组并在进入 FFT、Conv1D 和 Attention-MIL 前切除 padding，
因此补零不参与频率分解或 MIL instance 聚合。不同长度 activity 仍分别得到固定维度
embedding，随后才进入原有 subject-level activity fusion。

v2 预处理仅执行：Acc XYZ 的 L1 trend removal（lambda=50）、Gyro 原样保留、删除
前 48 点。没有增加额外滤波或增强。Latest-v2 cleanup 后，active 配置只保留当前
subject-level 主线；left/right/bilateral 与 acc/gyro/acc_gyro 参数接口继续可用，旧
v1、固定 1024 点、single-activity 和 CNN1D 产物位于项目外隔离区。
