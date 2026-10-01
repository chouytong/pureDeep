# MFAM 当前项目技术说明（服务器真实代码与运行产物审计版）

审计日期：2026-08-21  
服务器项目：`/home/zyt/MFAM`  
当前主配置：`/home/zyt/MFAM/configs/pads_multi_activity_v2.yaml`  
当前主任务：PADS、subject-level、11 activity、双腕 Acc+Gyro、PD-vs-DD 二分类  
清理状态：2026-08-21 latest-v2 cleanup 后；旧 v1、single-activity 与调参产物已移至项目外可恢复隔离区。  

> 本文按“服务器当前源码 → 正式运行目录内保存的 config/split/checkpoint/predictions → 文档”排序取证。代码、配置和文档不一致时，以实际执行代码和正式运行产物为准。本文不是依据 README 或历史设计反推的架构说明。

## 1. 项目目标与任务边界

### 1.1 当前实际解决的问题

输入同一受试者在 11 种规定 activity 下的左右腕加速度计和陀螺仪信号，输出一个受试者级别的二分类结果：

- 类别索引 0：`PD`，原始条件为 `Parkinson's`；
- 类别索引 1：`DD`，合并 `Other Movement Disorders`、`Essential Tremor`、`Multiple Sclerosis`、`Atypical Parkinsonism`；
- `Healthy` 不进入当前任务。

数据共 390 名受试者：PD 276、DD 114。每人当前均有 11 个 activity、每个 activity 均有左右腕记录，因此共有 4,290 个双腕 activity pair、8,580 个腕侧信号文件。

### 1.2 当前 11 个 activity（固定顺序）

1. CrossArms
2. DrinkGlas
3. Entrainment
4. HoldWeight
5. LiftHold
6. PointFinger
7. Relaxed
8. RelaxedTask
9. StretchHold
10. TouchIndex
11. TouchNose

这个顺序同时决定：输入 activity 轴、每个 activity 的 normalization 索引、可学习 activity embedding 索引，以及输出 attention 的含义，不能随意重排。

### 1.3 当前明确没有做的内容

- 没有做 MDS-UPDRS/UPDRS 连续严重程度回归、序数分级或临床量表估计；
- 没有把 Healthy 纳入分类，也没有分别预测四种 DD 亚型；
- 没有外部数据集验证、跨中心/跨设备验证或临床部署验证；
- 不声称是论文指标复现；
- v2 没有数据增强、重采样、异常点裁剪/剔除、时间戳插值或 L1 以外的额外滤波；
- 没有做左右腕逐采样点同步或原始时间轴直接拼接；两腕独立编码后晚期融合；
- 当前正式主模型没有 class weighting、label smoothing、focal loss、balanced sampler 或校准模型；
- 还没有完成 leave-one-activity-out、多 activity 缺失压力测试、传统机器学习强基线或外部 holdout；
- 虽然接口保留 left/right/bilateral 和 acc/gyro/acc_gyro，但当前主结果是 bilateral + acc_gyro。

## 2. 数据流程

### 2.1 原始数据读取与双腕配对

项目内原始数据副本位于 `/home/zyt/MFAM/data/raw/pads`，来源为 `/home/zyt/pads`；训练不再直接解析服务器外部原始目录。原始副本与预处理数据彼此独立，v2 输出位于：

`/home/zyt/MFAM/data/processed/pads_multi_activity/v2_l1_full_length`

读取和配对规则来自 `src/datasets/prepare_pads.py` 与 `scripts/prepare_multi_activity_data_v2.py`：

1. 从 `patients/patient_*.json` 读取 `id` 与 `condition`，生成 PD/DD 标签并排除 Healthy；
2. 从 `movement/observation_*.json` 查找该 subject；
3. 对每个配置 activity，要求恰好一个同名 `session.record_name`；
4. 在同一个 observation JSON、同一个 session 对象内，要求记录集合恰好为 `LeftWrist` 和 `RightWrist`；
5. 两腕元数据通道必须严格为 `Time + Accelerometer XYZ + Gyroscope XYZ`；
6. 两个文件必须存在、样本数相等、时间戳严格递增、有效采样率在标称采样率 ±20% 内；
7. 记录两腕最大时间偏移，但不据此对齐或重采样；
8. 每行 manifest 表示一个已经验证的双腕 pair，`pair_id` 为 `subject_activity_sessionXX`。

因此，配对依据不是文件名猜测；但“同一 session”也不等于两块手表具有硬件级逐点同步。

### 2.2 v2 预处理

原始 TXT/CSV 按 `[time, 6 channels]` 读取，时间列不进入模型。预处理顺序是：

1. 对 AccX/AccY/AccZ 分别求解 L1 trend filtering：
   `0.5 * ||y - trend||²₂ + 50 * ||D² trend||₁`；
2. 加速度输出为 `raw_acc - trend`；实际实现为 batched ADMM，`rho=40`、最多 5,000 次迭代、绝对/相对容差均为 `1e-4`，不收敛会直接失败；
3. GyroX/GyroY/GyroZ 保持原值；
4. 六个通道统一删除开头 48 点；
5. 保存 float32 NPY，单腕布局 `[6,T]`；
6. 不执行其他滤波、增强、裁剪、插值、重采样或异常值删除。

### 2.3 真实输入长度

采样率配置为 100 Hz。v2 在删除前 48 点后，实际保存并送入模型的长度为：

| Activity | 每腕实际 T | MIL instance 数 I | Top-K 保留数 K |
|---|---:|---:|---:|
| CrossArms | 976 | 18 | 6 |
| DrinkGlas | 976 | 18 | 6 |
| Entrainment | 2000 | 39 | 12 |
| HoldWeight | 976 | 18 | 6 |
| LiftHold | 976 | 18 | 6 |
| PointFinger | 976 | 18 | 6 |
| Relaxed | 2000 | 39 | 12 |
| RelaxedTask | 2000 | 39 | 12 |
| StretchHold | 976 | 18 | 6 |
| TouchIndex | 976 | 18 | 6 |
| TouchNose | 976 | 18 | 6 |

这里的 I、K 来自当前实际 MIL 参数：窗口 100 点、步长 50 点，`I=floor((T-100)/50)+1`，`K=ceil(0.3I)`。

### 2.4 Subject-level split 与防泄漏

当前正式运行使用冻结文件：

`/home/zyt/MFAM/splits/pads_classification/pd_vs_dd_crossarms_seed42.json`

尽管文件名保留了 `crossarms`，它保存的是 subject ID 集合，因此被 11 个 activity 共用。正式产物记录的 SHA-256 为：

`62fe2451c9b4071ec7b563e064142e4815d8fd7fee5a130bc19aa836cfdeb207`

| Split | Subject 数 | PD | DD |
|---|---:|---:|---:|
| Train | 264 | 187 | 77 |
| Validation | 47 | 33 | 14 |
| Test | 79 | 56 | 23 |

代码先取得 subject 集合并强制三集合两两不相交，再选择所有对应 activity records。因此同一 subject 的不同 activity、左右腕不会跨 split。

三个正式 seed 42/43/44 使用完全相同的冻结 split；seed 只改变参数初始化、训练 DataLoader shuffle 等随机过程。

### 2.5 Normalization

v2 使用 `activity × wrist × channel` 统计：

- mean shape：`[11,2,6,1]`；
- std shape：`[11,2,6,1]`；
- 统计只遍历 train subjects 的完整预处理信号；
- 每个 activity、left/right、六个通道分别累计所有训练时间点的 sum 与 square sum；
- `std = sqrt(E[x²]-E[x]²)`，下限 `1e-6`；
- validation、test、inference 复用 checkpoint 中保存的 train-only mean/std；
- 没有 per-subject normalization。

left/right/bilateral 只通过 `wrist_mask` 控制有效腕。统计函数仍保留两腕统计，使三个腕侧对照可共享同一统计定义。acc/gyro/acc_gyro 通过通道索引选择 `[0:3]`、`[3:6]` 或全部六轴；改成 3 轴时必须同步把 `model.input_channels` 改为 3。

### 2.6 Dataset/DataLoader 最终输出

`SubjectActivityDataset.__getitem__` 为一个 subject 返回：

- `x`：`[A,2,C,T_subject_max]`；当前 A=11、C=6、`T_subject_max=2000`；短 activity 在右侧补零；
- `wrist_mask`：`[A,2]`，left/right/bilateral 分别为 `[1,0]`、`[0,1]`、`[1,1]`；
- `activity_mask`：`[A]` bool；缺失 activity 为 False；
- `activity_lengths`：`[A]`；当前为 976 或 2000，缺失 activity 为 0；
- `y`：标量 long；
- subject/pair/path/activity 等审计元数据。

`collate_subject_activities` 再把同一 mini-batch 的所有 subject 右侧补到该 batch 最大 T，最终主配置输出：

- `x=[B,11,2,6,T_batch_max]`，当前完整数据通常 `T_batch_max=2000`；
- `wrist_mask=[B,11,2]`；
- `activity_mask=[B,11]`；
- `activity_lengths=[B,11]`；
- `y=[B]`。

关键点：padding 只是容器。`SubjectMFAM` 在进入 FFT 前按 `activity_lengths` 切回真实长度，所以补零不参与频率分解、卷积或 MIL。

## 3. 当前 SubjectMFAM 真实架构

### 3.1 参数共享关系

- 左腕与右腕：共享同一个 `WristMFAMEncoder` 对象，参数完全共享；
- 11 个 activity：共享同一个完整 `MFAM` activity encoder，包括频率分解、卷积、Channel Attention、MIL 和 bilateral fusion 逻辑；
- activity 之间只有可学习 activity embedding 的行不同；activity attention scorer 本身共享；
- bilateral fusion 在 SubjectMFAM 中只产生 514 维特征，其原有 per-activity classifier 被替换为 `Identity`；
- 最终只在 subject embedding 上做一次 PD/DD 分类。

### 3.2 单 activity、单腕编码路径

对同一真实长度组，输入为 `[N,2,6,T]`。双腕展平并按 `wrist_mask` 选择有效腕，bilateral 模式进入共享腕编码器的是 `[2N,6,T]`。

#### 频率分解

- 输入：`[2N,6,T]`；
- 对时间维做 rFFT；
- 频带为 `[0.5,3.0)`、`[3.0,7.0)`、`[7.0,12.0]`；最后一个上界包含 12 Hz；
- 每个频带在频域置零后 irFFT 回原 T；
- `include_original=false`，不保留原始全频流；
- 输出：`[2N,18,T]`；
- 无可学习参数。

#### Multi-scale Conv1D

三个并行 `Conv1d(18→64,kernel=3)` 分支，dilation 分别为 1、2、4，使用 exact same padding、BatchNorm、SiLU：

- 三分支输出均为 `[2N,64,T]`；
- 有效感受野分别为 3、5、9 点；
- 拼接为 `[2N,192,T]`；
- `1×1 Conv1d(192→128) + BatchNorm + SiLU`；
- 输出 `[2N,128,T]`；
- 当前 encoder dropout 为 0。

旧配置中的 `model.cnn.channels/kernel_size/dropout` 从不参与 SubjectMFAM，已在 latest-v2 cleanup 中移除；当前实际读取 `model.encoder.*`。

#### Channel Attention

- 对 `[2N,128,T]` 分别做时间维 global average pooling 和 max pooling，得到两个 `[2N,128,1]`；
- 二者经过同一个共享 MLP：`Conv1d 128→16 → ReLU → Conv1d 16→128`；
- 两路结果相加后 sigmoid，得到 channel weights `[2N,128,1]`；
- 与原特征逐通道相乘，输出仍为 `[2N,128,T]`。

#### Attention-MIL 与 Top-K

- 以 100 点窗口、50 点步长在时间轴 unfold；
- 每个窗口在窗口内取均值，形成 instances `[2N,I,128]`；若 T<100，则整段均值成为唯一 instance；
- tanh scorer：`Linear 128→64 → Tanh → Linear 64→1`，输出 scores `[2N,I]`；
- `raw_attention=softmax(scores)`；
- 当前 `hard_gating=true` 且 `hard_gating_train_only=false`，所以训练、validation、test、inference 都启用 Top-K；
- 保留 `K=ceil(0.3I)` 个最大权重，其余置零，再对保留权重重新归一化；
- 加权求和得到每腕固定 128 维 embedding。

当前两种长度分别为：T=976 → I=18/K=6；T=2000 → I=39/K=12。instance 数没有写死。

### 3.3 Bilateral fusion

两腕 MIL embedding 恢复为 `[N,2,128]`，按 `wrist_mask` 清零无效腕。融合向量严格为：

`[left, right, masked_mean, abs(left-right)*both_valid, wrist_mask]`

维度为 `128+128+128+128+2=514`，输出每个 activity 的 `[N,514]` embedding。

- 双腕有效时 mean 为两腕均值，difference 为绝对差；
- 单腕有效时无效腕为 0、mean 等于有效腕、difference 强制为 0；
- SubjectMFAM 中 bilateral head 的 per-activity classifier 已被 `Identity` 替换，所以这里不产生 activity-level logits。

### 3.4 Activity embedding 与 activity-level fusion

所有有效 activity 经共享 MFAM 后恢复为 `features=[B,11,514]`；缺失槽位保持 0，但不会进入 MFAM。

Activity Attention Aggregator 的真实计算是：

1. 位置索引 `0...10` 查可学习 embedding table `[11,514]`；
2. `enriched = LayerNorm(features + activity_embedding)`，shape `[B,11,514]`；
3. scorer：`Linear 514→128 → Tanh → Dropout(0.1) → Linear 128→1`；
4. 无效 activity score 填为该 dtype 的最小有限值；
5. 在 activity 轴 softmax，再把无效位置显式置 0，得到 `[B,11]` attention；
6. `subject_embedding = Σ enriched_a × attention_a`，输出 `[B,514]`。

注意加权的是加过 activity embedding 并经 LayerNorm 的 `enriched`，不是原始 bilateral feature。

### 3.5 最终分类头

`Dropout(0.2) → Linear(514,2)`：

- logits：`[B,2]`；
- probabilities：对 logits 做 softmax，`[B,2]`；
- `probabilities[:,1]` 是 DD 概率。

### 3.6 参数量

当前正式运行 `model.json` 与现场实例化均得到：总参数量 121,906，全部可训练。

| 参数组 | 参数量 |
|---|---:|
| Frequency decomposition | 0 |
| 共享 Multi-scale Conv + Channel Attention encoder | 39,824 |
| 共享 Attention-MIL scorer | 8,321 |
| Bilateral fusion（classifier 已为 Identity） | 0 |
| Activity embedding + LayerNorm + Activity scorer | 72,731 |
| Subject classifier | 1,030 |
| **总计** | **121,906** |

## 4. 完整前向数据流

### 4.1 文字流程图

```text
PADS patient/observation JSON + 双腕信号
  → 11 份 bilateral manifest
  → subject-level grouping + train-only normalization
  → batch x=[B,11,2,6,2000], lengths∈{976,2000}
  → 按 activity_mask 选有效槽位、按真实 T 分组并在 FFT 前切片
  → 每组 [N_T,2,6,T]
  → 展平有效腕（当前 bilateral 为 [2N_T,6,T]）
  → rFFT 三频带分解 [2N_T,18,T]
  → 三路 dilation Conv1D [2N_T,3×64,T]
  → concat + 1×1 fusion [2N_T,128,T]
  → Channel Attention [2N_T,128,T]
  → 100 点/50 步长 instances [2N_T,I,128]
  → tanh attention + Top-K 30% + 加权和 [2N_T,128]
  → 恢复左右腕 [N_T,2,128]
  → [left,right,mean,abs-diff,mask] [N_T,514]
  → 恢复 11 个 activity [B,11,514]
  → + 可学习 activity embedding + LayerNorm [B,11,514]
  → masked activity attention [B,11]
  → 加权和 subject embedding [B,514]
  → Dropout(0.2) + Linear(514→2)
  → logits/probabilities [B,2]
  → 单模型默认 argmax；正式三 seed 协议对 DD 概率均值后使用 validation 阈值
```

### 4.2 各模块 shape 表

| 阶段 | 输入 | 输出 | 当前核心参数 |
|---|---|---|---|
| DataLoader | 每 subject 11 个 activity | `[B,11,2,6,Tmax]` | B=8；Tmax 当前为 2000 |
| 长度恢复 | padded batch | 短组 `[8B,2,6,976]`、长组 `[3B,2,6,2000]`（满 batch 且全 activity 时） | padding 在 FFT 前切除 |
| 有效腕选择 | `[N,2,6,T]` | `[N_valid_wrist,6,T]` | bilateral 时 `N_valid_wrist=2N` |
| Frequency | `[Nv,6,T]` | `[Nv,18,T]` | 0.5–3、3–7、7–12 Hz |
| 三路 Conv | `[Nv,18,T]` | 3×`[Nv,64,T]` | k=3，dilation=1/2/4 |
| Conv fusion | `[Nv,192,T]` | `[Nv,128,T]` | 1×1 Conv |
| Channel Attention | `[Nv,128,T]` | feature `[Nv,128,T]`、weight `[Nv,128,1]` | reduction=8 |
| MIL windowing | `[Nv,128,T]` | `[Nv,I,128]` | window=100，stride=50 |
| MIL attention | `[Nv,I,128]` | wrist embedding `[Nv,128]` | tanh 128→64→1，Top-K 30% |
| 恢复腕轴 | `[Nv,128]` | `[N,2,128]` | 左右共享 encoder |
| Bilateral fusion | `[N,2,128]` + `[N,2]` | `[N,514]` | left/right/mean/diff/mask |
| 恢复 activity 轴 | 各长度组 feature | `[B,11,514]` | 共享 activity encoder |
| Activity enrich | `[B,11,514]` | `[B,11,514]` | Embedding `[11,514]` + LN |
| Activity attention | `[B,11,514]` + mask | attention `[B,11]` | 514→128→1 |
| Subject aggregation | enriched + attention | `[B,514]` | masked weighted sum |
| Classifier | `[B,514]` | logits `[B,2]` | Dropout .2 + Linear |

## 5. 训练流程

当前 v2 三 seed 正式运行的实际配置（保存于各 run 的 `config.yaml`）为：

- seeds：42、43、44；`deterministic=true`；
- 最大 epochs：50；
- train batch size：8；evaluation batch size：8；
- loss：未加权 CrossEntropyLoss，label smoothing=0；
- optimizer：AdamW，lr=`2e-4`，weight decay=`1e-4`，betas=`[0.9,0.999]`；
- scheduler：CosineAnnealingLR，`T_max=50`，minimum lr=`1e-6`，每 epoch step；
- CUDA AMP：开启；
- gradient clipping：global norm 5.0；
- early stopping：validation macro-F1 最大化，patience=12，minimum delta=0；
- checkpoint：每 epoch 保存 `last.pt`，validation macro-F1 严格提升时保存 `best.pt`；不保存周期 epoch checkpoint；
- 正式训练时 `run_test_after_training=false`，先完成训练和 validation 导出，再按冻结流程评估 test。

三个 seed 的训练轨迹：

| Seed | best epoch（1-based） | best val macro-F1 | 同 epoch train macro-F1 | 实际停止 epoch |
|---:|---:|---:|---:|---:|
| 42 | 2 | 0.4125 | 0.4658 | 14 |
| 43 | 13 | 0.4503 | 0.6266 | 25 |
| 44 | 23 | 0.5712 | 0.8844 | 35 |

seed 44 已显示明显 train-validation gap；三个 seed 的最佳 epoch 与验证表现差异也较大。

### 多 seed、阈值与概率融合

正式协议不是 logits 平均，而是：

1. 每个 seed 使用自己的 `best.pt` 在 validation 生成 subject-level softmax 概率；
2. 按 subject 对三个 seed 的 `P(DD)` 做算术平均；
3. 候选阈值由 validation 平均概率的所有唯一值、相邻中点、0、1 构成；
4. 优先最大化 validation macro-F1；并列时最大化 balanced accuracy；再并列时选离 0.5 最近者；
5. 得到阈值 `0.2778912236293157`；
6. 每个 seed 的 test 概率只导出一次，再按 subject 平均；
7. test 使用上述固定 validation 阈值，不在 test 上重新选阈值；
8. test CI 使用分层 subject bootstrap，5,000 次，seed 20260821。

## 6. 评估与推理

### 6.1 当前指标

基础 `classification_metrics` 输出：

- accuracy、balanced accuracy；
- macro/weighted precision、recall、F1；
- per-class precision、recall、F1、support；
- confusion matrix；
- macro/per-class AUROC；
- Brier score、negative log-likelihood；
- loss、样本数、耗时、batch 数。

正式 ensemble 分析输出 accuracy、balanced accuracy、macro precision/recall/F1、confusion matrix、AUROC、DD-only Brier、NLL，并可输出 bootstrap CI。

注意：基础 metrics 的 Brier 是两类 one-hot 误差平方和；ensemble 脚本的 Brier 只基于 `P(DD)`。二分类时前者约为后者的 2 倍，两套数值不能直接混用。

### 6.2 Validation/test 如何产生 subject prediction

一个 DataLoader item 就是一名 subject，模型只输出一组 `[PD,DD]` 概率，所以没有额外的 activity-level vote。11 个 activity 在 embedding 层通过 activity attention 聚合后直接形成 subject probability。

`evaluate.py` 默认使用 softmax argmax，即二分类的 0.5 阈值；它会保存每名 subject 的 ID、目标、预测、概率、activity mask、activity attention、activity pair IDs 与路径信息。正式三 seed 结果另由 `analyze_probability_ensemble.py` 执行上节所述概率平均与固定阈值协议。

### 6.3 `inference.py` 的真实输入输出

Subject 模式输入不是任意 NPY/TXT 路径，而是：

- `--config`：必须给出与 checkpoint 输入元数据一致的 subject 配置；
- `--checkpoint`：通常为某个 seed 的 `best.pt`；
- `--subject-id`：从配置的 11 个 manifests 中选择该 subject；或
- `--pair-id`：先定位一个 pair，再把它映射为 subject，实际仍加载该 subject 的全部配置 activity。

处理流程：读取所有配置 manifest → 选择 subject records → 用 checkpoint mean/std 构造 `SubjectActivityDataset` → 得到 `[11,2,C,Tmax]`、mask、length → 加 batch 轴 → 单个模型前向 → softmax → argmax。

JSON 输出包含：subject ID、配置 activity 顺序、activity mask、每个 activity 的真实长度、activity attention、activity pair IDs、预测索引/标签、PD/DD 概率。

当前 `inference.py` 不支持：

- 自动加载三 seed 并融合；
- 自动使用正式 validation 阈值 0.277891；
- 直接传入一组未经 manifest 验证的左右腕文件；
- 输出 bootstrap CI。

## 7. 当前实验状态

### 7.1 已完成实验

- 历史 CrossArms bilateral single-activity 正式基线；
- v1 11-activity SubjectMFAM 的探索与三 seed 正式运行；
- v2 full-length/L1/variable-length SubjectMFAM 三 seed 正式运行；
- v1-v2 paired preprocessing comparison：相同 batch size=8、max epochs=50、seeds=42/43/44、冻结 split、模型/loss/优化器/早停/阈值协议；
- 11 activity × v1/v2 × 3 seeds = 66 次 single-activity paired 正式训练；
- v2 preprocessing 数据质量审计与 variable-length smoke test；
- latest-v2 cleanup 后服务器测试：40 passed in 2.13 s；减少的两个测试仅属于已移除的 CNN1D baseline。

### 7.2 当前主结果：v2 多活动三 seed ensemble

固定 validation 阈值 `0.2778912236293157`，test n=79：

| 指标 | 结果 |
|---|---:|
| Accuracy | 0.6709 |
| Balanced accuracy | 0.6013 |
| Macro-F1 | 0.6013 |
| AUROC | 0.6203 |
| DD recall | 0.4348 |
| NLL | 0.5888 |
| DD-only Brier | 0.1993 |
| Confusion matrix（rows true PD/DD, cols pred PD/DD） | `[[43,13],[13,10]]` |

95% subject-bootstrap CI：accuracy `[0.5696,0.7722]`、balanced accuracy `[0.4876,0.7240]`、macro-F1 `[0.4840,0.7190]`、AUROC `[0.4720,0.7655]`。

0.5 阈值下同一 ensemble 的 macro-F1 只有 0.5385、DD recall 4/23；因此阈值协议对结果影响很大。

### 7.3 严格 paired v1-v2 预处理比较

| 版本 | Accuracy | Balanced acc. | Macro-F1 | AUROC | DD recall |
|---|---:|---:|---:|---:|---:|
| v1 | 0.6329 | 0.5489 | 0.5495 | 0.5970 | 0.3478 |
| v2 | 0.6709 | 0.6013 | 0.6013 | 0.6203 | 0.4348 |
| v2-v1 | +0.0380 | +0.0524 | +0.0519 | +0.0233 | +0.0870 |

但所有 paired bootstrap 差值区间均跨 0，不能据此宣称 v2 显著优于 v1。

### 7.4 Single-activity 结果定位

按相同三 seed + validation threshold ensemble 协议：

- v1 单动作最高 macro-F1：HoldWeight 0.6517，AUROC 0.6980；
- v2 单动作最高 macro-F1：HoldWeight 0.6302，AUROC 0.6731；
- v2 单动作最高 AUROC：CrossArms 0.7368，macro-F1 0.6220。

这些是不同实验分支的结果，不能与多活动模型做未经统计检验的“绝对最佳模型”结论。

### 7.5 Activity attention 行为

v2 三个 seed 在 test 上的归一化 attention entropy 分别约 0.905、0.686、0.510，平均 attention（先跨 seed、再跨 subject）约为：

`HoldWeight .343, DrinkGlas .153, CrossArms .114, Entrainment .107, StretchHold .102, LiftHold .073, PointFinger .032, Relaxed .024, TouchIndex .023, RelaxedTask .022, TouchNose .008`

这不是完全的单 activity collapse，但 seed 44 已明显集中，且不同 seed 权重分布差异大。attention 只能描述模型行为，不能当作临床重要性或因果证据；需要 activity 消融验证。

## 8. 代码实现与设计一致性审查

### 8.1 已确认一致/正确的部分

- subject split 在 activity 组织、裁剪/MIL 之前完成；三个 split 强制互斥；
- v2 normalization 只使用 train subjects，真实 shape `[11,2,6,1]`；
- current SubjectMFAM 无 `T=1024` 写死；频率分解、Conv1D、MIL instance 数均由真实 T 决定；
- v2 Dataset 返回真实 `activity_lengths`；模型在 FFT 前切除 padding；
- 缺失 activity 不进入 MFAM，activity attention 被 mask 为 0；
- 左右腕共享同一个 encoder；所有 activity 共享同一个 MFAM encoder；
- active 配置只保留 v2 subject-level 主线；left/right/bilateral 与 acc/gyro/acc_gyro 参数接口仍保留，v1、single-activity 配置与独立 CNN1D baseline 已隔离；
- checkpoint 保存模型、优化器、scheduler、AMP scaler、class names、normalization 与输入 metadata；
- 正式训练产物保存 config、environment、split、normalization、逐 epoch 日志、best/last checkpoint 与逐 subject 预测。

### 8.2 固定长度/固定数量审查结论

- `T=1024`：不在当前 active 源码、配置或脚本中写死；相关 v1/single-activity 配置和实验脚本已移至隔离区；
- MIL instance 数：不固定，由 T、window=100、stride=50 动态计算；
- activity 数：源码没有写死数字 11，但 `SubjectMFAM.forward` 要求输入 A 必须等于当前 config/checkpoint 的 `activity_count`；
- activity 顺序：固定并有语义，不能调换；
- variable-length 的含义：支持每个 activity 不同 T，并不表示任意 A/任意顺序；
- 当前全量数据每人都有 11 个 activity，所以 `activity_mask` 的功能通过测试验证过，但没有真实缺失数据上的正式性能证据。

### 8.3 README/配置/代码一致性

latest-v2 cleanup 已解决原来的主要漂移：

1. README 现在只描述 `[B,A,2,C,T]` 的 v2 subject-level 主线；
2. active `configs/` 只保留 `pads_multi_activity_v2.yaml`；
3. v2 YAML 已移除未使用的 `model.cnn.*` 和 `model.classifier.*`；最终分类 dropout 只由 `model.activity_fusion.classifier_dropout` 控制；
4. canonical YAML 已与正式协议一致设为 50 epochs、train/eval batch size 8、`run_test_after_training=false`；
5. 独立 CNN1D 注册、旧 v1 配置与对应脚本已移至项目外隔离区；
6. 仍存在一个数据审计产物口径差异：v2 `audit.json` 标记 `status=pass` 且未列出 timestamp gap；后生成的 `quality_summary.json` 为 `pass_with_warnings`，检测到 4 条 >10× median-dt 间隙。应以更完整的 quality summary 明示警告。

### 8.4 当前已知问题

1. **类别不平衡**：390 人中 PD/DD=276/114；当前 loss 无 class weight，阈值 0.5 明显偏向 PD。
2. **样本与验证集较小**：validation 仅 47 人、test 79 人，置信区间宽。
3. **过拟合与 seed 不稳定**：best epoch、val F1、attention 分布差异明显；seed 44 best epoch train macro-F1 0.884 vs val 0.571。
4. **Activity attention 集中**：HoldWeight 长期占优，seed 44 更明显；尚不能证明多活动互补性。
5. **阈值接口不统一**：`evaluate.py` 和 `inference.py` 默认 argmax/0.5，不会读取正式 ensemble 阈值；单 checkpoint inference 也不是三 seed ensemble。
6. **Brier 定义不统一**：基础 metrics 与 ensemble 脚本使用不同二分类 Brier 定义。
7. **测试集已被反复查看**：单次训练代码没有用 test 选 checkpoint/阈值，但同一冻结 test 已用于 v1/v2、多 seed、调参探索及 66 个单动作实验。对整个研究过程而言它不再是完全未触碰的最终盲测集，后续模型选择存在人为 test feedback / multiple-comparison 风险。
8. **Checkpoint provenance 校验不完整**：`validate_checkpoint_input` 检查活动顺序、腕侧/传感器模式和通道等，但不检查 manifest hash、split hash或数据 audit hash；`evaluate.py` 也只检查 checkpoint normalization 与当前 bundle 的 shape 后直接覆盖，未校验数值和 class_names。错误配置仍可能产生语义错位评估。
9. **真实 missing-activity 未验证**：代码和单元测试支持 mask，但正式 390 人均为完整 11 activity。
10. **数据质量警告未处理**：4 个大时间戳间隙；2,095/8,580 个腕侧记录至少一个通道触发 >20 MAD robust-outlier flag。当前未删除、裁剪、插值或重采样，这是正确的审计边界，但仍需分层复核。
11. **BatchNorm 共享**：同一个 encoder/BatchNorm 被不同 activity 和腕侧共享；SubjectMFAM 又按短/长组顺序调用。它符合“共享 encoder”设计，但运行统计混合了 activity/腕侧分布，是否合适尚未做消融。
12. **传感器消融配置有联动要求**：修改 `data.sensor_mode` 时必须同步修改 `model.input_channels`，否则会 shape error；目前不是自动推导。

### 8.5 待确认问题

- 四种 DD 亚型在冻结 train/validation/test 中的具体分层是否足够均衡；当前 split 只按 PD/DD 总标签冻结；
- 4 条 timestamp gap 对具体频谱和分类输出的影响；
- robust-outlier flags 是真实病理运动、设备尖峰还是量纲/采集问题；
- learned activity attention 是否优于简单概率平均、embedding 平均、majority vote；
- HoldWeight 的优势是否在 leave-one-activity-out、重复 subject CV 或外部 split 上稳定；
- 缺 1/3/5 个 activity 时的性能退化；
- v2 对 v1 的提升能否在更多独立 subject splits 上重复；当前 paired bootstrap CI 均跨 0；
- 最终部署究竟采用单 seed checkpoint 还是三 seed ensemble，以及选定阈值应如何固化进 inference artifact；
- 是否需要建立一个从未用于调参和实验筛选的最终 holdout，或改用 nested/repeated subject-level CV。

## 9. 主要代码与证据位置

### 源码

- 数据配对：`/home/zyt/MFAM/src/datasets/prepare_pads.py`
- v2 预处理：`/home/zyt/MFAM/scripts/prepare_multi_activity_data_v2.py`
- L1 trend filter：`/home/zyt/MFAM/src/datasets/preprocessing.py`
- manifest/Dataset：`/home/zyt/MFAM/src/datasets/manifest.py`
- subject grouping/collate：`/home/zyt/MFAM/src/datasets/subject_activity.py`
- split/normalization/DataLoader：`/home/zyt/MFAM/src/datasets/builders.py`
- SubjectMFAM：`/home/zyt/MFAM/src/models/subject_mfam.py`
- shared wrist MFAM：`/home/zyt/MFAM/src/models/mfam.py`
- frequency/encoder/attention/MIL：`/home/zyt/MFAM/src/models/{frequency,encoder,attention,mil}.py`
- bilateral fusion：`/home/zyt/MFAM/src/models/bilateral.py`
- activity fusion：`/home/zyt/MFAM/src/models/activity_fusion.py`
- train/eval/inference：`/home/zyt/MFAM/{train,evaluate,inference}.py`
- ensemble threshold：`/home/zyt/MFAM/scripts/analyze_probability_ensemble.py`

### 当前正式证据

- v2 数据审计：`/home/zyt/MFAM/data/processed/pads_multi_activity/v2_l1_full_length/{audit.json,quality_summary.json}`
- v2 seeds：`/home/zyt/MFAM/outputs/pads_classification/formal_v2_full_length_seed{42,43,44}_20260821`
- v2 ensemble：`/home/zyt/MFAM/outputs/pads_classification/formal_v2_full_length_ensemble_20260821`
- 历史 paired v1-v2、66 次单动作和旧调参产物：`/home/zyt/MFAM_cleanup_archive_20260821-224905/quarantine/outputs/pads_classification/`
