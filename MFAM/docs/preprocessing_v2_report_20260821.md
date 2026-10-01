# PADS 多活动完整时长预处理与变长输入修改报告

日期：2026-08-21  
服务器项目：`/home/zyt/MFAM`  
范围：数据预处理、质量审计、variable-length activity 输入和 normalization；未修改 Activity Attention、loss、frequency bands 或主体网络参数结构。

> Latest-v2 cleanup 说明：本文中的 v1 对照只用于记录当时的迁移证据。v1 数据、配置和历史实验现已移至 `/home/zyt/MFAM_cleanup_archive_20260821-224905`，不属于当前 active 运行链。

## 1. 审查结论

旧 v1 数据本身保存了原始 1024/2048 点信号；统一成 1024 点发生在 Dataset 的 `crop_or_pad`。MFAM 的 FFT、Conv1D 和 Attention-MIL 原本已经根据输入 `T` 动态工作，真正需要修改的是：

- SubjectActivityDataset 原来要求所有 activity 可直接 `torch.stack`；
- 默认 collate 不支持不同 subject tensor 的时间维差异；
- SubjectMFAM 原来把所有 activity 按相同 `T` 一次展平编码；
- normalization 原来只有 `[wrist,channel,1]`，没有 activity 维。

服务器 `/home/zyt/pads/scripts/run_preprocessing.py` 已核实官方逻辑：Acc XYZ 使用 `raw - l1_trend_filter(raw, lambda=50)`，Gyro 不处理，然后删除前 48 点。官方脚本会删除 LiftHold、PointFinger、TouchIndex；本实现按当前实验要求保留全部 11 个 activity。

## 2. 实施内容

### 2.1 独立 v2 数据

新增：

- `scripts/prepare_multi_activity_data_v2.py`
- `src/datasets/preprocessing.py`
- `scripts/summarize_v2_quality.py`
- `configs/pads_multi_activity_v2.yaml`

输出目录：

`/home/zyt/MFAM/data/processed/pads_multi_activity/v2_l1_full_length`

没有覆盖或修改 `data/processed/pads_multi_activity/v1`。

v2 处理顺序：

1. 读取不可变 raw copy；
2. 检查 7 列格式、有限值、时间戳严格递增和有效采样率；
3. Acc XYZ 求解 `0.5||y-x||² + 50||D²x||₁`，输出 `raw - trend`；
4. Gyro XYZ 原样保留；
5. 删除每段前 48 点；
6. 保存 float32 `[6,time]` NPY、manifest、solver audit 和逐腕质量记录。

服务器环境没有 CVXPY/CVXOPT，因此使用批量稀疏 ADMM 求解同一个凸目标，`rho=40`、最多 5000 次，并强制检查 primal/dual convergence。它与官方目标相同，但不能声称与 CVXOPT 输出逐位一致。

### 2.2 Variable-length activity

- v2 配置的 `sequence_length: null`，train/eval/inference 都使用 `full`；
- Dataset 返回 `activity_lengths`；
- batch tensor 只在右侧补零到当前 batch 的最大 `T`；
- SubjectMFAM 按真实长度对有效 activity 分组，并在 FFT 前切片；
- padding 不进入 FrequencyDecomposition、Conv1D 或 Attention-MIL；
- 每个 activity 仍输出固定 514 维 bilateral embedding，再进入原有 activity-level fusion。

没有新增网络参数，旧 subject checkpoint strict-load 通过。

### 2.3 Normalization

v2 subject 配置使用 `[activity,wrist,channel,1]` mean/std，实际形状为 `[11,2,6,1]`。统计量只遍历冻结 train subjects 的完整预处理信号；validation/test 复用 checkpoint 数值。没有 per-subject normalization。

旧配置没有 `normalization_scope` 时仍使用原来的 `[2,C,1]` 行为。

## 3. 完整数据审计结果

- 状态：结构与预处理 `pass`；独立质量汇总 `pass_with_warnings`
- subjects：390
- 标签：PD 276、DD 114
- 每个 activity：390 bilateral pairs
- 输出腕侧文件：8580
- NaN/Inf：0
- 非单调时间戳：0
- 最大常数段：3 点
- 有效采样率范围：99.2065–100.8077 Hz
- L1 solver batches：275
- 最大收敛迭代数：4538（低于 5000）

长度：

- 2000 点：Entrainment、Relaxed、RelaxedTask，各 780 个腕侧文件；
- 976 点：其余 8 个 activity，各 780 个腕侧文件。

真实文件核对确认：Gyro 输出与原始信号 `[48:]` 的 float32 值逐点完全一致；Acc 已执行去趋势；所有输出有限。

### 质量 warning

使用 `maximum_dt / median_dt > 10` 标记时间戳间隙，共 4 条：

- subject 006 / StretchHold / left：约 0.647 s；
- subject 189 / TouchIndex / right：约 0.218 s；
- subject 118 / TouchIndex / left：约 0.103 s；
- subject 017 / StretchHold / left：约 0.102 s。

共有 2095 个腕侧记录至少出现一个 `20 × MAD` 鲁棒异常点，主要集中于 CrossArms 和 LiftHold。这是审计 flag，不代表这些点一定是错误；本轮未删除、裁剪、插值或重采样任何记录。

## 4. 防泄漏与兼容性

- 冻结 split SHA256 保持为 `62fe2451c9b4071ec7b563e064142e4815d8fd7fee5a130bc19aa836cfdeb207`；
- train/validation/test 为 264/47/79 subjects；
- 同一 subject 的所有 activity 仍属于同一 split；
- 11 个 activity 均保留，包括 LiftHold、PointFinger、TouchIndex；
- 旧 v1 固定 1024 配置继续有效；
- 旧正式 checkpoint strict-load 通过；
- v2 `left+acc`、`right+gyro`、`bilateral+acc_gyro` 前向通过；
- v2 CrossArms single-activity `[B,2,6,976]` baseline 前向通过；
- subject inference 返回每个 activity 的真实长度。

## 5. 测试与 smoke

- Pytest：`42 passed in 2.21s`
- 静态编译：51 个 Python 文件通过
- 配置验证：旧 single activity、旧固定长度 multi activity、新 v2 均通过
- Padding invariance：修改 activity 的 padded tail 不改变 logits
- 真实 GPU smoke：1 train batch + 1 validation batch，成功前向、反向、保存 best/last checkpoint
- GPU peak memory：221,435,904 bytes（该数值仅对应 batch=2、单 batch smoke）
- CPU subject inference：subject 006 成功，长度为 976/2000 的完整映射正确

Smoke 只证明数据与训练链路可运行，不代表模型性能。

## 6. 产物与哈希

- 完整 audit：`data/processed/pads_multi_activity/v2_l1_full_length/audit.json`
  - SHA256 `bb0b012a46944e32e093914172565c31e31d583286d1be5e559b05f35c44738c`
- quality records：`quality_records.jsonl`
  - SHA256 `4686ecd0631fe7042d0cbfb1fe8537699d9dd7ccacfd81a5bbe8abc41fad7eaf`
- quality summary：`quality_summary.json`
  - SHA256 `61ee103f8658b59858dd5f907e9b9ae425d7436036981eb3d516a0e280efc2f6`
- v2 smoke：`outputs/pads_classification/smoke_v2_variable_length_20260821`
- smoke best checkpoint SHA256：`659541172b30e6c802b2bb6bd5d6bce0087094fb135bbdc8a2de8ed8d369b4d8`

第一次 `rho=1` 的求解器 smoke 按收敛门禁失败，失败目录 `outputs/preprocessing_smoke_v2_20260821` 被保留作审计，不可用于训练。其他 `outputs/preprocessing_*` 目录是小规模 smoke/benchmark；正式训练只应使用 `data/processed/pads_multi_activity/v2_l1_full_length`。

## 7. 后续建议

1. 下一轮正式实验应只比较 v1 与 v2 数据版本，保持模型、loss、frequency bands、split、seed 和调参预算一致。
2. 对 4 条明显时间戳间隙做单独敏感性实验；在用户批准前不要把插值或重采样混入当前 v2。
3. 对鲁棒异常点按 activity/channel 画分布并人工查看代表性波形，不要根据 MAD flag 自动删除。
4. 后续 activity 消融继续保留 11 个活动作为完整母集，从同一 v2 数据派生配置，避免重新预处理。
