# WSSL 现行数据预处理审查

日期：2026-10-04。范围为 retained WSSL 的现行代码、冻结 development 资产和独立只读测试；不训练新模型，不访问 outer predictions/outcomes，不改变历史正式结果。

## 1. 审查对象

服务器正式项目 `/home/zyt/deep_final`，原始/预处理数据项目 `/home/zyt/MFAM`。

现行代码快照见本轮 `current_source/`：

- `/home/zyt/MFAM/scripts/prepare_multi_activity_data_v2.py`；
- `/home/zyt/deep_final/foundation_validation/src/datasets/{preprocessing,manifest,subject_activity,builders,folds}.py`；
- `/home/zyt/deep_final/foundation_validation/src/models/pure_deep.py`；
- `/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928/scripts/{extract_ssl,transfer_hooks}.py`。

历史独立窗口重提取的辅助证据为 `/home/zyt/deep_final/artifacts/phase4_window_dynamics_20261002/scripts/extract_windows.py`，其 exact mean-reconstruction check 覆盖旧 frozen cache；本轮不引入该研究中的新增 window branch。

## 2. 两条输入路径的实际契约

| 环节 | STR local 路径 | frozen HarNet SSL 路径 |
|---|---|---|
| 原始信息 | 双腕 Acc XYZ + Gyro XYZ | 同 subject/activity/wrist 的 raw Acc XYZ |
| 原始单位 | metadata：Acc g、Gyro rad/s、Time s | metadata 已为 g，无 `m/s²→g` 乘除因子 |
| 原始采样 | metadata 标称 100 Hz | 同一 raw 记录标称 100 Hz |
| 去趋势 | 对完整 raw Acc 解 `0.5‖y−x‖²+50‖D²x‖₁`，输出 raw−trend | 不使用 L1-detrended Acc，不使用 STR-normalized Acc |
| Gyro | 保留原值 | 不进入 HarNet |
| trim | 完成 Acc 去趋势后，删除前 48 点 | raw Acc 固定删除前 48 点；对应相同有效区间 |
| 有效长度 | 976 或 2000 点，全部进入 local encoder | 976：左右各 edge-pad 12 至 1000；2000：固定前/后各 1000 点 |
| clip/filter | 无新增 clip/filter；solver 结果保存 float32 | 先逐轴 clip 到 ±3g，再 `scipy.signal.resample_poly(segment,3,10,axis=0)` |
| HarNet 输入 | 不适用 | `[3,300]`、标称 30 Hz/10 s；两个窗口的 1024D features 算术均值 |
| 标准化 | train-only `[11,2,6,1]` mean/std | 不用数据集/受试者/fold拟合 normalization；分类器内逐1024D向量 LayerNorm |

实际 v2 solver 运行配置为 λ50、rho40、最多5000次、atol/rtol1e−4。`l1_trend_filter_batch` 函数默认 rho1/max2000 不是本次正式预处理的有效配置，不能仅查看默认签名后认定正式流程未收敛。

缓存提取每个 HarNet 都调用 `eval()`，参数 `requires_grad=False`，推理在 `inference_mode()` 下进行；没有用 PADS train/validation 重新估计 HarNet 运行统计。特征缓存可一次包含 390 个 development subjects，因为外部 encoder 和所有变换是固定的逐记录映射，不从这些受试者估计参数、选择窗口或计算 pooled statistics。缓存“包含 validation subject 的 features”不能单独作为 leakage 的证据；真正需要检查的是是否有跨subject拟合、验证集参与选择或 HarNet train-mode 状态更新。

## 3. 顺序、ID、normalization 与 padding

活动固定顺序：CrossArms、DrinkGlas、Entrainment、HoldWeight、LiftHold、PointFinger、Relaxed、RelaxedTask、StretchHold、TouchIndex、TouchNose。

左右腕固定 `[left,right]`，每腕通道固定 `[AccX,AccY,AccZ,GyroX,GyroY,GyroZ]`；PD=0，DD=1。缓存以字符串 subject ID 建 index，`cache.batch(subject_ids,...)` 显式查ID，再取相应 `[11,2,1024]`，不是通过 minibatch数组位置匹配subject。活动维度仍依赖正式固定顺序，因此需要同时检查cache.activities、config.data.activities及manifest活动选择。

`build_subject_fold_datasets`先检查 subject-role disjointness；`_select_records`只把train IDs传给raw train dataset；`_activity_statistics`只遍历该raw train dataset的真实完整序列。统计float64累加后保存float32，方差使用总体分母，不是无偏估计。validation复用同一mean/std，不进行 per-subject normalization。相同split的三个seed应使用相同normalization；本轮根任务已完成15真实fold重新拟合与45checkpoints核验，mean/std全部逐位一致，记录为 `analysis/assets_normalization_audit.json` 的PASS；本子任务未重复重算。

`sequence_length=null`、train/eval crop均full。SubjectActivityDataset先对真实长度序列标准化，再以0右侧补齐batch容器；缺失腕在标准化后清零并提供wrist_mask。`activity_lengths`保留实际长度；PureDeepSubjectModel按有效activity真实长度分组、切片，再编码有效wrist，所以有限值的 padded tail 不参与 local encoder。所有活动缺失、重复subject/activity、标签不一致、左右腕长度/通道不一致均有拒绝检查。

## 4. 已发现的质量边界

现行资产描述：8580个腕侧文件均finite，时间戳严格递增；275个solver batch全部converged，最大4538次。有效采样率约99.2065–100.8077 Hz，4条记录存在超过10倍median dt的timestamp gap，最大约0.646577 s。根任务从服务器现行asset读取该描述；本轮脚本已独立重新核验全部raw/processed记录，结果与质量描述一致。

重要实现事实：当前local路径按样本索引处理；SSL `resample_poly`也按均匀采样索引运行，**没有按照每条原始时间戳插值到均匀时间网格**。因此“anti-alias resampling”成立，“按真实timestamp修正不规则采样”不成立。左右腕时钟偏移被记录，没有在sample-level进行同步矫正；当前网络为两腕独立编码后的feature-level融合。

这属于已知采样近似及质量限制，尚不能从这些flags推出性能偏差大小、临床phenotype被改变、特定病类受到影响，或提出已证实的最佳timestamp correction。MAD异常点同样是审计标记，不能自动作为错误值删除。

clip发生在polyphase filter之前；滤波输出可能有轻微overshoot，因此不能把“输入clip±3g”写成“所有resampled输出必在±3g”。未使用随机time reversal、permutation或幅度augmentation。976点的edge-padding重复边界，2000点前后窗口平均；这些是已冻结的映射，不能宣称临床最优，也不能通过本轮validation选择新的窗口策略。

目前代码审查未发现需改写正式输入的实质bug。物理单位由metadata声明核验，无法证明逐设备校准精度；算法求解收敛和finite不能证明每种运动病理语义完整保留。

## 5. 独立只读测试与执行边界

新增 `scripts/test_preprocessing_contract.py`。该脚本：

1. 只由15固定inner train/validation IDs的并集确定development subjects；不读取outer test IDs、predictions或outcomes。
2. 遍历390个development observations，逐个验证11活动×2腕的units、channels、device_location和标称100Hz，避免以单个metadata示例推断全数据。
3. 遍历8580 wrists：raw/processed有限、时间戳单调、post-trim976/2000、processed Gyro与raw[48:]逐点float32一致。
4. 按正式算法重建全部HarNet输入，核对resampled-window SHA、clip计数、cache SHA、subject IDs/activity order和window_counts。此项没有重新运行HarNet；feature bytes凭正式提取hash核验，不能写成“本轮独立重跑HarNet”。
5. 用明确synthetic小样例验证 train-only mean/std、subject标签、活动顺序、wrist mask、collate右侧padding，以及Acc去趋势不改变Gyro。synthetic测试只验证实现契约，不支持临床有效性或模型性能结论。
6. 只输出aggregate审计和合成fixture，不修改正式signals、cache、split、normalization、checkpoint或预测。

静态编译和服务器完整执行均 **PASS**，证据为本轮 `analysis/preprocessing_contract.json`：

| 核验项 | 本轮实测 |
|---|---:|
| development metadata observations | 390 |
| 活动记录 / 腕侧文件 | 4290 / 8580 |
| 976 / 2000 点腕侧序列 | 6240 / 2340 |
| 固定 HarNet 输入窗口 | 10920 |
| processed Gyro == raw `[48:]` | 8580 wrists 全部逐点float32一致 |
| 原始 metadata单位、通道、腕侧、100Hz、文件名映射 | 全部一致 |
| 缓存ID、activity顺序、window counts和输入SHA | 全部一致 |
| clip前超过±3g的Acc标量数 | 2044 / 32310720（0.006326%） |
| 实测有效采样率 | 99.206547–100.807653 Hz |
| 超过10倍median dt的gap记录 | 4 |
| 最大gap / median dt | 64.407448 |
| 最大左右腕时钟偏移 | 0.731371 s |
| 合成train-only normalization / masks / padding / Gyro保持测试 | 全部PASS |

Frozen cache SHA256：`f87829558938c3e50f78e53fe2f54156fc0f3768be3f73f97b089c0587c25376`；重建全部resampled-window输入SHA256：`7f77341dafe0225910cc736d80898b33fe8f098e8b9649693e1eaa78e353b0fe`，均与正式资产一致。

本轮没有重新运行HarNet，也没有独立重新求解8580份Acc去趋势。Acc处理链的依据是实际处理实现、有效运行配置、275批次收敛记录及现行资产；不得把输入SHA和Gyro一致性扩大解释为完整Acc solver逐位重算。15实际train fold normalization则由根任务独立重算，45个checkpoint mean/std逐位一致。

全部测试未训练模型，未修改正式inputs，未访问outer predictions/outcomes。PASS支持数据契约及资产复现，不支持宣称临床语义最优、每种phenotype无损或现行处理是性能最优。

## 6. 当前审查判断

- 两条输入路径有意保留不同信息：local Acc去趋势+Gyro，SSL raw Acc固定外部特征。不能把这种差异误报为输入单位错误或normalized cache泄漏。
- 主要有待核验的是完整资产契约、采样近似和物理metadata一致性，只读测试已对完整development资产通过。
- 不能据现有质量flags指定新的“最优”预处理。测试若失败，应先区分测试假设错误、资产损坏与正式流程问题并报告；不能自行修正历史正式结果或启动新的模型搜索。
