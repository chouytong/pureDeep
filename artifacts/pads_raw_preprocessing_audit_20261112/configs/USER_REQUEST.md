# Phase 9：PADS Raw IMU Data Quality, Preprocessing and Subject-level Data Protocol Audit

## 角色设定

你是一名具有丰富研究经验的计算机科学与工程研究人员，研究方向包括：

- Wearable IMU signal processing
- Time-series analysis
- Digital biomarkers for Parkinson's disease
- Deep learning
- Sensor fusion
- Signal denoising
- Statistical signal analysis
- Subject-level machine learning evaluation

你需要对当前 PURE-DEEP 项目的 PADS 数据处理流程进行一次系统、深入、可复现的数据与预处理审查。

本轮不是普通代码 review。

你需要从原始传感器数据出发，沿着整个数据处理链，判断目前的预处理是否合理，是否存在可能损失 PD/DD 判别信息的步骤，以及是否存在数据质量、时间对齐、信号处理和数据划分问题。

最终目标：

**找到有真实数据依据、具有明确物理或统计解释、并且有可能改善 Frozen WSSL-STR 性能的预处理改进方向。**

不允许为了提高指标而随意删除信号、改变滤波或修改数据划分。

---

# 一、项目正式背景

GitHub：

https://github.com/chouytong/pureDeep

当前主线代码：

/home/zyt/deep_final/mainline

服务器项目：

/home/zyt/deep_final

原始数据：

/home/zyt/deep_final/data

当前正式实验：

/home/zyt/deep_final/artifacts/wssl_guided_optimization_phase8_20261111/

当前最佳模型：

Frozen WSSL-STR

正式 development 指标：

Accuracy = 0.745604
BA = 0.719644
AUROC = 0.759553
Macro-F1 = 0.706589
PD Recall = 0.782344
DD Recall = 0.656945

当前任务：

PADS 双腕 IMU 数据上的受试者级 PD/DD 二分类。

数据基本组成：

- 390名受试者。
- PD：276。
- DD：114。
- 11项活动。
- 双腕传感器。
- Acc XYZ + Gyro XYZ。
- 共6个通道。

DD构成：

- Other Movement Disorders：60。
- Essential Tremor：28。
- Atypical Parkinsonism：15。
- Multiple Sclerosis：11。

当前最佳模型结构：

- STR wrist encoder。
- Frozen HarNet10 external Acc representation。
- Wrist residual fusion。
- Bilateral fusion。
- Activity fusion。
- Subject Path + STR Path。

当前正式模型和历史checkpoint均必须保持不变。

---

# 二、本轮核心研究问题

重点研究以下问题。

H1：PADS原始IMU数据是否存在启动、结束阶段的设备噪声或无效记录？

H2：当前是否存在不合理的滤波、重采样、时间同步或坐标系处理？

H3：当前固定长度、裁剪、padding和mask是否损失有效运动信息？

H4：当前normalization是否正确，是否存在统计信息泄漏或个体运动信息被过度消除？

H5：是否存在可以通过更合理的预处理提高PD/DD判别能力的真实机会？

H6：不同活动是否需要不同的预处理策略？

H7：PD和DD的信号质量、活动完成情况、时序结构是否存在系统性差异？

H8：当前subject-level data split和整个preprocessing pipeline是否完全符合受试者级评估要求？

这些是假设，不是已经确认的错误。

不能因为某种预处理方法在其他HAR或PD论文中常用，就直接认定当前项目需要采用。

---

# 三、执行权限与安全边界

本轮分三个权限级别。

## Level 0：只读审计

允许：

- 读取原始数据。
- 读取当前预处理源码。
- 读取配置和日志。
- 读取历史实验报告。
- 读取数据manifest。
- 统计文件和数据质量。
- 分析现有预处理。
- 检查数据划分。
- 计算只读诊断统计。

禁止：

- 修改mainline。
- 修改原始数据。
- 覆盖历史文件。
- 修改正式checkpoint。
- 重新拟合normalization。
- 重新训练模型。
- 删除或修改原始信号。

## Level 1：独立诊断代码

允许：

- 在独立目录创建分析脚本。
- 生成统计表。
- 绘制信号图。
- 绘制频谱图。
- 生成质量报告。
- 生成候选预处理方案。
- 编写correctness tests。

禁止：

- 修改正式预处理入口。
- 修改正式数据。
- 自动将候选应用于正式模型。
- 训练模型。
- 根据validation结果反向修改诊断规则。

## Level 2：受控预处理对照实验

本轮默认禁止自动启动。

只有Level 0和Level 1全部完成，并生成完整报告后，等待用户明确批准，才能开展。

即使获批，也必须使用独立实验目录和固定subject-level splits。

本轮深度训练预算：

0。

---

# 四、首先完整审查项目代码

不要直接从原始数据开始处理。

首先阅读：

- AGENTS.md
- README.md
- CONTEXT_HANDOFF
- 最新mainline
- Phase 3B、3C、3D、3E报告
- Phase 4完整报告
- Phase 5最终报告
- Phase 6最终报告
- Phase 7最终报告
- Phase 8最终报告
- 历史data preparation报告
- 数据集manifest
- Dataset类
- DataLoader
- Preprocessing
- Normalization
- Training
- Evaluation
- Inference
- Cache generation
- Feature extraction

建立完整的数据流图：

Raw data
   ↓
File parsing
   ↓
Timestamp processing
   ↓
Sensor calibration / axis mapping
   ↓
Signal cleaning
   ↓
Resampling
   ↓
Filtering
   ↓
Activity segmentation
   ↓
Windowing / fixed length
   ↓
Padding / masking
   ↓
Normalization
   ↓
Dataset / DataLoader
   ↓
STR input
   ↓
HarNet raw Acc input
   ↓
Frozen WSSL-STR

必须为每一步标注：

1. 对应源码文件。
2. 函数名称。
3. 输入形状。
4. 输出形状。
5. 输入单位。
6. 输出单位。
7. 采样率。
8. 是否使用label。
9. 是否需要fit。
10. fit使用哪些subject IDs。
11. 是否在train/validation split前执行。
12. 是否存在潜在信息泄漏。

不能仅根据README推断。

必须以真实代码为准。

---

# 五、原始数据完整性审计

对原始数据进行只读检查。

## 5.1 文件级审计

统计：

- 总文件数量。
- 文件格式。
- 每种文件大小。
- 文件是否损坏。
- 是否有重复文件。
- 是否存在空文件。
- 是否存在缺失受试者。
- 是否存在命名不一致。
- 是否存在同一受试者多个版本。

生成：

raw_file_inventory.csv

## 5.2 受试者完整性

对每个受试者检查：

- 是否存在11项活动。
- 每项活动是否存在Left和Right。
- 每只腕是否包含6个通道。
- 每条记录是否具有有效时间戳。
- 是否存在重复活动。
- 是否存在活动缺失。
- 是否存在腕侧缺失。

生成：

subject_activity_completeness.csv

## 5.3 类别条件完整性

分别统计：

PD：
- 完整活动数量。
- 缺失活动比例。
- 缺失腕侧比例。
- 平均活动长度。
- 长度分布。

DD：
- 相同统计。

进一步按DD粗来源类别报告。

但不能把粗来源类别直接当成临床亚型解释。

特别检查：

缺失活动是否与PD/DD标签相关。

如果存在类别差异，必须区分：

- 数据采集缺失。
- 设备故障。
- 受试者未完成任务。
- 真实任务执行困难。

不能直接删除缺失活动患者。

---

# 六、原始IMU信号质量审计

这是本轮核心内容。

对每个活动、腕侧、通道进行信号质量检查。

## 6.1 基础统计

至少计算：

- Mean
- Median
- Standard deviation
- Variance
- Minimum
- Maximum
- Range
- IQR
- RMS
- MAD
- Zero fraction
- NaN fraction
- Inf fraction
- Constant segment ratio
- Unique value count
- Saturation candidate ratio

对Acc和Gyro分别报告。

不得仅凭标准差或Z-score删除异常样本。

## 6.2 物理单位检查

确认：

Acc单位：

- m/s²？
- g？
- 其他？

Gyro单位：

- rad/s？
- deg/s？
- 其他？

检查：

- 单位转换是否正确。
- 不同受试者是否使用相同单位。
- 是否存在单位混用。
- 是否存在量纲不合理的数据。
- 左右腕是否使用相同坐标约定。

不能仅凭数值范围猜测单位。

必须结合原始数据说明、代码和数据文档。

## 6.3 传感器异常

检查：

- 突然跳变。
- 传感器饱和。
- 数据截断。
- 长时间常数序列。
- 丢包。
- 重复采样点。
- 不规则时间戳。
- 多通道同步异常。
- 加速度重力分量异常。
- Gyro零漂。

异常检测需要输出原始样本位置和受影响记录。

不能自动删除。

## 6.4 真实运动峰值与传感器异常的区分

特别注意：

PD震颤、动作迟缓和运动不稳定可能产生真实的高频或高幅度信号。

不能将高幅值、低幅值、较大jerk直接判定为噪声。

需要结合：

- 时间连续性。
- 多通道一致性。
- 物理量程。
- 频谱结构。
- 左右腕关系。
- 其他活动记录。

对每类异常给出证据。

生成：

signal_quality_summary.csv

signal_anomaly_candidates.csv

signal_quality_by_activity.csv

signal_quality_by_class.csv

---

# 七、活动开始和结束阶段审计

重点审查用户提出的前几秒噪声问题。

不要预设必须删除前3秒或5秒。

## 7.1 分段统计

对每个活动和腕侧，将原始序列划分为：

- 前0.5秒。
- 前1秒。
- 前2秒。
- 前3秒。
- 前5秒。
- 中间稳定区间。
- 最后0.5秒。
- 最后1秒。
- 最后2秒。
- 最后3秒。
- 最后5秒。

以上仅用于诊断。

如果记录长度不足，标记NA，不补造数据。

## 7.2 计算时序指标

每个时间段计算：

- Acc magnitude。
- Gyro magnitude。
- RMS。
- Variance。
- MAD。
- Jerk。
- Spectral energy。
- Dominant frequency。
- Zero-crossing rate。
- Short-time energy。
- Autocorrelation。
- Signal-to-noise proxy。

## 7.3 PD/DD分层分析

分别比较：

- PD前段与中段。
- DD前段与中段。
- PD末段与中段。
- DD末段与中段。

检查是否存在：

1. 设备启动噪声。
2. 人体动作启动。
3. 姿势调整。
4. 稳定动作。
5. 动作停止。
6. 设备收尾。

不能仅根据PD/DD的平均曲线判断所有人的时间结构。

必须检查受试者级分布。

## 7.4 生成可视化

每个活动至少生成：

- Acc XYZ原始时序。
- Gyro XYZ原始时序。
- Acc magnitude。
- Gyro magnitude。
- RMS时间曲线。
- Jerk时间曲线。
- 频谱。
- PD/DD分组分布。

对于大量受试者：

采用代表性个体图和群体汇总图。

代表性个体只能按预先固定的ID选择，不得挑选最符合预期的曲线。

## 7.5 启动阶段删除候选

可以提出：

- 不删除。
- 删除前0.5秒。
- 删除前1秒。
- 删除前2秒。
- 删除前3秒。
- 活动自适应起始点。

但这些只是候选。

只有同时满足以下条件，才建议进入Level 2：

1. 有明确的传感器或采集协议证据支持无效区间。
2. 该区间在受试者间具有可重复性。
3. 不同PD/DD类别不存在明显的有价值启动信息差异，或者该差异已经被单独评估。
4. 删除不会改变活动定义。
5. 有固定的、独立于validation标签的确定性规则。

如果前几秒包含疾病相关动作启动信息，不应删除。

生成：

temporal_boundary_report.md

temporal_boundary_metrics.csv

temporal_boundary_figures/

---

# 八、采样率、时间戳与双腕同步审计

## 8.1 原始采样率

逐记录核对：

- 原始采样率。
- 时间戳间隔。
- 平均采样率。
- 中位采样率。
- 最大时间间隔。
- 时间戳抖动。
- 重复时间戳。
- 丢包率。

不要假设所有记录都是100Hz。

## 8.2 当前重采样流程

核对当前：

- 是否重采样。
- 目标采样率。
- 插值方式。
- 抗混叠滤波。
- 降采样实现。
- 边界处理。
- 是否存在频率混叠。

特别检查HarNet输入和STR输入是否使用不同采样率，以及差异是否有明确依据。

## 8.3 双腕同步

检查：

t_left - t_right

统计：

- 平均偏移。
- 中位偏移。
- 标准差。
- 最大绝对偏移。
- 不同活动的偏移。
- PD/DD的偏移分布。

如果没有可靠时间戳，不能假设左右腕完全同步。

不要自动使用DTW。

## 8.4 输出

sampling_rate_audit.csv

timestamp_quality.csv

bilateral_sync_report.csv

resampling_audit.md

---

# 九、滤波与频域处理审计

## 9.1 当前滤波方法

逐项核对：

- Filter type。
- Cutoff frequency。
- Filter order。
- Sampling frequency。
- Causal/non-causal。
- Zero-phase。
- Padding。
- Boundary handling。
- Numerical stability。

分别检查Acc和Gyro。

## 9.2 频域分析

计算：

- FFT。
- Welch PSD。
- Spectral centroid。
- Spectral bandwidth。
- Dominant frequency。
- Band energy。
- Spectral entropy。

如果有合理依据，可探索：

- Tremor-related band。
- Low-frequency movement。
- High-frequency noise。

但不得根据当前validation分类结果挑选频带。

频带设置应优先依据原始设备采样能力、物理知识和已发表研究。

## 9.3 滤波效果检查

对同一原始记录绘制：

- Raw signal。
- Current filtered signal。
- Candidate filtered signal。

比较：

- 幅值变化。
- 相位变化。
- 频谱变化。
- 边缘效应。
- 震颤频带能量损失。

不要为了让曲线更平滑就选择滤波器。

平滑并不等于信息保留。

输出：

filter_audit.md

filter_frequency_response.csv

spectral_quality_by_activity.csv

---

# 十、坐标系、重力分量和传感器物理解释

检查：

1. Acc是否包含重力。
2. 是否进行了重力去除。
3. 是否使用设备姿态融合。
4. X/Y/Z轴方向。
5. 左右腕坐标是否一致。
6. 是否存在轴向反转。
7. Gyro单位及方向。
8. 是否存在传感器安装方向差异。

如果没有原始设备坐标说明：

明确标记UNKNOWN。

不要通过模型指标选择轴向翻转方式。

如有多个可能的坐标变换：

只提出候选，不修改正式输入。

输出：

sensor_coordinate_audit.md

sensor_unit_audit.csv

---

# 十一、活动切分、固定长度和Padding审计

检查：

- 当前活动长度分布。
- 固定长度设置。
- 裁剪规则。
- Padding规则。
- Mask规则。
- 最后不足窗口的处理。
- 真实长度是否保留。
- Padding是否参与统计。
- 不同活动是否具有不同有效时长。

重点检查：

如果活动长短不一，当前固定长度策略是否导致：

- 有效信号被截断。
- 不同活动被过度padding。
- 动作结束信息丢失。
- 短记录受到特殊影响。

必须按活动分别分析。

输出：

activity_duration_report.csv

padding_mask_audit.md

sequence_truncation_report.csv

---

# 十二、Normalization审计

这是最重要的数据泄漏检查之一。

完整恢复：

- Normalization文件。
- 训练受试者。
- 拟合函数。
- 统计量。
- 适用通道。
- 适用活动。
- 适用腕侧。
- 适用模型分支。

检查：

1. 是否train-only fit。
2. 是否在split前fit。
3. 是否使用validation。
4. 是否混合不同split。
5. 是否对左右腕分别归一化。
6. 是否对不同活动分别归一化。
7. 是否存在个体级normalization。
8. 是否丢失幅度信息。
9. 是否存在HarNet和STR不同normalization规则。

特别分析：

全局Z-score、通道级Z-score、活动级Z-score、腕侧级Z-score、subject-level normalization之间的差异。

但本轮不重新拟合或替换正式normalization。

只提出有证据支持的候选。

输出：

normalization_provenance.md

normalization_statistics.csv

normalization_leakage_audit.md

---

# 十三、数据划分与潜在泄漏审计

必须检查整个pipeline的执行顺序。

正确原则：

Subject split
    ↓
Train-only fit
    ↓
Train transform
    ↓
Validation transform
    ↓
Model selection within development protocol

检查：

- 同一受试者是否跨集合。
- 同一活动记录是否跨集合。
- 不同窗口是否跨集合。
- 缓存是否包含不同split数据。
- Normalization是否在全队列fit。
- PCA是否在全队列fit。
- 特征选择是否读取validation标签。
- Augmentation是否发生在split前。
- 受试者重复appearance如何统计。
- Checkpoint selection是否读取outer。

需要明确区分：

- Label leakage。
- Subject leakage。
- Preprocessing leakage。
- Validation model-selection bias。
- Historical outer exposure。

历史outer接触偏差永久保留。

本轮不能宣称整个项目完全没有历史outer信息接触。

不访问outer-test performance。

不建立outer-test loader。

不利用outer信息进行任何新分析。

输出：

split_integrity_report.md

preprocessing_leakage_audit.md

subject_overlap_check.csv

---

# 十四、PD/DD类别条件数据质量比较

重点研究：

当前PD与DD的性能差距是否可能部分来自数据质量或活动完成情况差异。

分别比较：

- 活动时长。
- 信号完整率。
- 缺失通道比例。
- 缺失腕侧比例。
- 信号幅值。
- RMS。
- Jerk。
- 频域能量。
- 震颤相关统计。
- 启动阶段质量。
- 双腕同步质量。

DD进一步按照现有粗来源类别分组。

但是：

- 不将粗来源类别视为精确临床亚型。
- 不根据validation错误名单选择数据。
- 不将相关性解释为因果。
- 不根据结果删除DD患者。

需要检验：

某些数据质量指标是否与PD/DD类别有关？

某些活动是否在DD组中具有更高的缺失或异常率？

这些差异是否可能影响当前模型的分类结果？

本轮只做数据质量关联诊断。

不训练新的分类器。

输出：

class_conditional_data_quality.md

class_conditional_data_quality.csv

---

# 十五、预处理候选方案设计

只有完成前面所有审计后，才提出候选。

每个候选必须包含：

1. 当前处理方式。
2. 发现的问题。
3. 原始数据证据。
4. 物理或统计依据。
5. 修改内容。
6. 可能损失的信息。
7. 可能获得的收益。
8. 是否改变输入分布。
9. correctness tests。
10. 实验成本。
11. 停止条件。

候选方向可以包括：

A. 删除经证实无效的设备启动区间。

B. 改进抗混叠重采样。

C. 改进时间戳处理。

D. 改善双腕同步。

E. 修正不合理滤波。

F. 改善活动长度与padding。

G. 改善normalization。

H. 对不同活动采用有物理依据的不同预处理。

但不能为了覆盖所有方向而强行提出修改。

没有证据的方向标记 NO_ACTION。

---

# 十六、Level 2未来预处理对照实验的设计要求

本轮不执行Level 2。

但必须为未来实验准备规范。

如果某个预处理候选通过审计：

每次只修改一个主要因素。

例如：

PREP-BASE：
当前正式预处理。

PREP-START：
仅修改启动阶段处理。

PREP-FILTER：
仅修改滤波。

PREP-RESAMPLE：
仅修改重采样。

PREP-SYNC：
仅修改双腕同步。

PREP-NORM：
仅修改normalization。

不得将多个改动组合在同一次实验后，无法判断真正有效因素。

所有候选：

- 同一subject-level splits。
- 同一seed集合。
- 同一Frozen WSSL-STR训练recipe。
- 同一checkpoint selection规则。
- 同一0.5决策规则。
- 同一评价指标。
- 预先固定保留标准。

只有通过单因素实验的改动，才允许进入后续组合实验。

如果一个预处理修改改善Accuracy但损害DD Recall，不能自动保留。

---

# 十七、必须生成的报告和产物

独立目录：

/home/zyt/deep_final/artifacts/pads_raw_preprocessing_audit_20261112/

必须生成：

1. FINAL_REPORT.md
2. DATA_PIPELINE_MAP.md
3. RAW_DATA_INTEGRITY_REPORT.md
4. SIGNAL_QUALITY_REPORT.md
5. TEMPORAL_BOUNDARY_REPORT.md
6. SAMPLING_RATE_AND_TIMESTAMP_REPORT.md
7. BILATERAL_SYNCHRONIZATION_REPORT.md
8. FILTER_AND_SPECTRAL_AUDIT.md
9. SENSOR_COORDINATE_AND_UNIT_AUDIT.md
10. ACTIVITY_SEGMENTATION_REPORT.md
11. NORMALIZATION_PROVENANCE_REPORT.md
12. DATA_SPLIT_AND_LEAKAGE_AUDIT.md
13. CLASS_CONDITIONAL_DATA_QUALITY_REPORT.md
14. PREPROCESSING_CANDIDATES.md
15. LEVEL2_EXPERIMENT_PROTOCOL.md
16. CORRECTNESS_AND_PROVENANCE.md

同时生成：

analysis/raw_file_inventory.csv

analysis/subject_activity_completeness.csv

analysis/signal_quality_summary.csv

analysis/signal_anomaly_candidates.csv

analysis/signal_quality_by_activity.csv

analysis/signal_quality_by_class.csv

analysis/temporal_boundary_metrics.csv

analysis/sampling_rate_audit.csv

analysis/timestamp_quality.csv

analysis/bilateral_sync_report.csv

analysis/filter_frequency_response.csv

analysis/spectral_quality_by_activity.csv

analysis/sensor_unit_audit.csv

analysis/activity_duration_report.csv

analysis/padding_mask_audit.csv

analysis/normalization_statistics.csv

analysis/subject_overlap_check.csv

analysis/class_conditional_data_quality.csv

analysis/preprocessing_candidate_priority.csv

analysis/phase9_integrity.json

所有图像保存在：

figures/

---

# 十八、数据安全和工程保护

原始数据只读。

不得修改：

/home/zyt/deep_final/mainline

不得覆盖：

- Frozen WSSL-STR checkpoint。
- 历史预处理配置。
- 历史normalization。
- 历史实验。
- 原始数据。
- 历史正式报告。

禁止上传：

- 原始IMU数据。
- 私有manifest。
- 受试者ID。
- 逐受试者统计。
- 私有信号图。
- checkpoint。
- feature cache。
- 受保护文件。

对所有受保护文件建立哈希清单。

最终确认审计前后哈希一致。

公开发布时只允许：

- 通用分析代码。
- 不含个体信息的协议。
- 不可反推出受试者身份的聚合统计。
- 研究方法报告。

所有统计报告必须披露：

- 数据来源。
- 统计单位。
- 受试者数量。
- split数量。
- 缺失值处理。
- 多重比较方法。
- 是否接触validation信息。
- 是否属于历史重复development。

---

# 十九、最终必须回答的研究问题

请以严格审稿人的角度给出最终判断。

1. 当前PADS数据是否存在明确的原始数据质量问题？
2. 前几秒是否真的存在设备噪声？
3. 该问题是否在所有11项活动中一致？
4. PD和DD是否存在不同的启动阶段运动特征？
5. 是否应该删除前0.5、1、2、3或5秒？
6. 当前采样率和重采样流程是否合理？
7. 是否存在抗混叠不足？
8. 当前滤波是否损失震颤相关信息？
9. 双腕是否同步？
10. 左右腕坐标系是否一致？
11. 当前固定长度是否截断重要运动？
12. Padding和mask是否正确？
13. Normalization是否可能损失疾病信息？
14. 是否存在统计信息泄漏？
15. 哪些预处理步骤是正确且应该保留的？
16. 哪些步骤值得开展独立对照？
17. 哪些修改可能真正改善Frozen WSSL-STR？
18. 哪些方向不值得继续投入？

最后给出：

- 当前预处理流程图。
- 已确认问题。
- 潜在问题。
- 无问题且应保留的步骤。
- 预处理候选优先级。
- Level 2实验设计。
- 每项候选的预期收益、风险和停止条件。

必须明确区分：

FACT：实际代码或数据验证的事实。

EVIDENCE：具有统计证据的观察。

HYPOTHESIS：合理但尚未验证的解释。

UNKNOWN：缺少必要数据或元信息。

NO_ACTION：目前没有证据支持修改。

---

# 二十、执行结束条件

本轮只完成Level 0和Level 1。

禁止：

- 自动删除原始信号前几秒。
- 自动修改滤波。
- 自动修改重采样。
- 自动修改normalization。
- 自动重新切分正式数据。
- 自动重新训练Frozen WSSL-STR。
- 自动执行预处理消融。
- 自动访问outer-test。
- 自动启动下一阶段。

最终报告完成后停止。

等待用户审查和批准后，才允许开展Level 2。

**本轮最重要的目标不是找到一种看起来更先进的滤波器，而是确认当前PADS原始数据经过的每一步处理是否合理，并找到有充分数据证据支持的预处理改进方向。**