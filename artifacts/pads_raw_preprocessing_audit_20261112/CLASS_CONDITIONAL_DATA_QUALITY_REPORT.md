# PD/DD与粗来源的质量关联

预先固定12指标×11活动=132个患者级Mann–Whitney，双腕先平均，DD−PD效果量、患者bootstrap2000 CI和BH132均保留。37项q<.05，不表示37种设备故障；RMS/jerk/3–12Hz能量可能同时包含真实疾病运动、任务执行和采集因素。所有组活动/腕完整率均100%，没有证据把DD识别差归咎于缺失活动。

| activity | planned_tests | BH05_count |
| --- | --- | --- |
| CrossArms | 12 | 4 |
| DrinkGlas | 12 | 5 |
| Entrainment | 12 | 1 |
| HoldWeight | 12 | 6 |
| LiftHold | 12 | 5 |
| PointFinger | 12 | 3 |
| Relaxed | 12 | 0 |
| RelaxedTask | 12 | 2 |
| StretchHold | 12 | 5 |
| TouchIndex | 12 | 4 |
| TouchNose | 12 | 2 |


| device | DD | PD | association_only_no_causal_claim |
| --- | --- | --- | --- |
| Apple Watch Series 3 | 11 | 93 | True |
| Apple Watch Series 4 | 103 | 183 | True |


设备类别不平衡是明确的潜在混杂：Series3占PD的93/276而DD仅11/114。device_quality_descriptive.csv与source×device交叉表用于背景，但小于10人交叉格的数值已抑制，完整交叉表仅私有。不拟合“预测疾病/预测设备”的新分类器，不按错误名单分析，不把相关性说成当前模型利用的因果证据。没有单次质控剔除或针对DD的数据清理。


---
数据来源：服务器PADS v1.0.0的390人PD/DD任务子集，原始数据、既有v2信号及45 Frozen WSSL-STR checkpoint元数据；不是原库全部469人。统计单位为患者，同患者双腕先平均；原始记录/通道数只作描述性计数，不作为独立患者。15重叠development splits、seeds42/43/44用于来源核查，不是独立外部验证。缺失/不可计算值为NA，不补造；频谱超过记录Nyquist不外推；稀有交叉分层<10人的数值统计抑制，完整表仅私有。检验族132/440/880分别BH，患者bootstrap2000次；其余描述性表无额外显著性声明。接触全390人诊断标签及validation成员元数据用于质量关联，未读分类错误名单/分数，未构建outer loader、未解码outer性能/预测；历史outer接触永久披露。全队列探索会增加未来选择偏差，不能叫盲态独立验证。训练/分类器拟合/normalization重拟合均0，Level2未执行。


## 发布前数值正确性修订

本版结合[L1求解器证据](L1_SOLVER_CORRECTNESS_REPORT.md)。此前Phase9发布前草稿与全部源码/聚合输出保存在audit/pre_solver_witness_snapshot；旧CLOSURE_FREEZE按该版本命名空间核验，替代关系见configs/PREPUBLICATION_REVISION.json。正式Phase3–8、数据、模型及normalization均未改；原始质量关联表的p/q未改变。

**修订结论优先于此前逐记录独立假设**：既有L1有限迭代使用全batch共同停止，数值输出可随其他记录变化；真实输出保守目标值证据见新报告。此发现不改原始信号质量统计，不表示标签泄漏或已证明模型性能虚高。
