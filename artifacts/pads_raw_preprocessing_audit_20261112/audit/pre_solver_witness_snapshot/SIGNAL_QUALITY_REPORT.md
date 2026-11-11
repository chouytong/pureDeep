# 六轴信号质量

FACT：8580腕记录、51480轴记录，原始与处理信号无NaN/Inf；所有Gyro处理值与raw[48:]转float32逐点相等。51,480轴分别计算均值、中位、SD、方差、min/max/range/IQR/RMS/MAD、零比例、非有限比例、常数段、unique值和极端重复候选。最长完全常数run为3点，不据低幅或jerk判错。

| type | flagged_wrist_records | affected_subject_count | channel_candidate_rows | interpretation |
| --- | --- | --- | --- | --- |
| gap_over10median | 4 | 4 | 5 | candidate only; no automatic rejection |
| robust_derivative_candidate | 6426 | 390 | 18419 | candidate only; no automatic rejection |
| robust_level_candidate | 4467 | 389 | 12486 | candidate only; no automatic rejection |


20×scaledMAD幅值/差分flag只是候选，真实震颤、姿势变化和低幅量化均可能触发。signal_anomaly_evidence.csv同时报告多通道并发和时间间隙背景；多通道共变既可能为运动，也可能共模伪迹，不能据此自动裁剪。原始位置、时间、受影响记录及全部通道特征只存私有同名表。硬件满量程UNKNOWN，极端重复候选不等于饱和；HarNet±3g是适配裁剪，不是设备量程。Gyro均值或静息活动RMS不能在缺少静止参考时充作零漂校准。high_to_low_proxy只是频段比，没有参考噪声时不能当真实SNR或临床信号质量标签。


---
数据来源：服务器PADS v1.0.0的390人PD/DD任务子集，原始数据、既有v2信号及45 Frozen WSSL-STR checkpoint元数据；不是原库全部469人。统计单位为患者，同患者双腕先平均；原始记录/通道数只作描述性计数，不作为独立患者。15重叠development splits、seeds42/43/44用于来源核查，不是独立外部验证。缺失/不可计算值为NA，不补造；频谱超过记录Nyquist不外推；稀有交叉分层<10人的数值统计抑制，完整表仅私有。检验族132/440/880分别BH，患者bootstrap2000次；其余描述性表无额外显著性声明。接触全390人诊断标签及validation成员元数据用于质量关联，未读分类错误名单/分数，未构建outer loader、未解码outer性能/预测；历史outer接触永久披露。全队列探索会增加未来选择偏差，不能叫盲态独立验证。训练/分类器拟合/normalization重拟合均0，Level2未执行。
