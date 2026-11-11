# 活动长度、窗口和mask

FACT：Entrainment/Relaxed/RelaxedTask原2048点，其余8活动1024点；现有裁48后为2000/976点。STR sequence_length:null、crop:full，保留全部剩余长度；没有额外固定1024截断。11活动顺序与left/right顺序固定，不把window或activity当患者。

| activity | record_count | true_samples | container_padding_fraction | HarNet_padded_samples | HarNet_contexts | HarNet_coverage_fraction | post48_tail_lost_samples |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CrossArms | 780 | 976 | 0.512 | 24 | 1 | 1 | 0 |
| DrinkGlas | 780 | 976 | 0.512 | 24 | 1 | 1 | 0 |
| Entrainment | 780 | 2000 | 0 | 0 | 2 | 1 | 0 |
| HoldWeight | 780 | 976 | 0.512 | 24 | 1 | 1 | 0 |
| LiftHold | 780 | 976 | 0.512 | 24 | 1 | 1 | 0 |
| PointFinger | 780 | 976 | 0.512 | 24 | 1 | 1 | 0 |
| Relaxed | 780 | 2000 | 0 | 0 | 2 | 1 | 0 |
| RelaxedTask | 780 | 2000 | 0 | 0 | 2 | 1 | 0 |
| StretchHold | 780 | 976 | 0.512 | 24 | 1 | 1 | 0 |
| TouchIndex | 780 | 976 | 0.512 | 24 | 1 | 1 | 0 |
| TouchNose | 780 | 976 | 0.512 | 24 | 1 | 1 | 0 |


976点在container补到2000会有51.2%占位，但归一化在真实数据上完成，encoder按activity_lengths切片后才编码；它不是51.2%的虚假信号进入模型。合成输入改变padding、屏蔽活动及腕侧占位后logits逐位相同。原始元数据6传感器加Time已纠正schema，不把它计成缺失通道。

HarNet短记录976点前后edge-pad共24点到1000；长记录2000点取first1000/last1000，完全覆盖剩余记录且没有重叠/尾部截断，2窗口特征取均值丢失窗口先后关系但不生成独立患者。当前无明显尾部丢失证据，不建议额外固定长度/删除未完成活动者。学习型temporal pooling不等于已证明所有起止信息均保留。


---
数据来源：服务器PADS v1.0.0的390人PD/DD任务子集，原始数据、既有v2信号及45 Frozen WSSL-STR checkpoint元数据；不是原库全部469人。统计单位为患者，同患者双腕先平均；原始记录/通道数只作描述性计数，不作为独立患者。15重叠development splits、seeds42/43/44用于来源核查，不是独立外部验证。缺失/不可计算值为NA，不补造；频谱超过记录Nyquist不外推；稀有交叉分层<10人的数值统计抑制，完整表仅私有。检验族132/440/880分别BH，患者bootstrap2000次；其余描述性表无额外显著性声明。接触全390人诊断标签及validation成员元数据用于质量关联，未读分类错误名单/分数，未构建outer loader、未解码outer性能/预测；历史outer接触永久披露。全队列探索会增加未来选择偏差，不能叫盲态独立验证。训练/分类器拟合/normalization重拟合均0，Level2未执行。
