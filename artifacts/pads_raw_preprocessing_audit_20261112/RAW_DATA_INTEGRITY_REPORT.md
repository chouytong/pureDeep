# 原始数据完整性

FACT：任务子集为390人，PD276/DD114；DD粗来源Other60、ET28、Atypical15、MS11。原始目录只存此390人任务子集，不以原库469人减去本地390人来报告“79人丢失”。390患者JSON、390 observation JSON、8580 TXT均解析。每人11活动、每活动左右腕各1条；录制文件完整不等于已验证患者动作完成正确，缺少视频/逐任务完成评分。未发现空文件、精确字节重复、缺失活动/腕侧、重复session或完整六轴相邻重复点；实际轴定义为Time加6传感器轴。

| format | file_count | total_bytes | minimum_bytes | median_bytes | maximum_bytes | empty_count | exact_duplicate_group_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| .json | 780 | 5142632 | 365 | 6631 | 12790 | 0 | 0 |
| .txt | 8580 | 1052861729 | 95031 | 96364 | 195459 | 0 | 0 |


文件路径、逐文件hash、每条大小、逐人完整性和schema明细位于restricted_manifests/analysis/同名表。公开analysis/raw_file_inventory.csv按格式汇总、subject_activity_completeness.csv按活动/类别汇总；不含个体ID或filehash。所有预期TXT与实际路径集合完全一致，因此没有额外版本TXT被静默忽略；命名额外检查见最终correctness。metadata字段单位全一致不代表已验证硬件校准。


---
数据来源：服务器PADS v1.0.0的390人PD/DD任务子集，原始数据、既有v2信号及45 Frozen WSSL-STR checkpoint元数据；不是原库全部469人。统计单位为患者，同患者双腕先平均；原始记录/通道数只作描述性计数，不作为独立患者。15重叠development splits、seeds42/43/44用于来源核查，不是独立外部验证。缺失/不可计算值为NA，不补造；频谱超过记录Nyquist不外推；稀有交叉分层<10人的数值统计抑制，完整表仅私有。检验族132/440/880分别BH，患者bootstrap2000次；其余描述性表无额外显著性声明。接触全390人诊断标签及validation成员元数据用于质量关联，未读分类错误名单/分数，未构建outer loader、未解码outer性能/预测；历史outer接触永久披露。全队列探索会增加未来选择偏差，不能叫盲态独立验证。训练/分类器拟合/normalization重拟合均0，Level2未执行。
