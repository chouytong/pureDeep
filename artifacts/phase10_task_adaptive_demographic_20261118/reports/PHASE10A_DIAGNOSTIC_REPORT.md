# Phase10A 完成

| condition | runs | splits | accuracy | ba | auroc | macro_f1 | pd_recall | dd_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0-DEMOGRAPHIC-ONLY | 15 | 15 | 0.602816 | 0.574716 | 0.60788 | 0.560924 | 0.642318 | 0.507114 |
| STR-CERTIFIED-L1 | 45 | 15 | 0.717855 | 0.686585 | 0.712168 | 0.673253 | 0.761759 | 0.61141 |
| WSSL-CERTIFIED-L1 | 45 | 15 | 0.741534 | 0.7189 | 0.759285 | 0.703561 | 0.773894 | 0.663905 |


DD高置信错误appearance=135，边界appearance=516；均有同人/seed重复，不是独立人数。

| source_category | validation_count | recall | ci_low | ci_high |
| --- | --- | --- | --- | --- |
| Atypical Parkinsonism | 15 | 0.6 | 0.357468 | 0.801755 |
| Essential Tremor | 28 | 0.785714 | 0.604614 | 0.897875 |
| Multiple Sclerosis | 11 | 0.636364 | 0.353801 | 0.848335 |
| Other Movement Disorders | 60 | 0.483333 | 0.36175 | 0.606922 |


当前DD错误不能只由极小单一粗来源类解释与否由固定gate逐项记录；不据错误名单赋权。合法Certified OOF不存在，阈值部署研究BLOCKED；只保存validation描述曲线。四人口字段完整且定义/单位有来源，D0实际15fit已完成；年龄/性别/设备混杂用聚合分布及关联报告。

10B：可执行已授权FT-L1 pilot；它复核同一末stage适配并非结构创新。10C：字段门槛PASS。10D：满足固定启动诊断，待B/C报告后执行。

Phase8 legacy AI不混入Certified主比较；已有历史结果仅在协议中作为既有PD/DD取舍引用，本轮不解码旧整队列缓存。

固定15个重叠development splits；同人重复出现。三seed不是独立患者队列。人口记录和逐人预测不公开，正式阈值/模型指针不改。历史outer文字接触和多轮development选择偏差保留；本轮未读取outer性能。
