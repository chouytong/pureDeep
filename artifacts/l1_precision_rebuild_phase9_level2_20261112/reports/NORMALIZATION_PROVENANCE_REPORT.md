# 15份train-only normalization

PASS。逐split调用原mainline `_activity_statistics`，完整实际样本、不含padding，float64 sum/square后按原float32格式，[11,2,6,1]活动/腕/通道，epsilon1e−6。15份各自只fit inner-train成员，三个seed及两模型共享各split同一份。成员集合、数据版本、manifest SHA和norm SHA绑定检查；旧norm或错误版本直接拒绝，非train读取负测试通过。没有全队列fit/validationfit/每人归一化。

| sensor | mean_abs_mean_change | maximum_abs_mean_change | minimum_std_ratio | mean_std_ratio | maximum_std_ratio |
| --- | --- | --- | --- | --- | --- |
| Acc | 4.5062e-05 | 0.000212245 | 0.996646 | 1.062 | 1.43455 |
| Gyro | 0 | 0 | 1 | 1 | 1 |


Acc统计改变是精确执行同一目标的处理后果；Gyro未变。std比值只是数值分布变化，不表示诊断信息必然增加。完整拟合值及train IDs仅私有，公开按活动/腕/通道聚合的新旧变化。
