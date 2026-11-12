# 独立数据重建

**PASS**。390授权PD/DD development成员，11活动×双腕=8580记录/25740 Acc轴，无患者删除、失败轴或旧新混合。raw1024/2048逐条确认输出976/2000，6通道float32与原顺序。完整Acc求解→raw−trend→trim48→float32；Gyro仅trim48转float32，全量精确相等。原raw和legacy信号只读，不调整timestamp、同步、坐标或gravity。

版本`pads_l1_lambda50_certified_v1`；manifest SHA `cdc5e26db77ccb101c7157fc1a7aacde132de9d5e6d5449203f8bf84aa393f16`。真实信号/全长trend和dual/证书/ID映射存restricted_manifests。逐记录raw/source/config/cert/output hash、原子写入和完成标记齐全；部分失败不生成完整成功门槛。不完整文件保留并阻断，不静默覆盖。完整版本再次执行build-data只经hash核验复用。

Acc单位g、通道和原文件可核验；是否含完整重力向量、SDK导出前变换、左右腕物理轴/绝对同步仍UNKNOWN。本次只修复固定索引L1目标，不声称物理重力去除，不改变HarNet输入。
