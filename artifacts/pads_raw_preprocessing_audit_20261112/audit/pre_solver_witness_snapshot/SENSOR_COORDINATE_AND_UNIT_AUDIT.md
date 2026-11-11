# 物理单位与坐标

FACT：8580元数据记录一致为Time(s)、AccXYZ(g)、GyroXYZ(rad/s)，无m/s²或deg/s转换。单位g由元数据确认，不能由此确认是否包含完整重力向量。官方论文/脚本称L1用于重力趋势处理，这是算法意图；本地原始Acc幅值普遍远低于1g，采集SDK字段及导出前变换未核实。重力语义标UNKNOWN，而非断言已去重力或一定保留重力；详见ACCELERATION_SEMANTICS_ADDENDUM.md。当前没有执行新的姿态融合、安装方向校准或左右轴翻转。

UNKNOWN：X/Y/Z的具体佩戴方向、左右物理坐标等价、安装角误差、硬件fullscale/精确零漂标定。左右都叫XYZ不证明两腕物理朝向相同；Gyro符号也不能通过模型成绩选择。设备metadata包括Series3(104人)/Series4(286人)，与只描述Series4的官方概述存在范围差异；这是metadata事实，硬件日志未验证，不擅自改写设备字段。

| device | DD | PD | association_only_no_causal_claim |
| --- | --- | --- | --- |
| Apple Watch Series 3 | 11 | 93 | True |
| Apple Watch Series 4 | 103 | 183 | True |


EVIDENCE：Series3中PD93/DD11，Series4中PD183/DD103，提示设备与诊断类别混杂，可能影响统计可分性/泛化，但未检验模型是否依赖设备。单位metadata一致不能排除校准差异。NO_ACTION：不自动换单位、旋转/镜像或校准；先补充硬件/佩戴协议，设备分层质量只作描述。


---
数据来源：服务器PADS v1.0.0的390人PD/DD任务子集，原始数据、既有v2信号及45 Frozen WSSL-STR checkpoint元数据；不是原库全部469人。统计单位为患者，同患者双腕先平均；原始记录/通道数只作描述性计数，不作为独立患者。15重叠development splits、seeds42/43/44用于来源核查，不是独立外部验证。缺失/不可计算值为NA，不补造；频谱超过记录Nyquist不外推；稀有交叉分层<10人的数值统计抑制，完整表仅私有。检验族132/440/880分别BH，患者bootstrap2000次；其余描述性表无额外显著性声明。接触全390人诊断标签及validation成员元数据用于质量关联，未读分类错误名单/分数，未构建outer loader、未解码outer性能/预测；历史outer接触永久披露。全队列探索会增加未来选择偏差，不能叫盲态独立验证。训练/分类器拟合/normalization重拟合均0，Level2未执行。


补充物理输入门槛：见[Acc语义说明](ACCELERATION_SEMANTICS_ADDENDUM.md)。Line候选可审阅但不能在输入契约UNKNOWN时直接进入训练。
