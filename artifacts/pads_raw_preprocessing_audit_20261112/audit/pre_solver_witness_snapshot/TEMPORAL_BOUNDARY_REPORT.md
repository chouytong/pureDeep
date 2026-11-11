# 启动和结束区间

FACT：[PADS官方数据说明](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/)说明每次开始有振动提示并建议约0.5s裁剪；本项目已裁48点，近似0.48s，不是当前完全未处理启动振动。EVIDENCE：按真实时间计算前/末0.5、1、2、3、5s与中心参考的22项sensor-window组合，每窗含magnitude、原始/中心化RMS、方差、MAD、实际dt jerk、PSD/主频/ZCR/短时能量/lag1相关和频段proxy；总188760行私有明细。

固定PD/DD boundary−mid比较440项中0个启动项、9个结束项经BH<.05；结束差异主要见下表。631/880个同组边界−mid配对差异经校正，但不能把这些差异都解释为设备噪声。没有启动显著项也不等于两类启动信息等价或无价值；未做等效性检验。

| activity | sensor | window | measurement | mean_difference | CI_low | CI_high | BH440_q |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CrossArms | acc | end3 | jerk_rms | 12.3785 | 4.15663 | 22.915 | 0.0234939 |
| CrossArms | acc | end5 | jerk_rms | 9.97946 | 2.95016 | 18.8408 | 0.0087406 |
| CrossArms | gyro | end5 | jerk_rms | 32.2943 | 1.20905 | 69.8774 | 0.0087406 |
| RelaxedTask | acc | end1 | dynamic_rms | -0.00611389 | -0.0121679 | -0.00105021 | 0.0361867 |
| RelaxedTask | acc | end3 | dynamic_rms | -0.00606403 | -0.00999793 | -0.00221201 | 0.011088 |
| RelaxedTask | acc | end5 | dynamic_rms | -0.00601525 | -0.0094848 | -0.00280913 | 0.0111175 |
| RelaxedTask | gyro | end2 | dynamic_rms | -0.0258556 | -0.0475699 | -0.00630132 | 0.0299095 |
| RelaxedTask | gyro | end3 | dynamic_rms | -0.0267729 | -0.0465849 | -0.00867669 | 0.0111175 |
| RelaxedTask | gyro | end5 | dynamic_rms | -0.0253292 | -0.0439757 | -0.00822154 | 0.0191724 |


中心参考：先去两端5s，若剩余不足1s，使用中心20%且至少0.5s；不能声称它已被验证为稳定运动。10.24s记录的5s前/后窗口会覆盖大量活动且与中心参考重叠，差异是配对描述，不是互斥独立片段。短窗频率分辨率有限，如0.5s约2Hz，不做精细震颤频率诊断。

HYPOTHESIS：头尾变化混合设备振动、运动启动/停止和姿势调整，当前数据不能逐人逐点分解。NO_ACTION：维持正式48点，不推荐无差别再裁1–5s；更长裁剪尚不满足无效区间明确、跨人重复且保留任务定义的前提。固定每类前2ID的44代表图仅在私有figures，每活动公开群体图见figures/group_quality_*.png。没有按异常或错误名单挑图。


---
数据来源：服务器PADS v1.0.0的390人PD/DD任务子集，原始数据、既有v2信号及45 Frozen WSSL-STR checkpoint元数据；不是原库全部469人。统计单位为患者，同患者双腕先平均；原始记录/通道数只作描述性计数，不作为独立患者。15重叠development splits、seeds42/43/44用于来源核查，不是独立外部验证。缺失/不可计算值为NA，不补造；频谱超过记录Nyquist不外推；稀有交叉分层<10人的数值统计抑制，完整表仅私有。检验族132/440/880分别BH，患者bootstrap2000次；其余描述性表无额外显著性声明。接触全390人诊断标签及validation成员元数据用于质量关联，未读分类错误名单/分数，未构建outer loader、未解码outer性能/预测；历史outer接触永久披露。全队列探索会增加未来选择偏差，不能叫盲态独立验证。训练/分类器拟合/normalization重拟合均0，Level2未执行。
