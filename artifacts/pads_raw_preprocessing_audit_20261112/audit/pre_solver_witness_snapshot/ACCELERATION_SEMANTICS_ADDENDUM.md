# 原始Acc重力语义的审慎判断

FACT：元数据单位一致为g，真实原始记录平均向量模/RMS与轴均值范数由下表给出。EVIDENCE：静息及多数任务的典型幅值远低于完整静止1g向量，不支持把每条原始记录视为已确认的含完整重力输入。UNKNOWN：究竟是采集SDK已分离重力、导出前转换、或其他数据语义，缺少采集API/导出契约，不能只凭范围判定。官方论文和脚本说明L1意在处理重力趋势，不等于验证本地native数据实际保留重力。

| activity | label | measurement | subject_count | median | p25 | p75 | unit | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CrossArms | DD | acc_magnitude_mean | 114 | 0.0681449 | 0.0447547 | 0.11436 | g | raw tensor values; not proof of a particular acquisition SDK API |
| CrossArms | DD | acc_rms | 114 | 0.12614 | 0.0826178 | 0.190358 | g | raw tensor values; not proof of a particular acquisition SDK API |
| CrossArms | DD | norm_of_axis_means_g | 114 | 0.0236526 | 0.014748 | 0.0351904 | g | raw tensor values; not proof of a particular acquisition SDK API |
| CrossArms | PD | acc_magnitude_mean | 276 | 0.0650664 | 0.0486978 | 0.0884173 | g | raw tensor values; not proof of a particular acquisition SDK API |
| CrossArms | PD | acc_rms | 276 | 0.117767 | 0.0857526 | 0.164674 | g | raw tensor values; not proof of a particular acquisition SDK API |
| CrossArms | PD | norm_of_axis_means_g | 276 | 0.0255651 | 0.0192544 | 0.0367067 | g | raw tensor values; not proof of a particular acquisition SDK API |
| DrinkGlas | DD | acc_magnitude_mean | 114 | 0.101372 | 0.0705602 | 0.145998 | g | raw tensor values; not proof of a particular acquisition SDK API |
| DrinkGlas | DD | acc_rms | 114 | 0.148439 | 0.102442 | 0.216196 | g | raw tensor values; not proof of a particular acquisition SDK API |
| DrinkGlas | DD | norm_of_axis_means_g | 114 | 0.0176785 | 0.0134726 | 0.0306373 | g | raw tensor values; not proof of a particular acquisition SDK API |
| DrinkGlas | PD | acc_magnitude_mean | 276 | 0.0793161 | 0.0603888 | 0.10296 | g | raw tensor values; not proof of a particular acquisition SDK API |
| DrinkGlas | PD | acc_rms | 276 | 0.110157 | 0.0838694 | 0.150532 | g | raw tensor values; not proof of a particular acquisition SDK API |
| DrinkGlas | PD | norm_of_axis_means_g | 276 | 0.0165309 | 0.0122088 | 0.0221838 | g | raw tensor values; not proof of a particular acquisition SDK API |
| Entrainment | DD | acc_magnitude_mean | 114 | 0.0696696 | 0.0489171 | 0.110965 | g | raw tensor values; not proof of a particular acquisition SDK API |
| Entrainment | DD | acc_rms | 114 | 0.0775181 | 0.0572603 | 0.131128 | g | raw tensor values; not proof of a particular acquisition SDK API |
| Entrainment | DD | norm_of_axis_means_g | 114 | 0.00475163 | 0.00315721 | 0.00774043 | g | raw tensor values; not proof of a particular acquisition SDK API |
| Entrainment | PD | acc_magnitude_mean | 276 | 0.0622356 | 0.0475249 | 0.0882798 | g | raw tensor values; not proof of a particular acquisition SDK API |
| Entrainment | PD | acc_rms | 276 | 0.0748086 | 0.0556046 | 0.106039 | g | raw tensor values; not proof of a particular acquisition SDK API |
| Entrainment | PD | norm_of_axis_means_g | 276 | 0.0047029 | 0.00313684 | 0.00717022 | g | raw tensor values; not proof of a particular acquisition SDK API |
| HoldWeight | DD | acc_magnitude_mean | 114 | 0.0219298 | 0.0134679 | 0.0397165 | g | raw tensor values; not proof of a particular acquisition SDK API |
| HoldWeight | DD | acc_rms | 114 | 0.0257345 | 0.0171053 | 0.0576989 | g | raw tensor values; not proof of a particular acquisition SDK API |
| HoldWeight | DD | norm_of_axis_means_g | 114 | 0.00333701 | 0.00255622 | 0.00454224 | g | raw tensor values; not proof of a particular acquisition SDK API |
| HoldWeight | PD | acc_magnitude_mean | 276 | 0.0152425 | 0.0114493 | 0.0218984 | g | raw tensor values; not proof of a particular acquisition SDK API |
| HoldWeight | PD | acc_rms | 276 | 0.0182764 | 0.01436 | 0.0272576 | g | raw tensor values; not proof of a particular acquisition SDK API |
| HoldWeight | PD | norm_of_axis_means_g | 276 | 0.00323004 | 0.0025746 | 0.00399424 | g | raw tensor values; not proof of a particular acquisition SDK API |
| LiftHold | DD | acc_magnitude_mean | 114 | 0.0407879 | 0.0292952 | 0.0617822 | g | raw tensor values; not proof of a particular acquisition SDK API |
| LiftHold | DD | acc_rms | 114 | 0.061926 | 0.0402134 | 0.0990722 | g | raw tensor values; not proof of a particular acquisition SDK API |
| LiftHold | DD | norm_of_axis_means_g | 114 | 0.00895162 | 0.0056354 | 0.0161179 | g | raw tensor values; not proof of a particular acquisition SDK API |
| LiftHold | PD | acc_magnitude_mean | 276 | 0.0306262 | 0.0234815 | 0.0433783 | g | raw tensor values; not proof of a particular acquisition SDK API |
| LiftHold | PD | acc_rms | 276 | 0.042832 | 0.0308271 | 0.0756726 | g | raw tensor values; not proof of a particular acquisition SDK API |
| LiftHold | PD | norm_of_axis_means_g | 276 | 0.00805926 | 0.0051756 | 0.0145327 | g | raw tensor values; not proof of a particular acquisition SDK API |
| PointFinger | DD | acc_magnitude_mean | 114 | 0.251134 | 0.1903 | 0.352214 | g | raw tensor values; not proof of a particular acquisition SDK API |
| PointFinger | DD | acc_rms | 114 | 0.324506 | 0.247329 | 0.451889 | g | raw tensor values; not proof of a particular acquisition SDK API |
| PointFinger | DD | norm_of_axis_means_g | 114 | 0.0525372 | 0.0293942 | 0.091722 | g | raw tensor values; not proof of a particular acquisition SDK API |
| PointFinger | PD | acc_magnitude_mean | 276 | 0.247839 | 0.174223 | 0.338122 | g | raw tensor values; not proof of a particular acquisition SDK API |
| PointFinger | PD | acc_rms | 276 | 0.324631 | 0.22192 | 0.450619 | g | raw tensor values; not proof of a particular acquisition SDK API |
| PointFinger | PD | norm_of_axis_means_g | 276 | 0.0509887 | 0.0303303 | 0.0835721 | g | raw tensor values; not proof of a particular acquisition SDK API |
| Relaxed | DD | acc_magnitude_mean | 114 | 0.00830979 | 0.005768 | 0.0158844 | g | raw tensor values; not proof of a particular acquisition SDK API |
| Relaxed | DD | acc_rms | 114 | 0.0112383 | 0.00805634 | 0.0219792 | g | raw tensor values; not proof of a particular acquisition SDK API |
| Relaxed | DD | norm_of_axis_means_g | 114 | 0.00257223 | 0.00189386 | 0.00316333 | g | raw tensor values; not proof of a particular acquisition SDK API |
| Relaxed | PD | acc_magnitude_mean | 276 | 0.00741471 | 0.00530489 | 0.0130925 | g | raw tensor values; not proof of a particular acquisition SDK API |
| Relaxed | PD | acc_rms | 276 | 0.00972309 | 0.00744066 | 0.0178951 | g | raw tensor values; not proof of a particular acquisition SDK API |
| Relaxed | PD | norm_of_axis_means_g | 276 | 0.00269906 | 0.00204966 | 0.00343703 | g | raw tensor values; not proof of a particular acquisition SDK API |
| RelaxedTask | DD | acc_magnitude_mean | 114 | 0.0107424 | 0.00724667 | 0.0235878 | g | raw tensor values; not proof of a particular acquisition SDK API |
| RelaxedTask | DD | acc_rms | 114 | 0.0150421 | 0.00972253 | 0.0283211 | g | raw tensor values; not proof of a particular acquisition SDK API |
| RelaxedTask | DD | norm_of_axis_means_g | 114 | 0.00278536 | 0.00201516 | 0.00355438 | g | raw tensor values; not proof of a particular acquisition SDK API |
| RelaxedTask | PD | acc_magnitude_mean | 276 | 0.00876167 | 0.00622133 | 0.0198911 | g | raw tensor values; not proof of a particular acquisition SDK API |
| RelaxedTask | PD | acc_rms | 276 | 0.0114566 | 0.0083712 | 0.0245807 | g | raw tensor values; not proof of a particular acquisition SDK API |
| RelaxedTask | PD | norm_of_axis_means_g | 276 | 0.00288927 | 0.00227011 | 0.00367155 | g | raw tensor values; not proof of a particular acquisition SDK API |
| StretchHold | DD | acc_magnitude_mean | 114 | 0.0181209 | 0.0121416 | 0.0380077 | g | raw tensor values; not proof of a particular acquisition SDK API |
| StretchHold | DD | acc_rms | 114 | 0.0215806 | 0.0149428 | 0.0484202 | g | raw tensor values; not proof of a particular acquisition SDK API |
| StretchHold | DD | norm_of_axis_means_g | 114 | 0.00225987 | 0.0017525 | 0.00355276 | g | raw tensor values; not proof of a particular acquisition SDK API |
| StretchHold | PD | acc_magnitude_mean | 276 | 0.0132102 | 0.0101361 | 0.0205527 | g | raw tensor values; not proof of a particular acquisition SDK API |
| StretchHold | PD | acc_rms | 276 | 0.0161642 | 0.0131729 | 0.0258611 | g | raw tensor values; not proof of a particular acquisition SDK API |
| StretchHold | PD | norm_of_axis_means_g | 276 | 0.00208994 | 0.00166898 | 0.00287315 | g | raw tensor values; not proof of a particular acquisition SDK API |
| TouchIndex | DD | acc_magnitude_mean | 114 | 0.287133 | 0.199836 | 0.400151 | g | raw tensor values; not proof of a particular acquisition SDK API |
| TouchIndex | DD | acc_rms | 114 | 0.329521 | 0.236492 | 0.438203 | g | raw tensor values; not proof of a particular acquisition SDK API |
| TouchIndex | DD | norm_of_axis_means_g | 114 | 0.118064 | 0.0824563 | 0.179557 | g | raw tensor values; not proof of a particular acquisition SDK API |
| TouchIndex | PD | acc_magnitude_mean | 276 | 0.286337 | 0.179605 | 0.403751 | g | raw tensor values; not proof of a particular acquisition SDK API |
| TouchIndex | PD | acc_rms | 276 | 0.329223 | 0.211794 | 0.457987 | g | raw tensor values; not proof of a particular acquisition SDK API |
| TouchIndex | PD | norm_of_axis_means_g | 276 | 0.133945 | 0.0897541 | 0.221565 | g | raw tensor values; not proof of a particular acquisition SDK API |
| TouchNose | DD | acc_magnitude_mean | 114 | 0.354525 | 0.2441 | 0.45462 | g | raw tensor values; not proof of a particular acquisition SDK API |
| TouchNose | DD | acc_rms | 114 | 0.409009 | 0.29159 | 0.533347 | g | raw tensor values; not proof of a particular acquisition SDK API |
| TouchNose | DD | norm_of_axis_means_g | 114 | 0.122725 | 0.0848795 | 0.160278 | g | raw tensor values; not proof of a particular acquisition SDK API |
| TouchNose | PD | acc_magnitude_mean | 276 | 0.34516 | 0.25846 | 0.507032 | g | raw tensor values; not proof of a particular acquisition SDK API |
| TouchNose | PD | acc_rms | 276 | 0.416734 | 0.303899 | 0.612654 | g | raw tensor values; not proof of a particular acquisition SDK API |
| TouchNose | PD | norm_of_axis_means_g | 276 | 0.125331 | 0.0812618 | 0.181146 | g | raw tensor values; not proof of a particular acquisition SDK API |


这不影响元数据g单位或已测得的零延拓边界效应，但合成1g图不能被当成真实典型幅度。HarNet输入重力契约同样需要核实；未解决前，不启动Level2，不添加假定1g或按模型成绩选重力轴。此前报告草稿保存在restricted_manifests/drafts_before_acc_semantics，最终使用reports9_v2.py。


---
数据来源：服务器PADS v1.0.0的390人PD/DD任务子集，原始数据、既有v2信号及45 Frozen WSSL-STR checkpoint元数据；不是原库全部469人。统计单位为患者，同患者双腕先平均；原始记录/通道数只作描述性计数，不作为独立患者。15重叠development splits、seeds42/43/44用于来源核查，不是独立外部验证。缺失/不可计算值为NA，不补造；频谱超过记录Nyquist不外推；稀有交叉分层<10人的数值统计抑制，完整表仅私有。检验族132/440/880分别BH，患者bootstrap2000次；其余描述性表无额外显著性声明。接触全390人诊断标签及validation成员元数据用于质量关联，未读分类错误名单/分数，未构建outer loader、未解码outer性能/预测；历史outer接触永久披露。全队列探索会增加未来选择偏差，不能叫盲态独立验证。训练/分类器拟合/normalization重拟合均0，Level2未执行。
