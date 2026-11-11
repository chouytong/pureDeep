# Phase9 Level0/1冻结诊断协议

实际执行2026-11-12；官方起点4745549dabf34ccc89a740ad361a0410c6a93a4d。仅数据质量/源码审查与独立诊断，无模型训练、分类器拟合、normalization重拟合、原数据修改、正式candidate应用、outer loader/性能访问。Level2需完整报告后另获用户明确批准，本轮不执行。

先完整扫描当前mainline、准备/HarNet/缓存/训练/推理源码及指定历史报告并核对远端；后才读取原始传感器数组。依本轮明确授权，对服务器当前390名PD/DD任务子集全体作cohort质量诊断，包含既往反复development出现过的人；不按outer身份分组、选择记录或读取其性能，canonical split只用于ID交集和norm来源核查。此不是新独立验证，也不使历史outer重新untouched。整队列label-conditioned探索会增加未来方法选择偏差，永久披露；今后性能评估不能以此声称结果盲态。

所有原始文件、处理信号、mainline及P8保护链和P8产物建立审计前hash，最终复核；不改原文件，不删除任何文件，不运行训练/预处理生产入口。私有ID、逐人统计、样本位置及代表性信号图存restricted_manifests，必须文件名以private_开头或处于被发布器排除的目录；公开analysis只含患者聚合分布/匿名计数，不公开稳定可联接的化名或原始file hashes。要求的逐人表同名保存在restricted_manifests/analysis/；公开同名表为隐私聚合版本并明确统计单位，不用混淆为逐人数据。

代表图在读取信号前按ID字典序固定每类前2人，所有11活动同一规则，不按波形/异常/预测筛选；名单私有，图只服务器。公开figures仅群体汇总/合成验证。异常位置全保留为候选，不等同设备故障，不删除：非有限/时间非单调为结构错误；dt>10×本记录median为大间隙；近似缺包以round(dt/median)-1估计而非已证实packet loss；完全常数run≥50点为候选；robust channel level/derivative >20×1.4826MAD仅候选；相同全6通道连续点重复与单通道量化重复区分。原生硬件满量程未经文档证实时，saturation只能报极端重复值候选，不能把HarNet±3g裁剪门限冒充硬件量程。每类异常附同步多通道/时间/频谱证据，不按类别更改阈值。

时间诊断按每记录真实相对时间选择前/末0.5/1/2/3/5秒，mid为去掉两端5秒后的中段；若该中段不足1秒，改用中心20%且至少0.5秒；不足则NA。窗口不是互斥分区，前后5秒可能覆盖短记录大部；不得按诊断crop正式信号。频谱以本记录median fs作为近似均匀采样轴，并报告其不规则性，不能用Welch证明跨缺包信号真频率。所有六轴基础统计、向量模/中心化dynamic RMS、MAD、jerk(真实dt)、zcr、lag1 autocorr、0.25s短时能量、PSD诊断均只读。固定频带0–3/3–12/12–15/15–Nyquist Hz，仅描述信号带宽与HarNet目标Nyquist，不当作确诊震颤或数据驱动择带。

当前L1 detrend是数据自适应非线性，不存在单个LTI transfer function；以同原始记录的currentprocessed和raw[48:]配对幅相/频段差、预固定合成正弦频率×幅度响应诊断，不能把非线性响应叫固定cutoff。允许Level1仅内存候选图：固定raw-minus-recordmean（不作为norm重拟合/生产滤波）、HarNet resample_poly的padtype=line边界敏感性；不输出候选sensor文件、不运行模型、不作性能消融。合成输入可验证采样/抗混叠和mask，不能据此宣称真实数据改进。

统计单位始终患者。先在同患者/活动内平均双腕，不能把8580记录、6通道、窗口当独立人。主class关联固定11活动×12指标=132项Mann–Whitney（DD−PD、rank-biserial与患者bootstrap95%CI）；12指标预设duration_seconds,acc_dynamic_rms,gyro_rms,acc_jerk_rms,gyro_jerk_rms,acc_fraction_3_12,gyro_fraction_3_12,gap_fraction,constant_fraction,clip3g_fraction,start1_acc_rms_ratio,end1_acc_rms_ratio。缺项仍占family按p1调整、自身p/qNA。时间边界固定11活动×2sensor×10boundary×2指标(RMS/jerk)的PD/DD差族440项；同组边界−mid配对族为前述×2class=880项Wilcoxon。BH分别校正132/440/880，不事后缩小族。全部其他统计/粗DD来源仅描述，不把相关性说成因果。Bootstrap2000次，seed20261112，以同人的双腕联合抽样。无误名单或原模型分数进入诊断。

normalization只读取45匹配文件及checkpoint已有统计、fit_ID集合与源码，不调用拟合函数或从原数据再计算训练norm；合成hand-test只测已给定均值标准差的apply。双腕relative timestamp差不等同真实物理同步：官方时间重置零、硬件trigger与clock校准未知，互相关也不能作为强制同步ground truth。不DTW、不自动轴翻转。

预处理候选只能在完整审计后提出，按FACT/EVIDENCE/HYPOTHESIS/UNKNOWN/NO_ACTION区分。保留原48点裁剪是当前对照；无证据不把0.5s扩大到1–5s。公开发布沿用现有选择发布器，不修改过滤器，必须逐项验证私有表/信号图未包含。本轮结束Level0/1后停止，Level2待明确批准。
