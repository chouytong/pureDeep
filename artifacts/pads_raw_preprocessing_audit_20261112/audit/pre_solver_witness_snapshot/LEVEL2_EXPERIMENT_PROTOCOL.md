# Level2待批准协议：未执行

**状态：NOT_RUN_REQUIRES_EXPLICIT_USER_APPROVAL；本轮训练预算0。** 本文件是完整报告后的可审阅候选规范，不是自动启动授权。若用户批准且PADS导出Acc重力语义与HarNet官方输入契约已核实，才只测试PREP-RESAMPLE-BOUNDARY一个主因素：原1000point rawAcc上下文的resample_poly padtype由constant0改为line；window/clip±3g/30Hz/300points/HarNet权重、STR真实信号、归一化和模型结构均不变。PREP-BASE为原冻结流程，原45checkpoint和概率可作配对基线，不覆盖。

先冻结新独立目录/源码/配置与上限：候选外部特征label-independent导出一次，仅受控allowlist；45个classifier-side模型，15相同subject splits×3seeds42/43/44，HarNet固定，原seed初始化，CE/AdamW2e−4/WD1e−4/batch8/cosine50/min1e−6/clip5/max50/patience12及原BA checkpoint selection不变。由于STR输入未变，沿用各匹配inner-train既有norm，无重新拟合。禁止新阈值/temperature/seed子集/配方搜索，不按验证错误名单改规则。若需要改norm、window或rate，已超出这一审批对象，不能偷偷组合。

Smoke门槛：原baseline exact输入/feature/预测重现；constant/sine/alias/边缘shape与finite测试，300points、通道单位、ID/activity/wrist对齐，所有fit/query/excluded集合与原hash；候选不得把验证人传入拟合或把全feature缓存先解码再任意筛选。smoke失败先停止并归档，不开展45run。不得因固定旧权重推理略好就替代匹配重训练对照；也不把边界更平滑当成功。

评价9项Acc/BA/AUROC/MacroF1/PD/DDRecall/NLL/Brier/ECE，先split内平均seed再15split等权；paired10000split bootstrap/Wilcoxon/effect-size，主体class-stratified patient cluster10000；15split不是独立临床队列。唯一候选主族BA/AUROC固定2，次族其余7固定7，缺失项保留按p1调整/自身NA。device分层只作预定敏感性，不作为事后选seed/阈值的选择器，Series3 DD仅11人的精度限制必须报告。

联合门槛：meanΔBA>0，≥10/15BAwins，paired95%CI下界>0，AUROC/F1不下降，PD/DDRecall各下降≤.01，BA/AUC mean within-split3seedSD≤baseline1.25，correctness/provenance全过。q/clusterCI完整报告；通过仅development保留，不覆盖正式mainline，不声称外部泛化。成本上限45候选run及一次固定export，无额外重试选最好。任一保护hash异常、来源/维度错误、不可解释设备差异、门槛失败即停止；不以再换padmode/滤波/阈值挽救。时间戳/同步/L1/norm候选尚不满足可执行前提，不在本批准对象和预算内。


---
数据来源：服务器PADS v1.0.0的390人PD/DD任务子集，原始数据、既有v2信号及45 Frozen WSSL-STR checkpoint元数据；不是原库全部469人。统计单位为患者，同患者双腕先平均；原始记录/通道数只作描述性计数，不作为独立患者。15重叠development splits、seeds42/43/44用于来源核查，不是独立外部验证。缺失/不可计算值为NA，不补造；频谱超过记录Nyquist不外推；稀有交叉分层<10人的数值统计抑制，完整表仅私有。检验族132/440/880分别BH，患者bootstrap2000次；其余描述性表无额外显著性声明。接触全390人诊断标签及validation成员元数据用于质量关联，未读分类错误名单/分数，未构建outer loader、未解码outer性能/预测；历史outer接触永久披露。全队列探索会增加未来选择偏差，不能叫盲态独立验证。训练/分类器拟合/normalization重拟合均0，Level2未执行。


补充物理输入门槛：见[Acc语义说明](ACCELERATION_SEMANTICS_ADDENDUM.md)。Line候选可审阅但不能在输入契约UNKNOWN时直接进入训练。
