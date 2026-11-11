# Level2修订规范：只供用户审阅，未执行

训练预算本轮仍0，所有候选均NOT_RUN_REQUIRES_EXPLICIT_USER_APPROVAL。优先只审阅PREP-L1-PRECISION一个因素；不改lambda50、trim48、单位/轴、采样、活动、HarNet、结构、loss、threshold或seed。目的为把原目标实现成可验证的逐记录数值解，而不是用成绩挑更平滑滤波。

先固定一个现有依赖可支持的确定性solver实现及source/hash，不能并行搜索多种实现后择优。接受标准是每个series可行dual |u|<=50，Primal−Dual gap<=1e−8+1e−4 E_aff，E_aff为该series减合法仿射参考的残差能量；不能用单一converged标志代替gap。低幅合成KKT例、不同频率/幅值、常数/斜坡、float32保存误差、同一输入单独/混批/重排最大差<=1e−7g都要通过。每条独立终止，缓存factorization可共享而停止条件不能受其他患者影响。

先做限定CPU预算的可行性评测，未定义/无法满足精度和成本上限则STOP，不放宽证书以追求BA。建议审批上限6 CPU-hours(按单核累计)用于solver QA/候选导出，超出停止；实际新solver耗时未测，模型run上限固定45；未经该审批不开始真实候选重处理。如果通过，候选signals只写新独立目录，旧数据和checkpoint不变；本轮未实现/应用该solver候选。

真实候选处理和训练必须在明确批准后：同15subject-level splits×seeds42/43/44，同原Frozen WSSL-STR recipe（batch8/AdamW2e−4/WD1e−4/cosine50/clip5/patience12），HarNet frozen旧rawAcc缓存保持，原seed初始化，原validation BA selection和0.5规则。STR信号改变后，按原函数在各inner-train上重新fit该候选norm（15唯一split，三个seed复用同split值），这是固定数据变换后的必要配套，不是另一个norm搜索。不能沿用旧norm制造分布失配，也不能拟合全队列norm。本轮没有做任何一次norm重拟合。

基线为完整保护的legacy近似L1流程及原45模型。比较先split内3seed指标平均再15split等权；9指标、10000split bootstrap、patient-cluster10000、Wilcoxon/effectsize/seedSD。唯一候选主族BA/AUROC固定2，次族其余7固定7，缺失测试按p1计入、自身NA。报告数值修复与方法学独立性的成功，以及模型是否值得提升为正式模型，两者区分。

模型提升仍需meanΔBA>0、≥10/15wins、paired95%CI下界>0、AUROC/F1不降、PD/DDrecall各下降<=.01、BA/AUC within-split3seedSD<=baseline1.25、correctness/provenance全部通过。原15split是重复development；旧指标不因发现耦合而重写，新对照不能叫独立临床验证。数值改善不等于BA改善，失败不追加lambda/threshold/架构搜索。原始Acc/HarNet物理语义仍需核实，不能由数值证书替代临床物理解释。

PREP-RESAMPLE-BOUNDARY暂不安排联合或并发训练。它需先解决solver/输入契约和参照协议，再另行审阅一个单独padtype=line对照；不把它悄悄加到L1数值修复，不能累加两个因素后归因。时间戳/同步/轴/新滤波/norm搜索无执行授权。全部Level2/下一阶段本轮未启动。
