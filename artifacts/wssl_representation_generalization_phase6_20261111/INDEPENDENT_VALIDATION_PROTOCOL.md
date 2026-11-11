# 独立跨受试者验证协议：尚无外部评估结果

2026-11-11；唯一目标是固定Frozen WSSL-STR的PD/DD受试者级外部验证。不得把15个重叠development splits、不同seed、同人的其他session或PADS重打包视作独立队列。旧outer文字接触/整表读取偏差永久披露，旧outer不作为本模型的新独立证据。

## 锁定模型与决策

按不依赖performance的字典序选择现有seed42/context0/inner0 Frozen checkpoint，SHA及全部pipeline哈希见analysis/external_validation_lock.json。保留原143,172参数分类侧和官方10,457,408参数冻结HarNet10；不新训练全390人模型、不做45模型ensemble。该实例训练于其原inner-train（实际人数记于lock），不能把45run均值当其单模型期望。

采用此checkpoint原训练normalization，不在外部数据fit任何均值、Scaler、PCA、阈值或校准器。决策为原两类argmax，tie PD，相当于P(DD)>.5。primary calibration为NONE；正温度仅历史diagnostic，不能看到外部标签后决定加不加T。

预处理固定：local六通道AccXYZ/GyroXYZ、原L1/trimming和原训练归一化；Raw Acc分支trim48、clip±3g、既有短窗edge padding/首末真实10秒段、100→30Hz resample_poly3/10和HarNet10。保持11活动、双腕、真实长度、同一活动定义/顺序。模型权重、norm、sample单位（Acc g/Gyro rad/s）、方向与clock必须有可审阅映射。外部文件格式转换只允许无标签的确定映射；若需插补活动、补Gyro、学device映射或修改架构，则该队列不具备直接测试条件，应另立任务而非强行报告性能。

未来外部批处理固定无标签ID排序、外部encoder64窗/末真实batch，classifier8人/末真实batch，eval/no-grad、BN不更新。Phase6为了归档重现而用的历史batch零padding布局只服务内部数值核验；不能依赖内部人的临床信息处理外部输入。外部格式适配器尚未因真实数据验收，必须在标签揭盲前完成代码/hash冻结和合成边界测试，不能冒称已实现且验证外部评估runner。

## 数据进入条件

1. 数据使用协议/伦理/同意范围支持本研究及拟公布聚合指标；由数据管理方确认，不擅自签约/申请/下载。
2. 新队列真实来源、医院、时间、设备、采样、任务、校准、诊断流程可核验。由数据方安全核对本项目390人及全部监督/参考fit人群的重叠；不同伪ID/仓库名不是无重叠证明。外部SSL预训练成员重叠若无法核查也必须声明。
3. 标签明确为临床PD与可比DD，记录ET/atypical/MS/other构成、诊断确定性、用药/ON-OFF、年龄/性别与任务执行条件。Healthy control不能映射成DD。
4. 确认双腕11任务Acc/Gyro映射。若设备/任务不匹配，只能形成兼容性/迁移研究，不能与development直接合并算同一任务性能。
5. 连续入组/先验纳入排除标准先定，不能按模型错误、距离或预测概率筛人。缺失/无效记录报告人数和原因，拒绝评价不能隐匿为正确预测。测试label由独立评价者保管；外部不用于选择checkpoint或阈值。

## 一次性评价

先在无标签输入上冻结数据清单、格式适配、模型/pipeline/hash和逐人预测；锁定后由独立评价者一次性揭盲。首个合格session/预定记录优先级在揭盲前确定；重复session不是独立患者，若评价多session须以subject为cluster。不得看到结果后回流训练或挑test子集。

必须报告Accuracy、BA、AUROC、Macro-F1、PD/DD Recall、proper unweighted NLL、two-class sum Brier及10等宽top-confidence ECE；补充P(DD)可靠性和classwise ECE。原始双logits用于稳定NLL计算，不用weighted训练CE冒充proper loss。距离没有概率映射时不计算其校准指标。

95% CI使用预先固定seed20261111、10,000次按PD/DD分层的subject bootstrap；重复记录同一人一起抽样。报告置信区间、样本数、类别先验和缺失情况；ECE小样本不可靠，不能作为唯一结论。设备、医院/时间/采集条件、DD来源亚型各自分层，样本不足则明确区间宽/NA，不事后合并最有利子组。对预先声明的探索子组比较分别定义BH族，不能临时扩张后挑显著。

与development差异仅描述，不把不同入组/标签定义/设备的accuracy直接比较成改进。当前只有一个固定外部模型，不搜索方法或更新retained。若后续需模型适配，应在新的development资源独立训练，保留外部测试锁，另获授权。

## 当前准备状态与停止点

已锁模型实例与评价规则；**尚无已授权、已验收且PD/DD输入兼容的独立队列**。因此外部全部指标NOT_EVALUATED，不能给外部泛化95% CI。公开来源核查见DATASET_COMPATIBILITY_AUDIT.md；尚需数据方授权/重叠核验、输入adapter验收与标签隔离。协议完成不等于实际验证完成。不自动下载数据、申请访问、联系他人或启动Phase7。
