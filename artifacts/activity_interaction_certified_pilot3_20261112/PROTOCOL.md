# Certified L1 Activity Token Interaction：固定三seed复核

用户指定context0/inner0、seeds42/43/44，正式新增训练上限3 run；不是根据新旧性能选择split。历史Phase8已完成旧L1的15×3，本轮仅对一个split复核，不与历史15split均值直接比较，不自动保留、不扩大其余42次、不改变当前模型指针。全部实验在服务器新独立目录，阶段audit→export→correctness-test→train3→analyze→integrity→publish/readback。新模块结果出现前冻结本协议和原源码/配置来源。

来源：Phase8提交4745549dabf34ccc89a740ad361a0410c6a93a4d；Certified L1提交beb57167a7b8b54792f1efed672927396fe8294d。只新增本目录；既有源码/数据/norm/WSSL/STR/HarNet/旧AI与模型指针只读。私有资产用restricted_manifests以沿用原发布器明确排除策略。只核验本轮必要输入及既有hash来源链，不重做全项目历史审计。excluded只读成员元数据，不读取其信号、特征、预测或outer性能。

固定数据pads_l1_lambda50_certified_v1，复用context0/inner0已fit的train-only normalization，复用各seed匹配的Certified WSSL checkpoint和原官方逐人HarNet缓存。无L1重算、normalizationfit、WSSL重训或HarNet提取。三个实例同时包含基础模型和新增模块seed变化，不声称仅隔离模块初始化随机性。

按train/validation分角色导出实际bilateral token[N,11,258]及实际mask、baseline logits、IDs/labels/活动顺序和来源SHA，模型eval/no_grad、true lengths，读取allowlist只包含本split train∪validation。在原activity_aggregator输入处只读hook，经过完整Certified WSSL forward，不拿旧Phase8监督token替代。模型/原参数buffer前后指纹不变。

事先固定复现标准：float32归档。相同路径的零初始化AI输出与其baseline logits必须torch.equal；重复eval/reload同路径要求bitwise。跨完整signal→head token路径和归档路径max absolute logits error≤1e−5、probability error≤2e−7，rtol=0；不因失败放宽。记录实际误差和决策一致性。归档prob为torch float32 softmax；按Phase8B读出prob为float64 exp(logits−logsumexp)，有限精度差别披露，主比较双方统一Phase8定义。

直接import冻结interaction8.py的ActivityInteraction及原explicit_model：258→64、11 identity embedding、单层4head attention/dropout.1、两residual+LayerNorm、FFN64→128→64、masked mean、64→258零初始化输出，再用原classifier线性weight且bias=None映射residual；原subject+STR logits只加一次。全部143172基础分类侧参数冻结且始终eval，HarNet10457408冻结且不进入图。原源码若实际不同则报告，不能替换成近似Transformer。不得加载旧AI训练权重。

正确性至少包括来源/ID/角色/版本/shape/mask、masked NaN/Inf不影响、全无效拒绝、[B,2]finite、零residual/重复eval/reload精确、base无梯度/不变/保持eval、新模块有效梯度/更新、baseline不重复相加。固定仅2次合成smoke更新，首步上游零梯度属于零输出初始化的预期；第二步检验梯度向上游传播。正式训练重新seed与初始化，无smoke权重带入。

原Phase8B token训练：train subject sorted order，batch8、shuffle torch.Generator(seed)、num_workers0，AdamW lr2e−4/WD1e−4/betas(.9,.999)、balanced CE仅train标签、cosineTmax50/min1e−6、clip5、max50/patience12、validation BA严格改善保存best。无AMP、seed_everything deterministicTrue/warn_only，torch线程4、CUDA同步计时，TF32保持当前同环境默认matmulFalse/cuDNNTrue。验证按8批；mask实际全有效，与原ones规则一致。完整3seed，无实验级结果提前停止。失败与恢复保留，不超3个正式实例；已完成hash匹配run可复用，未完成run保留并BLOCKED不暗中重试。

DD p>.5、tie PD；9指标用原Phase8 pred_metrics：macro两概率列AUC、unweighted proper raw-logit NLL、两列平方和Brier、10等宽topconfidenceECE。每seed各列baseline/AI/delta、混淆矩阵、best/实际epochs、参数/时间，再报三seed均值与样本SD及配对delta。绝不ensemble。3seed不是3个患者队列，不做seed显著性检验，不套10/15门槛。可补10000次患者PD/DD分层bootstrap，同一人的3seed与两模型共用权重，估计固定模型/本split条件下seedmean metric delta，仅描述性敏感性。

描述模式：若均值PDrec提高且DDrec下降超过.01，优先标“PD/DD权衡仍存在”；若BA/AUC/F1均提高、两类recall均值各不降超过.01，则“多项指标同向改善”；若BA/AUC/F1均不提高则“未见改善”；其他为“结果混合”。另逐seed报告DD降>.01和方向一致性。该描述不是RETAIN门槛，不改变模型指针。是否值得申请完整15×3作为有理由的研究判断单列，既不自动申请也不自动执行。

公平token→logits计时：同一inner-train前8人resident float32 tokens/mask，batch8、warmup3/repeat10、CUDA同步，比较原head与head+AI，3seed分别测、逐模型核对state不变。给batch ms和batch/8平均耗时、吞吐/峰值；不冒称单患者独立延迟或端到端。排除原始编码、L1、norm、HarNet和IO；完整唯一部署参数含原base与HarNet。

监督train token是in-sample，base与AI都经历本development validation选epoch，validation不是独立test，token不是无监督或end-to-end OOF表示。bootstrap不能消除选择/重复开发偏差。历史旧15split汇总仅背景，不宣称超过它。本轮完成3run、报告、保护与公开发布回读后停止。
