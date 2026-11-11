# Phase6 frozen protocol：成对表征、覆盖交互与独立验证

2026-11-11，起点161b904462e754a892c3d3d2e4f9ea68b33d18c0。已知Phase5结果和研究假设；本轮是同一开发队列上的机制诊断，不是独立确认性验证。retained Frozen WSSL-STR和所有历史文件不变。新深训练/Scaler/PCA/reference fit/分类头fit/阈值搜索预算均0；允许已有权重推理与新特征导出。输出仅本独立Phase6目录。

## 6A：输入、来源和推理

先锁定历史source/config/split/normalization/checkpoint/预测/Phase5输出hash和官方HarNet10权重c64f9135d99e2dcdfc9ae7cc0672f2bcc438df9ceb8215665882f92cddd162a6，完成全部45对资产/ID/标签/normalization/输入来源检查后才能导出。缺失或错位只阻断对应项，不借别fold/seed数据替代。

只读取当前validation allowlist受试者的逐文件raw Acc与处理后Acc+Gyro，原始/处理文件hash对照既有data migration ledger；不解码历史390人压缩sensor/feature数组。CSV只使用当前ID的标签/文件信息，其他行临时解析与全局窗口顺序/长度元数据读取并不构成系统级物理隔离。历史outer汇总文字接触、全表/全数组读取偏差永久保留；不能宣称整个研究从未接触outer。原始信号读取由ID/path检查阻断，当前pair的train/excluded输入不送入模型；不新建outer/test loader。

复用canonical mainline dataset的validation模式、train-only normalization、六通道顺序AccXYZ/GyroXYZ、左右腕、11活动顺序、真实长度与padding切片。HarNet单独沿原Raw Acc路径：trim48，clip±3g，短记录edge pad到1000点，长记录取首/末1000点，resample_poly(3,10)到3×300。名义100Hz→30Hz，窗口数1或2，绝不虚构两个完整10秒窗。

为复现历史CUDA形状，外部encoder在原始全局窗口布局的64行batch位置只放入当前query的实际窗口，其余行是无subject/label的零padding；不读取被排除人的信号。layer4与frozen layer5按历史提取源码的batch64/末batch形状与eval模式计算。全局顺序只用于数值布局provenance，不拟合、不用excluded标签。源代码和元数据来源记录完整；不能把padding说成读取了其他人的真实数据。新外部中间表示可按subject文件缓存，但各目标模型仅能取其自己的validation IDs。

layer4：保存每activity/wrist/真实window的512×3及mask/count；汇总成512D时先时间均值再真实window等权。last-block适配不含layer1–4权重，因此layer4共享且预期漂移0，这不是疾病机制发现。layer5保存每activity/wrist1024D，实际窗口feature可同步保留。frozen与adapted classifier均fresh显式IndependentWSSL strict load；adapted的base与layer5分开从原state严格加载。完整投影residual含LayerNorm/Linear bias；最终wrist=local+完整residual，保留11×2结构。所有模块eval/inference_mode/requires_grad False，BN、参数和buffer起止fingerprint不变，不触发优化器。

每pair重新计算两模型完整validation概率/决策，与原归档逐ID对照，概率绝对容差固定1e-12、决策完全一致；若不通过先定位数值路径，不能放宽容差掩盖身份差异。source/输入无法确认的pair为BLOCKED，不参与“完整45”推断。保存原始双logits以计算同口径log-odds变化，不用概率反推冒充raw logits。测试独立接口与原计算logits、mask/padding/无效window、自身输入重排与ID错位拒绝；不使用旧闭包wrapper的deepcopy。

每个subject、layer、activity、wrist计算cos distance、L2、Drel=L2/(frozen norm+1e-12)、两侧norm，zero norm另计：两侧均0 cos distance0，一侧0 cos distance1。不能横比不同层原始L2大小。主subject drift为22个合法activity/wrist等权均值，window先等权；activity/wrist明细仅描述，不作为独立样本。

预测事件：每seed/appearance wrong→correct=improved、correct→wrong=worsened，其余unchanged；margin为(2y−1)*(zDD−zPD)，概率margin为P(true class)−.5；flip单列。旧43/46/301为四appearance内三seed概率均值的另一个患者口径，仅敏感性分组，不与单seed混写。分组由validation结局定义，不能路由或训练。

预登记A primary：layer5 subject Drel与true signed log-odds margin change的Spearman，all/PD/DD共3项BH。探索族：layer5/projected/final_wrist三层×margin/flip两个结局×all/PD/DD=18项Spearman；Pearson附录无新选择检验。组差族：三可变层Drel的improved−worsened、DD−PD，共6项。先每split内均值三seed，再15split的signed-rank、paired bootstrap、dz、rank-biserial；不足15完整unit则明确NA，不能静默drop得到显著性。layer4常量相关为NA，不填0。

subject-cluster敏感性使用每个患者12次合法appearance/seed的标量均值（不拼embedding坐标），PD/DD分层10,000重采样，全部appearance共同保留；相关为subject-mean Spearman，与split平均相关是不同estimand。固定患者事件组和类间Drel差另给cluster CI，仍是已拟合模型的条件性证据，非重训/外部不确定性。

## 6B：仅原评分的直接交互

逐行复用Phase5已有u、dPD、dDD、relative和原q25/q75固定覆盖组，容差1e-12核验，不fit bank/PCA/Scaler、不找cutoff。报告class high/low error、abs probability margin、true signed probability margin、relative。primary方向定义error interaction=(PD高−低error)−(DD高−低error)。另两项secondary直接交互为true signed probability margin与abs margin的(DD高−低)−(PD高−低)；三项为B interaction BH族，不能以一类显著另一类不显著代替直接差中差。

同时报告15split等权seed-first交互的Wilcoxon/split CI/effect和按患者联合重采样的pooled exposure交互CI。后者先合计各类high/low exposure分子分母，避免少数DD小组单fold零分母，估计对象不同必须注明，不能混用。种子20261111、10,000次；固定validation切点，不bootstrap重新挑阈值。

confounder检查：活动/腕mask和window counts来自本轮合法输入或已有合法元数据；若11活动/双腕恒完整，说明缺失composition无变异，但执行质量仍未知。context内两类bank等人数且人数固定，不能由人数差解释同split类间方向；不同context bank数相关仅描述，空间覆盖/亚型异质性不因人数相同而被排除。若没有保存per-activity/wrist reference距离，就标BLOCKED，不重算距离。每人4个validation appearances、3seed不当12独立患者。DD来源亚型仅已有标签元数据的描述性分层，无新门控或测试选择。

## 6C：独立验证协议、不是评估

仅浏览官方dataset元数据/原始论文/授权规则，不下载数据、个体表、权重或受限资源，不申请/接受新的数据协议。核查PADS同源、mPower、BEAT-PD、WATCH-PD等真实来源；不存在用户明确提供的医院/时间/设备独立队列就写unknown/not ready。检查participant overlap需要来源方确认/安全链接，不因不同repository名假定独立。PD/HC不等于PD/DD；缺11任务、双腕Acc/Gyro或不同单位不能填零合并算性能。

唯一外部验证模型的实例按不依赖结果的字典序选seed42/context0/inner0现有Frozen checkpoint，锁SHA、原训练normalization、预处理、0.5/argmax tie PD。它代表当前retained配置的一个既有模型，训练人数不是全部390；不能把45fold均值当该单checkpoint的性能。没有新全队列训练。primary calibration NONE，既有temperature仅diagnostic，外部不fit T或任何normalizer。

未来外部指标Accuracy/BA/AUC/F1/PD/DD recall、proper unweighted NLL、two-class sum Brier、10等宽confidence ECE+DD可靠性；95% subject bootstrap CI、设备/条件/DD亚型分层、开发差异描述。没有可用合法兼容外部数据则全部外部metrics=NOT_EVALUATED，不伪造结果。外部标签由独立评价人员保管，锁预测后一次性揭盲，不能回流调模型或阈值。

## 工程、统计与停止

所有fit/query/excluded IDs、原始文件hash、新feature hash及source/checkpoint/normalization/preprocess/split/seed/context映射只放私有目录；公开CSV为聚合，含subject ID的表绝不发布。核心source/协议在新输出前注册hash；代码失败保留旧版本，修复不改变统计口径/容差。保留负结果、MISSING/BLOCKED和旧偏差，不覆盖历史。source guards结束再次验证。

本轮输出用户要求的全部报告/CSV/JSON、额外完整性和统计附录、独立验证准备状态，最后按既有选择性Git流程非force发布相同审核tree，私有数据/权重/预测不上传。保留Frozen，完成后停止，不自动Phase7。
