# Phase 5：互补性、融合数值及表征可靠性诊断协议

日期2026-11-11；正式起点38c89251a442bb245a77d4bffc539da050fcdeb1。根目录/home/zyt/deep_final/artifacts/wssl_complementarity_phase5_20261111。已知旧结果在用户请求和上轮正式报告中明示；本轮是重复development的描述性/机制审计，不假装未见数据的独立确认性试验。

Frozen WSSL-STR始终保留。新深模型训练0、轻量分类器拟合0、温度拟合0、Scaler/PCA拟合0、HarNet推理/提取0、阈值搜索0。禁止历史结果再生成。只恢复保存的模型读出参数并核验已有validation输出，计算新的描述统计/不确定性。所有拟合状态若缺失标MISSING并阻断依赖它的步骤，绝不自动重拟合或补提特征。

## 资产与访问

启动先读取指定四份上轮文档、README、AGENTS、当前CONTEXT_HANDOFF，及相关Phase3B/3C/3D、贡献/模态/参考/校准报告和分析源码。跳过包含旧outer performance的历史综合报告；永久继承其已暴露事实。历史outer汇总文字接触和整表/缓存整数组读取偏差不能撤销。本轮不声称整个研究历史完全未接触outer信息。

历史受保护SHA清单与新启动锁仅放restricted_manifests，不能上传；公开仅汇总数量/状态。主线、历史实验文件、225个目标/OOF checkpoint和预测、split、normalization及上轮全部输出起止核验。原始IMU信号不读，全球缓存不解码。哈希核验读取完整文件字节但不解析其特征/指标。优先读每个合法development split保存的预测和特征包：先检查ID全部属于当前inner-train∪inner-validation，再读相应标签/分数。全局development融合预测用CSV逐行filter，再核验当前allowlist。不能宣称CSV物理读取了零无关行，也不把源代码检查冒充系统级I/O隔离。

训练/query/excluded IDs，checkpoint、feature、normalization和split hash、seed/context及bank来源写私有lineage。split的test IDs只用于排除断言，任何test performance均不读取。若历史fit-ID、PCA/bank/参数状态不完整，分别说明“来源记录可核验”和“完整数值重放不可完成”的区别。不得混合不同fold encoder坐标；不同split只汇总标量统计，不拼接embedding坐标。

## Phase 5A

四组核心方向：WSSL的log odds z_DD-z_PD及P(DD)，d_PD，-d_DD（DD方向），r=d_PD-d_DD。raw d_DD保留并明确与DD方向相反；raw z_DD另作描述，它不是决策margin且有共同logit平移不识别问题。使用保存的双logits仅为取得实际融合输入，不将log odds误称raw DD logit。

逐split/seed、PD/DD/全体统计均值、sample SD、q05/q25/q50/q75/q95、min/max。核心4个定向score两两Pearson/Spearman，15 split内先均值seed；6对×2相关方法×3标签组=36项单独探索BH族，检验split相关系数对0。概率版本及距离原值可作补充但不增添选择检验。不同尺度只比较相关和其本单位统计。

正式模型指标保持原规则：每split先平均三个seed的六指标，再平均15split。逐人四格错误表提供两套口径：(a)45单seed/appearance与确定reference对照；(b)每appearance先均值三seed概率再决策，与reference对照。旧stable-error>=.75、stable-correct<=.25、unstable介于其间；全四次错/全12次错仅敏感性定义，不能代换。患者级描述汇总四appearance的正确比例，不构造新的跨context部署分类器。

H1补充排序审计：每split/seed的PD×DD配对，只在本split内构造；WSSL排序严格错误的pairs内，报告relative正确排序比例（relative ties计.5），及全部pairs的双向纠错/破坏/净变化。WSSL ties单列，不作为严格错误pairs。pair不是独立样本。此条件诊断不能证明条件独立、临床因果或可部署信息增量，不以全体错误组上的rescue替代净收益。

预登记A-primary四项：relative−WSSL的PD正确率、DD正确率、BA差；WSSL严格错序pairs内relative正确比例−.5。四项一族BH，不给gross rescue>0套“有增量”结论。已有10bin条件AUC只引用上轮结果，非本轮新发现。支持H1必须区分“有部分不一致且正确的排序”和“统计上的独立增量”；后一命题无新增模型检验，本轮可判证据不足。

u=min(d_PD,d_DD)。每个validation split整体、与标签无关的q25/q75固定为低/高覆盖描述组，不是训练/部署阈值。报告全体/PD/DD的WSSL错误率high−low及u与|P(DD)-.5|、r、|r|的相关。覆盖三项high−low差为独立BH族；若分组为空/退化则NA，不改切点。分组由validation定义，只描述，禁止路由/重加权。u高不能解释临床严重度/错标/异常疾病。

## Phase 5B

恢复且仅恢复已有45份fusion NPZ的coef、intercept、scaler mean/scale。按原输入顺序(log odds,r)重建：f=b+w_W(s_W−mu_W)/sigma_W+w_R(r−mu_R)/sigma_R，P_DD=expit(f)，argmax tie→PD。与保存的全部validation概率、决策逐行比较，概率绝对容差固定1e-12；不改容差掩盖问题，不新训练分类头。

检查编码PD0/DD1、输入方向、标准化、fit train比例、OOF和deployment来源。报告b与训练DD先验log odds（并非声称两者恒等）、标准化输入、两项线性贡献及协方差；报告PD/DD正确→错误等四格中已有f的数值分解、DD预测率/平均概率及敏感度。代数恢复隐式边界仅用于解释已有决策，不把任何改系数/去截距/重中心化后的预测当新结果。

OOF/full：WSSL合法OOF与full validation分数按类分布、KS统计量（不提供把独立样本假设用错的KS p）、标准化均值差和Wasserstein；同一train subject的OOF vs full in-sample分数作辅助，明确full in-sample不等同validation。reference逐人OOF未找到时仅用保存的OOF mean/SD和Scaler state核验总体尺度，不伪造类条件/联合分布，也不重算PCA/bank。OOF fit均值标准化为0可推出线性fusion训练均值为b；不能由此推出mean sigmoid=b的sigmoid或重构训练DD recall。

B机制探索四项独立BH族：validation标准化WSSL均值对0、标准化reference均值对0、validation融合均值−b对0、mean|W contribution|−mean|R contribution|对0。它们量化固定已拟合映射下的偏移，不能因果识别迁移原因。若OOF/reference输入或完整fit状态缺失，明确相关机制验证BLOCKED。

## Phase 5C

已有Frozen/adapted验证预测按split/seed/ID/label对齐，六指标引用并复核历史聚合，统计全体与PD/DD四格、分组/各seed稳定性。预测变化用双方保存概率推导log odds；FP32概率有量化误差，不能冒充adapted原始双logits。若仅Frozen有双logits，则raw DD logit displacement不可识别。

cosine/L2/norm漂移只在已有、同一input/preprocess且同语义层的Frozen/adapted特征成对存在时计算，不跨坐标拼接；若完整适配特征不存在，representation_drift.csv中保留45×层的MISSING/BLOCKED记录，值NA，报告缺口，不新推理。预测层关联不冒充表征机制或疾病因果。单个受试者改善不能触发validation-informed训练。

## 统计与决策

预先固定seed20261111、10,000 paired split bootstrap，Wilcoxon双侧、Cohen dz、rank-biserial、mean/median delta及wins/ties/losses(abs<=1e-12为tie)，每个声明BH族完整保留；探索相关允许NA，不填零。15splits重叠，45模型不是45患者；各seed均值、within-split seed SD和split SD分开。

对relative/fusion/adapted相对baseline六指标提供class-stratified subject-cluster bootstrap：390 ID中PD276/DD114分层重采样，所有appearance/seed同一multiplicity，原15unit seed-first estimand保持。另对H1条件排序、患者级错误组净变化和覆盖pooled appearance error rates提供cluster CI。覆盖cluster估计对象为pooled exposure率，不能与15split等权CI混称同一对象；小组零分母剔除的draw数需披露，不更换阈值。均仅固定已拟合模型下条件性不确定性，不覆盖重训、研究选择或独立外部泛化。

历史p/q完整引用不改写；新统计族单列。本轮无任何模型选择/训练保留门槛，retained配置始终不变。H1/H2/H3按事实、合理推测、证据不足/阻断分级，候选只写下一阶段研究假设、单一改动、成本与停止条件，不自动执行。

## 交付与停止

生成用户指定8份专题报告、全部指定CSV/phase5a_integrity，另有FINAL_REPORT、CORRECTNESS_AND_PROVENANCE、ASSET_AUDIT、NEXT_STAGE_CANDIDATES、decision、REPRODUCE、统计附录与私有lineage。逐人表、fit/excluded ID、状态数组只留服务器；公开analysis均为聚合/状态表。无安装/升级依赖。最后再次核验历史/source lock，通过既有选择性publisher与GitHub连接发布同一审核tree，expected old SHA、非force，远端与镜像一致后结束。
