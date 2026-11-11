# Phase8 固定执行与统计协议

实际执行日2026-11-12（Asia/Shanghai），目录按用户指定保留20261111。起点35b0a7cb78b28f29385724cf98f4f1dbdde8b2b2。当前新授权允许按A1→A2→B→C→D完成完整实验；覆盖此前“结束Phase7不自动训练”的范围，不改变主线、历史负结果、outer与发布边界。每阶段报告和correctness/source gate归档后才进入下一阶段。不同新深度候选不同时完整训练。

## 事先已知与解释

历史Phase3B已经报告三seed概率集成（BA约.7200、AUROC约.7816），因此A1是新门槛下的复核，不能冒称新发现或结果盲态预注册。所有阶段都是同390人重复development，不是独立临床验证。原监督encoder的train embedding是in-sample supervised representation；不能称无监督独立表征。过去outer文字接触/整manifest和NPZ读取及公开检索摘要个体暴露永久披露；本轮不读旧outer性能/信号，不新建outer loader。

## 资产/输入

所有45模型配对canonical split/seed/provenance/norm/标签。首先核对已有合法per-model train+validation feature NPZ的IDs/角色，再读取其数组；允许集合为该模型train∪validation，excluded只作私有排除元数据。禁止用全390人sensor/feature缓存先解码再筛选。旧合法特征包包含bilateral[N,11,258]、subject[N,258]、structured[N,176]和raw2logits/probabilities；来源旧extract_baseline_v2。旧提取当时曾调用整缓存，永久保留这项偏差，不因本轮复用分模型包而抹去。P5保护hash、创建代码freeze与Phase6完整validation重现作为当前来源证据；不能虚构缺失的创建时逐特征hash。

每项新拟合只接触当前inner-train，validation仅预测/既定epoch selection。原normalization直接复用经过核对的train-only统计，不重fit。A1/2/B使用既有checkpoint/特征，C才使用allowlist下的原处理后六轴及Phase6每人官方frozen-HarNet export（与监督checkpoint无关）。记录fit/query/excluded、checkpoint/source/preprocessing/norm/feature hashes，IDs和预测只在私有目录。

## 固定方法

A1主候选3seed概率均值，辅助raw2logits均值后softmax；DD>0.5，tiePD，无seed子集/阈值/温度搜索。Ensemble是3模型系统，不称单模型。固定ensemble没有可估计的重复seed方差，seedSD=NA；另用15split BA/AUROC SD≤baseline1.25作为“无明显稳定性恶化”的诊断替代，明确它不是seed方差估计。

A2仅一个主要特征位置：concat(original subject258, original structured176)=434D，一行一个患者，恰对应原最终两项线性readout的联合输入，不从多层挑最好。StandardScaler只fit train，无PCA/feature selection/L2-normalize。固定LR(C1,L2,balanced,lbfgs,max_iter2000)；Linear SVM与RBF-SVM均SVC(C1,balanced,probability=True,random_state=模型seed)，RBF gamma=scale。SVC自带train-only内部5fold Platt概率估计是预先声明的分类器步骤，不用validation、不拟合温度、不搜索。RF(n_estimators300,max_depth4,min_samples_leaf5,max_features=sqrt,balanced,n_jobs4)。XGBoost只在已有兼容环境时运行：300浅树depth3,lr.05,subsample1,colsample1,reg_lambda1,min_child_weight5,固定seed；没有依赖则NOT_RUN_DEPENDENCY_MISSING，不安装。train/validation监督encoder先前使用方式如实披露，不将SVC内部概率CV误称整个encoder crossfit。

B只有审计确定无完整token-to-token机制才运行一个主候选。现有attention有softmax竞争与池化后的非线性，不称完全没有跨活动关系；缺口限定为pooling前的内容依赖token mixing。固定新模块258→64、11activity embedding、1层4head self-attention、两次residual+LayerNorm、FFN扩展2、dropout.1、masked mean→64→258零初始化residual。原classifier权重把delta映到logits，原subject/STR logits不变作anchor；不另设gate或新分支。全部原WSSL模块冻结，只训练新interaction模块，使用该模型合法bilateral缓存。零初始化输出须与原baseline逐位相同。一个预定inference attention-off诊断，不另训练消融、不择优保留。

C在B归档后进行。只改原CE→CE+.1 SupCon，tau=.1，不加projection/head/augmentation/sampler，不解冻HarNet。为避免强制异质DD全部相互吸引，固定单一asymmetric subject SupCon：仅PD作positive anchors，其他不同患者PD为positives，DD作为negatives；DD不做跨亚型positive吸引。缺≥2不同PD或缺DD的batch该项=0并记录，禁止复制患者凑样本。粗Other/ET/Atypical/MS来源不是可靠细粒度亚型监督，本轮不用作附加loss。LocalWSSL分类侧按原seed初始化/同50epoch、batch8、AdamW2e-4,WD1e-4,cosine50,clip5,patience12选val BA最佳，HarNet固定；无任何其他recipe变化。此不代表所有常规双类SupCon都已被检验。若valid-anchor构成或梯度/identity correctness失败，停止C并记录，不扩大搜索。

B/C均先独立smoke，再45完整run；一阶段全部归档后才运行下阶段。C的pre/post embedding geometry按同split/seed患者在两模型上的subject258计算cosine距离、norm、margin，train与validation分开，不跨模型直接比较未对齐坐标的逐维向量差。类内/类间同模型标量距离可比较；DD粗类别之间同样只描述。DD类内或亚型间距离相对压缩>25%是事先固定的解释警示，不是临床结构受损证明；不根据geometry重选lambda/模型。

## 评价和保留

9指标：Acc/BA/AUROC/MacroF1/PDrec/DDrec/proper unweightedNLL/two-class sumBrier/10等宽top-confidenceECE。使用原始raw logits计算NLL，不任意clip概率。非概率分数不能冒充calibration；无法定义的指标NA。所有新模型采用明确概率>.5决策，tiePD，AUROC用完整两类概率macro定义与原程序一致。

先split内平均3seed指标，再15split等权；ensemble每split一个系统指标直接配对seed-first基线。配对10000 bootstrap（seed20261112）、均值/中位/wins-ties-losses/CI/dz/rank-biserial/Wilcoxon。A1主族2、A2主族10（含XGB未运行两项）、B主族2、C主族2，主指标BA/AUROC；最后全局主族固定16，缺项内部按p=1校正、缺项自身p/q仍NA。次要族固定56=8候选×7其余指标（含XGB）；logit ensemble敏感性独立9项。不得缩小family以删除负结果。BH不修复split依赖和多轮开发偏差。

患者cluster bootstrap按PD/DD分层10000次，同人的全部appearance/seed共用抽样次数，估计固定已拟合模型条件下seed-first15split指标差；与split bootstrap、seedSD、外部泛化分开。零分母/不可检验draw如实记录，不改为最有利pooled口径。模型经validation早停与多轮研究选择产生的乐观偏差不由此消除。

统一gate：ΔBA>0、BA至少10/15wins、配对splitCI下界>0、AUROC/MacroF1均值不下降、PD/DDrec均值下降≤.01，BA及AUROC平均within-split3seedSD≤baseline1.25（ensemble另述），correctness/provenance通过。没有新增q阈值条件，但完整报告p/q与cluster敏感性；通过仅development保留，不改正式mainline或外部单checkpoint锁。Logit ensemble永不替换A1主要候选。所有负结果、条件未运行和失败脚本/日志保留。

公开仅代码/协议/配置/汇总/报告，features/checkpoints/逐人结果/受保护manifest/logs私有。全部实验在服务器。本轮按新授权完整推进，Phase8D完成后发布并停止，不启动Phase9。
