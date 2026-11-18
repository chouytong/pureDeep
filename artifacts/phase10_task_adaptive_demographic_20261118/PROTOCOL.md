# Phase 10：预先固定的执行协议（2026-11-18）

本轮授权来自`configs/USER_REQUEST.md`。只在服务器的既有`mfam_analysis`环境计算，严格10A→10B→10C→10D→Final串行。当前起点为`99db920b97b81004327a575796c70476f8381090`。本协议冻结后方可查看本轮诊断/候选结果。原数据、正式模型、normalization、split、checkpoint和当前指针不可修改。私有产物只在`restricted_manifests/`；`private/`如存在仅为其内部别名并由发布器跳过。新阶段不继承历史停止规则作为训练授权，使用本轮明确预算和条件。

## 基线与来源

实验对照为Phase9的WSSL-CERTIFIED-L1，匹配STR-CERTIFIED-L1仅用于诊断。不得将CURRENT_MODEL内历史L1指标、旧adapted权重或旧supervised embedding当作本轮初始化/对照。HarNet输入继续为原Raw Acc：trim48、clip±3g、固定first/last1000样本（短记录edge-pad至1000）、polyphase3/10至30Hz；它不使用Certified detrended/normalized Acc。Certified L1处理后的6轴仅进入STR支路。使用15份已保存的inner-train-only normalization，不重新fit。原始来源类仅是粗source labels，不称精确临床亚型。

10A读既有Certified逐run预测，完整核对checkpoint、数据manifest、norm、fit/query身份与哈希；不读取封存outer性能。人口字段只允许age/gender/height/weight；age是采集阶段年龄，age_at_diagnosis及诊断评论、用药、病程、问卷答案不得进入输入。原字段gender不冒称核实的生物学sex。官方数据说明和原论文确认来源及year/cm/kg；设备原始标记只用于分层，永不输入模型。

## 10A固定分析

signed margin=(2y−1)(zDD−zPD)，yDD=1；高置信错误为p_true<0.10，边界为0.45≤pDD≤0.55。按类/正确性报告mean、median、sample SD、IQR与1/5/25/50/75/95/99分位数。图为聚合直方图和class-conditional KDE，不画个体可识别点。错误转移以同seed/split逐人严格配对。每run source subtype n<10时仅报告人数，性能及CI为NA；合并unique-subject的固定三seed/appearance均值概率诊断单列Wilson CI，不能冒充单模型性能。小组不做显著性检验。

人口统计D0：固定LogisticRegression(C=1,max_iter=2000,solver=lbfgs,class_weight=balanced)。四列输入为age、female0/male1、height、weight；连续值train-median imputation，gender train-mode imputation，固定映射；train-only standardization，未做feature selection，不加入missingness indicator。确定性D0每split仅fit一次，共15次，不复制为45个独立fit。比较时可匹配同split三个神经seed作为同一D0的重复参考，seed SD=NA。年龄分层固定<60、60–69、≥70；性别与Series3/4标记分层。组n<10或不足任一类5人时分类九指标标NA，保留人数。数据字段缺失/不可信则阻断对应模型，不伪造。

阈值grid=0.05至0.95步长0.01（包含0.5），曲线仅描述validation操作点。主规则pDD>0.5、tiePD不变。AUROC、proper NLL、two-column Brier及10等宽top-confidence ECE在固定概率下不随决策阈值改变；曲线中显式记录不变，不以阈值声称排序/概率校准改善。仅可使用严格匹配Certified版本、目标inner-train subject separation和完整checkpoint/norm来源的既有OOF分数；没有则BLOCKED，不新增135个OOF训练，不借旧L1 OOF。

## 10B、10C、10D与pilot门槛

FT-L1仅训练官方HarNet的实际最后一个stage；FT-L2为最后两个stage。审计实际ResBlock/BN/Conv及形状后固定范围，不能将stage名字冒称ResBlock。两者backbone LR1e-5、classifier LR2e-4；所有BN running统计和affine均冻结，其他被冻结state必须不变。官方初始化在每split/seed独立加载；分类侧从对应原seed初始化，不warm-start旧adapted。其余保持Certified训练recipe：AdamW WD1e-4、betas(.9,.999)、batch8、workers4、max50/cosine/min classifier LR1e-6、clip5、patience12、strict best validation BA、AMPfalse、原增强0。HarNet scheduler保持与classifier相同cosine相对因子（1e-5→5e-8），不能被classifier最低LR抬高。可缓存同一官方冻结prefix的Raw Acc表示，但仅按当前fit/query allowlist读取；不读取旧整队列压缩缓存。

FT-L1先context0/inner0三seed。资源门槛：meanΔBA>0、BA至少2/3提高、meanΔAUROC≥0（将“不明显下降”保守固定为不下降）、meanΔDDRecall≥−.01、proper NLL增量≤.05、有限loss/gradient、BN/冻结state/来源正确。NLL阈值是本轮“明显概率退化”的预先工程门槛，不是临床量纲。失败即停止FT-L1；报告后只有明确支持增加深度的独立证据才考虑FT-L2，不把单个最好seed当证据。用户已授权由研究者裁量预算歧义；采用整个10B最多45 formal runs、优先FT-L1，绝不默认两个完整矩阵。FT-L2未执行时所有比较问题回答NOT_TESTED，不编造。

C仅在10A字段可信可用时执行。D2=固定Certified原logits+W_demo·MLP四人口字段；MLP4→8 ReLU→16 ReLU，W_demo=Linear16→2、bias=False、零初始化，共216新增参数。原HarNet与WSSL所有参数/BN均冻结且eval；只训练新分支，初始化logits逐位等于基线。CE沿用对应train-balanced权重、无DD额外系数。其余训练/选epoch规则固定，与原recipe匹配。先同一单split三seed；C资源门槛为BA均值>0且≥2/3seed提高、AUROC不降、两类Recall均不降超过.01、NLL增量≤.05、correctness通过。若D0在BA和AUROC均严格超过D2则判人口信息主导而fusion无额外增量，拒绝扩展。可计算的group中任何类别Recall较同组baseline下降>.15为极端不稳定，拒绝扩展；小组NA不是稳定性通过的证据，另作限制。通过后最多增加42正式run；pilot属于development，非独立确认。

D启动必须由A固定诊断支持：DD Recall低于PD至少10/15split、DD signed-margin均值低于PD至少10/15split、错误覆盖至少3个实际source类，且不是n<20单一来源占≥80%DD错误；输入/证书/finite质量检查通过。不能用validation错误名单赋权。保持原weighted-mean balanced CE，仅DD weight×1.10，再以train人数加权均值归一化回原尺度；weight scale常数会在PyTorch weighted-mean中约掉，它不保证新的梯度norm不变，也不是固定分母CE。测试手工分子/实际batch权重分母、常数缩放不变性与梯度，禁止更改reduction或搜索其他系数。先3seed；资源门槛为DDRecall和BA均值>0、PDRecall≥−.01、AUROC/F1≥0、NLL增量≤.05、正确稳定。失败REJECT，不搜权重。通过最多增加42run。不能叠加B/C/AI等改动。

## 完整统计与保留

每split先平均三个seed指标，再对15个重叠split等权。九指标均含mean/sample split SD及seed稳定性，ensemble不混入。每候选与同seed/split的Certified WSSL配对；完整15split用10,000 paired bootstrap、mean/median Δ、wins/ties/losses、Wilcoxon、dz与rank-biserial。主BH族预先为FT-L1/FT-L2/DEMO/DD四候选×BA/AUROC=8；其余7项×4=28为secondary，未运行/非有限项以p=1占位、其自身q=NA，不缩族。pilot不做完整显著性或保留判定。patient-cluster bootstrap10,000次，PD/DD分层，同一人的所有appearance/seed共用权重，固定拟合条件性敏感度，不当独立验证。

完整保留逐项沿用用户9门槛；stability固定为BA和AUROC的mean within-split三seed sample SD均≤baseline1.25倍。correctness/provenance全部通过是必要条件。没有新候选达标就保持WSSL-CERTIFIED研究基线与现有指针。即使候选通过，也仅单独登记development candidate，不覆盖纯IMU模型或CURRENT_MODEL。

每阶段报告归档后进入下一阶段；每轮已完成实验及时使用原选择式发布器同步代码、协议、测试、报告、安全聚合与累计实验CSV，负结果同样发布，远端commit/tree/源快照/镜像回读一致才称完成。日志、checkpoint/optimizer、IDs、逐人预测及人口记录仅服务器。最终报告完整回答12问题，停止，不启动Phase11。
