# Patch-based Temporal STR：正式训练前冻结协议（2026-10-08）

## 目标与边界
仅检验固定local patch与ordered patch aggregation的增量；“过早压缩”是待验证假设。只用15固定inner-development splits，seed42/43/44；outer/context为历史目录索引，不读取outer-test信号、预测或性能。P0采用审计PASS的45正式STR状态/预测，当前source匹配Phase2重训；无需重复P0。旧f0命名仅baseline辅助文件名，不执行Frequency模块。FrozenWSSL仍总体retained best，本轮不使用WSSL/HarNet、Frequency/FFT、augmentation、triplet、InstanceNorm、memory bank或kNN。

## 固定条件
P0原STR75,524参数。P1和P2保留完整temporal/learned-moment/wrist/bilateral/activity/structured路径。
输入仅原train-fold normalization后的6-channel Acc+Gyro。沿用原按activity真长度分组、valid-wrist筛选逻辑；每个valid wrist只接收slice到真长度的输入。200sample patch、100stride，完整patch加唯一tail-aligned start。976对应9patch，2000对应19patch。长度不足200无patch，residual0；缺失activity/wrist排除。
共享patch encoder：DWConv6k15pad7→PW6→32→GELU→DWConv32k7pad3→PW32→64→GELU→patch-time mean。bias全部True，无新normalization/dropout。P1 masked mean。P2 DWConv64k3pad1groups64→GELU→masked Linear64→1 attention pooling；conv前invalid输入清零，conv后再mask，empty行权重和residual为0。二者64→16→GELU→zero-init Linear16→64 residual加原64D wrist token。
P1总80,340/新增4,816；P2总80,661/新增5,137；P2比P1多321。先构造所有原STR参数，再用fork_rng隔离新分支；两候选encoder/head先初始化，后构造P2额外算子，保证共用初始化匹配、原参数和DataLoader/训练RNG一致。全部从匹配seed随机初始化scratch训练，不从正式trained STR warm-start。
P2-P1同时改变序列卷积、pooling及321参数；只能支持当前ordered组合，不能隔离order、attention、capacity的因果贡献。

## 原recipe逐项锁定
AdamW LR2e-4、WD1e-4、betas(.9,.999)、batch8、FP32、clip5、cosine50epoch floor1e-6、max50；train-fold balanced CE、label smoothing0；原strict validation BA first-best、patience12、min_delta0；threshold.5/argmax、tiePD。原activity/wrist顺序、sampler、输入、normalization、loss均不变。不会按结果更改patch/stride/hidden/pooling/threshold/epochs/patience，不做搜索或P2救援。

## 正式入口与审计
P0全45validation复现、15norm refit PASS；extraction/P1/P2正确性PASS；原engine P1/P2各one-batch one-epoch smoke PASS，smoke不用于选择。源码/协议/统计与测试SHA冻结并推送Git tag后，才能训练90次(P1/P2各45)。同一条件seed连续15split，禁止覆盖或resume已存在stage。每run核对ID/labels、train-only norm、原recipe、params、checkpoint config/source、严格BA选择/stop、probabilities/metrics、outer loader不存在；记录best/last、epoch loss/metrics、实际训练预测耗时。训练后重跑全run hash/audit；冻结正式STR资产不修改。

## 统计
六metric Accuracy/BA/AUROC/Macro-F1/PDRecall/DDRecall；原两列概率tie-aware macro AUROC实现不更换。每split三个seed的metric先取均值，再15配对unit比较；不是平均probability ensemble。三个比较P1-P0、P2-P0、P2-P1。固定bootstrap seed20261008、10,000 paired split resamples、percentile95%CI；wins/ties/loss eps1e-12、paired dz、rank-biserial、双侧Wilcoxon signed-rank。三个BA为primary BH family，另外18比较BH仅探索性。分别报告bootstrap与Wilcoxon/BH，不互相替代。45run、patch、subject/pair均不增加统计n；split共享subjects/training pool，仅repeated-development robustness而非独立外部推断。
完整报告seed mean/seed-mean SD、mean within-split seed SD、15split SD、同seed方法agreement/score Spearman、跨seedagreement、best/stop epochs、train/validation分类loss与best-last变化。online train含Dropout，不能当eval-mode泛化gap；没有中间checkpoint，不伪造或追溯EMA。

## 预先采用的正式保留规则
直接复用最近同STR、三条件residual实验Frequency-Prior的正式规则，替换为P2对P0及P1两个matched比较。仓库原STR相对V8的BA+.01/AUC+.015晋级规则属于另一比较对象；本轮不混用，也不按结果自行选门槛。
P2必须对两个reference均满足：meanΔBA>0、至少10/15胜、BA bootstrap下界>0、三个primary BA的BH q<.05；AUROCΔ≥−.005、Macro-F1Δ≥0、两类Recall各Δ≥−.01；BA/AUROC mean within-split seed SD≤1.25×max(reference SD,.001)。全部满足RETAIN。若P2-P0 meanBA≤0或任一两reference辅助保护失败，REJECT；其余INCONCLUSIVE。本轮主保留对象P2；P1是locality机制对照，报告其完整P0证据，不事后改为新的第三候选选择流程。

## 条件解释与后续门槛
在分析代码内预注册：P1利用诊断仅当P1-P0 meanBA>0、AUC/F1Δ≥0、两RecallΔ≥−.01；P2利用与order shuffle仅当P2对P0/P1 meanBA均>0且两比较辅助指标保护通过。否则明确skip，不根据图挑选activity/wrist/patch。
符合门槛时，描述patch attention分布和residual/original norm、subject probability effect；order shuffle固定seed20261008，只打乱有效embedding，不改embedding或mask。按15split描述，非新候选、非超参数选择。
DD subtype仅P2正式RETAIN后描述，不能据此reweight/resample。P3 adjacency pretext仅P2正式RETAIN后方可进入：同subject/activity/wrist相邻positive、|j-i|≥2negative；training-only192→32→1 head；λ.1在前10% planned optimizer steps线性到0，固定、不搜索。P0/P1/P2阶段不实现或运行P3。若未满足，STOP，不换window/stride/Transformer/pretext救援。WSSL+Patch不在本阶段启动；Frequency组合始终禁止。

## 输出与Git
独立artifact目录，不覆盖正式资产。六milestones audit/extraction/P1/P2/preformal/final分别commit+tag+push。仅code/config/aggregate tables/report/manifest/hash，排除原始数据、clinical个体表、privatecheckpoint、individualprediction、大cache。中文报告Q1–Q4及RETAIN/REJECT/INCONCLUSIVE，完整正负结果/限制写README；development-only，不写论文。
