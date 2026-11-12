# L1 Precision Rebuild Phase9 Level2固定协议

正式依据4da80813a98586acf23a89507359e4ca93689289。按audit→solver-test→pilot→build-data→fit-normalization→train-wssl→WSSL报告归档→train-str→analyze→integrity-check→publish-dry-run→publish执行。A/B数值/独立性/CPU验收未全通过，不导出完整数据或训练。任何待用轴证书失败，不删除患者、不混合旧新数据、不生成完整成功标记。

私有产物使用restricted_manifests/（对应用户建议的private），以兼容既有发布器的明确排除规则；子目录manifests/numerical_certificates/processed_signals/normalization/checkpoints/predictions/subject_diagnostics。原始数据、legacy信号/norm/split/源码/权重/HarNet全只读，新增记录版本标记，错误组合fail closed。无hardlink写旧文件、无依赖安装或全局包修改。

数学目标固定完整原始series P=.5||y-x||²+50||D2x||1，D2为索引[1,-2,1]，无时间缩放/标准化/lambda搜索。生产float64、每轴独立初始化和停止，不共享患者数值warmstart/rho/容差。先评测独立停止+gap的ADMM参考；允许为满足同目标精度/成本锁定一种独立数值算法，依据证书和CPU而非分类性能。最终生产source/config/init/maxiter/failure及依赖在全量导出前冻结。任何solver失败显式返回FAIL，无仿射/旧算法回退。

E_aff固定为.5||y−x_aff||²，该条自身合法仿射参考，与Phase9witness一致。证书完整原序列计算，dual q满足|q|∞≤50，D(q)=y·D2ᵀq−.5||D2ᵀq||²，真实q与ADMM scaled dual区分。raw gap=P−D不clip，不使用max(gap,0)。接受gap≤1e−8+1e−4E_aff且gap≥−64*eps*(1+|P|+|D|+sum|y*v|+50sum|x|)，v=D2ᵀq；微负值独立标记。任何更负gap、非有限、stationarity>1e−9*(1+max|y|)或dual不可行失败。若投影q则记录投影前违反量/最大改变量，并以投影后q核验证书；不伪造可行性。

保存规则：原完整Acc求解→raw−trend→裁48→float32，Gyro仅裁48转float32；不新增filter/对齐/插值/裁1–5s。float32绝对误差≤一个float32 ULP加8eps64*(1+maxabs residual)的逐点保守界；量化误差与float64 gap分开，不称无损。保存前raw=trend+residual须满足5e−13*(1+maxabs raw)。同输入单独/混批/重排/轴分组/重复/进程和cache差异maxabs≤1e−7g。maxiter无证书不得生成成功标记。

CPU solver QA+pilot+候选完整导出累计上限21600秒(6CPUh)，按进程及子进程user+system CPU累计，walltime另报。每阶段/失败尝试入账，先有界benchmark再估全量；预计明显超预算或运行接近上限则BLOCKED_COMPUTE_BUDGET，不放宽精度。预算不包含后续GPU训练与normfit，但分别报其CPU/GPU/墙钟成本。真实pilot固定每活动按原rawSHA字典序前2条腕记录，共22条，分类结果出现前登记，覆盖1024/2048与全部11activity；不按模型误例或结果选样。

授权development split的train∪validation并集核对390PD/DD；只对此并集逐记录生成，与cohort数值无关。各run只读本inner train/query真实信号和对应HarNet逐人fixed cache，不建outer loader、不读excluded预测/性能。存creation raw/source/config/cert/output hash和版本；原子写入，续跑只复用全部匹配且证书PASS的记录。15份norm严格原activity/wrist/channel/full-length/epsilon算法inner-train-only fit，3seed与STR/WSSL共享各split同份，绑定数据manifest与version。

HarNet保持原official权重、raw解析/trim48/context/clip±3g/100→30/zero-extension/aggregation/frozen策略，优先复用Phase6已合法逐人缓存；不重提取、不读旧supervised embedding代替新信号训练。Frozen指外部HarNet，WSSL分类侧143172参数按原seed初始化训练，STR75524参数按其原formal factory训练。两组架构/原balanced CE/采样/augmentation/8batch/AdamW2e−4WD1e−4/betas.9,.999/cosine50min1e−6/clip5/50epochs/patience12/validationBA strictbest/AMPfalse/workers4/deterministic设置逐项审查锁定，无checkpoint warmstart。TF32/precision/cuDNN实际值和历史证据限度记录。不按分类结果换solver或recipe。

9指标Acc/BA/macro两概率列AUROC/MacroF1/PDrec/DDrec/proper unweightedNLL/两类平方和Brier/10等宽topconfidenceECE；DD正类,pDD>.5，tiePD。NLL优先rawlogits的float64logsumexp；历史若仅有合法概率则−log真类概率无clip，来源与数值精度区别披露。每split先平均3seed指标再15split等权，主表无ensemble。旧45行从真实历史stage/pred逐一提取，不用总均值复制。

预登记C1 WSSLnew−old：主familyBA/AUC2、次7；C2 STRnew−old与WSSLnew−STRnew：主4、次14。探索差中差(WSSLnew−STRnew)−(WSSLold−STRold)另独立BA/AUC2、次7。缺项按p1占完整family、自身p/qNA；不改族大小。配对10000split bootstrap/Wilcoxon/dz/rank-biserial/wins-ties-losses及within-split3seedSD。10000PD/DD分层subject-cluster，每人全部appearance/seed共用抽样权重，estimand同15split seed-first差；条件于已训练固定模型，不代表外部泛化。历史outer和重复development偏差永久保留。

性能提升WSSL的gate固定ΔBA>0、≥10/15wins、paired95CI下界>0、AUROC/F1不降、PD/DDRecall各降≤.01、BA/AUC within-split seedSD≤legacy1.25、correctness/provenance全通过。p/q和clusterCI完整报告，不事后增删门槛。数值/独立性/数据来源与PERFORMANCE_PROMOTION分开判定；未涨点不取消STR复核、不否定数值修复、不恢复旧耦合为独立方法。

device分层与新旧信号RMS/trend/差异/边缘内部/固定0–3/3–12/12–15/15–Nyquist能量仅辅助；少样本NA/抑制，不选型/删组/加设备输入。参数量、CPU预处理与GPUforward分离；GPU统一batch8/warmup3/repeat10同步计时，缓存forward不冒称在线端到端。无全HarNet重提取预算，超范围的端到端明确NOT_MEASURED。

本轮最多90正式run；smoke单列且不用于选型，训练失败/重试保留。完成报告和原发布器审计/远端回读后停止；不启动interaction/SupCon/adapter/LoRA/阈值/lambda/line边界/重力/坐标/时间戳或outer研究。
