# 受试者划分与泄漏类型

FACT：15canonical train/validation/excluded集合逐一互斥、并集390；45norm对应fit成员精确匹配；所有activity/wrist/window按patient继承集合，原始信号精确hash跨subject重复组0。每人出现在4个validation development unit，重复appearance不能算4人。本轮没有调用构建outer dataset/loader或训练entry。

需区分：Label leakage——未发现mean/std或记录局部L1使用validation标签拟合；Subject leakage——当前ID集合互斥/无跨人exactduplicate；Preprocessing leakage——逐记录label-independent detrend/外部冻结HarNet提前导出本身不等于跨人fit，norm必须split后train-only；Validation selection bias——原checkpoint按val BA早停并多轮研究，因此仍存在；Historical outer exposure——历史文字、整缓存/manifest等接触永久保留，不能再宣称untouched。

原mainline FrozenEmbeddingCache初始化会整体解码390人pretrained及random数组，然后按ID选batch；这是数据访问边界弱点，不自动等于标签泄漏。旧监督feature提取包含全缓存接触及train embedding in-sample限制也保留。本轮仅查源码/hash，未解码全局feature缓存；源码旧normalization成员证明不等于OS物理隔离。原Dataset的trainaugmentation在split后，当前formal配置没有新增旋转/活动dropout。PCA/feature selection不在正式Frozen WSSL-STR预处理链，本轮也没fit。

全390人质量审计是本次明确授权的cohort探索，不按outer角色作新分析，不评估任何outer性能；全体也已在既往development反复出现。质量关联用了诊断标签及val成员元数据，今后的候选选择不能称未见全队列信息。整体历史偏差不由本轮PASS消除。


---
数据来源：服务器PADS v1.0.0的390人PD/DD任务子集，原始数据、既有v2信号及45 Frozen WSSL-STR checkpoint元数据；不是原库全部469人。统计单位为患者，同患者双腕先平均；原始记录/通道数只作描述性计数，不作为独立患者。15重叠development splits、seeds42/43/44用于来源核查，不是独立外部验证。缺失/不可计算值为NA，不补造；频谱超过记录Nyquist不外推；稀有交叉分层<10人的数值统计抑制，完整表仅私有。检验族132/440/880分别BH，患者bootstrap2000次；其余描述性表无额外显著性声明。接触全390人诊断标签及validation成员元数据用于质量关联，未读分类错误名单/分数，未构建outer loader、未解码outer性能/预测；历史outer接触永久披露。全队列探索会增加未来选择偏差，不能叫盲态独立验证。训练/分类器拟合/normalization重拟合均0，Level2未执行。
