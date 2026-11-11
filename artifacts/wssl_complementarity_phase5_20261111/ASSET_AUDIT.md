# Phase5 资产审计

225个checkpoint及预测/归一化/split通过当前hash与ID/标签审计：Frozen45、adapted45、deep-OOF135。旧审计对同SHA的135状态做过严格加载；本轮不重新实例化模型或加载其tensor做推理。

| asset | status | blocked |
| --- | --- | --- |
| saved reference OOF row-level score vector | MISSING | joint/class-conditional OOF reference distribution and exact training-fusion score replay |
| serialized fitted reference Scaler/PCA and bank vectors | MISSING | independent numeric replay of reference fit/transform chain; ID/hash records only |
| post-adaptation final HarNet/local/subject features | MISSING | cosine/L2/norm representation drift |
| adapted raw two-class logits | MISSING | raw DD logit displacement; probability-implied log odds remain available |


60条reference fit/bank ID记录（15×全训练与3个OOF子折）均与合法train/query/excluded ID、类别和等人数bank对齐；对应22块变换hash记录存在。它们不是完整的序列化Scaler/PCA/bank向量。逐人的reference OOF输入未保存；仅有原报告总体均值/SD、输入Scaler参数与来源脚本。故“可以独立重放整个reference链/完整fusion训练OOF联合分布”是BLOCKED，不能以报告写了PASS推定状态文件必然存在。

合法Frozen包保存local、SSL residual、bilateral、subject、structured与双logits；其当前hash、ID、标签和概率一致性已核验，但没有独立的创建时逐特征hash清单。来源依据是上轮已验证提取脚本/模型/概率记录，而非本轮重新推理证明每一个embedding。Adapted源码在forward里计算layer5后送原base，未导出post-adaptation表征。stage4_windows.npz是390人共享的pre-adaptation中间缓存，不能当作成对的最终adapted特征；未解码。

查找范围与文件路径/hash在私有feature_discovery_inventory；缺失是指定来源范围内未找到合法资产，不声称已搜索整台机器所有文件。没有用文件名相似的其他fold表示替代缺失。原分区full-model split包含canonical excluded outer IDs，元数据存在不等于outer performance被读取；本轮不构造任何信号dataset。

逐对象完整fit/query/excluded IDs、source checkpoint/feature/normalization/split SHA、seed/context放restricted_manifests/all_model_reference_lineage.json。公开asset_inventory与reference_asset_lineage_summary只含聚合、路径hash及状态，不上传受保护manifest。


永久继承上轮偏差：旧V8 outer汇总文字曾被读取，旧manifest/缓存曾整表/整数组读取，不能撤销或将本研究包装为完全未接触outer。本轮没有解码全球HarNet/stage4缓存、读取原始IMU、创建outer-test loader或访问outer performance；只用固定development产物。当前特征包先验证IDs属于本inner-train/validation，再读取保存分数；全局development融合CSV逐行过滤。哈希核验仍按完整文件读取字节，CSV过滤也不构成物理零无关行I/O；不声称系统级完全隔离。分区test IDs仅用于排除名单元数据。
