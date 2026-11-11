# 原reference pipeline来源恢复审计

总体PARTIAL；原方法精确对新队列评分BLOCKED。不是“所有reference分数均不存在”，也不是“代码可见所以已经恢复”。审计范围为原reference artifact、Phase5和Phase6，目录外资产不作全局不存在断言；原创建代码本身未序列化所需reference变换对象。

## 算法来源已恢复

原typicality.py SHA256=f329439f96575205ca8da50fce3baa1cd913816de8d7eaa124b472129d741361，与B_freeze创建时source hash一致；全文和调用者源代码已读取、AST有效，未导入/执行其fit函数。common_reference先在各split train的每个activity/wrist1024D特征上fit StandardScaler，再PCA(n_components=min(16,n−1,d), whiten=false, randomized, iterated_power4, random_state=原SEED)，去掉近零奇异方向并L2归一化。PD/DD平衡bank按固定SHA顺序选min(nPD,nDD)人，各59–62人，仍来自该split train；cosine distance截到[0,2]、k5均值、自身排除，零范数距离1。先等均左右腕再均11活动；relative=dPD−dDD，正方向DD，u=min(dPD,dDD)。原strict5fold类内q95诊断，与后续Phase5的u q25/q75用途不同。

深度WSSL OOF、strict类内5fold距离、fusion用的3fold全链reference OOF是不同对象。135个深度OOF checkpoint/45个保存的deepOOF特征包不等于保存了reference OOF分数。原reference OOF函数只在内存返回逐人scores；其fit/query3fold manifest可以核对，但缺持久化的完整逐人reference训练矩阵。

## 实际资产状态

| 对象 | 状态 | 本轮证据与限制 |
| --- | --- | --- |
| 原算法与创建时source hash | COMPLETE | B_freeze和typicality一致 |
| 15split最终query逐人dPD/dDD/u/relative | COMPLETE，私有 | 唯一ID、query集合、u/relative公式与Phase5对应记录一致 |
| 原train/query及canonical excluded | COMPLETE，私有 |15split及reference/deep OOF3fold分区对齐；不展示ID |
| PD/DD bank身份与平衡规则 | COMPLETE，私有 | 不能由身份清单代替变换后bank向量 |
| u q25/q75与保存组划分 | COMPLETE |15split×3seed原组重现；阈值是旧validation quantile，不是OOF训练阈值 |
| reference Scaler完整状态 | MISSING_IN_AUDITED_SCOPE |5280行中有scaler_mean hash，缺各维mean/scale等完整状态值 |
| reference PCA完整状态 | MISSING_IN_AUDITED_SCOPE |有components hash/rank记录，缺完整components/mean/keep等值 |
| 变换后bank向量 | MISSING_IN_AUDITED_SCOPE |无可验证持久化状态，无法直接给新query计距离 |
| 完整训练reference OOF逐人score矩阵 | MISSING |只有来源分区/使用代码/聚合和最终融合参数 |
| 45个fusion二维Scaler | COMPLETE，另一个Scaler |NPZ header形状2，coef1×2；不能补回1024D参考空间 |
| 逐活动/腕reference query距离 | MISSING |只保存聚合dPD/dDD，不能分解 |

本轮检查630个序列化NPZ目录/必要header，未解码其中个体数组。最初恢复审计按关键字发现45个scaler命名文件并主动停止；随后核实它们是fusion二维Scaler，并非reference feature Scaler。失败v1脚本和原日志保留，active审计明确分类后通过。未忽略这些候选、没有把二维Scaler误称缺失，也没有把它当原reference几何已恢复。

历史audit_bank_norms曾重fit5280个固定变换以记录norm/hash；本轮只核对该记录存在且hash格式有效，**没有再次运行**。哈希不可逆，理论上根据原数据重算并不等于已有状态可直接部署；本轮任何重fit均NOT_AUTHORIZED。

Phase6直接交互适合进入可独立证伪的研究假设，但原算法的“独立复核执行”仍受缺失状态与合格队列双重阻断。新方法如后续单独授权必须另立版本、保存所有状态/OOF/fit-query边界，不混同旧方法，不训练新的gate或fusion。

历史outer性能文字接触、旧manifest整表/旧NPZ整数组读取与Phase6公开搜索结果的个体元数据摘要暴露永久保留。本轮宽检索再次意外返回公开PADS个体元数据摘要，未主动打开、保存个体行或用于实验。本轮未读取旧outer性能；历史文档逐行扫描时屏蔽3行可能涉及outer数值的段落，不用其作性能证据。保护哈希读取完整文件字节、旧私有分数表和fit/query身份元数据仍属应用层审计，不能称OS物理隔离或从未接触个体信息。15重叠development splits、seed和患者cluster区间不构成独立临床验证。
