# Phase5 最终报告

保留Frozen WSSL-STR；新深度训练0、新轻量模型/Scaler/PCA拟合0、新神经推理/HarNet提取0。完成现存分数的分析，明确阻断缺失资产所需的子分析，不把部分来源审计冒充完整数值重放。

| work | status | reason |
| --- | --- | --- |
| 资产审计 | 完成、存在MISSING子项 | 45+45+135；60 reference ID链 |
| 5A分布/互补/覆盖 | 完成 | 固定分数、新统计；无新分类器 |
| 5B fusion恢复 | 完成 | 45概率精确重放；不改参数 |
| 5B OOF reference联合分布 | BLOCKED | 逐人OOF分数/变换状态未保存 |
| 5C预测变化 | 完成 | 全体与PD/DD、错误组、各seed |
| 5C表征漂移 | BLOCKED | 未保存post-adaptation同层特征 |
| 新训练/拟合/神经推理 | 未运行 | 全部0 |
| 发布快照 | 准备并核验；最终提交receipt另存源目录外 | 仅代码、协议、报告、聚合结果，不在报告中自指提交SHA |


| question | evidence_strength | support | limit |
| --- | --- | --- | --- |
| H1：部分不同且正确的信息 | 中等、描述性 | 有gross rescue/错序救回，但整体net<0，条件错序正确率<.5 | 独立、可部署增量未建立 |
| H1：无法形成稳定净收益 | 较强、限当前固定方法 | PD损失、BA下降、stable-correct伤害，cluster一致 | 不否定所有表示/算法家族 |
| H2：固定操作点的数值解释 | 强、代数可重放 | 45概率误差2.22e-16，负b+有限DD证据→PDD降低 | 不是负截距应被修改的授权 |
| H2：OOF/full总体均值迁移是主因 | 不支持该强命题 | 最终均值迁移近0、CI跨0；reference均值向DD | 联合/类条件reference OOF缺失，不能排除其他分布变化 |
| H3：预测变化具有异质性 | 较强、描述性 | 43改善/46恶化/301不变 | 指标聚合口径不同不能混写 |
| H3：表征变化导致改善机制 | BLOCKED | adapted同层成对表征缺失 | 不能给cosine/L2或因果结论 |
| 独立外部泛化 | 未验证 | 无新独立队列/outer evaluation | 三种开发不确定性均不能替代外部验证 |


| priority | candidate | research_hypothesis | existing_evidence | single_change | cost | stop_condition |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 补齐OOF/reference来源资产 | 保存状态缺口使迁移解释不可识别 | 当前缺逐人reference OOF与完整transform states | 仅增加合法既有流程的状态导出与创建时hash；不改评分算法 | 0深训练；若需重放须另授权，时长未测 | 任一fit/query来源无法独立证实则停止机制推断；不顺带重拟合头 |
| 2 | 独立数据上的覆盖方向复核 | u不是跨疾病类别通用可靠性指标 | PD与DD高低u错误方向相反，cluster均有同向证据 | 在独立合法队列重复同一冻结评分和预定class interaction诊断 | 0新训练；新数据访问/推理需另授权，成本未知 | 无法重复或存在队列/设备混杂则不发展为门控，不用真实类别路由 |
| 3 | 既有checkpoint成对表征导出 | 适配改变不同患者的表示但不一定带来一致收益 | 43改善/46恶化；缺成对adapted表示 | 仅导出同输入同层Frozen/adapted特征，不改模型 | 45对已有模型推理；0训练，实际时长未测 | 身份/预处理不一致即停；漂移不能对应净改善则不进入新适配训练 |
| NO-GO | 新深度模型训练 | 当前没有稳定可部署增量支撑 | reference净损失、融合REJECT、local无稳定增量 | 本轮无单一结构改动获执行 | 预算0 | 先补来源与独立验证；不按个别人改善恢复搜索 |
| STOP | 同队列融合/阈值/CE/LoRA/Adapter/Gate/Patch/Gyro搜索 | 不得用不同命名重启失败或未过闸门候选 | 既有停止规则持续有效 | 不执行 | 0 | 不能用本轮描述性子组、单一AUC或训练内指标重新开门 |


正式六指标（原预测复核，非新模型）：

| condition | accuracy | ba | auroc | macro_f1 | pd_recall | dd_recall |
| --- | --- | --- | --- | --- | --- | --- |
| adapted | 0.751827 | 0.724402 | 0.768908 | 0.712829 | 0.790493 | 0.658310 |
| baseline | 0.745604 | 0.719644 | 0.759553 | 0.706589 | 0.782344 | 0.656945 |
| fusion | 0.752378 | 0.627638 | 0.763640 | 0.631057 | 0.927570 | 0.327707 |
| relative | 0.675813 | 0.675842 | 0.735639 | 0.647599 | 0.675737 | 0.675946 |


1. Dual-reference有部分WSSL未做出的正确排序/预测，但尚不能确认独立、可稳定利用的信息增量。WSSL错序pairs内relative正确率.416706；全体每unit救回9.51同时破坏16.78，净−7.27。不能只在错误组报获救。
2. 不能稳定修复全体错误。PD recall Δ−.106607、DD Δ+.019001（CI跨0），BA Δ−.043803；stable-error获益伴随stable-correct损失，不能逐患者路由。
3. Fusion直接数值原因可恢复：b平均−1.0225，DD两项平均证据+.422764、+.209945不足抵消负截距，DD平均f−.389791；DD预测率.345980→.147096、DD Recall .656945→.327707。概率重放误差2.22e-16，未发现已有部署数值中的全局方向/编码错用。AUC+.004087的split与cluster CI均跨0，排序与固定操作点不是同一件事。
4. OOF/full确有尺度和类条件变化，但最终整体均值移位−.005711、CI跨0；reference移位反而向DD。没有证据把失败单独归因“总体迁移”。缺reference OOF逐人联合分布和transform state，完整机制归因BLOCKED；自然类先验与balanced-CE/readout目标差异仅是相容解释，未做因果识别。
5. Adapted改变一部分人的预测，非一致改善。四appearance、三seed均值概率诊断下43改善、46恶化、301不变；正式单seed BA仅+.004757且cluster CI跨0。表征机制本轮无法回答，缺同层adapted特征，不能用预测变化冒充cosine漂移证据。
6. 最大未解问题是独立泛化、可靠性可识别性与可追溯资产完整性。u高低错误关系在PD/DD方向相反，不是通用可靠性；旧outer文字接触永久保留，同队列多轮开发不能提供独立验证。
7. 当前无新深度训练方向获证据支持。先补状态与独立数据复核，再决定是否值得另授权的成对既有模型推理；同队列fusion/threshold/CE/adapter/patch/gyro搜索停止。
8. 可检验新假设是类条件覆盖几何与读出源分布/操作点不匹配；这是方法学贡献的研究入口，尚非已验证新算法。候选的单一改动、成本和停止条件见上表，不执行。

全部predeclared p/q、effect、seed/split SD、subject-cluster CI见STATISTICAL_APPENDIX和analysis。统计、源代码、失败脚本版本与日志均留存；缺失值明确NA。已有温度诊断、CE NOT_RUN_GATE_FAIL及所有历史REJECT不改写。输出只在/home/zyt/deep_final/artifacts/wssl_complementarity_phase5_20261111，mainline及历史正式产物不修改。


永久继承上轮偏差：旧V8 outer汇总文字曾被读取，旧manifest/缓存曾整表/整数组读取，不能撤销或将本研究包装为完全未接触outer。本轮没有解码全球HarNet/stage4缓存、读取原始IMU、创建outer-test loader或访问outer performance；只用固定development产物。当前特征包先验证IDs属于本inner-train/validation，再读取保存分数；全局development融合CSV逐行过滤。哈希核验仍按完整文件读取字节，CSV过滤也不构成物理零无关行I/O；不声称系统级完全隔离。分区test IDs仅用于排除名单元数据。

本轮属于同一390人、15个重叠development splits的重复开发。正式指标先同split平均三个seed的指标；三seed概率均值仅用于错误分组。split bootstrap、seed SD、subject-cluster CI均不能充当独立外部泛化证据。cluster bootstrap按PD276/DD114分层，10,000次，同一人的全部appearance/seed共享采样次数，仅固定已拟合模型条件下的不确定性；不覆盖重训练和多轮开发选择偏差。不同条件/估计对象的CI是否跨0，不构成它们之间差异的检验。
