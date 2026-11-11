# Phase 7 最终研究报告

**保留Frozen WSSL-STR，不改变任何正式权重、norm、预处理或决策。** 本轮完成20项数据源调查、五项锁定资产核验、严格内存adapter及25项服务器合成测试、独立评价协议和reference来源审计。尚未取得直接可用的独立PD/DD队列，也未恢复原reference完整外部评分状态，因此没有真实外部性能、没有独立复核成功结论。新增训练/优化器更新/分类器/温度/阈值/Scaler/PCA拟合、外部个体下载、申请/联系均0。

## 完整工作状态

| 工作 | 状态 | 证据及保留缺口 |
| --- | --- | --- |
| 最新GitHub/服务器核对 | PASS |起点6de6480fce87920a899181fa7c8586c01ce616f9；mainline45文件一致 |
| 必读历史与源码 | COMPLETE_WITH_DISCLOSURE |19文档全文工具扫描、当前hash；Python全文/AST；屏蔽3行outer数值，未重新读旧outer性能 |
| 五项模型资产及205人norm来源 | PASS |实际文件SHA全匹配、checkpoint内部norm一致 |
| 数据源扩展 | COMPLETE inventory20项 |16个新增条目；全字段矩阵含unknown，无完整合格独立队列 |
| adapter与合成correctness | PASS_SYNTHETIC_ONLY |25检查、两次确定输出、BN/全部state不变；真实parser/设备映射未验收 |
| 独立评价协议 | DESIGN_READY_WITH_PRE_DATA_GATES |主指标/CI/盲法/NA规则已定；数据字典相关子组检验须揭盲前进一步冻结 |
| reference来源恢复 | PARTIAL |原算法/最终query分数/ID来源完整，原reference Scaler/PCA/bank/训练OOF矩阵缺失 |
| 真正临床分类性能 | NOT_EVALUATED |全部external metrics=null，不将development15split均值代入 |
| 类条件覆盖独立复核 | BLOCKED |无新合格队列且缺原reference变换状态；不是假设被否定 |

## 数据候选优先级

| dataset | priority_score | compatibility | blocking_gap |
| --- | --- | --- | --- |
| PLA General Hospital PD/ET temporal study (Xia2026) | 52.500000 | Partial | Missing9tasks, placement/rate/raw-data-access/overlap unverified |
| FINGER-TAPPING-PARKINSONISMS-DATABASE | 47.500000 | Partial | No wrist mapping; Acc not established; no11tasks |
| Maryland HOME PD/non-PD parkinsonism (Khalil2025) | 45.000000 | Blocked | Wrong body site and activities; derived features cannot supply raw wrist streams |
| Multi-segment upper limb tremor PD/ET (IEEE2023) | 42.500000 | Partial | 4tasks vs11; release/axis mapping/permissions unknown |
| NeurDNet hand tremor study (2021) | 36.250000 | Partial | Counts per diagnosis, raw axes, placement and11task mapping unknown |


PADS的80分不进入此排序：已经明确重复同一390人。首选现存合作线索PLA PD/ET52.5分也缺9个任务，不能直接用于当前模型。没有任何排名可代替数据授权、受试者独立、临床标签与完整输入硬门槛。最佳完整验证路线仍是有临床方支持的PADS兼容独立采集；目前它只是计划，不是假设已有医院或人数。完整差异和原始来源见[调查报告](DATASET_DISCOVERY_REPORT.md)、[矩阵](DATASET_COMPATIBILITY_MATRIX.csv)、[访问准备](DATA_ACCESS_PREPARATION.md)。

## 十一项研究判断

1. **没有发现已经满足全部条件的独立队列。** 找到PD/ET、PD/PSP/MSA和非PD帕金森综合征等新临床线索，但没有同时满足授权、身份核查、双腕原始六轴及全部11任务者。调查范围之外不能断言不存在。
2. **现存优先线索为PLA总医院PD/ET双手研究**；首个完整模型验证最可能仍需新的同协议临床合作采集。Belgrade是公开来源明确的PSP/MSA标签资源；与前者的用途不同。
3. PLA仅ET作为DD、手背传感器和rest/posture2任务，与PADS多亚型DD、腕位、11任务不同；采样/格式与独立身份未核实。Belgrade为右手拇指/食指gyro200Hz而非双腕Acc/Gyro。两者均不可无条件合并或重标。
4. **模型不能直接应用于这些现有不匹配数据并称原任务外部验证。** 代码可处理合成正确结构，不代表临床适用性。只在PD/DD临床定义、完整活动和设备映射均核实后，才有固定模型推理的工程条件，仍需独立性能证据。
5. **需要格式适配器，已完成内存接口合成验收。** 名称/单位/规则采样转换允许，真实文件parser、时钟和坐标/放置仍待数据方信息；重采样/设备滤波/放置差异可改变分布。缺腕/任务/临床DD不能靠格式转换解决。
6. **锁定checkpoint和normalization完整可追溯。** 五项SHA全匹配，norm与checkpoint相同且来自原205人train；分类侧143172+冻结encoder10457408参数。仍是seed42/context0/inner0这个具体模型，非ensemble或全390人重新训练。
7. **原reference算法只能部分恢复。** 原source、bank身份、最终query评分和cutoff可核对；缺原参考feature Scaler/PCA/bank向量与完整reference训练OOF矩阵，无法精确对新query评分。45个fusion二维Scaler存在，但不是原1024D参考Scaler，不以重新拟合补齐。
8. **Phase6交互有资格作为独立可证伪假设，尚无执行资格。** 需要合格新队列和完整旧状态；否则只能另授权登记新方法，不混用新旧分数/阈值。未来直接检验class×distance交互，不比较两类各自显著性，不训练gate。
9. **development证据较充分的结论**：Frozen外部SSL prior的既有匹配开发增量、90次成对预测重现、实际layer5表示变化和直接类条件交互。它们仍属重复development，不能据此证明临床机制。
10. **缺独立临床证据的结论**：这个具体冻结实例的外部判别/校准/设备与DD亚型稳定性；类条件覆盖交互能否重复及其病理原因；表示漂移是否解释个体获益。旧outer、PD/HC或同人随访不能代替。
11. **无充分理由重启深度训练。** 最优先缺口是数据和可追溯资产，而非新模块。Adapter/LoRA/解冻、fusion/门控/拒绝分类器、阈值/CE/Patch搜索保持停止。

## 证据等级与准备程度

| 层级 | 当前证据 | 可支持的判断 |
| --- | --- | --- |
| 工程资产与合成正确性 |本轮实际哈希/strict load、25测试通过 |固定内存推理接口可复核；不是临床验证 |
| 同队列development |Phase6 error交互.439291，splitCI[.326885,.557102]、BHq.000183；另一pooled patient-cluster估计.422650[.237459,.603781] |值得独立复核；两个估计对象不同，不能当两次独立复制 |
| 独立临床 |无合格数据/授权/身份核对结果 |全部性能与交互NOT_EVALUATED |
| 机制因果 |无设计识别DD异质性/任务质量/几何 |仅假设，不把高u称临床异常 |

总体准备程度为“模型锁定+合成工程准备+评价设计完成；临床数据、真实格式验收与原reference外部评分受阻”。不写没有依据的准备百分比。后续每个方向的假设、成本和停止条件见[NEXT_STAGE_RESEARCH_PLAN](NEXT_STAGE_RESEARCH_PLAN.md)；所有真实访问步骤须新获具体授权，未发送申请或消息。

历史outer性能文字接触、旧manifest整表/旧NPZ整数组读取与Phase6公开搜索结果的个体元数据摘要暴露永久保留。本轮宽检索再次意外返回公开PADS个体元数据摘要，未主动打开、保存个体行或用于实验。本轮未读取旧outer性能；历史文档逐行扫描时屏蔽3行可能涉及outer数值的段落，不用其作性能证据。保护哈希读取完整文件字节、旧私有分数表和fit/query身份元数据仍属应用层审计，不能称OS物理隔离或从未接触个体信息。15重叠development splits、seed和患者cluster区间不构成独立临床验证。

最终代码/协议/研究级兼容矩阵/汇总/审计通过既有选择式Git流程发布；privates和logs留服务器。目录 `/home/zyt/deep_final/artifacts/independent_validation_phase7_20261111/`。本轮完成后停止，不自动启动Phase8。
