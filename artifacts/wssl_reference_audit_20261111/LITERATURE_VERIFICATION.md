# 原始论文与适用边界核查（2026-11-11）

正式论文集/元数据可以确认发表位置；详细受试者隔离若未取得全文则标待核验，不能用摘要替代。未安装依赖、下载代码仓库或模型权重。以下不会触发新训练。

| 论文 | 可核验正式来源与任务/模态 | 受试者隔离 | 本项目状态与借鉴边界 |
|---|---|---|---|
| PaAno，Park/Kang，ICLR2026 | [正式全文](https://proceedings.iclr.cc/paper_files/paper/2026/file/5991785875cf5eee9c29e2ed7d26d1c0-Paper-Conference.pdf)：正常序列训练，TSB-AD单/多变量时间点异常检测 | 任务不是PD/DD受试者分类；不能声称其协议给本项目提供跨患者验证 | 本次只借参考距离概念。未做patch CNN/triplet/时序邻接pretext；patch instance normalization可能去除本任务幅度信息，正负与邻接假设不能直接移植 |
| XTransfer，Zhang等，ICML2026 | [PMLR306:160794–160814](https://proceedings.mlr.press/v306/zhang26iy.html)：跨模态少样本人类感知，model repairing/layer recombining | 具体训练/测试subject协议待全文核验；官方PDF端点被当前读取器以octet-stream拒绝 | 未执行完整XTransfer；既有layer5/adapter失败不否定整个迁移家族，本次不做新层搜索 |
| Bio-Inspired SSL / BioPM，Tarale等，ICML2026 | [PMLR306:119735–119757](https://proceedings.mlr.press/v306/tarale26a.html)：腕部Acc movement segment token+masked Transformer；NHANES约28k小时/11k人 | 摘要明确6个subject-disjoint HAR benchmark；详细划分/标签预算待全文核验，官方PDF读取同上失败 | 本项目Phase3E已测受限Acc neural encoder导出，不是完整BioPM。不能列为完全未测，也不能把受限配置负结果推广完整pipeline |
| Cycle-Aware Masked SSL for PD Diagnosis，Yang等 | [DOI原始登记入口](https://doi.org/10.1109/TBME.2026.3712830)、[PubMed登记](https://pubmed.ncbi.nlm.nih.gov/42441478/)；先前检索元数据显示IEEE TBME，2026-07-13 ahead of print | 正式全文取得失败，详细subject隔离/模态/标签预算均BLOCKED待核验；PubMed此次打开只返回HHS页脚，不伪称取得全文 | 未测试cycle-aware mask；既有随机/连续mask不等价。需先有本数据周期结构/失败机制证据，不启动预训练；也不以其accuracy设本项目目标 |
| Deep Nearest Neighbors，Sun等，ICML2022 | [正式PMLR162](https://proceedings.mlr.press/v162/sun22d.html)：深度近邻OOD检测 | 不提供本项目独立subject验证 | 普通kNN是经典对照，不单独作为创新或疾病因果证据 |
| Calibration，Guo等，ICML2017 | [正式PMLR70:1321–1330](https://proceedings.mlr.press/v70/guo17a.html)：神经网络置信度校准 | 本项目仍须独立审计OOF来源 | 正T诊断不改变单模型排序/.5分类；不能宣传为AUROC改进 |

PaAno全文可读；XTransfer/BioPM正式会议年份、标题、作者及卷页已从论文集核验。Cycle正式全文、最终卷页/发表状态仍需官方出版全文进一步核验，不冒充已完成。所有JCR分区/CCF分级均未取得可审计权威年度依据，标**待核验**；不称预印本为SCI Q1/Q2或凭“ICML级别”推定临床有效性。

本次没有证明新增稳定分类收益。CE reduction或普通kNN本身不构成方法创新；未来贡献仍需稳定任务增量、机制证据、完整对照和独立验证。疾病人群间参考距离不能称正常/异常生理判别，分类结果也不能冒充连续严重度回归。
