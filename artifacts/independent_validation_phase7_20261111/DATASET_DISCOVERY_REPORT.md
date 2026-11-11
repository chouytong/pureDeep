# 数据源调查报告

截至2026-11-11，本轮扩展为20个数据源/研究条目，比原PADS/mPower/BEAT-PD/WATCH-PD四项增加16项。未确认任何已授权、独立性已核查且满足当前完整输入的PD/DD队列。此结论限定于本轮核查范围，不是全球不存在相关数据的证明。

## 准备优先级与硬门槛

评分=30×标签/4+25×传感器活动/4+20×独立可信度/4+15×访问可行性/4+10×样本亚型/4。它是研究人员基于已验证元数据的保守准备判断，非模型性能、成功率或经过外部验证的量表。未知项不加乐观分；各行分项和理由可审查。

标签4=明确PD及多类DD诊断，3=ET单亚型，0=无合格DD或不能核实；传感器活动4=全11任务双腕六通道已证实，2=已知多项双手上肢惯性任务但缺完整映射，1=相关腕/手信号或少量任务，0=明显不符或未知。独立性1只表示不同研究来源有依据，绝非身份核对通过；0为已知重复或不能判断，4须真正核查通过（本表无此项）。访问4=明确当前开放许可，2=明确proposal/request路径或开放入口但条款缺口，1=线索/入口尚待核实，0=无证据。UCI开放研究页面并不构成用户本轮下载授权。样本/亚型仅PD/DD可用人群计分：4=两类至少100且有亚型，2=两类至少20且有DD定义，1=有明确小型诊断样本，0=缺DD/数量未知。未用的分档不用于补分。

已知重复的PADS即使得80分，也被独立性硬门槛排除；不在外部数据优先队列中。其余具有诊断标签的线索排序如下。A_ET/A_DD_SUBSET只说明标签方向相关，**不是已经满足A类完整兼容的队列**。

| dataset | priority_score | compatibility | blocking_gap |
| --- | --- | --- | --- |
| PLA General Hospital PD/ET temporal study (Xia2026) | 52.500000 | Partial | Missing9tasks, placement/rate/raw-data-access/overlap unverified |
| FINGER-TAPPING-PARKINSONISMS-DATABASE | 47.500000 | Partial | No wrist mapping; Acc not established; no11tasks |
| Maryland HOME PD/non-PD parkinsonism (Khalil2025) | 45.000000 | Blocked | Wrong body site and activities; derived features cannot supply raw wrist streams |
| Multi-segment upper limb tremor PD/ET (IEEE2023) | 42.500000 | Partial | 4tasks vs11; release/axis mapping/permissions unknown |
| NeurDNet hand tremor study (2021) | 36.250000 | Partial | Counts per diagnosis, raw axes, placement and11task mapping unknown |


最高的既有临床准备线索是PLA总医院Xia2026双手PD/ET研究（52.5/100），但只有2项任务、手背放置，缺其余9项及核实采样/原始格式，不能直接验证当前模型。其次Belgrade有公开作者数据入口和PSP/MSA标签，但两只手指的陀螺仪不是双腕Acc/Gyro。Maryland临床PD/非PD帕金森综合征样本量较大，但腰部移动性数据和已公布derived features不能代替腕部原始输入。IEEE多节段研究是相关上肢合作线索，未核实原始数据发布。由此不能把“拿到其中任何一项”当成当前模型可部署。

## 核查来源与差异

下列均为研究级来源；未访问个体文件。完整字段、unknown、具体人数/传感器/活动/授权和阻断原因见CSV。样本量不将双手、重复session、滑窗或assessments计为人。

- **PADS v1.0.0**：A；Blocked。 [官方入口](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/)；[原始论文/数据引用](https://doi.org/10.1038/s41531-023-00625-7)。Same390PD/DD; not independent。
- **PLA General Hospital PD/ET temporal study (Xia2026)**：A_ET_SUBSET；Partial。 [官方入口](https://pmc.ncbi.nlm.nih.gov/articles/PMC13591743/)；[原始论文/数据引用](https://doi.org/10.1186/s12911-026-03650-7)。Missing9tasks, placement/rate/raw-data-access/overlap unverified。
- **FINGER-TAPPING-PARKINSONISMS-DATABASE**：A_DD_SUBSET；Partial。 [官方入口](https://github.com/innovation-center-etf/FINGER-TAPPING-PARKINSONISMS-DATABASE)；[原始论文/数据引用](https://pmc.ncbi.nlm.nih.gov/articles/PMC10107087/)。No wrist mapping; Acc not established; no11tasks。
- **Maryland HOME PD/non-PD parkinsonism (Khalil2025)**：A_DD_SUBSET；Blocked。 [官方入口](https://doi.org/10.13016/2qhy-961s)；[原始论文/数据引用](https://doi.org/10.3390/biomedicines13030572)。Wrong body site and activities; derived features cannot supply raw wrist streams。
- **Multi-segment upper limb tremor PD/ET (IEEE2023)**：A_ET_SUBSET；Partial。 [官方入口](https://ieeexplore.ieee.org/document/10223245)；[原始论文/数据引用](https://ieeexplore.ieee.org/ielx7/7333/4359219/10223245.pdf)。4tasks vs11; release/axis mapping/permissions unknown。
- **NeurDNet hand tremor study (2021)**：A_ET_SUBSET；Partial。 [官方入口](https://pmc.ncbi.nlm.nih.gov/articles/PMC8099874/)；[原始论文/数据引用](https://doi.org/10.1038/s41598-021-88919-9)。Counts per diagnosis, raw axes, placement and11task mapping unknown。
- **Daphnet Freezing of Gait**：C_PD_FOG；Blocked。 [官方入口](https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait)；[原始论文/数据引用](https://doi.org/10.24432/C56K78)。NoDD; lower-body FOG task, missing wrists/Gyro。
- **UCI HAR Using Smartphones**：D_HAR；Blocked。 [官方入口](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)；[原始论文/数据引用](https://doi.org/10.24432/C54S4K)。HAR volunteers != PD/DD; wrong body/task。
- **BIOCLITE v2**：B_PD_HC；Partial。 [官方入口](https://zenodo.org/records/20644246)；[原始论文/数据引用](https://doi.org/10.2196/72820)。8!=11tasks; noDD; bilateral mapping absent。
- **REMAP real-world mobility**：B_PD_HC；Blocked。 [官方入口](https://doi.org/10.5523/bris.21h9f9e30v9cl2fapjggz4q1x7)；[原始论文/数据引用](https://doi.org/10.1038/s41597-023-02663-5)。Open skeleton != openIMU; noDD and wrong tasks。
- **WATCH-PD**：B_PD_HC；Blocked。 [官方入口](https://www.nature.com/articles/s41531-024-00721-2)；[原始论文/数据引用](https://doi.org/10.1038/s41531-024-00721-2)。No clinical DD; device/task mapping missing。
- **BEAT-PD DREAM (CIS-PD/REAL-PD)**：C_PD_SEVERITY；Blocked。 [官方入口](https://www.synapse.org/Synapse:syn20825169/wiki/600903)；[原始论文/数据引用](https://pmc.ncbi.nlm.nih.gov/articles/PMC10047543/)。NoDD task; heterogeneous protocols。
- **PDWearML**：B_PD_HC+C_SEVERITY；Partial。 [官方入口](https://ieee-dataport.org/documents/pdwearml-leveraging-daily-activities-fast-parkinsons-disease-severity-assessment-wearable)；[原始论文/数据引用](https://pubmed.ncbi.nlm.nih.gov/41447488/)。NoDD; full bilateral/task/units mapping unverified。
- **Parkinson@Home tremor reference release**：B_PD_HC；Partial。 [官方入口](https://data.ru.nl/collections/di/dcmn/DSC_pdtremor_t0000359a_966)；[原始论文/数据引用](https://doi.org/10.1038/s41531-025-01056-2)。NoDD; daily life task mismatch; official release counts differ from analysis24/24。
- **PPMI Verily digital substudy**：B_PD_HC+C_LONGITUDINAL；Blocked。 [官方入口](https://www.ppmi-info.org/access-data-specimens/download-data)；[原始论文/数据引用](https://doi.org/10.1038/s41531-025-01034-8)。No compatible DD labels/11bilateraltasks; patient-level overlap unknown。
- **COPS continuous observation (2026)**：C_PD_SYMPTOMS；Blocked。 [官方入口](https://doi.org/10.1038/s41597-026-06999-6)；[原始论文/数据引用](https://pubmed.ncbi.nlm.nih.gov/41957027/)。NoDD/Gyro/11task mapping。
- **Mobilise-D TVS / standardization examples**：E_TARGET_LABEL_NOT_HARMONIZED；Partial。 [官方入口](https://doi.org/10.5281/zenodo.7185429)；[原始论文/数据引用](https://doi.org/10.1038/s41597-023-01930-9)。MS cohort cannot be relabeled clinicalDD without protocol; 11task/wrist mapping absent。
- **PD-BioStampRC21**：B_PD_HC；Blocked。 [官方入口](https://doi.org/10.21227/g2g8-1503)；[原始论文/数据引用](https://doi.org/10.1038/s41531-021-00248-w)。NoDD; noGyro; placement/task mismatch。
- **mPower**：B_PD_HC；Blocked。 [官方入口](https://www.synapse.org/Synapse:syn4993293/wiki/247860)；[原始论文/数据引用](https://doi.org/10.1038/sdata.2016.11)。PD/control notPD/DD; handheld task mismatch。
- **CAPTURE-24**：D_HAR+E_NO_CLINICAL_TARGET；Blocked。 [官方入口](https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001)；[原始论文/数据引用](https://github.com/OxWearables/capture24)。NoclinicalPD/DD/Gyro/11task mapping。


Parkinson@Home需区分官方release描述25PD/24controls与原论文招募25/25、特定分析24/24；不把不同范围数字混合。REMAP的开放部分主要是coarsened skeleton，原始腕加速度在controlled部分；不能因有open DOI就声称IMU开放。PDWearML作者代码MIT不等于DataPort数据许可。Mobilise-D标准化示例不等于完整临床队列，MS等疾病标签也不能未经诊断协议就自动映射DD。PPMI prodromal/SWEDD不自动变DD。NeurDNet缺各诊断人数/传感器细节的字段保持unknown。

## 访问准备

本轮没有申请、联系作者、代签协议或下载外部个体数据。拟申请材料与原文明确条件分别列在DATA_ACCESS_PREPARATION.md。当前最现实的完整模型验证路径是与临床方另行协商真正PADS兼容的独立PD/DD采集；尚未确认医院、人数、预算和同意范围，不能写成已拥有队列。探索现存ET/PSP/MSA数据仍有方法学价值，但若任务不完整应另立任务，不能悄悄删活动或重训练当前模型。

完整检索式、失败页面、来源层级和意外摘要暴露见SEARCH_LOG.md。不声称未经核对的期刊分区或CCF等级。历史outer性能文字接触、旧manifest整表/旧NPZ整数组读取与Phase6公开搜索结果的个体元数据摘要暴露永久保留。本轮宽检索再次意外返回公开PADS个体元数据摘要，未主动打开、保存个体行或用于实验。本轮未读取旧outer性能；历史文档逐行扫描时屏蔽3行可能涉及outer数值的段落，不用其作性能证据。保护哈希读取完整文件字节、旧私有分数表和fit/query身份元数据仍属应用层审计，不能称OS物理隔离或从未接触个体信息。15重叠development splits、seed和患者cluster区间不构成独立临床验证。
