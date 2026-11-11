# 公开来源检索记录与边界

检索日2026-11-11。此为有明确任务边界的多轮资料调查，不宣称PRISMA穷尽检索或全球所有数据已排除。不凭期刊名称推断JCR分区、SCI Q1/Q2或CCF等级；未取得对应年份官方分区证据，均不作等级断言。重点覆盖npj Parkinson's Disease、Scientific Data、TBME、TNSRE、临床运动障碍研究、PhysioNet、Synapse、机构仓储和官方作者代码页。原始论文、官方数据入口与可读性分别记录；代码开源不等于个体数据开源。

代表性检索式（同主题曾作标题、采样率和data availability限定复核）：

- Parkinson differential diagnosis essential tremor wearable inertial dataset bilateral wrist open data
- Parkinson atypical parkinsonism multiple sclerosis smartwatch inertial open dataset PD differential diagnosis
- "Parkinson" "essential tremor" wearable "dataset" "data availability" -filetype:csv -site:physionet.org
- "Intelligent differentiation" "temporal validation" Parkinson wearable 2026
- "Quick computer aided" "patients" "data" finger tapping
- "A Wearable Multi-segment" "100" "Hz" / "available" / "tasks"
- "Parkinson" "differential" "dataset" "TBME" inertial -filetype:csv -file_list -site:physionet.org
- "Parkinson" "essential tremor" "dataset" "JBHI" wearable -filetype:csv -file_list -site:physionet.org
- Parkinson PPMI wearable Verily study data access watch 2025
- Parkinson at Home validation dataset radboud 25 100 Hz
- PDWearML dataset Zenodo wrist Parkinson healthy 2025
- "BIOCLITE" dataset Parkinson 50 Hz
- Oxford real world mobility Parkinson dataset open bilateral accelerometry
- "Mobilise-D" "dataset" "Parkinson" "multiple sclerosis" data access wearable
- "Continuous observation of Parkinsonian symptoms" dataset 66 2026 doi
- "Daphnet" dataset "10" "64" official
- "WATCH-PD" "data availability" "3DT"

部分标题/路径限定查询返回无关论文，未作为证据。候选可以被检索命中，但没有原始来源支持的数值留unknown；例如NeurDNet各诊断人数与多项数据的采样率不从综述补填。论文中多次试验、双手记录和滑窗数量均不得计作人数。

网页可读性限制：PMC有时返回CAPTCHA，后续从同一原始文章的索引/可读页面复核；Nature部分页面跳转身份服务，使用官方索引PDF段落核对；IEEE文章入口有机器人检查，但官方作者版本PDF可读；IEEE DataPort若干入口不可访问；Synapse只呈现公开项目元数据，完整现行访问条件未呈现；DRUM DOI入口与Radboud/Bristol部分入口访问失败或超时。可访问性的技术失败不等于数据不存在或法律拒绝，矩阵分别标记。

仅通过GitHub连接器读取三个官方作者README：Belgrade finger-tapping、PDWearML、OxWearables CAPTURE-24。未clone仓库、执行其安装命令、打开个体传感器/元数据文件或下载数据。公开README中的字段模式、研究规模和通用ID示例仅用于格式核查，未保存逐人行。

永久披露：此前outer性能文字接触、旧manifest整表和旧NPZ整数组读取、Phase6公开搜索摘要个体元数据暴露仍有效。本轮首轮宽检索再次意外返回PADS公开file_list个体行摘要；未主动打开该文件、未持久化这些行、未将其加入实验或数据表。后续查询排除csv/file_list；PADS只打开官方项目landing，不打开个体文件。不能宣称浏览从未接触个体信息。新研究产物只保存研究级元数据、链接、证据边界与汇总。

核心来源逐行列在DATASET_COMPATIBILITY_MATRIX.csv和DATASET_DISCOVERY_REPORT.md。所有外部个体数据下载、申请、联系、代签协议动作均0。
