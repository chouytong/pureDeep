"""Report actual completed audits; never fabricate external metrics."""
from common7 import *
import datetime,importlib.metadata
def save(name,s):(HERE/name).write_text(s.strip()+'\n')
def main():
 guard();data=pd.read_csv(HERE/'DATASET_COMPATIBILITY_MATRIX.csv');startup=load(HERE/'analysis/startup_integrity.json');synthetic=load(HERE/'analysis/synthetic_integrity.json');ref=load(HERE/'analysis/reference_provenance_integrity.json');hashes=pd.read_csv(HERE/'analysis/model_lock_hash_audit.csv');tests=pd.read_csv(HERE/'tests/synthetic_correctness.csv')
 ranked=data[data.target_category.str.startswith('A')&~data.dataset.str.startswith('PADS')][['dataset','priority_score','compatibility','blocking_gap']]
 disclosure='''历史outer性能文字接触、旧manifest整表/旧NPZ整数组读取与Phase6公开搜索结果的个体元数据摘要暴露永久保留。本轮宽检索再次意外返回公开PADS个体元数据摘要，未主动打开、保存个体行或用于实验。本轮未读取旧outer性能；历史文档逐行扫描时屏蔽3行可能涉及outer数值的段落，不用其作性能证据。保护哈希读取完整文件字节、旧私有分数表和fit/query身份元数据仍属应用层审计，不能称OS物理隔离或从未接触个体信息。15重叠development splits、seed和患者cluster区间不构成独立临床验证。'''
 save('DATASET_DISCOVERY_REPORT.md',f'''
# 数据源调查报告

截至2026-11-11，本轮扩展为20个数据源/研究条目，比原PADS/mPower/BEAT-PD/WATCH-PD四项增加16项。未确认任何已授权、独立性已核查且满足当前完整输入的PD/DD队列。此结论限定于本轮核查范围，不是全球不存在相关数据的证明。

## 准备优先级与硬门槛

评分=30×标签/4+25×传感器活动/4+20×独立可信度/4+15×访问可行性/4+10×样本亚型/4。它是研究人员基于已验证元数据的保守准备判断，非模型性能、成功率或经过外部验证的量表。未知项不加乐观分；各行分项和理由可审查。

标签4=明确PD及多类DD诊断，3=ET单亚型，0=无合格DD或不能核实；传感器活动4=全11任务双腕六通道已证实，2=已知多项双手上肢惯性任务但缺完整映射，1=相关腕/手信号或少量任务，0=明显不符或未知。独立性1只表示不同研究来源有依据，绝非身份核对通过；0为已知重复或不能判断，4须真正核查通过（本表无此项）。访问4=明确当前开放许可，2=明确proposal/request路径或开放入口但条款缺口，1=线索/入口尚待核实，0=无证据。UCI开放研究页面并不构成用户本轮下载授权。样本/亚型仅PD/DD可用人群计分：4=两类至少100且有亚型，2=两类至少20且有DD定义，1=有明确小型诊断样本，0=缺DD/数量未知。未用的分档不用于补分。

已知重复的PADS即使得80分，也被独立性硬门槛排除；不在外部数据优先队列中。其余具有诊断标签的线索排序如下。A_ET/A_DD_SUBSET只说明标签方向相关，**不是已经满足A类完整兼容的队列**。

{table(ranked)}

最高的既有临床准备线索是PLA总医院Xia2026双手PD/ET研究（52.5/100），但只有2项任务、手背放置，缺其余9项及核实采样/原始格式，不能直接验证当前模型。其次Belgrade有公开作者数据入口和PSP/MSA标签，但两只手指的陀螺仪不是双腕Acc/Gyro。Maryland临床PD/非PD帕金森综合征样本量较大，但腰部移动性数据和已公布derived features不能代替腕部原始输入。IEEE多节段研究是相关上肢合作线索，未核实原始数据发布。由此不能把“拿到其中任何一项”当成当前模型可部署。

## 核查来源与差异

下列均为研究级来源；未访问个体文件。完整字段、unknown、具体人数/传感器/活动/授权和阻断原因见CSV。样本量不将双手、重复session、滑窗或assessments计为人。

'''+''.join(f"- **{r.dataset}**：{r.target_category}；{r.compatibility}。 [官方入口]({r.official_entry})；[原始论文/数据引用]({r.original_paper})。{r.blocking_gap}。\n" for r in data.itertuples())+f'''

Parkinson@Home需区分官方release描述25PD/24controls与原论文招募25/25、特定分析24/24；不把不同范围数字混合。REMAP的开放部分主要是coarsened skeleton，原始腕加速度在controlled部分；不能因有open DOI就声称IMU开放。PDWearML作者代码MIT不等于DataPort数据许可。Mobilise-D标准化示例不等于完整临床队列，MS等疾病标签也不能未经诊断协议就自动映射DD。PPMI prodromal/SWEDD不自动变DD。NeurDNet缺各诊断人数/传感器细节的字段保持unknown。

## 访问准备

本轮没有申请、联系作者、代签协议或下载外部个体数据。拟申请材料与原文明确条件分别列在DATA_ACCESS_PREPARATION.md。当前最现实的完整模型验证路径是与临床方另行协商真正PADS兼容的独立PD/DD采集；尚未确认医院、人数、预算和同意范围，不能写成已拥有队列。探索现存ET/PSP/MSA数据仍有方法学价值，但若任务不完整应另立任务，不能悄悄删活动或重训练当前模型。

完整检索式、失败页面、来源层级和意外摘要暴露见SEARCH_LOG.md。不声称未经核对的期刊分区或CCF等级。{disclosure}
''')
 save('DATA_ACCESS_PREPARATION.md','''
# 数据访问准备（未提交、未联系）

| 路线 | 官方入口与已核实条件 | 建议先准备的材料（非冒称官方强制项） | 当前停止点 |
| --- | --- | --- | --- |
| PLA PD/ET | [原论文](https://doi.org/10.1186/s12911-026-03650-7)写明敏感数据不公开，向通讯作者reasonable request | 机构/PI、研究目的、伦理与数据安全说明；先问无个体信息的采集字典、原始六轴是否保留、是否仅2任务 | 用户未授权申请/联系；不提交。不能假设可补出9项任务 |
| Belgrade | [作者README](https://github.com/innovation-center-etf/FINGER-TAPPING-PARKINSONISMS-DATABASE)公开说明.mat模式 | 核实数据再用许可、是否有额外腕部/Acc数据和独立性证明；不得直接复制finger为wrist | 本轮只读README；不下载个体文件。许可证未知不当作已批准 |
| Maryland | [DRUM引用](https://doi.org/10.13016/2qhy-961s)，原论文称deidentified demographics/clinical/sensor-derived数据归档 | 先确认是否有原始六轴、许可和诊断字典；即便授权也为腰部任务 | 门户可读性受限；无用户申请指令、无兼容腕数据 |
| IEEE上肢研究 | [TNSRE论文](https://doi.org/10.1109/TNSRE.2023.3306203)，未查到明确原始数据发布路径 | 仅提出未来询问事项：原始Gyro、腕部、活动、许可/伦理/身份重叠 | 不凭论文开放许可推断数据许可，不联系 |
| PPMI | [官方Access Data](https://www.ppmi-info.org/access-data-specimens/download-data)：qualified researcher、签DUA、在线申请、遵守publication policy，官方说明年度分析更新/派生数据回馈 | PI/机构与研究计划、数据字段、人员、安全与用途；raw digital stream是否须额外审批待官方确认 | 未注册/申请/签约；无完整PD/DD任务匹配 |
| WATCH-PD | [原论文](https://www.nature.com/articles/s41531-024-00721-2)：3DT Stage2或向steering committee proposal | 研究方案、所需streams、伦理/安全及DD问题说明 | 没有DD组；不联系/申请 |
| REMAP | [原论文](https://doi.org/10.1038/s41597-023-02663-5)：accelerometry属于controlled部分，须Bristol Research Data Access Agreement | 研究者资质、目的、所需受控数据和安全措施；以官网实际表格为准 | 不申请；PD/HC、活动不符 |
| 其余Synapse/DataPort/Radboud/Zenodo | 矩阵逐项官方入口；部分当前条款不可完整核实 | 首先核实条款/许可与是否存在合格DD/原始输入；不预设开放下载就是研究授权 | 本轮零下载，缺核心标签不优先走访问流程 |

以上为可审阅的未来准备清单，不是已发送邮件、已批准申请或新的访问许可。只有用户明确授权具体来源、具体用途和访问动作后才考虑申请/下载，另核对数据方批准。
''')
 save('MODEL_DEPLOYMENT_AUDIT.md',f'''
# 冻结模型部署审计

实际读取当前文件而非按旧对话推定状态。Git mirror HEAD及刚fetch的origin/main均为{START}，worktree干净；主线45个文件逐项与该已发布快照匹配。主线Python全文读取/AST解析和源文件哈希见mainline_source_audit.csv，关键forward/数据预处理另行审阅。启动前重读19个必需历史/入口文档，MISSING0；全文由审计工具扫描，数值表保留原文件哈希，提供narrative审阅，outer相关数值3行不对研究结果重新展示。此阅读控制不抹除历史接触。

{table(hashes[['asset','expected_sha256','actual_sha256','status']])}

Norm文件、checkpoint内部mean/std逐值一致；norm fit IDs与原split的205名train完全一致，query/excluded保持分离，精确身份私有保存。配置11活动/2腕/6通道固定；官方HarNet loader所有131 feature-extractor键和权重与原checkpoint逐项一致，官方随机classifier不用于特征。显式IndependentWSSL strict load原143172参数分类侧，HarNet10457408参数固定，不创建45模型ensemble或新全390人模型。

合成推理所有参数requires_grad=false、inference_mode、eval；全部state（包括BN）前后fingerprint一致。原两logits、原argmax、pDD>.5/tiePD、CalibratorNONE。没有使用mainline中会加载全局cache的FrozenEmbeddingCache，也没有触发任何训练入口。源码相同/哈希锁定和合成正确性不等于这一个checkpoint已经取得独立临床验证。

当前阶段只测试固定内存接口。实际外部文件解析、坐标/时钟/任务定义核验、授权与身份独立、临床评价runner的真实数据验收仍待完成。技术上能生成合成logits不等于可以给任何PD/HC或少任务数据报告PD/DD性能。{disclosure}
''')
 save('EXTERNAL_INPUT_ADAPTER_REPORT.md',f'''
# 外部输入适配与合成correctness

状态PASS_SYNTHETIC_ONLY；真实外部adapter验收尚未完成。所有执行在服务器已有mfam_analysis环境，没有安装或升级依赖；3个程序生成、无临床身份的fixture，固定种子20261111。主线/原资产未修改。

## 已实现接口

external_adapter.py接收内存中的11活动→left/right→Stream结构，不扫描文件系统或读取外部个体文件。输入不含诊断标签；要求有限[T,6]、明确通道映射、独立设备/腕侧来源、已确认的原PADS轴向与wrist placement、规律时间戳和单位。未知映射、不规则时间、缺活动/腕/通道/单位拒绝；没有复制对侧或伪造任务。许可/重叠等真实clinical gate的布尔检查仅是应用接口，不能替代数据方核查和用户批准，不能称系统强制隔离。

确定性重命名、g与m/s²、rad/s与deg/s、ms与s转换，固定polyphase到100Hz。原local预处理trim48、Acc减固定L1趋势、Gyro原样和既有norm；external rawAcc按原clip/窗口/resample路径。两支共享实际活动/腕结构和真实长度；不足1000的edge padding仅为原HarNet合法窗口处理，不是补缺失活动。batch末端容器padding不改变真实长度。只接受调用者核实的raw device stream，预滤波/Euler角/特征表不具有等价性。

所有允许的单位/采样转换仍须记录源设备带宽、硬件滤波、轴向与安装差异；upsampling不会恢复丢失频带，ms→s不能修复时钟错位。代码当前不擅自插值不规则时钟或学习设备变换。源格式未知时不能凭合成内存结构宣称.mat/TXT真实解析器验收完成。

## 实际测试结果

{table(tests)}

重复两次完整合成推理最大logit差={synthetic['determinism_max_error']:.1f}（bitwise相同），两次的classifier batches均[2,1]，encoder末批每人22窗，满足64上限。另测试长记录首末双窗44的窗口映射；该长窗测试不冒称做过额外完整临床推理。padding perturbation对照两次forward一致，输入未变、logits有限、状态及BN未变。合成运行记录elapsed={synthetic['elapsed_seconds']:.6f}s（不含启动资产哈希成本，未测峰值内存）。

这25项涵盖用户要求14类检查及多个拒绝路径，不是25个临床验证病例。所有真实clinical sensor读取0、外部下载0、新fit/optimizer更新0。可进一步验证实际file parser、放置/时间同步和真实设备信号，但须先取得具体数据授权与兼容性证据；本轮不实施。
''')
 save('REFERENCE_PIPELINE_PROVENANCE.md',f'''
# 原reference pipeline来源恢复审计

总体PARTIAL；原方法精确对新队列评分BLOCKED。不是“所有reference分数均不存在”，也不是“代码可见所以已经恢复”。审计范围为原reference artifact、Phase5和Phase6，目录外资产不作全局不存在断言；原创建代码本身未序列化所需reference变换对象。

## 算法来源已恢复

原typicality.py SHA256={ref['source_sha256']}，与B_freeze创建时source hash一致；全文和调用者源代码已读取、AST有效，未导入/执行其fit函数。common_reference先在各split train的每个activity/wrist1024D特征上fit StandardScaler，再PCA(n_components=min(16,n−1,d), whiten=false, randomized, iterated_power4, random_state=原SEED)，去掉近零奇异方向并L2归一化。PD/DD平衡bank按固定SHA顺序选min(nPD,nDD)人，各59–62人，仍来自该split train；cosine distance截到[0,2]、k5均值、自身排除，零范数距离1。先等均左右腕再均11活动；relative=dPD−dDD，正方向DD，u=min(dPD,dDD)。原strict5fold类内q95诊断，与后续Phase5的u q25/q75用途不同。

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

{disclosure}
''')
 save('NEXT_STAGE_RESEARCH_PLAN.md','''
# 后续研究计划：只列候选，不执行Phase8

| 优先级 | 研究假设/目标 | 下一步具体工作 | 成本/训练 | 停止条件 |
| --- | --- | --- | --- | --- |
| P1 | 固定Frozen WSSL-STR在独立、同协议PD/DD队列仍有判别能力 | 临床合作方先确认能否连续入组、诊断随访、双腕11任务原始六轴及伦理；用户授权后才联系/申请 | 临床/伦理与数据管理为主；训练0；预算尚未核实不报虚假金额 | 任一授权、独立性、任务/设备硬门槛失败则停止直接模型验证 |
| P2 | 现有PLA PD/ET、Belgrade PSP/MSA或IEEE上肢资源可为独立临床方案提供先验 | 仅在另授权后核实数据字典/许可/未公布的任务来源，不下载后再发现缺输入 | 文档协调；不训练 | 只有2/4/1任务且无法提供真实额外任务时，不进入原11任务模型验证；不能复制补全 |
| P3 | 确定性格式映射足以运行固定模型而无代码数值错误 | 在获批的真实原始格式上完成parser、轴/单位/时钟和设备QA，冻结source/hash、盲预测 | 小规模工程与已有模型推理；训练0 | raw/preprocessed不明、数据丢失、强行轴/placement映射或合成与实际格式不一致即停止 |
| P4 | 类×参考距离交互在独立PD/DD重复 | 优先寻找与旧hash吻合的完整reference状态；若找回才按原bank/旧cutoff独立评分 | 资产审计+现有参数推理；新fit0 | 原1024DScaler/PCA/bank状态缺失则原方法执行BLOCKED；不凭旧hash造状态 |
| P5 | DD异质性、任务质量或norm/几何可解释交互 | 临床定义与盲质量评分先定，预先冻结探索检验/样本量与机制可证伪预测 | 新临床元数据与统计；本轮不实施 | 样本太小、亚型不可靠、中心/设备与疾病完全混杂则NA，不作因果结论 |

这些任务需要不同的新条件；Phase7没有已获批准的外部数据访问，也没有新训练依据。不应为填补资料缺口立即训练。停止同队列Adapter/LoRA/解冻/gate/拒绝分类器/dual-reference融合、阈值/CE/Patch搜索。若未来提出新reference protocol，必须与原方法分开登记并另行授权所有新fit，不能改名复活失败融合。PD/HC或严重度数据可以服务另一个明确任务，不是本次PD/DD外部验证的替代结果。

独立临床样本量、伦理、时间和预算均未核实；不提供不存在的医院合作、招募承诺或实验进度。申请材料准备见DATA_ACCESS_PREPARATION.md，具体未来主检验与停止规则见INDEPENDENT_VALIDATION_PROTOCOL.md。完成本轮发布后停止，不自动启动Phase8。
''')
 save('CORRECTNESS_AND_PROVENANCE.md',f'''
# Correctness与来源边界

正式起点{START}。当前root/mainline/history和Phase6资产启动/结束受{startup['protected_files']}项SHA256保护。45个主线文件与新fetch的GitHub snapshot一致，19个指定历史/入口文档当前读取/hash，无MISSING，3行outer数值屏蔽。没有按旧对话假定现存状态；没有运行训练入口或旧outer loader。历史正式产物不覆盖。

所有新代码与报告/配置/汇总source SHA在source_output_manifest.json；五项模型/config/norm/preprocessing哈希见model_lock_hash_audit.csv。SYNTHETIC_SOURCE_FREEZE在任何合成结果前锁住common/bootstrap/adapter/tests/PROTOCOL；数据源调查、reference分类审计与报告源随后登记，不冒称在Phase6结果前预注册。Phase6推理资产/旧统计保持原样，未重算成新临床证据。

合成测试25项通过，重复logits逐位一致、norm与原checkpoint相同、131官方feature keys相同、全部模型状态/BN未变、无梯度/优化器更新。3个fixture仅程序生成；所有输入内存对象无临床标签，不读临床sensor、不解码旧全队列sensor/feature数组。L1趋势求解是既有确定预处理，不是拟合分类器、Scaler或温度。

Reference审计只读取现有query评分/身份清单/源代码与序列化header；已有15×3query分组一致。为审计全部15个development单元，本轮完整加载既有Phase5 development分数表后再按context/inner分组，未把这种读取声称为subject级物理隔离；它不是旧outer-test性能表，也没有进入新的模型训练或临床输入。初版审计发现45个Scaler命名NPZ而停止，人工核实后按二维fusion Scaler分类，保留reference_audit_v1.py、logs/reference_audit.log和成功v2日志；active script仍为reference_audit.py。这个审计修订没有产生新距离、改变阈值或拟合状态。不能把首次停止隐藏为所有脚本首次均通过。

完整私有fit/query/excluded与哈希、状态搜索目录/headers在restricted_manifests；日志在logs，不发布。新临床外部query ID空，新fit ID空，逐人预测未公开。公共兼容矩阵只有研究级样本数/来源，不存公开检索个体行。源README只经连接器读取，未clone仓库；网页technical访问失败按未知记录，未绕过授权或身份验证。

外部下载、申请/联系人、代签协议、依赖安装/升级、新分类器/深度训练、温度/阈值/reference Scaler/PCA拟合均0。本轮不是全真实数据评价runner验收：真实格式parser/设备和时钟映射、数据访问/去重、临床标签盲法、子组方法按真实字典的预先冻结仍待完成。所有真实external_metrics与reference_external_interaction均null。

{disclosure}

最终验收会逐项检查需求文件、原模型锁、source freeze、25项结果、630个状态header的分类、全部保护hash和公开过滤。整体只能称PASS_WITH_BLOCKED_CLINICAL_AND_REFERENCE_EXECUTION，不应写全能力PASS。发布沿用未修改的选择式publisher与既有GitHub连接器exact-tree/CAS规则，保留历史MFAM快照且不重同步；不上传raw、IDs、逐人结果、特征、checkpoint、受保护manifest和运行日志。
''')
 save('REPRODUCE.md','''
# 复核与运行入口

全部实验仅在服务器 `/home/zyt/deep_final/artifacts/independent_validation_phase7_20261111` 执行，Python `/home/zyt/envs/mfam_analysis/bin/python -B`，无新依赖。原正式运行顺序：bootstrap7.py → synthetic_tests.py → reference_audit.py(v1停止并保留) → reference_audit.py(active分类修复) → dataset_discovery.py → build_reports7.py → verify7.py。通过前后的日志和协议/源freeze分别保留，不能把重跑覆盖原冻结记录。

代码提供严格内存Stream接口，未实现或暗中调用外部数据下载。现有private模型与元数据授权是复核前提，public仓库不包含权重或数据。运行到新独立目录前必须显式适配HERE与原路径引用；不要原位覆盖本轮已发布输出，不更改mainline/历史源或checkpoint，不放松锁定hash和测试容差来通过检查。

synthetic_tests不是外部临床evaluation入口；未来真实clinical推理须授权/重叠/协议/设备/标签分离与实际格式验收。reference_audit不调用原Space.fit；运行旧typicality/audit_bank_norms以重fit明确不属本轮复核授权。
''')
 save('FINAL_REPORT.md',f'''
# Phase 7 最终研究报告

**保留Frozen WSSL-STR，不改变任何正式权重、norm、预处理或决策。** 本轮完成20项数据源调查、五项锁定资产核验、严格内存adapter及25项服务器合成测试、独立评价协议和reference来源审计。尚未取得直接可用的独立PD/DD队列，也未恢复原reference完整外部评分状态，因此没有真实外部性能、没有独立复核成功结论。新增训练/优化器更新/分类器/温度/阈值/Scaler/PCA拟合、外部个体下载、申请/联系均0。

## 完整工作状态

| 工作 | 状态 | 证据及保留缺口 |
| --- | --- | --- |
| 最新GitHub/服务器核对 | PASS |起点{START}；mainline45文件一致 |
| 必读历史与源码 | COMPLETE_WITH_DISCLOSURE |19文档全文工具扫描、当前hash；Python全文/AST；屏蔽3行outer数值，未重新读旧outer性能 |
| 五项模型资产及205人norm来源 | PASS |实际文件SHA全匹配、checkpoint内部norm一致 |
| 数据源扩展 | COMPLETE inventory20项 |16个新增条目；全字段矩阵含unknown，无完整合格独立队列 |
| adapter与合成correctness | PASS_SYNTHETIC_ONLY |25检查、两次确定输出、BN/全部state不变；真实parser/设备映射未验收 |
| 独立评价协议 | DESIGN_READY_WITH_PRE_DATA_GATES |主指标/CI/盲法/NA规则已定；数据字典相关子组检验须揭盲前进一步冻结 |
| reference来源恢复 | PARTIAL |原算法/最终query分数/ID来源完整，原reference Scaler/PCA/bank/训练OOF矩阵缺失 |
| 真正临床分类性能 | NOT_EVALUATED |全部external metrics=null，不将development15split均值代入 |
| 类条件覆盖独立复核 | BLOCKED |无新合格队列且缺原reference变换状态；不是假设被否定 |

## 数据候选优先级

{table(ranked)}

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

{disclosure}

最终代码/协议/研究级兼容矩阵/汇总/审计通过既有选择式Git流程发布；privates和logs留服务器。目录 `/home/zyt/deep_final/artifacts/independent_validation_phase7_20261111/`。本轮完成后停止，不自动启动Phase8。
''')
 write(HERE/'analysis/runtime_environment.json',dict(python=sys.version,versions={x:importlib.metadata.version(x) for x in ['torch','numpy','pandas','scipy','scikit-learn']},dependency_installs=0,all_experimental_execution='authorized server only',synthetic_only=True,new_training=0,new_fits=0,external_individual_downloads=0,contact_or_application_actions=0))
 write(HERE/'decision.json',dict(retained='Frozen WSSL-STR',model_changed=False,new_training=0,optimizer_updates=0,new_fits=0,external_metrics=None,reference_external_interaction=None,independent_clinical_validation='NOT_EVALUATED_NO_ELIGIBLE_COHORT',synthetic_adapter='PASS_SYNTHETIC_ONLY',reference_recovery='PARTIAL_EXACT_SCORING_BLOCKED',dataset_candidates=20,start_phase8=False))
 guard();print('PHASE7_REPORTS_COMPLETE')
if __name__=='__main__':main()
