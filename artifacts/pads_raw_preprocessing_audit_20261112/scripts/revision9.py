"""Revise this unpublished Phase9 report after deterministic solver counterexamples."""
from common9 import *

def main():
 w=load(HERE/'analysis/solver_objective_witness_integrity.json');assert w['status']=='CONFIRMED_NUMERICAL_OBJECTIVE_AND_BATCH_DEPENDENCE';assert load(HERE/'analysis/solver_batch_lineage_integrity.json')['status'].startswith('PASS');version=load(HERE/'configs/PREPUBLICATION_REVISION.json');archive=HERE/version['snapshot_prefix']
 for r in version['snapshot_files']:assert sha(archive/r['path'])==r['sha256']
 def replace(name,a,b):
  p=HERE/name;s=p.read_text();assert a in s,(name,a);p.write_text(s.replace(a,b))
 common='\n\n## 发布前数值正确性修订\n\n本版结合[L1求解器证据](L1_SOLVER_CORRECTNESS_REPORT.md)。此前Phase9发布前草稿与全部源码/聚合输出保存在audit/pre_solver_witness_snapshot；旧CLOSURE_FREEZE按该版本命名空间核验，替代关系见configs/PREPUBLICATION_REVISION.json。正式Phase3–8、数据、模型及normalization均未改；原始质量关联表的p/q未改变。\n'
 p=HERE/'DATA_PIPELINE_MAP.md';s=p.read_text();s=s.replace('record-local; no cross-subject statistics','parsing is local; later ADMM stopping is batch-coupled');s=s.replace('offline noncausal adaptive operation; no LTI cutoff','offline adaptive objective; GLOBAL batch stopping causes confirmed numerical cross-record coupling; no LTI cutoff');p.write_text(s+common)
 df=pd.read_csv(HERE/'analysis/data_pipeline_steps.csv');ix=df.step=='Acc detrend / Gyro';df.loc[ix,'requires_fit']='separable per-series objective but globally shared finite-iteration stopping';df.loc[ix,'fit_subject_ID_scope']='original same-activity/length batch up to32 wrist records; IDs private';df.loc[ix,'leakage_or_limitation']='confirmed numerical cross-record dependence before splits; not label-supervised fit; cannot claim strict record-local transform';df.to_csv(HERE/'analysis/data_pipeline_steps.csv',index=False)
 report=f'''# L1求解精度、批次独立性与既有输出的目标值证据

这是发布前最终复核发现的首要问题，不是重新训练或重做真实信号滤波。原正式文件全部只读，当前输出/源输入均与原创建时hash匹配（8580/8580）。

FACT：原L1代码求解目标 P(x)=0.5||y-x||²+50||D²x||₁。ADMM的primal/dual范数和容差在整个batch所有series上联合计算，用同一个停止时刻；实际准备按活动/长度、subject排序，左右腕、32文件一批，在subject split之前处理。目标可分不等于有限迭代实现具有逐记录独立性。

数值反例：同一1Hz、0.01g合成目标独立处理在{w['synthetic_alone_iterations']}次迭代后标converged，与固定42个其他series合批后在{w['synthetic_batched_iterations']}次停止；同一目标trend最大相差{w['synthetic_same_input_max_trend_change']:.9g}g。两次都使用原lambda50/rho40/max5000/tol1e-4，无参数搜索。

0.5Hz/0.001g合成例有数值KKT证书：仿射参考的dual最大值{w['exact_certificate_dual_max']:.9g}<50，stationarity最大残差{w['exact_certificate_stationarity_max']:.9g}。该参考目标约{w['exact_affine_optimum_objective']:.12g}，原solver在1次迭代、converged=True时约{w['exact_case_existing_solver_objective']:.12g}。无论是否把浮点KKT称为最优性证明，仅“存在更低目标的可行解”就已足够反驳高精度目标一致性。

## 既有真实输出的保守证据

未重跑真实L1、未生成候选信号。对已保存float32 residual r_hat，真实保存前残差与它的差用一整个float32 ULP作保守上界 e。旧trend的post48部分为y_tail-r。定义下界：

LB = 0.5 sum max(|r_hat|-e,0)² +50 sum max(|D²(y_tail-r_hat)|-(e_i+2e_(i+1)+e_(i+2)),0)。

原始完整目标P_old≥LB，因为省略的前48点及其边界贡献都非负。对完整原始记录构造任意合法仿射参考x_aff，有P*≤U=P(x_aff)。若LB>U+max(1e-9,1e-4 U)，即可确认旧输出在目标值上优于该参考的说法不成立，且其目标次优至少LB-U；不需要知道原始前48点trend，也不把float32舍入当错误。这是确定性凸目标见证，不是统计p值，也不是新的滤波候选。

EVIDENCE：25,740条Acc轴记录中{w['flagged_axis_records']}条通过上述保守见证，涉及{w['affected_wrist_records']}个腕记录、{w['affected_subject_count']}名患者至少一条轴。**不是“这些都是坏信号/噪声”，也不表示未flag的输出已获最优性证明。** 逐轴数值仅私有，公开solver_objective_witness_summary.csv按活动/通道汇总。Gyro不经过此L1运算，原raw[48:]精确检验仍成立。

## 数据协议含义

实际275个准备batch迭代范围{w['original_minimum_iterations']}–{w['original_maximum_iterations']}。按原代码顺序重建batch并逐项匹配solver_audit日志；15split的inner-train腕记录有97.10%–100%与non-train成员共享停止准则。这里只检查来源成员关系，未按outer角色计算信号/性能，不读outer结果。

FACT：原实现存在跨记录数值依赖；EVIDENCE：原准备batch跨subject角色。应标为**split前无监督数值耦合/传导性风险**，不能再宣称预处理完全逐受试者独立。它不是已证实的诊断标签泄漏；solver未用标签，本轮也没有估计分类分数被提高多少。train-only normalization成员正确并不能消除上游这种依赖。旧Frozen WSSL-STR数字和checkpoint不改，但方法学解释必须加入该限制。

## 不作超出证据的推论

原来的合成“L1频率响应”同时包含非线性目标与有限迭代精度/合批背景，不能当精确L1解的理论响应，亦不能仅由它宣称震颤保留/慢动作损失。原raw质量、边界窗口、时间戳、设备分布检验未依赖新solver参考，因此原132/440/880统计族不改。原始Acc的gravity/API契约仍UNKNOWN。

优先候选改为PREP-L1-PRECISION：固定同一lambda50目标，只修复逐记录独立性与数值证书，不搜索滤波强度/频带。性能收益UNKNOWN，必须经用户批准后的独立单因素对照；本轮训练、normalization重拟合、真实L1重处理均0。
'''
 (HERE/'L1_SOLVER_CORRECTNESS_REPORT.md').write_text(report)
 for name in ['SIGNAL_QUALITY_REPORT.md','FILTER_AND_SPECTRAL_AUDIT.md','NORMALIZATION_PROVENANCE_REPORT.md','DATA_SPLIT_AND_LEAKAGE_AUDIT.md','CORRECTNESS_AND_PROVENANCE.md','CLASS_CONDITIONAL_DATA_QUALITY_REPORT.md']:
  p=HERE/name;p.write_text(p.read_text()+common+'\n**修订结论优先于此前逐记录独立假设**：既有L1有限迭代使用全batch共同停止，数值输出可随其他记录变化；真实输出保守目标值证据见新报告。此发现不改原始信号质量统计，不表示标签泄漏或已证明模型性能虚高。\n')
 replace('DATA_SPLIT_AND_LEAKAGE_AUDIT.md','逐记录label-independent detrend/外部冻结HarNet提前导出本身不等于跨人fit','外部冻结HarNet提前导出本身不等于跨人fit；但本轮新证据确认L1的全batch停止造成split前跨记录数值依赖')
 replace('FILTER_AND_SPECTRAL_AUDIT.md','3/6/12Hz稳态正弦基本保留，1Hz响应具有幅值依赖','3/6/12Hz在这次42series合批下稳态正弦基本保留，1Hz响应具有幅值/批次/迭代精度依赖；这不是精确L1目标的理论滤波特性')
 cand=pd.read_csv(HERE/'analysis/preprocessing_candidate_priority.csv');precision=dict(priority=1,candidate='PREP-L1-PRECISION',status='REVIEW_ELIGIBLE_NUMERICAL_CORRECTNESS',current='separable L1 objective with globalbatch ADMM residual stopping before split',problem='large objective suboptimality and target output depending on companion records',evidence='20173/25740axis conservative witnesses;7490wristrecords; same synthetic input changes0.008199g; source/log membership verified',physical_basis='numerical fidelity to the already-declared lambda50 objective; not new denoising theory',change='same objective/lambda50; deterministic independent per-series solve with dual-gap certificate; no rawdata/units/axes/rate/trim change',possible_loss='substantial STR input distribution shift; higher CPU cost; exact L1 itself may remove clinically useful trends',expected_gain='restore numerical consistency and independence; BA/DD gain UNKNOWN',input_distribution_changed=True,correctness='per-series primal-dual gap<=1e-8+1e-4*affine-residual-energy; target alone/batched/reordered agreement; float32 roundtrip; source hashes',future_cost='solver QA+bounded CPU benchmark first; only after approval and all certificates: train-only norm15 splits +45 matchedclassifier-side runs; HarNet cache unchanged',stop='any certificate failure;batch-dependence remains;CPU budget exceeded;provenance failure;no automatic model promotion without joint gate')
 cand.loc[cand.candidate=='PREP-RESAMPLE-BOUNDARY','priority']=2;cand.loc[cand.candidate=='PREP-RESAMPLE-BOUNDARY','status']='DEFER_AFTER_SOLVER_AND_INPUT_CONTRACT';cand.loc[cand.candidate=='PREP-FILTER','status']='NO_ACTION_FOR_NEW_FILTER_SEPARATE_SOLVER_PRECISION';cand=pd.concat([pd.DataFrame([precision]),cand],ignore_index=True);cand=cand.sort_values(['priority','candidate'],kind='stable');cand.to_csv(HERE/'analysis/preprocessing_candidate_priority.csv',index=False)
 (HERE/'PREPROCESSING_CANDIDATES.md').write_text('# 修订候选优先级\n\n首个具体预处理候选是PREP-L1-PRECISION，而不是增加滤波器或扩大裁剪。算法目标lambda50不变，问题是数值求解精度和逐记录独立性。设备/Acc物理语义澄清仍为优先来源工作，HarNet line边界参考退居其后；不要把两改动组合。原始质量统计保持不变，旧Phase9草稿已版本化保留。\n\n'+table(cand)+'\n\n所有收益仍是HYPOTHESIS。数值正确性通过不等于新模型联合门槛通过；若性能不通过，不能强行保留模型，也不能继续把legacy输出叫成高精度L1解来合理化结果。\n')
 level2='''# Level2修订规范：只供用户审阅，未执行

训练预算本轮仍0，所有候选均NOT_RUN_REQUIRES_EXPLICIT_USER_APPROVAL。优先只审阅PREP-L1-PRECISION一个因素；不改lambda50、trim48、单位/轴、采样、活动、HarNet、结构、loss、threshold或seed。目的为把原目标实现成可验证的逐记录数值解，而不是用成绩挑更平滑滤波。

先固定一个现有依赖可支持的确定性solver实现及source/hash，不能并行搜索多种实现后择优。接受标准是每个series可行dual |u|<=50，Primal−Dual gap<=1e−8+1e−4 E_aff，E_aff为该series减合法仿射参考的残差能量；不能用单一converged标志代替gap。低幅合成KKT例、不同频率/幅值、常数/斜坡、float32保存误差、同一输入单独/混批/重排最大差<=1e−7g都要通过。每条独立终止，缓存factorization可共享而停止条件不能受其他患者影响。

先做限定CPU预算的可行性评测，未定义/无法满足精度和成本上限则STOP，不放宽证书以追求BA。建议审批时明确CPU上限，模型run上限固定45；未经该审批不开始真实候选重处理。如果通过，候选signals只写新独立目录，旧数据和checkpoint不变；本轮未实现/应用该solver候选。

真实候选处理和训练必须在明确批准后：同15subject-level splits×seeds42/43/44，同原Frozen WSSL-STR recipe（batch8/AdamW2e−4/WD1e−4/cosine50/clip5/patience12），HarNet frozen旧rawAcc缓存保持，原seed初始化，原validation BA selection和0.5规则。STR信号改变后，按原函数在各inner-train上重新fit该候选norm（15唯一split，三个seed复用同split值），这是固定数据变换后的必要配套，不是另一个norm搜索。不能沿用旧norm制造分布失配，也不能拟合全队列norm。本轮没有做任何一次norm重拟合。

基线为完整保护的legacy近似L1流程及原45模型。比较先split内3seed指标平均再15split等权；9指标、10000split bootstrap、patient-cluster10000、Wilcoxon/effectsize/seedSD。唯一候选主族BA/AUROC固定2，次族其余7固定7，缺失测试按p1计入、自身NA。报告数值修复与方法学独立性的成功，以及模型是否值得提升为正式模型，两者区分。

模型提升仍需meanΔBA>0、≥10/15wins、paired95%CI下界>0、AUROC/F1不降、PD/DDrecall各下降<=.01、BA/AUC within-split3seedSD<=baseline1.25、correctness/provenance全部通过。原15split是重复development；旧指标不因发现耦合而重写，新对照不能叫独立临床验证。数值改善不等于BA改善，失败不追加lambda/threshold/架构搜索。原始Acc/HarNet物理语义仍需核实，不能由数值证书替代临床物理解释。

PREP-RESAMPLE-BOUNDARY暂不安排联合或并发训练。它需先解决solver/输入契约和参照协议，再另行审阅一个单独padtype=line对照；不把它悄悄加到L1数值修复，不能累加两个因素后归因。时间戳/同步/轴/新滤波/norm搜索无执行授权。全部Level2/下一阶段本轮未启动。
'''
 (HERE/'LEVEL2_EXPERIMENT_PROTOCOL.md').write_text(level2)
 judge=pd.read_csv(HERE/'analysis/research_question_judgments.csv');changes={0:('FACT/EVIDENCE','结构完整且4时间间隙；另有确认的L1目标次优/批次耦合，不等同传感器噪声。'),7:('EVIDENCE/LIMITATION','原经验响应混入数值精度与合批停止，不能解释为精确L1的理论保留率；临床震颤信息UNKNOWN。'),13:('FACT/LIMITATION','ID与norm成员正确，但L1 split前全批停止有确认的无监督数值耦合，不能声称全链严格subject-independent；标签泄漏/性能虚高未被证明。'),15:('HYPOTHESIS_READY_FOR_REVIEW','优先单因素L1逐记录数值精度/证书修复；line边界后续单独审阅，不能组合。'),16:('HYPOTHESIS','数值修复有确定正确性依据，实际BA/DD改善未知；先澄清设备与Acc输入契约。')}
 for i,(status,text) in changes.items():judge.loc[i,'evidence_status']=status;judge.loc[i,'judgment']=text
 judge.to_csv(HERE/'analysis/research_question_judgments.csv',index=False)
 final=f'''# Phase9最终报告：包含L1数值证据的发布前修订

**Level0/1已完成；训练0、normalization重拟合0、真实L1重处理0，Level2未执行。** 此版本取代audit/pre_solver_witness_snapshot内发布前草稿。原Phase3–8、数据、正式模型和原阈值均未改变。

## 首要发现

1. **FACT/EVIDENCE：L1数值精度和批次独立性存在实证问题。** 原始/处理文件创建hash匹配8580条；25,740条Acc轴中{w['flagged_axis_records']}条具有保守目标次优见证，涉及{w['affected_wrist_records']}腕记录。见证考虑float32舍入和未知前48点，并不把它们当坏信号。相同合成输入单独与合批trend最大差{w['synthetic_same_input_max_trend_change']:.7f}g。原实际275批在split前混合，train成员97.10%–100%共享non-train停止准则。不能继续声称全链预处理严格逐受试者独立；这不是已证实标签泄漏，也没有证明BA被提高多少。
2. **FACT：结构完整，局部timestamp问题存在。** 390人×11活动×双腕8580记录，空文件/缺活动/缺腕/NaN/Inf/exactduplicate均0，4条大间隙max0.646577s。Gyro与raw[48:]float32全量精确一致。
3. **EVIDENCE/UNKNOWN：设备和物理输入契约需澄清。** Series3 PD93/DD11、Series4 PD183/DD103；原始Acc单位g明确，但是否已分离gravity/导出前变换UNKNOWN，不能加假定1g或按性能选坐标轴。
4. **NO_ACTION：不扩大裁剪，不重复添加已有抗混叠。** 官方启动振动已有48点近似裁剪；无依据再删1–5s。当前HarNet FIR抗混叠存在，零延拓边界效应另有信号证据，但不优先于solver问题，也没有分类改进证据。
5. **FACT：split/norm/padding局部检查通过，但不能抵消上游耦合。** 45norm成员及tensor一致、零refit；15集合内部互斥；STRfull976/2000且真长度切片，HarNet覆盖post48；每人4个val appearance仍只有1人。

## 当前数据流

完整源码/函数/形状/单位/fit/label/split表见DATA_PIPELINE_MAP.md及data_pipeline_steps.csv。Raw解析→原全batch近似L1(Acc)/Gyro原样→trim48→float32/full length→subject split→train-only norm→mask/true-length STR；parallel rawAcc→clip/context→30Hzpolyphase→FrozenHarNet→bilateral/activity/subject+STR。关键修订是L1目标数学可分、实现停止却非逐记录独立。

## 数值证据与解释

[L1_SOLVER_CORRECTNESS_REPORT.md](L1_SOLVER_CORRECTNESS_REPORT.md)提供公式、合成数值KKT证书、原输出下界/参考上界、原solver日志和批次来源。原“合成频率响应”必须解释成legacy有限迭代实现响应，不能当精确L1理论响应；原cohort原始质量/边界统计不变，132/440/880固定族不缩小、不重选。37/132质量关联与9/440边界class关联仍为探索性，不能说是病理机制或故障标签。历史outer接触、原全cache访问及多轮development偏差永久保留。

![Numerical objective and existing-output witnesses](figures/solver_objective_evidence.png)

## 十八项最终判断

'''+table(judge)+'''
## 候选与停止

'''+table(cand[['priority','candidate','status','expected_gain','future_cost','stop']])+'''

优先可审阅的具体改动是固定lambda50目标的数值精度/逐记录独立性修复，不是新滤波强度搜索。需要独立源码、逐series dual-gap证书、顺序/批次不变性、CPU上限，再经用户明确批准才能真实重处理、train-only norm和45配对run。性能收益UNKNOWN；正确性与模型提升分别判定，不能强行保留模型。Line边界仅作为后续单因素，不与L1组合。

16必需报告、Acc语义补充、L1正确性报告、全部CSV均已生成。逐人表/异常位置/44原始代表图仅私有；公开为聚合/合成，稀有交叉格数值抑制。24,180旧资产保护hash最终复核；旧Phase9发布前snapshot保留，旧冻结声明按其命名空间验证，原正式历史不覆盖。完整未来规范见LEVEL2_EXPERIMENT_PROTOCOL.md。**在Level0/1停止，未经用户批准不启动Level2。**
'''
 (HERE/'FINAL_REPORT.md').write_text(final)
 p=HERE/'CORRECTNESS_AND_PROVENANCE.md';p.write_text(p.read_text()+'\n\n数值见证属于确定性Level1正确性，不是预处理性能消融；可行仿射参考只用于目标值上界，不作为生产变换、classifier或normalization拟合。所有真实输出仍是原文件。旧CLOSURE被发布前新证据取代，原文、hash、121文件snapshot和未发布localcommit9fd6adbb完整保留，不删除失败或旧结论。\n')
 (HERE/'README.md').write_text('# Phase9当前入口（L1数值证据修订）\n\n[最终报告](FINAL_REPORT.md) · [L1数值正确性](L1_SOLVER_CORRECTNESS_REPORT.md) · [真实链路](DATA_PIPELINE_MAP.md) · [Acc语义](ACCELERATION_SEMANTICS_ADDENDUM.md) · [候选](PREPROCESSING_CANDIDATES.md) · [待批准Level2规范](LEVEL2_EXPERIMENT_PROTOCOL.md)。\n\nLevel0/1完成，训练/真实L1重处理/normrefit均0。发现L1目标次优及批次依赖，原正式模型与数据保持。所有旧草稿见audit/pre_solver_witness_snapshot，仅为版本追踪；本页链接为当前结论。私有IDs/逐人表/44信号图不公开。\n')
 p=HERE/'REPRODUCE.md';p.write_text(p.read_text()+'\n\n发布前修订链：在旧Level0/1输出上依次solver_witness9.py → solver_batch_lineage9.py(先保存旧snapshot) → revision9.py → solver_plot9.py → finalize9_v4.py。原finalize9_v3及CLOSURE只代表旧draft，最新closure按supersession规则复核旧档与新档。所有真实L1重处理/模型训练仍0。\n')
 write(HERE/'analysis/report_revision_integrity.json',dict(status='PASS_REVISED_FOR_CONFIRMED_SOLVER_FINDING',revision=2,old_scientific_draft_preserved=True,all_prior_protected_assets_unchanged=True,new_primary_candidate='PREP-L1-PRECISION',labels_not_used_for_objective_witness=True,Level2_started=False,normalization_refits=0,model_training=0))
 print('PASS_SCIENTIFIC_REPORT_REVISION')
if __name__=='__main__':main()
