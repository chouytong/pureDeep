"""Compile verified saved-score findings without fitting or changing decisions."""
from common import *

def read(n):return pd.read_csv(HERE/'analysis'/n)
def put(n,s):(HERE/n).write_text(s.rstrip()+'\n')
def seedfirst(f,columns):return f.groupby(['context','inner'])[columns].mean().mean()

LIMIT='''本轮属于同一390人、15个重叠development splits的重复开发。正式指标先同split平均三个seed的指标；三seed概率均值仅用于错误分组。split bootstrap、seed SD、subject-cluster CI均不能充当独立外部泛化证据。cluster bootstrap按PD276/DD114分层，10,000次，同一人的全部appearance/seed共享采样次数，仅固定已拟合模型条件下的不确定性；不覆盖重训练和多轮开发选择偏差。不同条件/估计对象的CI是否跨0，不构成它们之间差异的检验。'''
BOUNDARY='''永久继承上轮偏差：旧V8 outer汇总文字曾被读取，旧manifest/缓存曾整表/整数组读取，不能撤销或将本研究包装为完全未接触outer。本轮没有解码全球HarNet/stage4缓存、读取原始IMU、创建outer-test loader或访问outer performance；只用固定development产物。当前特征包先验证IDs属于本inner-train/validation，再读取保存分数；全局development融合CSV逐行过滤。哈希核验仍按完整文件读取字节，CSV过滤也不构成物理零无关行I/O；不声称系统级完全隔离。分区test IDs仅用于排除名单元数据。'''

def main():
    guard();audit=load(HERE/'analysis/asset_audit.json');integrity=load(HERE/'analysis/phase5a_integrity.json')
    # Existing publisher treats a bare 'subjects' CSV header as a patient-ID field.
    # Preserve scientific outputs and supply unambiguous count-only public aliases.
    aliases={'error_group_subject_cluster_bootstrap.csv':'error_group_cluster_summary.csv','adaptation_subject_consistency.csv':'adaptation_patient_change_summary.csv'}
    for source,target in aliases.items():read(source).rename(columns={'subjects':'subject_count'}).to_csv(HERE/'analysis'/target,index=False)
    write_json(HERE/'analysis/public_aggregate_aliases.json',dict(aliases=aliases,change='subjects -> subject_count header only; scalar patient counts, no identifiers; preserve existing privacy filter',scientific_values_changed=False))
    form=read('formal_metric_identity.csv');means=form.groupby(['condition','context','inner'])[METRICS].mean().groupby('condition').mean().reset_index()
    tests=read('predeclared_tests.csv');primary=tests[tests.family=='A_primary'];coverage_tests=tests[tests.family=='A_coverage'];mechanism=tests[tests.family=='B_mechanism']
    comp=read('subject_complementarity.csv');single=comp[(comp.aggregation=='single_seed')&(comp.error_group=='all')]
    ct=single.groupby('label')[['n','gross_rescue','newly_broken','net_rescue','net_rescue_rate']].mean().reset_index()
    group=read('error_group_subject_cluster_bootstrap.csv');groupall=group[group.label=='all'][['condition','error_group','subjects','gross_rescue_appearance_fraction','newly_broken_appearance_fraction','mean_delta','ci_low','ci_high']]
    p=read('fusion_parameter_audit.csv');paramcols=['W_coefficient','R_coefficient','intercept','W_scaler_mean','R_scaler_mean','W_scaler_scale','R_scaler_scale','training_DD_prevalence','training_prior_log_odds','W_standardized_validation_mean','R_standardized_validation_mean','W_mean_contribution','R_mean_contribution','validation_mean_logit_shift','W_mean_absolute_contribution','R_mean_absolute_contribution']
    param_summary=p[paramcols].agg(['mean','min','max']).T.reset_index(names='parameter')
    bias=read('fusion_class_bias.csv');bt=bias[bias.event=='all'].groupby('label')[['baseline_DD_prediction_rate','fusion_DD_prediction_rate','mean_baseline_pDD','mean_fusion_pDD','mean_W_contribution','mean_R_contribution','intercept','mean_fusion_logit']].mean().reset_index()
    co=read('score_correlations.csv');co=co[co.method=='spearman'].groupby(['label','left','right']).correlation.mean().reset_index()
    sc=read('score_distribution.csv');score_summary=sc.groupby(['label','score'])[['mean','sd','q05','q25','q50','q75','q95']].mean().reset_index();score_summary.to_csv(HERE/'analysis/score_distribution_summary.csv',index=False)
    context=sc.groupby(['context','inner','label','score'])[['mean','sd','q05','q25','q50','q75','q95']].mean().groupby(['context','label','score']).mean().reset_index();context.to_csv(HERE/'analysis/score_context_summary.csv',index=False)
    shift=read('oof_full_distribution_shift.csv');shift_summary=shift.groupby(['score','label'])[['OOF_mean','OOF_sd','full_validation_mean','full_validation_sd','standardized_mean_shift']].mean().reset_index()
    put('ASSET_AUDIT.md','# Phase5 资产审计\n\n225个checkpoint及预测/归一化/split通过当前hash与ID/标签审计：Frozen45、adapted45、deep-OOF135。旧审计对同SHA的135状态做过严格加载；本轮不重新实例化模型或加载其tensor做推理。\n\n'+table(pd.DataFrame(audit['missing']))+'''

60条reference fit/bank ID记录（15×全训练与3个OOF子折）均与合法train/query/excluded ID、类别和等人数bank对齐；对应22块变换hash记录存在。它们不是完整的序列化Scaler/PCA/bank向量。逐人的reference OOF输入未保存；仅有原报告总体均值/SD、输入Scaler参数与来源脚本。故“可以独立重放整个reference链/完整fusion训练OOF联合分布”是BLOCKED，不能以报告写了PASS推定状态文件必然存在。

合法Frozen包保存local、SSL residual、bilateral、subject、structured与双logits；其当前hash、ID、标签和概率一致性已核验，但没有独立的创建时逐特征hash清单。来源依据是上轮已验证提取脚本/模型/概率记录，而非本轮重新推理证明每一个embedding。Adapted源码在forward里计算layer5后送原base，未导出post-adaptation表征。stage4_windows.npz是390人共享的pre-adaptation中间缓存，不能当作成对的最终adapted特征；未解码。

查找范围与文件路径/hash在私有feature_discovery_inventory；缺失是指定来源范围内未找到合法资产，不声称已搜索整台机器所有文件。没有用文件名相似的其他fold表示替代缺失。原分区full-model split包含canonical excluded outer IDs，元数据存在不等于outer performance被读取；本轮不构造任何信号dataset。

逐对象完整fit/query/excluded IDs、source checkpoint/feature/normalization/split SHA、seed/context放restricted_manifests/all_model_reference_lineage.json。公开asset_inventory与reference_asset_lineage_summary只含聚合、路径hash及状态，不上传受保护manifest。
'''+ '\n\n'+BOUNDARY)
    put('SCORE_DISTRIBUTION_REPORT.md','# Phase5A：分数分布和排序\n\n'+table(score_summary[score_summary.score.isin(['wssl_logit','baseline_pdd','d_pd','d_dd','dd_aligned','relative'])])+'\n\n核心score Spearman（先split内seed均值，再15split）：\n\n'+table(co)+'''

WSSL融合输入为真实保存的 z_DD−z_PD，绝非raw z_DD；raw DD logit单独统计。PD-reference distance dPD为DD方向，DD-reference原距离dDD方向相反，相关分析用−dDD。relative=dPD−dDD，固定界0。分位数表是各unit分位数的seed-first平均，不是把重复appearance当独立人拟合总体分布。不同context的完整同口径结果见score_context_summary.csv；未跨监督encoder坐标拼embedding。

WSSL与relative总体Spearman .664238，PD .599875/DD .611911；相关但不相同不等于有独立增量。DD内dPD与−dDD高度反向（−.718442），两个单参考不是互相独立证据源。36项相关检验属于A_correlations探索族，全部p/q保留，不能选最有利一项论创新。

OOF/full均值和SD：\n\n'''+table(shift_summary)+'''

WSSL OOF/DD均值.060103、full validation .417361；PD为−1.122893→−1.410561。总体尺度变化与类条件位置变化存在，但并非所有context同向。reference只有OOF总体均值/SD，类条件与联合分布缺失，不补做Scaler/PCA或距离重算。same_subject_OOF_full_train_shift另比较同一训练人的OOF/full in-sample分数，只用于区分数据角色，不能当泛化验证。
'''+ '\n\n'+LIMIT)
    put('SUBJECT_COMPLEMENTARITY_REPORT.md','# Phase5A：受试者与排序互补性\n\n单seed、每split/seed平均四格纠错：\n\n'+table(ct)+'\n\n预登记四项A-primary：\n\n'+table(primary)+'\n\n四appearance患者汇总（每appearance三seed均值概率决策）：\n\n'+table(groupall[groupall.condition=='relative'])+'''

在WSSL严格排错的PD–DD pairs中，relative正确排序比例.416706，比.5低.083294（1/15正，split CI[-.113138,-.053701]，dz−1.362684，BH四项q=.001221）；subject-cluster中心差CI[-.136756,-.020932]。平均每unit救回223个严格错序pair，同时破坏276.47个原正确pair；pair不算独立样本。上轮固定10bin条件AUC为另一个条件问题，不能与“严格错序子集”混同，也不把前者当本轮新证据。

主stable-error68（PD35/DD33）、stable-correct279、unstable43均按既有定义逐人复核。relative对stable-error正确比例改善.154412，却对stable-correct损失.149642，整体患者四appearance正确比例−.081410。这些组按WSSL validation错误定义，天然有不同rescue/harm机会，不能解释为预先存在的临床可路由亚型。错误组中有获救者不等于全体净收益，也不能用于重加权。

H1结论：弱命题“存在与WSSL不同且部分正确的排序/预测”有直接证据；强命题“存在可稳定利用的WSSL未充分利用的独立疾病信息”仍证据不足。当前“互补不足以获得稳定分类净收益”得到支持，不能从gross rescue推导新分类器价值。主指标与三seed概率均值分组不是同一个估计对象，因此总体−.069791单seed Accuracy差与患者分组−.081410不能混写。
'''+ '\n\n'+LIMIT)
    coverage=read('reference_coverage.csv').groupby('label')[['low_n','high_n','low_error_rate','high_error_rate','high_minus_low_error_rate']].mean().reset_index()
    put('REFERENCE_COVERAGE_REPORT.md','# Phase5A：参考覆盖度\n\nu=min(dPD,dDD)，每validation split标签无关的q25/q75只定义描述组。\n\n'+table(coverage)+'\n\n'+table(coverage_tests)+'\n\n患者cluster、pooled exposure估计（与上表15split等权估计不同）：\n\n'+table(read('coverage_subject_cluster_bootstrap.csv'))+'''

高u不是通用“低可靠性”标记：PD高u错误率.300587 vs低u .150468；DD却为.229848 vs .519021，方向相反。两类各自的条件性cluster差仍为PD+.148441[.065230,.231629]、DD−.274209[−.439300,−.105957]。总体pooled差+.049538的cluster CI跨0；不把它与15split CI不跨0解释成两方法显著不同。

u与WSSL absolute probability margin的Spearman总体−.041976，PD−.081290/DD+.107506，弱且分组方向不同；u与relative总体+.370494。完整分组相关见coverage_correlations.csv。高/低组PD/DD组成不同（平均DD人数5.33 vs10.53），整体错误差同时混合类别/几何，不能视为可靠性因果证据。这里的u也不是临床严重度、异常疾病、错标或健康偏离指数。

禁止按本表进行筛人、采样、患者路由或训练权重改变。若未来检验该现象，应在独立数据重复同一固定诊断，不能基于验证标签构建门控。
'''+ '\n\n'+LIMIT)
    broken=bias[(bias.label=='DD')&(bias.event=='newly_broken')].dropna(subset=['mean_fusion_logit'])[['n','mean_baseline_pDD','mean_fusion_pDD','mean_W_contribution','mean_R_contribution','intercept','mean_fusion_logit']].mean().to_frame('mean_across_nonempty_units').reset_index(names='quantity')
    put('FUSION_FAILURE_ANALYSIS.md','# Phase5B：既有融合失败的直接数值解释\n\n45份既有系数恢复所有validation概率，最大误差2.22e-16，原决策完全相同。没有重新训练、选系数或调阈值。\n\n```text\nf = b + wW * (zDD-zPD-muW)/sigmaW + wR * (r-muR)/sigmaR\nP(DD) = sigmoid(f); DD iff f>0 (tie PD)\n```\n\n'+table(param_summary)+'\n\n类别输出及线性分解：\n\n'+table(bt)+'\n\n原DD正确被融合破坏的子集：\n\n'+table(broken)+'\n\n预登记B机制检验：\n\n'+table(mechanism)+'''

直接数值事实：训练DD比例均值.292298，b均值−1.022500，45个b全部为负；W系数45个为正，R有44个正/1个微负（最小−.003749），不是整体输入方向反转。DD组两项平均贡献+.422764和+.209945，仍不足抵消负截距，平均f=−.389791。原DD被破坏者保存概率由.653471降到.406604，平均f=−.385399；上表只对该事件非空的unit平均，以保持各线性项与截距同一权重，逐行恒等式完全通过。不能用不同权重的总体均值反推精确个体f。

全体DD预测比例.345980→.147096；DD mean P从.577047→.420457，PD false positive比例.217656→.072430。固定.5规则将大量原来略偏DD的样本判为PD；同一个排序分数加截距/改变概率尺度可保留较好排序而显著移动决策。融合AUROC+.004087只是点估计，历史split CI[-.004272,.010426]、cluster[-.007039,.015225]，不能说有可靠AUROC增益。

自然类先验解释“相容”而非已证明因果：融合是未加class_weight的C1 logistic，训练DD约29.23%，原WSSL用balanced CE。b接近但不等于训练prior log odds均值−.884272，分布形状和L2也影响b。validation−train DD比例均值仅−.0000166，不能用明显新队列prior漂移解释本例。没有用重加权或去截距重算任何新性能。

OOF/full迁移证据受限：reference标准化均值迁移+.061573，split CI[.022955,.103785]，BH4项q=.024902；其方向向DD，不是整体向PD。WSSL总体标准化均值−.088738，CI跨0；最终线性均值val−OOF(=b)仅−.005711，CI[-.135326,.128281]、q=.977966。因此“整体均值迁移单独导致DD崩塌”不受支持。两输入均有尺度/类条件差异，但缺reference OOF逐人联合分布及完整状态，不能因果归因OOD/full shift或计算完整训练混淆矩阵。

保存记录中未发现编码反转、输入标准化错用或跨fold bank混用，且deployment线性重放精确；这不等于缺失reference变换链已获独立数值验证。W的mean absolute贡献.637394，大于R .251440，差的CI[.218325,.554594]；不是R全面支配。协方差完整保存，不能仅以各方差比例把相关输入的贡献分解为独立因果份额。保持原REJECT，停止融合/阈值搜索。
'''+ '\n\n'+LIMIT)
    ad=read('adaptation_error_transitions.csv');ad=ad[(ad.aggregation=='single_seed')&(ad.error_group=='all')].groupby('label')[['n','gross_rescue','newly_broken','net_rescue','net_rescue_rate']].mean().reset_index()
    patterns=read('adaptation_subject_consistency.csv')
    put('ADAPTATION_SUBJECT_ANALYSIS.md','# Phase5C：Frozen与Adapted受试者变化\n\n'+table(ad)+'\n\n患者四appearance模式（每appearance三seed均值概率）：\n\n'+table(patterns[patterns.error_group=='all'])+'\n\n'+table(groupall[groupall.condition=='adapted'])+'''

单seed正式口径平均每unit救回5.844444、破坏5.2、净+.644444人；PD净+.6人，DD净+.044444。患者四appearance口径有43人改善、46人恶化、301不变；17人跨appearance既有rescue也有harm，不能声称所有人一致改善。后一个口径不是正式单模型性能，不能用其净数替换前者。

现有adapted vs frozen正式BA+.004757，split CI[-.000612,.010149]、历史本轮重分析6项q=.301647；subject-cluster[-.008245,.017479]。AUROC+.009355，cluster[.001011,.017972]，但历史开发联合门槛未通过、不复活候选。AUC的单一CI不抵消BA/recall/seed与既有停止规则。

stable-error中的改善伴随stable-correct上的损失，且这些组由原validation错误定义；这是异质预测变化的描述，不是可部署路由机制。按PD/DD与四事件的probability-implied log odds、真实标签signed margin变化见adaptation_prediction_drift.csv。rescued组平均true-signed log-odds变化约+1.116321、newly-broken约−1.186739，这与事件定义有关，不能拿同义数值关联当机制发现。

H3结论：预测/受试者改善异质性有直接证据；“由特定表征变化造成”无法核验，表征资产缺失，不能给出一致的疾病机制解释或据此新增适配训练。
'''+ '\n\n'+LIMIT)
    put('REPRESENTATION_DRIFT_REPORT.md','''# Phase5C：表征漂移缺口

状态：BLOCKED_ADAPTED_REPRESENTATION_MISSING。

Frozen的local、ssl_residual、bilateral、subject、structured包存在；对应adapted post-model特征未找到。既有stage4是pre-adaptation共享中间缓存，不能与Frozen最终特征冒充同层成对表示，也不能将不同fold embedding拼接。按用户限制，本轮不新提取HarNet、不重放adapted forward、不重新训练。

analysis/representation_drift.csv保留45×6=270条model/layer状态行；cosine、L2、Frozen/adapted norm为NA，绝不填0。每行附同输入预处理来源检查状态及可合法计算的预测变化摘要，但预测变化不是embedding drift。Adapted原始双logits也未保存，raw DD logit displacement无法由softmax反推；双方保存概率可计算量化后的log odds，名称明确为probability-implied，不能伪称原始双logits差。

若未来另获推理授权，最小补充是对同一split/seed、同一ID和预处理成对导出既有checkpoint的指定层，保存源hash/IDs/形状/拟合状态，不训练任何网络。之后只做匹配坐标内的描述，cosine/norm本身仍不是疾病因果或性能收益证据。当前不自动执行。
''')
    candidates=pd.DataFrame([
      ['1','补齐OOF/reference来源资产','保存状态缺口使迁移解释不可识别','当前缺逐人reference OOF与完整transform states','仅增加合法既有流程的状态导出与创建时hash；不改评分算法','0深训练；若需重放须另授权，时长未测','任一fit/query来源无法独立证实则停止机制推断；不顺带重拟合头'],
      ['2','独立数据上的覆盖方向复核','u不是跨疾病类别通用可靠性指标','PD与DD高低u错误方向相反，cluster均有同向证据','在独立合法队列重复同一冻结评分和预定class interaction诊断','0新训练；新数据访问/推理需另授权，成本未知','无法重复或存在队列/设备混杂则不发展为门控，不用真实类别路由'],
      ['3','既有checkpoint成对表征导出','适配改变不同患者的表示但不一定带来一致收益','43改善/46恶化；缺成对adapted表示','仅导出同输入同层Frozen/adapted特征，不改模型','45对已有模型推理；0训练，实际时长未测','身份/预处理不一致即停；漂移不能对应净改善则不进入新适配训练'],
      ['NO-GO','新深度模型训练','当前没有稳定可部署增量支撑','reference净损失、融合REJECT、local无稳定增量','本轮无单一结构改动获执行','预算0','先补来源与独立验证；不按个别人改善恢复搜索'],
      ['STOP','同队列融合/阈值/CE/LoRA/Adapter/Gate/Patch/Gyro搜索','不得用不同命名重启失败或未过闸门候选','既有停止规则持续有效','不执行','0','不能用本轮描述性子组、单一AUC或训练内指标重新开门']
    ],columns=['priority','candidate','research_hypothesis','existing_evidence','single_change','cost','stop_condition'])
    candidates.to_csv(HERE/'analysis/next_stage_candidate_priority.csv',index=False)
    put('NEXT_STAGE_CANDIDATES.md','# 下一阶段候选：不执行\n\n'+table(candidates)+'''

当前没有足够证据支持新的深度训练。值得先检验的是两个可反驳命题：H4，reference距离与错误的关系受疾病类别/覆盖几何影响，不能当通用uncertainty；H5，balanced-CE原生分数与自然先验OOF readout在固定操作点不兼容，缺失的联合来源状态阻碍可靠归因。H4需独立队列重复预定交互诊断；H5首先需完整保存源状态而非搜索新阈值。

可能的方法学贡献是可复用、可证伪的“来源审计—双向rescue/harm—类条件覆盖—读出数值分解”评估框架；当前仅为研究假设与审计工程，不是已验证的新算法创新。普通kNN、CE reduction、相关系数或某个子组获救本身不构成论文贡献。所谓STOP针对这些固定配置在相同重复开发队列上的继续搜索，不是全方法家族被科学上永久否定。
''')
    evidence=pd.DataFrame([
      ['H1：部分不同且正确的信息','中等、描述性','有gross rescue/错序救回，但整体net<0，条件错序正确率<.5','独立、可部署增量未建立'],
      ['H1：无法形成稳定净收益','较强、限当前固定方法','PD损失、BA下降、stable-correct伤害，cluster一致','不否定所有表示/算法家族'],
      ['H2：固定操作点的数值解释','强、代数可重放','45概率误差2.22e-16，负b+有限DD证据→PDD降低','不是负截距应被修改的授权'],
      ['H2：OOF/full总体均值迁移是主因','不支持该强命题','最终均值迁移近0、CI跨0；reference均值向DD','联合/类条件reference OOF缺失，不能排除其他分布变化'],
      ['H3：预测变化具有异质性','较强、描述性','43改善/46恶化/301不变','指标聚合口径不同不能混写'],
      ['H3：表征变化导致改善机制','BLOCKED','adapted同层成对表征缺失','不能给cosine/L2或因果结论'],
      ['独立外部泛化','未验证','无新独立队列/outer evaluation','三种开发不确定性均不能替代外部验证']
    ],columns=['question','evidence_strength','support','limit'])
    status=pd.DataFrame([['资产审计','完成、存在MISSING子项','45+45+135；60 reference ID链'],['5A分布/互补/覆盖','完成','固定分数、新统计；无新分类器'],['5B fusion恢复','完成','45概率精确重放；不改参数'],['5B OOF reference联合分布','BLOCKED','逐人OOF分数/变换状态未保存'],['5C预测变化','完成','全体与PD/DD、错误组、各seed'],['5C表征漂移','BLOCKED','未保存post-adaptation同层特征'],['新训练/拟合/神经推理','未运行','全部0'],['发布快照','准备并核验；最终提交receipt另存源目录外','仅代码、协议、报告、聚合结果，不在报告中自指提交SHA']],columns=['work','status','reason'])
    put('FINAL_REPORT.md','# Phase5 最终报告\n\n保留Frozen WSSL-STR；新深度训练0、新轻量模型/Scaler/PCA拟合0、新神经推理/HarNet提取0。完成现存分数的分析，明确阻断缺失资产所需的子分析，不把部分来源审计冒充完整数值重放。\n\n'+table(status)+'\n\n'+table(evidence)+'\n\n'+table(candidates)+'\n\n正式六指标（原预测复核，非新模型）：\n\n'+table(means)+'''

1. Dual-reference有部分WSSL未做出的正确排序/预测，但尚不能确认独立、可稳定利用的信息增量。WSSL错序pairs内relative正确率.416706；全体每unit救回9.51同时破坏16.78，净−7.27。不能只在错误组报获救。
2. 不能稳定修复全体错误。PD recall Δ−.106607、DD Δ+.019001（CI跨0），BA Δ−.043803；stable-error获益伴随stable-correct损失，不能逐患者路由。
3. Fusion直接数值原因可恢复：b平均−1.0225，DD两项平均证据+.422764、+.209945不足抵消负截距，DD平均f−.389791；DD预测率.345980→.147096、DD Recall .656945→.327707。概率重放误差2.22e-16，未发现已有部署数值中的全局方向/编码错用。AUC+.004087的split与cluster CI均跨0，排序与固定操作点不是同一件事。
4. OOF/full确有尺度和类条件变化，但最终整体均值移位−.005711、CI跨0；reference移位反而向DD。没有证据把失败单独归因“总体迁移”。缺reference OOF逐人联合分布和transform state，完整机制归因BLOCKED；自然类先验与balanced-CE/readout目标差异仅是相容解释，未做因果识别。
5. Adapted改变一部分人的预测，非一致改善。四appearance、三seed均值概率诊断下43改善、46恶化、301不变；正式单seed BA仅+.004757且cluster CI跨0。表征机制本轮无法回答，缺同层adapted特征，不能用预测变化冒充cosine漂移证据。
6. 最大未解问题是独立泛化、可靠性可识别性与可追溯资产完整性。u高低错误关系在PD/DD方向相反，不是通用可靠性；旧outer文字接触永久保留，同队列多轮开发不能提供独立验证。
7. 当前无新深度训练方向获证据支持。先补状态与独立数据复核，再决定是否值得另授权的成对既有模型推理；同队列fusion/threshold/CE/adapter/patch/gyro搜索停止。
8. 可检验新假设是类条件覆盖几何与读出源分布/操作点不匹配；这是方法学贡献的研究入口，尚非已验证新算法。候选的单一改动、成本和停止条件见上表，不执行。

全部predeclared p/q、effect、seed/split SD、subject-cluster CI见STATISTICAL_APPENDIX和analysis。统计、源代码、失败脚本版本与日志均留存；缺失值明确NA。已有温度诊断、CE NOT_RUN_GATE_FAIL及所有历史REJECT不改写。输出只在/home/zyt/deep_final/artifacts/wssl_complementarity_phase5_20261111，mainline及历史正式产物不修改。
'''+ '\n\n'+BOUNDARY+'\n\n'+LIMIT)
    appendix='# Phase5 完整统计附录\n\n'+LIMIT+'\n\n新BH族：A_primary4、A_correlations36、A_coverage3、B_mechanism4。历史六指标p/q按原族逐字复制，不用本轮新族覆盖。relative是同一确定reference评分，formal identity表中按baseline seed重复用于配对，不是3个新训练seed；其seed SD为NA。\n\n'
    for n in ['predeclared_tests.csv','seed_split_stability.csv','per_seed_metrics.csv','context_sensitivity.csv','subject_cluster_bootstrap.csv','coverage_subject_cluster_bootstrap.csv','error_group_subject_cluster_bootstrap.csv','reused_B_relative_vs_WSSL_paired.csv','reused_B_fusion_paired.csv','reused_A1_adapted_paired.csv']:
        appendix+='## '+n+'\n\n'+table(read(n))+'\n\n'
    put('STATISTICAL_APPENDIX.md',appendix)
    write_json(HERE/'decision.json',dict(status='COMPLETE_AVAILABLE_ANALYSES_WITH_BLOCKED_SUBANALYSES',retained_model='Frozen WSSL-STR',retained_config_changed=False,new_deep_training_runs=0,new_lightweight_classifier_fits=0,new_scaler_PCA_fits=0,new_temperature_fits=0,new_neural_inference=0,HarNet_feature_reextractions=0,threshold_or_coefficient_searches=0,H1='partial distinct correctness observed; stable exploitable independent increment not established; net benefit absent',H2='deployment numeric decomposition verified; overall mean-shift-only explanation unsupported; full source-distribution mechanism blocked by missing reference OOF rows/states',H3='heterogeneous prediction changes verified; representation mechanism blocked',missing_assets=audit['missing'],fusion_remains='REJECT',CE_remains='NOT_RUN_GATE_FAIL',calibration_remains='DIAGNOSTIC_ONLY',historical_outer_exposure_permanently_inherited=True,strict_entire_research_no_outer_exposure_claim=False,new_outer_evaluation=False,external_validation=False,stop='Publish approved aggregate audit snapshot using existing workflow; no automatic next phase'))
    # Source code is read as evidence, not executed. No source contains outer performance tables.
    reviewed=[]
    for pth in [OLD/'scripts/calibration_fusion.py',OLD/'scripts/typicality.py',OLD/'scripts/extract_baseline_v2.py',ROOT/'artifacts/phase3c_ssl_adaptation_20260930/scripts/phase3c_hooks_e1_frozen.py',ROOT/'mainline/src/metrics/classification.py']:
        txt=pth.read_text();reviewed.append(dict(path=str(pth),sha256=sha(pth),bytes=pth.stat().st_size,complete_source_read=True))
    pd.DataFrame(reviewed).to_csv(HERE/'analysis/reviewed_source_code.csv',index=False)
    guard();print('PHASE5_REPORTS_COMPILED',flush=True)

if __name__=='__main__':main()
