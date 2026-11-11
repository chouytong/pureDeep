"""Compile existing results; never fit or select another candidate."""
from common import *
import shutil, subprocess

def table(f, digits=6):
    f=f.copy()
    def fmt(v):
        if pd.isna(v): return 'NA'
        if isinstance(v,(float,np.floating)): return f'{v:.{digits}f}'
        return str(v).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(map(str,f.columns))+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+''.join('| '+' | '.join(fmt(v) for v in row)+' |\n' for row in f.itertuples(index=False,name=None))

def read(n): return pd.read_csv(HERE/'analysis'/n)
def put(n,s): (HERE/n).write_text(s.rstrip()+'\n')
def jt(n): return json.loads((HERE/'analysis'/n).read_text())
def means(f,cols=METRICS):return f.groupby(['context','inner'])[cols].mean().mean()
def metric_table():return read('ALL_stability_summary.csv').pivot(index='condition',columns='metric',values='mean')[METRICS].reset_index()

LIMIT='''15 个 development splits 共享受试者和训练池，45 次已有 seed/run 不是 45 个独立样本。主统计先在同一 split 平均三个 seed 的指标，再汇总 15 splits；不是概率 ensemble。主 bootstrap seed=20261111、10,000 次；Wilcoxon 和 BH family 见各表。受试者分层 cluster bootstrap 将同一人的全部 appearance 一起重采样，仅描述固定已拟合模型下的条件性不确定性，不覆盖重训练、开发多重选择或外部泛化。5 个 context 汇总仅为 inner-development 分组，不是 outer-test。'''
DEVIATION='''边界偏差（必须保留）：bootstrap 历史审计完整读取的旧综合报告 `reports/PURE_DEEP_RESEARCH_FINAL_REPORT.md` 含旧 V8 outer-test 汇总，派生的初版 ledger/摘录也包含该内容。因此本轮“完全未接触历史 outer 指标”不成立。未加载 outer-test 原始信号/标签表、逐受试者预测，也未运行 outer evaluation；但不能事后撤销已有的文字接触。该历史值未进入新拟合器或选择规则，所有新候选已停止，当前主线未改变。初版 `protected_assets.json` 的 `outer_access:false` 仅是当时声明，由本最终审计更正，不能作为合规证明。新分析只能作为有此偏差披露的 development 诊断。另：压缩 NPZ 缓存按整数组解压后仅索引合法 train/validation 行，存储读取粒度不等于逐行物理隔离；未用 outer 行拟合或评分。不能声称存在覆盖整个进程的强制系统级 outer-access sentinel。'''

def ledger():
    src=HERE/'EXPERIMENT_LEDGER.csv'
    if not (HERE/'restricted_manifests/EXPERIMENT_LEDGER_initial.csv').exists():
        shutil.copy2(src,HERE/'restricted_manifests/EXPERIMENT_LEDGER_initial.csv')
        shutil.copy2(HERE/'EXPERIMENT_LEDGER.md',HERE/'restricted_manifests/EXPERIMENT_LEDGER_initial.md')
    f=pd.read_csv(HERE/'restricted_manifests/EXPERIMENT_LEDGER_initial.csv').fillna('')
    inv=[]
    specs={'V8':(Path('/home/zyt/MFAM/outputs/pads_classification'),'pure_deep_normfree_moments_v8*development_20260916'),
      'V8-GN':(ROOT/'foundation_validation/outputs/foundation_validation','groupnorm_nr01_seed*'),
      'STR-01':(ROOT/'foundation_validation/outputs/structured_token_residual','str01_seed*'),
      'WSR-01':(ROOT/'foundation_validation/outputs/wrist_structured_residual','wsr01_seed*'),
      'prototype':(ROOT/'foundation_validation/outputs/invariant_representation','pa0*_lambda*_seed*'),
      'strong-backbone':(ROOT/'foundation_validation/outputs/strong_backbone_benchmark','*seed*')}
    for eid,(root,pattern) in specs.items():
        for folder in sorted(root.glob(pattern)):
            if 'smoke' in folder.name:continue
            for p in sorted(folder.glob('outer_*/inner_*/stage_status.json')):
                obj=json.loads(p.read_text())
                inv.append(dict(experiment=eid,variant=folder.name,path=str(p),sha256=sha(p),complete=obj.get('status')=='complete'))
    counts=pd.DataFrame(inv)
    counts.to_csv(HERE/'analysis/foundation_inner_stage_inventory.csv',index=False)
    actual=counts[counts.complete].groupby('experiment').size().to_dict()
    expected={'V8':45,'V8-GN':45,'STR-01':45,'WSR-01':45,'prototype':90,'strong-backbone':135,'phase3c':360,'wssl-ema':90}
    dates={'V8':'2026-09-16','V8-GN':'2026-09-17','STR-01':'2026-09-21','WSR-01':'2026-09-22','prototype':'2026-09-19','strong-backbone':'2026-09-21','DSG-01':'2026-09-26','RGD-01':'2026-09-26','PRR-01':'2026-09-26','PAG-01':'2026-09-26','phase3c':'2026-10-01','phase3d':'2026-10-01','phase3e':'2026-10-01','phase4':'2026-10-02','wssl-ema':'2026-10-02','wssl-review':'2026-10-04','frequency':'2026-10-07','wssl-contribution':'2026-10-07','activity-scaling':'2026-10-08','patch-str':'2026-10-08','patch-recheck':'2026-11-11 recheck; original training 2026-10-08'}
    for j,row in f.iterrows():
        eid=row.experiment_id
        if eid in actual:f.loc[j,'actual_completed_inner_stages']=actual[eid]
        if eid in expected:f.loc[j,'expected_runs']=str(expected[eid])
        f.loc[j,'actual_new_deep_runs']=str(f.loc[j,'actual_completed_inner_stages'])+' historical runs; 0 this task'
        f.loc[j,'formal_conclusion_date']=dates.get(eid,'unknown; artifact date is not a verified formal conclusion date')
        f.loc[j,'major_metrics']='See complete original report and history_sources.csv; initial automatic mixed-table extraction is superseded.'
        f.loc[j,'rejection_reason']=row.conclusion+'; fixed configuration only, not family-wide falsification'
        if eid in ['DSG-01','RGD-01','PRR-01','PAG-01']:f.loc[j,'status']='completed-diagnostic; conditional neural candidate not-run'
        if eid=='phase3a':f.loc[j,'status']='rejected'
        if eid in specs:f.loc[j,'protocol_and_sources']=row.protocol_and_sources+';analysis/foundation_inner_stage_inventory.csv'
    # Existing records account for all supervised runs, separate the previously omitted OOF chain.
    new=dict(f.iloc[0]);new.update(experiment_id='phase3c-deep-OOF',target_baseline='Frozen WSSL',unique_change='3-fold fixed epoch10 within target inner-train',expected_runs='135',actual_completed_inner_stages=135,actual_new_deep_runs='135 historical; 0 this task',status='completed-diagnostic',formal_conclusion_date='2026-10-01',major_metrics='provenance PASS; not a new classification candidate',conclusion='135 actual checkpoints and prediction sets reused',rejection_reason='not applicable',protocol_and_sources='analysis/OOF_inventory_135.csv;analysis/OOF_provenance.json',source_hash=sha(HERE/'analysis/OOF_inventory_135.csv'),baseline_reused=True)
    f=pd.concat([f,pd.DataFrame([new])],ignore_index=True)
    f.to_csv(src,index=False)
    put('EXPERIMENT_LEDGER.md','# 历史实验总账\n\n初版只数 artifact/runs，遗漏 foundation_validation/outputs，已按真实 inner stage 元数据补齐。初版保留于私有 restricted_manifests。来源路径/hash 见 CSV、history_sources 和 foundation_inner_stage_inventory。阶段日期无法核实时写 unknown。\n\n'+table(f[['experiment_id','expected_runs','actual_completed_inner_stages','status','conclusion']])+'''

计数口径：prototype=两种 loss×45=90；strong-backbone=ResNet45+InceptionTime45+GyroFFT45=135。Phase2=baseline重训45+16候选×45=765。Phase3C=360监督训练+135独立 deep-OOF，p100 frozen/STR复用。EMA=ordinary复现45+EMA45，不是90个新结构。Phase3D表内135监督run；train-only masked pretext另有历史报告记录45，未把它混入监督run，也未重新加载其权重。smoke不计入正式run。所有这些均非本轮新训练。

Phase3D条件activity gate当时NOT_RUN；后续11-scalar activity scaling确已做45次，INCONCLUSIVE/DO NOT RETAIN，不能改写有害。Patch P1/P2在10月8日训练，11月11日仅复核。Phase C当时未启动独立loss数值审计；本轮才补做。Full HarNet fine-tuning、Patch P3、双prior均不因未做而自动获准。prototype alignment训练loss与当前冻结参考距离分类不是同一实验。
''')
    return f

def main():
    guard();hist=ledger();mt=metric_table();base=mt[mt.condition=='baseline']
    put('AUDIT_REPORT.md','# 主线与边界审计\n\n当前 retained single-model 仍为 Frozen WSSL-STR；A0候选未运行，B融合拒绝，校准只保留诊断sidecar。\n\n'+DEVIATION+'\n\n'+LIMIT+'''

完整历史文档157项、24实验组在启动时读入，来源清单与SHA256见 analysis/history_sources.csv；后续补充实际foundation inner stage和135 OOF资产。历史原文、checkpoint、split、归一化、预测与负结果均受hash锁保护，不覆盖历史报告。冻结入口与每次失败修订分别保存 configs/*_freeze.json；最终完整性见 analysis/final_integrity.json。

已确认：HarNet layer5、middle+final、输出adapter、gated fusion、train-only masked adaptation已做且未保留；deep-OOF阈值诊断已做，本轮未重复阈值网格。Activity scaling不是从未尝试，也不是证明有害。BioPM已测受限acc neural export，不是完整BioPM方法。历史固定配置失败不能泛化整个方法家族。

范围限制：本项目当前为PD/DD受试者分类诊断；尚不能当成临床手部连续严重程度回归结果。DD不是健康正常人群。无新的临床因果结论、独立外部验证或新最佳模型声明。
''')
    put('BASELINE_REPRODUCTION.md','# 冻结基线复现\n\n'+table(base)+'''

45个checkpoint全部独立重新推理；完整validation概率最大绝对差1.1102230246251565e-16，决策/指标一致。归档没有logits列，故未声称与归档logits比较；新logits保存在私有features。PD=0/DD=1，原argmax在tie选0；不是三seed概率ensemble。运行import根目录 /home/zyt/deep_final/mainline，非旧foundation副本。

实际local六通道Acc+Gyro temporal/learned moment wrist64D；外部frozen HarNet10 Acc1024D，经LayerNorm+Linear到64D后相加；bilateral258D、activity/subject path、11×16 ordered STR路径保持，最终logits相加。分类侧143,172可训练参数，外部10,457,408冻结；本轮不实例化更新外部encoder，以受保护缓存推理。完整zero-residual含投影bias，zero SSL input不能等价替代。独立显式接口避免deepcopy旧闭包wrapper。

tests/model_correctness.json覆盖三seed代表batch的真长度/padding、无效腕活动、ID重排、错误ID/重复ID/标签错位、reload、实例参数/cache/RNG隔离、分支logits分解。所有45概率核验不是只有代表batch测试。复用此前15个完全精确的train normalization refit证明（hash见baseline_reproduction.json）；OOF另重算45组，覆盖135模型。checkpoint来源/hash由protected_assets及OOF inventory锁定。

135历史deep-OOF状态和预测实际可用，均严格加载；训练/归一化排除query，全部来源排除当前target validation。固定epoch10、forced last，不以query指标选checkpoint。OOF较小训练池与full best-epoch分数尺度不等同。

初版失败：CPU softmax与CUDA归档差2.98e-8且train/val混合batch；v2分离loader并在CUDA softmax恢复原路径，未放宽1e-12 tolerance。旧脚本/冻结记录保留。
'''+LIMIT)
    gr=pd.DataFrame(jt('A0_real_gradients.json')['cases'])
    sampler=read('A0_sampler_summary.csv')
    put('LOSS_NUMERICAL_AUDIT.md','# A0 weighted CE 数值审计\n\n结论：AUDIT_COMPLETE / PRECISION_LIMITED；候选 NOT_RUN_GATE_FAIL，新增deep训练0。五项闸门a/c/d通过，b历史明确全局风险目标未确认，e实际TF32参数梯度严格精度门槛未通过。不能把未训练候选写成性能REJECT。\n\n'+table(gr)+'''

实际PyTorch2.8 hard-index、无smoothing路径：L_batch=Σw_y CE/Σw_y；固定目标L_fixed=Σw_y CE/(B·mean_train(w_y))。同一batch两者梯度相差公共标量Σw_y/(B·mean_train(w_y))，类别间相对权重不变。默认weighted mean正确，不是PyTorch bug；有限自然随机batch的期望比值不必等于固定全局平衡经验风险。逆频率train_balanced本身不能证明历史明确声明了后一目标，更不能证明DD错误由前者导致。[安装版本官方定义](https://docs.pytorch.org/docs/2.8/generated/torch.nn.CrossEntropyLoss.html)。

RandomSampler自然采样，batch8、drop_last=False，seed+原loader offset；45配置各50epoch标签组成shadow，2250记录，不更新optimizer。实际类别比例、末batch、归一化系数和类别总系数见A0_sampler_summary.csv，逐epoch风险见A0_epoch_risk.csv。epoch实现以batch大小加权平均batch-ratio，通常不等于全集weighted-risk，也不是proper unweighted NLL。

Float64手算/自动微分覆盖单PD/DD、B1、balanced、末小batch、未加权/加权，容差1e-12通过；soft-label默认按样本数mean，不能套hard-label分母。真实三seed固定模型/输入的logit梯度、参数梯度及裁剪均记录。seed44实际TF32相对误差6.412217e-5超过预定2e-5；未改容差。只在独立审计进程关闭TF32的IEEE-FP32影子均通过，正式训练设置未变。数学恒等式不等于有限精度逐位恒等。

失败修订：v1暴露TF32参数梯度误差；v2另发现以torch FP32重算类权重与正式Python双精度除法再转FP32在42/45组相差1ULP；v3直接用正式权重，并逐位复核原路径。修订未消除实际TF32失败、未开启训练。早期collection status=PASS仅代表收集完成，以A0_precision_assessment.json及A0_gate.json为最终解释。
''')
    cats=read('A1_category_error_summary.csv')
    def corr(n):
        f=read(n);cols=['corrected','newly_wrong','net'];keys=['condition','label'] if 'condition' in f else ['label']
        return f.groupby(keys)[cols].mean().reset_index()
    put('MODALITY_ERROR_DIAGNOSTICS.md','# A1 模态和错误诊断\n\n主错误定义每appearance先均值三seed概率，再按四appearance错误比例>=.75：68人（PD35/DD33）；全部四appearance错38（PD14/DD24），全部12次错12，是不同敏感性定义。\n\n'+table(cats)+'\n\nAdapted相对Frozen每split/seed平均纠错（对全体人，非只筛错误者）：\n\n'+table(corr('A1_adapted_corrections.csv'))+'\n\n固定probe完整配对：\n\n'+table(read('A1_probe_paired.csv'))+'''

既有45个last-block adapted预测复用，normalization hash与baseline一致；未重跑37-condition contribution矩阵。各条件全部六指标和稳定性见STATISTICAL_APPENDIX.md。source_category仅从既有CrossArms manifest来源字段聚合，未新增临床个体表。分数/margin/seed与appearance稳定性见上表及私有错误表，不用于逐患者路由、采样、删人或重加权。

唯一预固定probe：原WSSL logit + flattened22×1024外部Acc cache的train StandardScaler/PCA16为control；candidate仅加ordered11×2×64 local特征经独立train Scaler/PCA16。最终StandardScaler+C=1 L2 logistic，natural class weights=None，共90个轻量fit。ΔBA=+.000476、ΔAUC=+.000166，CI均跨0，没有稳定local增量。local混合Acc+Gyro，未分离Gyro-only；不能归因Gyro，也不能证明Gyro无用。底层监督local encoder见过这些inner-train人，probe不是整个模型链OOF。

新特征含local、SSL投影完整residual、bilateral、subject、structured，来源由baseline提取脚本固定。没有将输入删除/置换解释为重新训练单模态或疾病因果；未新训练encoder/内部adapter。
''')
    cal=read('A2_calibration_metrics.csv');ct=cal.groupby(['condition','context','inner'])[['nll','brier','ece','classwise_ece','calibration_intercept','calibration_slope']].mean().groupby('condition').mean().reset_index()
    put('CALIBRATION_DIAGNOSTICS.md','# A2 概率校准\n\n'+table(ct)+'\n\n'+table(read('A2_calibration_directional.csv'))+'''

全部45个T仅在对应target inner-train的合格deep-OOF log odds上拟合，固定正温度T∈[.05,20]，无阈值搜索；样本端点概率0行，无clip后伪造logit。validation截距/斜率90个descriptive fit只用于诊断，不部署；参数、梯度收敛、逐seed/split指标与固定10等宽reliability bins保留。proper unweighted NLL、双类sum Brier（2×二分类平方误差）、top-confidence ECE、classwise ECE均与weighted训练CE区分。

NLL .616720→.577543，Δ-.039178，12/15改善，CI[-.064443,-.016595]，BH三项q=.016113。Brier .386441→.383837、ECE .110333→.107569，两者CI跨0。预登记mean NLL<0且Brier<=0的诊断规则通过，标记RETAIN_DIAGNOSTIC_ONLY；不代表校准已外部验证，更不改变原单模型身份。正T保留每个已有模型的排序及原.5决策，AUROC/BA无提升。

OOF训练池更小、epoch10固定，deployment是full pool/best epoch；分数/参考库尺度迁移见A2_Bfusion_score_transfer.csv。该分布差不能由OOF来源合法自动消除；不使用其他见过当前validation的context模型补分数。按PD/DD/来源组的NLL/Brier/ECE及DD概率可靠性见A2_grouped_calibration、A2_DD_reliability_bins；这是描述性分组而非亚型校准器。reliability.svg为聚合图，重复appearance不当独立样本。

方法参考：[Guo等，ICML2017](https://proceedings.mlr.press/v70/guo17a.html)。'''+ '\n\n'+LIMIT)
    put('TYPICALITY_REPORT.md','# B 冻结双参考典型性\n\n'+table(mt[mt.condition.isin(['strict_pd','strict_dd','common_pd','common_dd','relative','centroid_relative','full_pd','full_dd','full_relative'])])+'\n\n预登记主比较（2项AUROC的primary BH）：\n\n'+table(read('B_primary_paired.csv'))+'\n\n与WSSL全体人纠错，单位为每split/seed平均人数：\n\n'+table(corr('B_all_subject_corrections.csv'))+'\n\n全体人相关/决策一致率：\n\n'+table(read('B_agreement_with_WSSL.csv').groupby('condition')[['score_spearman','prediction_agreement']].mean().reset_index())+'''

strict PD-only/DD-only所有PADS拟合仅该类；各自5fold subject crossfit拟合整个链，95%参考类距离定阈值，不使用另一类人数。外部HarNet独立冻结无PADS标签训练。common空间则监督class-conditional，PD/DD均疾病群体，不能称无监督PD发现或生理正常异常检测。

固定22 activity/wrist各自Scaler→PCA(min16,合法rank、无whitening、randomized seed20261111)→L2，cosine k5个不同subject，双腕等权后11活动等权；零向量距离1、rank0规定、少于5不降低k。训练query排除自身。common主分析以固定hash等人数bank，完整不平衡bank为预定敏感性；阈值95%参考类训练leave-self-out（并非全链OOF），relative界0。strict阈值与common阈值来源不能混同。

5,280次固定训练变换重放hash/norm审计：实际最小rank16、拟合向量与选中bank零向量均0，无降k/退化失败。fit-ID manifests私有；同一外部encoder缓存坐标系，未拼接不同监督fold embedding。仅用保存的subject聚合cache，无伪造2秒patch或将976点虚构两个完整10秒窗。距离未映射合法概率，NLL/Brier/ECE=NA。

relative相对两个单参考AUROC稳定占优；同空间centroid/full-bank敏感性完整24项BH见B_all_paired。固定10个WSSL概率bin的条件AUC screen增量CI[.086286,.129363]只提供粗分箱互补迹象，不能证明独立临床增量。relative自身BA .675842/AUC .735639低于WSSL。

闸门通过后仅执行一次两分数C1 L2 logistic：train deep-OOF WSSL logit+完全同算法train-chain OOF relative distance，各OOF变换/参考库排除query；45个fit，部署full inner-train链。BA .627638（Δ-.092006，0/15胜）、DD recall .327707（Δ-.329238），F1与seed稳定性护栏也失败。融合REJECT，不搜索权重/阈值/类别权重补救，不取代143,172参数原生模型。独立pipeline的输入scaler/coefficients私有保存，不能把它描述为原模型结果。

普通深度kNN本身不是创新：[Sun等，ICML2022](https://proceedings.mlr.press/v162/sun22d.html)。'''+ '\n\n'+LIMIT)
    put('NEGATIVE_RESULTS_DO_NOT_REPEAT.md','''# 本轮与历史负结果边界

本轮：固定CE candidate未运行（b/e闸门失败），不是性能负结果；local probe无稳定增量；两分数融合REJECT（BA/F1/DD/BA seed稳定性）；relative不是更佳主模型；temperature只诊断保留，Brier/ECE CI跨0。实际TF32参数梯度seed44未过，不隐藏在collection PASS后。

历史：STR后WSR、prototype loss、strong backbone/GyroFFT、Phase2各16配置、Phase3A、HarNet layer5/multi、output adapter/gate/domain-masked、HarNet5/BioPM受限export、window dynamics、EMA、LR1e-4、frequency/capacity残差、Patch P1/P2均未成为当前retained。Activity scaling是INCONCLUSIVE，不称有害。详见完整ledger与原来源hash。

不得更换名字/rank/宽度/seed/阈值重复以上固定失败配置。也不由单配置拒绝否定PEFT/SSL/Gyro/patch整个家族。Phase3C的135模型OOF阈值诊断已完成，不重跑；PhaseC loss当时未启动，本次补齐但不训练。Full HarNet、Patch P3、cycle预训练、PaAno triplet、XTransfer层搜索均NOT_RUN/UNAUTHORIZED，本轮结束。
''')
    priority=pd.DataFrame([
      ['0','原Frozen WSSL-STR','RETAIN_CURRENT','45推理精确；无新联合门槛赢家'],
      ['1','正温度校准sidecar','DIAGNOSTIC_ONLY','NLL均值规则通过；Brier/ECE不确定；不部署替换主模型'],
      ['2','双参考relative评分','COMPLETED_EXPLORATORY','优于单参考，低于WSSL；受历史outer文字接触偏差限制'],
      ['stop','两分数融合','REJECT','BA/F1/DD/seed guard失败'],
      ['stop','固定分母CE','NOT_RUN_GATE_FAIL','历史风险目标未确认；实际TF32精度失败'],
      ['future','新内部适配/局部网络','NOT_AUTHORIZED','缺少足够稳定增量和机制证据，不起训练']
    ],columns=['priority','candidate','status','reason'])
    put('CANDIDATE_PRIORITY.md','# 候选与停止点\n\n'+table(priority)+'\n\n后续仅可先审阅明确的机制假设与周期/模态隔离证据；当前无足够稳定补救增量，不启动新内部适配或局部训练协议。若未来另获授权，应先隔离开发资料、重新冻结范围与真正独立验证方案。不得把本轮诊断重新标记为未接触outer结果。')
    paired_files=['B_fusion_paired.csv','B_relative_vs_WSSL_paired.csv','B_all_paired.csv','A1_probe_paired.csv','A1_adapted_paired.csv','A2_calibration_directional.csv','subject_cluster_bootstrap.csv']
    appendix='# 全部配对统计与稳定性\n\n'+LIMIT+'\n\nBH family：B_primary两项AUROC；B_all探索24项；融合六项；relative-vs-WSSL六项；probe六项；adapted重分析六项；calibration三项。历史p/q不覆盖，新的seed/family不能替换原报告结论。calibration原wins计正delta=更差，以improved_count为改善。B_all保留局部bh_q列，跨24项取exploratory_bh_q；主两项取primary_bh_q。\n\n'+table(mt)+'\n\n'
    for n in paired_files:appendix+='## '+n+'\n\n'+table(read(n))+'\n\n'
    appendix+='## SD（deterministic B不伪造3seed）\n\n'+table(read('ALL_stability_summary.csv'))+'\n\n## 五context敏感性\n\n'+table(read('five_context_sensitivity.csv'))
    seedrows=[]
    for n in ['baseline_metrics.csv','B_fusion_metrics.csv','A1_adapted_metrics.csv','A1_probe_metrics.csv']:
        q=read(n);keys=['condition','seed'] if 'condition' in q else ['seed'];v=q.groupby(keys)[METRICS].mean().reset_index();v.insert(0,'source',n);seedrows.append(v)
    seeds=pd.concat(seedrows,ignore_index=True);seeds.to_csv(HERE/'analysis/per_seed_means.csv',index=False)
    appendix+='\n\n## 各seed均值\n\n'+table(seeds)
    put('STATISTICAL_APPENDIX.md',appendix)
    decision=dict(status='COMPLETE_WITH_DISCLOSED_PROTOCOL_DEVIATION',retained_native_model='Frozen WSSL-STR',retained_native_model_changed=False,new_deep_training_runs=0,deep_optimizer_updates=0,A0='NOT_RUN_GATE_FAIL',A0_actual_precision='FAIL_SEED44_UNCHANGED_TOLERANCE',A1='NO_STABLE_LOCAL_INCREMENT',B_reference='EXPLORATORY_RELATIVE_OVER_SINGLE_REFERENCE',B_fusion='REJECT',A2_temperature='RETAIN_DIAGNOSTIC_ONLY_NOT_DEPLOYED',outer_evaluation_run=False,historical_outer_summary_text_exposed=True,strict_outer_no_access_certified=False,external_validation=False,stop='No new candidate or training. Preserve native WSSL and submit audited snapshot.',publication='Authorized; use existing selective publishing workflow, exclude private arrays and tables.')
    write_json(HERE/'decision.json',decision)
    status=pd.DataFrame([['历史审计','完成并披露边界偏差','24组+135 OOF资产'],['基线','复用+45精确推理','不重训'],['A0','完成；CE候选未运行','b/e闸门失败'],['A1','完成/复用','90轻量probe；旧adapted45'],['A2','完成；诊断保留','45温度；90描述斜率'],['B','完成；融合拒绝','9确定评分方法；45融合fit'],['新深模型/outer','未运行','0 / 0'],['论文全文/分区','部分待核验','见LITERATURE_VERIFICATION.md']],columns=['工作','状态','证据/原因'])
    put('FINAL_REPORT.md','# Frozen WSSL-STR 主线审计与双参考诊断：最终报告\n\n原Frozen WSSL-STR继续保留；本轮新增deep training **0**，optimizer更新0，未运行outer evaluation。没有符合联合判别门槛的新主模型。\n\n'+table(status)+'\n\n'+table(priority)+'\n\n'+table(base)+'\n\n'+DEVIATION+'''

45次完整validation重推理概率误差最大1.11e-16；135历史deep-OOF checkpoint及预测确实保存并通过来源审计。A0正式weighted mean不是bug；历史全局目标未确认，seed44实际TF32梯度误差超预定容差，因此CE候选未运行。local probe的ΔBA+.000476、ΔAUROC+.000166，CI跨0；不能确认Gyro独有增量。

双参考relative AUC .735639，相对common PD-only +.067150（13/15，CI[.042404,.088420]，primary q=.000305）、DD-only +.236204（15/15，CI[.207170,.265064]，q=.000122）；但低于WSSL .759553。一次OOF两分数融合BA .627638，Δ-.092006（0/15胜，CI[-.110506,-.073277]，q=.000122）；DD recall降至.327707，拒绝。互补筛查不等于可部署融合收益。

正温度诊断NLL Δ-.039178（12/15改善，CI[-.064443,-.016595]，q=.016113），Brier/ECE的CI跨0；仅按预登记均值规则保留诊断sidecar，不改变AUROC/.5分类或当前原生模型。完整六指标、mean/median delta、wins/ties/losses、SD、p/q、effect、配对/subject cluster CI及5context敏感性见STATISTICAL_APPENDIX.md。距离无合法概率映射，校准指标NA。

已确认事实与机制推测区分：数值路径/资产/配对结果是本轮证据；“batch reduction导致DD错误”“融合失败由OOF尺度/先验迁移导致”均未证实；Gyro独有增量、cycle结构、论文详细subject协议/JCR分区仍待核验。15split重复开发不能替代独立验证，也不能用不相同CI跨0状态证明方法间显著不同。

服务器产物：/home/zyt/deep_final/artifacts/wssl_reference_audit_20261111。主线代码/mainline、数据/data与历史模型/artifacts均继续位于/home/zyt/deep_final。私有features/predictions/checkpoints/restricted_manifests仅服务器；公开代码/配置/报告/聚合表按既有工具发布。环境/资源见analysis/runtime_environment.json：RTX5090D v2，threads4，正式依赖未改，无依赖安装；推理65.3s、参考25.5s、校准融合16.6s分别为核心进程计时，不是GPU基准，峰值显存未测。

复现命令和执行顺序见REPRODUCE.md。停止点：完成最终hash核验和选择性GitHub提交后结束本轮；不后台扩展搜索，不重训baseline、OOF、adapter、Patch或full HarNet。完整性结论允许“保护资产hash未变”，不允许“所有合规测试通过”。
'''+ '\n\n'+LIMIT)
    guard();print('REPORTS_COMPILED',flush=True)

if __name__=='__main__':main()
