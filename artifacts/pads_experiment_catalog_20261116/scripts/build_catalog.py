"""Server-only consolidation of existing development results; no fitting/inference."""
from pathlib import Path
from collections import defaultdict, Counter
import csv, datetime, gzip, hashlib, json, math, re, statistics, sys

ROOT = Path('/home/zyt/deep_final')
HERE = ROOT / 'artifacts/pads_experiment_catalog_20261116'
OUT = ROOT / 'conclusions/EXPERIMENT_RESULTS_ALL_20261116.csv'
BASE = '7c6573aeed86dcec792b3a086618ee63653eb9cb'
METRICS = ['Accuracy','AUROC','BA','PD_Recall','DD_Recall','Macro_F1']
ALIASES = {
 'Accuracy':['accuracy','acc'], 'AUROC':['auroc','macro_auroc'],
 'BA':['ba','balanced_accuracy'], 'PD_Recall':['pd_recall'],
 'DD_Recall':['dd_recall'], 'Macro_F1':['macro_f1']}
ROWS, SOURCES, CHECKS = [], {}, []

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def source(rel):
 p=ROOT/rel
 assert p.is_file() and not p.is_symlink(),rel
 assert 'foe01_final_locked_outer' not in str(p) and 'final_nested_cv' not in str(p)
 SOURCES[rel]=dict(path=rel,sha256=sha(p),bytes=p.stat().st_size)
 return p

def readcsv(rel):
 with source(rel).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def readjson(rel):return json.loads(source(rel).read_text())

def number(x):
 try:
  v=float(x)
  return v if math.isfinite(v) else None
 except (ValueError,TypeError):return None

def mean(v):return statistics.mean(v) if v else None
def sd(v):return statistics.stdev(v) if len(v)>1 else None
def value(row,aliases):
 for k in aliases:
  v=number(row.get(k))
  if v is not None:return v
 return None

def canonical(row, mapping=None):
 return {m:value(row,(mapping or ALIASES).get(m,[])) for m in METRICS}

def splitid(row):
 for a,b in [('context','inner'),('outer','inner'),('outer_context','inner_fold'),('outer_fold','inner_fold')]:
  if a in row and b in row:return str(row[a])+'/'+str(row[b])
 if 'fold_id' in row:return row['fold_id']
 raise ValueError('No split identity')

def emit(stage,method,records,purpose,problem,result,status='DESCRIPTIVE',role='development',
         files=(),report='',notes='',data='historical_as_recorded_not_Certified_L1',selection=''):
 assert records or status.startswith(('NOT_RUN','NON_PERFORMANCE','BLOCKED'))
 if report:source(report)
 # Every statistical row represents exactly one existing seed/split combination.
 keys=[(x['split'],x.get('seed')) for x in records]
 assert len(keys)==len(set(keys)),(stage,method,'duplicate seed/split')
 seeds=sorted({str(x['seed']) for x in records if x.get('seed') not in [None,'','-1','0']})
 splits=sorted({x['split'] for x in records})
 row=dict(阶段=stage,方法=method,实验角色=role,数据版本=data,采用目的=purpose,
          要解决的问题=problem,达到的结果=result,历史判定=status,
          development_split数=len(splits),seed数=len(seeds),已有统计记录数=len(records),
          seed列表=';'.join(seeds) or 'NA',指标单位='比例[0,1]',
          SD定义='NA',指标来源筛选=selection,
          缺失值与限制=notes,来源文件=';'.join(files),来源报告=report or 'NA',
          来源提交=BASE,来源链接=('https://github.com/chouytong/pureDeep/blob/'+BASE+'/'+(report or (files[0] if files else 'conclusions/MODEL_INDEX.md'))))
 missing=[]
 for m in METRICS:
  bysplit=defaultdict(list);byseed=defaultdict(list)
  for x in records:
   v=x.get(m)
   if v is None:continue
   assert -1e-12 <= v <= 1+1e-12,(stage,method,m,v)
   bysplit[x['split']].append(v)
   if x.get('seed') not in [None,'','-1','0']:byseed[str(x['seed'])].append(v)
  sv=[mean(v) for v in bysplit.values()]; ev=[mean(v) for v in byseed.values()]
  if len(sv)>1:
   primary=sd(sv);row['SD定义']='同一split先平均seed指标，再跨development splits计算样本SD(ddof=1)'
  elif len(ev)>1:
   primary=sd(ev);row['SD定义']='仅一个development split，跨seed计算样本SD(ddof=1)；不代表跨split波动'
  else:primary=None
  row[m+'_mean']=mean(sv);row[m+'_SD']=primary;row[m+'_seed_SD']=sd(ev)
  if not sv:missing.append(m)
 if missing:row['缺失值与限制']+='；源结果未保存/不适用：'+','.join(missing)+'；对应均值和SD均NA'
 if len(seeds)<2 and records:row['缺失值与限制']+='；不能估计跨seed SD，seed_SD=NA'
 if len(splits)==1:row['缺失值与限制']+='；单split pilot，不能替代15-split实验'
 ROWS.append(row)
 return row

def table(stage,filename,group,purpose,problem,result,report='',status='REJECT',role='development',
          mapping=None,skip=(),data='historical_as_recorded_not_Certified_L1',notes='',filterfn=None):
 rel='artifacts/'+stage+'/'+filename; groups=defaultdict(list)
 for raw in readcsv(rel):
  if filterfn and not filterfn(raw):continue
  key=' | '.join(k+'='+raw.get(k,'') for k in group) if group else Path(filename).stem
  if any(raw.get(k) in skip for k in group):continue
  z=canonical(raw,mapping);z.update(split=splitid(raw),seed=raw.get('seed'))
  groups[key].append(z)
 for key,items in sorted(groups.items()):
  decision=status(key) if callable(status) else status
  if key in ['variant=baseline','condition=baseline','variant=str','condition=F0','condition=P0']:
   decision='MATCHED_REFERENCE_REUSED'
  if key=='variant=frozen':decision='RETAIN_CURRENT_REFERENCE'
  if key=='variant=lastblock':decision='PREVIOUSLY_REJECTED_REFERENCE'
  if key=='condition=WSSL-CERTIFIED-L1' and 'pilot' in stage:decision='MATCHED_SINGLE_SPLIT_REFERENCE'
  if key=='condition=AI-WSSL-CERTIFIED-L1':decision='EXPERIMENTAL_CANDIDATE_PILOT_ONLY'
  label=key
  rowrole=role
  if 'ensemble' in filename:
   label='evaluated_probability_ensemble | '+key
   rowrole='inference_ensemble'
  details={
   'lr_1e4':'降低学习率','wd_1e3':'增强weight decay','smooth_005':'标签平滑0.05',
   'ce_unweighted':'去除CE类别权重','ce_sqrtweighted':'平方根逆频率CE',
   'focal_balanced':'类别平衡focal loss','ce_effective099':'effective-number CE',
   'balanced_sampler':'类别平衡采样加原平衡CE','jitter_equal':'两类等策略微扰',
   'jitter_dd_only':'仅DD微扰','scaling_equal':'两类幅值缩放','scaling_dd_only':'仅DD幅值缩放',
   'temporal_equal':'两类时间扰动','temporal_dd_only':'仅DD时间扰动',
   'mixup_equal':'两类Mixup','mixup_dd_only':'仅DD Mixup',
   'e1_balanced_unweighted':'平衡采样加无权CE','e1_balanced_balanced':'平衡采样加平衡CE',
   'e3_sam_rho005':'SAM rho0.05','e4_aux_lambda01':'活动辅助loss lambda0.1',
   'e4_aux_lambda03':'活动辅助loss lambda0.3','a_ssl_only_pretrained':'仅冻结外部SSL',
   'b_str_pretrained':'Frozen WSSL-STR','c_str_random':'随机encoder容量控制',
   'adapt':'HarNet末层适配','multi':'中层与末层feature融合','adapter':'瓶颈adapter',
   'gate':'门控wrist融合','domain_masked':'训练集masked域适配',
   'harnet5':'HarNet5 prior','biopm':'BioPM acc residual','biopm_only':'仅BioPM acc',
   'biopm_random':'随机BioPM容量控制','ema':'指数移动平均参数','activity_scaling':'活动特征缩放',
   'activity_interaction':'AI-WSSL-STR（旧L1 ActivityInteraction）',
   'subject_supcon':'受试者监督对比loss','linear_svm':'线性SVM读出','rbf_svm':'RBF SVM读出',
   'logistic_regression':'Logistic Regression读出','random_forest':'随机森林读出',
   'probability_ensemble':'三seed概率ensemble','logit_ensemble':'三seed原logits ensemble'}
  rawname=key.split('=',1)[-1]
  specific=details.get(rawname)
  rowpurpose=purpose+('；本条件：'+specific if specific else '')
  rowresult=result
  if decision in ['MATCHED_REFERENCE_REUSED','RETAIN_CURRENT_REFERENCE','MATCHED_SINGLE_SPLIT_REFERENCE']:
   rowresult='本阶段的既有匹配参考；复用/复现不是新的候选性能晋升。'+result
  emit(stage,label,items,rowpurpose,problem,rowresult,decision,rowrole,[rel],report,notes,data,
       selection=key)

def summaries(stage,method,paths,purpose,problem,result,report='',status='REJECT',notes=''):
 records=[]
 for j,rel in enumerate(paths):
  d=readjson(rel)
  assert d['status']=='complete' and len(d['fold_summaries'])==15,rel
  for flag in ['outer_test_loader_created','outer_test_signal_accessed','outer_test_predictions_accessed']:
   assert d.get(flag) is False,(rel,flag)
  match=re.search(r'seed(42|43|44)',rel);seed=match.group(1) if match else '42'
  for fold in d['fold_summaries']:
   z=canonical(fold['validation_metrics']);z.update(split=fold['fold_id'],seed=seed)
   records.append(z)
 return emit(stage,method,records,purpose,problem,result,status,'development',paths,report,notes)

def baseprotection():
 sys.path.insert(0,str(ROOT/'artifacts/pads_maintenance_audit_20261116/scripts'))
 from audit import protected
 return protected()

def main():
 assert Path(sys.executable).resolve()==Path('/home/zyt/envs/mfam_analysis/bin/python').resolve()
 assert not OUT.exists(),'Preserve existing CSV'
 before=baseprotection()
 manifest=json.loads(Path('/home/zyt/pureDeep-github-upload/PUBLICATION_MANIFEST.json').read_text())
 public={x['path'] for x in manifest['files']}
 # Early screening: retain all actually recorded configurations, without counting
 # the two independent seed-stability summaries as additional experiments.
 early=defaultdict(list)
 for rel in sorted(public):
  if rel.startswith('artifacts/development/') and rel.endswith('/development_summary.json'):
   name=Path(rel).parent.name
   name=re.sub(r'_seed(?:43|44)','',name)
   early[name].append(rel)
 for name,paths in sorted(early.items()):
  summaries('early_development_20260915_16',name,paths,
            '比较小样本纯深度手腕时序表示和活动聚合的初期可行方案',
            '确定基础模型方向：容量、传感器分支、频谱、统计矩和聚合是否有用',
            '初期开发筛选记录；后续经GroupNorm、STR和WSSL阶段演进，不能作为当前模型的独立外部证据',
            status='HISTORICAL_SCREENING',notes='部分候选仅seed42；历史预处理和配置各异，不与Certified L1混算')
 # Reuse exact source links recorded in original three-seed comparison JSONs.
 json_specs=[
 ('foundation_validation','NR-01 / V8-GN','foundation_validation/outputs/foundation_validation/groupnorm_nr01_three_seed_analysis.json','归一化的batch组成敏感性','用GroupNorm替代BatchNorm，保持容量和训练配方','开发BA提高，历史保留为后续V8-GN基座','artifacts/foundation_validation/nr01_groupnorm_backbone/README.md','HISTORICALLY_RETAINED'),
 ('foundation_validation','LR-01 long-range TCN','foundation_validation/outputs/foundation_validation/longrange_lr01_three_seed_analysis.json','局部感受野可能不足','增大TCN dilation以覆盖长活动动态','关键判别指标下降；不保留','artifacts/foundation_validation/nr01_groupnorm_backbone/README.md','REJECT'),
 ('foundation_validation','AGG-01 masked mean','foundation_validation/outputs/foundation_validation/masked_mean_agg01_three_seed_analysis.json','活动attention在小样本下可能不稳定','用无参数均值聚合检验稳定性','改变PD/DD召回取舍；保留消融证据，不替换模型','artifacts/foundation_validation/nr01_groupnorm_backbone/README.md','ABLATION_ONLY'),
 ('foundation_validation','AGG-02 mean + attention residual','foundation_validation/outputs/foundation_validation/mean_attention_residual_agg02_three_seed_analysis.json','平均与attention的稳定性/选择性取舍','以mean锚定并加入门控attention残差','波动变小未转化为可靠判别增益','artifacts/foundation_validation/nr01_groupnorm_backbone/README.md','REJECT'),
 ('training_recipe','TR-FINAL LR1e-4 three-seed confirmation','artifacts/training_recipe/v8_gn_final_recipe/analysis/tr_final_three_seed_analysis.json','单seed低学习率改善是否可靠','三seed确认学习率1e-4','未达保留门槛，DD召回和种子稳定性有代价，恢复原配方','artifacts/training_recipe/v8_gn_final_recipe/README.md','REJECT'),
 ('mean_only_fusion','MF-01 bilateral mean-only','artifacts/ablation/mf01_mean_only_20260920/results/mf01_comparison_20260920.json','双腕full融合是否需要左右身份与差异信息','仅保留双腕masked mean作为消融','BA、AUROC和类别召回下降，保留full双腕融合','artifacts/ablation/mf01_mean_only_20260920/MF01_FINAL_REPORT.md','REJECT')]
 for stage,method,rel,problem,purpose,result,report,status in json_specs:
  d=readjson(rel);paths=[]
  for item in d['per_seed']:
   p=Path(item['run'])/'development_summary.json'
   assert p.is_relative_to(ROOT)
   paths.append(str(p.relative_to(ROOT)))
  summaries(stage,method,paths,purpose,problem,result,report,status)
 for category,report,purpose,problem,result in [
  ('training_recipe','artifacts/training_recipe/v8_gn_final_recipe/README.md','固定架构下小范围确认LR/batch/WD/dropout/scheduler','原训练配方是否造成性能瓶颈','单seed方向筛选；未形成可替代原配方的最终多seed方案'),
  ('invariant_representation','artifacts/invariant_representation/pa_stage3_20260919/PROTOTYPE_ALIGNMENT_REPORT.md','训练集活动条件疾病prototype alignment','减少subject变化并保存疾病判别信息','两档预先固定的alignment权重未替换原基座'),
  ('strong_backbone_benchmark','artifacts/strong_backbone_benchmark_20260921/STRONG_BACKBONE_BENCHMARK_REPORT.md','比较成熟CNN容量和唯一Gyro FFT分支','是否存在主干表达能力或频域信息利用不足','ResNet退化、Inception无稳定优势、FFT分支退化；保持V8-GN'),
  ('structured_token_residual','artifacts/str01_structured_token_residual_20260921/STR01_REPORT.md','增加保留活动顺序的零初始化结构残差读出','活动聚合是否损失疾病判别信息','STR-01历史保留；现为matched baseline'),
  ('wrist_structured_residual','artifacts/wsr01_wrist_structured_residual_20260922/WSR01_REPORT.md','加入活动×腕侧有序残差读出','双腕融合前的信息是否被进一步损失','AUROC微升但联合门槛失败，不替换STR-01')]:
  groups=defaultdict(list)
  for rel in sorted(public):
   if rel.startswith('foundation_validation/outputs/'+category+'/') and rel.endswith('/development_summary.json'):
    name=Path(rel).parent.name
    if category=='training_recipe' and name.startswith('tr_final'):continue
    name=re.sub(r'_seed(?:42|43|44)_\d{8}(?:_r\d+)?$','',name)
    groups[name].append(rel)
  for method,paths in groups.items():
   status='MATCHED_BASELINE' if category=='structured_token_residual' else 'REJECT'
   summaries(category,method,paths,purpose,problem,result,report,status)
 # Named stages: raw per-seed/per-split metrics are always preferred to rounded reports.
 specs=[
 ('phase2_str_training_20260926','analysis/variant_seed_split_metrics.csv',['variant'],'验证STR训练、损失与增强','学习率/正则、类别不平衡及DD数据不足是否限制STR','17配置均完成；所有16候选未通过联合闸门','PHASE2_REPORT.md','REJECT'),
 ('phase3a_puredeep_20260928','analysis/phase3_seed_split_metrics.csv',['variant'],'分离采样与loss补偿并测试SAM/活动辅助监督','重复类别补偿、训练泛化差距及活动证据不足','未替换STR；SAM等候选存在指标/召回取舍','PHASE3A_REPORT.md','REJECT'),
 ('phase3b_external_ssl_20260928','analysis/phase3b_seed_split_metrics.csv',['variant'],'引入冻结的外部可穿戴SSL先验及随机容量对照','从小样本PADS单独学习手腕表示是否受限','预训练冻结HarNet+STR形成开发增量；保留Frozen WSSL-STR','PHASE3B_REPORT.md','MIXED_SEE_METHOD'),
 ('phase3c_ssl_adaptation_20260930','analysis/model_seed_split_metrics.csv',['variant'],'测试HarNet末层适配及中末层融合','冻结prior是否存在任务适配不足','末层适配与中末层融合均未替换Frozen WSSL','PHASE3C_REPORT.md','REJECT'),
 ('phase3c_ssl_adaptation_20260930','analysis/e2_single_seed_split.csv',['rule'],'比较既有固定阈值与train-only OOF阈值','判别排序增益是否因操作点选择未转为召回','OOF阈值不保留；只整理历史已执行阈值，不重搜索','PHASE3C_REPORT.md','INFERENCE_REFERENCE'),
 ('phase3c_ssl_adaptation_20260930','analysis/e2_ensemble_15split.csv',['rule'],'检查三seed概率ensemble与OOF阈值','单模型与ensemble操作点能否兼顾PD/DD','ensemble是独立推理估计量，不替代单模型','PHASE3C_REPORT.md','INFERENCE_REFERENCE'),
 ('phase3c_ssl_adaptation_20260930','analysis/e4_seed_split_metrics.csv',['variant','fraction'],'固定25/50/75/100%训练人数学习曲线','冻结SSL增量是否随训练受试者数量变化','100%端点重用历史结果；学习曲线是开发证据','PHASE3C_REPORT.md','LEARNING_CURVE'),
 ('phase3d_ssl_adaptation_20261001','analysis/model_seed_split_metrics.csv',['variant'],'测试adapter、gate和train-only masked域适配','冻结外部prior与PADS间域差异是否可被轻量适配改善','adapter/gate/域适配不保留；activity gate条件未满足未运行','PHASE3D_REPORT.md','REJECT'),
 ('phase3e_external_priors_20261001','analysis/model_seed_split_metrics.csv',['variant'],'比较HarNet5与BioPM acc-only外部prior及随机对照','HarNet10之外的时长和表征先验是否提供增量','没有新single-model候选获保留；不能单独归因于时间尺度','PHASE3E_REPORT.md','REJECT'),
 ('phase3e_external_priors_20261001','analysis/e0_seed_metrics.csv',['model'],'固定等权seed/模型ensemble','冻结与适配模型是否有可用互补信息','六模型ensemble仅推理AUROC参考，BA/DD代价仍在','PHASE3E_REPORT.md','INFERENCE_REFERENCE'),
 ('wssl_review_20261004','analysis/lr_seed_split_metrics.csv',['variant'],'固定classifier学习率对照','WSSL的训练/验证loss差距是否由学习率造成','较低学习率未通过整体门槛，保留原recipe','FINAL_REPORT.md','REJECT'),
 ('wssl_ema_20261002','analysis/ema_seed_split_metrics.csv',['variant'],'测试参数EMA','验证波动是否可由权重平滑缓解','EMA未替换原Frozen WSSL','FINAL_REPORT.md','REJECT'),
 ('wssl_activity_scaling_20261008','analysis/scaling_seed_split_metrics.csv',['variant'],'学习活动维度缩放','11活动贡献不均是否限制固定WSSL表示','结果INCONCLUSIVE，不保留，不追加搜索','FINAL_REPORT.md','INCONCLUSIVE'),
 ('frequency_prior_str_20261007','analysis/seed_split_metrics.csv',['condition'],'固定五频带残差与近等容量残差对照','频段先验能否提供超出STR和额外容量的判别增量','F2未形成稳定联合增量，REJECT','FINAL_REPORT.md','REJECT'),
 ('patch_str_20261008','analysis/seed_split_metrics.csv',['condition'],'固定200/100局部patch及顺序建模','局部动态和顺序能否弥补整段时序表示不足','P1/P2均未建立稳定BA增量；P3未运行','PATCH_STR_REPORT.md','REJECT'),
 ('wssl_reference_audit_20261111','analysis/baseline_metrics.csv',[],'回读复现Frozen WSSL基线','历史模型与当前数值路径是否一致','45次既有模型重推理匹配，原模型保留','FINAL_REPORT.md','RETAIN_CURRENT'),
 ('wssl_reference_audit_20261111','analysis/A1_adapted_metrics.csv',[],'复核历史末层adapted模型','轻量适配的既有表现和来源是否可靠','复用既有45结果，不作为新训练','FINAL_REPORT.md','HISTORICAL_REUSE'),
 ('wssl_reference_audit_20261111','analysis/A1_probe_metrics.csv',['condition'],'score SSL控制与加入local表示probe','Gyro/local表示是否提供独有增量','local增量CI跨零，不能确认稳定独有增益','FINAL_REPORT.md','DIAGNOSTIC_ONLY'),
 ('wssl_reference_audit_20261111','analysis/B_metrics.csv',['condition'],'PD/DD双参考与相对距离评分','仅单一参考是否限制疾病边界的表达','相对评分高于单参考但低于WSSL；受历史outer文字接触偏差限制','FINAL_REPORT.md','EXPLORATORY_WITH_BOUNDARY_DEVIATION'),
 ('wssl_reference_audit_20261111','analysis/B_fusion_metrics.csv',[],'OOF两分数融合','互补筛查能否转为可部署融合收益','BA/F1/DD下降，融合REJECT；保留历史边界偏差披露','FINAL_REPORT.md','REJECT'),
 ('wssl_guided_optimization_phase8_20261111','analysis/all_model_metrics.csv',['condition'],'传统读出、subject SupCon、ActivityInteraction和固定ensemble','冻结WSSL读出及跨活动信息能否形成稳定新增量','全部新增候选未获保留；AI旧L1存在PD/DD召回取舍','reports/FINAL_REPORT.md','REJECT'),
 ('wssl_guided_optimization_phase8_20261111','analysis/activity_interaction_ablation.csv',['condition'],'训练后关闭attention的单次推理消融','交互attention在现有候选中的贡献','推理干预证据，不是attention-off重新训练对照','reports/ACTIVITY_INTERACTION_REPORT.md','ABLATION_ONLY'),
 ('l1_precision_rebuild_phase9_level2_20261112','analysis/metrics_by_split_seed.csv',['condition'],'独立数值正确的L1重建并成对重训STR/WSSL','旧L1批次耦合与数值来源问题','证书/独立性PASS；未证明相对历史WSSL提升；新数据WSSL相对STR有开发增量','reports/FINAL_REPORT.md','CERTIFIED_COMPARISON'),
 ('activity_interaction_certified_pilot3_20261112','analysis/metrics_by_seed.csv',['condition'],'仅在Certified L1单split三seed复核AI交互','旧L1上的ActivityInteraction能否迁移到Certified数据','三个seed的DD Recall均下降超过0.01；pilot完成，不扩大，不晋升','reports/FINAL_REPORT.md','PILOT_ONLY')]
 for stage,filename,group,purpose,problem,result,report,status in specs:
  if stage=='phase3b_external_ssl_20260928':
   status=lambda k:'RETAIN_CURRENT' if 'b_str_pretrained' in k else ('MATCHED_BASELINE' if 'baseline' in k else 'CONTROL_NOT_RETAINED')
  if 'certified_pilot' in stage:data='pads_l1_lambda50_certified_v1'
  else:data='historical_as_recorded_not_Certified_L1'
  start=len(ROWS)
  table(stage,filename,group,purpose,problem,result,'artifacts/'+stage+'/'+report,status,data=data,
        notes=('此阶段披露历史outer文字接触及整表I/O边界偏差；不当作独立验证' if stage=='wssl_reference_audit_20261111' else '重复development；同人和训练池跨split重叠；非独立外部验证'))
  if stage=='l1_precision_rebuild_phase9_level2_20261112':
   for row in ROWS[start:]:row['数据版本']='pads_l1_lambda50_certified_v1' if 'CERTIFIED' in row['方法'] else 'legacy_L1'
 # Phase4 four single-factor studies: avoid counting their repeated baseline four times.
 for name,desc in [('baseline','原Frozen WSSL基线'),('a1_mean','多窗mean SSL特征'),('a2_delta','窗口间delta动态'),('b1_raw','raw时序残差'),('c1_dd_aux','DD辅助监督')]:
  filename='analysis/'+name+'_seed_split_metrics.csv'
  table('phase4_window_dynamics_20261002',filename,['variant'],desc,
        '固定WSSL是否遗漏活动内动态或DD可分信息','四项候选均未通过闸门，保持Frozen WSSL',
        'artifacts/phase4_window_dynamics_20261002/FINAL_REPORT.md',
        'RETAIN_CURRENT' if name=='baseline' else 'REJECT',skip=() if name=='baseline' else (('baseline','a1_mean') if name=='a2_delta' else ('baseline',)))
 # Recorded ensemble results are evaluated probabilities, not averages of metrics.
 table('phase3a_puredeep_20260928','analysis/e2_ensemble_split_metrics.csv',[],
       '三seed概率ensemble','模型seed之间是否存在推理互补',
       '排序提高但DD召回降低；非默认单模型',
       'artifacts/phase3a_puredeep_20260928/PHASE3A_REPORT.md','INFERENCE_REFERENCE')
 table('phase3b_external_ssl_20260928','analysis/phase3b_ensemble_split_metrics.csv',['variant'],
       '冻结各分支的三seed概率ensemble','推理方差与单模型的区别',
       '保留为ensemble参考，不与单模型指标平均混算',
       'artifacts/phase3b_external_ssl_20260928/PHASE3B_REPORT.md','INFERENCE_REFERENCE')
 # H1 diagnostic and family decomposition, without reading patient-level records.
 table('h1_deep_gap_diagnosis_20260921','feature_family_decomposition/fold_metrics.csv',['variant'],
       '固定H1统计特征族的留一/单族/时域频域分解','深度表示与传统特征的疾病信息差距来自哪些分量',
       '提供诊断参考；不把H1输入到当前纯深度主线',
       'artifacts/h1_deep_gap_diagnosis_20260921/H1_DEEP_REPRESENTATION_GAP_DIAGNOSIS_REPORT.md','DIAGNOSTIC_ONLY')
 table('str01_gain_mechanism_diagnosis_20260923','analysis/head_metrics_45_seed_folds.csv',['head'],
       '比较V8/STR原头、残差头和最终头','STR增量由哪条既有读出路径贡献',
       '表示/读出诊断，不是重新训练的新候选',
       'artifacts/str01_gain_mechanism_diagnosis_20260923/STR01_GAIN_MECHANISM_DIAGNOSIS_REPORT.md','DIAGNOSTIC_ONLY')
 # WSSL contribution interventions: use masked metrics (not repeated full metrics).
 mm={m:[ALIASES[m][0]+'_masked'] for m in METRICS}
 table('wssl_contribution_20261007','analysis/metrics_45runs.csv',['condition'],
       '活动/腕侧/全部SSL分支的推理mask干预','冻结SSL的增量是否集中于特定活动或腕侧',
       '消融仅为已拟合模型条件性证据；遮蔽不是重训，也不等同临床重要性',
       'artifacts/wssl_contribution_20261007/REPRODUCTION_REPORT.md','ABLATION_ONLY',mapping=mm)
 # PAG accuracy is exactly recoverable from saved class recalls and the immutable
 # matched validation supports. Integer-confusion checks prevent assuming weights.
 support_source='foundation_validation/outputs/structured_token_residual/str01_seed42_20260921/development_summary.json'
 supports={f['fold_id'].replace('outer_','').replace('inner_',''):f['validation_metrics']['support']
           for f in readjson(support_source)['fold_summaries']}
 pag='artifacts/pag01_preaggregation_disease_info_20260926/analysis/disease_probe_45.csv'
 groups=defaultdict(list)
 for raw in readcsv(pag):
  z=canonical(raw);sid=splitid(raw);n0,n1=supports[sid]
  tp0=z['PD_Recall']*n0;tp1=z['DD_Recall']*n1
  assert abs(tp0-round(tp0))<1e-8 and abs(tp1-round(tp1))<1e-8
  z['Accuracy']=(round(tp0)+round(tp1))/(n0+n1);z.update(split=sid,seed=raw['seed'])
  groups[(raw['stage'],raw['pca_k'])].append(z)
 for (stage,k),records in sorted(groups.items()):
  emit('pag01_preaggregation_disease_info_20260926',stage+' | PCA='+k,records,
       '控制最终分数的活动前聚合疾病probe','最终score之外是否有可读疾病信息、是否在聚合时衰减',
       'raw activity存在增量排名信息，但训练闸门未全部通过；不启动新深度训练',
       'DIAGNOSTIC_ONLY','diagnostic',[pag,support_source],
       'artifacts/pag01_preaggregation_disease_info_20260926/PAG01_REPORT.md',
       'Accuracy由保存的PD/DD recall×同split支持人数恢复，已验证整数正确数；未读取个体表')
 # RGD stores only ranking/BA; do not invent unavailable threshold confusion counts.
 table('rgd01_h1_str_residual_ranking_20260926','analysis/fusion_45_seed_splits.csv',['model'],
       'H1/STR分数的固定轻量融合诊断','两类既有分数是否含互补排序信息',
       '融合AUROC存在开发增量，但仅为手工特征诊断，不替代纯深度模型',
       'artifacts/rgd01_h1_str_residual_ranking_20260926/RGD01_REPORT.md','DIAGNOSTIC_ONLY')
 # DSG native/PCA probes deliberately retain only the metrics they actually saved.
 for probe in ['native','pca']:
  mapping={'BA':[probe+'_probe_ba'],'AUROC':[probe+'_probe_auroc']}
  table('dsg01_cross_subject_disease_subspace_20260926','analysis/native_probes.csv',['model','stage'],
        probe+'冻结表示probe','疾病子空间跨seed/split是否稳定且能迁移',
        '诊断性表示probe；不能由稳定性相关推断因果机制',
        'artifacts/dsg01_cross_subject_disease_subspace_20260926/DSG01_REPORT.md',
        'DIAGNOSTIC_ONLY',mapping=mapping,notes=probe+' probe；原产物仅保存BA/AUROC')
  for row in ROWS[-8:]:
   # Distinguish native vs PCA records even if stage counts change.
   if row['阶段']=='dsg01_cross_subject_disease_subspace_20260926' and row['采用目的'].startswith(probe):row['方法']=probe+' | '+row['方法']
 # PRR saved private validation scores: derive all six metrics without any fits.
 # Patient IDs/status columns are never retained, printed or exported.
 from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score,f1_score,recall_score
 rel='artifacts/prr01_frozen_str_pair_rescue_20260926/analysis/subject_level_records.csv.gz'
 scores=defaultdict(lambda:([],[]))
 with gzip.open(source(rel),'rt') as f:
  for raw in csv.DictReader(f):
   y,s=scores[(raw['stage'],raw['outer']+'/'+raw['inner'],raw['seed'])]
   y.append(int(raw['label']));s.append(float(raw['corrected_score']))
 groups=defaultdict(list);reference=readcsv('artifacts/prr01_frozen_str_pair_rescue_20260926/analysis/method_metrics_45.csv')
 lookup={(x['stage'],splitid(x),x['seed']):x for x in reference}
 for key,(y,score) in scores.items():
  pred=[int(s>=0) for s in score]
  z=dict(Accuracy=accuracy_score(y,pred),AUROC=roc_auc_score(y,score),BA=balanced_accuracy_score(y,pred),
         PD_Recall=recall_score(y,pred,pos_label=0),DD_Recall=recall_score(y,pred,pos_label=1),
         Macro_F1=f1_score(y,pred,average='macro'),split=key[1],seed=key[2])
  ref=lookup[key]
  assert abs(z['AUROC']-float(ref['corrected_auroc']))<1e-12
  assert abs(z['BA']-float(ref['corrected_ba']))<1e-12
  groups[key[0]].append(z)
 for method,records in sorted(groups.items()):
  emit('prr01_frozen_str_pair_rescue_20260926',method,records,
       '冻结STR各表示的pair-rescue修正诊断','能否恢复H1能排对而STR排错的PD-DD配对且不破坏原正确配对',
       'activity raw存在总体排名信息但target-specific净救回不稳定；不启动新训练',
       'DIAGNOSTIC_ONLY','diagnostic',['artifacts/prr01_frozen_str_pair_rescue_20260926/analysis/method_metrics_45.csv'],
       'artifacts/prr01_frozen_str_pair_rescue_20260926/PRR01_REPORT.md',
       '六指标仅在服务器由既有corrected_score>=0恢复；AUROC/BA逐run与原表核对；不输出IDs/个体score')
 # Explicitly record non-executed and non-performance work to avoid presenting
 # registration, data audits or blocked external validation as performance runs.
 extra=[
  ('wssl_guided_optimization_phase8_20261111','XGBoost','NOT_RUN_DEPENDENCY_MISSING','传统读出对照','检查另一传统读出是否提供增量','依赖缺失，未运行，不填虚构分数'),
  ('phase3d_ssl_adaptation_20261001','activity-conditioned gate','NOT_RUN_GATE_FAIL','条件式活动门控','只在前置gate成功时追加','前置gate失败，未运行'),
  ('patch_str_20261008','P3 / WSSL+Patch','NOT_RUN_GATE_FAIL','后续组合候选','仅在P1/P2满足预定闸门时扩展','前置条件失败，未运行'),
  ('wssl_reference_audit_20261111','fixed-denominator CE','NOT_RUN_GATE_FAIL','审计weighted CE风险目标','历史目标与数值精度是否允许改CE','历史目标未确认且实际TF32梯度误差超限，未运行'),
  ('independent_validation_phase7_20261111','independent PD/DD validation','BLOCKED_NO_ELIGIBLE_COHORT','验证外部可迁移性','当前缺独立PD/DD队列和完整reference状态','已调查20数据源并完成合成adapter测试，无真实外部性能'),
  ('pads_raw_preprocessing_audit_20261112','raw/preprocessing audit','NON_PERFORMANCE_AUDIT','审查原始数据、时间戳、同步和L1数值来源','数据处理及归一化证据是否可信','原审计与限制见报告；后续Certified重建单独列行'),
  ('wssl_representation_generalization_phase6_20261111','representation/coverage diagnosis','NON_PERFORMANCE_DIAGNOSTIC','表示漂移与类条件覆盖分析','适配变化与不同临床类别覆盖是否关联','描述性机制证据，不产生新模型分类分数或独立验证'),
  ('wssl_complementarity_phase5_20261111','existing-model complementarity reanalysis','NON_PERFORMANCE_REANALYSIS','重分析既有Frozen/adapted/reference互补性','个别受试者改善是否意味着可部署净收益','重复既有结果；未授权新深度模型，机制及外部验证仍缺证据'),
  ('ai_wssl_str_registration_20261116','AI-WSSL-STR registration','NON_PERFORMANCE_REGISTRATION','独立工程登记、加载与保存接口','避免模型名称、checkpoint来源与生命周期混淆','3个Certified加载/前向/恢复逐位一致；0训练；登记不等于晋升'),
  ('patch_str_recheck_20261111','Patch archived-result recheck','NON_PERFORMANCE_RECHECK','回读历史负结果','核对现有Patch归档是否一致','与原归档一致；未重复正式训练'),
  ('pads_maintenance_audit_20261116','maintenance state verification','NON_PERFORMANCE_AUDIT','核验公开快照、保护资产和发布状态','服务器/GitHub/镜像是否一致','25,467公开文件最终核对，0训练，当前模型指针未改变')]
 for stage,method,status,purpose,problem,result in extra:
  reports=[p for p in public if p.startswith('artifacts/'+stage+'/') and Path(p).name in ['FINAL_REPORT.md','RECHECK_REPORT.md','MODEL_REGISTRATION_REPORT.md','MAINTENANCE_REPORT.md']]
  emit(stage,method,[],purpose,problem,result,status,'non_performance_or_not_run',
       report=sorted(reports)[0] if reports else '',notes='未运行/非分类性能阶段；所有分类均值、SD均NA，不把测试PASS当性能')
 # Distinct identifiers and meaningful algebra/independent arithmetic checks.
 ids=[(r['阶段'],r['方法']) for r in ROWS]
 assert len(ids)==len(set(ids)),[x for x,c in Counter(ids).items() if c>1]
 for row in ROWS:
  if all(row[k] is not None for k in ['BA_mean','PD_Recall_mean','DD_Recall_mean']):
   assert abs(row['BA_mean']-(row['PD_Recall_mean']+row['DD_Recall_mean'])/2)<1e-10
  assert row['采用目的'] and row['要解决的问题'] and row['达到的结果']
  for m in METRICS:
   if row[m+'_mean'] is not None:assert row[m+'_SD'] is not None,(row['阶段'],row['方法'],m)
 # Exact identities against retained model provenance; independent NumPy SD
 # checks are run by verify_catalog.py after this export.
 for stage,method in [('phase3b_external_ssl_20260928','variant=b_str_pretrained'),('activity_interaction_certified_pilot3_20261112','condition=AI-WSSL-CERTIFIED-L1')]:
  row=next(x for x in ROWS if x['阶段']==stage and x['方法']==method)
  CHECKS.append(dict(stage=stage,method=method,accuracy=row['Accuracy_mean'],ba=row['BA_mean'],auroc=row['AUROC_mean']))
 wssl=next(r for r in ROWS if r['阶段']=='phase3b_external_ssl_20260928' and r['方法']=='variant=b_str_pretrained')
 current=readjson('conclusions/CURRENT_MODEL.json')['retained_model']['metrics']
 for k in ['accuracy','ba','auroc','macro_f1','pd_recall','dd_recall']:
  label=next(m for m,a in ALIASES.items() if k in a)
  assert abs(wssl[label+'_mean']-current[k])<1e-12
  CHECKS.append(dict(check='retained_pointer_metric_identity',metric=k,status='PASS'))
 before2=baseprotection();assert before==before2
 columns=['阶段','方法','实验角色','数据版本','采用目的','要解决的问题','达到的结果','历史判定']
 columns += [m+s for m in METRICS for s in ['_mean','_SD','_seed_SD']]
 columns += ['SD定义','development_split数','seed数','seed列表','已有统计记录数','指标单位','缺失值与限制','指标来源筛选','来源文件','来源报告','来源提交','来源链接']
 with OUT.open('x',encoding='utf-8-sig',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=columns,extrasaction='raise');writer.writeheader()
  for row in ROWS:writer.writerow({k:('NA' if v is None else (format(v,'.12g') if isinstance(v,float) else v)) for k,v in row.items()})
 saved=list(csv.DictReader(OUT.open(encoding='utf-8-sig')));assert len(saved)==len(ROWS)
 for rel,record in SOURCES.items():assert sha(ROOT/rel)==record['sha256']
 # Private input manifests only contain file-level hashes. Public summary never
 # serializes the private record path, input IDs, labels or individual predictions.
 (HERE/'restricted_manifests').mkdir(exist_ok=True)
 with (HERE/'restricted_manifests/input_sources.json').open('x') as f:json.dump(list(SOURCES.values()),f,indent=2)
 public_sources=[v for k,v in SOURCES.items() if k in public]
 for name,data in [('source_manifest.json',public_sources),('validation.json',dict(
     status='PASS',execution_host='server',python_executable=sys.executable,source_commit=BASE,
     rows=len(ROWS),stages=len({r['阶段'] for r in ROWS}),columns=len(columns),
     rows_with_all_six_metrics=sum(all(r[m+'_mean'] is not None for m in METRICS) for r in ROWS),
     rows_with_partial_metrics=sum(any(r[m+'_mean'] is not None for m in METRICS) and not all(r[m+'_mean'] is not None for m in METRICS) for r in ROWS),
     nonperformance_or_not_run_rows=sum(all(r[m+'_mean'] is None for m in METRICS) for r in ROWS),
     csv_sha256=sha(OUT),csv_bytes=OUT.stat().st_size,source_files_verified=len(SOURCES),
     immutable_sources_unchanged=True,protected=before,checks=CHECKS,
     new_training_runs=0,new_model_inference_runs=0,new_threshold_fits=0,new_normalization_fits=0,
     cached_PRR_scores_reaggregated_only=True,locked_outer_results_opened=False,
     omitted_scope=['sealed historical outer/final results','non-PADS paper replications','retired MFAM source resynchronization'],
     timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))]:
  with (HERE/'analysis'/name).open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps(load_validation(),ensure_ascii=False,indent=2))

def load_validation():return json.loads((HERE/'analysis/validation.json').read_text())

if __name__=='__main__':main()
