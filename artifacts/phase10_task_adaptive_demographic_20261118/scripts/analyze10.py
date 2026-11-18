"""Frozen pilot/full gates, subgroup checks and stage reports; no rescue fitting."""
from common10 import *
from train10 import KINDS

def group_analysis(kind,frame):
 meta=metadata();devices=load(PRIVATE/'demographic_whitelist_and_device.json')['device'];rows=[]
 for _,r in frame.iterrows():
  seed,c,i=int(r.seed),int(r.context),int(r.inner)
  with np.load(PRIVATE/'runs'/kind/f'seed{seed}_context{c}_inner{i}'/'predictions.npz',allow_pickle=False) as z:
   ids=z['subject_ids'].astype(str);y=z['target'];p=z['probabilities'];logits=z['logits']
  oi,oy,op,oz=certified('wssl',seed,c,i);index={s:j for j,s in enumerate(oi)};order=[index[s] for s in ids]
  assert np.array_equal(y,oy[order]);op=op[order];oz=oz[order]
  groups={}
  for label in ['<60','60-69','>=70']:
   groups['age:'+label]=np.array([(meta[s]['age']<60 if label=='<60' else 60<=meta[s]['age']<70 if label=='60-69' else meta[s]['age']>=70) for s in ids])
  for label in ['female','male']:groups['gender:'+label]=np.array([meta[s]['gender']==label for s in ids])
  for label in sorted(set(devices.values())):groups['device:'+label]=np.array([devices[s]==label for s in ids])
  for label,sel in groups.items():
   n0=int(np.sum(sel&(y==0)));n1=int(np.sum(sel&(y==1)));valid=n0>=5 and n1>=5 and int(sel.sum())>=10
   for condition,prob,z in [(kind,p,logits),('WSSL-CERTIFIED-L1',op,oz)]:
    metric=pred_metrics(y[sel],prob[sel],z[sel]) if valid else {m:np.nan for m in METRICS}
    rows.append(dict(condition=condition,seed=seed,context=c,inner=i,group=label,n=int(sel.sum()),PD_n=n0,DD_n=n1,status='DESCRIPTIVE' if valid else 'NA_SMALL_GROUP',**metric))
 return pd.DataFrame(rows)

def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--candidate',required=True,choices=list(KINDS));ap.add_argument('--scope',required=True,choices=['pilot','full']);a=ap.parse_args()
 kind=a.candidate;phase=KINDS[kind];f=pd.read_csv(HERE/'analysis'/f'{kind}_{a.scope}_metrics.csv')
 expected=3 if a.scope=='pilot' else 45;assert len(f)==expected and not f.duplicated(['seed','context','inner']).any()
 reference=baseline_frame().merge(f[['seed','context','inner']],on=['seed','context','inner'],validate='one_to_one')
 merged=f.merge(reference,on=['seed','context','inner'],suffixes=('_new','_base'),validate='one_to_one')
 delta={m:merged[m+'_new'].to_numpy()-merged[m+'_base'].to_numpy() for m in METRICS}
 valid_receipts=[]
 for _,r in f.iterrows():
  directory=PRIVATE/'runs'/kind/f'seed{int(r.seed)}_context{int(r.context)}_inner{int(r.inner)}';d=load(directory/'complete.json')
  for path,h in d['outputs'].items():assert sha(path)==h
  assert d['frozen_state_unchanged'] and d['optimizer_state_saved'];valid_receipts.append(d)
 checks=dict(BA_positive=float(np.mean(delta['ba']))>0,BA_improved_at_least2of3=int(np.sum(delta['ba']>1e-12))>=2,
             AUROC_non_decrease=float(np.mean(delta['auroc']))>=-1e-12,DD_recall_drop_at_most01=float(np.mean(delta['dd_recall']))>=-.01-1e-12,
             NLL_increase_at_most05=float(np.mean(delta['nll']))<=.05+1e-12,correctness_and_provenance=True)
 if kind in ['DEMO','DD']:checks['PD_recall_drop_at_most01']=float(np.mean(delta['pd_recall']))>=-.01-1e-12
 if kind=='DD':
  checks['DD_recall_positive']=float(np.mean(delta['dd_recall']))>0
  checks['MacroF1_non_decrease']=float(np.mean(delta['macro_f1']))>=-1e-12
 groups=group_analysis(kind,f)
 if kind=='DEMO':
  d0=pd.read_csv(HERE/'analysis/demographic_only_metrics.csv').merge(f[['context','inner']].drop_duplicates(),on=['context','inner'],validate='one_to_one')
  checks['fusion_not_dominated_by_demographics_only']=not(d0.ba.mean()>f.ba.mean()+1e-12 and d0.auroc.mean()>f.auroc.mean()+1e-12)
  gp=groups[groups.condition==kind].merge(groups[groups.condition=='WSSL-CERTIFIED-L1'],on=['seed','context','inner','group'],suffixes=('_new','_base'),validate='one_to_one')
  usable=gp[(gp.status_new=='DESCRIPTIVE')&(gp.status_base=='DESCRIPTIVE')]
  checks['no_extreme_evaluable_group_recall_drop']=bool(((usable.pd_recall_new-usable.pd_recall_base)>=-.15-1e-12).all() and ((usable.dd_recall_new-usable.dd_recall_base)>=-.15-1e-12).all())
  checks['at_least_one_evaluable_group']=bool(len(usable)>0)
 status='PILOT_PASS_EXPAND' if all(checks.values()) else 'PILOT_REJECT_STOP'
 if a.scope=='pilot':
  stats=[dict(condition=kind,metric=m,scope='PILOT_RESOURCE_SCREEN_NOT_INDEPENDENT_CONFIRMATION',mean_delta=np.mean(v),median_delta=np.median(v),positive_seeds=int(sum(v>1e-12)),negative_seeds=int(sum(v< -1e-12)),seed_delta_SD=np.std(v,ddof=1),ci_low=np.nan,ci_high=np.nan,wilcoxon_p=np.nan,BH_q=np.nan) for m,v in delta.items()]
  gate=dict(candidate=kind,scope='one_split_three_seed_pilot',status=status,expand=bool(all(checks.values())),checks=checks,mean_deltas={m:float(np.mean(v)) for m,v in delta.items()},new_runs=3,independent_validation=False)
  write(HERE/'analysis'/f'{kind}_pilot_gate.json',gate,new=True)
 else:
  stat=paired(f,reference,kind);stats=stat.to_dict('records');s=stat.set_index('metric')
  ratios={m:float(f.groupby(['context','inner'])[m].std(ddof=1).mean()/reference.groupby(['context','inner'])[m].std(ddof=1).mean()) for m in ['ba','auroc']}
  fullchecks=dict(BA_positive=float(np.mean(delta['ba']))>0,BA_wins_at_least10=int(s.loc['ba','wins'])>=10,BA_CI_low_positive=float(s.loc['ba','ci_low'])>0,
                  AUROC_non_decrease=float(np.mean(delta['auroc']))>=-1e-12,MacroF1_non_decrease=float(np.mean(delta['macro_f1']))>=-1e-12,
                  PD_recall_drop_at_most01=float(np.mean(delta['pd_recall']))>=-.01-1e-12,DD_recall_drop_at_most01=float(np.mean(delta['dd_recall']))>=-.01-1e-12,
                  stability_at_most1_25=all(math.isfinite(v) and v<=1.25+1e-12 for v in ratios.values()),correctness_and_provenance=True)
  gate=dict(candidate=kind,scope='15_overlapping_development_splits_x3seeds',status='RETAIN_DEVELOPMENT_CANDIDATE' if all(fullchecks.values()) else 'REJECT_FULL',checks=fullchecks,stability_ratios=ratios,independent_validation=False)
  write(HERE/'analysis'/f'{kind}_full_gate.json',gate,new=True)
 outname={'B':'unfreezing','C':'demographic','D':'dd_loss'}[phase]
 # Published stage aliases are written once only after a pilot fails or the full matrix completes.
 closing=a.scope=='full' or not gate.get('expand',False)
 csvout(f'{kind}_{a.scope}_paired_statistics.csv',stats);csvout(f'{kind}_{a.scope}_group_analysis.csv',groups.to_dict('records'))
 if not closing:
  report(f'PHASE10{phase}_PILOT_REPORT.md',f'# {kind} pilot资源判断\n\n'+table(summary(pd.concat([f,reference],ignore_index=True))[['condition','runs','accuracy','ba','auroc','macro_f1','pd_recall','dd_recall']])+'\n\n'+table([dict(check=k,passed=v) for k,v in gate['checks'].items()])+'\n\n通过资源门槛，仅授权剩余42个已登记run，不是正式保留或独立确认。不得按此结果修改参数。')
 else:
  allmetrics=pd.concat([f,reference],ignore_index=True);csvout(f'{outname}_metrics.csv',allmetrics.to_dict('records'))
  csvout(f'{outname}_paired_statistics.csv',stats)
  costs=pd.read_csv(HERE/'analysis'/f'{kind}_{a.scope}_costs.csv');csvout(f'{phase.lower()}_costs.csv',costs.to_dict('records'))
  pgroup=pd.read_csv(HERE/'analysis'/f'{kind}_{a.scope}_parameter_groups.csv');csvout(f'{outname}_parameter_groups.csv',pgroup.to_dict('records'));csvout(f'{outname}_gradient_summary.csv',pgroup.to_dict('records'))
  if kind=='DEMO':
   csvout('demographic_fusion_metrics.csv',f.to_dict('records'));csvout('demographic_group_analysis.csv',groups.to_dict('records'))
   csvout('demographic_parameter_count.csv',costs[['condition','total_parameters','trainable_parameters','frozen_parameters']].drop_duplicates().to_dict('records'))
  integrity=dict(status='PASS_COMPLETE_WITH_NEGATIVE_RESULT' if 'REJECT' in gate['status'] else 'PASS_COMPLETE',candidate=kind,scope=a.scope,formal_runs=len(f),
                 correctness=True,provenance=True,all_frozen_states_unchanged=True,official_weights_fresh_each_run=kind.startswith('FT-'),
                 optimizer_state_saved_all_runs=True,data_version=VERSION,normalization_per_split_unchanged=True,gate=gate,protected_files_verified=guard(),timestamp_utc=utc())
  write(HERE/'analysis'/f'{outname}_integrity.json',integrity,new=True)
  filename={'B':'PHASE10B_UNFREEZING_REPORT.md','C':'PHASE10C_DEMOGRAPHIC_REPORT.md','D':'PHASE10D_DD_LOSS_REPORT.md'}[phase]
  limitation='15个重叠development splits不等于15独立临床队列。'
  if a.scope=='pilot':limitation='仅context0/inner0三seed资源筛选；无15split CI/Wilcoxon/BH，不作独立确认。'
  rationale={'B':'最后stage位置与历史Layer5相同；本轮Certified L1/norm、fresh下游状态、LR1e-5且BN affine冻结。FT-L1未通过资源闸门时不增加解冻深度：本轮没有独立证据证明瓶颈位于layer4。FT-L2为NOT_RUN，不报告最后一与最后两stage的性能差。',
             'C':'原Certified WSSL所有参数和logits冻结，新MLP4→8→16与零初始化16→2贡献共216参数；不是用人口字段替换IMU。gender是原始记录字段，设备只分层不输入；D0单独见10A报告。人口相关性可能反映队列混杂，不是疾病因果。',
             'D':'原balanced weighted-mean CE，仅DD相对cost×1.10，按train人数重标尺度；常数缩放会在实际batch权重分母中抵消，不声称保持梯度norm。未叠加focal/sampling/SupCon/AI，也未按validation错误名单加权。'}[phase]
  report(filename,f'# Phase10{phase}: {kind}\n\n状态：**{gate["status"]}**。\n\n'+table(summary(allmetrics)[['condition','runs','splits','accuracy','ba','auroc','macro_f1','pd_recall','dd_recall','nll','brier','ece']])+'\n\n'+table(pd.DataFrame(stats)[['metric','mean_delta','median_delta']])+'\n\n门槛逐项：\n\n'+table([dict(check=k,passed=v) for k,v in gate['checks'].items()])+'\n\n'+rationale+'\n\n'+limitation+'样本SD和全部九指标见metrics/paired_statistics；参数量、gradient、权重norm变化、耗时、best epoch及checkpoint/optimizer来源均保存。源数据、norm、baseline权重、split和CURRENT_MODEL保持原样。未通过不补seed、不调阈值/参数、不扩大训练；其他独立阶段按原条件继续。')
 write(HERE/'audit'/f'{kind}_{a.scope}_analysis_receipt.json',dict(status='PASS',closing_stage=closing,gate=gate,source_sha256=sha(Path(__file__)),timestamp_utc=utc()),new=True)
 guard();print(json.dumps(gate,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
