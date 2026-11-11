from common8 import *
def main():
 assert load(HERE/'audit/startup_integrity.json')['status']=='PASS_BEFORE_PHASE8_FITS';guard();rows=[];complement=[]
 for c,i,sp,ex in splits():
  parts=[]
  for seed in [42,43,44]:
   f=features(seed,c,i);q=f['is_validation'];ids=f['subject_ids'][q];y=f['target'][q];p=f['probabilities'][q];z=f['logits'][q]
   order=np.argsort(ids);ids=ids[order];y=y[order];p=p[order];z=z[order]
   if parts:assert np.array_equal(ids,parts[0][0]) and np.array_equal(y,parts[0][1])
   parts.append((ids,y,p,z));rows.append(save_prediction('baseline',seed,c,i,ids,y,p,z))
  p=np.mean([a[2].astype(float) for a in parts],0);z=np.mean([a[3].astype(float) for a in parts],0);logp=np.exp(z-logsumexp(z,1,keepdims=True))
  rows.append(save_prediction('probability_ensemble',-1,c,i,ids,y,p));rows.append(save_prediction('logit_ensemble',-1,c,i,ids,y,logp,z))
  correct=np.stack([(a[2][:,1]>.5)==y for a in parts]);new=(p[:,1]>.5)==y
  for label in [-1,0,1]:
   keep=np.ones(len(y),bool) if label==-1 else y==label;t=correct[:,keep];n=new[keep]
   complement.append(dict(context=c,inner=i,label='all' if label==-1 else ['PD','DD'][label],subject_count=int(keep.sum()),all_seeds_correct=int(t.all(0).sum()),all_seeds_wrong=int((~t.any(0)).sum()),mixed_seed_correctness=int((t.any(0)&~t.all(0)).sum()),ensemble_correct=int(n.sum()),correct_single_seed_mean=float(t.sum(1).mean()),ensemble_minus_mean_correct=float(n.sum()-t.sum(1).mean()),ensemble_misses_despite_any_seed_correct=int((t.any(0)&~n).sum()),ensemble_recovers_nonunanimous_cases=int((~t.all(0)&n).sum())))
 frame=pd.DataFrame(rows);frame.to_csv(HERE/'analysis/ensemble_metrics.csv',index=False);pd.DataFrame(complement).to_csv(HERE/'analysis/ensemble_error_complementarity.csv',index=False)
 baseline=frame[frame.condition=='baseline'];stats=[];decisions=[]
 for name in ['probability_ensemble','logit_ensemble']:
  selected=frame[frame.condition==name];s=paired(selected,baseline,name);mask=s.metric.isin(['ba','auroc']);s['primary_bh_q']=np.nan;s.loc[mask,'primary_bh_q']=bh_planned(s.loc[mask,'wilcoxon_p']);s['stage_all9_sensitivity_q']=bh_planned(s.wilcoxon_p);stats.append(s)
  decisions.append(gate(selected,s,baseline,name,ensemble=True))
 pd.concat(stats).to_csv(HERE/'analysis/ensemble_paired_statistics.csv',index=False)
 means=frame.groupby('condition')[METRICS].mean().reset_index()
 text='# Phase8A1 三seed集成复核\n\n历史Phase3B已报告概率集成，本轮不是首次发现。原45checkpoint和per-model prediction来源通过；无新训练/推理/温度/阈值拟合。\n\n'+table(means)+'\n\n'+table(pd.concat(stats)[['condition','metric','mean_delta','wins','ci_low','ci_high','wilcoxon_p','primary_bh_q']])+'\n\n'+''.join(f"{d['condition']}: **{d['status']}**；失败条件：{', '.join(k for k,v in d['checks'].items() if not v)}。\n\n" for d in decisions)+'固定3模型ensemble的seedSD不可识别，不报告虚假0；between-split稳定性与seed方差不同。Logit ensemble仅敏感性，不按结果替换主要候选。所有15split共享患者，区间不是独立临床验证。患者cluster敏感性在最终统一统计阶段追加，不改变此处原始paired/gate。完成本报告后进入A2。\n'
 (HERE/'reports/ENSEMBLE_REPORT.md').write_text(text)
 write(HERE/'analysis/ensemble_integrity.json',dict(status='PASS_COMPLETE_A1',paired_units=15,baseline_models=45,probability_primary=True,raw_logit_sensitivity=True,new_training=0,new_model_forward=0,seed_subset_search=False,temperature_threshold_fit=False,decisions=decisions,report_sha256=sha(HERE/'reports/ENSEMBLE_REPORT.md')))
 guard();print(json.dumps(clean(dict(stage='A1',means=means.to_dict('records'),decisions=decisions)),ensure_ascii=False))
if __name__=='__main__':main()
