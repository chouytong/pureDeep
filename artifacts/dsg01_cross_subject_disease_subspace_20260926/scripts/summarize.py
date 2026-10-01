#!/usr/bin/env python3
"""Split-first DSG-01 inference and primary stable-error analysis."""
from __future__ import annotations
import json,sys,warnings
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr,wilcoxon,mannwhitneyu
BASE=Path('/home/zyt/deep_final/artifacts/dsg01_cross_subject_disease_subspace_20260926')
OUT=BASE/'analysis';sys.path.insert(0,str(BASE/'scripts'))
from analyze import INST,SPLITS,ACTIVITIES,payload,features,fitted,centroid,decide
OFF=Path('/home/zyt/deep_final/artifacts/str01_structured_token_residual_20260921/analysis/classification_15_paired_splits.csv')
GAIN=Path('/home/zyt/deep_final/artifacts/str01_gain_mechanism_diagnosis_20260923/analysis/validation_records_with_h1.csv')
def finite_mean(x):
 v=np.asarray(x,dtype=float);return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
def bh(values):
 p=np.asarray(values,dtype=float);q=np.full(len(p),np.nan);ok=np.isfinite(p);n=ok.sum()
 if n:
  ids=np.where(ok)[0];order=ids[np.argsort(p[ids])];z=np.minimum.accumulate((p[order]*n/np.arange(1,n+1))[::-1])[::-1];q[order]=np.minimum(1,z)
 return q
def paired(x,y):
 x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);mask=np.isfinite(x)&np.isfinite(y);d=y[mask]-x[mask]
 if len(d)<8:return (np.nan,np.nan,int(len(d)),np.nan)
 try:p=float(wilcoxon(d).pvalue) if np.any(d) else 1.0
 except ValueError:p=np.nan
 return (float(np.mean(d)),p,int(len(d)),int(np.sum(d>0)))
def boot_rho(x,y,n=4000):
 x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);mask=np.isfinite(x)&np.isfinite(y);x=x[mask];y=y[mask]
 if len(x)<8 or len(np.unique(x))<3 or len(np.unique(y))<3:return np.nan,np.nan,np.nan,np.nan,len(x)
 rho,p=spearmanr(x,y);rng=np.random.default_rng(260126);vals=[]
 for _ in range(n):
  idx=rng.integers(len(x),size=len(x))
  if len(np.unique(x[idx]))>=3 and len(np.unique(y[idx]))>=3:vals.append(spearmanr(x[idx],y[idx]).statistic)
 return float(rho),float(p),float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5)),len(x)
def per_split():
 cka=pd.read_csv(OUT/'cka_pairs_with_unbiased.csv');pairs=pd.read_csv(OUT/'alignment_transfer_pairs.csv');native=pd.read_csv(OUT/'native_probes.csv');activity=pd.read_csv(OUT/'activity_alignment_pairs.csv');aprobe=pd.read_csv(OUT/'activity_native_probes.csv');official=pd.read_csv(OFF)
 rows=[]
 for model in ('v8','str'):
  for o,i in SPLITS:
   for stage in (['activity_raw_flat','activity_context_flat','subject_embedding']+(['structured_embedding','decision_input'] if model=='str' else [])):
    c=cka[(cka.model==model)&(cka.stage==stage)&(cka.pair_type=='cross_split_same_seed')]
    c=c[((c.a_outer==o)&(c.a_inner==i))|((c.b_outer==o)&(c.b_inner==i))]
    p=pairs[(pairs.model==model)&(pairs.stage==stage)&(pairs.pair_type=='cross_split_same_seed')&(pairs.source_outer==o)&(pairs.source_inner==i)]
    inc=pairs[(pairs.model==model)&(pairs.stage==stage)&(pairs.pair_type=='cross_split_same_seed')&(pairs.target_outer==o)&(pairs.target_inner==i)]
    n=native[(native.model==model)&(native.stage==stage)&(native.outer==o)&(native.inner==i)]
    row=dict(model=model,outer=o,inner=i,stage=stage)
    for col,key in [('cka','cka'),('cka_unbiased','cka_unbiased'),('direction_cos','direction_cos'),('probe_cos','probe_cos'),('principal_angle_mean_deg','principal_angle_mean_deg')]:
     source=c if col.startswith('cka') else p
     row[key]=finite_mean(source[col])
    for col in ('native_probe_ba','native_probe_auroc','pca_probe_ba','pca_probe_auroc'):row[col]=finite_mean(n[col])
    for col in ('transfer_ba','transfer_auroc','transfer_drop_ba','transfer_drop_auroc','eval_n'):row[col]=finite_mean(inc[col])
    row['transfer_valid_pairs']=int(np.isfinite(inc.transfer_ba).sum())
    for metric in ('accuracy','balanced_accuracy','macro_f1','macro_auroc','pd_recall','dd_recall'):
     row[metric]=float(official.loc[(official.outer==o)&(official.inner==i),('str01_' if model=='str' else 'baseline_')+metric].iloc[0])
    rows.append(row)
 frame=pd.DataFrame(rows);frame.to_csv(OUT/'split_metrics_15.csv',index=False)
 act=[]
 for model in ('v8','str'):
  for kind in (['activity_raw','activity_context']+(['structured_tokens'] if model=='str' else [])):
   for name in ACTIVITIES:
    for o,i in SPLITS:
     a=activity[(activity.model==model)&(activity.kind==kind)&(activity.activity==name)&(activity.outer==o)&(activity.inner==i)]
     n=aprobe[(aprobe.model==model)&(aprobe.kind==kind)&(aprobe.activity==name)&(aprobe.outer==o)&(aprobe.inner==i)]
     act.append(dict(model=model,kind=kind,activity=name,outer=o,inner=i,direction_cos=finite_mean(a.direction_cos),probe_cos=finite_mean(a.probe_cos),principal_angle_mean_deg=finite_mean(a.principal_angle_mean_deg),probe_ba=finite_mean(n.probe_ba),probe_auroc=finite_mean(n.probe_auroc)))
 af=pd.DataFrame(act);af.to_csv(OUT/'activity_split_metrics_15.csv',index=False)
 return frame,af

def inference(frame,af):
 comparisons=[]
 for vstage,sstage in [('subject_embedding','subject_embedding'),('subject_embedding','decision_input'),('activity_raw_flat','activity_raw_flat'),('activity_context_flat','activity_context_flat')]:
  a=frame[(frame.model=='v8')&(frame.stage==vstage)].sort_values(['outer','inner']);b=frame[(frame.model=='str')&(frame.stage==sstage)].sort_values(['outer','inner'])
  for col in ('cka','cka_unbiased','direction_cos','probe_cos','principal_angle_mean_deg','native_probe_ba','native_probe_auroc','transfer_drop_ba','transfer_drop_auroc'):
   delta,p,n,wins=paired(a[col],b[col]);comparisons.append(dict(v8_stage=vstage,str_stage=sstage,metric=col,v8_mean=finite_mean(a[col]),str_mean=finite_mean(b[col]),delta=delta,p=p,n=n,str_wins=wins))
 cf=pd.DataFrame(comparisons);cf['q']=bh(cf.p);cf.to_csv(OUT/'model_paired_inference.csv',index=False)
 ac=[]
 for kind in ('activity_raw','activity_context'):
  for name in ACTIVITIES:
   a=af[(af.model=='v8')&(af.kind==kind)&(af.activity==name)].sort_values(['outer','inner']);b=af[(af.model=='str')&(af.kind==kind)&(af.activity==name)].sort_values(['outer','inner'])
   for metric in ('direction_cos','probe_cos','probe_ba','probe_auroc'):
    d,p,n,w=paired(a[metric],b[metric]);ac.append(dict(kind=kind,activity=name,metric=metric,v8_mean=finite_mean(a[metric]),str_mean=finite_mean(b[metric]),delta=d,p=p,n=n,str_wins=w))
 ac=pd.DataFrame(ac);ac['q']=np.nan
 for _,ix in ac.groupby(['kind','metric']).groups.items():ac.loc[ix,'q']=bh(ac.loc[ix,'p'])
 ac.to_csv(OUT/'activity_paired_inference_bh.csv',index=False)
 # Split is the unit; compare each model's stability and disease accessibility with its own classification.
 correlations=[]
 for model,stage in [('v8','subject_embedding'),('str','subject_embedding'),('str','decision_input')]:
  g=frame[(frame.model==model)&(frame.stage==stage)].sort_values(['outer','inner'])
  for predictor in ('cka_unbiased','direction_cos','probe_cos','native_probe_ba','native_probe_auroc','transfer_drop_ba'):
   for outcome in ('balanced_accuracy','macro_auroc','macro_f1','pd_recall','dd_recall'):
    rho,p,lo,hi,n=boot_rho(g[predictor],g[outcome]);correlations.append(dict(model=model,stage=stage,predictor=predictor,outcome=outcome,rho=rho,p=p,ci95_low=lo,ci95_high=hi,n=n))
 corr=pd.DataFrame(correlations);corr['q']=np.nan
 for _,ix in corr.groupby(['model','stage']).groups.items():corr.loc[ix,'q']=bh(corr.loc[ix,'p'])
 corr.to_csv(OUT/'stability_classification_correlations.csv',index=False)

def stable_errors():
 historical=pd.read_csv(GAIN,dtype={'subject_id':str})
 rows=[]
 for model in ('v8','str'):
  stage='subject_embedding' if model=='v8' else 'decision_input'
  for o,i in SPLITS:
   by_seed=[]
   for s in (42,43,44):
    inst=(s,o,i);tr=payload(model,inst,'train');va=payload(model,inst,'validation');ft=fitted(model,inst,stage)
    X=features(tr,stage).astype(float);y=tr['label'].astype(int);V=features(va,stage).astype(float);v=va['label'].astype(int)
    c0=X[y==0].mean(0);c1=X[y==1].mean(0);axis=(c1-c0);axis/=np.linalg.norm(axis)
    mid=(c0+c1)/2
    axis_true=(2*v-1)*((V-mid)@axis)
    own=np.where(v[:,None]==0,c0,c1);wrong=np.where(v[:,None]==0,c1,c0)
    centroid_distance=np.linalg.norm(V-wrong,axis=1)-np.linalg.norm(V-own,axis=1)
    probe_true=(2*v-1)*decide(ft['sc'],ft['probe'],V)
    slog=va['subject_logits'][:,1]-va['subject_logits'][:,0]
    rlog=va['structured_logits'][:,1]-va['structured_logits'][:,0] if model=='str' else np.full(len(v),np.nan)
    flog=va['final_logits'][:,1]-va['final_logits'][:,0]
    by_seed.append(pd.DataFrame(dict(subject_id=va['subject_id'].astype(str),label=v,axis_true_margin=axis_true,centroid_distance=centroid_distance,probe_true_margin=probe_true,subject_logits_true_margin=(2*v-1)*slog,structured_logits_true_margin=(2*v-1)*rlog,final_logits_true_margin=(2*v-1)*flog)))
   merged=pd.concat(by_seed).groupby(['subject_id','label'],as_index=False).mean(numeric_only=True);merged['model']=model;merged['outer']=o;merged['inner']=i;rows.append(merged)
 frame=pd.concat(rows,ignore_index=True)
 labels=historical[['subject_id','target','v8_stability','str_stability']].drop_duplicates('subject_id')
 assert len(labels)==390
 frame=frame.merge(labels,on='subject_id',validate='many_to_one')
 frame['primary_status']=np.where(frame.model=='str',frame.str_stability,frame.v8_stability)
 frame.to_csv(OUT/'stable_error_validation_appearances.csv',index=False)
 subjects=frame.groupby(['model','subject_id','label','primary_status'],as_index=False).mean(numeric_only=True)
 subjects.to_csv(OUT/'stable_error_subject_means.csv',index=False)
 counts=subjects.groupby(['model','primary_status']).size().to_dict()
 count_json={f'{m}:{s}':int(n) for (m,s),n in counts.items()}
 # Historical 12-seed-fold definition is sensitivity only, kept separate.
 old=Path('/home/zyt/deep_final/artifacts/str01_gain_mechanism_diagnosis_20260923/analysis/subject_stability_transitions.csv')
 count_json['historical_sensitivity']={'v8_stable_error':54,'str_stable_error':42,'source':str(old)}
 (OUT/'stable_error_definitions.json').write_text(json.dumps({'primary':'three seeds aggregated first within each of four validation appearances; error rate >=0.75 stable-error, <=0.25 stable-correct','counts':count_json},indent=2))
 tests=[]
 for model in ('v8','str'):
  for label in ('all','PD','DD'):
   g=subjects[subjects.model==model]
   if label!='all':g=g[g.label==(0 if label=='PD' else 1)]
   good=g[g.primary_status=='stable_correct'];bad=g[g.primary_status=='stable_error']
   for metric in ('axis_true_margin','centroid_distance','probe_true_margin','subject_logits_true_margin','structured_logits_true_margin','final_logits_true_margin'):
    x=good[metric].dropna().to_numpy();y=bad[metric].dropna().to_numpy()
    if len(x)<5 or len(y)<5:continue
    stat,p=mannwhitneyu(y,x,alternative='two-sided');delta=2*stat/(len(x)*len(y))-1
    tests.append(dict(model=model,label=label,metric=metric,stable_correct_n=len(x),stable_error_n=len(y),correct_mean=float(x.mean()),error_mean=float(y.mean()),error_minus_correct=float(y.mean()-x.mean()),cliffs_delta=float(delta),p=float(p)))
 tests=pd.DataFrame(tests);tests['q']=np.nan
 for _,ix in tests.groupby(['model','label']).groups.items():tests.loc[ix,'q']=bh(tests.loc[ix,'p'])
 tests.to_csv(OUT/'stable_error_group_inference_bh.csv',index=False)
if __name__=='__main__':
 frame,af=per_split();inference(frame,af);stable_errors()
 print('summary complete',flush=True)
