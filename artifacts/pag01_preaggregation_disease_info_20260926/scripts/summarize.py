#!/usr/bin/env python3
"""Seed-first 15-split paired inference, sensitivities, ambiguity and controls."""
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import wilcoxon,rankdata
B=Path('/home/zyt/deep_final/artifacts/pag01_preaggregation_disease_info_20260926');O=B/'analysis';rng=np.random.default_rng(20260926)
def bh(p):
 p=np.asarray(p,float);order=np.argsort(p);q=np.empty(len(p));q[order]=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];return np.minimum(1,q)
def ci(d):
 d=np.asarray(d,float);v=d[rng.integers(0,len(d),(10000,len(d)))].mean(axis=1);return [float(x) for x in np.quantile(v,[.025,.975])]
def effect(d):
 d=np.asarray(d,float);nz=d[np.abs(d)>1e-12]
 if not len(nz):return 0.0
 r=rankdata(np.abs(nz));return float((r[nz>0].sum()-r[nz<0].sum())/r.sum())
def paired(stage,metric,d):
 d=np.asarray(d,float);bounds=ci(d);return dict(stage=stage,metric=metric,mean_delta=float(d.mean()),sd_delta=float(d.std(ddof=1)),median_delta=float(np.median(d)),ci95_low=bounds[0],ci95_high=bounds[1],improved_splits=int(sum(d>0)),worsened_splits=int(sum(d<0)),rank_biserial=effect(d),p=float(wilcoxon(d).pvalue) if np.any(np.abs(d)>1e-12) else 1.0)
def split(frame,keys):return frame.groupby(keys,as_index=False).mean(numeric_only=True)
probe=pd.read_csv(O/'disease_probe_45.csv');p15=split(probe,['outer','inner','pca_k','stage']);p15.to_csv(O/'disease_probe_15.csv',index=False)
sig=pd.read_csv(O/'increment_signal_45.csv');s15=split(sig,['outer','inner','pca_k','stage']);s15.to_csv(O/'increment_signal_15.csv',index=False)
summary=[]
for k,gk in p15.groupby('pca_k'):
 for stage,g in gk.groupby('stage'):
  for metric in ['auroc','delta_auroc','ba','macro_f1','pd_recall','dd_recall','log_loss']:
   summary.append(dict(pca_k=k,stage=stage,metric=metric,mean=g[metric].mean(),sd=g[metric].std(ddof=1),median=g[metric].median()))
pd.DataFrame(summary).to_csv(O/'disease_probe_summary.csv',index=False)
summary=[]
for (k,stage),g in s15.groupby(['pca_k','stage']):
 for metric in ['increment_r2','increment_mae','increment_spearman','corrected_auroc_delta']:
  summary.append(dict(pca_k=k,stage=stage,metric=metric,mean=g[metric].mean(),sd=g[metric].std(ddof=1),median=g[metric].median()))
pd.DataFrame(summary).to_csv(O/'increment_signal_summary.csv',index=False)
# Primary stage vs score-only AUROC, with descriptive other metrics separately.
rows=[]
for k in (8,16,32):
 gk=p15[p15.pca_k==k];base=p15[(p15.pca_k==16)&(p15.stage=='score_only')].set_index(['outer','inner'])
 for stage in ['activity_raw','activity_context','subject_embedding','structured_embedding','decision_input']:
  g=gk[gk.stage==stage].set_index(['outer','inner'])
  for metric in ['auroc','ba','macro_f1','pd_recall','dd_recall','log_loss']:
   sign=-1 if metric=='log_loss' else 1;rows.append(dict(pca_k=k,**paired(stage,metric,sign*(g[metric]-base[metric]))))
inf=pd.DataFrame(rows);inf['q_bh_within_k']=np.nan
for k,g in inf.groupby('pca_k'):inf.loc[g.index,'q_bh_within_k']=bh(g.p)
inf.to_csv(O/'probe_vs_baseline_paired_bh.csv',index=False)
primary=inf[(inf.pca_k==16)&(inf.metric=='auroc')].copy();primary['q_bh_primary_auc']=bh(primary.p);primary.to_csv(O/'primary_auc_inference_bh.csv',index=False)
# Direct stage comparisons are paired, not inferred from individual significance.
rows=[]
comparisons=[('raw_minus_context','activity_raw','activity_context'),('context_minus_subject','activity_context','subject_embedding'),('subject_minus_decision','subject_embedding','decision_input'),('raw_minus_decision','activity_raw','decision_input'),('raw_minus_subject','activity_raw','subject_embedding'),('structured_minus_subject','structured_embedding','subject_embedding')]
for k in (8,16,32):
 for name,left,right in comparisons:
  for metric in ['auroc']:
   a=p15[(p15.pca_k==k)&(p15.stage==left)].set_index(['outer','inner']);b=p15[(p15.pca_k==k)&(p15.stage==right)].set_index(['outer','inner']);rows.append(dict(pca_k=k,comparison=name,source='disease_probe',**paired(name,metric,a[metric]-b[metric])))
  for metric in ['increment_r2','increment_mae','increment_spearman','corrected_auroc_delta']:
   a=s15[(s15.pca_k==k)&(s15.stage==left)].set_index(['outer','inner']);b=s15[(s15.pca_k==k)&(s15.stage==right)].set_index(['outer','inner']);sign=-1 if metric=='increment_mae' else 1;rows.append(dict(pca_k=k,comparison=name,source='increment_signal',**paired(name,metric,sign*(a[metric]-b[metric]))))
direct=pd.DataFrame(rows);direct['q_bh_within_k_source']=np.nan
for (k,source),g in direct.groupby(['pca_k','source']):direct.loc[g.index,'q_bh_within_k_source']=bh(g.p)
direct.to_csv(O/'direct_stage_paired_bh.csv',index=False)
# Explicit fixed progression direction per split; a rebound means no monotone attenuation.
prog=[]
for k in (8,16,32):
 for metric,table in [('auroc',p15),('increment_r2',s15),('increment_spearman',s15),('corrected_auroc_delta',s15)]:
  z=table[table.pca_k==k].pivot(index=['outer','inner'],columns='stage',values=metric)
  for ix,row in z.iterrows():
   vals=[row[s] for s in ['activity_raw','activity_context','subject_embedding','decision_input']]
   prog.append(dict(outer=ix[0],inner=ix[1],pca_k=k,metric=metric,raw_to_context_drop=vals[0]-vals[1],context_to_subject_drop=vals[1]-vals[2],subject_to_decision_drop=vals[2]-vals[3],raw_to_decision_drop=vals[0]-vals[3],strict_monotone_attenuation=all(vals[j]>vals[j+1] for j in range(3))))
pd.DataFrame(prog).to_csv(O/'stage_progression_15.csv',index=False)
# Conditional permutation: same score-decile distribution in train, no validation permutation.
perm=pd.read_csv(O/'conditional_permutation_45.csv');n15=split(perm.groupby(['outer','inner','seed','stage'],as_index=False).mean(numeric_only=True),['outer','inner','stage']);n15.to_csv(O/'conditional_permutation_15.csv',index=False)
control=[]
for stage in ['activity_raw','activity_context','subject_embedding','structured_embedding','decision_input']:
 a=p15[(p15.pca_k==16)&(p15.stage==stage)].set_index(['outer','inner']);z=n15[n15.stage==stage].set_index(['outer','inner']);d=a.delta_auroc-z.permuted_delta_auroc
 control.append(dict(**paired(stage,'real_minus_permuted_delta_auc',d),real_mean=float(a.delta_auroc.mean()),permutation_mean=float(z.permuted_delta_auroc.mean())))
control=pd.DataFrame(control);control['q_bh']=bh(control.p);control.to_csv(O/'conditional_permutation_inference_bh.csv',index=False)
# Ambiguity by subject, derived before seeing val label/correctness.
amb=pd.read_csv(O/'ambiguity_subject_groups_45.csv');amb15=split(amb,['outer','inner','ambiguity']);amb15.to_csv(O/'ambiguity_subject_groups_15.csv',index=False)
ambinf=[]
for metric in ['str_error_rate','raw_error_rate','raw_auroc_gain']:
 z=amb15.pivot(index=['outer','inner'],columns='ambiguity',values=metric)
 for right in ['medium','low']:
  d=(z['high']-z[right]).dropna();ambinf.append(paired(f'high_minus_{right}',metric,d))
ambinf=pd.DataFrame(ambinf);ambinf['q_bh']=bh(ambinf.p);ambinf.to_csv(O/'ambiguity_subject_inference_bh.csv',index=False)
# Pair groups: highest ambiguity of either subject; pair counts are descriptive, 15 splits inference.
pair=pd.read_csv(O/'ambiguity_pair_records.csv.gz',dtype={'dd_subject_id':str,'pd_subject_id':str})
rows=[]
for (o,i,seed,group),g in pair.groupby(['outer','inner','seed','pair_ambiguity']):
 wrong=g[~g.original_correct];good=g[g.original_correct];recovered=int(g.recovered.sum());broken=int(g.newly_broken.sum())
 rows.append(dict(outer=o,inner=i,seed=seed,ambiguity=group,pairs=len(g),original_wrong=len(wrong),original_correct=len(good),recovered=recovered,newly_broken=broken,recovery_rate=recovered/len(wrong) if len(wrong) else np.nan,harm_rate=broken/len(good) if len(good) else np.nan,net_pair_gain=recovered-broken,net_pair_gain_per_pair=(recovered-broken)/len(g),stable_error_participation=g.any_stable_error.mean()))
pg=pd.DataFrame(rows);pg.to_csv(O/'ambiguity_pair_groups_45.csv',index=False);pg15=split(pg,['outer','inner','ambiguity']);pg15.to_csv(O/'ambiguity_pair_groups_15.csv',index=False)
pgi=[]
for metric in ['recovery_rate','harm_rate','net_pair_gain_per_pair']:
 z=pg15.pivot(index=['outer','inner'],columns='ambiguity',values=metric)
 for right in ['medium','low']:
  d=(z['high']-z[right]).dropna();pgi.append(paired(f'high_minus_{right}',metric,d))
pgi=pd.DataFrame(pgi);pgi['q_bh']=bh(pgi.p);pgi.to_csv(O/'ambiguity_pair_inference_bh.csv',index=False)
# Primary stable-error is strictly sensitivity, never used to define threshold or train probe.
sens=[]
for (o,i,seed,hard),g in pair.groupby(['outer','inner','seed','any_stable_error']):
 wrong=g[~g.original_correct];good=g[g.original_correct];sens.append(dict(outer=o,inner=i,seed=seed,any_stable_error=hard,pairs=len(g),recovery_rate=g.recovered.sum()/len(wrong) if len(wrong) else np.nan,harm_rate=g.newly_broken.sum()/len(good) if len(good) else np.nan,net_pair_gain_per_pair=(g.recovered.sum()-g.newly_broken.sum())/len(g)))
ss=pd.DataFrame(sens);ss.to_csv(O/'stable_error_pair_sensitivity_45.csv',index=False);split(ss,['outer','inner','any_stable_error']).to_csv(O/'stable_error_pair_sensitivity_15.csv',index=False)
print('primary_auc');print(primary.round(4).to_string(index=False));print('direct_raw_decision');print(direct[(direct.pca_k==16)&(direct.comparison=='raw_minus_decision')].round(4).to_string(index=False));print('permutation');print(control.round(4).to_string(index=False));print('ambiguity_subject');print(amb15.groupby('ambiguity')[['n','str_error_rate','raw_auroc_gain']].mean().round(4).to_string());print('ambiguity_pairs');print(pg15.groupby('ambiguity')[['pairs','recovery_rate','harm_rate','net_pair_gain_per_pair']].mean().round(4).to_string())
