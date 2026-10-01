#!/usr/bin/env python3
"""Split-level inference, negative controls, stable-error sensitivity and pair attribution."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy.stats import wilcoxon,mannwhitneyu
B=Path('/home/zyt/deep_final/artifacts/prr01_frozen_str_pair_rescue_20260926');O=B/'analysis';rng=np.random.default_rng(20260926)
def bh(p):
 p=np.asarray(p,float);ix=np.argsort(p);q=np.empty(len(p));q[ix]=np.minimum.accumulate((p[ix]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];return np.minimum(1,q)
def bootci(d):
 d=np.asarray(d,float);v=d[rng.integers(0,len(d),(10000,len(d)))].mean(axis=1);return [float(x) for x in np.quantile(v,[.025,.975])]
a=pd.read_csv(O/'method_metrics_45.csv');s=a.groupby(['outer','inner','stage'],as_index=False).mean(numeric_only=True);s.to_csv(O/'method_metrics_15.csv',index=False)
metrics=['corrected_auroc','auroc_delta_vs_str','corrected_ba','recovered_pairs','unrecovered_h1_rescue_pairs','recovery_rate','newly_broken_pairs','harm_rate','str_correct_preservation','all_wrong_recovered_pairs','net_pair_gain','targeted_net_pair_gain']
summary=[]
for stage,g in s.groupby('stage'):
 for m in metrics:summary.append(dict(stage=stage,metric=m,mean=g[m].mean(),sd=g[m].std(ddof=1),median=g[m].median(),min=g[m].min(),max=g[m].max()))
pd.DataFrame(summary).to_csv(O/'method_summary_15.csv',index=False)
base=s[s.stage=='B_score_only'].set_index(['outer','inner']);comparisons=[]
for stage in ['C_activity_raw','D_subject','E_structured','F_decision']:
 g=s[s.stage==stage].set_index(['outer','inner'])
 for metric in ['auroc_delta_vs_str','recovery_rate','net_pair_gain','targeted_net_pair_gain','harm_rate','corrected_ba']:
  d=g[metric]-base[metric];sign=-1 if metric=='harm_rate' else 1
  ci=bootci(d);comparisons.append(dict(stage=stage,metric=metric,mean_difference=d.mean(),sd_difference=d.std(ddof=1),median_difference=d.median(),ci95_low=ci[0],ci95_high=ci[1],improved_splits=int(sum(sign*d>0)),worsened_splits=int(sum(sign*d<0)),p=wilcoxon(d).pvalue if np.any(d) else 1.0))
c=pd.DataFrame(comparisons);c['q_bh_all']=bh(c.p);c.to_csv(O/'paired_vs_score_only_bh.csv',index=False)
# For multiplicity-controlled formal decision emphasize AUROC, recovery, net gain; broader table retains all comparisons.
main=c[c.metric.isin(['auroc_delta_vs_str','recovery_rate','net_pair_gain'])].copy();main['q_bh_primary_family']=bh(main.p);main.to_csv(O/'primary_pair_inference_bh.csv',index=False)
neg=pd.read_csv(O/'class_conditional_permutation_45.csv');ns=neg.groupby(['outer','inner','seed','stage'],as_index=False).mean(numeric_only=True);ns=ns.groupby(['outer','inner','stage'],as_index=False).mean(numeric_only=True);ns.to_csv(O/'permutation_mean_15.csv',index=False)
ni=[]
for stage in ['C_activity_raw','D_subject','E_structured','F_decision']:
 g=s[s.stage==stage].set_index(['outer','inner']);z=ns[ns.stage==stage].set_index(['outer','inner'])
 for metric in ['auroc_delta_vs_str','recovery_rate','net_pair_gain']:
  d=g[metric]-z[metric];ci=bootci(d);ni.append(dict(stage=stage,metric=metric,real_mean=g[metric].mean(),permutation_mean=z[metric].mean(),mean_difference=d.mean(),ci95_low=ci[0],ci95_high=ci[1],wins=int(sum(d>0)),p=wilcoxon(d).pvalue if np.any(d) else 1.0))
ni=pd.DataFrame(ni);ni['q_bh']=bh(ni.p);ni.to_csv(O/'negative_control_inference_bh.csv',index=False)
# Descriptive hard-subject pair strata; no pairwise p-values and no claim of independent pairs.
pair=pd.read_csv(O/'pair_level_records.csv.gz',dtype={'dd_subject_id':str,'pd_subject_id':str})
pair['recovered_h1_rescue']=(pair.original_category=='h1_rescue')*(pair.corrected_margin>0).astype(float)
pair['broken_original_correct']=((pair.original_category=='both_correct')|(pair.original_category=='str_rescue'))*(pair.corrected_margin<0).astype(float)
hr=[]
for (o,i,seed,stage,hard),g in pair.groupby(['outer','inner','seed','stage','any_stable_error']):
 rescue=g[g.original_category=='h1_rescue'];origgood=g[g.str_original_margin>0];allbad=g[g.str_original_margin<0]
 rec=int((rescue.corrected_margin>0).sum());harm=int((origgood.corrected_margin<0).sum());allrec=int((allbad.corrected_margin>0).sum())
 hr.append(dict(outer=o,inner=i,seed=seed,stage=stage,any_stable_error=hard,pairs=len(g),h1_rescue_pairs=len(rescue),recovered_h1_rescue=rec,recovery_rate=rec/len(rescue) if len(rescue) else np.nan,str_correct_pairs=len(origgood),newly_broken=harm,harm_rate=harm/len(origgood) if len(origgood) else np.nan,net_pair_gain=allrec-harm,targeted_net_pair_gain=rec-harm))
h=pd.DataFrame(hr);h.to_csv(O/'hard_subject_pairs_45.csv',index=False);h15=h.groupby(['outer','inner','stage','any_stable_error'],as_index=False).mean(numeric_only=True);h15.to_csv(O/'hard_subject_pairs_15.csv',index=False)
# Conditional attribution only for best net-AUC stage; these are associations induced partly by category selection.
attr=[]
for stage in ['C_activity_raw','F_decision']:
 x=pair[pair.stage==stage].copy();x['outcome']=np.where((x.original_category=='h1_rescue')&(x.corrected_margin>0),'recovered_h1_rescue',np.where((x.original_category=='h1_rescue')&(x.corrected_margin<=0),'unrecovered_h1_rescue',np.where((x.str_original_margin>0)&(x.corrected_margin<0),'newly_broken','other')))
 for outcome,g in x[x.outcome!='other'].groupby('outcome'):
  for metric in ['str_original_margin','h1_margin','predicted_residual_pair_margin','activity_rep_residual_magnitude']:
   # split-seed then split mean, avoiding pair pseudoreplication
   by=g.groupby(['outer','inner','seed'])[metric].mean().groupby(['outer','inner']).mean()
   attr.append(dict(stage=stage,outcome=outcome,metric=metric,pair_instances=len(g),split_count=len(by),split_mean=by.mean(),split_sd=by.std(ddof=1),split_median=by.median(),stable_error_fraction=g.any_stable_error.mean()))
pd.DataFrame(attr).to_csv(O/'conditional_pair_attribution.csv',index=False)
print('primary');print(main.round(4).to_string(index=False));print('negative');print(ni.round(4).to_string(index=False));print('hard');print(h15.groupby(['stage','any_stable_error'])[['recovery_rate','harm_rate','net_pair_gain','targeted_net_pair_gain']].mean().round(4).to_string())
