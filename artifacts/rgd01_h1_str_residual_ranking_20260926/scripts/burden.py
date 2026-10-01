#!/usr/bin/env python3
"""Unique-subject ranking burden and leave-hard-subjects-out sensitivity."""
from pathlib import Path
import gzip,json
import numpy as np,pandas as pd
from scipy.stats import mannwhitneyu,wilcoxon
from sklearn.metrics import roc_auc_score
BASE=Path('/home/zyt/deep_final/artifacts/rgd01_h1_str_residual_ranking_20260926');OUT=BASE/'analysis';ROOT=Path('/home/zyt/deep_final')
EM=ROOT/'artifacts/dsg01_cross_subject_disease_subspace_20260926/extraction/embeddings';H1DIR=BASE/'h1_reproduction'
def bh(p):
 p=np.asarray(p,float);ix=np.argsort(p);q=np.empty(len(p));q[ix]=np.minimum.accumulate((p[ix]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];return np.minimum(1,q)
sub=pd.read_csv(OUT/'subject_ranking_burden_15.csv',dtype={'subject_id':str})
unique=sub.groupby(['subject_id','label','status'],as_index=False).mean(numeric_only=True);assert len(unique)==390
unique.to_csv(OUT/'subject_ranking_burden_390.csv',index=False)
status=unique.set_index('subject_id').status.to_dict();assert unique.status.value_counts().to_dict()=={'stable_correct':284,'stable_error':74,'unstable':32}
comp=[]
for metric in ('str_error_fraction','h1_error_fraction','h1_rescue_fraction','str_rescue_fraction','net_h1_rescue_fraction'):
 for other in ('stable_correct','unstable'):
  a=unique[unique.status=='stable_error'][metric].to_numpy();b=unique[unique.status==other][metric].to_numpy();u,p=mannwhitneyu(a,b,alternative='two-sided')
  comp.append(dict(metric=metric,comparison=f'stable_error_vs_{other}',error_mean=float(a.mean()),other_mean=float(b.mean()),delta=float(a.mean()-b.mean()),cliffs_delta=float(2*u/(len(a)*len(b))-1),p=float(p)))
comp=pd.DataFrame(comp);comp['q']=bh(comp.p);comp.to_csv(OUT/'subject_burden_group_inference_bh.csv',index=False)
# Pair categories counted once per PD-DD pair; H1 is deterministic, STR averaged over 3 seeds at split level.
pairs=pd.read_csv(OUT/'ranking_pairs.csv.gz',dtype={'dd_subject_id':str,'pd_subject_id':str})
pairs['any_stable_error']=(pairs.dd_status=='stable_error')|(pairs.pd_status=='stable_error')
pairs['both_stable_error']=(pairs.dd_status=='stable_error')&(pairs.pd_status=='stable_error')
pair_group=pairs.groupby(['outer','inner','seed','any_stable_error','category'],as_index=False).size();pair_group.to_csv(OUT/'pair_category_by_stable_error.csv',index=False)
counts=pairs.groupby(['any_stable_error','category']).size().unstack(fill_value=0)
counts.to_csv(OUT/'pair_category_status_counts.csv')
# Top 5/10% are selected descriptively from all validation appearances. No performance selection uses this.
hard=unique.sort_values(['str_error_fraction','subject_id'],ascending=[False,True]);sets={'none':set(),'str_stable_error':set(unique.loc[unique.status=='stable_error','subject_id']),'top_5pct_str_rank_burden':set(hard.head(20).subject_id),'top_10pct_str_rank_burden':set(hard.head(39).subject_id)}
rows=[]
for o in range(5):
 for i in range(3):
  with np.load(H1DIR/f'outer{o}_inner{i}.npz',allow_pickle=False) as h:
   hi={str(x):j for j,x in enumerate(h['validation_subject_id'])}
   for s in (42,43,44):
    with np.load(EM/f'str_seed{s}_outer{o}_inner{i}_validation.npz',allow_pickle=False) as z:
     ids=z['subject_id'].astype(str);y=z['label'].astype(int);ss=z['final_logits'][:,1]-z['final_logits'][:,0]
    hs=h['validation_h1_probability_dd'][[hi[x] for x in ids]]
    for name,excluded in sets.items():
     keep=np.array([x not in excluded for x in ids]);yy=y[keep]
     if len(np.unique(yy))<2:continue
     hauc=float(roc_auc_score(yy,hs[keep]));sauc=float(roc_auc_score(yy,ss[keep]));rows.append(dict(outer=o,inner=i,seed=s,condition=name,removed=int((~keep).sum()),kept=int(keep.sum()),pd_n=int(sum(yy==0)),dd_n=int(sum(yy==1)),pairs=int(sum(yy==0)*sum(yy==1)),h1_auroc=hauc,str_auroc=sauc,gap=hauc-sauc))
frame=pd.DataFrame(rows);frame.to_csv(OUT/'leave_hard_out_45.csv',index=False)
split=frame.groupby(['outer','inner','condition'],as_index=False).mean(numeric_only=True);split.to_csv(OUT/'leave_hard_out_15.csv',index=False)
base=split[split.condition=='none'].set_index(['outer','inner']);summary=[]
for name in sets:
 g=split[split.condition==name].set_index(['outer','inner']);d=g.gap-base.gap
 summary.append(dict(condition=name,excluded_unique_subjects=len(sets[name]),overlap_str_stable_error=len(sets[name]&sets['str_stable_error']),mean_kept=float(g.kept.mean()),mean_pairs=float(g.pairs.mean()),mean_h1_auroc=float(g.h1_auroc.mean()),mean_str_auroc=float(g.str_auroc.mean()),mean_gap=float(g.gap.mean()),gap_change_vs_full=float(d.mean()),gap_change_p=float(wilcoxon(d).pvalue) if name!='none' else np.nan))
pd.DataFrame(summary).to_csv(OUT/'leave_hard_out_summary.csv',index=False)
print(unique.groupby('status')[['str_error_fraction','h1_error_fraction','h1_rescue_fraction','str_rescue_fraction','net_h1_rescue_fraction']].mean().round(4).to_string())
print(pd.DataFrame(summary).round(4).to_string(index=False))
print('pair categories by status',counts.to_string())
