#!/usr/bin/env python3
"""Contrast cross-split and same-split cross-seed stability controls."""
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import wilcoxon
BASE=Path('/home/zyt/deep_final/artifacts/dsg01_cross_subject_disease_subspace_20260926/analysis')
c=pd.read_csv(BASE/'cka_pairs_with_unbiased.csv');a=pd.read_csv(BASE/'alignment_transfer_pairs.csv')
rows=[]
for model,stage in sorted(set(zip(a.model,a.stage))):
 for o in range(5):
  for i in range(3):
   for metric in ('cka_unbiased','direction_cos','probe_cos'):
    source=c if metric=='cka_unbiased' else a
    z=source[(source.model==model)&(source.stage==stage)]
    if metric=='cka_unbiased':
     same=z[(z.pair_type=='same_split_cross_seed')&(z.a_outer==o)&(z.a_inner==i)]
     cross=z[(z.pair_type=='cross_split_same_seed')&(((z.a_outer==o)&(z.a_inner==i))|((z.b_outer==o)&(z.b_inner==i)))]
    else:
     same=z[(z.pair_type=='same_split_cross_seed')&(z.source_outer==o)&(z.source_inner==i)]
     cross=z[(z.pair_type=='cross_split_same_seed')&(z.source_outer==o)&(z.source_inner==i)]
    rows.append(dict(model=model,stage=stage,outer=o,inner=i,metric=metric,same_split_cross_seed=float(same[metric].mean()),cross_split_same_seed=float(cross[metric].mean()),difference=float(cross[metric].mean()-same[metric].mean()),same_pair_count=len(same),cross_pair_count=len(cross),same_common_train_mean=float(same.common_train_n.mean()),cross_common_train_mean=float(cross.common_train_n.mean())))
f=pd.DataFrame(rows);f.to_csv(BASE/'seed_split_control_15_splits.csv',index=False)
t=[]
for (model,stage,metric),g in f.groupby(['model','stage','metric']):
 d=g.difference.to_numpy();t.append(dict(model=model,stage=stage,metric=metric,same_mean=g.same_split_cross_seed.mean(),cross_mean=g.cross_split_same_seed.mean(),cross_minus_same=d.mean(),cross_greater=int((d>0).sum()),p=wilcoxon(d).pvalue))
t=pd.DataFrame(t);t['q']=np.nan
for _,ix in t.groupby('metric').groups.items():
 p=t.loc[ix,'p'].to_numpy();order=np.argsort(p);q=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];t.loc[np.asarray(ix)[order],'q']=q
t.to_csv(BASE/'seed_split_control_inference_bh.csv',index=False)
print(t.round(4).to_string(index=False))
