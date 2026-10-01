#!/usr/bin/env python3
"""Independent-subject error consistency sensitivity, primary seed-first status."""
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import mannwhitneyu
BASE=Path('/home/zyt/deep_final/artifacts/dsg01_cross_subject_disease_subspace_20260926/analysis')
d=pd.read_csv(BASE/'stable_error_validation_appearances.csv',dtype={'subject_id':str})
metrics=['axis_true_margin','centroid_distance','probe_true_margin','subject_logits_true_margin','final_logits_true_margin']
rows=[]
for (model,sid,label,status),g in d.groupby(['model','subject_id','label','primary_status']):
 row=dict(model=model,subject_id=sid,label=label,primary_status=status,appearances=len(g))
 for m in metrics:
  x=g[m].to_numpy(dtype=float);row[m+'_mean']=x.mean();row[m+'_sd']=x.std(ddof=1);row[m+'_positive_fraction']=np.mean(x>0)
 rows.append(row)
f=pd.DataFrame(rows);assert f.appearances.eq(4).all();f.to_csv(BASE/'error_consistency_subjects.csv',index=False)
tests=[]
for model in ('v8','str'):
 for group in ('stable_correct','unstable'):
  a=f[(f.model==model)&(f.primary_status==group)];b=f[(f.model==model)&(f.primary_status=='stable_error')]
  for metric in ['axis_true_margin_sd','probe_true_margin_sd','axis_true_margin_positive_fraction','probe_true_margin_positive_fraction']:
   x=a[metric].to_numpy();y=b[metric].to_numpy();u,p=mannwhitneyu(y,x,alternative='two-sided')
   tests.append(dict(model=model,reference_group=group,metric=metric,reference_n=len(x),error_n=len(y),reference_mean=x.mean(),error_mean=y.mean(),error_minus_reference=y.mean()-x.mean(),cliffs_delta=2*u/(len(x)*len(y))-1,p=p))
t=pd.DataFrame(tests);t['q']=np.nan
for _,ix in t.groupby(['model','reference_group']).groups.items():
 p=t.loc[ix,'p'].to_numpy();order=np.argsort(p);q=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];t.loc[np.asarray(ix)[order],'q']=q
t.to_csv(BASE/'error_consistency_tests_bh.csv',index=False)
print(t.round(4).to_string(index=False))
