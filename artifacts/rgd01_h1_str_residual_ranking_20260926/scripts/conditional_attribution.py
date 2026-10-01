#!/usr/bin/env python3
"""Descriptive H1 group contribution among rescued validation pairs; no feature selection."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from scipy.stats import wilcoxon
B=Path('/home/zyt/deep_final/artifacts/rgd01_h1_str_residual_ranking_20260926'); O=B/'analysis'
sys.path.insert(0,'/home/zyt/deep_final/foundation_validation/scripts')
from run_h1_deep_representation_gap_diagnosis import full_estimator,FAMILY_DEFINITIONS
F=Path('/home/zyt/MFAM/outputs/pads_classification/v3_analysis_and_baselines/analysis_and_baselines_20260828/features')
with np.load(F/'handcrafted_features.npz',allow_pickle=False) as z:X=z['X']; ids=z['subject_ids'].astype(str);y=z['labels']
cols=json.loads((F/'feature_schema.json').read_text())['columns'];parts=[c.split('|',3) for c in cols];idx={s:j for j,s in enumerate(ids)}
groups={}
for fam,names in FAMILY_DEFINITIONS.items():groups[('family',fam)]=np.array([k for k,p in enumerate(parts) if p[3] in names],int)
for act in sorted(set(p[0] for p in parts)):groups[('activity',act)]=np.array([k for k,p in enumerate(parts) if p[0]==act],int)
for wr in sorted(set(p[1] for p in parts)):groups[('wrist',wr)]=np.array([k for k,p in enumerate(parts) if p[1]==wr],int)
assert sum(len(v) for (kind,name),v in groups.items() if kind=='family')==len(cols)
pairs=pd.read_csv(O/'ranking_pairs.csv.gz',dtype={'dd_subject_id':str,'pd_subject_id':str});pairs=pairs[pairs.category.isin(['h1_rescue','str_rescue'])]
rows=[];checks=[]
for outer in range(5):
 for inner in range(3):
  with np.load(B/f'h1_reproduction/outer{outer}_inner{inner}.npz',allow_pickle=False) as z: tr=z['train_subject_id'].astype(str);va=z['validation_subject_id'].astype(str);ref=z['validation_h1_probability_dd']
  model=full_estimator().fit(X[[idx[s] for s in tr]],y[[idx[s] for s in tr]])
  im=model.named_steps['imputer']; filt=model.named_steps['variance']; sc=model.named_steps['scaler'];lr=model.named_steps['model']
  xv=sc.transform(filt.transform(im.transform(X[[idx[s] for s in va]])))
  original=np.flatnonzero(filt.get_support());fullcoef=np.zeros(len(cols));fullcoef[original]=lr.coef_[0]; fullz=np.zeros((len(va),len(cols)));fullz[:,original]=xv
  contributions=fullz*fullcoef
  pred=lr.decision_function(xv);assert np.max(np.abs(pred-(contributions.sum(axis=1)+lr.intercept_[0])))<1e-8
  assert np.max(np.abs(1/(1+np.exp(-pred))-ref))<1e-10
  pos={s:j for j,s in enumerate(va)}
  for seed in (42,43,44):
   pp=pairs[(pairs.outer==outer)&(pairs.inner==inner)&(pairs.seed==seed)].copy()
   if len(pp)==0:continue
   di=np.array([pos[s] for s in pp.dd_subject_id]);pi=np.array([pos[s] for s in pp.pd_subject_id])
   for (kind,name),ix in groups.items():
    margin=contributions[di][:,ix].sum(axis=1)-contributions[pi][:,ix].sum(axis=1)
    for cat in ['h1_rescue','str_rescue']:
     sel=pp.category.to_numpy()==cat
     rows.append(dict(outer=outer,inner=inner,seed=seed,kind=kind,group=name,category=cat,pairs=int(sel.sum()),mean_h1_logit_margin_contribution=float(np.mean(margin[sel])) if sel.any() else np.nan))
   checks.append(dict(outer=outer,inner=inner,seed=seed,n_h1_rescue=int((pp.category=='h1_rescue').sum()),n_str_rescue=int((pp.category=='str_rescue').sum())))
r=pd.DataFrame(rows);r.to_csv(O/'conditional_h1_group_contributions_45.csv',index=False)
a=r.groupby(['outer','inner','kind','group','category'],as_index=False).mean(numeric_only=True);a.to_csv(O/'conditional_h1_group_contributions_15.csv',index=False)
wide=a.pivot(index=['outer','inner','kind','group'],columns='category',values='mean_h1_logit_margin_contribution').reset_index()
out=[]
for (kind,name),g in wide.groupby(['kind','group']):
 d=(g.h1_rescue-g.str_rescue).dropna()
 out.append(dict(kind=kind,group=name,splits=len(d),mean_h1_rescue=float(g.h1_rescue.mean()),mean_str_rescue=float(g.str_rescue.mean()),mean_delta=float(d.mean()),wins=int((d>0).sum()),p=float(wilcoxon(d).pvalue) if len(d)>0 and np.any(d) else np.nan))
out=pd.DataFrame(out);out['q_bh_within_kind']=np.nan
for kind,g in out.groupby('kind'):
 p=g.p.fillna(1).to_numpy(); order=np.argsort(p); q=np.empty(len(p));q[order]=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];out.loc[g.index,'q_bh_within_kind']=np.minimum(1,q)
out.sort_values(['kind','q_bh_within_kind']).to_csv(O/'conditional_h1_group_attribution_bh.csv',index=False)
print(out.sort_values(['kind','q_bh_within_kind']).round(4).to_string(index=False))
