#!/usr/bin/env python3
"""Analyze only complete Phase-2 inner-development variants, seed-first."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score,f1_score,recall_score
from scipy.stats import wilcoxon
B=Path('/home/zyt/deep_final/artifacts/phase2_str_training_20260926');O=B/'analysis';O.mkdir(exist_ok=True);V=json.loads((B/'manifest.json').read_text())['variants'];rng=np.random.default_rng(20260926)
M=['accuracy','ba','auroc','macro_f1','pd_recall','dd_recall']
def boot(d):
 d=np.asarray(d,float);return np.quantile(d[rng.integers(len(d),size=(10000,len(d)))].mean(1),[.025,.975]).tolist()
rows=[];exposure=[];complete=[]
for variant,v in V.items():
 paths=[B/'runs'/variant/f'seed{s}'/f'outer_{o}'/f'inner_{i}' for s in (42,43,44) for o in range(5) for i in range(3)]
 finished=sum((p/'stage_status.json').is_file() and json.loads((p/'stage_status.json').read_text()).get('status')=='complete' for p in paths)
 complete.append(dict(variant=variant,family=v.get('family','baseline'),complete_units=finished,required_units=45))
 if finished!=45:continue
 for s in (42,43,44):
  for o in range(5):
   for i in range(3):
    p=B/'runs'/variant/f'seed{s}'/f'outer_{o}'/f'inner_{i}';f=pd.read_csv(p/'predictions/validation.csv',dtype={'subject_id':str});y=f.target.to_numpy(int);prob=f.probability_dd.to_numpy(float);pred=f.prediction.to_numpy(int);status=json.loads((p/'stage_status.json').read_text())['summary'];h=[json.loads(z) for z in (p/'logs/epochs.jsonl').read_text().splitlines() if z];e=status['best_epoch'];assert len(f)==f.subject_id.nunique()
    rows.append(dict(variant=variant,family=v.get('family','baseline'),seed=s,outer=o,inner=i,n=len(y),best_epoch=e,epochs_run=len(h),best_train_loss=h[e-1]['train']['loss'],best_val_loss=h[e-1]['validation']['loss'],last_train_loss=h[-1]['train']['loss'],last_val_loss=h[-1]['validation']['loss'],accuracy=accuracy_score(y,pred),ba=balanced_accuracy_score(y,pred),auroc=roc_auc_score(y,prob),macro_f1=f1_score(y,pred,average='macro'),pd_recall=recall_score(y,pred,pos_label=0),dd_recall=recall_score(y,pred,pos_label=1)))
    if 'phase2_exposure' in h[0]['train']:
     for z in h:
      c=z['train']['phase2_exposure'];n=c['pd_exposures']+c['dd_exposures'];a=c['augmented_pd_exposures']+c['augmented_dd_exposures'];exposure.append(dict(variant=variant,seed=s,outer=o,inner=i,epoch=z['epoch']+1,**c,augmented_original_ratio=a/max(1,n-a),dd_exposure_fraction=c['dd_exposures']/n,augmented_fraction=a/n))
pd.DataFrame(complete).to_csv(O/'variant_completion.csv',index=False)
if not rows:raise SystemExit('No complete variants')
raw=pd.DataFrame(rows);raw.to_csv(O/'variant_seed_split_metrics.csv',index=False)
fold=raw.groupby(['variant','outer','inner'],as_index=False)[M].mean();fold.to_csv(O/'variant_15split_seedfirst.csv',index=False)
seed=raw.groupby(['variant','seed'],as_index=False)[M].mean();seed.to_csv(O/'variant_seed_means.csv',index=False)
if exposure:
 exp=pd.DataFrame(exposure);exp.to_csv(O/'augmentation_epoch_exposure.csv',index=False)
 exp.groupby('variant',as_index=False)[['pd_exposures','dd_exposures','augmented_pd_exposures','augmented_dd_exposures','augmented_original_ratio','dd_exposure_fraction','augmented_fraction']].mean().to_csv(O/'augmentation_exposure_summary.csv',index=False)
base=fold[fold.variant=='baseline'].set_index(['outer','inner']);base_seed_sd=seed[seed.variant=='baseline'][M].std(ddof=1)
summary=[];paired=[]
for variant,g in fold.groupby('variant'):
 gs=g.set_index(['outer','inner']);ss=seed[seed.variant==variant][M].std(ddof=1)
 entry=dict(variant=variant,family=V[variant].get('family','baseline'),**{m:gs[m].mean() for m in M},**{m+'_seed_sd':ss[m] for m in M},best_epoch_median=float(raw[raw.variant==variant].best_epoch.median()),best_epoch_max=int(raw[raw.variant==variant].best_epoch.max()))
 summary.append(entry)
 if variant=='baseline':continue
 for m in M:
  d=(gs[m]-base[m]).to_numpy();ci=boot(d);paired.append(dict(variant=variant,family=V[variant].get('family'),metric=m,mean_delta=d.mean(),improved_splits=int(sum(d>0)),worsened_splits=int(sum(d<0)),ci95_low=ci[0],ci95_high=ci[1],seed_sd=ss[m],baseline_seed_sd=base_seed_sd[m],p_wilcoxon=float(wilcoxon(d).pvalue) if np.any(d) else 1.0))
pd.DataFrame(summary).to_csv(O/'variant_summary.csv',index=False);p=pd.DataFrame(paired)
if not p.empty:
 mask=p.metric.isin(['ba','auroc']);idx=np.flatnonzero(mask.to_numpy());pv=p.loc[mask,'p_wilcoxon'].to_numpy();order=np.argsort(pv);q=np.empty(len(pv))
 # Benjamini-Hochberg over all completed primary BA/AUROC comparisons.
 ranked=pv[order]*len(pv)/(np.arange(len(pv))+1);ranked=np.minimum.accumulate(ranked[::-1])[::-1];q[order]=np.minimum(1,ranked);p['q_primary_bh']=np.nan;p.loc[mask,'q_primary_bh']=q
p.to_csv(O/'variant_paired_vs_baseline.csv',index=False)
print('completion');print(pd.DataFrame(complete).to_string(index=False));print('summary');print(pd.DataFrame(summary)[['variant','ba','auroc','macro_f1','accuracy','pd_recall','dd_recall','ba_seed_sd','dd_recall_seed_sd','best_epoch_median']].round(4).to_string(index=False))
if not p.empty:print('primary_paired');print(p[p.metric.isin(['ba','auroc','macro_f1','pd_recall','dd_recall'])][['variant','metric','mean_delta','improved_splits','ci95_low','ci95_high']].round(4).to_string(index=False))
