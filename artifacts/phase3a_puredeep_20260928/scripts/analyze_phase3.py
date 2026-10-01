"""Seed-first paired Phase-3A inner-development analysis; no outer-final artifacts."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score,f1_score,recall_score
B=Path('/home/zyt/deep_final/artifacts/phase3a_puredeep_20260928')
P2=Path('/home/zyt/deep_final/artifacts/phase2_str_training_20260926')
O=B/'analysis';O.mkdir(exist_ok=True)
R=np.random.default_rng(20260928)
SOURCES={'baseline':P2/'runs/baseline','e1_balanced_balanced':P2/'runs/balanced_sampler',**{n:B/'runs'/n for n in json.loads((B/'manifest.json').read_text())['variants']}}
M=['accuracy','ba','auroc','macro_f1','pd_recall','dd_recall']
rows=[];complete=[];exposures=[]
for name,root in SOURCES.items():
 paths=[root/f'seed{seed}'/f'outer_{o}'/f'inner_{i}' for seed in (42,43,44) for o in range(5) for i in range(3)]
 done=sum((p/'stage_status.json').is_file() and json.loads((p/'stage_status.json').read_text())['status']=='complete' for p in paths)
 complete.append(dict(variant=name,completed=done,required=45))
 if done<45:continue
 for seed in (42,43,44):
  for o in range(5):
   for i in range(3):
    p=root/f'seed{seed}'/f'outer_{o}'/f'inner_{i}'
    f=pd.read_csv(p/'predictions/validation.csv',dtype={'subject_id':str});y=f.target.to_numpy(int);prob=f.probability_dd.to_numpy(float);pred=f.prediction.to_numpy(int)
    assert f.subject_id.is_unique
    history=[json.loads(z) for z in (p/'logs/epochs.jsonl').read_text().splitlines()]
    best=json.loads((p/'stage_status.json').read_text())['summary']['best_epoch']
    h=history[best-1];last=history[-1]
    rows.append(dict(variant=name,seed=seed,outer=o,inner=i,n=len(y),best_epoch=best,epochs_run=len(history),best_train_loss=h['train']['loss'],best_train_ce=h['train'].get('first_pass_classification_loss',h['train']['classification_loss']),best_val_ce=h['validation']['classification_loss'],last_train_loss=last['train']['loss'],last_train_ce=last['train'].get('first_pass_classification_loss',last['train']['classification_loss']),last_val_ce=last['validation']['classification_loss'],best_gap_ce=h['validation']['classification_loss']-h['train'].get('first_pass_classification_loss',h['train']['classification_loss']),last_gap_ce=last['validation']['classification_loss']-last['train'].get('first_pass_classification_loss',last['train']['classification_loss']),accuracy=accuracy_score(y,pred),ba=balanced_accuracy_score(y,pred),auroc=roc_auc_score(y,prob),macro_f1=f1_score(y,pred,average='macro'),pd_recall=recall_score(y,pred,pos_label=0),dd_recall=recall_score(y,pred,pos_label=1)))
    for z in history:
     c=z['train'].get('phase2_exposure')
     if c:exposures.append(dict(variant=name,seed=seed,outer=o,inner=i,epoch=z['epoch']+1,**c))
pd.DataFrame(complete).to_csv(O/'phase3_completion.csv',index=False)
if not rows:raise SystemExit('No complete condition')
raw=pd.DataFrame(rows);raw.to_csv(O/'phase3_seed_split_metrics.csv',index=False)
fold=raw.groupby(['variant','outer','inner'],as_index=False)[M].mean();fold.to_csv(O/'phase3_15split_seedfirst.csv',index=False)
seedmean=raw.groupby(['variant','seed'],as_index=False)[M].mean();seedmean.to_csv(O/'phase3_seed_means.csv',index=False)
summary=[]
for name,g in fold.groupby('variant'):
 ss=seedmean[seedmean.variant==name][M].std(ddof=1);r=raw[raw.variant==name]
 summary.append(dict(variant=name,**{m:g[m].mean() for m in M},**{m+'_split_sd':g[m].std(ddof=1) for m in M},**{m+'_seed_sd':ss[m] for m in M},best_epoch_mean=r.best_epoch.mean(),best_epoch_median=r.best_epoch.median(),best_train_ce=r.best_train_ce.mean(),best_val_ce=r.best_val_ce.mean(),last_train_ce=r.last_train_ce.mean(),last_val_ce=r.last_val_ce.mean(),best_gap_ce=r.best_gap_ce.mean(),last_gap_ce=r.last_gap_ce.mean()))
pd.DataFrame(summary).to_csv(O/'phase3_summary.csv',index=False)
if exposures:
 exp=pd.DataFrame(exposures);exp.to_csv(O/'e1_epoch_exposures.csv',index=False);exp.groupby('variant',as_index=False)[['pd_exposures','dd_exposures','augmented_pd_exposures','augmented_dd_exposures']].mean().to_csv(O/'e1_exposure_summary.csv',index=False)
def paired(a,b,metric):
 aa=fold[fold.variant==a].set_index(['outer','inner'])[metric];bb=fold[fold.variant==b].set_index(['outer','inner'])[metric]
 if len(aa)!=15 or len(bb)!=15:return None
 d=(aa-bb).to_numpy();ci=np.quantile(d[R.integers(0,15,size=(10000,15))].mean(1),[.025,.975]);sd=d.std(ddof=1)
 return dict(candidate=a,reference=b,metric=metric,mean_delta=d.mean(),positive_splits=int((d>0).sum()),negative_splits=int((d<0).sum()),ci95_low=ci[0],ci95_high=ci[1],cohen_dz=d.mean()/sd if sd else 0,wilcoxon_p=wilcoxon(d).pvalue if np.any(d) else 1)
comparisons=[]
for a in SOURCES:
 if a!='baseline':
  for m in M:
   z=paired(a,'baseline',m)
   if z:comparisons.append(z)
for m in M:
 z=paired('e1_balanced_unweighted','e1_balanced_balanced',m)
 if z:comparisons.append(z)
p=pd.DataFrame(comparisons)
if not p.empty:
 p['q_primary_bh']=np.nan;mask=(p.reference=='baseline') & p.metric.isin(['ba','auroc']);pv=p.loc[mask,'wilcoxon_p'].to_numpy()
 if len(pv):
  order=np.argsort(pv);ranked=np.minimum.accumulate((pv[order]*len(pv)/(np.arange(len(pv))+1))[::-1])[::-1];q=np.empty_like(ranked);q[order]=np.minimum(1,ranked);p.loc[mask,'q_primary_bh']=q
 p.to_csv(O/'phase3_paired.csv',index=False)
print('completion');print(pd.DataFrame(complete).to_string(index=False));print('summary');print(pd.DataFrame(summary)[['variant']+M+['ba_seed_sd','auroc_seed_sd','dd_recall_seed_sd','best_epoch_median','best_gap_ce','last_gap_ce']].round(4).to_string(index=False));print('paired');print(p[p.metric.isin(['ba','auroc','macro_f1','pd_recall','dd_recall'])][['candidate','reference','metric','mean_delta','positive_splits','ci95_low','ci95_high','cohen_dz','q_primary_bh']].round(4).to_string(index=False))
