"""E2: seed-aligned probability ensemble, fixed inner-development only."""
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score,f1_score,recall_score
B=Path('/home/zyt/deep_final/artifacts/phase2_str_training_20260926/runs/baseline')
O=Path('/home/zyt/deep_final/artifacts/phase3a_puredeep_20260928/analysis');O.mkdir(exist_ok=True)
R=np.random.default_rng(20260928)
metrics=['accuracy','ba','auroc','macro_f1','pd_recall','dd_recall']
def measure(f):
 y=f.target.to_numpy(int);p=f.probability_dd.to_numpy(float);pred=(p>.5).astype(int)
 return dict(accuracy=accuracy_score(y,pred),ba=balanced_accuracy_score(y,pred),auroc=roc_auc_score(y,p),macro_f1=f1_score(y,pred,average='macro'),pd_recall=recall_score(y,pred,pos_label=0),dd_recall=recall_score(y,pred,pos_label=1))
rows=[];base=[];subjects=[]
for o in range(5):
 for i in range(3):
  parts=[]
  for seed in (42,43,44):
   p=B/f'seed{seed}'/f'outer_{o}'/f'inner_{i}'/'predictions/validation.csv'
   f=pd.read_csv(p,dtype={'subject_id':str}).sort_values('subject_id').reset_index(drop=True)
   assert f.subject_id.is_unique
   parts.append(f)
   base.append(dict(outer=o,inner=i,seed=seed,**measure(f)))
  common=parts[0][['subject_id','target']]
  assert all(common.equals(f[['subject_id','target']]) for f in parts[1:])
  e=common.copy();e['probability_dd']=np.mean([f.probability_dd.to_numpy(float) for f in parts],axis=0)
  rows.append(dict(outer=o,inner=i,n=len(e),**measure(e)))
  e.insert(0,'inner',i);e.insert(0,'outer',o);subjects.append(e)
pd.DataFrame(subjects[0]).head(0)
sub=pd.concat(subjects,ignore_index=True);sub.to_csv(O/'e2_ensemble_subject_predictions.csv',index=False)
b=pd.DataFrame(base);e=pd.DataFrame(rows);b.to_csv(O/'e2_seed_split_metrics.csv',index=False);e.to_csv(O/'e2_ensemble_split_metrics.csv',index=False)
fold=b.groupby(['outer','inner'],as_index=False)[metrics].mean();fold.to_csv(O/'e2_seed_mean_split_metrics.csv',index=False)
paired=[]
for m in metrics:
 d=e[m].to_numpy()-fold[m].to_numpy();ci=np.quantile(d[R.integers(0,15,size=(10000,15))].mean(1),[.025,.975]);sd=d.std(ddof=1)
 paired.append(dict(metric=m,seed_metric_mean=fold[m].mean(),ensemble_metric_mean=e[m].mean(),mean_delta=d.mean(),positive_splits=int((d>0).sum()),negative_splits=int((d<0).sum()),ci95_low=ci[0],ci95_high=ci[1],cohen_dz=d.mean()/sd if sd else 0,seed_metric_split_sd=fold[m].std(ddof=1),ensemble_split_sd=e[m].std(ddof=1),wilcoxon_p=wilcoxon(d).pvalue if np.any(d) else 1))
p=pd.DataFrame(paired);p['q_primary_bh']=np.nan;idx=p.metric.isin(['ba','auroc']);pv=p.loc[idx,'wilcoxon_p'].to_numpy();order=np.argsort(pv);q=np.minimum.accumulate((pv[order]*len(pv)/(np.arange(len(pv))+1))[::-1])[::-1];result=np.empty_like(q);result[order]=np.minimum(q,1);p.loc[idx,'q_primary_bh']=result;p.to_csv(O/'e2_ensemble_paired.csv',index=False)
print(p.round(4).to_string(index=False))
