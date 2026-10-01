"""Train-only OOF operating points; 15 split units, no outer/validation tuning."""
import sys,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score,f1_score,recall_score
ROOT=Path('/home/zyt/deep_final/foundation_validation');BASE=Path('/home/zyt/deep_final/artifacts/phase3c_ssl_adaptation_20260930');P3B=Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928')
sys.path.insert(0,str(ROOT))
from src.utils.config import load_config
from src.engine.nested_training import _load_frozen_split
OUT=BASE/'analysis';rng=np.random.default_rng(20260930)
manifest=json.loads((BASE/'manifest.json').read_text());lo,hi,step=manifest['oof_threshold_grid'];grid=np.round(np.arange(lo,hi+step/2,step),2)
split,_,_,sha=_load_frozen_split(load_config(str(ROOT/'configs/str01_seed42.yaml')));assert sha==manifest['split_sha256']
def choose(y,p):
 scores=np.array([balanced_accuracy_score(y,(p>t).astype(int)) for t in grid]);best=np.flatnonzero(scores>=scores.max()-1e-12)
 order=sorted(best,key=lambda i:(abs(grid[i]-.5),grid[i]))
 return float(grid[order[0]]),float(scores.max())
def measure(y,p,t):
 pred=(p>t).astype(int)
 return dict(accuracy=accuracy_score(y,pred),ba=balanced_accuracy_score(y,pred),auroc=roc_auc_score(y,p),macro_f1=f1_score(y,pred,average='macro'),pd_recall=recall_score(y,pred,pos_label=0),dd_recall=recall_score(y,pred,pos_label=1))
M=['accuracy','ba','auroc','macro_f1','pd_recall','dd_recall'];rows=[];ens=[];thresholds=[]
for outer in split['outer']:
 oi=int(outer['outer_fold'])
 for inner in outer['inner_folds']:
  ii=int(inner['inner_fold']);oofparts=[];valparts=[]
  train=set(map(str,inner['train_subjects']));val=set(map(str,inner['validation_subjects']))
  for seed in (42,43,44):
   of=pd.read_csv(BASE/f'oof/seed{seed}/outer_{oi}/inner_{ii}/oof_predictions.csv',dtype={'subject_id':str}).sort_values('subject_id').reset_index(drop=True)
   vf=pd.read_csv(P3B/f'runs/b_str_pretrained/seed{seed}/outer_{oi}/inner_{ii}/predictions/validation.csv',dtype={'subject_id':str}).sort_values('subject_id').reset_index(drop=True)
   assert of.subject_id.is_unique and vf.subject_id.is_unique
   assert set(of.subject_id)==train and set(vf.subject_id)==val and not train&val
   assert (of[['subject_id','target']].equals(oofparts[0][['subject_id','target']]) if oofparts else True)
   assert (vf[['subject_id','target']].equals(valparts[0][['subject_id','target']]) if valparts else True)
   oofparts.append(of);valparts.append(vf)
   t,oofba=choose(of.target.to_numpy(int),of.probability_dd.to_numpy(float))
   y=vf.target.to_numpy(int);p=vf.probability_dd.to_numpy(float)
   for rule,th in [('fixed05',.5),('train_oof',t)]:rows.append(dict(outer=oi,inner=ii,seed=seed,rule=rule,threshold=th,n=len(vf),**measure(y,p,th)))
   thresholds.append(dict(outer=oi,inner=ii,seed=seed,kind='single',threshold=t,oof_ba=oofba,train_n=len(of)))
  oy=oofparts[0].target.to_numpy(int);op=np.mean([f.probability_dd.to_numpy(float) for f in oofparts],axis=0)
  vy=valparts[0].target.to_numpy(int);vp=np.mean([f.probability_dd.to_numpy(float) for f in valparts],axis=0)
  t,oofba=choose(oy,op)
  for rule,th in [('fixed05',.5),('train_oof',t)]:ens.append(dict(outer=oi,inner=ii,rule=rule,threshold=th,n=len(vy),**measure(vy,vp,th)))
  thresholds.append(dict(outer=oi,inner=ii,seed='ensemble',kind='ensemble',threshold=t,oof_ba=oofba,train_n=len(oy)))
raw=pd.DataFrame(rows);raw.to_csv(OUT/'e2_single_seed_split.csv',index=False)
fold=raw.groupby(['outer','inner','rule'],as_index=False)[M].mean();fold.to_csv(OUT/'e2_single_15split_seedfirst.csv',index=False)
e=pd.DataFrame(ens);e.to_csv(OUT/'e2_ensemble_15split.csv',index=False)
pd.DataFrame(thresholds).to_csv(OUT/'e2_train_oof_thresholds.csv',index=False)
paired=[]
for kind,table in [('single',fold),('ensemble',e)]:
 a=table[table.rule=='train_oof'].sort_values(['outer','inner']).reset_index(drop=True);b=table[table.rule=='fixed05'].sort_values(['outer','inner']).reset_index(drop=True)
 assert a[['outer','inner']].equals(b[['outer','inner']]) and len(a)==15
 for m in M:
  d=(a[m]-b[m]).to_numpy();ci=np.quantile(d[rng.integers(0,15,size=(10000,15))].mean(axis=1),[.025,.975]);sd=d.std(ddof=1)
  paired.append(dict(kind=kind,metric=m,fixed05=b[m].mean(),train_oof=a[m].mean(),mean_delta=d.mean(),positive_splits=int((d>0).sum()),negative_splits=int((d<0).sum()),ci95_low=ci[0],ci95_high=ci[1],cohen_dz=d.mean()/sd if sd else 0,wilcoxon_p=wilcoxon(d).pvalue if np.any(d) else 1))
p=pd.DataFrame(paired);p['q_ba_bh']=np.nan;mask=p.metric=='ba';pv=p.loc[mask,'wilcoxon_p'].to_numpy();order=np.argsort(pv);q=np.minimum.accumulate((pv[order]*len(pv)/(np.arange(len(pv))+1))[::-1])[::-1];qq=np.empty_like(q);qq[order]=np.minimum(1,q);p.loc[mask,'q_ba_bh']=qq
p.to_csv(OUT/'e2_threshold_paired.csv',index=False)
print(p.round(4).to_string(index=False));print(pd.DataFrame(thresholds).groupby('kind').threshold.describe().round(4).to_string())
