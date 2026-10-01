import sys,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr,wilcoxon
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score,f1_score,recall_score
B=Path(__file__).resolve().parents[1];O=B/'analysis';rng=np.random.default_rng(20261001)
F=Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928/runs/b_str_pretrained');L=Path('/home/zyt/deep_final/artifacts/phase3c_ssl_adaptation_20260930/runs/adapt/p100')
M=['accuracy','ba','auroc','macro_f1','pd_recall','dd_recall']
def metric(y,p):
 h=(p>.5).astype(int)
 return dict(accuracy=accuracy_score(y,h),ba=balanced_accuracy_score(y,h),auroc=roc_auc_score(y,p),macro_f1=f1_score(y,h,average='macro'),pd_recall=recall_score(y,h,pos_label=0),dd_recall=recall_score(y,h,pos_label=1))
rows=[];comp=[]
for o in range(5):
 for i in range(3):
  fa=[];la=[]
  for seed in (42,43,44):
   rel=Path(f'seed{seed}/outer_{o}/inner_{i}/predictions/validation.csv')
   a=pd.read_csv(F/rel,dtype={'subject_id':str}).sort_values('subject_id').reset_index(drop=True);b=pd.read_csv(L/rel,dtype={'subject_id':str}).sort_values('subject_id').reset_index(drop=True)
   assert a[['subject_id','target']].equals(b[['subject_id','target']]) and a.subject_id.is_unique
   y=a.target.to_numpy();p=a.probability_dd.to_numpy();q=b.probability_dd.to_numpy();fa.append(p);la.append(q)
   for name,z in [('frozen_single',p),('lastblock_single',q),('equal_mix_per_seed',(p+q)/2)]:rows.append(dict(model=name,outer=o,inner=i,seed=seed,**metric(y,z)))
   ca=(p>.5)==y;cb=(q>.5)==y
   comp.append(dict(outer=o,inner=i,seed=seed,n=len(y),spearman=spearmanr(p,q).statistic,disagreement=np.mean((p>.5)!=(q>.5)),frozen_only_correct=int((ca&~cb).sum()),lastblock_only_correct=int((cb&~ca).sum()),both_correct=int((ca&cb).sum()),both_wrong=int((~ca&~cb).sum())))
  for name,z in [('frozen_3seed_ensemble',np.mean(fa,axis=0)),('lastblock_3seed_ensemble',np.mean(la,axis=0)),('equal_6model_ensemble',(np.mean(fa,axis=0)+np.mean(la,axis=0))/2)]:rows.append(dict(model=name,outer=o,inner=i,seed=0,**metric(y,z)))
r=pd.DataFrame(rows);r.to_csv(O/'e0_seed_metrics.csv',index=False);f=r.groupby(['model','outer','inner'],as_index=False)[M].mean();f.to_csv(O/'e0_15split.csv',index=False);s=f.groupby('model')[M].mean();s.to_csv(O/'e0_summary.csv');pd.DataFrame(comp).to_csv(O/'e0_complementarity.csv',index=False)
pr=[]
for a,b in [('lastblock_single','frozen_single'),('equal_mix_per_seed','frozen_single'),('frozen_3seed_ensemble','frozen_single'),('equal_6model_ensemble','frozen_3seed_ensemble'),('equal_6model_ensemble','lastblock_3seed_ensemble')]:
 for m in M:
  x=f[f.model==a].set_index(['outer','inner'])[m];y=f[f.model==b].set_index(['outer','inner'])[m];d=(x-y).to_numpy();ci=np.quantile(d[rng.integers(0,15,(10000,15))].mean(1),[.025,.975]);pr.append(dict(candidate=a,reference=b,metric=m,delta=d.mean(),wins=int((d>0).sum()),ci_low=ci[0],ci_high=ci[1],dz=d.mean()/d.std(ddof=1),p=wilcoxon(d).pvalue if np.any(d) else 1))
p=pd.DataFrame(pr);pv=p.p.to_numpy();idx=np.argsort(pv);q=np.minimum.accumulate((pv[idx]*len(pv)/np.arange(1,len(pv)+1))[::-1])[::-1];qq=np.empty_like(q);qq[idx]=np.minimum(q,1);p['bh_q']=qq;p.to_csv(O/'e0_paired.csv',index=False)
print(s.round(4).to_string());print(pd.DataFrame(comp).groupby(['outer','inner']).mean().mean().to_string());print(p[p.metric.isin(['ba','auroc'])].round(4).to_string(index=False))
