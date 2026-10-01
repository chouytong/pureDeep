"""Subject-count learning curve; seed-first 15 fixed split units."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score,f1_score,recall_score
BASE=Path('/home/zyt/deep_final/artifacts/phase3c_ssl_adaptation_20260930');P3B=Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928');P2=Path('/home/zyt/deep_final/artifacts/phase2_str_training_20260926');OUT=BASE/'analysis';rng=np.random.default_rng(20260930)
M=['accuracy','ba','auroc','macro_f1','pd_recall','dd_recall']
def measure(f):
 y=f.target.to_numpy(int);p=f.probability_dd.to_numpy(float);pred=(p>.5).astype(int)
 return dict(accuracy=accuracy_score(y,pred),ba=balanced_accuracy_score(y,pred),auroc=roc_auc_score(y,p),macro_f1=f1_score(y,pred,average='macro'),pd_recall=recall_score(y,pred,pos_label=0),dd_recall=recall_score(y,pred,pos_label=1))
rows=[];integrity=[];valkeys={}
for frac in (25,50,75,100):
 for variant in ('str','frozen'):
  root=(P2/'runs/baseline' if variant=='str' else P3B/'runs/b_str_pretrained') if frac==100 else BASE/'runs'/variant/f'p{frac}'
  for outer in range(5):
   for inner in range(3):
    for seed in (42,43,44):
     stage=root/f'seed{seed}'/f'outer_{outer}'/f'inner_{inner}'
     status=json.loads((stage/'stage_status.json').read_text());assert status['status']=='complete' and not status['outer_test_loader_created']
     f=pd.read_csv(stage/'predictions/validation.csv',dtype={'subject_id':str}).sort_values('subject_id').reset_index(drop=True)
     assert f.subject_id.is_unique
     key=(outer,inner,seed)
     if key in valkeys:assert f[['subject_id','target']].equals(valkeys[key])
     else:valkeys[key]=f[['subject_id','target']]
     split=json.loads((stage/'split.json').read_text());train=set(split['train_subjects']);val=set(split['validation_subjects'])
     assert not train&val and set(f.subject_id)==val
     history=[json.loads(z) for z in (stage/'logs/epochs.jsonl').read_text().splitlines()]
     best=status['summary']['best_epoch'];h=history[best-1]
     rows.append(dict(variant=variant,fraction=frac,outer=outer,inner=inner,seed=seed,train_n=len(train),validation_n=len(val),best_epoch=best,best_train_loss=h['train']['loss'],best_val_loss=h['validation']['loss'],**measure(f)))
     integrity.append(dict(variant=variant,fraction=frac,outer=outer,inner=inner,seed=seed,train=sorted(train),validation=sorted(val),normalization_sha256=status['summary']['normalization_sha256']))
assert len(rows)==360
for frac in (25,50,75,100):
 for outer in range(5):
  for inner in range(3):
   group=[x for x in integrity if x['fraction']==frac and x['outer']==outer and x['inner']==inner]
   assert len(group)==6 and all(x['train']==group[0]['train'] and x['validation']==group[0]['validation'] and x['normalization_sha256']==group[0]['normalization_sha256'] for x in group)
for variant in ('str','frozen'):
 for outer in range(5):
  for inner in range(3):
   sets=[]
   for frac in (25,50,75,100):
    x=next(z for z in integrity if z['variant']==variant and z['fraction']==frac and z['outer']==outer and z['inner']==inner)
    sets.append(set(x['train']))
   assert all(a.issubset(b) for a,b in zip(sets,sets[1:]))
raw=pd.DataFrame(rows);raw.to_csv(OUT/'e4_seed_split_metrics.csv',index=False)
fold=raw.groupby(['variant','fraction','outer','inner'],as_index=False)[M+['train_n']].mean();fold.to_csv(OUT/'e4_15split_seedfirst.csv',index=False)
summary=[]
for (variant,frac),g in fold.groupby(['variant','fraction']):
 r=raw[(raw.variant==variant)&(raw.fraction==frac)];within=r.groupby(['outer','inner'])[M].std(ddof=1).mean()
 summary.append(dict(variant=variant,fraction=frac,train_n=g.train_n.mean(),**{m:g[m].mean() for m in M},**{m+'_seed_sd':within[m] for m in M}))
pd.DataFrame(summary).to_csv(OUT/'e4_summary.csv',index=False)
def paired(metric,fa,fb=None):
 def diff(frac):
  a=fold[(fold.variant=='frozen')&(fold.fraction==frac)].set_index(['outer','inner'])[metric].sort_index();b=fold[(fold.variant=='str')&(fold.fraction==frac)].set_index(['outer','inner'])[metric].sort_index()
  assert a.index.equals(b.index) and len(a)==15
  return (a-b).to_numpy()
 d=diff(fa) if fb is None else diff(fa)-diff(fb)
 ci=np.quantile(d[rng.integers(0,15,size=(10000,15))].mean(axis=1),[.025,.975]);sd=d.std(ddof=1)
 return dict(metric=metric,fraction=fa,reference_fraction=fb if fb is not None else '',mean_delta=d.mean(),positive_splits=int((d>0).sum()),ci95_low=ci[0],ci95_high=ci[1],cohen_dz=d.mean()/sd if sd else 0,wilcoxon_p=wilcoxon(d).pvalue if np.any(d) else 1)
comp=[]
for frac in (25,50,75,100):
 for m in M:comp.append(paired(m,frac))
for frac in (25,50,75):
 for m in ('ba','auroc','macro_f1','dd_recall'):comp.append(paired(m,frac,100))
p=pd.DataFrame(comp);p['q_primary_bh']=np.nan
mask=(p.reference_fraction=='')&p.metric.isin(['ba','auroc']);pv=p.loc[mask,'wilcoxon_p'].to_numpy();order=np.argsort(pv);q=np.minimum.accumulate((pv[order]*len(pv)/(np.arange(len(pv))+1))[::-1])[::-1];qq=np.empty_like(q);qq[order]=np.minimum(1,q);p.loc[mask,'q_primary_bh']=qq
p.to_csv(OUT/'e4_paired.csv',index=False)
print(pd.DataFrame(summary)[['variant','fraction','train_n']+M+['ba_seed_sd','auroc_seed_sd','dd_recall_seed_sd']].round(4).to_string(index=False));print(p[p.metric.isin(['ba','auroc','dd_recall'])][['metric','fraction','reference_fraction','mean_delta','positive_splits','ci95_low','ci95_high','q_primary_bh']].round(4).to_string(index=False))
