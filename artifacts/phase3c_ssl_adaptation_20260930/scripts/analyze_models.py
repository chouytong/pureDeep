"""E1/E3 fixed inner-development seed-first comparison and retention gates."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import wilcoxon,spearmanr
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score,f1_score,recall_score
BASE=Path('/home/zyt/deep_final/artifacts/phase3c_ssl_adaptation_20260930');P3B=Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928');P2=Path('/home/zyt/deep_final/artifacts/phase2_str_training_20260926')
O=BASE/'analysis';R=np.random.default_rng(20260930);M=['accuracy','ba','auroc','macro_f1','pd_recall','dd_recall']
S={'str':P2/'runs/baseline','frozen':P3B/'runs/b_str_pretrained','adapt':BASE/'runs/adapt/p100','multi':BASE/'runs/multi/p100'}
def measure(f):
 y=f.target.to_numpy(int);p=f.probability_dd.to_numpy(float);pred=(p>.5).astype(int)
 return dict(accuracy=accuracy_score(y,pred),ba=balanced_accuracy_score(y,pred),auroc=roc_auc_score(y,p),macro_f1=f1_score(y,pred,average='macro'),pd_recall=recall_score(y,pred,pos_label=0),dd_recall=recall_score(y,pred,pos_label=1))
rows=[];pairrows=[];complete=[];valkeys={}
for name,root in S.items():
 done=0
 for outer in range(5):
  for inner in range(3):
   f3=[]
   for seed in (42,43,44):
    stage=root/f'seed{seed}'/f'outer_{outer}'/f'inner_{inner}'
    path=stage/'stage_status.json'
    if not path.is_file() or json.loads(path.read_text()).get('status')!='complete':continue
    done+=1;status=json.loads(path.read_text());f=pd.read_csv(stage/'predictions/validation.csv',dtype={'subject_id':str}).sort_values('subject_id').reset_index(drop=True)
    assert f.subject_id.is_unique and set(f.target)=={0,1}
    key=(outer,inner,seed)
    if key in valkeys:assert f[['subject_id','target']].equals(valkeys[key])
    else:valkeys[key]=f[['subject_id','target']]
    h=[json.loads(z) for z in (stage/'logs/epochs.jsonl').read_text().splitlines()];best=status['summary']['best_epoch'];bh=h[best-1]
    rows.append(dict(variant=name,outer=outer,inner=inner,seed=seed,n=len(f),best_epoch=best,epochs_run=len(h),best_train_loss=bh['train']['loss'],best_val_loss=bh['validation']['loss'],best_gap=bh['validation']['loss']-bh['train']['loss'],last_train_loss=h[-1]['train']['loss'],last_val_loss=h[-1]['validation']['loss'],**measure(f)))
    f3.append((seed,f.probability_dd.to_numpy(float)))
   if len(f3)==3:
    for i,j in ((0,1),(0,2),(1,2)):
     a,b=f3[i][1],f3[j][1]
     pairrows.append(dict(variant=name,outer=outer,inner=inner,seed_a=f3[i][0],seed_b=f3[j][0],score_spearman=spearmanr(a,b).statistic,threshold_disagreement=np.mean((a>.5)!=(b>.5))))
 complete.append(dict(variant=name,completed=done,required=45))
 if name in ('str','frozen','adapt') and done!=45:raise RuntimeError(f'{name} {done}/45')
raw=pd.DataFrame(rows);raw.to_csv(O/'model_seed_split_metrics.csv',index=False)
pd.DataFrame(complete).to_csv(O/'model_completion.csv',index=False)
fold=raw.groupby(['variant','outer','inner'],as_index=False)[M].mean();fold.to_csv(O/'model_15split_seedfirst.csv',index=False)
paircor=pd.DataFrame(pairrows);paircor.to_csv(O/'model_seed_prediction_pairs.csv',index=False)
summary=[]
for name,g in fold.groupby('variant'):
 r=raw[raw.variant==name];sm=r.groupby('seed')[M].mean();within=r.groupby(['outer','inner'])[M].std(ddof=1).mean();pc=paircor[paircor.variant==name]
 summary.append(dict(variant=name,**{m:g[m].mean() for m in M},**{m+'_split_sd':g[m].std(ddof=1) for m in M},**{m+'_seed_sd':sm[m].std(ddof=1) for m in M},**{m+'_within_seed_sd':within[m] for m in M},score_spearman=pc.score_spearman.mean(),threshold_disagreement=pc.threshold_disagreement.mean(),best_epoch_median=r.best_epoch.median(),best_train_loss=r.best_train_loss.mean(),best_val_loss=r.best_val_loss.mean(),best_gap=r.best_gap.mean(),last_train_loss=r.last_train_loss.mean(),last_val_loss=r.last_val_loss.mean()))
s=pd.DataFrame(summary);s.to_csv(O/'model_summary.csv',index=False)
def paired(a,b,m):
 x=fold[fold.variant==a].set_index(['outer','inner'])[m].sort_index();y=fold[fold.variant==b].set_index(['outer','inner'])[m].sort_index()
 assert x.index.equals(y.index) and len(x)==15
 d=(x-y).to_numpy();ci=np.quantile(d[R.integers(0,15,size=(10000,15))].mean(axis=1),[.025,.975]);sd=d.std(ddof=1)
 return dict(candidate=a,reference=b,metric=m,mean_delta=d.mean(),positive_splits=int((d>0).sum()),negative_splits=int((d<0).sum()),ci95_low=ci[0],ci95_high=ci[1],cohen_dz=d.mean()/sd if sd else 0,wilcoxon_p=wilcoxon(d).pvalue if np.any(d) else 1)
comps=[]
for a,b in [('adapt','frozen'),('multi','frozen'),('adapt','multi')]:
 if set((a,b)).issubset(set(fold.variant)) and len(fold[fold.variant==a])==15 and len(fold[fold.variant==b])==15:
  for m in M:comps.append(paired(a,b,m))
p=pd.DataFrame(comps);p['q_primary_bh']=np.nan
mask=(p.reference=='frozen')&p.metric.isin(['ba','auroc']);pv=p.loc[mask,'wilcoxon_p'].to_numpy()
if len(pv):
 order=np.argsort(pv);q=np.minimum.accumulate((pv[order]*len(pv)/(np.arange(len(pv))+1))[::-1])[::-1];qq=np.empty_like(q);qq[order]=np.minimum(1,q);p.loc[mask,'q_primary_bh']=qq
p.to_csv(O/'model_paired.csv',index=False)
ss=s.set_index('variant');gates={}
for candidate in ('adapt','multi'):
 if candidate not in ss.index or len(fold[fold.variant==candidate])<15:continue
 c=ss.loc[candidate];b=ss.loc['frozen'];ba=p[(p.candidate==candidate)&(p.reference=='frozen')&(p.metric=='ba')].iloc[0];auc=p[(p.candidate==candidate)&(p.reference=='frozen')&(p.metric=='auroc')].iloc[0]
 gate={'ba_stable':bool(ba.mean_delta>0 and ba.positive_splits>=10 and ba.ci95_low>0),'auroc_noninferior':bool(auc.mean_delta>=0 and auc.ci95_low>-.01),'macro_f1_non_decrease':bool(c.macro_f1>=b.macro_f1),'dd_drop_at_most_001':bool(c.dd_recall>=b.dd_recall-.01),'pd_drop_at_most_001':bool(c.pd_recall>=b.pd_recall-.01),'ba_seed_sd_at_most_125x':bool(c.ba_within_seed_sd<=1.25*b.ba_within_seed_sd),'auroc_seed_sd_at_most_125x':bool(c.auroc_within_seed_sd<=1.25*b.auroc_within_seed_sd),'threshold_disagreement_nonincrease':bool(c.threshold_disagreement<=b.threshold_disagreement)}
 gate['retain']=all(gate.values());gates[candidate]=gate
(O/'model_gates.json').write_text(json.dumps(gates,indent=2))
print(pd.DataFrame(complete).to_string(index=False));print(s[['variant']+M+['ba_within_seed_sd','auroc_within_seed_sd','dd_recall_within_seed_sd','score_spearman','threshold_disagreement','best_gap']].round(4).to_string(index=False));print(p[p.reference=='frozen'][['candidate','metric','mean_delta','positive_splits','ci95_low','ci95_high','cohen_dz','q_primary_bh']].round(4).to_string(index=False));print('gates',gates)
