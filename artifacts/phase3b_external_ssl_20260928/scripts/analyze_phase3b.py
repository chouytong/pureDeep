"""Phase-3B frozen-SSL fixed inner development analysis; no outer artifacts."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import wilcoxon,spearmanr
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score,f1_score,recall_score
B=Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928')
BASE=Path('/home/zyt/deep_final/artifacts/phase2_str_training_20260926/runs/baseline')
O=B/'analysis';O.mkdir(exist_ok=True)
M=['accuracy','ba','auroc','macro_f1','pd_recall','dd_recall']
ROOTS={'baseline':BASE,**{n:B/'runs'/n for n in json.loads((B/'manifest.json').read_text())['variants']}}
R=np.random.default_rng(20260928)
def metrics(y,p):
 pred=(p>.5).astype(int)
 return dict(accuracy=accuracy_score(y,pred),ba=balanced_accuracy_score(y,pred),auroc=roc_auc_score(y,p),macro_f1=f1_score(y,pred,average='macro'),pd_recall=recall_score(y,pred,pos_label=0),dd_recall=recall_score(y,pred,pos_label=1))
rows=[];saved={};completion=[];seedpairs=[];ensembles=[]
for name,root in ROOTS.items():
 done=0
 for seed in (42,43,44):
  for outer in range(5):
   for inner in range(3):
    stage=root/f'seed{seed}'/f'outer_{outer}'/f'inner_{inner}'
    status=stage/'stage_status.json'
    if not status.is_file() or json.loads(status.read_text()).get('status')!='complete':continue
    done+=1
    f=pd.read_csv(stage/'predictions/validation.csv',dtype={'subject_id':str}).sort_values('subject_id').reset_index(drop=True)
    assert f.subject_id.is_unique and set(f.target.unique())=={0,1}
    key=(outer,inner,seed)
    common=f[['subject_id','target']]
    if key in saved:assert common.equals(saved[key][0]),(name,key,'ID or label mismatch')
    else:saved[key]=(common,None)
    p=f.probability_dd.to_numpy(float);y=f.target.to_numpy(int)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    if name=='baseline':saved[key]=(common,p)
    hist=[json.loads(z) for z in (stage/'logs/epochs.jsonl').read_text().splitlines()]
    best=json.loads(status.read_text())['summary']['best_epoch']
    bm=hist[best-1]
    rows.append(dict(variant=name,outer=outer,inner=inner,seed=seed,n=len(y),best_epoch=best,epochs_run=len(hist),train_loss=bm['train']['loss'],validation_loss=bm['validation']['loss'],**metrics(y,p)))
    saved[(name,outer,inner,seed)]=(common,p)
 completion.append(dict(variant=name,completed=done,required=45))
 if done!=45:raise RuntimeError(f'{name}: only {done}/45 completed')
 for outer in range(5):
  for inner in range(3):
   fs=[saved[(name,outer,inner,s)] for s in (42,43,44)]
   assert all(fs[0][0].equals(x[0]) for x in fs[1:])
   probs=np.stack([x[1] for x in fs]);y=fs[0][0].target.to_numpy(int)
   ens=probs.mean(axis=0)
   ensembles.append(dict(variant=name,outer=outer,inner=inner,n=len(y),**metrics(y,ens)))
   for a,b in ((0,1),(0,2),(1,2)):
    corr=spearmanr(probs[a],probs[b]).statistic
    seedpairs.append(dict(variant=name,outer=outer,inner=inner,seed_a=(42,43,44)[a],seed_b=(42,43,44)[b],score_spearman=corr,threshold_disagreement=np.mean((probs[a]>.5)!=(probs[b]>.5))))
raw=pd.DataFrame(rows);raw.to_csv(O/'phase3b_seed_split_metrics.csv',index=False)
pd.DataFrame(completion).to_csv(O/'phase3b_completion.csv',index=False)
fold=raw.groupby(['variant','outer','inner'],as_index=False)[M].mean();fold.to_csv(O/'phase3b_15split_seedfirst.csv',index=False)
seedmean=raw.groupby(['variant','seed'],as_index=False)[M].mean();seedmean.to_csv(O/'phase3b_seed_means.csv',index=False)
summary=[]
for name,g in fold.groupby('variant'):
 r=raw[raw.variant==name];sm=seedmean[seedmean.variant==name]
 within=r.groupby(['outer','inner'])[M].std(ddof=1).mean()
 summary.append(dict(variant=name,**{m:g[m].mean() for m in M},**{m+'_split_sd':g[m].std(ddof=1) for m in M},**{m+'_seed_sd':sm[m].std(ddof=1) for m in M},**{m+'_within_split_seed_sd':within[m] for m in M},best_epoch_median=r.best_epoch.median(),train_loss=r.train_loss.mean(),validation_loss=r.validation_loss.mean()))
pd.DataFrame(summary).to_csv(O/'phase3b_summary.csv',index=False)
def paired(a,b,m,ensemble=False):
 table=pd.DataFrame(ensembles) if ensemble else fold
 aa=table[table.variant==a].set_index(['outer','inner'])[m].sort_index();bb=table[table.variant==b].set_index(['outer','inner'])[m].sort_index()
 assert len(aa)==len(bb)==15 and aa.index.equals(bb.index)
 d=(aa-bb).to_numpy();ci=np.quantile(d[R.integers(0,15,size=(10000,15))].mean(axis=1),[.025,.975]);sd=d.std(ddof=1)
 return dict(candidate=a,reference=b,metric=m,mean_delta=d.mean(),positive_splits=int((d>0).sum()),negative_splits=int((d<0).sum()),ci95_low=ci[0],ci95_high=ci[1],cohen_dz=d.mean()/sd if sd else 0,wilcoxon_p=wilcoxon(d).pvalue if np.any(d) else 1)
comparisons=[]
for a,b in [('a_ssl_only_pretrained','baseline'),('b_str_pretrained','baseline'),('c_str_random','baseline'),('b_str_pretrained','c_str_random')]:
 for m in M:comparisons.append(paired(a,b,m))
pair=pd.DataFrame(comparisons);pair['q_primary_bh']=np.nan
mask=pair.metric.isin(['ba','auroc']);pv=pair.loc[mask,'wilcoxon_p'].to_numpy();order=np.argsort(pv);q=np.minimum.accumulate((pv[order]*len(pv)/(np.arange(len(pv))+1))[::-1])[::-1];ordered=np.empty_like(q);ordered[order]=np.minimum(1,q);pair.loc[mask,'q_primary_bh']=ordered
pair.to_csv(O/'phase3b_paired.csv',index=False)
ens=pd.DataFrame(ensembles);ens.to_csv(O/'phase3b_ensemble_split_metrics.csv',index=False)
e2=pd.read_csv('/home/zyt/deep_final/artifacts/phase3a_puredeep_20260928/analysis/e2_ensemble_split_metrics.csv').sort_values(['outer','inner']).reset_index(drop=True)
e2_current=ens[ens.variant=='baseline'].sort_values(['outer','inner']).reset_index(drop=True)
assert (e2[['outer','inner']]==e2_current[['outer','inner']]).all().all()
assert np.allclose(e2[M].to_numpy(),e2_current[M].to_numpy(),atol=1e-12,rtol=0)
ens.groupby('variant',as_index=False)[M].mean().to_csv(O/'phase3b_ensemble_summary.csv',index=False)
ens_pairs=[paired(a,'baseline',m,True) for a in ROOTS if a!='baseline' for m in M]
pd.DataFrame(ens_pairs).to_csv(O/'phase3b_ensemble_paired.csv',index=False)
sp=pd.DataFrame(seedpairs);sp.to_csv(O/'phase3b_seed_pair_disagreement.csv',index=False)
sp.groupby('variant',as_index=False)[['score_spearman','threshold_disagreement']].mean().to_csv(O/'phase3b_seed_disagreement_summary.csv',index=False)
s=pd.DataFrame(summary).set_index('variant'); b='b_str_pretrained';c='c_str_random';base='baseline'
gate={}
for ref in (base,c):
 for m in ('ba','auroc'):
  z=pair[(pair.candidate==b)&(pair.reference==ref)&(pair.metric==m)].iloc[0]
  gate[f'{ref}_{m}']=bool(z.mean_delta>0 and z.positive_splits>=10 and z.ci95_low>0)
gate['macro_f1_non_decrease']=bool(s.loc[b,'macro_f1']>=s.loc[base,'macro_f1'])
gate['dd_recall_non_decrease']=bool(s.loc[b,'dd_recall']>=s.loc[base,'dd_recall'])
gate['pd_recall_drop_at_most_001']=bool(s.loc[b,'pd_recall']>=s.loc[base,'pd_recall']-.01)
for m in ('ba','auroc'):gate[f'{m}_seed_sd_at_most_125x']=bool(s.loc[b,m+'_seed_sd']<=1.25*s.loc[base,m+'_seed_sd'])
gate['allow_phase3c']=all(gate.values())
(O/'phase3b_gate.json').write_text(json.dumps(gate,indent=2))
print(pd.DataFrame(completion).to_string(index=False));print(pd.DataFrame(summary)[['variant']+M+['ba_seed_sd','auroc_seed_sd','dd_recall_seed_sd']].round(4).to_string(index=False));print(pair[['candidate','reference','metric','mean_delta','positive_splits','ci95_low','ci95_high','q_primary_bh']].round(4).to_string(index=False));print('gate',gate)
