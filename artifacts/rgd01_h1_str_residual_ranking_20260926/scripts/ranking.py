#!/usr/bin/env python3
"""RGD-01 pairwise AUROC decomposition and subject burden; validation only."""
from pathlib import Path
import csv,gzip,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr,kendalltau,wilcoxon,mannwhitneyu
from sklearn.metrics import roc_auc_score
ROOT=Path('/home/zyt/deep_final');BASE=ROOT/'artifacts/rgd01_h1_str_residual_ranking_20260926';OUT=BASE/'analysis';OUT.mkdir(exist_ok=True)
EM=ROOT/'artifacts/dsg01_cross_subject_disease_subspace_20260926/extraction/embeddings';H1DIR=BASE/'h1_reproduction'
GAIN=ROOT/'artifacts/str01_gain_mechanism_diagnosis_20260923/analysis/validation_records_with_h1.csv'
STATUS=pd.read_csv(GAIN,dtype={'subject_id':str})[['subject_id','str_stability']].drop_duplicates('subject_id').set_index('subject_id').str_stability.to_dict();assert len(STATUS)==390

def load(o,i,s):
 with np.load(H1DIR/f'outer{o}_inner{i}.npz',allow_pickle=False) as h, np.load(EM/f'str_seed{s}_outer{o}_inner{i}_validation.npz',allow_pickle=False) as z, np.load(EM/f'str_seed{s}_outer{o}_inner{i}_train.npz',allow_pickle=False) as t:
  hidx={str(x):j for j,x in enumerate(h['validation_subject_id'])};ids=z['subject_id'].astype(str);assert set(ids)==set(hidx)
  j=np.array([hidx[x] for x in ids]);y=z['label'].astype(int);assert np.array_equal(y,h['validation_label'][j])
  hs=h['validation_h1_probability_dd'][j];ss=z['final_logits'][:,1]-z['final_logits'][:,0]
  ht=np.log(np.clip(h['train_h1_probability_dd'],1e-8,1-1e-8)/(1-np.clip(h['train_h1_probability_dd'],1e-8,1-1e-8)))
  st=t['final_logits'][:,1]-t['final_logits'][:,0]
  return ids,y,hs,ss,ht,st

def pair_arrays(ids,y,h,s):
 pdx=np.flatnonzero(y==0);ddx=np.flatnonzero(y==1);assert len(pdx)>0 and len(ddx)>0
 hm=h[ddx,None]-h[None,pdx];sm=s[ddx,None]-s[None,pdx]
 hc=np.where(hm>0,1.,np.where(hm<0,0.,.5));sc=np.where(sm>0,1.,np.where(sm<0,0.,.5))
 return pdx,ddx,hm,sm,hc,sc

def rank_auc(y,s):return float(roc_auc_score(y,s))
def boot_gap(split_data,B=2000):
 rng=np.random.default_rng(260126);samples=[]
 for _ in range(B):
  values=[]
  for y,h,seed_scores in split_data:
   pdx=np.flatnonzero(y==0);ddx=np.flatnonzero(y==1)
   ix=np.r_[rng.choice(pdx,len(pdx),replace=True),rng.choice(ddx,len(ddx),replace=True)]
   values.append(rank_auc(y[ix],h[ix])-np.mean([rank_auc(y[ix],s[ix]) for s in seed_scores]))
  samples.append(np.mean(values))
 return np.percentile(samples,[2.5,97.5]).tolist()

pair_rows=[];subject_rows=[];split_seed=[];score_rows=[];cache=[]
for o in range(5):
 for i in range(3):
  seed_scores=[];base=None
  for s in (42,43,44):
   ids,y,h,st,ht,str_train=load(o,i,s)
   if base is None:base=(ids.copy(),y.copy(),h.copy())
   else:
    assert np.array_equal(base[0],ids) and np.array_equal(base[1],y) and np.allclose(base[2],h)
   seed_scores.append(st)
   pdx,ddx,hm,sm,hc,sc=pair_arrays(ids,y,h,st)
   both=((hc==1)&(sc==1));wrong=((hc==0)&(sc==0));hres=((hc==1)&(sc==0));sres=((hc==0)&(sc==1));tie=((hc==.5)|(sc==.5))
   assert abs(hc.mean()-rank_auc(y,h))<1e-12 and abs(sc.mean()-rank_auc(y,st))<1e-12
   n=hc.size
   split_seed.append(dict(outer=o,inner=i,seed=s,pd_n=len(pdx),dd_n=len(ddx),pairs=n,h1_auc=hc.mean(),str_auc=sc.mean(),gap=hc.mean()-sc.mean(),both_correct=int(both.sum()),both_wrong=int(wrong.sum()),h1_rescue=int(hres.sum()),str_rescue=int(sres.sum()),ties=int(tie.sum()),h1_rescue_fraction=hres.mean(),str_rescue_fraction=sres.mean(),net_gain_pairs=int(hres.sum()-sres.sum())))
   # Subject participation burden counts each pair once for each participating subject.
   for idx in range(len(ids)):
    if y[idx]==0:
     e_str=1-sc[:,np.where(pdx==idx)[0][0]];e_h=1-hc[:,np.where(pdx==idx)[0][0]];r_h=hres[:,np.where(pdx==idx)[0][0]];r_s=sres[:,np.where(pdx==idx)[0][0]]
    else:
     e_str=1-sc[np.where(ddx==idx)[0][0],:];e_h=1-hc[np.where(ddx==idx)[0][0],:];r_h=hres[np.where(ddx==idx)[0][0],:];r_s=sres[np.where(ddx==idx)[0][0],:]
    subject_rows.append(dict(outer=o,inner=i,seed=s,subject_id=ids[idx],label=int(y[idx]),status=STATUS[ids[idx]],opposite_pairs=len(e_str),str_error_fraction=float(e_str.mean()),h1_error_fraction=float(e_h.mean()),h1_rescue_fraction=float(r_h.mean()),str_rescue_fraction=float(r_s.mean()),net_h1_rescue_fraction=float(r_h.mean()-r_s.mean())))
   for di,d in enumerate(ddx):
    for pi,p in enumerate(pdx):
     category='both_correct' if both[di,pi] else 'both_wrong' if wrong[di,pi] else 'h1_rescue' if hres[di,pi] else 'str_rescue' if sres[di,pi] else 'tie'
     pair_rows.append((o,i,s,ids[d],ids[p],float(hm[di,pi]),float(sm[di,pi]),category,STATUS[ids[d]],STATUS[ids[p]]))
   # Subject-level score agreement and pair-margin agreement.
   hlogit=np.log(np.clip(h,1e-8,1-1e-8)/(1-np.clip(h,1e-8,1-1e-8)))
   hz=(hlogit-ht.mean())/max(ht.std(),1e-8);sz=(st-str_train.mean())/max(str_train.std(),1e-8)
   _,_,hzm,szm,_,_=pair_arrays(ids,y,hz,sz)
   score_rows.append(dict(outer=o,inner=i,seed=s,spearman=float(spearmanr(h,st).statistic),kendall=float(kendalltau(h,st).statistic),pair_order_disagreement=float((hres|sres|tie).mean()),pair_margin_spearman=float(spearmanr(hzm.ravel(),szm.ravel()).statistic),pair_margin_abs_z_difference=float(np.mean(abs(hzm-szm))),subject_threshold_sign_disagreement=float(np.mean((h>=.5)!=(st>=0)))))
  cache.append((base[1],base[2],seed_scores))
seed=pd.DataFrame(split_seed);seed.to_csv(OUT/'pair_decomposition_45_seed_splits.csv',index=False)
seed.groupby(['outer','inner'],as_index=False).mean(numeric_only=True).to_csv(OUT/'pair_decomposition_15_splits.csv',index=False)
pd.DataFrame(score_rows).to_csv(OUT/'score_rank_complementarity_45.csv',index=False)
pd.DataFrame(score_rows).groupby(['outer','inner'],as_index=False).mean(numeric_only=True).to_csv(OUT/'score_rank_complementarity_15.csv',index=False)
sub=pd.DataFrame(subject_rows);sub.to_csv(OUT/'subject_ranking_burden_45.csv',index=False)
sub.groupby(['outer','inner','subject_id','label','status'],as_index=False).mean(numeric_only=True).to_csv(OUT/'subject_ranking_burden_15.csv',index=False)
with gzip.open(OUT/'ranking_pairs.csv.gz','wt',newline='') as f:
 w=csv.writer(f);w.writerow(['outer','inner','seed','dd_subject_id','pd_subject_id','h1_margin_dd_minus_pd','str_margin_dd_minus_pd','category','dd_status','pd_status']);w.writerows(pair_rows)
ci=boot_gap(cache)
summary=seed.groupby(['outer','inner']).mean(numeric_only=True)
np.random.default_rng(260127)
# Split bootstrap is a second, explicitly descriptive sensitivity.
rng=np.random.default_rng(260127);d=summary.gap.to_numpy();split_boot=np.array([d[rng.integers(len(d),size=len(d))].mean() for _ in range(10000)])
(OUT/'ranking_bootstrap.json').write_text(json.dumps({'stratified_subject_paired_bootstrap_mean_gap_ci95':ci,'split_resampling_paired_ci95':np.percentile(split_boot,[2.5,97.5]).tolist(),'mean_gap':float(d.mean()),'B_subject':2000,'B_split':10000,'unit':'PD/DD subjects resampled within each split; 3 STR seeds kept paired and averaged; 15 split means averaged'},indent=2))
print('mean H1/STR/gap',summary.h1_auc.mean(),summary.str_auc.mean(),d.mean(),'CI',ci,flush=True)
