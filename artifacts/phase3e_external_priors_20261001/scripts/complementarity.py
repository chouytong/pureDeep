"""Conditional E4 subject-ID alignment; pairs are descriptive, split is inferential unit."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
B=Path(__file__).resolve().parents[1];F=Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928/runs/b_str_pretrained');T=B/'runs/biopm';rng=np.random.default_rng(20261001);rows=[];sr=[]
for o in range(5):
 for i in range(3):
  pa=[];pb=[]
  for seed in (42,43,44):
   rel=Path(f'seed{seed}/outer_{o}/inner_{i}/predictions/validation.csv');a=pd.read_csv(F/rel,dtype={'subject_id':str}).sort_values('subject_id').reset_index(drop=True);b=pd.read_csv(T/rel,dtype={'subject_id':str}).sort_values('subject_id').reset_index(drop=True);assert a[['subject_id','target']].equals(b[['subject_id','target']]);y=a.target.to_numpy();p=a.probability_dd.to_numpy();q=b.probability_dd.to_numpy();ca=(p>.5)==y;cb=(q>.5)==y;pa.append(p);pb.append(q)
   rows.append(dict(outer=o,inner=i,seed=seed,spearman=spearmanr(p,q).statistic,disagreement=np.mean((p>.5)!=(q>.5)),frozen_only_correct_fraction=np.mean(ca&~cb),biopm_only_correct_fraction=np.mean(cb&~ca),both_wrong_fraction=np.mean(~ca&~cb),both_correct_fraction=np.mean(ca&cb)))
  p=np.mean(pa,axis=0);q=np.mean(pb,axis=0);a=p[y==1,None]-p[None,y==0];b=q[y==1,None]-q[None,y==0];ca=a>0;cb=b>0;za=rows[-3:]
  sr.append(dict(outer=o,inner=i,pairs=a.size,spearman=spearmanr(p,q).statistic,ranking_disagreement=np.mean(np.sign(a)!=np.sign(b)),frozen_only_rank_fraction=np.mean(ca&~cb),biopm_only_rank_fraction=np.mean(cb&~ca),frozen_only_correct_fraction=np.mean([z['frozen_only_correct_fraction'] for z in za]),biopm_only_correct_fraction=np.mean([z['biopm_only_correct_fraction'] for z in za]),ties_frozen=np.mean(a==0),ties_biopm=np.mean(b==0)))
pd.DataFrame(rows).to_csv(B/'analysis/e4_seed_complementarity.csv',index=False);r=pd.DataFrame(sr);r.to_csv(B/'analysis/e4_15split_complementarity.csv',index=False)
f=pd.read_csv(B/'analysis/model_15split_seedfirst.csv');out={}
for m in ('ba','auroc'):
 w=f.pivot(index=['outer','inner'],columns='variant',values=m);x=(w.frozen-w.str).to_numpy();y=(w.biopm-w.str).to_numpy();rr=spearmanr(x,y).statistic;bs=[]
 for _ in range(10000):
  ii=rng.integers(0,15,15);bs.append(spearmanr(x[ii],y[ii]).statistic)
 out[m+'_split_gain_correlation']={'spearman':rr,'ci95':np.nanquantile(bs,[.025,.975]).tolist()}
out['means']=r.mean(numeric_only=True).to_dict();out['two_sided_correct_complementarity_splits']=int(((r.frozen_only_correct_fraction>0)&(r.biopm_only_correct_fraction>0)).sum());out['two_sided_ranking_complementarity_splits']=int(((r.frozen_only_rank_fraction>=.01)&(r.biopm_only_rank_fraction>=.01)).sum());out['clear_complementarity']=bool(out['two_sided_correct_complementarity_splits']>=10 and out['two_sided_ranking_complementarity_splits']>=10)
s=pd.read_csv(B/'analysis/model_summary.csv').set_index('variant');p=pd.read_csv(B/'analysis/model_paired.csv');z=p[(p.candidate=='biopm')&(p.reference=='frozen')].set_index('metric');c=s.loc['biopm'];a=s.loc['frozen'];out['stable_near_harnet']=bool(all(z.loc[m,'ci95_low']>=-.01 and z.loc[m,'positive_splits']>=7 for m in ('ba','auroc')) and c.macro_f1>=a.macro_f1-.01 and c.dd_recall>=a.dd_recall-.01 and c.ba_within_seed_sd<=1.1*a.ba_within_seed_sd and c.auroc_within_seed_sd<=1.1*a.auroc_within_seed_sd)
control=p[(p.candidate=='biopm')&(p.reference=='biopm_random')].set_index('metric');out['pretrained_beats_random']=bool(len(control)>0 and all(control.loc[m,'ci95_low']>0 and control.loc[m,'positive_splits']>=10 for m in ('ba','auroc')))
out['execute_dual']=bool(out['clear_complementarity'] and out['stable_near_harnet'] and out['pretrained_beats_random']);(B/'analysis/e4_e5_gate.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
