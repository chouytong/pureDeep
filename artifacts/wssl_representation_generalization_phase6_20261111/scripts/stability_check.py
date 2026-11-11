"""Required seed/split stability and descriptive paired-seed rank repeatability."""
from common import *
from itertools import combinations

def main():
 guard();f=pd.read_csv(HERE/'analysis/representation_drift.csv');f=f[f.event=='all'];metrics=['cosine_distance','L2','relative_displacement','frozen_norm','adapted_norm'];rows=[];seeds=[]
 for (layer,label),t in f.groupby(['layer','label']):
  for m in metrics:
   unit=t.groupby(['context','inner'])[m].mean();sm=t.groupby('seed')[m].mean()
   rows.append(dict(layer=layer,label=label,metric=m,seed_first_mean=unit.mean(),split_SD=unit.std(ddof=1),mean_within_split_seed_SD=t.groupby(['context','inner'])[m].std(ddof=1).mean(),SD_of_three_seed_means=sm.std(ddof=1)))
  for seed,t2 in t.groupby('seed'):seeds.append(dict(layer=layer,label=label,seed=seed,**t2[metrics].mean().to_dict()))
 pd.DataFrame(rows).to_csv(HERE/'analysis/representation_seed_split_stability.csv',index=False);pd.DataFrame(seeds).to_csv(HERE/'analysis/representation_per_seed_summary.csv',index=False)
 patient=pd.read_csv(HERE/'restricted_manifests/subject_drift_private.csv',dtype={'subject_id':str});ranks=[]
 for (c,i,layer),t in patient.groupby(['context','inner','layer']):
  for label in [-1,0,1]:
   q=t if label==-1 else t[t.target==label];piv=q.pivot(index='subject_id',columns='seed',values='relative_displacement');assert list(piv.columns)==[42,43,44] and not piv.isna().any().any()
   for a,b in combinations([42,43,44],2):ranks.append(dict(context=c,inner=i,layer=layer,label='all' if label==-1 else ('PD' if label==0 else 'DD'),seed_pair=f'{a}-{b}',subject_count=len(piv),spearman=corr(piv[a],piv[b]),scope='descriptive matched-subject scalar drift ranks; no new hypothesis test or selection'))
 rank=pd.DataFrame(ranks);rank.to_csv(HERE/'analysis/representation_seed_rank_reproducibility.csv',index=False)
 rank.groupby(['context','inner','layer','label']).spearman.mean().groupby(['layer','label']).agg(['mean','std','min','max']).reset_index().to_csv(HERE/'analysis/representation_seed_rank_summary.csv',index=False)
 guard();print('SEED_SPLIT_STABILITY_AND_DESCRIPTIVE_RANK_CHECK_COMPLETE')

if __name__=='__main__':main()
