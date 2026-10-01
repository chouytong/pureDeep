"""15-split paired seed-variance and prediction-disagreement diagnostics."""
from pathlib import Path
import numpy as np,pandas as pd
B=Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928')
O=B/'analysis';rng=np.random.default_rng(20260928)
raw=pd.read_csv(O/'phase3b_seed_split_metrics.csv')
sp=pd.read_csv(O/'phase3b_seed_pair_disagreement.csv')
sd=raw.groupby(['variant','outer','inner'],as_index=False)[['ba','auroc','dd_recall']].std(ddof=1).rename(columns={m:m+'_seed_sd' for m in ('ba','auroc','dd_recall')})
s=sp.groupby(['variant','outer','inner'],as_index=False)[['score_spearman','threshold_disagreement']].mean()
f=sd.merge(s,on=['variant','outer','inner'])
f.to_csv(O/'phase3b_split_seed_stability.csv',index=False)
rows=[]
for ref in ('baseline','c_str_random'):
 a=f[f.variant=='b_str_pretrained'].set_index(['outer','inner']).sort_index();b=f[f.variant==ref].set_index(['outer','inner']).sort_index()
 assert a.index.equals(b.index) and len(a)==15
 for m in ['ba_seed_sd','auroc_seed_sd','dd_recall_seed_sd','score_spearman','threshold_disagreement']:
  d=(a[m]-b[m]).to_numpy();ci=np.quantile(d[rng.integers(0,15,size=(10000,15))].mean(axis=1),[.025,.975])
  rows.append(dict(candidate='b_str_pretrained',reference=ref,metric=m,candidate_mean=a[m].mean(),reference_mean=b[m].mean(),mean_delta=d.mean(),positive_splits=int((d>0).sum()),ci95_low=ci[0],ci95_high=ci[1]))
pd.DataFrame(rows).to_csv(O/'phase3b_seed_stability_paired.csv',index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))
