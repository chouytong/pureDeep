#!/usr/bin/env python3
"""Fixed 16-D unsupervised PCA sensitivity; train-only, no parameter search."""
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr,wilcoxon
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
B=Path('/home/zyt/deep_final/artifacts/rgd01_h1_str_residual_ranking_20260926');O=B/'analysis';E=Path('/home/zyt/deep_final/artifacts/dsg01_cross_subject_disease_subspace_20260926/extraction/embeddings')
def logit(p):p=np.clip(p,1e-8,1-1e-8);return np.log(p/(1-p))
rows=[]
for o in range(5):
 for i in range(3):
  with np.load(B/f'h1_reproduction/outer{o}_inner{i}.npz',allow_pickle=False) as h:
   ti={str(x):j for j,x in enumerate(h['train_subject_id'])};vi={str(x):j for j,x in enumerate(h['validation_subject_id'])}
   ht0=logit(h['train_h1_probability_dd']);hv0=logit(h['validation_h1_probability_dd'])
  for s in (42,43,44):
   with np.load(E/f'str_seed{s}_outer{o}_inner{i}_train.npz',allow_pickle=False) as t,np.load(E/f'str_seed{s}_outer{o}_inner{i}_validation.npz',allow_pickle=False) as v:
    tids=t['subject_id'].astype(str);vids=v['subject_id'].astype(str);ht=ht0[[ti[x] for x in tids]];hv=hv0[[vi[x] for x in vids]]
    st=t['final_logits'][:,1]-t['final_logits'][:,0];sv=v['final_logits'][:,1]-v['final_logits'][:,0]
    hm,hs=ht.mean(),max(ht.std(),1e-8);sm,ss=st.mean(),max(st.std(),1e-8)
    target_t=(ht-hm)/hs-(st-sm)/ss;target_v=(hv-hm)/hs-(sv-sm)/ss
    zt=((st-sm)/ss).reshape(-1,1);zv=((sv-sm)/ss).reshape(-1,1)
    stages={'score_only':None,'activity_raw':('activity_raw',),'subject_embedding':('subject_embedding',),'structured_embedding':('structured_embedding',),'decision_input':('subject_embedding','structured_embedding')}
    for stage,keys in stages.items():
     if keys is None:A,V=zt,zv
     else:
      Xt=np.concatenate([t[k].reshape(len(tids),-1) for k in keys],axis=1);Xv=np.concatenate([v[k].reshape(len(vids),-1) for k in keys],axis=1)
      scale=StandardScaler().fit(Xt);pc=PCA(n_components=16,svd_solver='randomized',random_state=0).fit(scale.transform(Xt))
      A=np.column_stack([zt,pc.transform(scale.transform(Xt))]);V=np.column_stack([zv,pc.transform(scale.transform(Xv))])
     pred=Ridge(alpha=1.0).fit(A,target_t).predict(V)
     rows.append(dict(outer=o,inner=i,seed=s,stage=stage,r2=r2_score(target_v,pred),mae=mean_absolute_error(target_v,pred),spearman=spearmanr(target_v,pred).statistic))
r=pd.DataFrame(rows);r.to_csv(O/'residual_pca16_45.csv',index=False)
a=r.groupby(['outer','inner','stage'],as_index=False).mean(numeric_only=True);a.to_csv(O/'residual_pca16_15.csv',index=False)
b=a[a.stage=='score_only'].set_index(['outer','inner']);out=[]
for stage in ('activity_raw','subject_embedding','structured_embedding','decision_input'):
 g=a[a.stage==stage].set_index(['outer','inner'])
 for metric,sign in [('r2',1),('mae',-1),('spearman',1)]:
  d=sign*(g[metric]-b[metric]);out.append(dict(stage=stage,metric=metric,baseline_mean=b[metric].mean(),stage_mean=g[metric].mean(),mean_improvement=d.mean(),wins=int((d>0).sum()),p=wilcoxon(d).pvalue))
out=pd.DataFrame(out);p=out.p.to_numpy();order=np.argsort(p);q=np.empty(len(p));q[order]=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];out['q_bh']=np.minimum(1,q)
out.to_csv(O/'residual_pca16_inference_bh.csv',index=False);print(out.round(4).to_string(index=False))
