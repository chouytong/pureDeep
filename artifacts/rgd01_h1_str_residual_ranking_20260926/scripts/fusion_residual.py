#!/usr/bin/env python3
"""Train-only score fusion and frozen-STR residual score accessibility probes."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy.stats import wilcoxon,spearmanr
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
ROOT=Path('/home/zyt/deep_final');BASE=ROOT/'artifacts/rgd01_h1_str_residual_ranking_20260926';OUT=BASE/'analysis';OUT.mkdir(exist_ok=True)
EM=ROOT/'artifacts/dsg01_cross_subject_disease_subspace_20260926/extraction/embeddings';H1DIR=BASE/'h1_reproduction'
def logit(x):
 x=np.clip(x,1e-8,1-1e-8);return np.log(x/(1-x))
def metrics(y,score):return float(roc_auc_score(y,score)),float(balanced_accuracy_score(y,score>=0.5))
def load(o,i,s):
 with np.load(H1DIR/f'outer{o}_inner{i}.npz',allow_pickle=False) as h,np.load(EM/f'str_seed{s}_outer{o}_inner{i}_train.npz',allow_pickle=False) as t,np.load(EM/f'str_seed{s}_outer{o}_inner{i}_validation.npz',allow_pickle=False) as v:
  ti={str(x):j for j,x in enumerate(h['train_subject_id'])};vi={str(x):j for j,x in enumerate(h['validation_subject_id'])}
  t_ids=t['subject_id'].astype(str);v_ids=v['subject_id'].astype(str);assert set(t_ids)==set(ti) and set(v_ids)==set(vi)
  tr_y=t['label'].astype(int);va_y=v['label'].astype(int)
  assert np.array_equal(tr_y,h['train_label'][[ti[x] for x in t_ids]]) and np.array_equal(va_y,h['validation_label'][[vi[x] for x in v_ids]])
  ht=logit(h['train_h1_probability_dd'][[ti[x] for x in t_ids]]);hv=logit(h['validation_h1_probability_dd'][[vi[x] for x in v_ids]])
  st=t['final_logits'][:,1]-t['final_logits'][:,0];sv=v['final_logits'][:,1]-v['final_logits'][:,0]
  tr={k:t[k] for k in t.files};va={k:v[k] for k in v.files}
  return t_ids,v_ids,tr_y,va_y,ht,hv,st,sv,tr,va

def bh(p):
 p=np.asarray(p,float);idx=np.argsort(p);q=np.empty(len(p));q[idx]=np.minimum.accumulate((p[idx]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];return np.minimum(1,q)

frows=[];rrows=[];predrows=[]
for o in range(5):
 for i in range(3):
  for s in (42,43,44):
   t_ids,v_ids,ty,vy,ht,hv,st,sv,tr,va=load(o,i,s)
   Zt=np.stack([st,ht],axis=1);Zv=np.stack([sv,hv],axis=1)
   scal=StandardScaler().fit(Zt);Zt=scal.transform(Zt);Zv=scal.transform(Zv)
   fitted={}
   for name,cols in [('STR_only',[0]),('H1_only',[1]),('STR_plus_H1',[0,1])]:
    m=LogisticRegression(C=1,solver='liblinear',random_state=0,max_iter=1000).fit(Zt[:,cols],ty)
    p=m.predict_proba(Zv[:,cols])[:,1];auc,ba=metrics(vy,p)
    fitted[name]=p
    frows.append(dict(outer=o,inner=i,seed=s,model=name,auroc=auc,ba=ba,train_n=len(ty),validation_n=len(vy),coef_str=float(m.coef_[0][cols.index(0)]) if 0 in cols else np.nan,coef_h1=float(m.coef_[0][cols.index(1)]) if 1 in cols else np.nan))
   # Baseline and augmented residual probes use train-only score scaling.
   hmean,hstd=ht.mean(),max(ht.std(),1e-8);smean,sstd=st.mean(),max(st.std(),1e-8)
   target_t=(ht-hmean)/hstd-(st-smean)/sstd;target_v=(hv-hmean)/hstd-(sv-smean)/sstd
   score_t=((st-smean)/sstd).reshape(-1,1);score_v=((sv-smean)/sstd).reshape(-1,1)
   stages={'score_only':(None,None),'activity_raw':(tr['activity_raw'].reshape(len(ty),-1),va['activity_raw'].reshape(len(vy),-1)),'subject_embedding':(tr['subject_embedding'],va['subject_embedding']),'structured_embedding':(tr['structured_embedding'],va['structured_embedding']),'decision_input':(np.concatenate([tr['subject_embedding'],tr['structured_embedding']],axis=1),np.concatenate([va['subject_embedding'],va['structured_embedding']],axis=1))}
   for stage,(Xt,Xv) in stages.items():
    if Xt is None:A,B=score_t,score_v
    else:
     xs=StandardScaler().fit(Xt);A=np.concatenate([score_t,xs.transform(Xt)],axis=1);B=np.concatenate([score_v,xs.transform(Xv)],axis=1)
    model=Ridge(alpha=1.0,solver='auto').fit(A,target_t)
    pred=model.predict(B);r2=float(r2_score(target_v,pred));mae=float(mean_absolute_error(target_v,pred));rho=float(spearmanr(target_v,pred).statistic)
    rrows.append(dict(outer=o,inner=i,seed=s,stage=stage,validation_r2=r2,validation_mae=mae,validation_spearman=rho,target_sd=float(target_v.std()),prediction_sd=float(pred.std()),train_r2=float(r2_score(target_t,model.predict(A)))))
    if stage in ('score_only','decision_input'):
     predrows.extend(dict(outer=o,inner=i,seed=s,subject_id=v_ids[j],label=int(vy[j]),stage=stage,target_residual=float(target_v[j]),predicted_residual=float(pred[j])) for j in range(len(v_ids)))
frame=pd.DataFrame(frows);frame.to_csv(OUT/'fusion_45_seed_splits.csv',index=False)
wide=frame.pivot_table(index=['outer','inner','seed'],columns='model',values=['auroc','ba']);wide.columns=[f'{a}_{b}' for a,b in wide.columns];wide=wide.reset_index();wide.to_csv(OUT/'fusion_45_paired.csv',index=False)
split=wide.groupby(['outer','inner'],as_index=False).mean(numeric_only=True);split.to_csv(OUT/'fusion_15_paired.csv',index=False)
inf=[]
for metric in ('auroc','ba'):
 for baseline in ('STR_only','H1_only'):
  delta=split[f'{metric}_STR_plus_H1']-split[f'{metric}_{baseline}']
  inf.append(dict(metric=metric,comparison=f'fusion_minus_{baseline}',mean_delta=float(delta.mean()),wins=int((delta>0).sum()),p=float(wilcoxon(delta).pvalue)))
inf=pd.DataFrame(inf);inf['q']=bh(inf.p);inf.to_csv(OUT/'fusion_paired_inference.csv',index=False)
r=pd.DataFrame(rrows);r.to_csv(OUT/'residual_probes_45.csv',index=False)
rsp=r.groupby(['outer','inner','stage'],as_index=False).mean(numeric_only=True);rsp.to_csv(OUT/'residual_probes_15.csv',index=False)
base=rsp[rsp.stage=='score_only'].set_index(['outer','inner']);res=[]
for stage in ('activity_raw','subject_embedding','structured_embedding','decision_input'):
 g=rsp[rsp.stage==stage].set_index(['outer','inner'])
 for metric,sign in [('validation_r2',1),('validation_mae',-1),('validation_spearman',1)]:
  d=sign*(g[metric]-base[metric]);res.append(dict(stage=stage,metric=metric,baseline_mean=float(base[metric].mean()),stage_mean=float(g[metric].mean()),mean_improvement=float(d.mean()),wins=int((d>0).sum()),p=float(wilcoxon(d).pvalue)))
ri=pd.DataFrame(res);ri['q']=bh(ri.p);ri.to_csv(OUT/'residual_probe_paired_inference.csv',index=False)
pd.DataFrame(predrows).to_csv(OUT/'residual_probe_predictions.csv',index=False)
(OUT/'fusion_residual_protocol.json').write_text(json.dumps({'scope':'inner-development only','outer_information_used':False,'H1_train_scores':'fixed formal H1 pipeline refit on inner-train, validation reproduction max 1.1e-16; train scores are in-sample','STR_train_scores':'frozen checkpoint on its own train subjects, in-sample','fusion':'C=1 logistic on train-only standardized STR DD logit margin and H1 DD logit; diagnostic only','residual_target':'H1 train-z-logit minus STR train-z-logit; both train standardizations applied to validation','residual_probe':'Ridge alpha=1 with train-only feature scaling, STR score covariate in all stages','warning':'in-sample base-model train scores versus out-of-sample validation score distribution; negative residual-probe result cannot prove information absence'},indent=2))
print(inf.round(4).to_string(index=False));print(ri.round(4).to_string(index=False))
