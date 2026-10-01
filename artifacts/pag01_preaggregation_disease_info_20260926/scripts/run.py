#!/usr/bin/env python3
"""Train-only disease probes, OOF incremental signal, score-conditional controls, ambiguity."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,f1_score,recall_score,log_loss,r2_score,mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
R=Path('/home/zyt/deep_final');B=R/'artifacts/pag01_preaggregation_disease_info_20260926';O=B/'analysis';E=R/'artifacts/dsg01_cross_subject_disease_subspace_20260926/extraction/embeddings'
ST=pd.read_csv(R/'artifacts/dsg01_cross_subject_disease_subspace_20260926/analysis/error_consistency_subjects.csv',dtype={'subject_id':str});ST=ST[ST.model=='str'].set_index('subject_id').primary_status.to_dict()
STAGES={'activity_raw':('activity_raw',),'activity_context':('activity_context',),'subject_embedding':('subject_embedding',),'structured_embedding':('structured_embedding',),'decision_input':('subject_embedding','structured_embedding')}
KS=(8,16,32)
def load(o,i,seed):
 with np.load(E/f'str_seed{seed}_outer{o}_inner{i}_train.npz',allow_pickle=False) as t,np.load(E/f'str_seed{seed}_outer{o}_inner{i}_validation.npz',allow_pickle=False) as v:
  tid=t['subject_id'].astype(str);vid=v['subject_id'].astype(str);ty=t['label'].astype(int);vy=v['label'].astype(int)
  assert len(set(tid))==len(tid) and len(set(vid))==len(vid) and not set(tid)&set(vid)
  tr={k:t[k] for k in t.files};va={k:v[k] for k in v.files}
 return tid,vid,ty,vy,tr,va

def raw_mats(tr,va,keys):
 n=len(tr['subject_id']);m=len(va['subject_id']);return np.concatenate([tr[k].reshape(n,-1) for k in keys],axis=1),np.concatenate([va[k].reshape(m,-1) for k in keys],axis=1)
def pca_mats(xt,xv,k):
 scaler=StandardScaler().fit(xt);a=scaler.transform(xt);b=scaler.transform(xv);pc=PCA(n_components=k,svd_solver='randomized',random_state=0).fit(a);return pc.transform(a),pc.transform(b)
def clf():return LogisticRegression(C=1,solver='liblinear',random_state=0,max_iter=1000)
def disease_metrics(y,logits):
 p=expit(logits);pred=(logits>=0).astype(int)
 return dict(auroc=float(roc_auc_score(y,p)),ba=float(balanced_accuracy_score(y,pred)),macro_f1=float(f1_score(y,pred,average='macro')),pd_recall=float(recall_score(y,pred,pos_label=0)),dd_recall=float(recall_score(y,pred,pos_label=1)),log_loss=float(log_loss(y,np.clip(p,1e-12,1-1e-12),labels=[0,1])))
def logit_model(x,y,v):return clf().fit(x,y).decision_function(v)
def oof_increment(ty,st,raw,rawv,sv,seed):
 cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=1000+seed);base=np.empty(len(ty));aug=np.empty(len(ty))
 for train,hold in cv.split(st,ty):
  sc=StandardScaler().fit(st[train,None]);zt=sc.transform(st[train,None]);zh=sc.transform(st[hold,None]);rtr,rhold=pca_mats(raw[train],raw[hold],16)
  base[hold]=logit_model(zt,ty[train],zh);aug[hold]=logit_model(np.column_stack([zt,rtr]),ty[train],np.column_stack([zh,rhold]))
 sc=StandardScaler().fit(st[:,None]);zt=sc.transform(st[:,None]);zv=sc.transform(sv[:,None]);baseval=logit_model(zt,ty,zv)
 rawtr,rawval=pca_mats(raw,rawv,16);augval=logit_model(np.column_stack([zt,rawtr]),ty,np.column_stack([zv,rawval]))
 return base,aug,baseval,augval,zt,zv,sc

def ambiguity(margin,q1,q2):
 a=np.abs(margin);return np.where(a<=q1,'high',np.where(a<=q2,'medium','low'))
def permute_by_score_bins(pc,score,rng):
 edges=np.quantile(score,np.linspace(0,1,11))[1:-1];bins=np.digitize(score,edges);out=pc.copy()
 for b in np.unique(bins):
  ix=np.flatnonzero(bins==b);out[ix]=pc[rng.permutation(ix)]
 return out

def run_one(o,i,seed,perms):
 tid,vid,ty,vy,tr,va=load(o,i,seed);st=tr['final_logits'][:,1]-tr['final_logits'][:,0];sv=va['final_logits'][:,1]-va['final_logits'][:,0]
 mats={stage:raw_mats(tr,va,keys) for stage,keys in STAGES.items()}
 boof,aoof,bval,aval,zt,zv,score_scaler=oof_increment(ty,st,*mats['activity_raw'],sv,seed)
 target=aoof-boof;targetval=aval-bval;q1,q2=np.quantile(np.abs(boof),[1/3,2/3]);amb=ambiguity(bval,q1,q2)
 assert min(np.bincount(ty))>=5 and len(set(amb))>=2
 base_metrics=disease_metrics(vy,bval);probe=[];signal=[];neg=[];pred=[]
 probe.append(dict(outer=o,inner=i,seed=seed,pca_k=16,stage='score_only',**base_metrics,delta_auroc=0.0))
 for stage,(xt,xv) in mats.items():
  for k in KS:
   pt,pv=pca_mats(xt,xv,k);at=np.column_stack([zt,pt]);av=np.column_stack([zv,pv]);dl=logit_model(at,ty,av);m=disease_metrics(vy,dl)
   probe.append(dict(outer=o,inner=i,seed=seed,pca_k=k,stage=stage,**m,delta_auroc=m['auroc']-base_metrics['auroc']))
   rg=Ridge(alpha=1.0).fit(at,target);signalhat=rg.predict(av);corr=bval+signalhat
   rho=spearmanr(targetval,signalhat).statistic
   signal.append(dict(outer=o,inner=i,seed=seed,pca_k=k,stage=stage,increment_r2=float(r2_score(targetval,signalhat)),increment_mae=float(mean_absolute_error(targetval,signalhat)),increment_spearman=float(rho),corrected_auroc=float(roc_auc_score(vy,expit(corr))),corrected_auroc_delta=float(roc_auc_score(vy,expit(corr))-base_metrics['auroc']),true_increment_sd=float(targetval.std()),pred_increment_sd=float(signalhat.std())))
   if k==16:
    for j,sid in enumerate(vid):pred.append(dict(outer=o,inner=i,seed=seed,subject_id=sid,label=int(vy[j]),stage=stage,str_score=float(sv[j]),base_disease_logit=float(bval[j]),raw_probe_disease_logit=float(aval[j]),true_increment_signal=float(targetval[j]),pred_increment_signal=float(signalhat[j]),corrected_disease_logit=float(corr[j]),ambiguity=str(amb[j]),stable_status=ST[sid],train_oof_ambiguity_q1=float(q1),train_oof_ambiguity_q2=float(q2)))
   if k==16:
    rng=np.random.default_rng(100000*o+1000*i+10*seed+13)
    for rep in range(perms):
     ptrain=permute_by_score_bins(pt,st,rng);null=logit_model(np.column_stack([zt,ptrain]),ty,av);na=float(roc_auc_score(vy,expit(null)))
     neg.append(dict(outer=o,inner=i,seed=seed,stage=stage,permutation=rep,baseline_auroc=base_metrics['auroc'],permuted_auroc=na,permuted_delta_auroc=na-base_metrics['auroc']))
 # Subject-group ambiguity from OOF baseline margin, held-out labels only for evaluation.
 groups=[]
 for group in ('high','medium','low'):
  sel=amb==group;y=vy[sel]
  rec=dict(outer=o,inner=i,seed=seed,ambiguity=group,n=int(sel.sum()),pd_n=int(sum(y==0)),dd_n=int(sum(y==1)),str_error_rate=float(np.mean((sv[sel]>=0)!=y)),score_only_error_rate=float(np.mean((bval[sel]>=0)!=y)),raw_error_rate=float(np.mean((aval[sel]>=0)!=y)),raw_log_loss=float(log_loss(y,expit(aval[sel]),labels=[0,1])) if len(y) else np.nan)
  rec['base_auroc']=float(roc_auc_score(y,expit(bval[sel]))) if len(set(y))==2 else np.nan
  rec['raw_auroc']=float(roc_auc_score(y,expit(aval[sel]))) if len(set(y))==2 else np.nan
  rec['raw_auroc_gain']=rec['raw_auroc']-rec['base_auroc'];groups.append(rec)
 # Pair recovery from final STR score to directly fitted activity-raw disease probe.
 pdix=np.flatnonzero(vy==0);ddix=np.flatnonzero(vy==1);d=np.repeat(ddix,len(pdix));p=np.tile(pdix,len(ddix));orig=sv[d]-sv[p];corrected=aval[d]-aval[p]
 rawp=expit(aval);pairs=[]
 for j in range(len(d)):
  dg,pg=amb[d[j]],amb[p[j]];group='high' if 'high' in (dg,pg) else ('medium' if 'medium' in (dg,pg) else 'low')
  pairs.append(dict(outer=o,inner=i,seed=seed,dd_subject_id=vid[d[j]],pd_subject_id=vid[p[j]],pair_ambiguity=group,dd_ambiguity=str(dg),pd_ambiguity=str(pg),any_stable_error=ST[vid[d[j]]]=='stable_error' or ST[vid[p[j]]]=='stable_error',original_margin=float(orig[j]),raw_probe_margin=float(corrected[j]),original_correct=bool(orig[j]>0),raw_probe_correct=bool(corrected[j]>0),recovered=bool(orig[j]<0 and corrected[j]>0),newly_broken=bool(orig[j]>0 and corrected[j]<0)))
 return probe,signal,neg,pred,groups,pairs,dict(outer=o,inner=i,seed=seed,train_n=len(ty),validation_n=len(vy),oof_q1=float(q1),oof_q2=float(q2),oof_base_auroc=float(roc_auc_score(ty,expit(boof))),oof_raw_auroc=float(roc_auc_score(ty,expit(aoof))),validation_base_auroc=base_metrics['auroc'],validation_raw_auroc=float(roc_auc_score(vy,expit(aval))),validation_str_auroc=float(roc_auc_score(vy,sv)),baseline_coef_positive=bool(np.corrcoef(sv,bval)[0,1]>0))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--permutations',type=int,default=10);args=ap.parse_args()
 jobs=[(0,0,42)] if args.smoke else [(o,i,s) for o in range(5) for i in range(3) for s in (42,43,44)]
 out=[[],[],[],[],[],[],[]]
 for o,i,seed in jobs:
  result=run_one(o,i,seed,args.permutations)
  for dest,part in zip(out,result):dest.extend(part if isinstance(part,list) else [part])
  print('finished',o,i,seed,flush=True)
 if args.smoke:
  a=out[0];assert len(a)==16 and len(out[1])==15 and len(out[2])==5*args.permutations and len(out[5])==sum(x['pd_n'] for x in out[4])*sum(x['dd_n'] for x in out[4])
  print(json.dumps({'status':'PASS','one_split_seed':jobs[0],'probe_rows':len(out[0]),'signal_rows':len(out[1]),'permutation_rows':len(out[2]),'pair_rows':len(out[5]),'ambiguity_groups':sorted(set(x['ambiguity'] for x in out[4]))},indent=2));(B/'smoke.json').write_text(json.dumps({'status':'PASS','probe_rows':len(out[0]),'signal_rows':len(out[1]),'pair_rows':len(out[5])},indent=2))
 else:
  names=['disease_probe_45.csv','increment_signal_45.csv','conditional_permutation_45.csv','subject_predictions_primary.csv.gz','ambiguity_subject_groups_45.csv','ambiguity_pair_records.csv.gz','crossfit_audit_45.csv']
  for name,rows in zip(names,out):pd.DataFrame(rows).to_csv(O/name,index=False,compression='gzip' if name.endswith('.gz') else None)
  assert len(out[0])==45*16 and len(out[1])==45*15 and len(out[2])==45*5*args.permutations and len(out[6])==45
  print(json.dumps({'status':'PASS','split_seed_instances':45,'disease_probe_rows':len(out[0]),'increment_signal_rows':len(out[1]),'permutation_rows':len(out[2]),'subject_prediction_rows':len(out[3]),'pair_rows':len(out[5]),'outer_information_used':False},indent=2))
