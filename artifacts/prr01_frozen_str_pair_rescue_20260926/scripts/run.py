#!/usr/bin/env python3
"""Frozen STR subject residual correction and validation PD-DD pair recovery."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
R=Path('/home/zyt/deep_final');B=R/'artifacts/prr01_frozen_str_pair_rescue_20260926';O=B/'analysis';E=R/'artifacts/dsg01_cross_subject_disease_subspace_20260926/extraction/embeddings';H=R/'artifacts/rgd01_h1_str_residual_ranking_20260926/h1_reproduction';RG=R/'artifacts/rgd01_h1_str_residual_ranking_20260926/analysis'
ST=pd.read_csv(RG/'subject_ranking_burden_390.csv',dtype={'subject_id':str}).set_index('subject_id').status.to_dict()
STAGES={'A_STR':None,'B_score_only':None,'C_activity_raw':('activity_raw',),'D_subject':('subject_embedding',),'E_structured':('structured_embedding',),'F_decision':('subject_embedding','structured_embedding')}
def logit(p):p=np.clip(p,1e-8,1-1e-8);return np.log(p/(1-p))
def load(o,i,s):
 with np.load(H/f'outer{o}_inner{i}.npz',allow_pickle=False) as h:
  ti={str(x):j for j,x in enumerate(h['train_subject_id'])};vi={str(x):j for j,x in enumerate(h['validation_subject_id'])};ht0=logit(h['train_h1_probability_dd']);hvraw0=h['validation_h1_probability_dd'];hv0=logit(hvraw0);ty0=h['train_label'].astype(int);vy0=h['validation_label'].astype(int)
 with np.load(E/f'str_seed{s}_outer{o}_inner{i}_train.npz',allow_pickle=False) as t,np.load(E/f'str_seed{s}_outer{o}_inner{i}_validation.npz',allow_pickle=False) as v:
  tid=t['subject_id'].astype(str);vid=v['subject_id'].astype(str);ty=t['label'].astype(int);vy=v['label'].astype(int)
  assert set(tid)==set(ti) and set(vid)==set(vi) and np.array_equal(ty,ty0[[ti[x] for x in tid]]) and np.array_equal(vy,vy0[[vi[x] for x in vid]])
  ht=ht0[[ti[x] for x in tid]];hv=hv0[[vi[x] for x in vid]];hvraw=hvraw0[[vi[x] for x in vid]];st=t['final_logits'][:,1]-t['final_logits'][:,0];sv=v['final_logits'][:,1]-v['final_logits'][:,0]
  tr={k:t[k] for k in t.files};va={k:v[k] for k in v.files}
 return tid,vid,ty,vy,ht,hv,hvraw,st,sv,tr,va

def pair_table(o,i,s,vid,vy,sv,hvraw,scores,predres,activity_mag):
 pdix=np.flatnonzero(vy==0);ddix=np.flatnonzero(vy==1);d=np.repeat(ddix,len(pdix));p=np.tile(pdix,len(ddix));n=len(d)
 sm=sv[d]-sv[p];hm=hvraw[d]-hvraw[p];orig=np.where((sm>0)&(hm>0),'both_correct',np.where((sm<0)&(hm<0),'both_wrong',np.where((sm<0)&(hm>0),'h1_rescue',np.where((sm>0)&(hm<0),'str_rescue','tie'))))
 if n!=len(pdix)*len(ddix):raise AssertionError('pair indexing')
 anyhard=np.array([(ST[vid[x]]=='stable_error' or ST[vid[y]]=='stable_error') for x,y in zip(d,p)])
 base={'outer':o,'inner':i,'seed':s,'dd_subject_id':vid[d],'pd_subject_id':vid[p],'dd_status':[ST[x] for x in vid[d]],'pd_status':[ST[x] for x in vid[p]],'any_stable_error':anyhard,'original_category':orig,'str_original_margin':sm,'h1_margin':hm,'activity_rep_residual_magnitude':activity_mag[d]+activity_mag[p]}
 rows=[];stats=[]
 for stage,score in scores.items():
  cm=score[d]-score[p];res=predres[stage];pb=base.copy();pb.update(stage=stage,corrected_margin=cm,predicted_residual_pair_margin=res[d]-res[p])
  if stage!='A_STR':rows.append(pd.DataFrame(pb))
  rescue=orig=='h1_rescue';strgood=sm>0;strbad=sm<0;rec=np.where(cm>0,1,np.where(cm==0,.5,0));harm=np.where(cm<0,1,np.where(cm==0,.5,0));recovered=float(rec[rescue].sum());newbroken=float(harm[strgood].sum());all_wrong_recovered=float(rec[strbad].sum());remaining=float(rescue.sum()-recovered);net=all_wrong_recovered-newbroken
  # STR ties are absent or tracked separately; exact AUROC identity tested if absent.
  stats.append(dict(outer=o,inner=i,seed=s,stage=stage,val_pd=len(pdix),val_dd=len(ddix),pairs=n,h1_rescue_pairs=int(rescue.sum()),str_correct_pairs=int(strgood.sum()),recovered_pairs=recovered,unrecovered_h1_rescue_pairs=remaining,all_wrong_recovered_pairs=all_wrong_recovered,targeted_net_pair_gain=recovered-newbroken,recovery_rate=recovered/max(1,rescue.sum()),newly_broken_pairs=newbroken,harm_rate=newbroken/max(1,strgood.sum()),str_correct_preservation=1-newbroken/max(1,strgood.sum()),net_pair_gain=net,corrected_auroc=float(roc_auc_score(vy,score)),corrected_ba=float(balanced_accuracy_score(vy,score>=0)),str_original_auroc=float(roc_auc_score(vy,sv)),h1_auroc=float(roc_auc_score(vy,hvraw)),str_tie_pairs=int(sum(sm==0)),corrected_tie_pairs=int(sum(cm==0))))
 return rows,stats

def run_one(o,i,s,permutations=10):
 tid,vid,ty,vy,ht,hv,hvraw,st,sv,tr,va=load(o,i,s)
 hm,hs=ht.mean(),max(ht.std(),1e-8);sm,ss=st.mean(),max(st.std(),1e-8)
 target=(ht-hm)/hs-(st-sm)/ss;zt=((st-sm)/ss).reshape(-1,1);zv=((sv-sm)/ss).reshape(-1,1)
 scores={'A_STR':sv};predres={'A_STR':np.zeros(len(vid))};train_mats={};val_mats={}
 for stage,keys in STAGES.items():
  if stage=='A_STR':continue
  if keys is None:Xt=zt;Xv=zv
  else:
   t0=np.concatenate([tr[k].reshape(len(tid),-1) for k in keys],axis=1);v0=np.concatenate([va[k].reshape(len(vid),-1) for k in keys],axis=1)
   scale=StandardScaler().fit(t0);pt=scale.transform(t0);pv=scale.transform(v0);pc=PCA(n_components=16,svd_solver='randomized',random_state=0).fit(pt)
   Xt=np.column_stack([zt,pc.transform(pt)]);Xv=np.column_stack([zv,pc.transform(pv)])
  train_mats[stage]=Xt;val_mats[stage]=Xv
  pred=Ridge(alpha=1.0).fit(Xt,target).predict(Xv)
  predres[stage]=pred;scores[stage]=sm+ss*(zv[:,0]+pred)
 # Fixed activity embedding residual magnitude, based on train-only mean; descriptive only.
 amean=tr['activity_raw'].mean(axis=0);activity_mag=np.linalg.norm((va['activity_raw']-amean).reshape(len(vid),-1),axis=1)
 pairrows,stats=pair_table(o,i,s,vid,vy,sv,hvraw,scores,predres,activity_mag)
 subrows=[]
 for stage in STAGES:
  for j,sid in enumerate(vid):subrows.append(dict(outer=o,inner=i,seed=s,subject_id=sid,label=int(vy[j]),status=ST[sid],stage=stage,str_original_score=float(sv[j]),h1_score=float(hvraw[j]),predicted_residual=float(predres[stage][j]),corrected_score=float(scores[stage][j]),activity_rep_residual_magnitude=float(activity_mag[j])))
 # Negative controls: class-conditional residual target permutations only within train.
 rng=np.random.default_rng(100000*o+1000*i+10*s+1);neg=[]
 for stage in ('C_activity_raw','D_subject','E_structured','F_decision'):
  Xt=train_mats[stage];Xv=val_mats[stage]
  for rep in range(permutations):
   perm=target.copy()
   for cls in (0,1):
    ix=np.flatnonzero(ty==cls);perm[ix]=target[rng.permutation(ix)]
   pred=Ridge(alpha=1.0).fit(Xt,perm).predict(Xv);corr=sm+ss*(zv[:,0]+pred)
   _,nr=pair_table(o,i,s,vid,vy,sv,hvraw,{'negative_control':corr},{'negative_control':pred},activity_mag)
   q=nr[0];neg.append(dict(outer=o,inner=i,seed=s,stage=stage,permutation=rep,corrected_auroc=q['corrected_auroc'],auroc_delta_vs_str=q['corrected_auroc']-q['str_original_auroc'],recovery_rate=q['recovery_rate'],net_pair_gain=q['net_pair_gain'],harm_rate=q['harm_rate']))
 return pairrows,stats,subrows,neg

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--permutations',type=int,default=10);args=ap.parse_args()
 jobs=[(0,0,42)] if args.smoke else [(o,i,s) for o in range(5) for i in range(3) for s in (42,43,44)]
 allpair=[];allstats=[];allsub=[];allneg=[]
 for o,i,s in jobs:
  p,q,u,n=run_one(o,i,s,args.permutations);allpair.extend(p);allstats.extend(q);allsub.extend(u);allneg.extend(n);print('finished',o,i,s,flush=True)
 if args.smoke:
  st=pd.DataFrame(allstats);assert len(st)==6 and st.str_tie_pairs.max()==0;assert np.allclose(st.net_pair_gain/st.pairs,st.corrected_auroc-st.str_original_auroc,atol=1e-9)
  print(json.dumps({'status':'PASS','split_seed':'outer0_inner0_seed42','methods':len(st),'h1_rescue_pairs':int(st.h1_rescue_pairs.iloc[0]),'permutation_rows':len(allneg),'pair_auc_identity_max_error':float(np.max(abs(st.net_pair_gain/st.pairs-(st.corrected_auroc-st.str_original_auroc))))},indent=2));(B/'smoke.json').write_text(json.dumps({'status':'PASS','methods':len(st),'permutation_rows':len(allneg),'pair_auc_identity_max_error':float(np.max(abs(st.net_pair_gain/st.pairs-(st.corrected_auroc-st.str_original_auroc))))},indent=2))
 else:
  st=pd.DataFrame(allstats);assert len(st)==270 and st.str_tie_pairs.max()==0;assert np.allclose(st.net_pair_gain/st.pairs,st.corrected_auroc-st.str_original_auroc,atol=1e-9)
  st['auroc_delta_vs_str']=st.corrected_auroc-st.str_original_auroc;st.to_csv(O/'method_metrics_45.csv',index=False)
  pd.concat(allpair,ignore_index=True).to_csv(O/'pair_level_records.csv.gz',index=False,compression='gzip')
  pd.DataFrame(allsub).to_csv(O/'subject_level_records.csv.gz',index=False,compression='gzip');pd.DataFrame(allneg).to_csv(O/'class_conditional_permutation_45.csv',index=False)
  print(json.dumps({'status':'PASS','split_seed_instances':len(jobs),'method_metrics':len(st),'pair_rows':sum(len(p) for p in allpair),'subject_rows':len(allsub),'permutation_rows':len(allneg),'pair_auc_identity_max_error':float(np.max(abs(st.net_pair_gain/st.pairs-st.auroc_delta_vs_str)))},indent=2))
