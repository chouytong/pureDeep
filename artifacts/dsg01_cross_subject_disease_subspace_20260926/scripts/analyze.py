#!/usr/bin/env python3
"""DSG-01 frozen representation diagnostics; no model forward or outer data."""
from __future__ import annotations
import json,sys,warnings
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import numpy as np,pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr,wilcoxon,mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
from sklearn.preprocessing import StandardScaler
ROOT=Path('/home/zyt/deep_final')
BASE=ROOT/'artifacts/dsg01_cross_subject_disease_subspace_20260926'
EM=BASE/'extraction/embeddings';OUT=BASE/'analysis';OUT.mkdir(exist_ok=True)
SEEDS=(42,43,44);SPLITS=tuple((o,i) for o in range(5) for i in range(3));INST=tuple((s,o,i) for s in SEEDS for o,i in SPLITS)
STAGES={'v8':['activity_raw_flat','activity_context_flat','subject_embedding'],
        'str':['activity_raw_flat','activity_context_flat','subject_embedding','structured_embedding','decision_input']}
ACTIVITIES=json.loads((BASE/'asset_audit.json').read_text())['activities']
def sid(inst):return f'{inst[0]}_{inst[1]}_{inst[2]}'
@lru_cache(maxsize=None)
def payload(model,inst,role):
 s,o,i=inst;p=EM/f'{model}_seed{s}_outer{o}_inner{i}_{role}.npz'
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def features(data,stage):
 if stage=='activity_raw_flat':return data['activity_raw'].reshape(len(data['subject_id']),-1)
 if stage=='activity_context_flat':return data['activity_context'].reshape(len(data['subject_id']),-1)
 if stage=='decision_input':return np.concatenate([data['subject_embedding'],data['structured_embedding']],axis=1)
 if stage.startswith('activity_raw:'):return data['activity_raw'][:,int(stage.split(':')[1]),:]
 if stage.startswith('activity_context:'):return data['activity_context'][:,int(stage.split(':')[1]),:]
 if stage.startswith('structured_tokens:'):return data['structured_tokens'][:,int(stage.split(':')[1]),:]
 return data[stage]
def aligned_rows(data_a,data_b):
 ia={str(x):j for j,x in enumerate(data_a['subject_id'])};ib={str(x):j for j,x in enumerate(data_b['subject_id'])}
 ids=sorted(ia.keys()&ib.keys());return ids,np.array([ia[x] for x in ids]),np.array([ib[x] for x in ids])
def cka(X,Y):
 X=X.astype(np.float64);Y=Y.astype(np.float64);X-=X.mean(0);Y-=Y.mean(0)
 K=X@X.T;L=Y@Y.T
 num=np.sum(K*L);den=np.sqrt(np.sum(K*K)*np.sum(L*L))
 return float(num/den) if den>0 else np.nan
def score(y,decision):
 if len(y)<20 or min(np.bincount(y,minlength=2))<5:return np.nan,np.nan
 return float(balanced_accuracy_score(y,decision>=0)),float(roc_auc_score(y,decision))
def direction(v):
 n=np.linalg.norm(v);return v/n if n>1e-12 else np.full_like(v,np.nan)
def centroid(X,y):return direction(X[y==1].mean(0)-X[y==0].mean(0))
def fitted_probe(X,y):
 sc=StandardScaler().fit(X);z=sc.transform(X)
 m=LogisticRegression(C=1,solver='liblinear',class_weight='balanced',max_iter=1000,random_state=0).fit(z,y)
 return sc,m,direction(m.coef_[0]/sc.scale_)
def decide(sc,m,X):return m.decision_function(sc.transform(X))
def principal_angles(a,b):
 qa=np.linalg.qr(np.stack(a,axis=1))[0];qb=np.linalg.qr(np.stack(b,axis=1))[0]
 sv=np.linalg.svd(qa.T@qb,compute_uv=False)
 return np.degrees(np.arccos(np.clip(sv,-1,1)))
@lru_cache(maxsize=None)
def fitted(model,inst,stage):
 tr=payload(model,inst,'train');va=payload(model,inst,'validation')
 X=features(tr,stage).astype(np.float64);y=tr['label'].astype(int)
 V=features(va,stage).astype(np.float64);vy=va['label'].astype(int)
 # Fixed unsupervised 16-D PCA avoids underdetermined 258/2838-D Procrustes.
 pca=PCA(n_components=16,svd_solver='randomized',random_state=0).fit(X)
 Z=pca.transform(X);Zv=pca.transform(V)
 sc,probe,w=fitted_probe(X,y);p_sc,p_probe,p_w=fitted_probe(Z,y)
 native_ba,native_auc=score(vy,decide(sc,probe,V));pca_ba,pca_auc=score(vy,decide(p_sc,p_probe,Zv))
 return dict(X=X,y=y,V=V,vy=vy,pca=pca,Z=Z,Zv=Zv,sc=sc,probe=probe,w=w,p_sc=p_sc,p_probe=p_probe,p_w=p_w,cent=centroid(Z,y),native_ba=native_ba,native_auc=native_auc,pca_ba=pca_ba,pca_auc=pca_auc)
def pairing(model,stage,a,b,transfer=True):
 fa=fitted(model,a,stage);fb=fitted(model,b,stage)
 ta=payload(model,a,'train');tb=payload(model,b,'train');ids,ia,ib=aligned_rows(ta,tb)
 row=dict(model=model,stage=stage,source_seed=a[0],source_outer=a[1],source_inner=a[2],target_seed=b[0],target_outer=b[1],target_inner=b[2],common_train_n=len(ids),direction_cos=np.nan,probe_cos=np.nan,principal_angle_mean_deg=np.nan,raw_centroid_cos_descriptive=np.nan,eval_n=0,eval_pd=0,eval_dd=0,native_ba=np.nan,native_auroc=np.nan,transfer_ba=np.nan,transfer_auroc=np.nan,transfer_drop_ba=np.nan,transfer_drop_auroc=np.nan)
 if len(ids)<32:return row
 A=fa['Z'][ia];B=fb['Z'][ib]
 # Unlabeled alignment only; disease labels never enter the Procrustes fit.
 A=A-A.mean(0);B=B-B.mean(0)
 if np.linalg.matrix_rank(A)<16 or np.linalg.matrix_rank(B)<16:return row
 R,_=orthogonal_procrustes(B/np.linalg.norm(B),A/np.linalg.norm(A))
 row['direction_cos']=float(fa['cent']@(fb['cent']@R))
 row['probe_cos']=float(fa['p_w']@(fb['p_w']@R))
 row['principal_angle_mean_deg']=float(principal_angles((fa['cent'],fa['p_w']),(fb['cent']@R,fb['p_w']@R)).mean())
 if fa['X'].shape[1]==fb['X'].shape[1]:row['raw_centroid_cos_descriptive']=float(centroid(fa['X'],fa['y'])@centroid(fb['X'],fb['y']))
 if not transfer:return row
 # Evaluation B.val excludes every subject used to fit the A probe.
 av=set(map(str,ta['subject_id']));bv=payload(model,b,'validation');eligible=np.array([str(x) not in av for x in bv['subject_id']]);y=fb['vy'][eligible]
 row.update(eval_n=int(len(y)),eval_pd=int(np.sum(y==0)),eval_dd=int(np.sum(y==1)))
 if len(y)<20 or min(np.bincount(y,minlength=2))<5:return row
 native=decide(fb['p_sc'],fb['p_probe'],fb['Zv'][eligible]);transfer_dec=decide(fa['p_sc'],fa['p_probe'],fb['Zv'][eligible]@R)
 nb,na=score(y,native);xb,xa=score(y,transfer_dec)
 row.update(native_ba=nb,native_auroc=na,transfer_ba=xb,transfer_auroc=xa,transfer_drop_ba=nb-xb,transfer_drop_auroc=na-xa)
 return row

def run():
 # A: 45 x 45 linear CKA matrices per model/stage on shared training subjects.
 cka_rows=[]
 for model in ('v8','str'):
  for stage in STAGES[model]:
   matrix=np.eye(len(INST))
   for u,a in enumerate(INST):
    da=payload(model,a,'train');Xa=features(da,stage)
    for v in range(u+1,len(INST)):
     b=INST[v];db=payload(model,b,'train');ids,ia,ib=aligned_rows(da,db)
     value=cka(Xa[ia],features(db,stage)[ib]) if len(ids)>=20 else np.nan
     matrix[u,v]=matrix[v,u]=value
     cka_rows.append(dict(model=model,stage=stage,a_seed=a[0],a_outer=a[1],a_inner=a[2],b_seed=b[0],b_outer=b[1],b_inner=b[2],common_train_n=len(ids),cka=value,pair_type='same_split_cross_seed' if a[1:]==b[1:] else ('cross_split_same_seed' if a[0]==b[0] else 'cross_split_cross_seed')))
   pd.DataFrame(matrix,index=[sid(x) for x in INST],columns=[sid(x) for x in INST]).to_csv(OUT/f'cka_matrix_{model}_{stage}.csv')
   print('CKA',model,stage,flush=True)
 pd.DataFrame(cka_rows).to_csv(OUT/'cka_pairs.csv',index=False)
 # B/C: ordered within-model pairs. Include cross-seed same-split sensitivity.
 pair_rows=[];native=[]
 for model in ('v8','str'):
  for stage in STAGES[model]:
   for a in INST:
    f=fitted(model,a,stage);native.append(dict(model=model,stage=stage,seed=a[0],outer=a[1],inner=a[2],native_probe_ba=f['native_ba'],native_probe_auroc=f['native_auc'],pca_probe_ba=f['pca_ba'],pca_probe_auroc=f['pca_auc']))
    for b in INST:
     if a==b or (a[0]!=b[0] and a[1:]!=b[1:]):continue
     row=pairing(model,stage,a,b,transfer=True)
     row['pair_type']='same_split_cross_seed' if a[1:]==b[1:] else 'cross_split_same_seed'
     pair_rows.append(row)
   print('pairs',model,stage,flush=True)
 pd.DataFrame(native).to_csv(OUT/'native_probes.csv',index=False)
 pd.DataFrame(pair_rows).to_csv(OUT/'alignment_transfer_pairs.csv',index=False)
 # D: all 11 activities, raw/context for both models, ordered tokens for STR.
 act_native=[];act_pair=[]
 for model in ('v8','str'):
  kinds=['activity_raw','activity_context']+(['structured_tokens'] if model=='str' else [])
  for kind in kinds:
   for aid,activity in enumerate(ACTIVITIES):
    stage=f'{kind}:{aid}'
    for a in INST:
     f=fitted(model,a,stage)
     act_native.append(dict(model=model,kind=kind,activity=activity,seed=a[0],outer=a[1],inner=a[2],probe_ba=f['native_ba'],probe_auroc=f['native_auc']))
     for b in INST:
      if a==b or a[0]!=b[0]:continue
      r=pairing(model,stage,a,b,transfer=False)
      act_pair.append(dict(model=model,kind=kind,activity=activity,seed=a[0],outer=a[1],inner=a[2],target_outer=b[1],target_inner=b[2],common_train_n=r['common_train_n'],direction_cos=r['direction_cos'],probe_cos=r['probe_cos'],principal_angle_mean_deg=r['principal_angle_mean_deg']))
    print('activity',model,kind,activity,flush=True)
 pd.DataFrame(act_native).to_csv(OUT/'activity_native_probes.csv',index=False)
 pd.DataFrame(act_pair).to_csv(OUT/'activity_alignment_pairs.csv',index=False)
 (OUT/'protocol.json').write_text(json.dumps({'scope':'inner-development only','outer_information_used':False,'model_training':False,'linear_CKA_subjects':'shared train subjects only','alignment':'unlabeled Orthogonal Procrustes on shared train subjects after fixed train-only PCA16','alignment_min_common_train':32,'probe':'train-only balanced logistic C=1','transfer':'target validation subjects absent source probe train; min 20, min 5/class; NA otherwise','primary_unit':'15 split means after 3-seed aggregation','stable_error_primary':'seed-first four validation appearances, >=0.75 error','stable_error_sensitivity':'12 individual seed-fold appearances','raw_cosine':'descriptive only'},indent=2))
if __name__=='__main__':run()
