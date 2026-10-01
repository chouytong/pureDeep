#!/usr/bin/env python3
"""Unbiased HSIC-normalized linear CKA sensitivity for unequal feature dimensions."""
import sys
from pathlib import Path
import numpy as np,pandas as pd
BASE=Path('/home/zyt/deep_final/artifacts/dsg01_cross_subject_disease_subspace_20260926');sys.path.insert(0,str(BASE/'scripts'))
from analyze import payload,features,aligned_rows,INST,STAGES,OUT,sid

def uhsic(K,L):
 n=K.shape[0]
 K=K.copy();L=L.copy();np.fill_diagonal(K,0);np.fill_diagonal(L,0)
 return (np.sum(K*L)+K.sum()*L.sum()/((n-1)*(n-2))-2*np.dot(K.sum(0),L.sum(0))/(n-2))/(n*(n-3))
def unbiased_cka(X,Y):
 X=np.asarray(X,dtype=np.float64);Y=np.asarray(Y,dtype=np.float64)
 K=X@X.T;L=Y@Y.T;xy=uhsic(K,L);xx=uhsic(K,K);yy=uhsic(L,L)
 return float(xy/np.sqrt(xx*yy)) if xx>0 and yy>0 else np.nan
rows=[]
for model in ('v8','str'):
 for stage in STAGES[model]:
  mat=np.eye(len(INST))
  for u,a in enumerate(INST):
   da=payload(model,a,'train');Xa=features(da,stage)
   for v in range(u+1,len(INST)):
    b=INST[v];db=payload(model,b,'train');ids,ia,ib=aligned_rows(da,db)
    val=unbiased_cka(Xa[ia],features(db,stage)[ib]) if len(ids)>=20 else np.nan
    mat[u,v]=mat[v,u]=val
    rows.append((model,stage,a[0],a[1],a[2],b[0],b[1],b[2],val))
  pd.DataFrame(mat,index=[sid(x) for x in INST],columns=[sid(x) for x in INST]).to_csv(OUT/f'cka_unbiased_matrix_{model}_{stage}.csv')
  print(model,stage,flush=True)
new=pd.DataFrame(rows,columns=['model','stage','a_seed','a_outer','a_inner','b_seed','b_outer','b_inner','cka_unbiased'])
old=pd.read_csv(OUT/'cka_pairs.csv');merged=old.merge(new,on=list(new.columns[:-1]),validate='one_to_one');merged.to_csv(OUT/'cka_pairs_with_unbiased.csv',index=False)
