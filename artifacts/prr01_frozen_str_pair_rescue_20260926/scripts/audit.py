#!/usr/bin/env python3
"""Read-only audit of PRR fixed inner-development inputs."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
R=Path('/home/zyt/deep_final');P=R/'artifacts/prr01_frozen_str_pair_rescue_20260926';E=R/'artifacts/dsg01_cross_subject_disease_subspace_20260926/extraction/embeddings';H=R/'artifacts/rgd01_h1_str_residual_ranking_20260926/h1_reproduction';RG=R/'artifacts/rgd01_h1_str_residual_ranking_20260926'
prior=json.loads((RG/'protocol.json').read_text());assert prior['H1_prediction_seed_count']==1 and prior['STR_V8_seed_count']==3 and prior['outer_signal_label_prediction_metric_accessed'] is False
status=json.loads((R/'artifacts/dsg01_cross_subject_disease_subspace_20260926/analysis/stable_error_definitions.json').read_text())['primary'];assert status==prior['stable_error_primary']
rows=[]
for o in range(5):
 for i in range(3):
  with np.load(H/f'outer{o}_inner{i}.npz',allow_pickle=False) as h:
   tr=h['train_subject_id'].astype(str);va=h['validation_subject_id'].astype(str);ht=h['train_label'].astype(int);hv=h['validation_label'].astype(int)
   assert len(set(tr))==len(tr) and len(set(va))==len(va) and not set(tr)&set(va)
   assert set(h.files)=={'train_subject_id','train_label','train_h1_probability_dd','validation_subject_id','validation_label','validation_h1_probability_dd'}
  for s in (42,43,44):
   with np.load(E/f'str_seed{s}_outer{o}_inner{i}_train.npz',allow_pickle=False) as t,np.load(E/f'str_seed{s}_outer{o}_inner{i}_validation.npz',allow_pickle=False) as v:
    for z,ids,labels in ((t,tr,ht),(v,va,hv)):
     sid=z['subject_id'].astype(str);ix={x:j for j,x in enumerate(ids)}
     assert set(sid)==set(ids) and np.array_equal(z['label'].astype(int),labels[[ix[x] for x in sid]])
     assert z['activity_raw'].shape==(len(sid),11,258) and z['subject_embedding'].shape==(len(sid),258) and z['structured_embedding'].shape==(len(sid),176) and z['final_logits'].shape==(len(sid),2)
    rows.append(dict(outer=o,inner=i,seed=s,train_n=len(tr),validation_n=len(va),train_pd=int(sum(ht==0)),train_dd=int(sum(ht==1)),val_pd=int(sum(hv==0)),val_dd=int(sum(hv==1)),subject_label_alignment=True))
a=pd.DataFrame(rows);assert len(a)==45;a.to_csv(P/'asset_audit.csv',index=False)
protocol={'scope':'fixed inner-development only','outer_information_used':False,'STR_seeds':[42,43,44],'H1_deterministic_per_split':True,'primary_statistical_unit':'15 splits; 3 STR seeds aggregated within split','stable_error_primary':status,'score_orientation':'DD positive; STR score=final_logits[1]-final_logits[0]; H1 probability_dd logit','residual_target':'H1 train-z-logit minus STR train-z-logit; train-only means and scales','representation':'activity_raw flatten, subject_embedding, structured_embedding, concatenated subject+structured; score-only baseline','readout':'train-only StandardScaler, fixed PCA-16 randomized SVD seed 0, Ridge alpha=1','negative_control':'class-conditional permutation of train residual target only','formal_model_changed':False}
(P/'protocol.json').write_text(json.dumps(protocol,indent=2));print(json.dumps({'status':'PASS','model_seed_splits':len(a),'unique_splits':a[['outer','inner']].drop_duplicates().shape[0],'train_range':[int(a.train_n.min()),int(a.train_n.max())],'validation_range':[int(a.validation_n.min()),int(a.validation_n.max())],'stable_error_primary':status},indent=2))
