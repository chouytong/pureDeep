#!/usr/bin/env python3
"""PAG-01 fixed development-only STR asset and status audit."""
from pathlib import Path
import json
import numpy as np,pandas as pd
R=Path('/home/zyt/deep_final');B=R/'artifacts/pag01_preaggregation_disease_info_20260926';E=R/'artifacts/dsg01_cross_subject_disease_subspace_20260926/extraction/embeddings';S=R/'artifacts/dsg01_cross_subject_disease_subspace_20260926/analysis/error_consistency_subjects.csv'
status=pd.read_csv(S,dtype={'subject_id':str});status=status[status.model=='str'];assert len(status)==390 and status.primary_status.value_counts().to_dict()=={'stable_correct':284,'stable_error':74,'unstable':32}
rows=[]
for o in range(5):
 for i in range(3):
  refs=[]
  for seed in (42,43,44):
   with np.load(E/f'str_seed{seed}_outer{o}_inner{i}_train.npz',allow_pickle=False) as t,np.load(E/f'str_seed{seed}_outer{o}_inner{i}_validation.npz',allow_pickle=False) as v:
    ids0=t['subject_id'].astype(str);ids1=v['subject_id'].astype(str);y0=t['label'].astype(int);y1=v['label'].astype(int)
    assert len(set(ids0))==len(ids0) and len(set(ids1))==len(ids1) and not set(ids0)&set(ids1)
    assert t['activity_raw'].shape==(len(ids0),11,258) and v['activity_raw'].shape==(len(ids1),11,258)
    assert t['activity_context'].shape==(len(ids0),11,258) and v['activity_context'].shape==(len(ids1),11,258)
    for z,n in ((t,len(ids0)),(v,len(ids1))):
     assert z['subject_embedding'].shape==(n,258) and z['structured_embedding'].shape==(n,176) and z['final_logits'].shape==(n,2)
    assert np.all(np.isfinite(t['final_logits'])) and np.all(np.isfinite(v['final_logits']))
    assert set(ids1).issubset(set(status.subject_id))
    refs.append((set(ids0),set(ids1),dict(zip(ids1,y1))))
    rows.append(dict(outer=o,inner=i,seed=seed,train_n=len(ids0),validation_n=len(ids1),train_pd=int(sum(y0==0)),train_dd=int(sum(y0==1)),validation_pd=int(sum(y1==0)),validation_dd=int(sum(y1==1))))
  assert refs[0]==refs[1]==refs[2]
a=pd.DataFrame(rows);assert len(a)==45 and a[['outer','inner']].drop_duplicates().shape[0]==15;a.to_csv(B/'asset_audit.csv',index=False)
protocol={'scope':'fixed inner-development only','outer_information_used':False,'formal_STR_frozen':True,'seeds':[42,43,44],'split_count':15,'primary_unit':'15 splits after mean across 3 seeds','disease_positive_class':'DD=1','STR_score':'final_logits[:,1]-final_logits[:,0]','stages':['activity_raw','activity_context','subject_embedding','structured_embedding','decision_input'],'score_baseline':'STR final DD-PD logit as covariate','primary_pca_components':16,'sensitivity_pca_components':[8,32],'disease_probe':'train-only LogisticRegression C=1 liblinear, standardized score+train-only standardized PCA features','increment_signal':'inner-train 5-fold subject-level cross-fitted disease logit(activity_raw+score)-disease logit(score-only)','signal_predictor':'train-only Ridge alpha=1 on standardized score+fixed PCA features','ambiguity':'abs(5-fold OOF final-score-only disease logit) tertiles from inner-train; validation scored by full-train model','conditional_permutation':'shuffle representation PC rows among inner-train subjects within STR-score deciles only','stable_error':'DSG-01 primary status sensitivity only, never defines ambiguity'}
(B/'protocol.json').write_text(json.dumps(protocol,indent=2));print(json.dumps({'status':'PASS','split_seed_instances':45,'unique_splits':15,'status_counts':status.primary_status.value_counts().to_dict(),'outer_information_used':False},indent=2))
