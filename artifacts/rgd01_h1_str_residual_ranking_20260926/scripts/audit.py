#!/usr/bin/env python3
"""Audit formal inner-development scores and reproduce deterministic H1."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
ROOT=Path('/home/zyt/deep_final');BASE=ROOT/'artifacts/rgd01_h1_str_residual_ranking_20260926';BASE.mkdir(exist_ok=True)
FV=ROOT/'foundation_validation';sys.path.insert(0,str(FV/'scripts'))
from run_h1_deep_representation_gap_diagnosis import full_estimator
EM=ROOT/'artifacts/dsg01_cross_subject_disease_subspace_20260926/extraction/embeddings'
H1FILE=Path('/home/zyt/MFAM/outputs/pads_classification/v3_analysis_and_baselines/analysis_and_baselines_20260828/baselines/H1_handcrafted_logistic/development_predictions_all.csv')
FEATURE=Path('/home/zyt/MFAM/outputs/pads_classification/v3_analysis_and_baselines/analysis_and_baselines_20260828/features/handcrafted_features.npz')
H1=pd.read_csv(H1FILE,dtype={'subject_id':str});assert len(H1)==1560 and H1.analysis_scope.eq('development_inner_cv').all()
assert set(H1.binary_label)=={0,1} and H1.loc[H1.binary_label==1,'diagnosis'].eq('DD').all() and H1.loc[H1.binary_label==0,'diagnosis'].eq('PD').all()
assert np.max(np.abs(H1.probability_pd+H1.probability_dd-1))<1e-12
with np.load(FEATURE,allow_pickle=False) as z:X=z['X'];ids=z['subject_ids'];labels=z['labels']
index={str(s):i for i,s in enumerate(ids)};assert len(index)==390
STABLE=json.loads((ROOT/'artifacts/dsg01_cross_subject_disease_subspace_20260926/analysis/stable_error_definitions.json').read_text())
assert STABLE['primary'].startswith('three seeds aggregated first')
checks=[];OUT=BASE/'h1_reproduction';OUT.mkdir(exist_ok=True)
for outer in range(5):
 for inner in range(3):
  ref=H1[(H1.outer_context==outer)&(H1.inner_fold==inner)].set_index('subject_id');assert not ref.index.has_duplicates
  all_ids=[]
  for model in ('str','v8'):
   for seed in (42,43,44):
    with np.load(EM/f'{model}_seed{seed}_outer{outer}_inner{inner}_validation.npz',allow_pickle=False) as z:
     got=pd.DataFrame({'subject_id':z['subject_id'].astype(str),'label':z['label'].astype(int),'score':z['final_logits'][:,1]-z['final_logits'][:,0]})
    assert not got.subject_id.duplicated().any() and set(got.subject_id)==set(ref.index)
    assert np.array_equal(got.set_index('subject_id').loc[ref.index].label.to_numpy(),ref.binary_label.to_numpy())
    assert np.all(np.isfinite(got.score))
    all_ids.append(tuple(sorted(got.subject_id)))
  assert len(set(all_ids))==1
  with np.load(EM/f'str_seed42_outer{outer}_inner{inner}_train.npz',allow_pickle=False) as z:
   train_ids=z['subject_id'].astype(str);train_y=z['label'].astype(int)
  assert not (set(train_ids)&set(ref.index))
  ti=np.array([index[s] for s in train_ids]);vi=np.array([index[s] for s in ref.index]);assert np.array_equal(labels[ti],train_y) and np.array_equal(labels[vi],ref.binary_label.to_numpy())
  est=full_estimator().fit(X[ti],labels[ti]);train_prob=est.predict_proba(X[ti])[:,1];val_prob=est.predict_proba(X[vi])[:,1]
  diff=float(np.max(np.abs(val_prob-ref.probability_dd.to_numpy())))
  assert diff<1e-10,(outer,inner,diff)
  np.savez_compressed(OUT/f'outer{outer}_inner{inner}.npz',train_subject_id=train_ids,train_label=train_y,train_h1_probability_dd=train_prob,validation_subject_id=ref.index.to_numpy(dtype=str),validation_label=ref.binary_label.to_numpy(dtype=int),validation_h1_probability_dd=val_prob)
  checks.append(dict(outer=outer,inner=inner,train_n=len(train_ids),validation_n=len(ref),h1_reproduction_max_probability_delta=diff,h1_auroc=roc_auc_score(ref.binary_label,val_prob),subject_label_alignment=True))
frame=pd.DataFrame(checks);frame.to_csv(BASE/'asset_audit.csv',index=False)
protocol={'scope':'fixed inner-development only','outer_signal_label_prediction_metric_accessed':False,'H1_prediction_seed_count':1,'STR_V8_seed_count':3,'H1_positive_class':'DD=1','H1_score':'probability_dd; logit used only for train-only fusion/residual target','STR_score':'final_logits[DD]-final_logits[PD]; sigmoid equals DD probability','stable_error_primary':STABLE['primary'],'H1_reproduction_max_delta':float(frame.h1_reproduction_max_probability_delta.max()),'H1_mean_split_AUROC':float(frame.h1_auroc.mean())}
(BASE/'protocol.json').write_text(json.dumps(protocol,indent=2));print(json.dumps(protocol,indent=2))
