#!/usr/bin/env python3
"""Locked H1 outer refit, once per outer fold, train-only preprocessing."""
from pathlib import Path
import csv,json,sys
import numpy as np,joblib
from sklearn.linear_model import LogisticRegression
R=Path('/home/zyt/deep_final');FV=R/'foundation_validation';B=R/'artifacts/foe01_final_locked_outer_20260926';sys.path.insert(0,str(FV))
from src.utils.provenance import sha256_file
from src.analysis.baselines import _feature_pipeline
p=B/'LOCKED_BEFORE_OUTER.json';assert sha256_file(p)==(B/'LOCKED_BEFORE_OUTER.json.sha256').read_text().split()[0];lock=json.loads(p.read_text());assert lock['status']=='LOCKED_BEFORE_OUTER_ACCESS' and lock['H1_pipeline']['test']=='outer-test once after full train fit'
feature=Path('/home/zyt/MFAM/outputs/pads_classification/v3_analysis_and_baselines/analysis_and_baselines_20260828/features/handcrafted_features.npz');assert sha256_file(feature)==lock['source_and_asset_sha256'][str(feature)]
with np.load(feature,allow_pickle=False) as z:X=z['X'];ids=z['subject_ids'].astype(str);y=z['labels'].astype(int)
idx={s:j for j,s in enumerate(ids)};assert len(idx)==390 and X.shape==(390,4928)
payload=json.loads((FV/'splits/pads_classification/v3_nested_cv/seed42/nested_cv_splits.json').read_text());seen=[]
for outer in payload['outer']:
 o=int(outer['outer_fold']);stage=B/'model_runs/H1'/f'outer_{o}';stage.mkdir(parents=True,exist_ok=True);out=stage/'predictions.csv';modelpath=stage/'h1_fitted.joblib'
 if out.exists() and modelpath.exists():
  rows=list(csv.DictReader(out.open()));assert set(r['subject_id'] for r in rows)==set(map(str,outer['test_subjects']));seen.extend(r['subject_id'] for r in rows);print(json.dumps({'status':'already_complete','model':'H1','outer_fold':o}),flush=True);continue
 train_ids=list(map(str,outer['train_subjects']));test_ids=list(map(str,outer['test_subjects']));assert not set(train_ids)&set(test_ids)
 ti=np.array([idx[s] for s in train_ids]);vi=np.array([idx[s] for s in test_ids]);est=_feature_pipeline(LogisticRegression(C=1.0,class_weight=None,max_iter=5000,solver='liblinear',random_state=42)).fit(X[ti],y[ti]);joblib.dump(est,modelpath)
 # Test scores read only after the frozen train-only pipeline is fully fitted.
 prob=est.predict_proba(X[vi])[:,1];pred=(prob>=0.5).astype(int)
 with out.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['subject_id','target','probability_pd','probability_dd','prediction','outer_fold','threshold']);w.writeheader();w.writerows({'subject_id':s,'target':int(y[j]),'probability_pd':float(1-pd),'probability_dd':float(pd),'prediction':int(pr),'outer_fold':o,'threshold':0.5} for s,j,pd,pr in zip(test_ids,vi,prob,pred))
 seen.extend(test_ids);print(json.dumps({'status':'complete','model':'H1','outer_fold':o,'test_n':len(test_ids),'checkpoint_sha256':sha256_file(modelpath)},ensure_ascii=False),flush=True)
assert len(seen)==len(set(seen))==390
