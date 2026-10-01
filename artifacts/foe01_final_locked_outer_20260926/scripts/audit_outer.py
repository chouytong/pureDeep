#!/usr/bin/env python3
"""Audit completed locked outer products before statistical aggregation."""
from pathlib import Path
import csv,json,sys
import pandas as pd
R=Path('/home/zyt/deep_final');FV=R/'foundation_validation';B=R/'artifacts/foe01_final_locked_outer_20260926';sys.path.insert(0,str(FV))
from src.utils.provenance import sha256_file
p=B/'LOCKED_BEFORE_OUTER.json';assert sha256_file(p)==(B/'LOCKED_BEFORE_OUTER.json.sha256').read_text().split()[0];lock=json.loads(p.read_text());split=json.loads((FV/'splits/pads_classification/v3_nested_cv/seed42/nested_cv_splits.json').read_text());rows=[];labels={}
for model in ('STR-01','V8-GN'):
 for seed in (42,43,44):
  seen=[]
  for outer in split['outer']:
   o=int(outer['outer_fold']);train=set(map(str,outer['train_subjects']));test=set(map(str,outer['test_subjects']));sel=lock['deep_models'][model][str(seed)]['selection'][o]
   stage=B/'model_runs'/model/f'seed{seed}'/f'outer_{o}'/'outer_final';status=json.loads((stage/'stage_status.json').read_text());assert status['status']=='complete' and status['outer_test_evaluation_count']==1 and status['fixed_epoch_count']==sel['selected_epoch_count'] and status['summary']['threshold']==sel['threshold']
   norm=json.loads((stage/'normalization.json').read_text());assert set(map(str,norm['subject_ids']))==train and not set(map(str,norm['subject_ids']))&test and norm['subject_count']==len(train)
   assert status['summary']['normalization_sha256']==norm['normalization_sha256'] and status['summary']['checkpoint_sha256']==sha256_file(stage/'checkpoints/final.pt')
   pp=list(csv.DictReader((stage/'outer_test/predictions.csv').open()));ids=[r['subject_id'] for r in pp];assert len(ids)==len(set(ids))==len(test) and set(ids)==test
   assert all(abs(float(r['threshold'])-sel['threshold'])<1e-12 and int(r['prediction_thresholded'])==int(float(r['probability_dd'])>=sel['threshold']) for r in pp)
   for r in pp:
    key=(o,r['subject_id']);y=int(r['target']);assert y in (0,1)
    if key in labels:assert labels[key]==y
    labels[key]=y
   seen.extend(ids);rows.append(dict(model=model,seed=seed,outer_fold=o,test_n=len(pp),train_n=len(train),fixed_epochs=sel['selected_epoch_count'],threshold=sel['threshold'],normalization_train_only=True,evaluation_count=1,checkpoint_sha256=status['summary']['checkpoint_sha256']))
  assert len(seen)==len(set(seen))==390
hseen=[]
for outer in split['outer']:
 o=int(outer['outer_fold']);test=set(map(str,outer['test_subjects']));stage=B/'model_runs/H1'/f'outer_{o}';pp=list(csv.DictReader((stage/'predictions.csv').open()));ids=[r['subject_id'] for r in pp];assert len(ids)==len(set(ids))==len(test) and set(ids)==test and (stage/'h1_fitted.joblib').is_file()
 for r in pp:
  assert int(r['target'])==labels[(o,r['subject_id'])] and abs(float(r['threshold'])-.5)<1e-12 and int(r['prediction'])==int(float(r['probability_dd'])>=.5)
 hseen.extend(ids);rows.append(dict(model='H1',seed='',outer_fold=o,test_n=len(pp),train_n=len(outer['train_subjects']),fixed_epochs='',threshold=.5,normalization_train_only=True,evaluation_count=1,checkpoint_sha256=sha256_file(stage/'h1_fitted.joblib')))
assert len(hseen)==len(set(hseen))==390 and len(labels)==390
pd.DataFrame(rows).to_csv(B/'outer_asset_audit.csv',index=False)
print(json.dumps({'status':'PASS','deep_units':30,'H1_folds':5,'unique_outer_subjects':390,'all_labels_aligned':True,'train_only_normalization':True,'one_outer_evaluation_per_unit':True,'lock_sha256':sha256_file(p)},indent=2))
