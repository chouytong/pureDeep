#!/usr/bin/env python3
"""Freeze all outer rules using source/config and inner-development artifacts only."""
from pathlib import Path
import csv,hashlib,json,sys
from datetime import datetime,timezone
import numpy as np
R=Path('/home/zyt/deep_final');FV=R/'foundation_validation';B=R/'artifacts/foe01_final_locked_outer_20260926';sys.path.insert(0,str(FV))
from src.utils.config import load_config
from src.models import build_model
from src.metrics.threshold import select_binary_threshold,final_epoch_from_inner_best
from src.utils.provenance import sha256_file
SPLIT=FV/'splits/pads_classification/v3_nested_cv/seed42/nested_cv_splits.json';SHA=SPLIT.with_suffix(SPLIT.suffix+'.sha256');expected=SHA.read_text().split()[0];assert sha256_file(SPLIT)==expected
payload=json.loads(SPLIT.read_text());assert len(payload['outer'])==5 and payload['inner_folds']==3
runs={};files={};audit=[]
for model,template in [('STR-01','str01_seed{seed}'),('V8-GN','nr01_seed{seed}')]:
 runs[model]={}
 for seed in (42,43,44):
  cfgpath=FV/'configs'/f'{template.format(seed=seed)}.yaml';cfg=load_config(str(cfgpath));params=sum(p.numel() for p in build_model(cfg).parameters());assert params==(75524 if model=='STR-01' else 71026)
  assert cfg['experiment']['seed']==seed and cfg['training']['epochs']==50 and cfg['training']['batch_size']==8 and cfg['training']['learning_rate']==2e-4 and cfg['training']['weight_decay']==1e-4
  assert cfg['loss']['class_weights']=='train_balanced' and cfg['data']['normalization_scope']=='activity_wrist_channel' and cfg['data']['standardize'] is True
  assert cfg['model_selection']['threshold']=={'source':'inner_oof_only','metric':'balanced_accuracy','positive_class':'DD'} and cfg['model_selection']['final_epoch']=={'strategy':'median_inner_best_epoch'}
  root=FV/('outputs/structured_token_residual/str01_seed'+str(seed)+'_20260921' if model=='STR-01' else 'outputs/foundation_validation/groupnorm_nr01_seed'+str(seed)+'_20260917')
  selection=[]
  for outer in payload['outer']:
   o=int(outer['outer_fold']);train=set(map(str,outer['train_subjects']));test=set(map(str,outer['test_subjects']));assert not train&test
   allrows=[];epochs=[];hashes=[]
   for inner in outer['inner_folds']:
    i=int(inner['inner_fold']);stage=root/f'outer_{o}/inner_{i}';status=json.loads((stage/'stage_status.json').read_text());assert status['status']=='complete' and status['phase']=='inner' and status['outer_test_loader_created'] is False
    epochs.append(int(status['summary']['best_epoch']));assert 1<=epochs[-1]<=50
    path=stage/'predictions/validation.csv';rows=list(csv.DictReader(path.open()));ids=[str(r['subject_id']) for r in rows];expected_ids=set(map(str,inner['validation_subjects']));assert set(ids)==expected_ids and len(ids)==len(expected_ids) and set(ids).issubset(train) and not set(ids)&test
    assert all(int(r['target']) in (0,1) and 0<=float(r['probability_dd'])<=1 for r in rows)
    allrows.extend(rows);hashes.append({'inner':i,'prediction_sha256':sha256_file(path),'best_checkpoint_sha256':status['summary']['checkpoint_sha256'],'best_epoch':epochs[-1],'validation_n':len(rows)})
   ids=[str(r['subject_id']) for r in allrows];assert len(ids)==len(set(ids))==len(train) and set(ids)==train
   th=select_binary_threshold([int(r['target']) for r in allrows],[float(r['probability_dd']) for r in allrows],metric='balanced_accuracy',zero_division=0.0)
   ep=final_epoch_from_inner_best(epochs,strategy='median_inner_best_epoch')
   selection.append({'outer_fold':o,'outer_train_n':len(train),'outer_test_n':len(test),'inner_epochs':epochs,'selected_epoch_count':int(ep['selected_epoch_count']),'threshold':float(th['threshold']),'threshold_source':th['source'],'threshold_selection_metric':th['metric'],'inner_oof_n':len(allrows),'inner_oof_prediction_files':hashes,'inner_oof_ba_at_selected_threshold':float(th['balanced_accuracy'])})
   audit.append({'model':model,'seed':seed,'outer_fold':o,'train_n':len(train),'test_n':len(test),'selected_epochs':ep['selected_epoch_count'],'threshold':th['threshold'],'inner_oof_n':len(allrows)})
  runs[model][str(seed)]={'config':str(cfgpath),'config_sha256':sha256_file(cfgpath),'params':params,'development_inner_root':str(root),'selection':selection}
  files[str(cfgpath)]=sha256_file(cfgpath)
# All matched recipe fields equal except architecture-specific structured path.
for seed in (42,43,44):
 a=load_config(str(FV/'configs'/f'str01_seed{seed}.yaml'));b=load_config(str(FV/'configs'/f'nr01_seed{seed}.yaml'))
 for field in ('training','loss','data','model_selection','nested_cv'):
  assert a[field]==b[field],(seed,field)
 bmodel=dict(b['model']);amodel=dict(a['model']);assert amodel.pop('structured_token_residual')=={'enabled':True,'projection_dim':16} and amodel==bmodel
for p in [FV/'src/engine/nested_training.py',FV/'src/metrics/threshold.py',FV/'src/analysis/baselines.py',FV/'src/models/pure_deep.py',FV/'src/datasets/folds.py',FV/'configs/pads_multi_activity_v3_nested_cv.yaml',SPLIT,FV/'scripts/run_nested_cv.py']:
 files[str(p)]=sha256_file(p)
# H1 source feature matrix is frozen; hashing bytes reveals no outcome or performance.
FEATURE=Path('/home/zyt/MFAM/outputs/pads_classification/v3_analysis_and_baselines/analysis_and_baselines_20260828/features/handcrafted_features.npz');SCHEMA=FEATURE.parent/'feature_schema.json';assert FEATURE.is_file() and SCHEMA.is_file();files[str(FEATURE)]=sha256_file(FEATURE);files[str(SCHEMA)]=sha256_file(SCHEMA)
assert len(audit)==30
lock={'schema_version':1,'status':'LOCKED_BEFORE_OUTER_ACCESS','created_at_utc':datetime.now(timezone.utc).isoformat(),'outer_information_accessed_for_this_protocol':False,'historical_outer_artifacts':'isolated; never used for this protocol or model selection','source_and_asset_sha256':files,'split_sha256':expected,'task':'PADS PD=0 versus DD=1; 390 subjects, 5 outer folds each tested once','model_roles':{'STR-01':'primary','V8-GN':'matched deep reference','H1':'handcrafted statistical reference'},'deep_models':runs,'architecture':{'STR-01_params':75524,'V8-GN_params':71026,'STR_change':'shared Linear(258,16)+GELU on each of 11 bilateral activity tokens, ordered flatten 176D, Linear(176,2) residual logits added to V8-GN base logits; zero initialization of residual head'},'recipe':{'full_bilateral':True,'activity_order':a['data']['activities'],'wrist_order':['left','right'],'input_channels':['AccX','AccY','AccZ','GyroX','GyroY','GyroZ'],'train_only_normalization':'per activity/wrist/channel mean+std fitted on corresponding outer-train subjects, epsilon 1e-6','loss':'train-balanced cross entropy, label smoothing 0','optimizer':'AdamW lr 2e-4, weight decay 1e-4, betas 0.9/0.999','scheduler':'cosine minimum lr 1e-6','batch_size':8,'maximum_inner_epochs':50,'inner_early_stopping':'validation balanced accuracy, patience 12, minimum delta 0, best checkpoint','outer_final_refit':'all corresponding outer-train subjects; train-only normalization and class weights; no validation/test early stopping; fixed epochs = rounded-up median of three inner best one-based epochs; final checkpoint only','deep_seeds':[42,43,44],'deterministic':True},'decision_rule':{'deep':'per model/seed/outer fold DD probability >= threshold maximizing inner OOF BA; exact candidate/tie rules in frozen src/metrics/threshold.py; no outer threshold fitting','H1':'DD probability >= 0.5','secondary_deep':'default argmax/0.5 for descriptive matched development comparison only'},'H1_pipeline':{'features':'frozen 4928D label-independent handcrafted matrix with fixed schema','fit':'outer-train only median imputer -> zero-variance filter -> StandardScaler -> LogisticRegression(C=1,class_weight=None,solver=liblinear,max_iter=5000,random_state=42)','test':'outer-test once after full train fit'},'metrics':{'primary':'balanced_accuracy','secondary':['accuracy','auroc','macro_f1','pd_recall','dd_recall','confusion_matrix'],'primary_aggregation':'per outer fold metric, average 3 deep seed metrics inside fold; then 5 outer folds as paired statistical units','confidence_interval':'paired bootstrap resampling the 5 outer folds, 10000 replicates, fixed RNG seed 20260926; report wide intervals and no claim of independent seeds/subject pairs','pooled_390':'descriptive only; each outer subject tested once per seed; no seed probability ensemble'},'comparison':'STR-01 minus V8-GN and STR-01 minus H1 on matched five folds; development vs outer descriptive, with default-0.5 secondary to match development decision','prohibited':'No outer-based model selection, hyperparameter/threshold/epoch/recipe change, diagnostic probes, fusion, correction, activity or mechanism search'}
path=B/'LOCKED_BEFORE_OUTER.json';path.write_text(json.dumps(lock,indent=2,ensure_ascii=False));h=sha256_file(path);(B/'LOCKED_BEFORE_OUTER.json.sha256').write_text(h+'  '+path.name+'\n')
with (B/'prelock_selection_audit.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(audit[0]));w.writeheader();w.writerows(audit)
print(json.dumps({'status':'LOCKED_BEFORE_OUTER_ACCESS','lock_sha256':h,'models':list(runs),'deep_refit_units':len(audit),'epoch_range':[min(x['selected_epochs'] for x in audit),max(x['selected_epochs'] for x in audit)],'outer_information_accessed':False},indent=2))
