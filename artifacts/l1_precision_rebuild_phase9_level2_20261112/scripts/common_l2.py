from pathlib import Path
import sys,json,hashlib,math,time
import numpy as np
import pandas as pd
ROOT=Path('/home/zyt/deep_final');MAIN=ROOT/'mainline';P9=ROOT/'artifacts/pads_raw_preprocessing_audit_20261112';P8=ROOT/'artifacts/wssl_guided_optimization_phase8_20261111';P6=ROOT/'artifacts/wssl_representation_generalization_phase6_20261111'
HERE=ROOT/'artifacts/l1_precision_rebuild_phase9_level2_20261112';PRIVATE=HERE/'restricted_manifests';RAW=ROOT/'data/raw/pads';OLD=ROOT/'data/processed/pads_multi_activity/v2_l1_full_length'
START='4da80813a98586acf23a89507359e4ca93689289';VERSION='pads_l1_lambda50_certified_v1';SEED=20261112
ACTS=['CrossArms','DrinkGlas','Entrainment','HoldWeight','LiftHold','PointFinger','Relaxed','RelaxedTask','StretchHold','TouchIndex','TouchNose'];SIDES=['left','right'];CHANNELS=['AccX','AccY','AccZ','GyroX','GyroY','GyroZ'];METRICS=['accuracy','ba','auroc','macro_f1','pd_recall','dd_recall','nll','brier','ece']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not math.isfinite(x):return None
 return x
def write(p,d):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(clean(d),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(f):
 def fmt(x):return 'NA' if pd.isna(x) else f'{x:.6g}' if isinstance(x,float) else str(x).replace('|','/')
 return '| '+' | '.join(f.columns)+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+''.join('| '+' | '.join(map(fmt,r))+' |\n' for r in f.itertuples(index=False,name=None))
def freeze(name,paths,**extra):
 p=HERE/'configs'/f'{name}_FREEZE.json';assert not p.exists();write(p,dict(files={str(p.relative_to(HERE)):sha(p) for p in paths},**extra))
def guard():
 d=load(PRIVATE/'manifests/protected_assets.json')['sha256']
 for p,h in d.items():assert sha(p)==h,'Protected changed '+p
 return len(d)
def splits():
 d=load(MAIN/'splits/pads_classification/v3_nested_cv/seed42/nested_cv_splits.json');return [(int(o['outer_fold']),int(i['inner_fold']),i,list(map(str,o['test_subjects']))) for o in d['outer'] for i in o['inner_folds']]
def records():return load(P9/'restricted_manifests/records.json')['records']
def baseline(model,seed,c,i):
 return ROOT/(f'artifacts/phase3b_external_ssl_20260928/runs/b_str_pretrained/seed{seed}/outer_{c}/inner_{i}' if model=='wssl' else f'foundation_validation/outputs/structured_token_residual/str01_seed{seed}_20260921/outer_{c}/inner_{i}')
