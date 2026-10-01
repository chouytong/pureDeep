"""Read-only E4 subject-access, frozen-BN, feature and checkpoint audit."""
import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
BASE=Path('/home/zyt/deep_final/artifacts/phase3d_ssl_adaptation_20261001')
ROOT=Path('/home/zyt/deep_final/foundation_validation')
sys.path.insert(0,str(ROOT))
from src.utils.config import load_config
from src.engine.nested_training import _load_frozen_split
split,_,_,sha=_load_frozen_split(load_config(str(ROOT/'configs/str01_seed42.yaml')))
assert sha==json.loads((BASE/'manifest.json').read_text())['split_sha256']
rows=[]
for outer in split['outer']:
 oi=int(outer['outer_fold'])
 for inner in outer['inner_folds']:
  ii=int(inner['inner_fold']);train=set(map(str,inner['train_subjects']));val=set(map(str,inner['validation_subjects']));assert not train&val
  raw=BASE/f'domain_raw/outer_{oi}/inner_{ii}';ra=json.loads((raw/'audit.json').read_text())
  assert set(ra['raw_subjects_opened'])==train|val and set(ra['train_subject_ids'])==train and set(ra['validation_subject_ids'])==val and ra['outer_raw_subjects_opened']==[]
  for role in ('train','validation'):
   assert hashlib.sha256((raw/f'{role}.npz').read_bytes()).hexdigest()==ra[f'{role}_sha256']
  for seed in (42,43,44):
   stage=BASE/f'domain_adaptation/seed{seed}/outer_{oi}/inner_{ii}';a=json.loads((stage/'adapt_status.json').read_text());f=stage/'features.npz';pt=stage/'adapted_layer5.pt'
   assert a['status']=='complete' and a['epochs']==5 and a['validation_windows_in_adaptation']==0 and a['outer_raw_access'] is False
   assert a['bn_running_stats_unchanged'] and a['layer5_parameter_delta_l2']>0
   assert a['train_n']==len(train) and a['validation_n']==len(val)
   assert hashlib.sha256(f.read_bytes()).hexdigest()==a['feature_sha256']
   assert hashlib.sha256(pt.read_bytes()).hexdigest()==a['layer5_sha256']
   z=np.load(f,allow_pickle=False)
   assert set(map(str,z['train_subject_ids']))==train and set(map(str,z['validation_subject_ids']))==val
   assert set(map(str,z['subject_ids']))==train|val and z['features'].shape==(len(train)+len(val),11,2,1024)
   assert np.isfinite(z['features']).all()
   rows.append(dict(seed=seed,outer=oi,inner=ii,train_n=len(train),validation_n=len(val),train_windows=ra['train_windows'],validation_windows_in_adaptation=0,layer5_delta_l2=a['layer5_parameter_delta_l2'],bn_unchanged=True,outer_raw_access=False,feature_sha256=a['feature_sha256']))
assert len(rows)==45
pd.DataFrame(rows).to_csv(BASE/'analysis/domain_integrity.csv',index=False)
print(json.dumps({'adaptations_verified':45,'raw_subject_access_fold_local':True,'validation_windows_in_adaptation':0,'bn_running_stats_unchanged':True,'feature_sha_match':True,'outer_raw_access':False},indent=2))
