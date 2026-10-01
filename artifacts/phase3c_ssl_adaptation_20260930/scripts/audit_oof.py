import json,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
ROOT=Path('/home/zyt/deep_final/foundation_validation');BASE=Path('/home/zyt/deep_final/artifacts/phase3c_ssl_adaptation_20260930')
sys.path.insert(0,str(ROOT))
from src.utils.config import load_config
from src.engine.nested_training import _load_frozen_split
split,_,_,sha=_load_frozen_split(load_config(str(ROOT/'configs/str01_seed42.yaml')))
assert sha==json.loads((BASE/'manifest.json').read_text())['split_sha256']
f=pd.read_csv('/home/zyt/MFAM/data/processed/pads_multi_activity/v2_l1_full_length/manifests/CrossArms.csv',dtype={'subject_id':str});labels=dict(zip(f.subject_id,f.label))
rows=[]
for outer in split['outer']:
 oi=int(outer['outer_fold'])
 for target in outer['inner_folds']:
  ii=int(target['inner_fold']);ids=np.array(sorted(map(str,target['train_subjects'])));y=np.array([1 if labels[s]=='DD' else 0 for s in ids]);target_val=set(target['validation_subjects'])
  cv=list(StratifiedKFold(n_splits=3,shuffle=True,random_state=20260930+3*oi+ii).split(ids,y))
  for seed in (42,43,44):
   allpred=pd.read_csv(BASE/f'oof/seed{seed}/outer_{oi}/inner_{ii}/oof_predictions.csv',dtype={'subject_id':str})
   assert allpred.subject_id.is_unique and set(allpred.subject_id)==set(ids) and not set(allpred.subject_id)&target_val
   for k,(tr,te) in enumerate(cv):
    stage=BASE/f'oof/seed{seed}/outer_{oi}/inner_{ii}/crossfit_{k}'
    status=json.loads((stage/'stage_status.json').read_text());sp=json.loads((stage/'split.json').read_text())
    assert status['status']=='complete' and status['summary']['best_epoch']==10 and status['outer_test_loader_created'] is False
    assert set(sp['train_subjects'])==set(ids[tr]) and set(sp['validation_subjects'])==set(ids[te]) and sp['test_subjects']==[]
    pred=pd.read_csv(stage/'predictions/validation.csv',dtype={'subject_id':str})
    assert pred.subject_id.is_unique and set(pred.subject_id)==set(ids[te])
    rows.append(dict(seed=seed,outer=oi,inner=ii,fold=k,train_n=len(tr),oof_n=len(te),epoch=10,outer_test_loader_created=False))
assert len(rows)==135
pd.DataFrame(rows).to_csv(BASE/'analysis/e2_oof_integrity.csv',index=False)
print(json.dumps({'crossfit_models_verified':135,'target_inner_train_oof_coverage':135,'target_validation_subjects_used_in_crossfit':False,'oof_labels_used_for_checkpoint_selection':False,'outer_test_loader_created':False},indent=2))
