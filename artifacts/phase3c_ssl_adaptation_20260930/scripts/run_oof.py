#!/usr/bin/env python3
"""Train-only 3-fold subject OOF scores for fixed Phase-3B frozen WSSL."""
import argparse,copy,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
ROOT=Path('/home/zyt/deep_final/foundation_validation');BASE=Path('/home/zyt/deep_final/artifacts/phase3c_ssl_adaptation_20260930');P3B=Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928')
sys.path[:0]=[str(ROOT),str(P3B/'scripts'),str(BASE/'scripts')]
from src.utils.config import load_config,require_pads_classification_config
from src.engine import nested_training as nt
from src.utils.device import select_device
from src.utils.provenance import sha256_file
from phase3c_hooks import activate
from restricted_manifests import restrict

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--outer',type=int);ap.add_argument('--inner',type=int);ap.add_argument('--fold',type=int);ap.add_argument('--smoke',action='store_true');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
 assert args.seed in (42,43,44)
 manifest=json.loads((BASE/'manifest.json').read_text())
 assert sha256_file(P3B/'manifest.json')==manifest['phase3b_manifest_sha256']
 c=load_config(str(ROOT/'configs'/f'str01_seed{args.seed}.yaml'));require_pads_classification_config(c)
 activate(c,'frozen')
 c['training']['epochs']=int(manifest['oof_fixed_epochs'])
 c['training']['early_stopping']['enabled']=False
 c.setdefault('development',{})['phase3c_oof_fixed_epoch']=int(manifest['oof_fixed_epochs'])
 c['experiment']['name']=f'phase3c_oof_seed{args.seed}'
 c['experiment']['output_root']=str(BASE/('smoke_oof' if args.smoke else 'oof')/f'seed{args.seed}')
 original_scheduler=nt.build_scheduler
 def fixed_trajectory(optimizer,cfg):
  cc=copy.deepcopy(cfg);cc['training']['epochs']=50
  return original_scheduler(optimizer,cc)
 nt.build_scheduler=fixed_trajectory
 nt._metric_improved=lambda *a,**k:True
 split,audit,path,sha=nt._load_frozen_split(c);assert sha==manifest['split_sha256']
 f=pd.read_csv('/home/zyt/MFAM/data/processed/pads_multi_activity/v2_l1_full_length/manifests/CrossArms.csv',dtype={'subject_id':str})
 label=dict(zip(f.subject_id,f.label));device=select_device('cuda')
 for outer in split['outer']:
  oi=int(outer['outer_fold'])
  if args.outer is not None and oi!=args.outer:continue
  for target in outer['inner_folds']:
   ii=int(target['inner_fold'])
   if args.inner is not None and ii!=args.inner:continue
   ids=np.array(sorted(map(str,target['train_subjects'])));y=np.array([1 if label[s]=='DD' else 0 for s in ids])
   cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=20260930+3*oi+ii)
   pieces=[]
   for k,(tr,te) in enumerate(cv.split(ids,y)):
    if args.fold is not None and k!=args.fold:continue
    sub=copy.deepcopy(target);sub['inner_fold']=k;sub['train_subjects']=ids[tr].tolist();sub['validation_subjects']=ids[te].tolist()
    assert not set(sub['train_subjects'])&set(sub['validation_subjects'])
    assert not set(sub['train_subjects'])&set(target['validation_subjects'])
    assert not set(sub['validation_subjects'])&set(target['validation_subjects'])
    stage=Path(c['experiment']['output_root'])/f'outer_{oi}'/f'inner_{ii}'/f'crossfit_{k}'
    cc=copy.deepcopy(c);restrict(cc,set(sub['train_subjects'])|set(sub['validation_subjects']),BASE/'restricted_manifests/oof'/f'outer_{oi}'/f'inner_{ii}')
    oo=copy.deepcopy(outer);oo['test_subjects']=[]
    result=nt._run_inner_fold(cc,oo,sub,stage,device,resume=args.resume,smoke=args.smoke)
    assert result['best_epoch']==(1 if args.smoke else 10)
    pred=pd.read_csv(stage/'predictions/validation.csv',dtype={'subject_id':str})
    assert set(pred.subject_id).issubset(set(ids[te])) and pred.subject_id.is_unique
    if not args.smoke:assert set(pred.subject_id)==set(ids[te])
    pred=pred[['subject_id','target','probability_dd']].copy();pred['crossfit_fold']=k;pieces.append(pred)
    print(json.dumps({'seed':args.seed,'outer':oi,'inner':ii,'crossfit':k,'train_n':len(tr),'oof_n':len(te),'epoch':result['best_epoch'],'outer_test_loader_created':False}),flush=True)
   if args.fold is None and not args.smoke:
    allpred=pd.concat(pieces,ignore_index=True).sort_values('subject_id')
    assert allpred.subject_id.is_unique and set(allpred.subject_id)==set(ids)
    out=Path(c['experiment']['output_root'])/f'outer_{oi}'/f'inner_{ii}'/'oof_predictions.csv';out.parent.mkdir(parents=True,exist_ok=True);allpred.to_csv(out,index=False)
 print(json.dumps({'status':'COMPLETE','seed':args.seed,'outer_test_evaluated':False}),flush=True)
if __name__=='__main__':main()
