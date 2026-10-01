#!/usr/bin/env python3
"""Phase-3C only fixed inner-development trainer; no outer evaluation."""
import argparse,copy,hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path('/home/zyt/deep_final/foundation_validation')
P3B=Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928')
BASE=Path('/home/zyt/deep_final/artifacts/phase3c_ssl_adaptation_20260930')
sys.path[:0]=[str(ROOT),str(P3B/'scripts'),str(BASE/'scripts')]
from src.utils.config import load_config,require_pads_classification_config
from src.engine.nested_training import _load_frozen_split,_run_inner_fold
from src.utils.device import select_device
from src.utils.provenance import sha256_file
from phase3c_hooks import activate
from restricted_manifests import restrict

def subset(ids,outer,inner,fraction):
 if fraction==100:return list(ids)
 f=pd.read_csv('/home/zyt/MFAM/data/processed/pads_multi_activity/v2_l1_full_length/manifests/CrossArms.csv',dtype={'subject_id':str})
 labels=dict(zip(f.subject_id,f.label));rng=np.random.default_rng(20260930+3*outer+inner)
 chosen=[]
 for label in ('PD','DD'):
  group=sorted(s for s in ids if labels[s]==label)
  order=rng.permutation(group);n=max(1,int(round(len(group)*fraction/100)))
  chosen.extend(order[:n].tolist())
 assert len(chosen)==len(set(chosen)) and set(chosen).issubset(set(ids))
 return sorted(chosen)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=('adapt','multi','str','frozen'),required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--fraction',type=int,choices=(25,50,75,100),default=100);ap.add_argument('--outer',type=int);ap.add_argument('--inner',type=int);ap.add_argument('--smoke',action='store_true');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
 assert args.seed in (42,43,44)
 manifest=json.loads((BASE/'manifest.json').read_text())
 assert sha256_file(P3B/'manifest.json')==manifest['phase3b_manifest_sha256']
 assert sha256_file(BASE/'analysis/stage4_windows.npz')==manifest['stage4_cache_sha256']
 c=load_config(str(ROOT/'configs'/f'str01_seed{args.seed}.yaml'));require_pads_classification_config(c)
 assert c['model']['structured_token_residual']['enabled'] is True
 cache=activate(c,args.variant)
 c.setdefault('development',{})['phase3c_variant']=args.variant;c['development']['train_fraction_percent']=args.fraction
 c['experiment']['name']=f'phase3c_{args.variant}_p{args.fraction}_seed{args.seed}'
 c['experiment']['output_root']=str(BASE/('smoke' if args.smoke else 'runs')/args.variant/f'p{args.fraction}'/f'seed{args.seed}')
 split,audit,split_path,sha=_load_frozen_split(c);assert sha==manifest['split_sha256']
 device=select_device('cuda')
 for outer in split['outer']:
  oi=int(outer['outer_fold'])
  if args.outer is not None and oi!=args.outer:continue
  for orig in outer['inner_folds']:
   ii=int(orig['inner_fold'])
   if args.inner is not None and ii!=args.inner:continue
   inner=copy.deepcopy(orig);inner['train_subjects']=subset(orig['train_subjects'],oi,ii,args.fraction)
   assert not set(inner['train_subjects'])&set(inner['validation_subjects'])
   stage=Path(c['experiment']['output_root'])/f'outer_{oi}'/f'inner_{ii}'
   if args.fraction<100:
    cc=copy.deepcopy(c);restrict(cc,set(inner['train_subjects'])|set(inner['validation_subjects']),BASE/'restricted_manifests/e4'/f'p{args.fraction}'/f'outer_{oi}'/f'inner_{ii}')
    oo=copy.deepcopy(outer);oo['test_subjects']=[]
   else:cc=c;oo=outer
   result=_run_inner_fold(cc,oo,inner,stage,device,resume=args.resume,smoke=args.smoke)
   print(json.dumps({'variant':args.variant,'fraction':args.fraction,'seed':args.seed,'outer':oi,'inner':ii,'train_n':len(inner['train_subjects']),'best_epoch':result['best_epoch'],'ba':result['validation_metrics']['balanced_accuracy'],'auroc':result['validation_metrics']['macro_auroc'],'outer_test_loader_created':False}),flush=True)
 print(json.dumps({'status':'COMPLETE','variant':args.variant,'fraction':args.fraction,'seed':args.seed,'outer_test_evaluated':False}),flush=True)
if __name__=='__main__':main()
