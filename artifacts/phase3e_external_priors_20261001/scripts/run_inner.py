#!/usr/bin/env python3
"""Phase-3D bounded supervised inner-development runner; no outer loader."""
import argparse,copy,json,sys
from pathlib import Path
BASE=Path('/home/zyt/deep_final/artifacts/phase3e_external_priors_20261001')
ROOT=Path('/home/zyt/deep_final/foundation_validation')
P3B=Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928')
sys.path[:0]=[str(ROOT),str(P3B/'scripts'),str(BASE/'scripts')]
from src.utils.config import load_config,require_pads_classification_config
from src.engine.nested_training import _load_frozen_split,_run_inner_fold
from src.utils.device import select_device
from src.utils.provenance import sha256_file
import phase3e_hooks as hooks
from restricted_manifests import restrict

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=('harnet5','biopm_only','biopm','biopm_random'),required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--outer',type=int);ap.add_argument('--inner',type=int);ap.add_argument('--smoke',action='store_true');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
 assert args.seed in (42,43,44)
 manifest=json.loads((BASE/'manifest.json').read_text())
 assert sha256_file(P3B/'manifest.json')==manifest['phase3b_manifest_sha256']
 assert sha256_file(P3B/'analysis/ssl_embeddings_all.npz')==manifest['embedding_sha256']
 c=load_config(str(ROOT/'configs'/f'str01_seed{args.seed}.yaml'));require_pads_classification_config(c)
 assert c['model']['structured_token_residual']['enabled'] is True
 hooks.FEATURE_DIM=512 if args.variant=='harnet5' else 384
 hooks.CACHE_PATH=BASE/'analysis'/('harnet5_embeddings_all.npz' if args.variant=='harnet5' else 'biopm_embeddings_all.npz')
 cache=hooks.FrozenEmbeddingCache()
 hooks.activate(c,{'embedding_kind':'random' if args.variant=='biopm_random' else 'pretrained','model_kind':'ssl_only' if args.variant=='biopm_only' else 'str_residual'},cache)
 c.setdefault('development',{})['phase3e_variant']=args.variant
 c['experiment']['name']=f'phase3e_{args.variant}_seed{args.seed}'
 c['experiment']['output_root']=str(BASE/('smoke' if args.smoke else 'runs')/args.variant/f'seed{args.seed}')
 split,audit,split_path,sha=_load_frozen_split(c);assert sha==manifest['split_sha256']
 device=select_device('cuda')
 for outer in split['outer']:
  oi=int(outer['outer_fold'])
  if args.outer is not None and oi!=args.outer:continue
  for orig in outer['inner_folds']:
   ii=int(orig['inner_fold'])
   if args.inner is not None and ii!=args.inner:continue
   inner=copy.deepcopy(orig)
   assert not set(inner['train_subjects'])&set(inner['validation_subjects'])
   cc=copy.deepcopy(c)
   restrict(cc,set(inner['train_subjects'])|set(inner['validation_subjects']),BASE/'restricted_manifests'/f'outer_{oi}'/f'inner_{ii}')
   oo=copy.deepcopy(outer);oo['test_subjects']=[]
   stage=Path(c['experiment']['output_root'])/f'outer_{oi}'/f'inner_{ii}'
   result=_run_inner_fold(cc,oo,inner,stage,device,resume=args.resume,smoke=args.smoke)
   print(json.dumps({'variant':args.variant,'seed':args.seed,'outer':oi,'inner':ii,'train_n':len(inner['train_subjects']),'best_epoch':result['best_epoch'],'ba':result['validation_metrics']['balanced_accuracy'],'auroc':result['validation_metrics']['macro_auroc'],'outer_test_loader_created':False}),flush=True)
 print(json.dumps({'status':'COMPLETE','variant':args.variant,'seed':args.seed,'outer_test_evaluated':False}),flush=True)
if __name__=='__main__':main()
