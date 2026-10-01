#!/usr/bin/env python3
"""Phase-3B fixed inner-development runner; no outer evaluation loader."""
import argparse,json,sys
from pathlib import Path
ROOT=Path('/home/zyt/deep_final/foundation_validation')
sys.path.insert(0,str(ROOT))
from src.utils.config import load_config,require_pads_classification_config
from src.engine.nested_training import _load_frozen_split,_run_inner_fold
from src.utils.device import select_device
from src.utils.provenance import sha256_file
from transfer_hooks import FrozenEmbeddingCache,activate,CACHE_PATH

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--variant',required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--resume',action='store_true');ap.add_argument('--smoke',action='store_true');ap.add_argument('--outer',type=int);ap.add_argument('--inner',type=int);args=ap.parse_args()
 manifest=json.loads(Path(args.manifest).read_text());v=manifest['variants'][args.variant];assert args.seed in (42,43,44)
 assert sha256_file(CACHE_PATH)==manifest['embedding_sha256']
 config=load_config(str(ROOT/'configs'/f'str01_seed{args.seed}.yaml'));require_pads_classification_config(config)
 assert config['model']['structured_token_residual']['enabled'] is True
 cache=FrozenEmbeddingCache();activate(config,v,cache)
 config.setdefault('development',{})['phase3b_variant']=args.variant
 config['development']['frozen_embedding_sha256']=manifest['embedding_sha256']
 config['experiment']['name']=f"phase3b_{args.variant}_seed{args.seed}"
 config['experiment']['output_root']=str(Path(args.manifest).parent/('smoke' if args.smoke else 'runs')/args.variant/f'seed{args.seed}')
 split,audit,split_path,split_sha=_load_frozen_split(config)
 assert split_sha==manifest['split_sha256']
 out=Path(config['experiment']['output_root']);out.mkdir(parents=True,exist_ok=True)
 device=select_device('cuda')
 for outer in split['outer']:
  oi=int(outer['outer_fold'])
  if args.outer is not None and oi!=args.outer:continue
  for inner in outer['inner_folds']:
   ii=int(inner['inner_fold'])
   if args.inner is not None and ii!=args.inner:continue
   stage=out/f'outer_{oi}'/f'inner_{ii}'
   summary=_run_inner_fold(config,outer,inner,stage,device,resume=args.resume,smoke=args.smoke)
   print(json.dumps({'variant':args.variant,'seed':args.seed,'outer':oi,'inner':ii,'best_epoch':summary['best_epoch'],'ba':summary['validation_metrics']['balanced_accuracy'],'auroc':summary['validation_metrics']['macro_auroc'],'outer_test_loader_created':False}),flush=True)
 print(json.dumps({'status':'COMPLETE','variant':args.variant,'seed':args.seed,'outer_test_evaluated':False}),flush=True)
if __name__=='__main__':main()
