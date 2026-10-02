"""B1 raw local Acc, frozen original WSSL architecture/recipe, no outer data."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent.parent
F=Path('/home/zyt/deep_final/foundation_validation');sys.path.insert(0,str(F))
from src.engine import nested_training as nt
from src.utils.config import load_config,require_pads_classification_config
from src.utils.device import select_device
from raw_acc_hooks import FrozenEmbeddingCache,activate,raw_index


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,choices=[42,43,44],required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--resume',action='store_true');ap.add_argument('--context',type=int);ap.add_argument('--inner',type=int);args=ap.parse_args()
    lock=json.loads((HERE/'analysis/baseline_lock.json').read_text())
    for path,digest in lock['source_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    cache=FrozenEmbeddingCache();index=raw_index()
    assert hashlib.sha256(Path(lock['cache_path']).read_bytes()).hexdigest()==lock['cache_sha256']
    cfg=load_config(str(F/f'configs/str01_seed{args.seed}.yaml'));require_pads_classification_config(cfg)
    cfg['data']['local_acc_preprocessing']='B1 raw posttrim48, original processed Gyro'
    cfg['development']=dict(phase4_variant='b1_raw',raw_input_audit=json.loads((HERE/'analysis/b1_raw_input_audit.json').read_text()),frozen_embedding_sha256=lock['cache_sha256'],no_outer_access=True)
    cfg['experiment']['name']=f'phase4_b1_raw_seed{args.seed}'
    cfg['experiment']['output_root']=str(HERE/('smoke' if args.smoke else 'runs')/'b1_raw'/f'seed{args.seed}')
    activate(cfg,cache,index)
    split,_,_,digest=nt._load_frozen_split(cfg);assert digest==lock['split_sha256']
    device=select_device('cuda')
    for context in split['outer']:
        oi=int(context['outer_fold'])
        if args.context is not None and oi!=args.context:continue
        for inner in context['inner_folds']:
            ii=int(inner['inner_fold'])
            if args.inner is not None and ii!=args.inner:continue
            stage=Path(cfg['experiment']['output_root'])/f'outer_{oi}/inner_{ii}'
            summary=nt._run_inner_fold(cfg,{'outer_fold':oi,'test_subjects':[]},inner,stage,device,resume=args.resume,smoke=args.smoke)
            oldstage=Path(lock['baseline_root'])/f'seed{args.seed}/outer_{oi}/inner_{ii}'
            old=json.loads((oldstage/'normalization.json').read_text());new=json.loads((stage/'normalization.json').read_text())
            for key in ('mean','std'):assert np.array_equal(np.asarray(old[key])[:,:,3:],np.asarray(new[key])[:,:,3:]),(oi,ii,key,'Gyro statistics changed')
            assert old['subject_ids_sha256']==new['subject_ids_sha256']==summary['train_subject_ids_sha256']
            assert old['normalization_sha256']!=new['normalization_sha256']
            print(json.dumps(dict(variant='b1_raw',seed=args.seed,context=oi,inner=ii,best_epoch=summary['best_epoch'],gyro_normalization_exact=True,train_subject_hash_matched=True,raw_acc_normalization_refitted=True,outer_test_loader_created=False)),flush=True)
    print(json.dumps(dict(status='COMPLETE',variant='b1_raw',seed=args.seed,smoke=args.smoke)),flush=True)


if __name__=='__main__':main()
