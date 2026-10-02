"""C1 conditional, fixed weight, original WSSL architecture and data paths."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent.parent
F=Path('/home/zyt/deep_final/foundation_validation');sys.path.insert(0,str(F))
from src.engine import nested_training as nt
from src.utils.config import load_config
from src.utils.device import select_device
from dd_aux_hooks import FrozenEmbeddingCache,activate


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,choices=[42,43,44],required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--resume',action='store_true');ap.add_argument('--context',type=int);ap.add_argument('--inner',type=int);args=ap.parse_args()
    decision=json.loads((HERE/'analysis/wssl_error_analysis_decision.json').read_text());assert decision['c1_eligible'] is True
    lock=json.loads((HERE/'analysis/baseline_lock.json').read_text())
    for path,digest in lock['source_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    assert hashlib.sha256(Path(lock['cache_path']).read_bytes()).hexdigest()==lock['cache_sha256']
    cfg=load_config(str(F/f'configs/str01_seed{args.seed}.yaml'))
    cfg['model']['dd_training_auxiliary']={'enabled':True,'input':'original_subject_embedding','input_dim':258,'classes':4}
    cfg['loss']['dd_auxiliary_weight']=.1
    cfg['development']=dict(phase4_variant='c1_dd_aux',source_categories=decision['source_categories'],auxiliary_training_only=True,no_inference_subtype_labels=True,no_outer_access=True,frozen_embedding_sha256=lock['cache_sha256'])
    cfg['experiment']['name']=f'phase4_c1_dd_aux_seed{args.seed}'
    cfg['experiment']['output_root']=str(HERE/('smoke' if args.smoke else 'runs')/'c1_dd_aux'/f'seed{args.seed}')
    split,_,_,digest=nt._load_frozen_split(cfg);assert digest==lock['split_sha256']
    labels=json.loads((HERE/'features/dd_source_category_labels.json').read_text())
    holder={};activate(cfg,FrozenEmbeddingCache(),holder)
    for context in split['outer']:
        oi=int(context['outer_fold'])
        if args.context is not None and oi!=args.context:continue
        for inner in context['inner_folds']:
            ii=int(inner['inner_fold'])
            if args.inner is not None and ii!=args.inner:continue
            holder['training_ids']=set(inner['train_subjects'])
            holder['dd_labels']={s:labels[s] for s in holder['training_ids'] if s in labels}
            assert set(holder['dd_labels'].values())=={0,1,2,3}
            stage=Path(cfg['experiment']['output_root'])/f'outer_{oi}/inner_{ii}'
            summary=nt._run_inner_fold(cfg,{'outer_fold':oi,'test_subjects':[]},inner,stage,select_device('cuda'),resume=args.resume,smoke=args.smoke)
            original=next(s for s in lock['stages'] if s['seed']==args.seed and s['context']==oi and s['inner']==ii)
            assert summary['normalization_sha256']==original['normalization_sha256'] and summary['train_subject_ids_sha256']==original['train_subject_ids_sha256']
            print(json.dumps(dict(variant='c1_dd_aux',seed=args.seed,context=oi,inner=ii,best_epoch=summary['best_epoch'],matched_normalization=True,auxiliary_weight=.1,auxiliary_validation_loss=0,outer_test_loader_created=False)),flush=True)
    print(json.dumps(dict(status='COMPLETE',variant='c1_dd_aux',seed=args.seed,smoke=args.smoke)),flush=True)


if __name__=='__main__':main()
