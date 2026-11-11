"""Single development-only entry for the retained STR and frozen WSSL mainline."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.engine import nested_training as nt
from src.utils.config import load_config, require_pads_classification_config
from src.utils.device import select_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--condition', choices=['str01', 'frozen_wssl'], required=True)
    p.add_argument('--seed', type=int, choices=[42, 43, 44], required=True)
    p.add_argument('--context', type=int, choices=range(5), required=True)
    p.add_argument('--inner', type=int, choices=range(3), required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    cfg = load_config(ROOT / 'configs' / f'str01_seed{args.seed}.yaml')
    require_pads_classification_config(cfg)
    assert not cfg['evaluation']['run_test_after_training']
    if args.condition == 'frozen_wssl':
        from src.models.wssl import FrozenEmbeddingCache, activate
        activate(cfg, dict(model_kind='str_residual', embedding_kind='pretrained'), FrozenEmbeddingCache())
    # Restrict manifests before any signal loading; no test dataset is built.
    from src.datasets import folds as fd
    original_records = fd.load_configured_records
    split = json.loads(Path(cfg['nested_cv']['split_file']).read_text())
    inner = split['outer'][args.context]['inner_folds'][args.inner]
    allowed = set(map(str, inner['train_subjects'])) | set(map(str, inner['validation_subjects']))
    fd.load_configured_records = lambda *a, **kw: [r for r in original_records(*a, **kw) if r.subject_id in allowed]
    cfg['experiment']['name'] = f'mainline_{args.condition}_seed{args.seed}'
    cfg['experiment']['output_root'] = str(ROOT.parent / 'results' / 'new_runs' / args.condition / f'seed{args.seed}')
    stage = Path(cfg['experiment']['output_root']) / f'context_{args.context}/inner_{args.inner}'
    if stage.exists():
        raise FileExistsError(f'Preserve existing run: {stage}')
    # Explicitly supply an empty test list, even though the engine is inner-only.
    nt._run_inner_fold(cfg, dict(outer_fold=args.context, test_subjects=[]), inner,
                       stage, select_device('cuda'), resume=False, smoke=args.smoke)


if __name__ == '__main__':
    main()
