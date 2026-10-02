"""One independent capacity/dynamics variant, fixed inner-development recipe."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
F = Path('/home/zyt/deep_final/foundation_validation')
sys.path.insert(0, str(F))
from src.engine.nested_training import _load_frozen_split, _run_inner_fold
from src.utils.config import load_config, require_pads_classification_config
from src.utils.device import select_device
from window_hooks import WindowCache, activate


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['a1_mean', 'a2_delta'], required=True)
    ap.add_argument('--seed', type=int, choices=[42,43,44], required=True)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--context', type=int)
    ap.add_argument('--inner', type=int)
    args = ap.parse_args()
    lock = json.loads((HERE / 'analysis/baseline_lock.json').read_text())
    test = json.loads((HERE / 'analysis/window_consistency_test.json').read_text())
    assert lock['status'] == test['status'] == 'PASS'
    assert sha(HERE / 'features/window_features.npz') == json.loads((HERE / 'analysis/window_extraction_audit.json').read_text())['window_cache_sha256']
    for path, digest in lock['source_sha256'].items():
        assert sha(path) == digest, f'Frozen source changed: {path}'
    config = load_config(str(F / f'configs/str01_seed{args.seed}.yaml'))
    require_pads_classification_config(config)
    config['development'] = dict(phase4_variant=args.variant, baseline='phase3b/b_str_pretrained', window_cache_sha256=sha(HERE / 'features/window_features.npz'), protocol_sha256=sha(HERE / 'PROTOCOL.md'), no_outer_access=True)
    config['experiment']['name'] = f'phase4_{args.variant}_seed{args.seed}'
    config['experiment']['output_root'] = str(HERE / ('smoke' if args.smoke else 'runs') / args.variant / f'seed{args.seed}')
    cache = WindowCache()
    activate(config, 'mean' if args.variant == 'a1_mean' else 'delta', cache)
    split, audit, split_path, digest = _load_frozen_split(config)
    assert digest == lock['split_sha256']
    device = select_device('cuda')
    out = Path(config['experiment']['output_root'])
    for context in split['outer']:
        oi = int(context['outer_fold'])
        if args.context is not None and oi != args.context:
            continue
        # Outer membership is deliberately not passed to dataset construction.
        development_context = {'outer_fold': oi, 'test_subjects': []}
        for inner in context['inner_folds']:
            ii = int(inner['inner_fold'])
            if args.inner is not None and ii != args.inner:
                continue
            stage = out / f'outer_{oi}/inner_{ii}'
            summary = _run_inner_fold(config, development_context, inner, stage, device, resume=args.resume, smoke=args.smoke)
            original = next(s for s in lock['stages'] if s['seed'] == args.seed and s['context'] == oi and s['inner'] == ii)
            assert summary['normalization_sha256'] == original['normalization_sha256']
            assert summary['train_subject_ids_sha256'] == original['train_subject_ids_sha256']
            print(json.dumps(dict(variant=args.variant, seed=args.seed, context=oi, inner=ii, best_epoch=summary['best_epoch'], matched_normalization=True, outer_test_loader_created=False)), flush=True)
    print(json.dumps(dict(status='COMPLETE', variant=args.variant, seed=args.seed, smoke=args.smoke)), flush=True)


if __name__ == '__main__':
    main()
