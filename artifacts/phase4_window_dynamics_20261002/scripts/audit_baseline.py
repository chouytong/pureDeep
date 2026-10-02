"""Audit only frozen inner-development WSSL stages; no outer evaluation reads."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score

ROOT = Path('/home/zyt/deep_final')
FOUNDATION = ROOT / 'foundation_validation'
sys.path.insert(0, str(FOUNDATION))
from src.utils.config import load_config
from src.utils.provenance import sha256_json

HERE = Path(__file__).resolve().parent.parent
BASE = ROOT / 'artifacts/phase3b_external_ssl_20260928'
SPLIT = FOUNDATION / 'splits/pads_classification/v3_nested_cv/seed42/nested_cv_splits.json'
M = ['accuracy', 'ba', 'auroc', 'macro_f1', 'pd_recall', 'dd_recall']


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def metrics(y, p):
    pred = (p > .5).astype(int)
    return dict(accuracy=accuracy_score(y, pred), ba=balanced_accuracy_score(y, pred),
                auroc=roc_auc_score(y, p), macro_f1=f1_score(y, pred, average='macro'),
                pd_recall=recall_score(y, pred, pos_label=0), dd_recall=recall_score(y, pred, pos_label=1))


def main():
    out = HERE / 'analysis'
    out.mkdir(exist_ok=True)
    manifest = json.loads((BASE / 'manifest.json').read_text())
    assert sha(SPLIT) == manifest['split_sha256']
    payload = json.loads(SPLIT.read_text())
    cache_path = BASE / 'analysis/ssl_embeddings_all.npz'
    assert sha(cache_path) == manifest['embedding_sha256']
    z = np.load(cache_path, allow_pickle=False)
    ids = z['subject_ids'].astype(str).tolist()
    assert len(ids) == len(set(ids)) == 390
    assert z['pretrained'].shape == (390, 11, 2, 1024)
    assert np.isfinite(z['pretrained']).all()
    assert set(np.unique(z['window_counts'])) == {1, 2}
    rows, locks, seen = [], [], {}
    for seed in (42, 43, 44):
        config = load_config(str(FOUNDATION / f'configs/str01_seed{seed}.yaml'))
        assert z['activities'].astype(str).tolist() == config['data']['activities']
        for context in payload['outer']:
            oi = int(context['outer_fold'])
            for inner in context['inner_folds']:
                ii = int(inner['inner_fold'])
                stage = BASE / f'runs/b_str_pretrained/seed{seed}/outer_{oi}/inner_{ii}'
                status = json.loads((stage / 'stage_status.json').read_text())
                assert status['status'] == 'complete' and status['outer_test_loader_created'] is False
                s = status['summary']
                assert s['smoke'] is False
                ckpath = stage / 'checkpoints/best.pt'
                assert sha(ckpath) == s['checkpoint_sha256']
                ck = torch.load(ckpath, map_location='cpu', weights_only=False)
                assert ck['class_names'] == ['PD', 'DD']
                assert ck['epoch'] + 1 == s['best_epoch']
                assert ck['provenance']['split_sha256'] == manifest['split_sha256']
                assert ck['config']['development']['frozen_embedding_sha256'] == manifest['embedding_sha256']
                for k in ('model', 'training'):
                    assert ck['config'][k] == config[k], (seed, oi, ii, k)
                loss = ck['config']['loss']
                counts = json.loads((stage / 'split.json').read_text())['class_counts']['train']
                assert config['loss']['class_weights'] == 'train_balanced'
                assert loss['class_weight_source'] == 'current_fold_train_subject_labels_only'
                assert loss['class_weight_train_counts'] == counts
                assert loss['label_smoothing'] == config['loss']['label_smoothing'] == 0
                total = sum(counts.values())
                assert np.allclose(loss['class_weights'], [total / (2 * counts[k]) for k in ('PD', 'DD')], atol=0, rtol=0)
                assert ck['input_metadata']['activities'] == config['data']['activities']
                assert ck['input_metadata']['wrist_order'] == ['left', 'right']
                assert ck['input_metadata']['channel_names'] == config['data']['channel_names']
                norm = json.loads((stage / 'normalization.json').read_text())
                train = set(map(str, inner['train_subjects']))
                val = set(map(str, inner['validation_subjects']))
                assert not train & val
                assert train | val <= set(ids)
                assert norm['fitted_on'] == 'explicit_training_subjects_only'
                assert set(norm['subject_ids']) == train
                assert norm['subject_ids_sha256'] == sha256_json(sorted(train)) == s['train_subject_ids_sha256']
                assert norm['normalization_sha256'] == s['normalization_sha256'] == ck['provenance']['normalization_sha256']
                for k in ('mean', 'std'):
                    assert np.array_equal(np.array(norm[k], dtype=np.float32), ck['normalization'][k].numpy())
                frame = pd.read_csv(stage / 'predictions/validation.csv', dtype={'subject_id': str}).sort_values('subject_id').reset_index(drop=True)
                assert frame.subject_id.is_unique and set(frame.subject_id) == val
                assert set(frame.target) == {0, 1}
                key = (oi, ii)
                aligned = frame[['subject_id', 'target']]
                if key in seen:
                    assert aligned.equals(seen[key])
                else:
                    seen[key] = aligned
                y = frame.target.to_numpy(int)
                p = frame.probability_dd.to_numpy(float)
                assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
                m = metrics(y, p)
                aliases = {'ba': 'balanced_accuracy', 'auroc': 'macro_auroc'}
                for k in M:
                    assert np.isclose(m[k], s['validation_metrics'][aliases.get(k, k)], atol=1e-12, rtol=0), (key, seed, k)
                hist = [json.loads(l) for l in (stage / 'logs/epochs.jsonl').read_text().splitlines()]
                best = hist[s['best_epoch'] - 1]
                nparams = sum(t.numel() for t in ck['model_state'].values())
                assert nparams == 143172, nparams
                rows.append(dict(variant='baseline', context=oi, inner=ii, seed=seed, n=len(y), best_epoch=s['best_epoch'], epochs_run=len(hist), train_loss=best['train']['loss'], validation_loss=best['validation']['loss'], **m))
                locks.append(dict(context=oi, inner=ii, seed=seed, checkpoint_sha256=sha(ckpath), prediction_sha256=sha(stage / 'predictions/validation.csv'), normalization_file_sha256=sha(stage / 'normalization.json'), normalization_sha256=s['normalization_sha256'], train_subject_ids_sha256=s['train_subject_ids_sha256'], validation_subject_ids_sha256=sha256_json(sorted(val))))
                del ck
    raw = pd.DataFrame(rows)
    assert len(raw) == 45 and len(seen) == 15
    raw.to_csv(out / 'baseline_seed_split_metrics.csv', index=False)
    fold = raw.groupby(['context', 'inner'], as_index=False)[M].mean()
    fold.to_csv(out / 'baseline_15split_seedfirst.csv', index=False)
    sm = raw.groupby('seed')[M].mean()
    within = raw.groupby(['context', 'inner'])[M].std(ddof=1).mean()
    summary = {**{k: float(fold[k].mean()) for k in M}, **{k + '_split_sd': float(fold[k].std(ddof=1)) for k in M}, **{k + '_seed_sd': float(sm[k].std(ddof=1)) for k in M}, **{k + '_within_split_seed_sd': float(within[k]) for k in M}, 'best_epoch_median': float(raw.best_epoch.median()), 'epochs_run_median': float(raw.epochs_run.median()), 'train_loss': float(raw.train_loss.mean()), 'validation_loss': float(raw.validation_loss.mean())}
    archived = pd.read_csv(BASE / 'analysis/phase3b_summary.csv').set_index('variant').loc['b_str_pretrained']
    for k in M:
        assert np.isclose(summary[k], archived[k], atol=1e-12, rtol=0)
    pd.DataFrame([summary]).to_csv(out / 'baseline_summary.csv', index=False)
    source_files = [BASE / 'scripts/transfer_hooks.py', BASE / 'scripts/extract_ssl.py', BASE / 'scripts/run_inner.py', FOUNDATION / 'src/engine/nested_training.py']
    source_files += sorted((FOUNDATION / 'src/models').rglob('*.py'))
    source_files += sorted((FOUNDATION / 'src/datasets').rglob('*.py'))
    lock = dict(status='PASS', completed_development_stages=45, fixed_development_splits=15, seeds=[42,43,44], baseline_root=str(BASE / 'runs/b_str_pretrained'), cache_path=str(cache_path), cache_sha256=sha(cache_path), split_path=str(SPLIT), split_sha256=sha(SPLIT), classifier_parameters=143172, frozen_harnet_parameters=10457408, decision_rule='DD probability > 0.5; tie PD', outer_artifacts_accessed=False, baseline_reused=True, source_sha256={str(p): sha(p) for p in source_files}, stages=locks, metrics=summary)
    (out / 'baseline_lock.json').write_text(json.dumps(lock, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'stages': 45, 'splits': 15, 'metrics': summary}, indent=2))


if __name__ == '__main__':
    main()
