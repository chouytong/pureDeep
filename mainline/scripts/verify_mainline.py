"""Migration verification, not training or model selection. Aggregate output only."""
import gc
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.models import build_model
from src.models.wssl import FrozenEmbeddingCache, ResidualSSLSubject
from src.datasets import folds as fd
from src.engine import nested_training as nt
from src.datasets.subject_activity import SubjectActivityDataset, collate_subject_activities
from src.metrics.classification import classification_metrics
from src.utils.config import load_config
from src.utils.seed import seed_everything


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def guards():
    lock = json.loads((PROJECT / 'artifacts/patch_str_20261008/analysis/f0_lock.json').read_text())
    for path, digest in lock['source_sha256'].items():
        assert sha(path) == digest, f'Frozen source changed: {path}'
    for stage in lock['stages']:
        for rel, digest in stage['files_sha256'].items():
            assert sha(Path(stage['path']) / rel) == digest, 'Frozen STR artifact changed'
    study = json.loads((PROJECT / 'artifacts/patch_str_20261008/analysis/study_lock.json').read_text())
    for rel, digest in study['files'].items():
        assert sha(PROJECT / 'artifacts/patch_str_20261008' / rel) == digest, 'Patch study changed'


def main():
    torch.set_num_threads(4)
    seed_everything(42, True)
    guards()
    assert all(str(Path(m.__file__).resolve()).startswith(str(ROOT)) for m in
               [sys.modules['src.models'], sys.modules['src.models.pure_deep'], fd])
    assert not any(n.startswith(('src.s4_final', 'src.r2_stabilization', 'src.targeted_ablation')) for n in sys.modules)
    cfg = load_config(ROOT / 'configs/str01_seed42.yaml')
    split = json.loads(Path(cfg['nested_cv']['split_file']).read_text())
    cache = FrozenEmbeddingCache()
    assert sha(PROJECT / 'data/ssl/harnet10_embeddings.npz') == 'f87829558938c3e50f78e53fe2f54156fc0f3768be3f73f97b089c0587c25376'
    rows, norm_count, differences = [], 0, []
    for outer in split['outer']:
        c = int(outer['outer_fold'])
        for inner in outer['inner_folds']:
            i = int(inner['inner_fold'])
            allowed = set(inner['train_subjects']) | set(inner['validation_subjects'])
            original = fd.load_configured_records
            fd.load_configured_records = lambda *a, **kw: [r for r in original(*a, **kw) if r.subject_id in allowed]
            try:
                bundle = fd.build_subject_fold_datasets(cfg, train_subject_ids=inner['train_subjects'],
                    validation_subject_ids=inner['validation_subjects'], test_subject_ids=(), fold_id=f'context_{c}/inner_{i}')
            finally:
                fd.load_configured_records = original
            assert bundle.test is None
            norm_count += 1
            loader = DataLoader(bundle.validation, batch_size=8, shuffle=False, num_workers=0,
                                collate_fn=collate_subject_activities)
            for condition in ['str01', 'frozen_wssl']:
                for seed in [42, 43, 44]:
                    if condition == 'str01':
                        stage = PROJECT / f'foundation_validation/outputs/structured_token_residual/str01_seed{seed}_20260921/outer_{c}/inner_{i}'
                    else:
                        stage = PROJECT / f'artifacts/phase3b_external_ssl_20260928/runs/b_str_pretrained/seed{seed}/outer_{c}/inner_{i}'
                    checkpoint = torch.load(stage / 'checkpoints/best.pt', map_location='cpu', weights_only=False)
                    for key in ['mean', 'std']:
                        assert torch.equal(getattr(bundle, key), checkpoint['normalization'][key]), 'Normalization changed'
                    # Only paths/_meta differ; training, data semantics and STR model config are unchanged.
                    current = load_config(ROOT / 'configs' / f'str01_seed{seed}.yaml')
                    for key in ['training', 'loss', 'evaluation', 'model']:
                        expected = checkpoint['config'][key]
                        if key == 'loss':
                            assert expected == nt._resolve_train_only_class_weights(current, bundle)['loss']
                        else:
                            assert current[key] == expected, f'Semantic config changed: {key}'
                    model = build_model(current)
                    if condition == 'frozen_wssl':
                        model = ResidualSSLSubject(model)
                    expected_params = 75524 if condition == 'str01' else 143172
                    assert sum(p.numel() for p in model.parameters()) == expected_params
                    model.load_state_dict(checkpoint['model_state'], strict=True)
                    model.cuda().eval()
                    saved = pd.read_csv(stage / 'predictions/validation.csv', dtype={'subject_id': str}).set_index('subject_id')
                    ids, labels, probabilities = [], [], []
                    with torch.no_grad():
                        for batch in loader:
                            inputs = [batch[k].cuda() for k in ['x', 'wrist_mask', 'activity_mask', 'activity_lengths']]
                            if condition == 'frozen_wssl':
                                model.ssl_features = cache.batch(batch['subject_id'], 'pretrained', 'cuda')
                            probabilities.append(model(*inputs)['probabilities'].cpu().numpy())
                            ids.extend(batch['subject_id'])
                            labels.extend(batch['y'].tolist())
                    probability = np.concatenate(probabilities)
                    reference = saved.loc[ids]
                    assert set(ids) == set(inner['validation_subjects']) and labels == reference.target.tolist()
                    delta = float(abs(probability - reference[['probability_pd', 'probability_dd']].to_numpy()).max())
                    assert delta <= 1e-12, f'Probability changed: {condition}'
                    assert np.array_equal(probability.argmax(1), reference.prediction.to_numpy())
                    metric = classification_metrics(np.asarray(labels), probability.argmax(1), 2, probabilities=probability)
                    summary = json.loads((stage / 'stage_status.json').read_text())['summary']['validation_metrics']
                    for key in ['accuracy', 'balanced_accuracy', 'macro_auroc', 'macro_f1', 'per_class_recall']:
                        assert np.allclose(metric[key], summary[key], atol=1e-12, rtol=0)
                    differences.append(delta)
                    rows.append(dict(condition=condition, seed=seed, context=c, inner=i,
                        accuracy=metric['accuracy'], ba=metric['balanced_accuracy'], auroc=metric['macro_auroc'],
                        macro_f1=metric['macro_f1'], pd_recall=metric['per_class_recall'][0], dd_recall=metric['per_class_recall'][1]))
                    del model, checkpoint
                    gc.collect()
            print(json.dumps(dict(context=c, inner=i, checkpoints_passed=6)), flush=True)
            del bundle, loader
    assert len(rows) == 90 and norm_count == 15
    guards()
    frame = pd.DataFrame(rows)
    metrics = ['accuracy', 'ba', 'auroc', 'macro_f1', 'pd_recall', 'dd_recall']
    means = frame.groupby(['condition', 'context', 'inner'])[metrics].mean().groupby('condition').mean()
    report = dict(status='PASS', source_location=str(ROOT), conditions=['str01', 'frozen_wssl'],
        checkpoint_validation_forwards=90, train_only_normalization_refits=15,
        max_probability_difference=max(differences), seed_first_means=means.to_dict(orient='index'),
        old_MFAM_runtime_dependency=False, frozen_assets_unchanged=True,
        outer_signal_or_outcomes_accessed=False, training_started=False,
        model_or_training_recipe_changed=False, data_copy_verified=True)
    out = PROJECT / 'conclusions/organization_20261111'
    out.mkdir(parents=True, exist_ok=True)
    for name in ['VERIFICATION.json', 'verification_90_runs.csv']:
        assert not (out / name).exists(), 'Preserve previous verification output'
    (out / 'VERIFICATION.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    frame.to_csv(out / 'verification_90_runs.csv', index=False)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
