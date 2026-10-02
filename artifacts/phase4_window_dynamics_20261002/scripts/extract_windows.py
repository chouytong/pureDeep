"""Preserve official frozen window features; verify old mean cache before saving."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent.parent
OLD = Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928')
VENDOR = OLD / 'vendor/ssl-wearables'
sys.path.insert(0, str(VENDOR))
import hubconf

DATA = Path('/home/zyt/MFAM/data/processed/pads_multi_activity/v2_l1_full_length/manifests')
RAW = Path('/home/zyt/MFAM/data/raw/pads/movement/timeseries')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    audit = json.loads((OLD / 'analysis/extraction_audit.json').read_text())
    old_path = OLD / 'analysis/ssl_embeddings_all.npz'
    assert sha(old_path) == audit['output_sha256']
    old = np.load(old_path, allow_pickle=False)
    activities = old['activities'].astype(str).tolist()
    ids = old['subject_ids'].astype(str).tolist()
    split_path = Path('/home/zyt/deep_final/foundation_validation/splits/pads_classification/v3_nested_cv/seed42/nested_cv_splits.json')
    split = json.loads(split_path.read_text())
    development_ids = set()
    for context in split['outer']:
        for inner in context['inner_folds']:
            development_ids.update(inner['train_subjects'])
            development_ids.update(inner['validation_subjects'])
    assert set(ids) == development_ids
    rows = pd.concat([pd.read_csv(DATA / f'{a}.csv', dtype={'subject_id': str}) for a in activities], ignore_index=True)
    assert len(rows) == 4290 and rows[['subject_id', 'activity']].duplicated().sum() == 0
    assert sorted(rows.subject_id.unique()) == ids
    index = {s: i for i, s in enumerate(ids)}
    ai = {a: i for i, a in enumerate(activities)}
    windows, keys, positions = [], [], []
    raw_sha = hashlib.sha256()
    mask = np.zeros((390, 11, 2, 2), dtype=bool)
    for row in rows.itertuples():
        for wi, wrist in enumerate(('LeftWrist', 'RightWrist')):
            raw = np.loadtxt(RAW / f'{row.subject_id}_{row.record_name}_{wrist}.txt', delimiter=',', dtype=np.float32)
            assert raw.ndim == 2 and raw.shape[1] == 7
            acc = raw[48:, 1:4]
            assert len(acc) == (row.left_length if wi == 0 else row.right_length)
            assert len(acc) in (976, 2000) and np.isfinite(acc).all()
            acc = np.clip(acc, -3, 3)
            k = (index[row.subject_id], ai[row.activity], wi)
            if len(acc) < 1000:
                segments = [np.pad(acc, ((12, 12), (0, 0)), mode='edge')]
            else:
                segments = [acc[:1000], acc[-1000:]]
            for pos, segment in enumerate(segments):
                x = resample_poly(segment, 3, 10, axis=0).T.astype(np.float32)
                assert x.shape == (3, 300) and np.isfinite(x).all()
                windows.append(x)
                keys.append(k)
                positions.append((*k, pos))
                mask[(*k, pos)] = True
                raw_sha.update(x.tobytes())
                raw_sha.update(np.asarray(k, dtype=np.int16).tobytes())
    assert raw_sha.hexdigest() == audit['resampled_window_sha256']
    assert mask[..., 0].all() and np.array_equal(mask.sum(-1), old['window_counts'])
    X = np.stack(windows)
    keys = np.asarray(keys, dtype=np.int16)
    positions = np.asarray(positions, dtype=np.int16)
    checkpoint = VENDOR / 'model_check_point/mtl_best.mdl'
    assert sha(checkpoint) == audit['official_checkpoint_sha256']
    # Reproduce original constructor RNG order and exact inference batching.
    torch.manual_seed(20260928)
    random_control = hubconf.harnet10(pretrained=False, class_num=2).feature_extractor.eval().cuda()
    model = hubconf.harnet10(pretrained=True, my_device='cpu', class_num=2).feature_extractor.eval().cuda()
    for param in model.parameters():
        param.requires_grad_(False)
    feat = np.zeros((390, 11, 2, 2, 1024), dtype=np.float32)
    mean = np.zeros((390, 11, 2, 1024), dtype=np.float32)
    counts = np.zeros((390, 11, 2), dtype=np.int16)
    with torch.inference_mode():
        for lo in range(0, len(X), 64):
            hi = min(lo + 64, len(X))
            encoded = model(torch.from_numpy(X[lo:hi]).cuda()).flatten(1).cpu().numpy()
            assert encoded.shape == (hi-lo, 1024) and np.isfinite(encoded).all()
            feat[tuple(positions[lo:hi].T)] = encoded
            np.add.at(mean, tuple(keys[lo:hi].T), encoded)
            np.add.at(counts, tuple(keys[lo:hi].T), 1)
    mean /= counts[..., None]
    recovered = feat.sum(axis=3) / counts[..., None]
    assert np.array_equal(recovered, mean), 'Window reconstruction differs from extraction aggregation'
    maxdiff = float(np.abs(mean - old['pretrained']).max())
    assert np.array_equal(mean, old['pretrained']), f'Frozen cache not exactly reproduced: maxdiff={maxdiff}'
    assert np.all(feat[..., 1, :][~mask[..., 1]] == 0)
    dest = HERE / 'features/window_features.npz'
    dest.parent.mkdir(exist_ok=True)
    assert not dest.exists(), 'Never overwrite completed window cache'
    np.savez_compressed(dest, subject_ids=np.asarray(ids), activities=np.asarray(activities), front=feat[..., 0, :], back=feat[..., 1, :], valid_window_mask=mask, mean=mean)
    summary = dict(status='PASS', records=8580, windows=len(X), single_window_records=int((counts == 1).sum()), double_window_records=int((counts == 2).sum()), feature_shape=[390,11,2,2,1024], mask_shape=list(mask.shape), official_checkpoint_sha256=sha(checkpoint), resampled_window_sha256=raw_sha.hexdigest(), old_cache_sha256=sha(old_path), window_cache_sha256=sha(dest), mean_exactly_equal=True, max_absolute_mean_difference=maxdiff, frozen_parameters=sum(p.numel() for p in model.parameters()), trainable_encoder_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad), outer_artifacts_accessed=False)
    (HERE / 'analysis/window_extraction_audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
