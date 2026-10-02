"""Real development batches, all 45 frozen models, exact unchanged logits."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent.parent
F = Path('/home/zyt/deep_final/foundation_validation')
B = Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928')
sys.path[:0] = [str(F), str(B / 'scripts')]
from src.datasets.builders import load_configured_records
from src.datasets.subject_activity import SubjectActivityDataset, collate_subject_activities
from src.models import build_model
from src.utils.config import load_config
from src.utils.seed import seed_everything
from transfer_hooks import ResidualSSLSubject, FrozenEmbeddingCache


def main():
    seed_everything(42, True)
    torch.set_num_threads(4)
    old = FrozenEmbeddingCache()
    z = np.load(HERE / 'features/window_features.npz', allow_pickle=False)
    assert z['subject_ids'].astype(str).tolist() == old.ids
    assert z['activities'].astype(str).tolist() == old.activities
    mask = z['valid_window_mask']
    assert mask.dtype == np.bool_ and mask.shape == (390, 11, 2, 2)
    assert mask[..., 0].all() and np.array_equal(mask.sum(-1), old.counts)
    assert np.all(z['back'][~mask[..., 1]] == 0)
    assert np.array_equal(z['mean'], old.pretrained.numpy())
    reverse = [old.ids[-1], old.ids[0]]
    lookup = {s: i for i, s in enumerate(z['subject_ids'].astype(str))}
    assert np.array_equal(z['mean'][[lookup[s] for s in reverse]], old.batch(reverse, 'pretrained', 'cpu').numpy())
    cfg = load_config(str(F / 'configs/str01_seed42.yaml'))
    records = load_configured_records(cfg['data'], ['PD', 'DD'])
    lock = json.loads((HERE / 'analysis/baseline_lock.json').read_text())
    rows, smoke = [], None
    for stage in lock['stages']:
        seed, oi, ii = stage['seed'], stage['context'], stage['inner']
        path = B / f'runs/b_str_pretrained/seed{seed}/outer_{oi}/inner_{ii}'
        frame = pd.read_csv(path / 'predictions/validation.csv', dtype={'subject_id': str}).sort_values('subject_id')
        selected = frame.iloc[:8]
        ids = selected.subject_id.tolist()
        ck = torch.load(path / 'checkpoints/best.pt', map_location='cpu', weights_only=False)
        ds = SubjectActivityDataset([r for r in records if r.subject_id in ids], old.activities, cfg['data'], ck['normalization']['mean'], ck['normalization']['std'], 'validation')
        batch = collate_subject_activities([ds[i] for i in range(len(ds))])
        args = [batch[k].cuda() for k in ('x', 'wrist_mask', 'activity_mask', 'activity_lengths')]
        assert args[0].shape[:4] == (8, 11, 2, 6)
        model = ResidualSSLSubject(build_model(cfg)).eval().cuda()
        model.load_state_dict(ck['model_state'], strict=True)
        model.ssl_features = old.batch(batch['subject_id'], 'pretrained', 'cuda')
        with torch.no_grad():
            before = model(*args)['logits'].clone()
            p = before.softmax(-1)[:, 1].cpu().numpy()
            assert np.allclose(p, selected.probability_dd.to_numpy(), atol=1e-6, rtol=0), (seed, oi, ii, p)
            model.ssl_features = torch.from_numpy(z['mean'][[lookup[s] for s in batch['subject_id']]]).cuda()
            after = model(*args)['logits'].clone()
        assert torch.equal(before, after)
        rows.append(dict(seed=seed, context=oi, inner=ii, max_abs_logit_difference=float((after-before).abs().max()), archived_probability_max_difference=float(np.abs(p-selected.probability_dd.to_numpy()).max())))
        if smoke is None:
            am = args[2].clone(); wm = args[1].clone()
            am[:, 0] = False; wm[:, 1, 0] = 0
            with torch.no_grad():
                a = model(args[0], wm, am, args[3])['logits']
                changed = model.ssl_features.clone()
                changed[:, 0] = 10000
                changed[:, 1, 0] = -10000
                model.ssl_features = changed
                b = model(args[0], wm, am, args[3])['logits']
            assert torch.equal(a, b), 'Invalid wrist/activity SSL leak'
            model.ssl_features = old.batch(batch['subject_id'], 'pretrained', 'cuda')
            model.train()
            optim = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
            initial = model.wrist_projection.weight.detach().clone()
            optim.zero_grad()
            logits = model(*args)['logits']
            loss = nn.CrossEntropyLoss(weight=torch.tensor(ck['config']['loss']['class_weights'], device='cuda'))(logits, batch['y'].cuda())
            loss.backward()
            assert model.wrist_projection.weight.grad is not None and torch.isfinite(model.wrist_projection.weight.grad).all()
            assert float(model.wrist_projection.weight.grad.abs().sum()) > 0
            assert model.ssl_features.requires_grad is False
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optim.step()
            assert not torch.equal(initial, model.wrist_projection.weight)
            model.eval()
            with torch.no_grad(): expected = model(*args)['logits'].clone()
            reload_path = HERE / 'smoke/window_reload.pt'
            reload_path.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), reload_path)
            reloaded = ResidualSSLSubject(build_model(cfg)).eval().cuda()
            reloaded.load_state_dict(torch.load(reload_path, map_location='cuda', weights_only=True), strict=True)
            reloaded.ssl_features = model.ssl_features
            with torch.no_grad(): actual = reloaded(*args)['logits']
            assert torch.equal(expected, actual)
            smoke = dict(masked_wrist_activity_invariance=True, real_batch_shape=list(args[0].shape), projection_gradient_finite_nonzero=True, frozen_features_requires_grad=False, one_batch_update=True, checkpoint_reload_exact=True, smoke_loss_finite=bool(torch.isfinite(loss)))
        del ck, model
    assert len(rows) == 45
    pd.DataFrame(rows).to_csv(HERE / 'analysis/window_logit_consistency.csv', index=False)
    result = dict(status='PASS', checkpoints_tested=45, real_validation_batches=45, mean_cache_exact=True, logits_exact_for_all_checkpoints=True, maximum_archived_probability_difference=max(r['archived_probability_max_difference'] for r in rows), explicit_subject_lookup=True, short_window_no_fake_back=True, smoke=smoke, outer_artifacts_accessed=False)
    (HERE / 'analysis/window_consistency_test.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
