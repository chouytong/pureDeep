"""Audit retained WSSL on existing development checkpoints; no candidate training.

The only optimizer steps below use a new, disposable model to check gradient
connectivity. No score or trained artifact from that scratch instance is kept.
"""
from __future__ import annotations

import gc
import hashlib
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent.parent
FOUNDATION = Path('/home/zyt/deep_final/foundation_validation')
BASELINE = Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928')
EMA = Path('/home/zyt/deep_final/artifacts/wssl_ema_20261002')
LOCK = Path('/home/zyt/deep_final/artifacts/phase4_window_dynamics_20261002/analysis/baseline_lock.json')
sys.path[:0] = [str(FOUNDATION), str(BASELINE / 'scripts'), str(EMA / 'scripts')]

from transfer_hooks import FrozenEmbeddingCache, ResidualSSLSubject
from independent_wssl import IndependentWSSL, independent_copy
from src.datasets.builders import load_configured_records
from src.datasets.subject_activity import SubjectActivityDataset, collate_subject_activities
from src.models import build_model
from src.utils.config import load_config
from src.utils.seed import seed_everything


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_rng():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone(), torch.cuda.get_rng_state_all()


def same_rng(a, b):
    return (a[0] == b[0] and a[1][0] == b[1][0]
            and np.array_equal(a[1][1], b[1][1]) and a[1][2:] == b[1][2:]
            and torch.equal(a[2], b[2]) and len(a[3]) == len(b[3])
            and all(torch.equal(x, y) for x, y in zip(a[3], b[3])))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def gradient_smoke(config, inputs, ssl, targets, class_weights):
    """A fresh zero-projection model, one batch, two updates; not a performance run."""
    scratch = IndependentWSSL(config).cuda().train()
    scratch.ssl_features = ssl.detach().clone()
    require(not scratch.ssl_features.requires_grad, 'Frozen cache unexpectedly has gradients')
    optimizer = torch.optim.AdamW(scratch.parameters(), lr=2e-4, weight_decay=1e-4)
    rows = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = scratch(*inputs, ssl_features=ssl.detach())
        loss = torch.nn.functional.cross_entropy(output['logits'], targets, weight=class_weights)
        require(torch.isfinite(loss).item(), 'Nonfinite scratch loss')
        loss.backward()
        gradients = {name: parameter.grad for name, parameter in scratch.named_parameters()}
        present = {name: value for name, value in gradients.items() if value is not None}
        require(all(torch.isfinite(g).all().item() for g in present.values()), 'Nonfinite gradient')
        projection = float(scratch.wrist_projection.weight.grad.abs().sum())
        normalization = float(scratch.norm.weight.grad.abs().sum())
        encoder = sum(float(g.abs().sum()) for n, g in present.items() if n.startswith('backbone.wrist_encoder.'))
        structured = float(scratch.backbone.structured_token_residual.residual_head.weight.grad.abs().sum())
        require(projection > 0 and encoder > 0 and structured > 0, 'Main/SSL/structured path disconnected')
        require(normalization == 0 if step == 0 else normalization > 0,
                'Zero-projection normalization gradient progression differs')
        require(scratch.ssl_features.grad is None and ssl.grad is None, 'Frozen feature received gradient')
        rows.append(dict(step=step + 1, ssl_projection_gradient_l1=projection,
                         ssl_norm_gradient_l1=normalization, wrist_encoder_gradient_l1=encoder,
                         structured_head_gradient_l1=structured,
                         finite_gradients=True, frozen_features_no_gradient=True))
        torch.nn.utils.clip_grad_norm_(scratch.parameters(), 5)
        optimizer.step()
    del scratch, optimizer
    return dict(scope='One fresh scratch model, one real inner-training batch, two optimizer steps; no performance evaluation',
                checkpoint_weights_modified=False, score_reported=False, steps=rows)


def main():
    seed_everything(42, True)
    torch.set_num_threads(4)
    require(torch.cuda.is_available(), 'This audit uses the same CUDA runtime as the retained model')
    lock = json.loads(LOCK.read_text())
    source_before = {path: sha(path) for path in lock['source_sha256']}
    require(source_before == lock['source_sha256'], 'Frozen foundation/source lock mismatch')
    require(sha(lock['cache_path']) == lock['cache_sha256'], 'Frozen feature cache SHA mismatch')
    require(sha(lock['split_path']) == lock['split_sha256'], 'Frozen development split SHA mismatch')
    cache = FrozenEmbeddingCache()
    require(torch.isfinite(cache.pretrained).all().item(), 'Retained cache contains nonfinite feature values')
    require(not cache.pretrained.requires_grad, 'Retained cache is not frozen')
    cfg = load_config(str(FOUNDATION / 'configs/str01_seed42.yaml'))
    require(cache.activities == cfg['data']['activities'], 'Ordered activities differ')
    records = load_configured_records(cfg['data'], ['PD', 'DD'])
    rows = []
    isolation = None
    gradient = None
    for stage in lock['stages']:
        seed, context, inner = stage['seed'], stage['context'], stage['inner']
        directory = BASELINE / f'runs/b_str_pretrained/seed{seed}/outer_{context}/inner_{inner}'
        checkpoint_path = directory / 'checkpoints/best.pt'
        require(sha(checkpoint_path) == stage['checkpoint_sha256'], 'Formal checkpoint changed')
        require(sha(directory / 'predictions/validation.csv') == stage['prediction_sha256'], 'Formal prediction file changed')
        predictions = pd.read_csv(directory / 'predictions/validation.csv', dtype={'subject_id': str})
        require(not predictions.subject_id.duplicated().any(), 'Duplicate validation subject ID')
        selected_ids = sorted(predictions.subject_id.tolist())[:8]
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        dataset = SubjectActivityDataset([r for r in records if r.subject_id in set(selected_ids)],
                                        cache.activities, cfg['data'], checkpoint['normalization']['mean'],
                                        checkpoint['normalization']['std'], 'validation')
        batch = collate_subject_activities([dataset[i] for i in range(len(dataset))])
        ids = list(map(str, batch['subject_id']))
        require(set(ids) == set(selected_ids) and len(ids) == 8, 'Evaluation scope or ID alignment changed')
        target_lookup = dict(zip(predictions.subject_id, predictions.target))
        require(batch['y'].tolist() == [int(target_lookup[s]) for s in ids], 'Labels do not agree by ID')
        inputs = [batch[key].cuda() for key in ('x', 'wrist_mask', 'activity_mask', 'activity_lengths')]
        ssl = cache.batch(ids, 'pretrained', 'cuda')
        model = IndependentWSSL(cfg).cuda().eval()
        model.load_state_dict(checkpoint['model_state'], strict=True)
        require(model.backbone.wrist_structured_residual is None, 'Unexpected retained wrist structured residual')
        require(model.backbone.structured_token_residual.projection_dim == 16, 'Structured dimension changed')
        require(not dict(model.named_buffers()), 'Unexpected classifier buffers')
        require(sum(p.numel() for p in model.parameters()) == 143172, 'Parameter count changed')
        old = ResidualSSLSubject(build_model(cfg)).cuda().eval()
        old.load_state_dict(checkpoint['model_state'], strict=True)
        old.ssl_features = ssl
        with torch.no_grad():
            expected = model(*inputs, ssl_features=ssl)['logits']
            original = old(*inputs)['logits']
            require(torch.equal(expected, original), 'Old/new retained logits differ')
            probability = expected.softmax(-1)[:, 1].cpu().numpy()
            probability_lookup = dict(zip(predictions.subject_id, predictions.probability_dd))
            archived = np.array([probability_lookup[s] for s in ids])
            require(np.allclose(probability, archived, atol=1e-6, rtol=0), 'Archived probabilities differ by ID')

            # A deliberately stale bridge cannot override the explicit feature input.
            model.ssl_features = torch.full_like(ssl, float('nan'))
            require(torch.equal(model(*inputs, ssl_features=ssl)['logits'], expected), 'Explicit input loses priority')
            model.ssl_features = None
            try:
                model(*inputs)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Missing SSL features not rejected')
            try:
                model(*inputs, ssl_features=ssl[..., :16])
            except RuntimeError:
                pass
            else:
                raise AssertionError('Wrong SSL shape not rejected')

            # Reorder all local inputs and obtain features by the correspondingly reordered IDs.
            order = torch.arange(len(ids) - 1, -1, -1, device='cuda')
            reordered_ids = list(reversed(ids))
            reordered_ssl = cache.batch(reordered_ids, 'pretrained', 'cuda')
            require(torch.equal(reordered_ssl, ssl.index_select(0, order)), 'Cache lookup depends on position')
            reordered = model(*(x.index_select(0, order) for x in inputs), ssl_features=reordered_ssl)['logits']
            reorder_error = float((reordered - expected.index_select(0, order)).abs().max())
            # Identical operations, different batch order; FP32 tolerance is fixed a priori.
            require(reorder_error <= 1e-5, 'Subject reorder equivariance fails FP32 tolerance 1e-5')

            activity_mask, wrist_mask = inputs[2].clone(), inputs[1].clone()
            activity_mask[:, 0] = False
            wrist_mask[:, 1, 0] = False
            masked_inputs = [inputs[0], wrist_mask, activity_mask, inputs[3]]
            masked_before = model(*masked_inputs, ssl_features=ssl)['logits']
            changed = ssl.clone()
            changed[:, 0] = 10000
            changed[:, 1, 0] = -10000
            masked_after = model(*masked_inputs, ssl_features=changed)['logits']
            require(torch.equal(masked_before, masked_after), 'Finite invalid wrist/activity SSL leaks')
            without_ssl = model(*inputs, ssl_features=torch.zeros_like(ssl))['logits']
            ssl_effect = float((expected - without_ssl).abs().max())
            require(ssl_effect > 0, 'Learned retained SSL path has no observable effect')

            # Check serialization/reload for every checkpoint without writing another checkpoint.
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            reloaded = IndependentWSSL(cfg).cuda().eval()
            reloaded.load_state_dict(torch.load(buffer, map_location='cuda', weights_only=True), strict=True)
            require(torch.equal(reloaded(*inputs, ssl_features=ssl)['logits'], expected), 'Fresh reload differs')
        if isolation is None:
            model.ssl_features = ssl.clone()
            before_rng = snapshot_rng()
            copied = independent_copy(model, cfg)
            require(same_rng(before_rng, snapshot_rng()), 'Independent copy consumes ordinary RNG')
            require(all(a.data_ptr() != b.data_ptr() for a, b in zip(model.parameters(), copied.parameters())),
                    'Models share parameter storage')
            require(model.ssl_features.data_ptr() != copied.ssl_features.data_ptr(), 'Models share feature storage')
            with torch.no_grad():
                copied.wrist_projection.weight.add_(1)
                copied.ssl_features.zero_()
                require(torch.equal(model(*inputs)['logits'], expected), 'Copied model mutation changes original')
            isolation = dict(scope='First frozen checkpoint, one real eight-subject batch',
                             parameters_separate=True, cache_separate=True,
                             copied_mutation_does_not_change_original=True,
                             python_numpy_cpu_cuda_rng_preserved=True)
            train_ids = set(json.loads((directory / 'split.json').read_text())['train_subjects'])
            require(not train_ids.intersection(ids), 'Scratch training subjects overlap validation batch')
            label_by_id = {r.subject_id: r.label for r in records if r.subject_id in train_ids}
            chosen = sorted(s for s in train_ids if label_by_id[s] == 0)[:4]
            chosen += sorted(s for s in train_ids if label_by_id[s] == 1)[:4]
            require(len(chosen) == 8, 'Scratch gradient batch needs four training subjects per class')
            train_dataset = SubjectActivityDataset([r for r in records if r.subject_id in set(chosen)],
                                                  cache.activities, cfg['data'], checkpoint['normalization']['mean'],
                                                  checkpoint['normalization']['std'], 'train')
            train_batch = collate_subject_activities([train_dataset[i] for i in range(len(train_dataset))])
            train_inputs = [train_batch[key].cuda() for key in ('x', 'wrist_mask', 'activity_mask', 'activity_lengths')]
            train_ssl = cache.batch(train_batch['subject_id'], 'pretrained', 'cuda')
            gradient = gradient_smoke(cfg, train_inputs, train_ssl, train_batch['y'].cuda(),
                                      torch.tensor(checkpoint['config']['loss']['class_weights'], device='cuda'))
            del train_dataset, train_batch, train_inputs, train_ssl
            del copied
        rows.append(dict(seed=seed, context=context, inner=inner, subjects_evaluated=len(ids),
                         old_new_logits_exact=True, archived_probability_max_error=float(np.abs(probability - archived).max()),
                         explicit_input_priority=True, missing_wrong_shape_rejected=True,
                         subject_id_cache_reorder_exact=True, reorder_logit_max_error=reorder_error,
                         finite_masked_ssl_no_leak=True, retained_ssl_effect_logit_max=ssl_effect,
                         checkpoint_reload_exact=True))
        del model, old, reloaded, checkpoint, inputs, ssl, dataset, batch
        gc.collect()
        print(json.dumps(dict(checkpoint=len(rows), seed=seed, context=context, inner=inner, status='PASS')), flush=True)
    require(len(rows) == 45, 'Incomplete frozen checkpoint coverage')
    require({path: sha(path) for path in source_before} == source_before, 'Frozen source changed during audit')
    require(sha(lock['cache_path']) == lock['cache_sha256'], 'Frozen cache changed during audit')
    (HERE / 'analysis').mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(HERE / 'analysis/fusion_properties_45checkpoints.csv', index=False)
    result = dict(status='PASS', frozen_checkpoints=45, batches_per_checkpoint=1,
                  subjects_per_batch=8, total_subject_evaluations=360,
                  repeated_records_are_not_independent_statistical_samples=True,
                  evaluation_scope='First eight validation subjects by ID per frozen checkpoint; not full validation',
                  all_old_new_logits_exact=True, explicit_ssl_input_priority=True,
                  subject_id_reorder_test=True, subject_reorder_logit_tolerance=1e-5,
                  max_subject_reorder_error=max(r['reorder_logit_max_error'] for r in rows),
                  max_archived_probability_error=max(r['archived_probability_max_error'] for r in rows),
                  finite_masked_ssl_no_leak=True, every_retained_checkpoint_ssl_has_observable_effect=True,
                  all_checkpoint_reloads_exact=True, retained_cache_finite=True, classifier_buffers=[],
                  classifier_parameters=143172, isolation=isolation, gradient_smoke=gradient,
                  frozen_source_cache_checkpoint_prediction_guards=True,
                  outer_test_loaders_created=False, outer_performance_accessed=False,
                  new_performance_candidate=False, scratch_performance_reported=False,
                  script_sha256=sha(__file__))
    (HERE / 'analysis/fusion_properties.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
