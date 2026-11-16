"""Independent token-training entry preserving Phase8B's exact fixed recipe.

This module is not executed by registration tests except validation-only input checks.
"""
from pathlib import Path
import json
import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.special import logsumexp
from src.models.registry import MAINLINE, build_registered_model
from src.models.registered_checkpoint import file_sha256, save_registered_checkpoint
from src.utils.seed import seed_everything
from src.metrics.classification import classification_metrics


def validate_recipe(config):
    t, d, loss = config['training'], config['data'], config['loss']
    expected = dict(batch_size=8, optimizer='adamw', learning_rate=.0002,
                    weight_decay=.0001, adam_betas=[.9, .999], epochs=50,
                    gradient_clip_norm=5.0, mixed_precision=False)
    if any(t.get(k) != v for k, v in expected.items()):
        raise ValueError('Keep the original Phase8B training recipe')
    if t['scheduler'] != dict(type='cosine', minimum_lr=1e-6):
        raise ValueError('Keep the original cosine50/min1e-6 schedule')
    early = t['early_stopping']
    if (early.get('metric'), early.get('mode'), early.get('patience'),
            early.get('minimum_delta')) != ('balanced_accuracy', 'max', 12, 0.0):
        raise ValueError('Keep strict validation BA/patience12 selection')
    if d.get('num_workers') != 0 or loss.get('class_weights') != 'train_balanced' or loss.get('label_smoothing') != 0.0:
        raise ValueError('Original token workers0 and train-balanced CE required')
    if config['registration']['status'] != 'Experimental Candidate':
        raise ValueError('AI-WSSL-STR is not a retained model')


def prepare_token_training(config, *, token_index, base_checkpoint, base_sha256,
                           context, inner):
    """No fit, optimizer, signal processing or HarNet extraction occurs here."""
    validate_recipe(config)
    seed = int(config['experiment']['seed'])
    if file_sha256(base_checkpoint) != base_sha256:
        raise ValueError('Base checkpoint hash mismatch')
    base = torch.load(base_checkpoint, map_location='cpu', weights_only=False)
    if (base.get('seed'), base.get('context'), base.get('inner')) != (seed, context, inner):
        raise ValueError('Base checkpoint seed/split mismatch')
    if base.get('version') != config['data']['version']:
        raise ValueError('Base checkpoint/data version mismatch')
    metadata = json.loads(Path(token_index).read_text())
    records = [r for r in metadata['packages'] if r['seed'] == seed]
    if len(records) != 2 or {r['role'] for r in records} != {'train', 'validation'}:
        raise ValueError('Exactly one train and one validation package per seed required')
    split_file = MAINLINE / 'splits/pads_classification/v3_nested_cv/seed42/nested_cv_splits.json'
    splits = json.loads(split_file.read_text())
    outer = next(o for o in splits['outer'] if int(o['outer_fold']) == context)
    split = next(i for i in outer['inner_folds'] if int(i['inner_fold']) == inner)
    packages = {}
    for r in records:
        if file_sha256(r['path']) != r['sha256']:
            raise ValueError('Token package hash mismatch')
        with np.load(r['path'], allow_pickle=False) as z:
            p = {k: z[k].copy() for k in z.files}
        role = r['role']
        expected_ids = set(map(str, split['train_subjects' if role == 'train' else 'validation_subjects']))
        ids = p['subject_ids'].astype(str)
        if len(ids) != len(set(ids)) or set(ids) != expected_ids or ids.tolist() != sorted(ids):
            raise ValueError('Token identities/role/order mismatch')
        if str(p['role']) != role or str(p['version']) != config['data']['version']:
            raise ValueError('Token role/data version mismatch')
        if str(p['base_checkpoint_sha256']) != base_sha256 or str(p['normalization_sha256']) != base['normalization_sha256']:
            raise ValueError('Token/base/normalization binding mismatch')
        if p['activities'].tolist() != config['data']['activities'] or p['bilateral'].shape != (len(ids), 11, 258):
            raise ValueError('Token activity/dimension mismatch')
        if p['mask'].dtype != bool or p['mask'].shape != (len(ids), 11) or not p['mask'].all():
            raise ValueError('The existing fixed Phase8B recipe requires all11 activities')
        if not np.isfinite(p['bilateral']).all() or set(np.unique(p['target'])) != {0, 1}:
            raise ValueError('Invalid tokens/labels')
        packages[role] = p
    if set(packages['train']['subject_ids']) & set(packages['validation']['subject_ids']):
        raise ValueError('Train/validation overlap')
    model = build_registered_model('AI-WSSL-STR', config)
    model.base.load_state_dict(base['model_state'], strict=True)
    return model, packages, dict(seed=seed, context=context, inner=inner,
                                 data_version=config['data']['version'],
                                 base_checkpoint_sha256=base_sha256,
                                 normalization_sha256=base['normalization_sha256'],
                                 token_index_sha256=file_sha256(token_index))


def _metrics(y, p, z):
    m = classification_metrics(y, (p[:, 1] > .5).astype(int), 2, probabilities=p)
    loss = logsumexp(z.astype(float), axis=1) - z[np.arange(len(y)), y]
    confidence = p.max(1)
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = sum(np.mean(bins == b) * abs(np.mean((p[:, 1] > .5)[bins == b] == y[bins == b])
                                      - confidence[bins == b].mean())
              for b in range(10) if np.any(bins == b))
    return dict(accuracy=m['accuracy'], ba=m['balanced_accuracy'], auroc=m['macro_auroc'],
                macro_f1=m['macro_f1'], pd_recall=m['per_class_recall'][0],
                dd_recall=m['per_class_recall'][1], nll=float(loss.mean()),
                brier=float(np.mean(np.sum((p - np.eye(2)[y]) ** 2, axis=1))), ece=ece)


def train_activity_interaction(config, *, token_index, base_checkpoint, base_sha256,
                               context, inner, output_dir, device='cuda'):
    """One explicit split/seed only. Requires a separate training authorization to run."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError('Preserve existing run directory')
    seed = int(config['experiment']['seed'])
    torch.set_num_threads(4)
    seed_everything(seed, True)
    model, packages, provenance = prepare_token_training(
        config, token_index=token_index, base_checkpoint=base_checkpoint,
        base_sha256=base_sha256, context=context, inner=inner)
    model.to(device)
    train, query = packages['train'], packages['validation']
    loader = DataLoader(TensorDataset(torch.from_numpy(train['bilateral']),
                                     torch.from_numpy(train['target']).long()),
                        batch_size=8, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    counts = np.bincount(train['target'], minlength=2)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(len(train['target']) / (2 * counts),
                                                       dtype=torch.float32, device=device))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=.0002, weight_decay=.0001, betas=(.9, .999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    best, wait, best_path, logs = -math.inf, 0, None, []
    output.mkdir(parents=True)
    for epoch in range(50):
        model.train()
        total, seen = 0.0, 0
        for h, target in loader:
            h, target = h.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(h, torch.ones(h.shape[:2], dtype=torch.bool, device=device))
            loss = criterion(out['logits'], target)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss; preserve failed run')
            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.)
            optimizer.step()
            total += float(loss.detach()) * len(target)
            seen += len(target)
        model.eval()
        values = []
        with torch.inference_mode():
            for k in range(0, len(query['target']), 8):
                h = torch.from_numpy(query['bilateral'][k:k+8]).to(device)
                mask = torch.from_numpy(query['mask'][k:k+8]).to(device)
                values.append(model(h, mask)['logits'].cpu().numpy())
        z = np.concatenate(values)
        p = np.exp(z.astype(float) - logsumexp(z.astype(float), axis=1, keepdims=True))
        metrics = _metrics(query['target'], p, z)
        improved = metrics['ba'] > best
        if improved:
            best, wait = metrics['ba'], 0
            best_path = output / f'AI-WSSL-STR_seed{seed}_context{context}_inner{inner}_best_epoch{epoch+1}.pt'
            save_registered_checkpoint('AI-WSSL-STR', model, best_path, config,
                                       provenance=dict(provenance, epoch=epoch+1,
                                                       source_sha256=file_sha256(Path(__file__))))
        else:
            wait += 1
        logs.append(dict(epoch=epoch+1, train_loss=total/seen, validation_BA=metrics['ba'],
                         improved=improved, best_checkpoint=str(best_path)))
        (output / 'epochs.json').write_text(json.dumps(logs, indent=2) + '\n')
        scheduler.step()
        if wait >= 12:
            break
    return dict(best_checkpoint=str(best_path), best_BA=best, epochs=len(logs),
                status='Experimental Candidate', seed=seed, context=context, inner=inner)
