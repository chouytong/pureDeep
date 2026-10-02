"""Independent window residual experiment; original source/forward stays frozen."""
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent.parent
OLD = Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928')
sys.path.insert(0, str(OLD / 'scripts'))
from transfer_hooks import ResidualSSLSubject
from src.engine import nested_training as nt
from src.datasets import folds as fd


class WindowCache:
    def __init__(self):
        z = np.load(HERE / 'features/window_features.npz', allow_pickle=False)
        self.ids = z['subject_ids'].astype(str).tolist()
        self.index = {s: i for i, s in enumerate(self.ids)}
        self.activities = z['activities'].astype(str).tolist()
        self.mean = torch.from_numpy(z['mean'].copy())
        self.front = torch.from_numpy(z['front'].copy())
        self.back = torch.from_numpy(z['back'].copy())
        self.mask = torch.from_numpy(z['valid_window_mask'].copy())
        assert len(self.ids) == len(self.index) == 390
        assert self.mean.shape == self.front.shape == self.back.shape == (390, 11, 2, 1024)
        assert self.mask.shape == (390, 11, 2, 2) and self.mask.dtype == torch.bool
        assert self.mask[..., 0].all() and (self.back[~self.mask[..., 1]] == 0).all()

    def attach(self, model, subject_ids, device):
        idx = [self.index[str(s)] for s in subject_ids]
        model.ssl_features = self.mean[idx].to(device)
        model.window_front = self.front[idx].to(device)
        model.window_back = self.back[idx].to(device)
        model.window_mask = self.mask[idx].to(device)


class WindowResidualSubject(ResidualSSLSubject):
    def __init__(self, backbone, mode):
        assert mode in ('mean', 'delta')
        super().__init__(backbone)
        self.window_mode = mode
        self.window_front = self.window_back = self.window_mask = None
        self.window_norm = nn.LayerNorm(1024, eps=1e-5, elementwise_affine=False)
        self.window_branch = nn.Sequential(nn.Linear(1024, 8), nn.GELU(), nn.Linear(8, 64))
        nn.init.zeros_(self.window_branch[-1].weight)
        nn.init.zeros_(self.window_branch[-1].bias)
        original_fused = self.backbone._encode_activities_with_wrists
        wrapper = self

        def window_encoding(_backbone, inputs, wrist_mask, activity_mask, activity_lengths):
            _, wrist = original_fused(inputs, wrist_mask, activity_mask, activity_lengths)
            added = wrapper.window_residual(wrist_mask, activity_mask)
            fused = wrist + added
            batch, activities = fused.shape[:2]
            wm = wrist_mask.bool() & activity_mask[..., None].bool()
            valid = activity_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
            sw = fused.reshape(-1, 2, 64).index_select(0, valid)
            sm = wm.reshape(-1, 2).index_select(0, valid)
            features, _ = _backbone.wrist_fusion.fusion_features(sw, sm)
            result = fused.new_zeros((batch * activities, 258)).index_copy(0, valid, features)
            return result.reshape(batch, activities, 258), fused

        self.backbone._encode_activities_with_wrists = types.MethodType(window_encoding, self.backbone)

    def window_residual(self, wrist_mask, activity_mask):
        f, b, mask = self.window_front, self.window_back, self.window_mask
        if f is None or b is None or mask is None:
            raise RuntimeError('Explicit subject-aligned front/back windows missing')
        assert f.shape == b.shape == (*wrist_mask.shape, 1024)
        assert mask.shape == (*wrist_mask.shape, 2)
        active = mask.all(-1) & wrist_mask.bool() & activity_mask[..., None].bool()
        x = (f + b) * .5 if self.window_mode == 'mean' else b - f
        return self.window_branch(self.window_norm(x)) * active[..., None]


class WindowLoader:
    def __init__(self, base, model, cache):
        self.base, self.model, self.cache = base, model, cache

    def __iter__(self):
        for batch in self.base:
            self.cache.attach(self.model, batch['subject_id'], next(self.model.parameters()).device)
            yield batch

    def __len__(self):
        return len(self.base)


def development_bundle(config, *, train_subject_ids, validation_subject_ids=(), test_subject_ids=(), fold_id):
    """Reuse original train-normalization implementation, excluding test partition."""
    assert not test_subject_ids, 'No outer subject partition enters this runner'
    allowed = set(map(str, train_subject_ids)) | set(map(str, validation_subject_ids))
    original_load = fd.load_configured_records
    # Original coverage assertion expects all configured records to be assigned.
    # Restrict to this development train/validation subset before construction.
    def development_records(*args, **kwargs):
        return [r for r in original_load(*args, **kwargs) if r.subject_id in allowed]
    fd.load_configured_records = development_records
    try:
        bundle = fd.build_subject_fold_datasets(config, train_subject_ids=train_subject_ids,
            validation_subject_ids=validation_subject_ids, test_subject_ids=(), fold_id=fold_id)
        assert bundle.test is None and bundle.split_summary['test_subject_count'] == 0
        return bundle
    finally:
        fd.load_configured_records = original_load


def activate(config, mode, cache):
    assert cache.activities == config['data']['activities']
    original_build, original_train, original_eval = nt.build_model, nt.train_epoch, nt.evaluate_epoch
    nt.build_model = lambda cfg: WindowResidualSubject(original_build(cfg), mode)
    nt.build_subject_fold_datasets = development_bundle
    def train(model, loader, *args, **kwargs):
        return original_train(model, WindowLoader(loader, model, cache), *args, **kwargs)
    def evaluate(model, loader, *args, **kwargs):
        return original_eval(model, WindowLoader(loader, model, cache), *args, **kwargs)
    nt.train_epoch, nt.evaluate_epoch = train, evaluate
