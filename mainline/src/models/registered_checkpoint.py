"""Hash-verified named checkpoint loading, with non-overwriting registered saves."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import os
import torch
from .registry import MAINLINE, get_model_spec, load_registered_config, build_registered_model


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def list_checkpoints(model_name='AI-WSSL-STR'):
    spec = get_model_spec(model_name)
    if not spec.get('checkpoint_catalog'):
        return []
    return deepcopy(json.loads((MAINLINE / spec['checkpoint_catalog']).read_text())['checkpoints'])


def resolve_checkpoint(model_name, checkpoint_id):
    spec = get_model_spec(model_name)
    records = [r for r in list_checkpoints(model_name) if r['checkpoint_id'] == checkpoint_id]
    if len(records) != 1 or records[0]['model_id'] != spec['model_id']:
        raise ValueError('Unknown checkpoint ID or model mismatch')
    return records[0]


def _state_model_id(state):
    if 'base.norm.weight' in state and 'output.weight' in state:
        return 'ai_wssl_str'
    if 'norm.weight' in state and 'wrist_projection.weight' in state:
        return 'frozen_wssl'
    if any(k.startswith('wrist_encoder.') for k in state):
        return 'str01'
    raise ValueError('Unrecognized checkpoint state layout')


def load_registered_checkpoint(model_name, *, checkpoint_id=None, checkpoint_path=None,
                               expected_sha256=None, config=None, device='cpu'):
    spec = get_model_spec(model_name)
    if (checkpoint_id is None) == (checkpoint_path is None):
        raise ValueError('Specify exactly one checkpoint_id or checkpoint_path')
    entry = resolve_checkpoint(model_name, checkpoint_id) if checkpoint_id is not None else None
    path = Path(entry['path'] if entry else checkpoint_path)
    expected = entry['sha256'] if entry else expected_sha256
    if not expected or file_sha256(path) != expected:
        raise ValueError('Checkpoint SHA256 missing or mismatched')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    state = payload.get('model_state')
    if not isinstance(state, dict) or _state_model_id(state) != spec['model_id']:
        raise ValueError('Checkpoint architecture/model identity mismatch')
    identity = payload.get('model_identity')
    if identity is not None:
        if identity['model_id'] != spec['model_id'] or identity['status'] != spec['status']:
            raise ValueError('Registered identity/status mismatch')
    if entry:
        for key in ['seed', 'context', 'inner']:
            if payload.get(key) != entry[key]:
                raise ValueError(f'Checkpoint catalog {key} mismatch')
        if payload.get('source_sha256') != entry['training_source_sha256']:
            raise ValueError('Checkpoint source provenance mismatch')
        for key in ['base_checkpoint_sha256', 'normalization_sha256']:
            if entry.get(key) is not None and payload.get(key) != entry[key]:
                raise ValueError(f'Checkpoint {key} provenance mismatch')
    seed = int(entry['seed'] if entry else payload.get('seed', 42))
    if config is None:
        config = payload.get('registered_config') or load_registered_config(model_name, seed)
        if entry is not None and spec['model_id'] == 'ai_wssl_str':
            config['data']['version'] = entry['data_version']
            if entry['evidence_scope'] == 'legacy_l1_15split_development':
                config['data']['root'] = '/home/zyt/deep_final/data/processed/pads_multi_activity/v2_l1_full_length'
    target_version = entry['data_version'] if entry else payload.get('data_version', payload.get('version'))
    if spec['model_id'] == 'ai_wssl_str' and config['data'].get('version') != target_version:
        raise ValueError('AI checkpoint/config data version mismatch')
    if int(config['experiment']['seed']) != seed:
        raise ValueError('Config/checkpoint seed mismatch')
    model = build_registered_model(model_name, config)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, dict(model_id=spec['model_id'], display_name=spec['display_name'],
                       status=spec['status'], checkpoint_id=checkpoint_id,
                       checkpoint_sha256=expected, seed=seed,
                       context=payload.get('context'), inner=payload.get('inner'),
                       source_commit=entry.get('source_commit') if entry else None,
                       data_version=target_version,
                       base_checkpoint_sha256=payload.get('base_checkpoint_sha256'),
                       normalization_sha256=payload.get('normalization_sha256'),
                       source_layout='historical_strict_compatible' if identity is None else 'registered_v1')


def save_registered_checkpoint(model_name, model, path, config, *, provenance):
    """New identity format; never rewrites historical weights or promotes a model."""
    spec = get_model_spec(model_name)
    if _state_model_id(model.state_dict()) != spec['model_id']:
        raise ValueError('Saving model identity mismatch')
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Preserve existing checkpoint: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    reserved = {'model_state', 'model_identity', 'registered_config'}
    if reserved & set(provenance):
        raise ValueError('Provenance cannot overwrite model identity/state')
    payload = dict(provenance)
    payload.update(model_state=model.state_dict(), registered_config=deepcopy(config),
                   model_identity=dict(format_version=1, model_id=spec['model_id'],
                                       display_name=spec['display_name'], status=spec['status'],
                                       architecture_version=spec['architecture_version']))
    partial = path.with_name(path.name + f'.partial.{os.getpid()}')
    if partial.exists():
        raise FileExistsError('Preserve existing incomplete checkpoint')
    try:
        with partial.open('xb') as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        # A hard link atomically refuses to replace an existing destination.
        os.link(partial, path)
        partial.unlink()
    except Exception:
        # Preserve incomplete writes for diagnosis; no fallback to old checkpoints.
        raise
    return file_sha256(path)
