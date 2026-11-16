"""Named model registry; original src.models.build_model stays byte-for-byte intact."""
from pathlib import Path
from copy import deepcopy
import json
import yaml

MAINLINE = Path(__file__).resolve().parents[2]


def _registry():
    return json.loads((MAINLINE / 'models' / 'registry.json').read_text())


def get_model_spec(name):
    key = ''.join(c for c in str(name).lower() if c.isalnum())
    for record in _registry()['models']:
        if key in {''.join(c for c in n.lower() if c.isalnum())
                   for n in [record['model_id'], record['display_name'], *record['aliases']]}:
            return deepcopy(record)
    raise ValueError(f'Unregistered model: {name!r}')


def list_models():
    return deepcopy(_registry()['models'])


def load_registered_config(name, seed=42):
    spec = get_model_spec(name)
    if seed not in [42, 43, 44]:
        raise ValueError('Existing registered configs cover seeds42/43/44 only')
    path = MAINLINE / spec['configuration_pattern'].format(seed=seed)
    config = yaml.safe_load(path.read_text())
    return config


def build_registered_model(name, config=None, *, seed=42):
    """Build by model name without changing retained pointers or historical factories."""
    spec = get_model_spec(name)
    config = load_registered_config(name, seed) if config is None else deepcopy(config)
    from .pure_deep import build_pure_deep_subject
    if spec['model_id'] == 'str01':
        return build_pure_deep_subject(config)
    if spec['model_id'] == 'frozen_wssl':
        from .wssl import ResidualSSLSubject
        return ResidualSSLSubject(build_pure_deep_subject(config))
    if spec['model_id'] == 'ai_wssl_str':
        from .ai_wssl_str import AIWSSLSTR
        return AIWSSLSTR(config)
    raise ValueError('Missing registered factory')
