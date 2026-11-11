"""Mainline only: STR-01. Historical MFAM/replication factories are excluded."""
from .pure_deep import PureDeepSubjectModel, build_pure_deep_subject

def build_model(config):
    if config['model'].get('name') != 'pure_deep_subject':
        raise ValueError('This mainline entry only supports the retained STR architecture')
    return build_pure_deep_subject(config)
