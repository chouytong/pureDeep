"""Restricted inner-development helpers; no SSL or outer evaluation imports."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path('/home/zyt/deep_final')
FOUNDATION = ROOT / 'foundation_validation'
HERE = Path(__file__).resolve().parent.parent
SPLIT = FOUNDATION / 'splits/pads_classification/v3_nested_cv/seed42/nested_cv_splits.json'
SPLIT_SHA = 'b3c52317cb12b73c66046bbd7c50c94a7e707a64a38d3ad4f0fd4bc0c6ba6b3e'
sys.path.insert(0, str(FOUNDATION))
from src.datasets import folds as fd
from src.utils.config import load_config


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def reference(seed, context, inner):
    return FOUNDATION / f'outputs/structured_token_residual/str01_seed{seed}_20260921/outer_{context}/inner_{inner}'


def config_for(seed):
    cfg = load_config(str(FOUNDATION / f'configs/str01_seed{seed}.yaml'))
    t = cfg['training']
    require(t['learning_rate'] == 2e-4 and t['weight_decay'] == 1e-4, 'LR/WD changed')
    require(t['batch_size'] == 8 and t['optimizer'] == 'adamw', 'Batch/optimizer changed')
    require(t['adam_betas'] == [.9, .999] and t['gradient_clip_norm'] == 5., 'Optimizer recipe changed')
    require(t['epochs'] == 50 and not t['mixed_precision'], 'Epoch/FP32 recipe changed')
    require(t['scheduler'] == {'type': 'cosine', 'minimum_lr': 1e-6}, 'Scheduler changed')
    require(t['early_stopping'] == {'enabled': True, 'patience': 12, 'metric': 'balanced_accuracy',
                                   'mode': 'max', 'minimum_delta': 0.}, 'Stopping changed')
    require(cfg['loss'] == {'label_smoothing': 0., 'class_weights': 'train_balanced'}, 'Loss changed')
    require(not cfg['evaluation']['run_test_after_training'], 'Outer flag enabled')
    require(cfg['data']['sample_rate'] == 100. and cfg['model']['sample_rate'] == 100., 'Sampling rate changed')
    require(cfg['data']['sensor_mode'] == 'acc_gyro' and cfg['data']['wrist_mode'] == 'bilateral', 'Input changed')
    require(not cfg.get('self_supervised', {}).get('enabled', False), 'SSL enabled')
    require(cfg['experiment']['seed'] == seed and cfg['experiment']['deterministic'], 'Seed/determinism changed')
    require(cfg['model']['structured_token_residual'] == {'enabled': True, 'projection_dim': 16}, 'STR changed')
    require(sha(SPLIT) == SPLIT_SHA, 'Split SHA changed')
    return cfg


def development_bundle(config, *, train_subject_ids, validation_subject_ids=(), test_subject_ids=(), fold_id):
    require(not test_subject_ids, 'Outer subjects supplied')
    allowed = set(map(str, train_subject_ids)) | set(map(str, validation_subject_ids))
    original = fd.load_configured_records
    def selected(*args, **kwargs):
        return [r for r in original(*args, **kwargs) if r.subject_id in allowed]
    fd.load_configured_records = selected
    try:
        bundle = fd.build_subject_fold_datasets(config, train_subject_ids=train_subject_ids,
                    validation_subject_ids=validation_subject_ids, test_subject_ids=(), fold_id=fold_id)
        require(bundle.test is None, 'Test dataset instantiated')
        return bundle
    finally:
        fd.load_configured_records = original


def guard(lock):
    for path, digest in lock['source_sha256'].items():
        require(sha(path) == digest, f'Frozen source/input identity changed: {path}')
    for stage in lock['stages']:
        for rel, digest in stage['files_sha256'].items():
            require(sha(Path(stage['path']) / rel) == digest, f'Frozen F0 artifact changed: {stage["path"]}/{rel}')


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
