"""One prespecified WSSL activity-scaling control, reusing the original inner engine.

No EMA, new feature, encoder adaptation or decision rule is added.
The original 2e-4 recipe is unchanged; exactly eleven residual scalars are added.
Never resume or overwrite an incomplete run: the old checkpoints do not save RNG.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
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
EMA_ARCHIVE = Path('/home/zyt/deep_final/artifacts/wssl_ema_20261002')
BASELINE_LOCK = Path('/home/zyt/deep_final/artifacts/phase4_window_dynamics_20261002/analysis/baseline_lock.json')
sys.path[:0] = [str(FOUNDATION), str(BASELINE / 'scripts'), str(EMA_ARCHIVE / 'scripts')]

from independent_wssl import IndependentWSSL
from scaled_wssl import ActivityScaledWSSL
from ema_hooks import development_bundle  # Helper only; EMA activate/update is never called.
from transfer_hooks import FrozenEmbeddingCache, SSLLoader, CACHE_PATH, ResidualSSLSubject
from src.engine import nested_training as nt
from src.metrics.classification import classification_metrics
from src.utils.config import load_config, require_pads_classification_config
from src.utils.device import select_device
from src.models import build_model as original_model_factory
from src.utils.seed import seed_everything
from src.datasets.subject_activity import SubjectActivityDataset, collate_subject_activities


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def guard_inputs(baseline_lock, study_lock):
    for relative, digest in study_lock['files'].items():
        require(sha(HERE / relative) == digest, f'Pre-registered file changed: {relative}')
    require('PROTOCOL.md' in study_lock['files'], 'Protocol not in the study lock')
    require('scripts/run.py' in study_lock['files'], 'Runner not in the study lock')
    for path, digest in baseline_lock['source_sha256'].items():
        require(sha(path) == digest, f'Frozen source changed: {path}')
    require(sha(CACHE_PATH) == baseline_lock['cache_sha256'], 'Frozen cache SHA mismatch')
    require(sha(baseline_lock['split_path']) == baseline_lock['split_sha256'], 'Fixed split SHA mismatch')
    for path, digest in study_lock.get('additional_source_sha256', {}).items():
        require(sha(path) == digest, f'Additional frozen source changed: {path}')
    weight = BASELINE / 'vendor/ssl-wearables/model_check_point/mtl_best.mdl'
    require(sha(weight) == 'c64f9135d99e2dcdfc9ae7cc0672f2bcc438df9ceb8215665882f92cddd162a6',
            'Official frozen HarNet weight SHA changed')
    # Both the independent architecture and the data-only helper were frozen in
    # the completed EMA study. Importing the helper does not instantiate EMA.
    archived_lock = json.loads((EMA_ARCHIVE / 'analysis/ema_protocol_lock.json').read_text())
    for relative in ('scripts/independent_wssl.py', 'scripts/ema_hooks.py'):
        require(sha(EMA_ARCHIVE / relative) == archived_lock['files'][relative],
                f'Previously validated independent interface changed: {relative}')


def fixed_recipe(config):
    require_pads_classification_config(config)
    training = config['training']
    require(training['optimizer'].lower() == 'adamw', 'Optimizer changed')
    require(float(training['learning_rate']) == 2e-4, 'Starting recipe is not archived WSSL')
    require(float(training['weight_decay']) == 1e-4, 'Weight decay changed')
    require(training['adam_betas'] == [0.9, 0.999], 'AdamW betas changed')
    require(int(training['batch_size']) == 8 and int(config['evaluation']['batch_size']) == 8, 'Batch size changed')
    require(training['scheduler'] == {'type': 'cosine', 'minimum_lr': 1e-6}, 'Scheduler changed')
    require(int(training['epochs']) == 50 and float(training['gradient_clip_norm']) == 5, 'Epoch/clip changed')
    require(not training['mixed_precision'], 'AMP changed')
    require(int(training.get('overfit_batches', 0)) == 0, 'Training batch limiter enabled')
    require(training['early_stopping'] == {'enabled': True, 'patience': 12,
            'metric': 'balanced_accuracy', 'mode': 'max', 'minimum_delta': 0.0}, 'Stopping rule changed')
    require(config['loss']['class_weights'] == 'train_balanced', 'Balanced CE protocol changed')
    require(float(config['loss']['label_smoothing']) == 0, 'Label smoothing changed')
    require(not config.get('self_supervised', {}).get('enabled', False), 'Pretraining unexpectedly enabled')
    require(not config.get('distributed', {}).get('enabled', False), 'Distributed recipe changed')
    require(config['experiment'].get('deterministic', False), 'Determinism disabled')
    require(config['data']['wrist_mode'] == 'bilateral' and config['data']['sensor_mode'] == 'acc_gyro', 'Input selection changed')
    require(config['model']['structured_token_residual'] == {'enabled': True, 'projection_dim': 16}, 'STR path changed')
    require(not config['evaluation'].get('run_test_after_training', False), 'Outer evaluation flag enabled')


def activate_no_ema(cache, config):
    """Patch this process only, leaving all source files and optimizer logic intact."""
    require(cache.activities == config['data']['activities'], 'Activity order differs')
    old = {name: getattr(nt, name) for name in
           ('build_model', 'build_subject_fold_datasets', 'train_epoch', 'evaluate_epoch', '_loader')}
    loaders = []

    def model_factory(cfg):
        model = ActivityScaledWSSL(cfg)
        require(sum(p.numel() for p in model.parameters()) == 143183, 'Model parameter count changed')
        require(not dict(model.named_buffers()), 'Unexpected classifier buffer')
        return model

    def train(model, loader, *args, **kwargs):
        return old['train_epoch'](model, SSLLoader(loader, model, cache, 'pretrained'), *args, **kwargs)

    def evaluate(model, loader, *args, **kwargs):
        return old['evaluate_epoch'](model, SSLLoader(loader, model, cache, 'pretrained'), *args, **kwargs)

    def tracked_loader(*args, **kwargs):
        loader = old['_loader'](*args, **kwargs)
        if loader is not None:
            loaders.append(loader)
        return loader

    nt.build_model = model_factory
    nt.build_subject_fold_datasets = development_bundle
    nt.train_epoch = train
    nt.evaluate_epoch = evaluate
    nt._loader = tracked_loader
    return old, loaders


def cleanup_loaders(loaders):
    for loader in loaders:
        iterator = getattr(loader, '_iterator', None)
        if iterator is not None:
            iterator._shutdown_workers()
            loader._iterator = None
    loaders.clear()
    gc.collect()


def audit_stage(stage, reference, inner, expected_rate, seed, context, inner_index,
                smoke, baseline_entry, study_lock):
    status = json.loads((stage / 'stage_status.json').read_text())
    require(status['status'] == 'complete' and status['phase'] == 'inner', 'Run is not complete inner development')
    require(status['outer_test_loader_created'] is False, 'Outer loader marker is not false')
    summary = status['summary']
    require(summary['smoke'] == smoke, 'Smoke/formal artifact mismatch')
    require(summary['selection_metric'] == 'balanced_accuracy', 'Selection metric differs')
    require(sha(reference / 'checkpoints/best.pt') == baseline_entry['checkpoint_sha256'], 'Reference checkpoint changed')
    require(sha(reference / 'predictions/validation.csv') == baseline_entry['prediction_sha256'], 'Reference predictions changed')
    checkpoint = torch.load(stage / 'checkpoints/best.pt', map_location='cpu', weights_only=False)
    baseline = torch.load(reference / 'checkpoints/best.pt', map_location='cpu', weights_only=False)
    actual, original = checkpoint['config'], baseline['config']
    require(checkpoint['model_state']['activity_ssl_scale'].shape == (11,), 'Scalar contract differs')
    require(torch.isfinite(checkpoint['model_state']['activity_ssl_scale']).all().item(), 'Nonfinite scales')
    for key in ('model', 'data', 'evaluation', 'loss', 'self_supervised', 'nested_cv'):
        require(actual.get(key) == original.get(key), f'Frozen recipe section changed: {key}')
    training = copy.deepcopy(actual['training'])
    require(float(training['learning_rate']) == expected_rate, 'Wrong candidate learning rate')
    training['learning_rate'] = 2e-4
    require(training == original['training'], 'Training hyperparameter changed')
    require(int(actual['experiment']['seed']) == seed, 'Seed changed')
    require(actual['experiment']['deterministic'], 'Determinism changed')
    require(checkpoint['class_names'] == baseline['class_names'] == ['PD', 'DD'], 'Class direction changed')
    for key in ('mean', 'std'):
        require(torch.equal(checkpoint['normalization'][key], baseline['normalization'][key]), 'Train-only normalization differs')
    reference_summary = json.loads((reference / 'stage_status.json').read_text())['summary']
    for key in ('train_subject_ids_sha256', 'normalization_sha256'):
        require(summary[key] == reference_summary[key], f'Data identity differs: {key}')
    require(summary['checkpoint_sha256'] == sha(stage / 'checkpoints/best.pt'), 'Selected checkpoint SHA differs')
    require(summary['best_epoch'] == checkpoint['epoch'] + 1, 'Selected epoch inconsistent')
    split = json.loads((stage / 'split.json').read_text())
    require(set(split['train_subjects']) == set(inner['train_subjects']), 'Train subjects differ')
    require(set(split['validation_subjects']) == set(inner['validation_subjects']), 'Validation subjects differ')
    require(not split['test_subjects'], 'Test subject partition entered this run')
    require(not set(split['train_subjects']).intersection(split['validation_subjects']), 'Subject overlap')
    require(not list((stage / 'checkpoints').glob('ema*')), 'Unexpected EMA artifact')
    history = [json.loads(line) for line in (stage / 'logs/epochs.jsonl').read_text().splitlines()]
    require(1 <= len(history) <= (1 if smoke else 50), 'Invalid epoch coverage')
    require([entry['epoch'] for entry in history] == list(range(len(history))), 'Non-contiguous epoch history')
    require(all(entry['selection_metric'] == 'balanced_accuracy' for entry in history), 'Selection rule changed in history')
    final_record = history[-1]
    require(final_record['best_epoch'] == summary['best_epoch'], 'History selected epoch differs')
    running_best, first_best, patience = float('-inf'), 0, 0
    for entry in history:
        value = float(entry['validation']['balanced_accuracy'])
        improved = value > running_best
        require(entry['improved'] == improved, 'Best selection is not strict ordinary BA improvement')
        if improved:
            running_best, first_best, patience = value, entry['epoch'] + 1, 0
        else:
            patience += 1
        require(entry['best_epoch'] == first_best and entry['patience_count'] == patience,
                'BA-selection or patience history differs')
    require(summary['best_epoch'] == first_best, 'Selected epoch is not first ordinary BA maximum')
    require(smoke or len(history) == 50 or patience == 12, 'Stopped outside fixed cap/patience rule')
    last = torch.load(stage / 'checkpoints/last.pt', map_location='cpu', weights_only=False)
    require(last['epoch'] == len(history) - 1, 'Last checkpoint does not match stopping epoch')
    prediction_path = stage / 'predictions/validation.csv'
    frame = pd.read_csv(prediction_path, dtype={'subject_id': str})
    require(not frame.subject_id.duplicated().any(), 'Duplicate prediction subjects')
    reference_frame = pd.read_csv(reference / 'predictions/validation.csv', dtype={'subject_id': str})
    labels = dict(zip(reference_frame.subject_id, reference_frame.target))
    require(all(s in set(inner['validation_subjects']) for s in frame.subject_id), 'Prediction outside validation')
    require(frame.target.tolist() == [int(labels[s]) for s in frame.subject_id], 'Labels differ by subject ID')
    if not smoke:
        require(set(frame.subject_id) == set(inner['validation_subjects']), 'Incomplete validation predictions')
    probabilities = frame[['probability_pd', 'probability_dd']].to_numpy()
    require(np.isfinite(probabilities).all(), 'Nonfinite probability')
    require(np.array_equal(frame.prediction.to_numpy(), probabilities.argmax(axis=1)), 'Decision rule differs')
    require(not any('threshold' in str(name) for name in frame.columns), 'Threshold tuning fields unexpectedly present')
    metrics = classification_metrics(frame.target.to_numpy(), frame.prediction.to_numpy(), 2, probabilities=probabilities)
    for metric in ('accuracy', 'balanced_accuracy', 'macro_auroc', 'macro_f1'):
        actual_metric, saved_metric = metrics[metric], summary['validation_metrics'][metric]
        if actual_metric is None or saved_metric is None:
            require(actual_metric is None and saved_metric is None, f'Archived metric availability mismatch: {metric}')
        else:
            require(abs(actual_metric - saved_metric) < 1e-12, f'Archived metric mismatch: {metric}')
    audit = dict(status='PASS', seed=seed, context=context, inner=inner_index,
                 learning_rate=expected_rate, smoke=smoke,
                 best_epoch=summary['best_epoch'], stop_epoch=len(history),
                 validation_subject_count=len(frame), normalization_exact_to_reference=True,
                 input_backbone_loss_optimizer_scheduler_stopping_unchanged=True, added_parameters=11,
                 threshold=0.5, threshold_selection_performed=False,
                 decision_rule='Original two-class probability argmax; exact tie PD',
                 ema_instantiated=False, encoder_adaptation=False, outer_test_loader_created=False,
                 protocol_sha256=study_lock['files']['PROTOCOL.md'],
                 runner_sha256=sha(__file__),
                 files_sha256={str(p.relative_to(stage)): sha(p) for p in (
                     stage / 'checkpoints/best.pt', stage / 'checkpoints/last.pt',
                     stage / 'stage_status.json', stage / 'logs/epochs.jsonl',
                     stage / 'config.yaml', stage / 'split.json', stage / 'normalization.json', prediction_path)})
    del checkpoint, baseline, last
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=[42, 43, 44], required=True)
    parser.set_defaults(learning_rate=2e-4)
    parser.add_argument('--context', type=int, choices=range(5))
    parser.add_argument('--inner', type=int, choices=range(3))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--reuse-complete', action='store_true')
    args = parser.parse_args()
    require(args.learning_rate == 2e-4, 'Only unchanged LR is allowed')
    require(args.smoke or (args.context is None and args.inner is None), 'Formal execution covers all 15 splits per seed')
    torch.set_num_threads(4)
    baseline_lock = json.loads(BASELINE_LOCK.read_text())
    study_lock = json.loads((HERE / 'analysis/study_lock.json').read_text())
    require(json.loads((HERE/'analysis/correctness.json').read_text())['status']=='PASS', 'Correctness gate failed')
    guard_inputs(baseline_lock, study_lock)
    config = load_config(str(FOUNDATION / f'configs/str01_seed{args.seed}.yaml'))
    fixed_recipe(config)
    variant = 'activity_scaling'
    config['training']['learning_rate'] = args.learning_rate
    config['experiment']['name'] = f'wssl_review_{variant}_seed{args.seed}'
    config['experiment']['output_root'] = str(HERE / ('smoke' if args.smoke else 'runs') / variant / f'seed{args.seed}')
    config['development'] = dict(single_activity_scaling=True, scalar_count=11, initialization=1., no_outer_access=True, no_ema=True, no_encoder_adaptation=True, protocol_sha256=study_lock['files']['PROTOCOL.md'])
    cache = FrozenEmbeddingCache()
    require(torch.isfinite(cache.pretrained).all().item(), 'Retained feature cache is nonfinite')
    original, loaders = activate_no_ema(cache, config)
    stage_lookup = {(s['seed'], s['context'], s['inner']): s for s in baseline_lock['stages']}
    completed = 0
    try:
        split, _, _, digest = nt._load_frozen_split(config)
        require(digest == baseline_lock['split_sha256'], 'Frozen split differs')
        device = select_device('cuda')
        if not args.smoke:
            smoke_path = HERE / 'smoke/activity_scaling/seed42/outer_0/inner_0/scaling_audit.json'
            smoke_audit = json.loads(smoke_path.read_text())
            require(smoke_audit['status'] == 'PASS' and smoke_audit['smoke'], 'Formal entry requires candidate engine smoke PASS')
            require(smoke_audit['runner_sha256'] == sha(__file__), 'Smoke uses different runner')
        for context in split['outer']:
            context_index = int(context['outer_fold'])
            if args.context is not None and args.context != context_index:
                continue
            for inner in context['inner_folds']:
                inner_index = int(inner['inner_fold'])
                if args.inner is not None and args.inner != inner_index:
                    continue
                stage = Path(config['experiment']['output_root']) / f'outer_{context_index}/inner_{inner_index}'
                reference = BASELINE / f'runs/b_str_pretrained/seed{args.seed}/outer_{context_index}/inner_{inner_index}'
                audit_path = stage / 'scaling_audit.json'
                try:
                    if stage.exists():
                        require(args.reuse_complete, f'Existing stage requires explicit --reuse-complete: {stage}')
                        require(audit_path.is_file(), f'Existing stage lacks completed audit; no resume/overwrite: {stage}')
                        saved = json.loads(audit_path.read_text())
                        require(saved['status'] == 'PASS' and saved['smoke'] == args.smoke, 'Completed audit identity differs')
                        require(saved['runner_sha256'] == sha(__file__) and saved['protocol_sha256'] == study_lock['files']['PROTOCOL.md'], 'Completed run uses different frozen code/protocol')
                        for relative, expected in saved['files_sha256'].items():
                            require(sha(stage / relative) == expected, f'Completed artifact changed: {stage / relative}')
                        audit = audit_stage(stage, reference, inner, args.learning_rate, args.seed,
                                            context_index, inner_index, args.smoke,
                                            stage_lookup[(args.seed, context_index, inner_index)], study_lock)
                        require(audit == saved, 'Completed-run audit no longer matches current state')
                        reused = True
                    else:
                        guard_inputs(baseline_lock, study_lock)
                        nt._run_inner_fold(config, {'outer_fold': context_index, 'test_subjects': []},
                                           inner, stage, device, resume=False, smoke=args.smoke)
                        audit = audit_stage(stage, reference, inner, args.learning_rate, args.seed,
                                            context_index, inner_index, args.smoke,
                                            stage_lookup[(args.seed, context_index, inner_index)], study_lock)
                        guard_inputs(baseline_lock, study_lock)
                        audit_path.write_text(json.dumps(audit, indent=2) + '\n')
                        reused = False
                    completed += 1
                    print(json.dumps(dict(status='PASS', seed=args.seed, context=context_index,
                                          inner=inner_index, learning_rate=args.learning_rate,
                                          smoke=args.smoke, reused_complete=reused,
                                          best_epoch=audit['best_epoch'], stop_epoch=audit['stop_epoch'],
                                          no_ema=True, outer_test_loader_created=False)), flush=True)
                finally:
                    cleanup_loaders(loaders)
        require(args.smoke or completed == 15, 'Formal seed run did not complete all 15 fixed splits')
        guard_inputs(baseline_lock, study_lock)
        print(json.dumps(dict(status='COMPLETE', seed=args.seed, learning_rate=args.learning_rate,
                              completed_inner_splits=completed, smoke=args.smoke,
                              no_ema=True, outer_performance_accessed=False)), flush=True)
    finally:
        cleanup_loaders(loaders)
        for name, function in original.items():
            setattr(nt, name, function)


if __name__ == '__main__':
    main()
