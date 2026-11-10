"""Run via SSH stdin, with no writes to the frozen server experiment.

Only aggregate verification evidence is printed. No credentials, identifiers,
clinical rows, checkpoints or individual predictions leave the server.
"""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ARTIFACT = Path('/home/zyt/deep_final/artifacts/patch_str_20261008')
MIRROR = Path('/home/zyt/pureDeep-github-upload')
sys.path.insert(0, str(ARTIFACT / 'scripts'))
import common


def emit(**record):
    print(json.dumps(record, ensure_ascii=False), flush=True)


def checked_json(path, value):
    """Compare an output in memory; never overwrite a historical report."""
    expected = json.loads(Path(path).read_text())
    if Path(path).name in {'p1_tests.json', 'p2_tests.json'}:
        # Gradient magnitudes can vary with runtime; the original test asserts
        # zero/nonzero, finiteness and actual updates independently.
        for key in expected:
            if key != 'gradient_updates':
                common.require(value[key] == expected[key], f'JSON mismatch: {key}')
        emit(check='fresh_gradient_updates', condition=value['condition'],
             gradient_updates=value['gradient_updates'])
    else:
        common.require(value == expected, f'JSON mismatch: {Path(path).name}')
    emit(check='json_recomputed', file=Path(path).name, status='PASS')


def compare_csv(self, path=None, *args, **kwargs):
    import pandas as pd
    expected = pd.read_csv(path)
    pd.testing.assert_frame_equal(self.reset_index(drop=True), expected,
                                  check_dtype=False, check_exact=False,
                                  atol=1e-12, rtol=1e-12)
    emit(check='csv_recomputed', file=Path(path).name, rows=len(self), status='PASS')


def guards():
    from run_condition import study_guard
    common.guard(json.loads((ARTIFACT / 'analysis/f0_lock.json').read_text()))
    study_guard(json.loads((ARTIFACT / 'analysis/study_lock.json').read_text()))


def integrity():
    guards()
    study = json.loads((ARTIFACT / 'analysis/study_lock.json').read_text())
    for rel in [*study['files'], 'analysis/study_lock.json', 'PATCH_STR_REPORT.md']:
        common.require(common.sha(ARTIFACT / rel) ==
                       common.sha(MIRROR / 'artifacts/patch_str_20261008' / rel),
                       f'Source/publishing mirror differ: {rel}')
    milestones = {}
    for receipt in sorted(Path('/home/zyt').glob('pureDeep-patch-str-*-receipt-20261008.json')):
        item = json.loads(receipt.read_text())
        actual = subprocess.check_output(
            ['git', '-C', str(MIRROR), 'rev-parse', item['tag'] + '^{}'], text=True).strip()
        common.require(actual == item['commit'], 'Milestone tag mismatch')
        milestones[item['stage']] = actual
    common.require(len(milestones) == 6, 'Missing milestone')
    release = json.loads((ARTIFACT / 'analysis/preformal_release.json').read_text())
    common.require(common.sha(ARTIFACT / 'analysis/study_lock.json') ==
                   release['study_lock_sha256'], 'Freeze SHA mismatch')
    for rel, digest in study['files'].items():
        content = subprocess.check_output(['git', '-C', str(MIRROR), 'show',
            release['commit'] + ':artifacts/patch_str_20261008/' + rel])
        common.require(hashlib.sha256(content).hexdigest() == digest,
                       f'Preformal Git content differs: {rel}')
    common.require(not subprocess.check_output(
        ['git', '-C', str(MIRROR), 'status', '--porcelain'], text=True).strip(),
        'Publishing mirror has uncommitted changes')
    head = subprocess.check_output(['git', '-C', str(MIRROR), 'rev-parse', 'HEAD'], text=True).strip()
    refs = ['refs/heads/main'] + [
        json.loads(p.read_text())['tag'] + '^{}'
        for p in sorted(Path('/home/zyt').glob('pureDeep-patch-str-*-receipt-20261008.json'))]
    remote = subprocess.run(['git', '-C', str(MIRROR), 'ls-remote', 'origin', *refs],
                            capture_output=True, text=True, timeout=30)
    remote_status = 'UNAVAILABLE'
    if remote.returncode == 0:
        actual_refs = dict(line.split()[::-1] for line in remote.stdout.splitlines())
        common.require(actual_refs.get('refs/heads/main') == head, 'Remote main differs')
        for p in Path('/home/zyt').glob('pureDeep-patch-str-*-receipt-20261008.json'):
            item = json.loads(p.read_text())
            common.require(actual_refs.get('refs/tags/' + item['tag'] + '^{}') == item['commit'],
                           'Remote tag differs')
        remote_status = 'PASS'
    emit(stage='integrity', status='PASS', frozen_files=len(study['files']),
         milestone_commits=milestones, head=head, remote_main_and_tags=remote_status,
         original_assets_unchanged=True, outer_test_accessed=False,
         aggregate_sha256={rel: common.sha(ARTIFACT / rel) for rel in [
             'PATCH_STR_REPORT.md', 'BASELINE_AUDIT.md', 'PROTOCOL.md',
             'analysis/summary.csv', 'analysis/paired_comparisons.csv',
             'analysis/15split_seedfirst.csv', 'analysis/decision.json',
             'analysis/seed_means.csv', 'analysis/prediction_agreement.csv',
             'analysis/trajectory_summary.csv', 'analysis/timing_summary.csv']})


def baseline():
    """Rerun the original P0 audit, replacing all writes with comparisons.

    Its one no-overwrite precondition is inapplicable in this read-only wrapper;
    every other assertion is kept, including 45 full validation forwards,
    15 train-only normalization fits, and exact Phase2 state reproduction.
    """
    import pandas as pd
    import audit_p0
    original_require = audit_p0.require
    skipped = []

    def readonly_require(condition, message):
        if message == 'Do not overwrite completed baseline audit':
            common.require(not condition, 'Expected an existing completed audit')
            skipped.append(message)
            return
        original_require(condition, message)

    audit_p0.require = readonly_require
    audit_p0.write_json = checked_json
    with contextlib.ExitStack() as stack:
        from unittest.mock import patch
        stack.enter_context(patch.object(pd.DataFrame, 'to_csv', compare_csv))
        audit_p0.main()
    common.require(len(skipped) == 1, 'Unexpected no-overwrite guard handling')
    guards()
    emit(stage='baseline', status='PASS', full_validation_checkpoints=45,
         normalization_refits=15, historical_files_written=False,
         formal_training_started=False, outer_test_accessed=False)


def correctness():
    import pandas as pd
    from unittest.mock import patch
    import test_extraction
    import test_models
    import test_analysis_contract
    guards()
    for module in [test_extraction, test_models, test_analysis_contract]:
        module.write_json = checked_json
    with patch.object(pd.DataFrame, 'to_csv', compare_csv):
        test_extraction.main()
        for condition in ['P1', 'P2']:
            sys.argv = ['test_models.py', '--condition', condition]
            test_models.main()
        test_analysis_contract.main()
    # Validate preserved one-epoch engine smoke using original stage auditor.
    from run_condition import audit_stage
    lock = json.loads((ARTIFACT / 'analysis/f0_lock.json').read_text())
    study = json.loads((ARTIFACT / 'analysis/study_lock.json').read_text())
    inner = json.loads(common.SPLIT.read_text())['outer'][0]['inner_folds'][0]
    for condition in ['P1', 'P2']:
        stage = ARTIFACT / f'smoke/{condition}/seed42/outer_0/inner_0'
        actual = audit_stage(stage, condition, 42, 0, 0, inner, lock, study, True)
        common.require(actual == json.loads((stage / 'patch_audit.json').read_text()),
                       'Preserved smoke audit differs')
    guards()
    emit(stage='correctness', status='PASS', fresh_model_test_conditions=['P1', 'P2'],
         historical_smoke_reaudited=True, smoke_retrained=False,
         historical_files_written=False, outer_test_accessed=False)


def analysis():
    import pandas as pd
    from unittest.mock import patch
    import analyze_results
    analyze_results.write_json = checked_json
    with patch.object(pd.DataFrame, 'to_csv', compare_csv):
        analyze_results.main()
    # The historical timing summary averages per-run totals. Also report the
    # sum across 45 runs explicitly to avoid calling a per-run mean a total.
    runs = pd.read_csv(ARTIFACT / 'analysis/seed_split_metrics.csv')
    timing_rows = []
    for _, row in runs.iterrows():
        seed, c, i = map(int, [row.seed, row.context, row.inner])
        stage = common.reference(seed, c, i) if row.condition == 'P0' else ARTIFACT / f'runs/{row.condition}/seed{seed}/outer_{c}/inner_{i}'
        history = [json.loads(line) for line in (stage / 'logs/epochs.jsonl').read_text().splitlines()]
        train = [epoch['train']['duration_seconds'] for epoch in history]
        val = [epoch['validation']['duration_seconds'] for epoch in history]
        timing_rows.append(dict(condition=row.condition, seed=seed, context=c, inner=i,
            train_epoch_seconds_total=sum(train), validation_epoch_seconds_total=sum(val),
            mean_train_epoch_seconds=sum(train)/len(train),
            mean_validation_epoch_seconds=sum(val)/len(val),
            current_training_prediction_wall_seconds=row.training_and_prediction_seconds))
    times = pd.DataFrame(timing_rows)
    compare_csv(times, ARTIFACT / 'analysis/timing_runs.csv')
    means = times.groupby('condition', as_index=False).mean(numeric_only=True).drop(columns=['seed', 'context', 'inner'])
    compare_csv(means, ARTIFACT / 'analysis/timing_summary.csv')
    totals = times.groupby('condition', as_index=False)[[
        'train_epoch_seconds_total', 'validation_epoch_seconds_total',
        'current_training_prediction_wall_seconds']].sum(min_count=1)
    emit(check='time_totals_across_45_runs', values=totals.to_dict(orient='records'))
    guards()
    emit(stage='analysis', status='PASS', candidate_runs_reaudited=90,
         metric_runs_recomputed=135, seed_first_paired_units=15,
         historical_files_written=False, formal_training_started=False,
         decision='REJECT', outer_test_accessed=False)


def validation():
    """Full fresh validation inference for every trained P1/P2 best checkpoint."""
    import gc
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from test_models import dataset
    from patch_models import build_patch
    from src.datasets.subject_activity import collate_subject_activities
    from src.metrics.classification import classification_metrics
    from src.utils.seed import seed_everything
    seed_everything(42, True)
    torch.set_num_threads(4)
    guards()
    split = json.loads(common.SPLIT.read_text())
    count, maximum_difference = 0, 0.0
    for condition in ['P1', 'P2']:
        for seed in [42, 43, 44]:
            for outer in split['outer']:
                for inner in outer['inner_folds']:
                    c, i = int(outer['outer_fold']), int(inner['inner_fold'])
                    stage = ARTIFACT / f'runs/{condition}/seed{seed}/outer_{c}/inner_{i}'
                    ck = torch.load(stage / 'checkpoints/best.pt', map_location='cpu', weights_only=False)
                    saved = pd.read_csv(stage / 'predictions/validation.csv', dtype={'subject_id': str}).set_index('subject_id')
                    ds = dataset(ck, inner['validation_subjects'], 'validation')
                    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0,
                                        collate_fn=collate_subject_activities)
                    model = build_patch(ck['config'], condition).cuda().eval()
                    model.load_state_dict(ck['model_state'], strict=True)
                    ids, labels, probabilities = [], [], []
                    with torch.no_grad():
                        for batch in loader:
                            inputs = [batch[k].cuda() for k in ['x', 'wrist_mask', 'activity_mask', 'activity_lengths']]
                            probabilities.append(model(*inputs)['probabilities'].cpu().numpy())
                            ids.extend(batch['subject_id'])
                            labels.extend(batch['y'].tolist())
                    probability = np.concatenate(probabilities)
                    ref = saved.loc[ids]
                    common.require(set(ids) == set(inner['validation_subjects']), 'Incomplete fresh validation')
                    common.require(labels == ref.target.tolist(), 'Fresh validation labels differ')
                    difference = float(np.max(abs(probability - ref[['probability_pd', 'probability_dd']].to_numpy())))
                    common.require(difference <= 1e-12, 'Trained candidate probability differs')
                    common.require(np.array_equal(probability.argmax(1), ref.prediction), 'Fresh candidate decisions differ')
                    metric = classification_metrics(np.asarray(labels), probability.argmax(1), 2,
                                                     probabilities=probability)
                    expected = json.loads((stage / 'stage_status.json').read_text())['summary']['validation_metrics']
                    for key in ['accuracy', 'balanced_accuracy', 'macro_auroc', 'macro_f1', 'per_class_recall']:
                        common.require(np.allclose(metric[key], expected[key], rtol=0, atol=1e-12),
                                       f'Fresh candidate metric differs: {key}')
                    count += 1
                    maximum_difference = max(maximum_difference, difference)
                    del model, ck, ds, loader
                    gc.collect()
            emit(check='trained_validation', condition=condition, seed=seed,
                 completed_runs=count, status='PASS')
    common.require(count == 90, 'Missing candidate checkpoint')
    guards()
    emit(stage='validation', status='PASS', full_validation_checkpoints=count,
         max_probability_difference=maximum_difference, individual_predictions_exported=False,
         historical_files_written=False, outer_test_accessed=False)


if __name__ == '__main__':
    phase = sys.argv[1]
    common.require(phase in {'integrity', 'baseline', 'correctness', 'analysis', 'validation'}, 'Unknown phase')
    started = time.perf_counter()
    globals()[phase]()
    emit(completed_phase=phase, elapsed_seconds=time.perf_counter() - started)
