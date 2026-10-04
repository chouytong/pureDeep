"""Fixed full-matrix analysis for the single existing-WSSL LR control.

No partial-matrix selection. Individual predictions remain server-only; outputs
are run/split aggregates. Fifteen overlapping development splits are descriptive
paired units, after averaging the three seed metrics within each split.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
sys.path.insert(0, '/home/zyt/deep_final/foundation_validation')
from src.metrics.classification import classification_metrics

HERE = Path(__file__).resolve().parent.parent
BASELINE = Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928/runs/b_str_pretrained')
BASELINE_LOCK = Path('/home/zyt/deep_final/artifacts/phase4_window_dynamics_20261002/analysis/baseline_lock.json')
METRICS = ['accuracy', 'ba', 'auroc', 'macro_f1', 'pd_recall', 'dd_recall']
SEEDS = (42, 43, 44)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def measures(y, probabilities, prediction):
    require(np.array_equal(prediction, probabilities.argmax(axis=1)), 'Decision rule is not original argmax')
    require(np.array_equal(prediction, (probabilities[:,1] > .5).astype(int)), 'Decision rule is not fixed .5/tiePD')
    actual = classification_metrics(y, prediction, 2, probabilities=probabilities)
    return dict(accuracy=actual['accuracy'], ba=actual['balanced_accuracy'], auroc=actual['macro_auroc'],
                macro_f1=actual['macro_f1'], pd_recall=actual['per_class_recall'][0], dd_recall=actual['per_class_recall'][1])


def bh(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    q = np.minimum.accumulate((pvalues[order] * len(order) / np.arange(1, len(order) + 1))[::-1])[::-1]
    output = np.empty_like(q)
    output[order] = np.minimum(q, 1)
    return output


def verify_lock():
    lock = json.loads((HERE / 'analysis/lr_control_lock.json').read_text())
    require('scripts/analyze_lr_control.py' in lock['files'], 'Analysis was not frozen before results')
    for relative, digest in lock['files'].items():
        require(sha(HERE / relative) == digest, f'Frozen file changed: {relative}')
    for path, digest in lock.get('additional_source_sha256', {}).items():
        require(sha(path) == digest, f'Frozen additional source changed: {path}')
    for name in ('assets_normalization_audit.json', 'fusion_properties.json', 'preprocessing_contract.json'):
        require(json.loads((HERE / 'analysis' / name).read_text())['status'] == 'PASS', f'Entry audit missing: {name}')
    base = json.loads(BASELINE_LOCK.read_text())
    for path, digest in base['source_sha256'].items():
        require(sha(path) == digest, f'Original source changed: {path}')
    require(sha(base['cache_path']) == base['cache_sha256'], 'Original cache changed')
    require(sha(base['split_path']) == base['split_sha256'], 'Original splits changed')
    return lock, base


def inspect_stage(stage, variant, seed, context, inner, expected, lock):
    status = json.loads((stage / 'stage_status.json').read_text())
    require(status['status'] == 'complete' and status['phase'] == 'inner', f'Incomplete run: {stage}')
    require(status['outer_test_loader_created'] is False and status['summary']['smoke'] is False, 'Wrong run scope')
    if variant == 'baseline':
        for filename, key in [('checkpoints/best.pt', 'checkpoint_sha256'),
                              ('predictions/validation.csv', 'prediction_sha256'),
                              ('normalization.json', 'normalization_file_sha256')]:
            require(sha(stage / filename) == expected[key], f'Original artifact changed: {stage / filename}')
    else:
        audit = json.loads((stage / 'lr_control_audit.json').read_text())
        require(audit['status'] == 'PASS' and audit['smoke'] is False, 'Candidate integrity audit missing')
        require((audit['seed'], audit['context'], audit['inner']) == (seed, context, inner), 'Candidate audit identity differs')
        require(audit['learning_rate'] == 1e-4 and audit['normalization_exact_to_reference'], 'Candidate is not sole LR control')
        require(audit['input_architecture_loss_optimizer_scheduler_stopping_unchanged'] is True,
                'Candidate audit does not confirm unchanged non-LR recipe')
        require(audit['threshold'] == .5 and audit['threshold_selection_performed'] is False, 'Threshold changed')
        require(audit['ema_instantiated'] is False and audit['encoder_adaptation'] is False, 'Unexpected candidate modification')
        require(audit['outer_test_loader_created'] is False, 'Outer access marker differs')
        require(audit['protocol_sha256'] == lock['files']['LR_CONTROL_PROTOCOL.md'], 'Protocol identity differs')
        require(audit['runner_sha256'] == lock['files']['scripts/run_lr_control.py'], 'Runner identity differs')
        required_files = {'checkpoints/best.pt','checkpoints/last.pt','stage_status.json','logs/epochs.jsonl',
                          'config.yaml','split.json','normalization.json','predictions/validation.csv'}
        require(required_files.issubset(audit['files_sha256']), 'Candidate integrity hashes omit required artifacts')
        for relative, digest in audit['files_sha256'].items():
            require(sha(stage / relative) == digest, f'Candidate artifact changed: {stage / relative}')
    summary = status['summary']
    require(summary['selection_metric'] == 'balanced_accuracy', 'Selection rule changed')
    frame = pd.read_csv(stage / 'predictions/validation.csv', dtype={'subject_id': str}).sort_values('subject_id').reset_index(drop=True)
    require(frame.subject_id.is_unique, 'Duplicate prediction ID')
    split = json.loads((stage / 'split.json').read_text())
    require(set(frame.subject_id) == set(split['validation_subjects']), 'Incomplete validation subject set')
    require(not set(split['train_subjects']).intersection(frame.subject_id), 'Train-validation overlap')
    # Original historical baseline split may contain historical context metadata;
    # only new control must declare an empty test assignment.
    if variant != 'baseline':
        require(not split['test_subjects'], 'Candidate includes test assignment')
    probability = frame.probability_dd.to_numpy(float)
    require(np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all(), 'Invalid probabilities')
    y = frame.target.to_numpy(int)
    require(set(y) == {0, 1}, 'Both classes required')
    prediction = frame.prediction.to_numpy(int)
    values = measures(y, frame[['probability_pd','probability_dd']].to_numpy(float), prediction)
    for metric, archived in [('accuracy', 'accuracy'), ('ba', 'balanced_accuracy'), ('auroc','macro_auroc'), ('macro_f1', 'macro_f1'),
                              ('pd_recall', 'pd_recall'), ('dd_recall', 'dd_recall')]:
        require(abs(values[metric] - summary['validation_metrics'][archived]) < 1e-12, f'Archived {metric} differs')
    history = [json.loads(line) for line in (stage / 'logs/epochs.jsonl').read_text().splitlines()]
    require(1 <= summary['best_epoch'] <= len(history) <= 50, 'Invalid epoch coverage')
    best = history[summary['best_epoch'] - 1]
    last = history[-1]
    row = dict(variant=variant, seed=seed, context=context, inner=inner, subject_count=len(frame),
               best_epoch=summary['best_epoch'], stop_epoch=len(history), epoch_cap_hit=len(history) == 50,
               train_online_loss=best['train']['loss'], validation_loss=best['validation']['loss'],
               best_train_online_classification_loss=best['train']['classification_loss'],
               best_validation_classification_loss=best['validation']['classification_loss'],
               last_train_online_loss=last['train']['loss'], last_validation_loss=last['validation']['loss'],
               last_train_online_classification_loss=last['train']['classification_loss'],
               last_validation_classification_loss=last['validation']['classification_loss'], **values)
    return frame, row, summary


def main():
    lock, base = verify_lock()
    lookup = {(s['seed'], s['context'], s['inner']): s for s in base['stages']}
    require(len(lookup) == 45, 'Original reference lock incomplete')
    roots = {'baseline': BASELINE, 'lr_1e4': HERE / 'runs/lr_1e4'}
    rows, errors, agreement = [], [], []
    for context in range(5):
        for inner in range(3):
            probabilities = {variant: [] for variant in roots}
            predictions = {variant: [] for variant in roots}
            seed_identity = None
            for seed in SEEDS:
                frames, summaries = {}, {}
                for variant, root in roots.items():
                    frame, row, summary = inspect_stage(root / f'seed{seed}/outer_{context}/inner_{inner}', variant,
                                                        seed, context, inner, lookup[(seed, context, inner)], lock)
                    rows.append(row); frames[variant] = frame; summaries[variant] = summary
                    probabilities[variant].append(frame.probability_dd.to_numpy(float))
                    predictions[variant].append(frame.prediction.to_numpy(int))
                a, b = frames['baseline'], frames['lr_1e4']
                require(a[['subject_id', 'target']].equals(b[['subject_id', 'target']]), 'Methods not ID/label aligned')
                if seed_identity is None:
                    seed_identity = a[['subject_id', 'target']]
                else:
                    require(seed_identity.equals(a[['subject_id', 'target']]), 'Seeds use different validation subjects')
                for key in ('train_subject_ids_sha256', 'normalization_sha256'):
                    require(summaries['baseline'][key] == summaries['lr_1e4'][key], 'Matched training/normalization differs')
                y = a.target.to_numpy(int)
                old_error, new_error = a.prediction.to_numpy(int) != y, b.prediction.to_numpy(int) != y
                for label, name in [(0, 'PD'), (1, 'DD')]:
                    mask = y == label
                    corrected = int(np.sum(mask & old_error & ~new_error))
                    added = int(np.sum(mask & ~old_error & new_error))
                    errors.append(dict(seed=seed, context=context, inner=inner, disease=name, subject_count=int(mask.sum()),
                                       baseline_error_count=int(np.sum(mask & old_error)), candidate_error_count=int(np.sum(mask & new_error)),
                                       corrected_count=corrected, newly_wrong_count=added, net_corrected_count=corrected-added,
                                       prediction_disagreement=float(np.mean((old_error != new_error)[mask]))))
            for variant in roots:
                p, pred = np.stack(probabilities[variant]), np.stack(predictions[variant])
                agreement.append(dict(variant=variant, context=context, inner=inner,
                                      all_three_seed_agreement=float(np.mean(np.all(pred == pred[0], axis=0))),
                                      mean_pair_prediction_agreement=float(np.mean([np.mean(pred[a] == pred[b]) for a,b in [(0,1),(0,2),(1,2)]])),
                                      mean_pair_score_spearman=float(np.mean([spearmanr(p[a], p[b]).statistic for a,b in [(0,1),(0,2),(1,2)]]))))
    raw = pd.DataFrame(rows)
    require(len(raw) == 90 and not raw.duplicated(['variant','seed','context','inner']).any(), 'Incomplete/duplicate matrix')
    fold = raw.groupby(['variant','context','inner'], as_index=False)[METRICS].mean()
    summary_rows = []
    for variant, g in fold.groupby('variant'):
        r = raw[raw.variant == variant]
        seed_means = r.groupby('seed')[METRICS].mean()
        within = r.groupby(['context','inner'])[METRICS].std(ddof=1).mean()
        summary_rows.append(dict(variant=variant, **{m:g[m].mean() for m in METRICS},
                                 **{m+'_split_sd':g[m].std(ddof=1) for m in METRICS},
                                 **{m+'_seed_mean_sd':seed_means[m].std(ddof=1) for m in METRICS},
                                 **{m+'_within_split_seed_sd':within[m] for m in METRICS},
                                 best_epoch_median=r.best_epoch.median(), stop_epoch_median=r.stop_epoch.median(),
                                 epoch_cap_hits=int(r.epoch_cap_hit.sum()), train_online_loss=r.train_online_loss.mean(),
                                 validation_loss=r.validation_loss.mean(), trainable_parameter_count=143172,
                                 best_train_online_classification_loss=r.best_train_online_classification_loss.mean(),
                                 best_validation_classification_loss=r.best_validation_classification_loss.mean(),
                                 last_train_online_loss=r.last_train_online_loss.mean(),
                                 last_validation_loss=r.last_validation_loss.mean(),
                                 last_train_online_classification_loss=r.last_train_online_classification_loss.mean(),
                                 last_validation_classification_loss=r.last_validation_classification_loss.mean(),
                                 frozen_parameter_count=10457408, total_parameter_count=10600580))
    summary = pd.DataFrame(summary_rows).set_index('variant')
    bootstrap = np.random.default_rng(20261004).integers(0,15,size=(10000,15))
    paired = []
    for metric in METRICS:
        a = fold[fold.variant == 'lr_1e4'].set_index(['context','inner'])[metric].sort_index()
        b = fold[fold.variant == 'baseline'].set_index(['context','inner'])[metric].sort_index()
        require(len(a) == len(b) == 15 and a.index.equals(b.index), 'Incomplete paired split matrix')
        d = (a-b).to_numpy(); low, high = np.quantile(d[bootstrap].mean(axis=1), [.025,.975]); sd = d.std(ddof=1)
        paired.append(dict(metric=metric, mean_delta=d.mean(), improved_splits=int(np.sum(d>0)),
                           tied_splits=int(np.sum(d==0)), worse_splits=int(np.sum(d<0)), ci95_low=low, ci95_high=high,
                           cohen_dz=d.mean()/sd if sd else (0. if not np.any(d) else None),
                           wilcoxon_p=float(wilcoxon(d).pvalue) if np.any(d) else 1.))
    paired = pd.DataFrame(paired).set_index('metric')
    paired['bh_q_all6_exploratory'] = bh(paired.wilcoxon_p.to_numpy())
    paired['bh_q_primary'] = np.nan
    paired.loc[['ba','auroc'],'bh_q_primary'] = bh(paired.loc[['ba','auroc'],'wilcoxon_p'].to_numpy())
    b, c = summary.loc['baseline'], summary.loc['lr_1e4']
    gate = dict(ba_mean_positive=bool(paired.loc['ba','mean_delta']>0),
                ba_improved_10_of_15=bool(paired.loc['ba','improved_splits']>=10),
                ba_ci_low_positive=bool(paired.loc['ba','ci95_low']>0),
                auroc_mean_positive=bool(paired.loc['auroc','mean_delta']>0),
                auroc_improved_10_of_15=bool(paired.loc['auroc','improved_splits']>=10),
                macro_f1_non_decrease=bool(c.macro_f1>=b.macro_f1),
                pd_recall_drop_at_most_001=bool(c.pd_recall>=b.pd_recall-.01),
                dd_recall_drop_at_most_001=bool(c.dd_recall>=b.dd_recall-.01))
    for metric in ['ba','auroc']:
        gate[metric+'_within_split_seed_sd_at_most_125x'] = bool(c[metric+'_within_split_seed_sd'] <= 1.25*max(b[metric+'_within_split_seed_sd'],.001))
    error = pd.DataFrame(errors)
    numeric = ['subject_count','baseline_error_count','candidate_error_count','corrected_count','newly_wrong_count','net_corrected_count','prediction_disagreement']
    error_fold = error.groupby(['context','inner','disease'],as_index=False)[numeric].mean()
    result = dict(status='complete', candidate='lr_1e4', required_candidate_runs=45, completed_candidate_runs=45,
                  primary_statistical_units=15, seeds_averaged_before_split_comparison=True,
                  overlapping_splits='Descriptive repeated-development robustness, not independent external inference',
                  gate=gate, decision='RETAIN' if all(gate.values()) else 'REJECT',
                  retained_learning_rate=1e-4 if all(gate.values()) else 2e-4,
                  outer_access=False, no_architecture_change=True, no_other_candidate=True,
                  training_loss_scope='Online dropout-active; not eval-mode train performance',
                  protocol_sha256=lock['files']['LR_CONTROL_PROTOCOL.md'], analysis_sha256=sha(__file__))
    out = HERE/'analysis'
    raw.to_csv(out/'lr_seed_split_metrics.csv',index=False)
    fold.to_csv(out/'lr_15split_seedfirst.csv',index=False)
    summary.reset_index().to_csv(out/'lr_summary.csv',index=False)
    paired.reset_index().to_csv(out/'lr_paired.csv',index=False)
    pd.DataFrame(agreement).to_csv(out/'lr_prediction_agreement.csv',index=False)
    error.to_csv(out/'lr_errors_45runs.csv',index=False)
    error_fold.to_csv(out/'lr_errors_15split_seedfirst.csv',index=False)
    error_fold.groupby('disease',as_index=False)[numeric].mean().to_csv(out/'lr_error_summary.csv',index=False)
    (out/'lr_decision.json').write_text(json.dumps(result,indent=2)+'\n')
    print(summary[METRICS].round(6).to_string());print(paired.round(6).to_string());print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
