"""Recheck all frozen STR predictions, 15 train-only norm fits, and prior exact reproduction."""
import gc
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from common import *
from src.models import build_model
from src.datasets.subject_activity import collate_subject_activities
from src.metrics.classification import classification_metrics
from src.engine import nested_training as nt
from src.utils.provenance import source_tree_manifest


def main():
    torch.set_num_threads(4)
    cfg = config_for(42)
    _, current_source = source_tree_manifest(FOUNDATION)
    split = json.loads(SPLIT.read_text())
    require(not (HERE/'analysis/f0_lock.json').exists(),'Do not overwrite completed baseline audit')
    paths = list((FOUNDATION / 'src').rglob('*.py')) + list((FOUNDATION / 'configs').glob('*.yaml'))
    paths += [SPLIT]
    # Processed input contract files, without reading any outer outcomes.
    processed = Path(cfg['data']['root'])
    for name in ['processing_config.json', 'audit.json', 'quality_summary.json', 'solver_audit.json']:
        if (processed / name).exists(): paths.append(processed / name)
    paths += [Path(p) for p in cfg['data']['activity_manifests'].values()]
    lock = dict(source_sha256={str(p): sha(p) for p in paths}, stages=[], split_sha256=SPLIT_SHA,
                outer_access=False, prior_exact_reproduction=True)
    rows = []
    norm_rows = []
    for outer in split['outer']:
        c = int(outer['outer_fold'])
        for inner in outer['inner_folds']:
            i = int(inner['inner_fold'])
            require(not set(inner['train_subjects']) & set(inner['validation_subjects']), 'Train/val overlap')
            bundle = development_bundle(cfg, train_subject_ids=inner['train_subjects'],
                       validation_subject_ids=inner['validation_subjects'], fold_id=f'outer_{c}/inner_{i}')
            actual_cfg = nt._resolve_train_only_class_weights(cfg, bundle)
            loader = DataLoader(bundle.validation, batch_size=8, shuffle=False, num_workers=0,
                                collate_fn=collate_subject_activities)
            for seed in [42, 43, 44]:
                stage = reference(seed, c, i)
                ck = torch.load(stage/'checkpoints/best.pt', map_location='cpu', weights_only=False)
                status = json.loads((stage/'stage_status.json').read_text())
                require(status['status'] == 'complete' and not status['outer_test_loader_created'], 'Invalid reference status')
                require(status['summary']['checkpoint_sha256'] == sha(stage/'checkpoints/best.pt'), 'Formal checkpoint SHA mismatch')
                frozen_split = json.loads((stage/'split.json').read_text())
                require(set(frozen_split['train_subjects']) == set(inner['train_subjects']) and
                        set(frozen_split['validation_subjects']) == set(inner['validation_subjects']), 'Reference split mismatch')
                require(ck['class_names'] == ['PD','DD'], 'Class direction changed')
                current = config_for(seed)
                for key in ['model','data','training','evaluation','nested_cv','self_supervised']:
                    require(current.get(key) == ck['config'].get(key), f'Resolved frozen config differs: {key}')
                require(actual_cfg['loss'] == ck['config']['loss'], 'Train-only class weights differ')
                require(all(torch.equal(getattr(bundle,k),ck['normalization'][k]) for k in ['mean','std']), 'Normalization differs')
                model = build_model(ck['config']).to('cuda').eval()
                require(sum(p.numel() for p in model.parameters()) == 75524, 'STR parameter count changed')
                model.load_state_dict(ck['model_state'], strict=True)
                # Validate phase-2 full retraining evidence directly, not only its report.
                reproduced = ROOT/f'artifacts/phase2_str_training_20260926/runs/baseline/seed{seed}/outer_{c}/inner_{i}'
                re_ck = torch.load(reproduced/'checkpoints/best.pt',map_location='cpu',weights_only=False)
                require(re_ck['provenance']['source_tree_sha256']==current_source,'Matched Phase2 P0 source differs from current source')
                require(ck['model_state'].keys() == re_ck['model_state'].keys() and
                        all(torch.equal(v,re_ck['model_state'][k]) for k,v in ck['model_state'].items()), 'Prior STR reproduction differs')
                require(ck['epoch'] == re_ck['epoch'], 'Prior selected epoch differs')
                frame = pd.read_csv(stage/'predictions/validation.csv',dtype={'subject_id':str}).set_index('subject_id')
                re_frame = pd.read_csv(reproduced/'predictions/validation.csv',dtype={'subject_id':str}).set_index('subject_id').loc[frame.index]
                require(not frame.index.duplicated().any() and set(frame.index) == set(inner['validation_subjects']), 'Incomplete reference IDs')
                require(np.array_equal(frame[['probability_pd','probability_dd']].to_numpy(), re_frame[['probability_pd','probability_dd']].to_numpy()), 'Prior predictions differ')
                all_prob, all_y, all_ids = [], [], []
                with torch.no_grad():
                    for batch in loader:
                        out = model(*(batch[k].to('cuda') for k in ['x','wrist_mask','activity_mask','activity_lengths']))
                        all_prob.append(out['probabilities'].cpu().numpy());all_y.extend(batch['y'].tolist());all_ids.extend(batch['subject_id'])
                probability = np.concatenate(all_prob)
                matched = frame.loc[all_ids]
                require(matched.target.tolist() == all_y, 'ID/label alignment differs')
                diff = float(np.max(np.abs(probability-matched[['probability_pd','probability_dd']].to_numpy())))
                require(diff <= 1e-12, 'Full validation forward differs from archived prediction')
                require(np.array_equal(probability.argmax(1), matched.prediction.to_numpy()), 'Reference decisions differ')
                metric = classification_metrics(matched.target.to_numpy(),matched.prediction.to_numpy(),2,
                            probabilities=matched[['probability_pd','probability_dd']].to_numpy())
                for key in ['accuracy','balanced_accuracy','macro_auroc','macro_f1','per_class_recall']:
                    require(np.allclose(metric[key],status['summary']['validation_metrics'][key],atol=1e-12,rtol=0), 'Metric implementation mismatch')
                rows.append(dict(seed=seed,context=c,inner=i,subject_count=len(frame),max_probability_difference=diff,
                    normalization_exact=True,prior_model_state_exact=True,best_epoch=ck['epoch']+1,
                    accuracy=metric['accuracy'],ba=metric['balanced_accuracy'],auroc=metric['macro_auroc'],
                    macro_f1=metric['macro_f1'],pd_recall=metric['per_class_recall'][0],dd_recall=metric['per_class_recall'][1]))
                lock['stages'].append(dict(seed=seed,context=c,inner=i,path=str(stage),
                    files_sha256={r:sha(stage/r) for r in ['checkpoints/best.pt','checkpoints/last.pt','predictions/validation.csv',
                        'stage_status.json','config.yaml','normalization.json','split.json','logs/epochs.jsonl']}))
                del model,ck,re_ck;gc.collect()
            norm_rows.append(dict(context=c,inner=i,normalization_exact_all_seeds=True,
                                 train_count=len(inner['train_subjects']),validation_count=len(inner['validation_subjects'])))
            print(json.dumps({'status':'PASS','context':c,'inner':i,'checkpoints_checked':3}),flush=True)
            del bundle,loader;gc.collect()
    guard(lock)
    pd.DataFrame(rows).to_csv(HERE/'analysis/f0_45run_audit.csv',index=False)
    pd.DataFrame(norm_rows).to_csv(HERE/'analysis/f0_15normalization_audit.csv',index=False)
    write_json(HERE/'analysis/f0_lock.json',lock)
    report=dict(status='PASS',checkpoints=45,full_validation_predictions_checked=True,train_only_normalization_refits=15,
        max_probability_difference=max(r['max_probability_difference'] for r in rows),prior_45_model_states_exact=True,
        current_source_sha_recorded=True,current_source_tree_sha256=current_source,matched_phase2_current_source_exact=True,original_historical_source_hash_not_inferred=True,outer_access=False,
        formal_baseline_retraining=False,means=pd.DataFrame(rows)[['accuracy','ba','auroc','macro_f1','pd_recall','dd_recall']].mean().to_dict())
    write_json(HERE/'analysis/f0_audit.json',report)
    print(json.dumps(report),flush=True)


if __name__ == '__main__':main()
