"""Exactly one registered condition/seed through all15 original inner-training units."""
import argparse
import copy
import gc
import json
import pandas as pd
import numpy as np
import torch
from common import *
from prior_model import build_prior
from src.engine import nested_training as nt
from src.metrics.classification import classification_metrics
from src.utils.device import select_device


def study_guard(lock):
    for rel,digest in lock['files'].items():require(sha(HERE/rel)==digest,f'Frozen study file changed: {rel}')


def audit_stage(stage,condition,seed,c,i,inner,f0lock,studylock,smoke):
    base=reference(seed,c,i)
    old=torch.load(base/'checkpoints/best.pt',map_location='cpu',weights_only=False)
    ck=torch.load(stage/'checkpoints/best.pt',map_location='cpu',weights_only=False)
    status=json.loads((stage/'stage_status.json').read_text())
    require(status['status']=='complete' and status['phase']=='inner' and
            status['outer_test_loader_created'] is False,'Not completed restricted inner run')
    actual=ck['config'];baseline=old['config']
    for key in ['data','training','evaluation','loss','nested_cv','self_supervised']:
        require(actual.get(key)==baseline.get(key),f'Frozen recipe differs: {key}')
    m=copy.deepcopy(actual['model']);metadata=m.pop('frequency_prior_control')
    require(m==baseline['model'],'Original STR architecture configuration changed')
    require(metadata['condition']==condition and metadata['embedding_dim']==16 and metadata['zero_init'], 'Condition metadata differs')
    require(ck['class_names']==['PD','DD'] and actual['experiment']['seed']==seed,'Seed/class order differs')
    require(sum(v.numel() for v in ck['model_state'].values())=={'F1':78005,'F2':77998}[condition],'Parameter count differs')
    require(all(torch.equal(ck['normalization'][k],old['normalization'][k]) for k in ['mean','std']), 'Normalization differs from F0')
    split=json.loads((stage/'split.json').read_text())
    require(not split['test_subjects'] and set(split['train_subjects'])==set(inner['train_subjects']) and
            set(split['validation_subjects'])==set(inner['validation_subjects']),'Partition/outer boundary differs')
    require(not set(split['train_subjects']) & set(split['validation_subjects']),'Train/validation overlap')
    history=[json.loads(l) for l in (stage/'logs/epochs.jsonl').read_text().splitlines()]
    require([r['epoch'] for r in history]==list(range(len(history))) and 1<=len(history)<= (1 if smoke else 50),'Invalid epoch history')
    best=float('-inf');selected=0;patience=0
    for r in history:
        v=r['validation']['balanced_accuracy'];improved=v>best
        require(r['selection_metric']=='balanced_accuracy' and r['improved']==improved,'Selection changed')
        if improved:best=v;selected=r['epoch']+1;patience=0
        else:patience+=1
        require(r['best_epoch']==selected and r['patience_count']==patience,'BA-best/patience changed')
    require(smoke or len(history)==50 or patience==12,'Stopping outside original rule')
    require(ck['epoch']+1==selected==status['summary']['best_epoch'],'Checkpoint selected epoch differs')
    last=torch.load(stage/'checkpoints/last.pt',map_location='cpu',weights_only=False)
    require(last['epoch']==len(history)-1,'Last epoch differs')
    frame=pd.read_csv(stage/'predictions/validation.csv',dtype={'subject_id':str})
    ref=pd.read_csv(base/'predictions/validation.csv',dtype={'subject_id':str}).set_index('subject_id')
    require(not frame.subject_id.duplicated().any() and set(frame.subject_id)<=set(inner['validation_subjects']),'Invalid prediction IDs')
    require(smoke or set(frame.subject_id)==set(inner['validation_subjects']),'Incomplete validation')
    require(frame.target.tolist()==ref.loc[frame.subject_id].target.tolist(),'Label alignment differs')
    prob=frame[['probability_pd','probability_dd']].to_numpy()
    require(np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=1e-6),'Invalid probabilities')
    require(np.array_equal(prob.argmax(1),frame.prediction.to_numpy()),'Threshold rule changed')
    metric=classification_metrics(frame.target.to_numpy(),frame.prediction.to_numpy(),2,probabilities=prob)
    for k in ['accuracy','balanced_accuracy','macro_auroc','macro_f1','per_class_recall']:
        saved=status['summary']['validation_metrics'][k]
        require(metric[k] is None and saved is None if metric[k] is None or saved is None else
                np.allclose(metric[k],saved,atol=1e-12,rtol=0), 'Saved metrics differ')
    original_branch = {'wrist_encoder.branch.residual.weight','wrist_encoder.branch.residual.bias'}
    require(any(ck['model_state'][k].abs().sum()>0 for k in original_branch),'Residual remained zero after training')
    require(not list((stage/'checkpoints').glob('ema*')),'EMA unexpectedly present')
    audit=dict(status='PASS',condition=condition,seed=seed,context=c,inner=i,smoke=smoke,
         total_parameters={'F1':78005,'F2':77998}[condition],added_parameters={'F1':2481,'F2':2474}[condition],
         best_epoch=selected,stop_epoch=len(history),validation_subject_count=len(frame),
         normalization_exact=True,original_recipe_unchanged=True,threshold=.5,threshold_tie='PD',
         outer_test_loader_created=False,outer_performance_accessed=False,no_wssl=True,no_posthoc_tuning=True,
         runner_sha256=sha(__file__),protocol_sha256=studylock['files']['PROTOCOL.md'],
         files_sha256={r:sha(stage/r) for r in ['checkpoints/best.pt','checkpoints/last.pt','stage_status.json',
                      'config.yaml','split.json','normalization.json','logs/epochs.jsonl','predictions/validation.csv']})
    return audit


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--condition',choices=['F1','F2'],required=True)
    parser.add_argument('--seed',type=int,choices=[42,43,44],required=True);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();torch.set_num_threads(4)
    f0lock=json.loads((HERE/'analysis/f0_lock.json').read_text())
    studylock=json.loads((HERE/'analysis/study_lock.json').read_text());guard(f0lock);study_guard(studylock)
    for name in ['f0_audit','filter_bank_tests','model_tests']:
        require(json.loads((HERE/'analysis'/f'{name}.json').read_text())['status']=='PASS',f'Entry gate missing: {name}')
    if not args.smoke:
        for condition in ['F1','F2']:
            audit=json.loads((HERE/f'smoke/{condition}/seed42/outer_0/inner_0/prior_audit.json').read_text())
            require(audit['status']=='PASS' and audit['runner_sha256']==sha(__file__) and
                    audit['protocol_sha256']==studylock['files']['PROTOCOL.md'],'Current smoke gate missing')
        release=json.loads((HERE/'analysis/preformal_release.json').read_text())
        require(release['protocol_and_source_pushed_before_formal_results'] and
                release['study_lock_sha256']==sha(HERE/'analysis/study_lock.json'),'Preformal Git gate missing')
    cfg=config_for(args.seed)
    cfg['experiment']['name']=f'frequency_prior_{args.condition}_seed{args.seed}'
    cfg['experiment']['output_root']=str(HERE/('smoke' if args.smoke else 'runs')/args.condition/f'seed{args.seed}')
    cfg['model']['frequency_prior_control']=dict(condition=args.condition,embedding_dim=16,zero_init=True,
        bands=[[.5,3],[3,4],[4,6],[6,8],[8,12]] if args.condition=='F2' else None,
        source='current_normalized_acc_gyro' if args.condition=='F2' else 'original_wrist_embedding',
        boundary='even_reflection_2L_halfopen_lastclosed' if args.condition=='F2' else None)
    cfg['development']=dict(protocol_sha256=studylock['files']['PROTOCOL.md'],condition=args.condition,
        random_initialization=True,trained_STR_warmstart=False,no_outer=True,no_wssl=True,no_recipe_change=True)
    original={n:getattr(nt,n) for n in ['build_model','build_subject_fold_datasets','_loader']};loaders=[]
    nt.build_model=lambda config:build_prior(config,args.condition)
    nt.build_subject_fold_datasets=development_bundle
    def tracked_loader(*a,**kw):
        loader=original['_loader'](*a,**kw)
        if loader is not None:loaders.append(loader)
        return loader
    nt._loader=tracked_loader
    def cleanup():
        for loader in loaders:
            iterator=getattr(loader,'_iterator',None)
            if iterator is not None:iterator._shutdown_workers();loader._iterator=None
        loaders.clear();gc.collect()
    completed=0;device=select_device('cuda');split=json.loads(SPLIT.read_text())
    try:
        for outer in split['outer']:
            c=int(outer['outer_fold'])
            for inner in outer['inner_folds']:
                i=int(inner['inner_fold'])
                if args.smoke and (c,i)!=(0,0):continue
                stage=Path(cfg['experiment']['output_root'])/f'outer_{c}/inner_{i}'
                require(not stage.exists(),f'Existing stage must be preserved, not resumed/overwritten: {stage}')
                guard(f0lock);study_guard(studylock)
                try:
                    nt._run_inner_fold(cfg,{'outer_fold':c,'test_subjects':[]},inner,stage,device,resume=False,smoke=args.smoke)
                    audit=audit_stage(stage,args.condition,args.seed,c,i,inner,f0lock,studylock,args.smoke)
                    guard(f0lock);study_guard(studylock);write_json(stage/'prior_audit.json',audit)
                    completed+=1
                    print(json.dumps(dict(status='PASS',condition=args.condition,seed=args.seed,context=c,inner=i,
                                          smoke=args.smoke,completed_splits=completed)),flush=True)
                finally:cleanup()
        require(completed==(1 if args.smoke else 15),'Incomplete condition/seed matrix')
    finally:
        cleanup()
        for name,func in original.items():setattr(nt,name,func)
    print(json.dumps(dict(status='COMPLETE',condition=args.condition,seed=args.seed,units=completed,smoke=args.smoke)),flush=True)


if __name__=='__main__':main()
