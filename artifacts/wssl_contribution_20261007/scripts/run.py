"""Full45 reproduction first, smoke second, then inference-only residual removal."""
import argparse
import gc
import io
from datetime import datetime,timezone
from common import *


def iter_splits(lock):
    split=json.loads(Path(lock['split_path']).read_text())
    for outer in split['outer']:
        for inner in outer['inner_folds']:
            yield int(outer['outer_fold']),int(inner['inner_fold']),inner


def validate_config(cfg, base):
    require(cfg['data']==base['data'] and cfg['model']==base['model'], 'Frozen model/input config changed')
    t=cfg['training']
    require(t['optimizer']=='adamw' and t['learning_rate']==2e-4 and t['weight_decay']==1e-4 and t['batch_size']==8, 'Recipe mismatch')
    require(t['epochs']==50 and not t['mixed_precision'] and t['early_stopping']['patience']==12, 'Stopping/FP32 mismatch')
    require(cfg['data']['activities']==ACTIVITIES and cfg['data']['wrist_order']==['left','right'], 'Activity/wrist order')
    require(cfg['loss']['label_smoothing']==0 and cfg['loss']['class_weight_source']=='current_fold_train_subject_labels_only', 'Loss provenance')
    require(not cfg['evaluation']['run_test_after_training'], 'Outer flag')


def load_model(seed,c,i):
    ck=torch.load(stage_path(seed,c,i)/'checkpoints/best.pt',map_location='cpu',weights_only=False)
    model=IndependentWSSL(ck['config']).cuda().eval().requires_grad_(False)
    model.load_state_dict(ck['model_state'],strict=True)
    require(sum(p.numel() for p in model.parameters())==143172, 'WSSL parameter count')
    require(not dict(model.named_buffers()), 'Unexpected buffers')
    return ck,model


def full():
    require(not (HERE/'analysis/full_reproduction.json').exists(), 'Full audit already exists')
    lock=prepare_lock() if not (HERE/'analysis/assets_lock.json').exists() else guard()
    torch.set_num_threads(4);seed_everything(42,True)
    base=load_config(str(FOUNDATION/'configs/str01_seed42.yaml'))
    rows=[];normrows=[];allparts=[]
    for c,i,inner in iter_splits(lock):
        train=set(map(str,inner['train_subjects']));val=set(map(str,inner['validation_subjects']))
        require(train and val and not(train&val), 'Train/validation overlap')
        bundle=development_bundle(base,train_subject_ids=sorted(train),validation_subject_ids=sorted(val),test_subject_ids=[],fold_id=f'outer_{c}/inner_{i}')
        require(bundle.test is None and bundle.split_summary['normalization_fitted_on']=='train_subjects_only', 'Normalization/test boundary')
        # The loader is validation only. Train signals are used solely for the original normalization refit.
        batches=[collate_subject_activities([bundle.validation[j] for j in range(k,min(k+8,len(bundle.validation)))])
                 for k in range(0,len(bundle.validation),8)]
        require(set(s for batch in batches for s in batch['subject_id'])==val, 'Validation dataset IDs')
        cache=ValidationCache(lock['cache_path'],val)
        for seed in [42,43,44]:
            seed_everything(seed,True)
            ck,model=load_model(seed,c,i);validate_config(ck['config'],base)
            require(torch.equal(bundle.mean,ck['normalization']['mean']) and torch.equal(bundle.std,ck['normalization']['std']), 'Train-only normalization not exact')
            archived=predictions(stage_path(seed,c,i)/'predictions/validation.csv')
            strpred=predictions(str_path(seed,c,i)/'predictions/validation.csv')
            require(set(archived.index)==val and archived.index.equals(strpred.index), 'STR/WSSL validation alignment')
            require(np.array_equal(archived.target.to_numpy(),strpred.target.to_numpy()), 'STR/WSSL labels')
            legacy=ResidualSSLSubject(build_model(ck['config'])).cuda().eval().requires_grad_(False)
            legacy.load_state_dict(ck['model_state'],strict=True)
            before=state_fingerprint(model);parts=[];result=[];max_error=0.
            with torch.no_grad():
                for batch in batches:
                    ids=list(map(str,batch['subject_id']));target=batch['y'].numpy()
                    require(np.array_equal(target,archived.loc[ids].target.to_numpy()), 'Batch ID/label alignment')
                    inputs=[batch[k].cuda() for k in ['x','wrist_mask','activity_mask','activity_lengths']]
                    ssl=cache.batch(ids)
                    original=model(*inputs,ssl_features=ssl)['logits'];legacy.ssl_features=ssl
                    old=legacy(*inputs)['logits']
                    require(torch.equal(original,old), 'Historical/independent full logits differ')
                    wrist,added=encode(model,inputs,ssl)
                    fast=from_parts(model,wrist,added,inputs[1],inputs[2],torch.ones(11,2))
                    require(torch.equal(fast,original), 'Reused-part full logits differ')
                    probability=original.softmax(-1).cpu().numpy()
                    expected=archived.loc[ids,['probability_pd','probability_dd']].to_numpy()
                    err=float(np.abs(probability-expected).max());max_error=max(max_error,err)
                    require(err<=1e-12, 'Archived full probabilities differ')
                    require(np.array_equal(probability.argmax(1),archived.loc[ids].prediction.to_numpy()), 'Archived decisions differ')
                    parts.append(dict(ids=ids,target=batch['y'].clone(),wrist=wrist.cpu(),added=added.cpu(),
                                      wm=batch['wrist_mask'].clone(),am=batch['activity_mask'].clone(),
                                      full_logits=original.cpu(),full_probability=torch.from_numpy(probability.copy())))
                    for j,sid in enumerate(ids):
                        result.append(dict(subject_id=sid,target=int(target[j]),probability_pd=float(probability[j,0]),probability_dd=float(probability[j,1])))
            re=pd.DataFrame(result).set_index('subject_id').sort_index()
            measured=metrics(re.target.to_numpy(),re[['probability_pd','probability_dd']].to_numpy())
            reference=metrics(archived.target.to_numpy(),archived[['probability_pd','probability_dd']].to_numpy())
            for m in METRICS:require(abs(measured[m]-reference[m])<=1e-12, 'Archived metric differs '+m)
            status=json.loads((stage_path(seed,c,i)/'stage_status.json').read_text())
            mapping={'accuracy':'accuracy','ba':'balanced_accuracy','auroc':'macro_auroc','macro_f1':'macro_f1','pd_recall':'pd_recall','dd_recall':'dd_recall'}
            for m,k in mapping.items():require(abs(measured[m]-status['summary']['validation_metrics'][k])<=1e-12, 'Formal summary metric differs')
            require(before==state_fingerprint(model), 'Frozen state modified')
            require(cache.fingerprint==tensor_sha(cache.values), 'Cache modified')
            path=HERE/f'features/seed{seed}/outer_{c}/inner_{i}/parts.pt'
            require(not path.exists(),'Do not overwrite encoded parts');path.parent.mkdir(parents=True,exist_ok=True)
            torch.save(dict(seed=seed,context=c,inner=i,batches=parts,checkpoint_sha256=sha(stage_path(seed,c,i)/'checkpoints/best.pt'),
                            normalization_sha256=status['summary']['normalization_sha256'],validation_cache_tensor_sha256=cache.fingerprint),path)
            allparts.append(dict(seed=seed,context=c,inner=i,path=str(path),sha256=sha(path)))
            rows.append(dict(seed=seed,context=c,inner=i,subject_count=len(val),cache_ID_aligned=True,
                normalization_exact=True,legacy_independent_fast_logits_exact=True,archive_probability_max_error=max_error,
                archive_decisions_exact=True,archive_metrics_exact=True,state_unchanged=True,**measured))
            print('FULL REPRODUCTION PASS',seed,c,i,'n=',len(val),'archive maxerr=',max_error,flush=True)
            del model,legacy,ck,parts;gc.collect();torch.cuda.empty_cache()
        normrows.append(dict(context=c,inner=i,train_count=len(train),validation_count=len(val),mean_max_difference=0.,std_max_difference=0.,checkpoints_matched=3))
        del bundle,batches,cache;gc.collect()
    guard();require(len(rows)==45 and len(normrows)==15,'Incomplete reproduction')
    pd.DataFrame(rows).to_csv(HERE/'analysis/full_reproduction_45runs.csv',index=False)
    pd.DataFrame(normrows).to_csv(HERE/'analysis/normalization_15refits.csv',index=False)
    write_json(HERE/'features/parts_manifest.json',allparts)
    write_json(HERE/'analysis/full_reproduction.json',dict(status='PASS',checkpoints=45,normalization_refits=15,
        probability_max_difference=max(r['archive_probability_max_error'] for r in rows),all_probabilities_decisions_metrics_match=True,
        full_legacy_independent_reused_logits_bit_exact=True,historical_logits_not_archived=True,
        formal_logits_validation='Historic forward equivalence; no claim of unavailable archived logits',
        training=False,outer_access=False,parameter_count=143172,normalization_train_only=True,
        validation_only_cache_rows_used=True,parts_manifest_sha256=sha(HERE/'features/parts_manifest.json')))
    print('FULL45 REPRODUCTION GATE PASS',flush=True)


def smoke():
    lock=guard();require(json.loads((HERE/'analysis/full_reproduction.json').read_text())['status']=='PASS','Full gate required')
    require(not (HERE/'analysis/smoke_test.json').exists(), 'Do not overwrite smoke')
    torch.set_num_threads(4);seed_everything(42,True)
    ck,model=load_model(42,0,0)
    p=predictions(stage_path(42,0,0)/'predictions/validation.csv');ids=list(p.index)[:8]
    records=fd.load_configured_records(ck['config']['data'],['PD','DD'])
    dataset=SubjectActivityDataset([r for r in records if r.subject_id in set(ids)],ACTIVITIES,
       ck['config']['data'],ck['normalization']['mean'],ck['normalization']['std'],'validation')
    batch=collate_subject_activities([dataset[j] for j in range(len(dataset))])
    cache=ValidationCache(lock['cache_path'],ids);ssl=cache.batch(batch['subject_id'])
    inputs=[batch[k].cuda() for k in ['x','wrist_mask','activity_mask','activity_lengths']]
    before=state_fingerprint(model);rows=[]
    with torch.no_grad():
        wrist,added=encode(model,inputs,ssl)
        for name,kind,a,w,mask in conditions():
            scale=torch.from_numpy(mask).cuda()
            handle=model.wrist_projection.register_forward_hook(lambda module,args,output:output*scale[None,...,None])
            try:direct=model(*inputs,ssl_features=ssl)['logits']
            finally:handle.remove()
            fast=from_parts(model,wrist,added,inputs[1],inputs[2],scale)
            require(torch.equal(fast,direct),'Mask not exact hook parity '+name)
            rows.append(dict(condition=name,residual_output_hook_logits_exact=True,original_STR_wrist_unchanged=True))
        # Zero input is intentionally different from residual-off because trained LN/projection bias survives.
        actual_off=from_parts(model,wrist,added,inputs[1],inputs[2],torch.zeros(11,2))
        zero_ssl=model(*inputs,ssl_features=torch.zeros_like(ssl))['logits']
        zero_input_difference=float((zero_ssl-actual_off).abs().max())
        order=torch.arange(7,-1,-1,device='cuda')
        rev=model(*(v.index_select(0,order) for v in inputs),ssl_features=cache.batch(list(reversed(batch['subject_id']))))['logits']
        full=model(*inputs,ssl_features=ssl)['logits'];require(float((rev-full.index_select(0,order)).abs().max())<=1e-5,'ID reorder parity')
        memory=io.BytesIO();torch.save(model.state_dict(),memory);memory.seek(0)
        fresh=IndependentWSSL(ck['config']).cuda().eval().requires_grad_(False)
        fresh.load_state_dict(torch.load(memory,map_location='cuda',weights_only=True),strict=True)
        require(torch.equal(full,fresh(*inputs,ssl_features=ssl)['logits']),'Reload differs')
    require(before==state_fingerprint(model) and cache.fingerprint==tensor_sha(cache.values),'State/cache changed')
    pd.DataFrame(rows).to_csv(HERE/'analysis/smoke_conditions.csv',index=False)
    write_json(HERE/'analysis/smoke_test.json',dict(status='PASS',conditions=37,hook_vs_reused_parts_exact=True,
          projection_output_including_bias_disabled=True,zero_SSL_input_not_equivalent_max_logit_difference=zero_input_difference,
          original_STR_parameters_changed=False,cache_changed=False,reload_exact=True,subject_ID_reorder_pass=True,
          training=False,optimizer_steps=0,outer_access=False,scope='First frozen checkpoint, eight existing validation subjects; smoke performance not selected'))
    guard();print('SMOKE37 PASS; no optimizer/backward/training',flush=True)


def masked():
    lock=guard();require((HERE/'analysis/diagnostic_lock.json').exists(),'Freeze required')
    require(json.loads((HERE/'analysis/full_reproduction.json').read_text())['status']=='PASS','Full gate required')
    require(json.loads((HERE/'analysis/smoke_test.json').read_text())['status']=='PASS','Smoke gate required')
    require(not (HERE/'analysis/execution_state.json').exists(),'Existing inference state: inspect before rerun')
    torch.set_num_threads(4);seed_everything(42,True)
    parts_manifest=json.loads((HERE/'features/parts_manifest.json').read_text())
    require(sha(HERE/'features/parts_manifest.json')==json.loads((HERE/'analysis/full_reproduction.json').read_text())['parts_manifest_sha256'],'Parts manifest changed')
    rows=[];prediction_files=[];started=datetime.now(timezone.utc).isoformat()
    write_json(HERE/'analysis/execution_state.json',dict(status='running',completed_checkpoints=0,started_utc=started,training=False))
    for unit in parts_manifest:
        seed,c,i=unit['seed'],unit['context'],unit['inner']
        require(sha(unit['path'])==unit['sha256'],'Full encoded part changed')
        ck,model=load_model(seed,c,i);before=state_fingerprint(model)
        data=torch.load(unit['path'],map_location='cpu',weights_only=False);pack={name:[] for name,*_ in conditions()}
        for batch in data['batches']:
            wrist=batch['wrist'].cuda();added=batch['added'].cuda();wm=batch['wm'].cuda();am=batch['am'].cuda()
            for name,kind,a,w,mask in conditions():
                logits=from_parts(model,wrist,added,wm,am,torch.from_numpy(mask))
                if name=='full':require(torch.equal(logits.cpu(),batch['full_logits']),'Formal full part changed')
                probability=logits.softmax(-1).cpu().numpy()
                for j,sid in enumerate(batch['ids']):
                    pack[name].append(dict(subject_id=sid,target=int(batch['target'][j]),
                       p_pd=float(probability[j,0]),p_dd=float(probability[j,1]),prediction=int(probability[j].argmax()),
                       logit_pd=float(logits[j,0]),logit_dd=float(logits[j,1])))
        formal=predictions(stage_path(seed,c,i)/'predictions/validation.csv');full_metric=metrics(formal.target,formal[['probability_pd','probability_dd']])
        frames=[]
        for name,kind,a,w,mask in conditions():
            frame=pd.DataFrame(pack[name]).set_index('subject_id').sort_index()
            require(frame.index.equals(formal.index) and np.array_equal(frame.target,formal.target),'Counterfactual ID/label mismatch')
            measured=metrics(frame.target,frame[['p_pd','p_dd']])
            frames.append(frame.reset_index().assign(seed=seed,context=c,inner=i,condition=name,kind=kind,activity=a,wrist=w,
                  p_full=formal.probability_dd.to_numpy(),prediction_full=formal.prediction.to_numpy(),
                  delta_probability=formal.probability_dd.to_numpy()-frame.p_dd.to_numpy()))
            row=dict(seed=seed,context=c,inner=i,condition=name,kind=kind,activity=a,wrist=w,subject_count=len(frame))
            for m in METRICS:row.update({m+'_full':full_metric[m],m+'_masked':measured[m],m+'_delta':full_metric[m]-measured[m]})
            rows.append(row)
        dest=HERE/f'predictions/seed{seed}/outer_{c}/inner_{i}/counterfactual.csv'
        require(not dest.exists(),'Do not overwrite private predictions');dest.parent.mkdir(parents=True,exist_ok=True)
        pd.concat(frames,ignore_index=True).to_csv(dest,index=False)
        prediction_files.append(dict(seed=seed,context=c,inner=i,path=str(dest),sha256=sha(dest)))
        require(before==state_fingerprint(model),'Checkpoint state changed by inference')
        write_json(HERE/'analysis/execution_state.json',dict(status='running',completed_checkpoints=len(prediction_files),started_utc=started,training=False))
        print('COUNTERFACTUAL CHECKPOINT COMPLETE',seed,c,i,'conditions37',flush=True)
        del model,ck,data,pack,frames;gc.collect();torch.cuda.empty_cache()
    require(len(prediction_files)==45 and len(rows)==45*37,'Incomplete counterfactual matrix')
    pd.DataFrame(rows).to_csv(HERE/'analysis/metrics_45runs.csv',index=False)
    write_json(HERE/'predictions/manifest.json',prediction_files)
    guard()
    write_json(HERE/'analysis/execution_state.json',dict(status='complete',completed_checkpoints=45,conditions=37,
        diagnostic_masked_conditions=36,user_requested_masked_conditions=35,metrics_rows=len(rows),started_utc=started,
        completed_utc=datetime.now(timezone.utc).isoformat(),training=False,optimizer_steps=0,outer_access=False,
        checkpoint_state_unchanged_all45=True,private_prediction_manifest_sha256=sha(HERE/'predictions/manifest.json')))
    print('FULL COUNTERFACTUAL MATRIX COMPLETE',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['full','smoke','freeze','masked']);arg=ap.parse_args()
    if arg.stage=='freeze':guard();freeze_diagnostics();print('PRE-MASK DIAGNOSTIC CODE/RULES FROZEN')
    else:globals()[arg.stage]()
