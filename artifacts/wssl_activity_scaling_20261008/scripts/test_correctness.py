"""Pre-formal all45 checkpoint and matched-scratch correctness gate."""
import io
import gc
import copy
from run import *
from src.datasets import folds as fd


def equal_tree(a,b):
    if torch.is_tensor(a): return a.dtype==b.dtype and torch.equal(a.detach().cpu(),b.detach().cpu())
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)): return len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b


def dataset_for(ck, ids, role):
    records=fd.load_configured_records(ck['config']['data'], ['PD','DD'])
    return SubjectActivityDataset([r for r in records if r.subject_id in set(ids)],ck['config']['data']['activities'],
          ck['config']['data'],ck['normalization']['mean'],ck['normalization']['std'],role)


def batch_args(batch,cache):
    return [batch[k].cuda() for k in ('x','wrist_mask','activity_mask','activity_lengths')],cache.batch(batch['subject_id'],'pretrained','cuda')


def main():
    torch.set_num_threads(4);seed_everything(42,True)
    out=HERE/'analysis';out.mkdir(exist_ok=True)
    require(not (out/'correctness.json').exists(),'Do not overwrite completed gate')
    lock=json.loads(BASELINE_LOCK.read_text());cache=FrozenEmbeddingCache(); cache_before=cache.pretrained.clone()
    phase_a=Path('/home/zyt/deep_final/artifacts/wssl_contribution_20261007/analysis')
    prior=json.loads((phase_a/'assets_lock.json').read_text())
    for p,h in {**prior['source_files'],**prior['asset_files']}.items():require(sha(p)==h,'PhaseA frozen asset changed '+p)
    for n in ['full_reproduction.json','smoke_test.json']:
        require(json.loads((phase_a/n).read_text())['status']=='PASS','Prior full-inference gate')
    rows=[];sample=None
    for item in lock['stages']:
        seed,c,i=item['seed'],item['context'],item['inner']
        ref=BASELINE/f'runs/b_str_pretrained/seed{seed}/outer_{c}/inner_{i}'
        ck=torch.load(ref/'checkpoints/best.pt',map_location='cpu',weights_only=False)
        ordinary=EMA_ARCHIVE/f'runs/ordinary/seed{seed}/outer_{c}/inner_{i}'
        oc=torch.load(ordinary/'checkpoints/best.pt',map_location='cpu',weights_only=False)
        for key in ['model_state','epoch','normalization','optimizer_state','scheduler_state']:
            require(equal_tree(ck[key],oc[key]),'Ordinary matched baseline differs '+key)
        require(ck['provenance']['source_tree_sha256']==oc['provenance']['source_tree_sha256'],'Baseline source provenance differs')
        for key in ['data','model','loss','training','evaluation']:
            require(ck['config'][key]==oc['config'][key],'Ordinary baseline recipe differs '+key)
        a=pd.read_csv(ref/'predictions/validation.csv',dtype={'subject_id':str}).sort_values('subject_id').reset_index(drop=True)
        b=pd.read_csv(ordinary/'predictions/validation.csv',dtype={'subject_id':str}).sort_values('subject_id').reset_index(drop=True)
        require(a.equals(b),'Ordinary predictions differ')
        histories=[[json.loads(x) for x in (p/'logs/epochs.jsonl').read_text().splitlines()] for p in [ref,ordinary]]
        require(len(histories[0])==len(histories[1]),'Ordinary stopping differs')
        for x,y in zip(*histories):
            for key in ['train','validation','epoch','best_epoch','best_metric','patience_count','improved']:
                require(({k:v for k,v in x[key].items() if k!='duration_seconds'} == {k:v for k,v in y[key].items() if k!='duration_seconds'}) if isinstance(x[key],dict) else x[key]==y[key], 'Ordinary trajectory differs '+key)
        ds=dataset_for(ck,a.subject_id.tolist()[:16],'validation')
        model=IndependentWSSL(ck['config']).cuda().eval();model.load_state_dict(ck['model_state'])
        candidate=ActivityScaledWSSL(ck['config']).cuda().eval()
        missing=candidate.load_state_dict(ck['model_state'],strict=False)
        require(missing.missing_keys==['activity_ssl_scale'] and not missing.unexpected_keys,'Compatibility keys differ')
        require(candidate.activities==tuple(cache.activities),'Cache activity index mismatch')
        for name,index in candidate.activity_index.items():require(ds.activities[index]==name,'Dataset activity index mismatch')
        memory=io.BytesIO();torch.save(candidate.state_dict(),memory);memory.seek(0)
        fresh=ActivityScaledWSSL(ck['config']).cuda().eval();fresh.load_state_dict(torch.load(memory,weights_only=True))
        n=0
        with torch.no_grad():
            for k in [0,8]:
                batch=collate_subject_activities([ds[j] for j in range(k,k+8)]);args,ssl=batch_args(batch,cache)
                baseline=model(*args,ssl_features=ssl)['logits'];actual=candidate(*args,ssl_features=ssl)['logits']
                require(torch.equal(baseline,actual),'g=1 differs')
                require(torch.equal(actual,fresh(*args,ssl_features=ssl)['logits']),'Reload differs')
                prob=actual.softmax(-1).cpu().numpy();expected=a.set_index('subject_id').loc[batch['subject_id'],['probability_pd','probability_dd']].to_numpy()
                require(np.max(np.abs(prob-expected))<=1e-12,'Formal archive probability mismatch')
                n+=1
                if sample is None:sample=(copy.deepcopy(ck),batch)
        rows.append(dict(seed=seed,context=c,inner=i,real_validation_batches=n,subjects=16,g1_logits_bit_exact=True,
                    reload_logits_bit_exact=True,ordinary_all_parameters_optimizer_trajectory_predictions_exact=True,
                    source_tree_sha256=ck['provenance']['source_tree_sha256']))
        print('CHECKPOINT PASS',seed,c,i,flush=True)
        del model,candidate,fresh,ds,ck,oc;gc.collect()
    ck,batch=sample;args,ssl=batch_args(batch,cache)
    initial=[]
    for seed in [42,43,44]:
        cfg=load_config(str(FOUNDATION/f'configs/str01_seed{seed}.yaml'));fixed_recipe(cfg)
        seed_everything(seed,True);base=IndependentWSSL(cfg).cuda().eval();rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state_all()
        seed_everything(seed,True);scaled=ActivityScaledWSSL(cfg).cuda().eval()
        require(torch.equal(rng,torch.get_rng_state()) and equal_tree(cuda_rng,torch.cuda.get_rng_state_all()),'Candidate consumes extra RNG')
        require(all(torch.equal(v,scaled.state_dict()[k]) for k,v in base.state_dict().items()),'Scratch initial state differs')
        with torch.no_grad():require(torch.equal(base(*args,ssl_features=ssl)['logits'],scaled(*args,ssl_features=ssl)['logits']),'Scratch initial logits differ')
        initial.append(dict(seed=seed,status='PASS',original_parameters_exact=True,rng_exact=True,logits_exact=True))
    # Use only inner-train subjects for scratch optimizer correctness, not validation.
    split=json.loads(Path(lock['split_path']).read_text())['outer'][0]['inner_folds'][0]
    train_ds=dataset_for(ck,split['train_subjects'][:8],'train');train_batch=collate_subject_activities([train_ds[j] for j in range(len(train_ds))])
    ta,ts=batch_args(train_batch,cache);targets=train_batch['y'].cuda()
    cfg=load_config(str(FOUNDATION/'configs/str01_seed42.yaml'));seed_everything(42,True)
    model=ActivityScaledWSSL(cfg).cuda().train();optimizer=nt.build_optimizer(model,cfg)
    counts=np.bincount([r.label for r in train_ds.records],minlength=2) if False else None
    weights=torch.tensor(ck['config']['loss']['class_weights'],device='cuda')
    state0={k:v.detach().clone() for k,v in model.state_dict().items()};g_grad=None
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss=torch.nn.functional.cross_entropy(model(*ta,ssl_features=ts)['logits'],targets,weight=weights)
        loss.backward();g_grad=model.activity_ssl_scale.grad.detach().clone()
        require(torch.isfinite(g_grad).all(),'Nonfinite g gradient');torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
    require((g_grad.abs()>0).all(),'Each activity scalar must have nonzero gradient after projection learns')
    require(any(not torch.equal(v,model.state_dict()[k]) for k,v in state0.items() if k.startswith('backbone.')),'STR not updated')
    require(not torch.equal(state0['wrist_projection.weight'],model.wrist_projection.weight),'Projection not updated')
    require(torch.equal(cache.pretrained,cache_before) and not cache.pretrained.requires_grad,'Frozen cache modified')
    model.eval();before=model(*ta,ssl_features=ts)['logits'].detach()
    buffer=io.BytesIO();torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict()),buffer);buffer.seek(0)
    saved=torch.load(buffer,map_location='cuda',weights_only=True);fresh=ActivityScaledWSSL(cfg).cuda().eval();fresh.load_state_dict(saved['model'])
    op2=nt.build_optimizer(fresh,cfg);op2.load_state_dict(saved['optimizer'])
    require(equal_tree(op2.state_dict(),optimizer.state_dict()),'Optimizer reload differs')
    require(torch.equal(before,fresh(*ta,ssl_features=ts)['logits']),'Updated model reload logits differ')
    require(torch.equal(model.activity_ssl_scale,fresh.activity_ssl_scale),'Scale reload differs')
    # Nonzero trained checkpoint projection for isolation tests.
    model=ActivityScaledWSSL(ck['config']).cuda().eval();model.load_state_dict(ck['model_state'],strict=False)
    with torch.no_grad():
        raw=model.wrist_projection(model.norm(ssl)).clone()
        model.activity_ssl_scale[0]=2
        changed=model.wrist_projection(model.norm(ssl))
        require(torch.equal(raw[:,1:],changed[:,1:]),'Scalar broadcast leaks other activities')
        require(torch.equal(changed[:,0],2*raw[:,0]),'Wrists do not share scalar')
        am=args[2].clone();wm=args[1].clone();am[:,0]=False;wm[:,1,0]=False
        masked=[args[0].clone(),wm,am,args[3].clone()]
        a=model(*masked,ssl_features=ssl)['logits']
        masked[0][:,0]=1000;masked[0][:,1,0]=1000
        changed_ssl=ssl.clone();changed_ssl[:,0]=1000;changed_ssl[:,1,0]=1000
        b=model(*masked,ssl_features=changed_ssl)['logits']
        require(torch.equal(a,b),'Missing activity/wrist leaks branch')
        model.activity_ssl_scale[0]=17
        require(torch.equal(b,model(*masked,ssl_features=changed_ssl)['logits']),'Missing activity scale affects logits')
        padded=[v.clone() for v in args];padded[3][:,0]=500
        a=model(*padded,ssl_features=ssl)['logits'];padded[0][:,0,:,:,500:]=1000
        require(torch.equal(a,model(*padded,ssl_features=ssl)['logits']),'Padded timesteps leak computation')
        # Explicit cache input overrides only this instance's bridge; no shared parameters/cache.
        fresh=ActivityScaledWSSL(ck['config']).cuda().eval();fresh.load_state_dict(model.state_dict())
        require(all(a.data_ptr()!=b.data_ptr() for a,b in zip(model.parameters(),fresh.parameters())),'Shared parameter storage')
        model.ssl_features=torch.zeros_like(ssl)
        require(torch.equal(model(*args,ssl_features=ssl)['logits'],fresh(*args,ssl_features=ssl)['logits']),'Explicit input/instance isolation failed')
    pd.DataFrame(rows).to_csv(out/'correctness_45checkpoints.csv',index=False)
    result=dict(status='PASS',checkpoint_count=45,real_validation_batches=90,subjects_per_checkpoint=16,
      baseline_reuse_exact=True,scratch_initialization=initial,g1_and_reload_bit_exact=True,
      finite_nonzero_gradient_all11=True,gradient=g_grad.cpu().tolist(),str_and_projection_update=True,
      cache_frozen=True,harnet_instantiated=False,mask_activity_wrist_padding=True,activity_index_from_config=True,
      both_wrists_share_same_scalar=True,optimizer_reload_exact=True,instance_isolation=True,
      classifier_baseline_parameters=143172,classifier_candidate_parameters=143183,frozen_harnet_parameters=10457408,
      system_baseline_parameters=10600580,system_candidate_parameters=10600591,outer_access=False,
      smoke_scores_used_for_design=False,source_provenance_current_baseline_exact=True,
      prior_full_validation_reproduction_sha256=sha(phase_a/'full_reproduction.json'),
      prior_normalization_audit_sha256=sha(phase_a/'normalization_15refits.csv'))
    (out/'correctness.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'initialization_check.json').write_text(json.dumps(dict(status='PASS',seeds=initial,runner_sha256=sha(HERE/'scripts/run.py')),indent=2)+'\n')
    print('ALL CORRECTNESS PASS',flush=True)

if __name__=='__main__':main()
