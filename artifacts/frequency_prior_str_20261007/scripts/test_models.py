"""All45 frozen zero-init cases plus meaningful gradient, mask and state-isolation tests."""
import copy
import gc
import io
import json
import pandas as pd
import torch
from common import *
from prior_model import build_prior, load_str_state, original_state
from src.models import build_model
from src.datasets.builders import load_configured_records
from src.datasets.subject_activity import SubjectActivityDataset,collate_subject_activities
from src.utils.seed import seed_everything


def forward(model, batch):
    return model(*(batch[k].to('cuda') for k in ['x','wrist_mask','activity_mask','activity_lengths']))['logits']


def sampled_batch(config,inner,norm,role='validation'):
    ids=set(inner['validation_subjects'] if role=='validation' else inner['train_subjects'])
    records=[r for r in load_configured_records(config['data'],['PD','DD']) if r.subject_id in ids]
    dataset=SubjectActivityDataset(records,config['data']['activities'],config['data'],norm['mean'],norm['std'],role)
    return collate_subject_activities([dataset[i] for i in range(min(8,len(dataset)))])


def main():
    torch.set_num_threads(4)
    lock=json.loads((HERE/'analysis/f0_lock.json').read_text());guard(lock)
    require(json.loads((HERE/'analysis/filter_bank_tests.json').read_text())['status']=='PASS','Filter gate missing')
    cfg=config_for(42)
    for seed in [42,43,44]:
        seed_everything(seed,True);base=build_model(cfg);post=torch.get_rng_state().clone()
        for condition in ['F1','F2']:
            seed_everything(seed,True);model=build_prior(cfg,condition)
            require(torch.equal(post,torch.get_rng_state()),'Candidate RNG differs')
            require(all(torch.equal(v,original_state(model)[k]) for k,v in base.state_dict().items()),'Initial core differs')
    split=json.loads(SPLIT.read_text());rows=[]
    for outer in split['outer']:
        c=int(outer['outer_fold'])
        for inner in outer['inner_folds']:
            i=int(inner['inner_fold'])
            for seed in [42,43,44]:
                ck=torch.load(reference(seed,c,i)/'checkpoints/best.pt',map_location='cpu',weights_only=False)
                batch=sampled_batch(ck['config'],inner,ck['normalization'])
                baseline=build_model(ck['config']).to('cuda').eval();baseline.load_state_dict(ck['model_state'])
                with torch.no_grad():a=forward(baseline,batch)
                for condition in ['F1','F2']:
                    model=build_prior(ck['config'],condition).to('cuda').eval();load_str_state(model,ck['model_state'])
                    with torch.no_grad():b=forward(model,batch)
                    require(torch.equal(a,b),'Frozen zero-init logits differ')
                    # Full model state roundtrip; no checkpoint file is retained for the sampled cases.
                    buf=io.BytesIO();torch.save(model.state_dict(),buf);buf.seek(0)
                    reloaded=build_prior(ck['config'],condition).to('cuda').eval()
                    reloaded.load_state_dict(torch.load(buf,weights_only=True,map_location='cuda'))
                    with torch.no_grad():d=forward(reloaded,batch)
                    require(torch.equal(b,d),'Checkpoint reload differs')
                    rows.append(dict(seed=seed,context=c,inner=i,condition=condition,
                        subject_count=len(batch['subject_id']),zero_init_max_logit_diff=float((a-b).abs().max()),
                        logits_exact=True,checkpoint_reload_exact=True))
                    del model,reloaded,buf
                del baseline,ck,batch;gc.collect()
            print(json.dumps(dict(status='PASS',context=c,inner=i,zero_init_cases=6)),flush=True)
    # Actual inner-training samples only for scratch updates and masks.
    inner=split['outer'][0]['inner_folds'][0]
    ck=torch.load(reference(42,0,0)/'checkpoints/best.pt',map_location='cpu',weights_only=False)
    batch=sampled_batch(ck['config'],inner,ck['normalization'],'train')
    tests=[]
    for condition in ['F1','F2']:
        seed_everything(42,True);model=build_prior(cfg,condition).to('cuda')
        branch=model.wrist_encoder.branch
        original={k:v.detach().clone() for k,v in branch.state_dict().items()}
        opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
        y=batch['y'].to('cuda');counts=ck['config']['loss']['class_weights'];w=torch.tensor(counts,device='cuda')
        upstream_values=[]
        for step in range(2):
            model.train();opt.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(forward(model,batch),y,weight=w).backward()
            require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()),'Missing/nonfinite grad')
            upstream=sum(float(p.grad.abs().sum()) for name,p in branch.named_parameters() if not name.startswith('residual.'))
            require(upstream==0 if step==0 else upstream>0,'Zero-init/second-step upstream gradient mismatch')
            upstream_values.append(upstream)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
        updated=[name for name,value in branch.state_dict().items() if not torch.equal(value,original[name])]
        require('residual.weight' in updated and len(updated)>2,'Branch parameters did not update')
        model.eval()
        # Finite ignored padding, one absent wrist, and one absent activity must not affect outputs.
        masked=copy.deepcopy(batch);masked['wrist_mask'][0,0,0]=0
        masked['activity_mask'][0,1]=False;masked['activity_lengths'][0,1]=0
        perturbed=copy.deepcopy(masked)
        perturbed['x'][0,0,0]=12345.
        perturbed['x'][0,1]=float('nan')
        for b in range(perturbed['x'].shape[0]):
            for a in range(11):
                length=int(masked['activity_lengths'][b,a])
                if masked['activity_mask'][b,a]:perturbed['x'][b,a,...,length:]=9876.
        with torch.no_grad():a=forward(model,masked);b=forward(model,perturbed)
        require(torch.equal(a,b),'Padding/wrist/activity isolation failed')
        invalid=copy.deepcopy(masked);invalid['x'][0,2,0,0,0]=float('inf')
        try:forward(model,invalid)
        except ValueError:pass
        else:raise AssertionError('Valid nonfinite input accepted')
        # Independently built and deep-copied model have separate parameter storage.
        twin=copy.deepcopy(model)
        require(all(p.data_ptr()!=q.data_ptr() for p,q in zip(model.parameters(),twin.parameters())),'Shared model storage')
        with torch.no_grad():before=forward(twin,masked).clone();branch.residual.weight.add_(1.)
        with torch.no_grad():after=forward(twin,masked)
        require(torch.equal(before,after),'Model copies cross-contaminate')
        tests.append(dict(condition=condition,total_parameters=sum(p.numel() for p in model.parameters()),
            added_parameters=sum(p.numel() for p in branch.parameters()),two_scratch_train_updates=True,
            upstream_gradients=upstream_values,updated_branch_tensors=len(updated),mask_padding_isolation_exact=True,
            valid_nonfinite_rejected=True,independent_copy_exact=True))
        del model,twin,opt;gc.collect()
    guard(lock)
    pd.DataFrame(rows).to_csv(HERE/'analysis/zero_init_90cases.csv',index=False)
    write_json(HERE/'analysis/model_tests.json',dict(status='PASS',frozen_checkpoints=45,candidate_cases=90,
        sampled_validation_batch_size=8,zero_init_logits_exact=True,reload_exact=True,
        fresh_initial_core_exact_all_seeds=True,cpu_rng_preserved=True,branch_tests=tests,
        outer_access=False,performance_selection=False,scratch_models_saved=False,
        invalid_wrist_contract='Finite placeholders required by original activity-level finite check; absent activity may contain nonfinite ignored values.'))
    print(json.dumps(dict(status='PASS',cases=90,tests=tests)),flush=True)


if __name__=='__main__':main()
