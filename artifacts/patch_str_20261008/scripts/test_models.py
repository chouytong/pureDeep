"""Per-condition pre-formal initialization/autograd/mask/reload gates."""
import argparse,io,gc,copy
import torch,numpy as np,pandas as pd
from common import *
from patch_models import build_patch,load_str_state,original_state,COUNTS
from patch_extractor import extract_patches
from src.models import build_model as build_str
from src.datasets.subject_activity import SubjectActivityDataset,collate_subject_activities
from src.engine import nested_training as nt
from src.utils.seed import seed_everything


def same(a,b):
    if torch.is_tensor(a):return a.dtype==b.dtype and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def dataset(ck,ids,role):
    records=fd.load_configured_records(ck['config']['data'],['PD','DD'])
    return SubjectActivityDataset([r for r in records if r.subject_id in set(ids)],ck['config']['data']['activities'],
           ck['config']['data'],ck['normalization']['mean'],ck['normalization']['std'],role)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--condition',choices=['P1'],required=True);args=parser.parse_args();condition=args.condition
    torch.set_num_threads(4);seed_everything(42,True);lock=json.loads((HERE/'analysis/f0_lock.json').read_text());guard(lock)
    rows=[];sample=None
    for item in lock['stages']:
        ck=torch.load(Path(item['path'])/'checkpoints/best.pt',map_location='cpu',weights_only=False)
        frame=pd.read_csv(Path(item['path'])/'predictions/validation.csv',dtype={'subject_id':str})
        ds=dataset(ck,frame.subject_id.tolist()[:8],'validation');batch=collate_subject_activities([ds[j] for j in range(len(ds))]);inputs=[batch[k].cuda() for k in ['x','wrist_mask','activity_mask','activity_lengths']]
        base=build_str(ck['config']).cuda().eval();base.load_state_dict(ck['model_state'])
        model=build_patch(ck['config'],condition).cuda().eval();load_str_state(model,ck['model_state'])
        with torch.no_grad():a=base(*inputs)['logits'];b=model(*inputs)['logits'];require(torch.equal(a,b),'Frozen checkpoint zero-init logits differ')
        buf=io.BytesIO();torch.save(dict(model=model.state_dict(),patch_config=dict(condition=condition,patch_length=200,stride=100,total_parameters=COUNTS[condition]),source_sha256=sha(HERE/'scripts/patch_models.py')),buf);buf.seek(0);saved=torch.load(buf,weights_only=True,map_location='cuda')
        require(saved['patch_config']==dict(condition=condition,patch_length=200,stride=100,total_parameters=COUNTS[condition]),'Config not serialized')
        fresh=build_patch(ck['config'],condition).cuda().eval();fresh.load_state_dict(saved['model'])
        with torch.no_grad():require(torch.equal(b,fresh(*inputs)['logits']),'Frozen checkpoint reload differs')
        rows.append(dict(seed=item['seed'],context=item['context'],inner=item['inner'],condition=condition,zero_logits_exact=True,reload_exact=True))
        if sample is None:sample=(copy.deepcopy(ck),batch)
        del base,model,fresh,ds;gc.collect()
    ck,batch=sample;inputs=[batch[k].cuda() for k in ['x','wrist_mask','activity_mask','activity_lengths']];init=[]
    for seed in [42,43,44]:
        cfg=config_for(seed);seed_everything(seed,True);base=build_str(cfg).cuda().eval();cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state_all()
        seed_everything(seed,True);model=build_patch(cfg,condition).cuda().eval()
        require(same(base.state_dict(),original_state(model)),'Original initialized parameters differ')
        require(same(cpu,torch.get_rng_state()) and same(cuda,torch.cuda.get_rng_state_all()),'Training/DataLoader RNG altered')
        with torch.no_grad():require(torch.equal(base(*inputs)['logits'],model(*inputs)['logits']),'Fresh zero-init logits differ')
        init.append(dict(seed=seed,status='PASS',parameters_logits_rng_exact=True))
    split=json.loads(SPLIT.read_text())['outer'][0]['inner_folds'][0]
    ds=dataset(ck,split['train_subjects'][:8],'train');batch=collate_subject_activities([ds[j] for j in range(len(ds))]);inputs=[batch[k].cuda() for k in ['x','wrist_mask','activity_mask','activity_lengths']];targets=batch['y'].cuda()
    cfg=config_for(42);seed_everything(42,True);model=build_patch(cfg,condition).cuda().train();optimizer=nt.build_optimizer(model,cfg);initial={k:v.clone() for k,v in model.state_dict().items()}
    weights=torch.tensor(ck['config']['loss']['class_weights'],device='cuda');branch=model.wrist_encoder.branch;grads=[]
    for step in range(2):
        optimizer.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(model(*inputs)['logits'],targets,weight=weights);loss.backward()
        require(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()),'Nonfinite gradients')
        def gradnorm(module):return float(sum(p.grad.square().sum() for p in module.parameters() if p.grad is not None).sqrt())
        enc=gradnorm(branch.encoder);out=gradnorm(branch.residual);bottle=gradnorm(branch.bottleneck)
        require(out>0,'Final projection has no gradient')
        require(enc==0 if step==0 else enc>0,'Upstream encoder gradient stage incorrect')
        if step==1:require(bottle>0,'Bottleneck gradient missing')
        if condition=='P2':
            agg=gradnorm(branch.order_conv)+gradnorm(branch.scorer)
            require(agg==0 if step==0 else agg>0,'Ordered aggregator gradient stage incorrect')
        else:agg=0.
        grads.append(dict(step=step+1,encoder_grad_norm=enc,projection_grad_norm=out,bottleneck_grad_norm=bottle,aggregator_grad_norm=agg))
        torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
    require(any(not torch.equal(v,model.state_dict()[k]) for k,v in initial.items() if 'branch.encoder' in k),'Encoder did not update')
    require(any(not torch.equal(v,model.state_dict()[k]) for k,v in initial.items() if '.branch.' not in k),'Original STR did not update')
    model.eval();buf=io.BytesIO();torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict()),buf);buf.seek(0);saved=torch.load(buf,map_location='cuda',weights_only=True)
    fresh=build_patch(cfg,condition).cuda().eval();fresh.load_state_dict(saved['model']);op2=nt.build_optimizer(fresh,cfg);op2.load_state_dict(saved['optimizer']);require(same(op2.state_dict(),optimizer.state_dict()),'Optimizer reload not exact')
    with torch.no_grad():
        original=model(*inputs)['logits'];require(torch.equal(original,fresh(*inputs)['logits']),'Updated logits reload differ')
        wm=inputs[1].clone();am=inputs[2].clone();wm[:,1,0]=0;am[:,0]=False
        altered=[inputs[0].clone(),wm,am,inputs[3].clone()];a=model(*altered)['logits'];altered[0][:,0]=999;altered[0][:,1,0]=999
        require(torch.equal(a,model(*altered)['logits']),'Missing activity/wrist leak')
        altered=[v.clone() for v in inputs];a=model(*altered)['logits']
        for activity in range(11):
            for subject in range(len(batch['subject_id'])):
                length=int(altered[3][subject,activity]);altered[0][subject,activity,:,:,length:]=999
        require(torch.equal(a,model(*altered)['logits']),'Padded samples leak into logits')
        # Add invalid patch slots at fixed shape, vary only their contents.
        method=branch.from_patches
        def feed(value):
            def wrapped(patches,mask):
                padded=torch.cat([patches,patches.new_full((patches.shape[0],3,6,200),value)],dim=1)
                return method(padded,torch.cat([mask,torch.zeros(mask.shape[0],3,device=mask.device,dtype=torch.bool)],dim=1))
            return wrapped
        branch.from_patches=feed(0);a=model(*inputs)['logits'];branch.from_patches=feed(float('nan'));b=model(*inputs)['logits'];branch.from_patches=method
        require(torch.equal(a,b),'Invalid raw patch content affects logits')
        aggregate=branch.aggregate
        def contaminated(value):
            def wrapped(z,mask):
                z=z.clone();z[~mask]=value;return aggregate(z,mask)
            return wrapped
        branch.from_patches=feed(0);branch.aggregate=contaminated(0);a=model(*inputs)['logits'];branch.aggregate=contaminated(999);b=model(*inputs)['logits'];branch.aggregate=aggregate;branch.from_patches=method
        require(torch.equal(a,b),'Invalid patch embeddings affect logits')
        z=torch.randn(2,7,64,device='cuda');mask=torch.tensor([[1,1,1,1,0,0,0],[1]*7],dtype=torch.bool,device='cuda')
        if condition=='P1':
            perm=torch.tensor([3,1,0,2,4,5,6],device='cuda');a=branch.aggregate(z,mask)[0];b=branch.aggregate(z[:,perm],mask[:,perm])[0];require(torch.allclose(a,b,atol=1e-6,rtol=0),'Bag summary uses explicit order')
        else:
            require(float((branch.aggregate(z,mask)[0]-branch.aggregate(z.flip(1),mask.flip(1))[0]).abs().max())>1e-6,'Ordered operator unexpectedly permutation invariant')
    guard(lock);pd.DataFrame(rows).to_csv(HERE/f'analysis/{condition.lower()}_45checkpoint_tests.csv',index=False)
    write_json(HERE/f'analysis/{condition.lower()}_tests.json',dict(status='PASS',condition=condition,checkpoints=45,initialization=init,gradient_updates=grads,mask_activity_wrist_samples_patches_pass=True,shared_encoder_all_activities_wrists=True,normalization_new_branch=False,reload_optimizer_logits_exact=True,parameter_count=COUNTS[condition],added_parameters=COUNTS[condition]-75524,outer_access=False,smoke_scores_selected=False,model_source_sha256=sha(HERE/'scripts/patch_models.py')))
    print(condition,'ALL TESTS PASS',flush=True)

if __name__=='__main__':main()
