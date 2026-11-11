from common import *
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from src.datasets.builders import make_loader
from src.utils.config import load_config
from independent_wssl import IndependentWSSL
import inspect,re,time

def fixed_ce(logits,y,w,mean_weight):
    return F.cross_entropy(logits,y,weight=w,reduction='sum')/(len(y)*mean_weight)

def tests():
    rows=[]
    for case,y in [('single_PD',[0]*8),('single_DD',[1]*8),('batch1',[1]),('balanced',[0,1]*4),('last_small',[0,0,1]),('unweighted',[0,1,1])]:
        y=torch.tensor(y);z=torch.linspace(-1,2,len(y)*2,dtype=torch.float64).reshape(-1,2).requires_grad_()
        w=torch.ones(2,dtype=torch.float64) if case=='unweighted' else torch.tensor([.7,1.75],dtype=torch.float64)
        ce=-torch.log_softmax(z,1)[torch.arange(len(y)),y]
        a=F.cross_entropy(z,y,weight=w);b=fixed_ce(z,y,w,1.)
        assert torch.allclose(a,(ce*w[y]).sum()/w[y].sum(),atol=1e-12,rtol=0)
        assert torch.allclose(b,(ce*w[y]).sum()/len(y),atol=1e-12,rtol=0)
        ga=torch.autograd.grad(a,z,retain_graph=True)[0];gb=torch.autograd.grad(b,z,retain_graph=True)[0]
        ratio=w[y].sum()/len(y)
        assert torch.allclose(gb,ga*ratio,atol=1e-12,rtol=0)
        soft=F.one_hot(y,2).double();softloss=F.cross_entropy(z,soft,weight=w)
        assert torch.allclose(softloss,(ce*w[y]).mean(),atol=1e-12,rtol=0)
        rows.append(dict(case=case,batch_size=len(y),weighted_mean=float(a.detach()),fixed=float(b.detach()),gradient_scale=float(ratio),soft_label_formula_separate=True))
    write_json(HERE/'tests/loss_correctness.json',dict(status='PASS',cases=rows,autograd_double_absolute_tolerance=1e-12,actual_FP32_relative_gradient_tolerance=2e-5))

def main():
    torch.set_num_threads(4);guard();tests()
    (HERE/'A0_PROTOCOL.md').write_bytes((HERE/'PROTOCOL.md').read_bytes())
    freeze('A0',[HERE/'A0_PROTOCOL.md',Path(__file__),HERE/'scripts/common.py',HERE/'tests/loss_correctness.json'])
    audit=[];epochs=[];gradients=[];documentary=[]
    # No new validation-performance inspection is used to decide reduction eligibility.
    source_table=pd.read_csv(HERE/'analysis/history_sources.csv')
    pattern=re.compile(r'global\s+(?:class[- ]balanced|balanced)\s+(?:empirical\s+)?risk|全局.{0,12}(?:经验风险|平衡风险)|固定分母.{0,20}(?:目标|风险)',re.I)
    for row in source_table.itertuples():
        for line in Path(row.path).read_text().splitlines():
            if pattern.search(line):documentary.append(dict(path=row.path,line=line[:500]))
    # Mere mention of a future fixed-denominator audit is not a declaration of
    # the original optimizer objective. Historical protocols prescribe weighted
    # mean CE without asserting a full-data fixed denominator.
    b_confirmed=False
    for c,i,sp in splits():
      for seed in [42,43,44]:
        ck=torch.load(baseline(seed,c,i)/'checkpoints/best.pt',map_location='cpu',weights_only=False)
        with np.load(HERE/'features'/f'seed{seed}_context{c}_inner{i}.npz') as z:
            keep=~z['is_validation'];ids=z['subject_ids'][keep].tolist();y=torch.from_numpy(z['target'][keep].copy()).long();logits=torch.from_numpy(z['logits'][keep].copy())
        assert set(ids)==set(sp['train_subjects']);assert ids==sorted(ids)
        assert ck['config']['loss']['label_smoothing']==0.
        fit_ids(f'A0_sampler_{seed}_{c}_{i}',ids,[],ids,validation_used=False)
        counts=torch.bincount(y,minlength=2);w=len(y)/(2*counts.float());assert torch.allclose(w,torch.tensor(ck['config']['loss']['class_weights']),atol=1e-7,rtol=0)
        mean_weight=float(w[y].mean());ce=F.cross_entropy(logits,y,reduction='none')
        full_risk=float((ce*w[y]).sum()/(len(y)*mean_weight))
        loader=make_loader(TensorDataset(torch.arange(len(y)),y,ce),8,True,0,False,seed)
        assert loader.sampler.__class__.__name__=='RandomSampler' and not loader.drop_last
        scales=[];dd_mass=[];weighted_dd=[];first_indices=None;small_batches=0
        for epoch in range(50):
            total_self=total_fixed=0.;n=0
            for index,by,bce in loader:
                B=len(by);weights=w[by];den=float(weights.sum());numerator=float((weights*bce).sum());ratio=den/(B*mean_weight)
                scales.append(ratio);dd_mass.append(float(weights[by==1].sum()/den));weighted_dd.append(float(weights[by==1].sum()/(B*mean_weight)))
                total_self+=B*numerator/den;total_fixed+=B*numerator/(B*mean_weight);n+=B
                small_batches+=int(B<8)
                if first_indices is None:first_indices=index.numpy().copy()
            assert n==len(y) and abs(total_fixed/n-full_risk)<2e-7
            epochs.append(dict(context=c,inner=i,seed=seed,epoch=epoch+1,self_normalized_epoch=total_self/n,fixed_denominator_epoch=total_fixed/n,full_balanced_empirical_risk=full_risk,delta_self_minus_fixed=total_self/n-full_risk))
        audit.append(dict(context=c,inner=i,seed=seed,train_n=len(y),pd_n=int(counts[0]),dd_n=int(counts[1]),weight_pd=float(w[0]),weight_dd=float(w[1]),mean_train_example_weight=mean_weight,natural_dd_fraction=float((y==1).float().mean()),scale_mean=float(np.mean(scales)),scale_sd=float(np.std(scales,ddof=1)),scale_min=float(min(scales)),scale_p05=float(np.quantile(scales,.05)),scale_p95=float(np.quantile(scales,.95)),scale_max=float(max(scales)),mean_self_normalized_DD_coefficient=float(np.mean(dd_mass)),mean_fixed_DD_coefficient=float(np.mean(weighted_dd)),small_batches=small_batches,drop_last=False,sampler='RandomSampler'))
        if (c,i)==(0,0):
            from src.datasets import folds as fd
            from src.datasets.subject_activity import SubjectActivityDataset,collate_subject_activities
            cfg=load_config(MAIN/f'configs/str01_seed{seed}.yaml');records=[r for r in fd.load_configured_records(cfg['data'],['PD','DD']) if r.subject_id in set(ids)]
            ds=SubjectActivityDataset(records,cfg['data']['activities'],cfg['data'],ck['normalization']['mean'],ck['normalization']['std'],'train')
            assert ds.subject_ids==ids
            batch=collate_subject_activities([ds[int(j)] for j in first_indices]);inputs=[batch[k].cuda() for k in ['x','wrist_mask','activity_mask','activity_lengths']]
            fit_ids(f'A0_gradient_batch_seed{seed}',batch['subject_id'],[],ids,validation_used=False)
            ssl=torch.from_numpy(cache(batch['subject_id'])).cuda();model=IndependentWSSL(cfg).cuda().eval();model.load_state_dict(ck['model_state'],strict=True)
            z=model(*inputs,ssl_features=ssl)['logits'];target=batch['y'].cuda();weights=w.cuda();a=F.cross_entropy(z,target,weight=weights);b=fixed_ce(z,target,weights,mean_weight)
            lg_a=torch.autograd.grad(a,z,retain_graph=True)[0];lg_b=torch.autograd.grad(b,z,retain_graph=True)[0];ratio=float(weights[target].sum()/(len(target)*mean_weight))
            pa=list(model.parameters());ga=torch.autograd.grad(a,pa,retain_graph=True,allow_unused=True);gb=torch.autograd.grad(b,pa,allow_unused=True)
            va=torch.cat([g.reshape(-1) for g in ga if g is not None]);vb=torch.cat([g.reshape(-1) for g in gb if g is not None]);relative=float((vb-va*ratio).norm()/vb.norm().clamp_min(1e-12));assert relative<=2e-5
            assert torch.allclose(lg_b,lg_a*ratio,atol=2e-7,rtol=2e-5)
            na=float(va.norm());nb=float(vb.norm());ca=va*min(1,5/(na+1e-6));cb=vb*min(1,5/(nb+1e-6))
            assert torch.isfinite(va).all() and torch.isfinite(vb).all()
            gradients.append(dict(seed=seed,logit_gradient_scale=ratio,parameter_gradient_relative_error=relative,batch_gradient_norm=na,fixed_gradient_norm=nb,batch_clipped_norm=float(ca.norm()),fixed_clipped_norm=float(cb.norm()),clipped_direction_cosine=float(F.cosine_similarity(ca,cb,dim=0)),optimizer_updates=0))
            del model,ds,z
        del ck
    pd.DataFrame(audit).to_csv(HERE/'analysis/A0_sampler_summary.csv',index=False)
    pd.DataFrame(epochs).to_csv(HERE/'analysis/A0_epoch_risk.csv',index=False)
    write_json(HERE/'analysis/A0_real_gradients.json',dict(status='PASS',cases=gradients))
    correctness=json.loads((HERE/'tests/model_correctness.json').read_text())['status']=='PASS'
    difference=bool(min(x['scale_sd'] for x in audit)>0 and any(abs(x['delta_self_minus_fixed'])>1e-8 for x in epochs))
    gate=dict(a_batch_ratio_path=True,b_explicit_historical_global_risk_target_confirmed=b_confirmed,c_train_only_difference_reproduced=difference,d_same_historical_denominator_control_absent=True,e_correctness_assets_pass=correctness)
    guard();allowed=all(gate.values())
    write_json(HERE/'analysis/A0_gate.json',dict(status='AUDIT_COMPLETE',gate=gate,training_allowed=allowed,new_deep_training_runs=0,b_status='NOT_CONFIRMED: inverse-frequency weights do not establish a historical global-denominator objective',documentary_hits=documentary,torch_version=torch.__version__,pytorch_documentation='https://docs.pytorch.org/docs/2.8/generated/torch.nn.CrossEntropyLoss.html',runtime_loss_source=str(MAIN/'src/losses/mfam_loss.py'),runtime_epoch_aggregation_source=str(MAIN/'src/engine/runner.py'),default_weighted_mean_is_not_a_bug=True,within_batch_relative_class_weights_unchanged=True,DD_error_causation_proven=False,optimizer_updates=0))
    assert not allowed,'A fully passing gate requires the registered45-run candidate before completion'
    print(json.dumps(dict(stage='A0',status='COMPLETE',training_allowed=allowed,new_deep_training_runs=0)),flush=True)

if __name__=='__main__':main()
