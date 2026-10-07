"""F1 stage verification on synthetic inputs; no validation performance."""
import json
import torch
from common import config_for, require, HERE, write_json
from prior_model import build_prior, original_state
from src.models import build_model
from src.utils.seed import seed_everything


def main():
    torch.set_num_threads(4)
    cfg = config_for(42)
    seed_everything(42,True);baseline=build_model(cfg);rng=torch.get_rng_state().clone()
    seed_everything(42,True);candidate=build_prior(cfg,'F1')
    require(torch.equal(rng,torch.get_rng_state()), 'F1 changed RNG after base initialization')
    require(all(torch.equal(v,original_state(candidate)[k]) for k,v in baseline.state_dict().items()), 'F1 base initial values differ')
    baseline=baseline.to('cuda').eval();candidate=candidate.to('cuda').eval()
    x=torch.randn(2,11,2,6,976,device='cuda');wm=torch.ones(2,11,2,device='cuda')
    am=torch.ones(2,11,dtype=torch.bool,device='cuda');lengths=torch.full((2,11),976,device='cuda')
    with torch.no_grad():
        a=baseline(x,wm,am,lengths)['logits'];b=candidate(x,wm,am,lengths)['logits']
    require(torch.equal(a,b),'F1 zero-init logits differ')
    candidate.train();opt=torch.optim.AdamW(candidate.parameters(),lr=2e-4,weight_decay=1e-4)
    y=torch.tensor([0,1],device='cuda');upstream=None
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(candidate(x,wm,am,lengths)['logits'],y).backward()
        require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in candidate.parameters()),'F1 gradient missing/nonfinite')
        upstream=sum(float(p.grad.abs().sum()) for p in candidate.wrist_encoder.branch.encode.parameters())
        if step==0:require(upstream==0,'Zero final layer should block first upstream gradient')
        else:require(upstream>0,'F1 upstream branch not connected after projection update')
        torch.nn.utils.clip_grad_norm_(candidate.parameters(),5.);opt.step()
    require(candidate.wrist_encoder.branch.residual.weight.abs().sum()>0,'F1 zero residual never updated')
    report=dict(status='PASS',condition='F1',total_parameters=78005,added_parameters=2481,
                initialization_exact=True,rng_preserved=True,zero_init_logits_exact=True,
                two_scratch_updates=True,second_upstream_gradient_nonzero=upstream>0,
                synthetic_only=True,performance_selection=False,trained_model_saved=False,outer_access=False)
    write_json(HERE/'analysis/f1_stage_tests.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
