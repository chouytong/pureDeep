"""Synthetic demographic zero-residual/frozen-base and round-trip correctness."""
from common10 import *
from models10 import DemographicResidual,fingerprint,parameter_groups
from train10 import config,TraceAdamW
from src.utils.seed import seed_everything
from torch.nn import functional as F
import torch,io,gc

def main():
 assert (HERE/'reports/PHASE10B_UNFREEZING_REPORT.md').exists()
 seed_everything(42,True);torch.set_num_threads(4)
 c=config(42);base=model_for('wssl',c).cuda().eval();r=line('wssl',42,0,0)
 assert sha(r['checkpoint'])==r['checkpoint_sha256']
 base.load_state_dict(torch.load(r['checkpoint'],map_location='cuda',weights_only=False)['model_state'])
 x=torch.randn(2,11,2,6,2000,device='cuda');wm=torch.ones(2,11,2,dtype=torch.bool,device='cuda');am=torch.ones(2,11,dtype=torch.bool,device='cuda');lengths=torch.full((2,11),2000,device='cuda');base.ssl_features=torch.randn(2,11,2,1024,device='cuda')
 with torch.no_grad():z=base(x,wm,am,lengths)['logits']
 before=fingerprint(base);model=DemographicResidual(base).cuda();d=torch.randn(2,4,device='cuda');y=torch.tensor([0,1],device='cuda')
 with torch.no_grad():assert torch.equal(model(d,z)['logits'],z)
 assert sum(p.numel() for p in model.parameters() if p.requires_grad)==216
 opt=TraceAdamW(parameter_groups(model,'DEMO'))
 for _ in range(2):
  model.train();assert not model.base.training and not any(p.requires_grad for p in model.base.parameters())
  opt.zero_grad(set_to_none=True);F.cross_entropy(model(d,z)['logits'],y).backward();opt.step()
 assert fingerprint(base)==before
 assert model.demographic_mlp[0].weight.grad is not None and torch.isfinite(model.demographic_mlp[0].weight.grad).all() and model.demographic_mlp[0].weight.grad.abs().sum()>0
 try:model(torch.zeros(2,5,device='cuda'),z)
 except ValueError:pass
 else:raise AssertionError('Extra demographic feature accepted')
 model.eval()
 with torch.no_grad():expected=model(d,z)['logits']
 stream=io.BytesIO();torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict()),stream);stream.seek(0)
 other=DemographicResidual(model_for('wssl',c)).cuda().eval();saved=torch.load(stream,map_location='cuda',weights_only=False);other.load_state_dict(saved['model']);otheropt=TraceAdamW(parameter_groups(other,'DEMO'));otheropt.load_state_dict(saved['optimizer'])
 with torch.no_grad():assert torch.equal(expected,other(d,z)['logits'])
 result=dict(status='PASS',original_certified_checkpoint_sha256=r['checkpoint_sha256'],zero_init_logits_bitwise=True,
             frozen_base_state_unchanged_after_two_updates=True,base_in_eval_while_demo_trains=True,trainable_parameters=216,
             four_features_only=True,MLP_gradient_nonzero_after_zero_projection_warmup=True,model_optimizer_roundtrip_bitwise=True,
             synthetic_optimizer_updates=2,clinical_training_runs=0,source_sha256=sha(Path(__file__)),protected_files_verified=guard())
 write(HERE/'tests/demographic_correctness.json',result,new=True);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
