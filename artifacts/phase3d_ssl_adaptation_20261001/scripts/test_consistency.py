"""Read-only initial-logit, checkpoint and parameter checks for Phase-3D hooks."""
import copy,json,sys
from pathlib import Path
import torch
ROOT=Path('/home/zyt/deep_final/foundation_validation');P3B=Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928');BASE=Path('/home/zyt/deep_final/artifacts/phase3d_ssl_adaptation_20261001')
sys.path[:0]=[str(ROOT),str(P3B/'scripts'),str(BASE/'scripts')]
from src.utils.config import load_config
from src.engine import nested_training as nt
from transfer_hooks import ResidualSSLSubject
from phase3d_hooks import AdapterResidual,GatedResidual
c=load_config(str(ROOT/'configs/str01_seed42.yaml'));torch.manual_seed(20261001)
b=nt.build_model(c);base=copy.deepcopy(b).eval();frozen=ResidualSSLSubject(copy.deepcopy(b)).eval();adapter=AdapterResidual(copy.deepcopy(b)).eval();gate=GatedResidual(copy.deepcopy(b)).eval();act=GatedResidual(copy.deepcopy(b),activity_gate=True).eval()
x=torch.randn(1,11,2,6,64);wm=torch.ones(1,11,2,dtype=torch.bool);am=torch.ones(1,11,dtype=torch.bool);lengths=torch.full((1,11),64,dtype=torch.long)
z=torch.randn(1,11,2,1024)
frozen.ssl_features=z;adapter.raw_features=z;gate.ssl_features=z;act.ssl_features=z
with torch.inference_mode():
 ref=base(x,wm,am,lengths)['logits']
 init={name:float((model(x,wm,am,lengths)['logits']-ref).abs().max()) for name,model in [('frozen',frozen),('adapter',adapter),('gate',gate),('activity_gate',act)]}
assert all(v==0 for v in init.values()),init
cp=P3B/'runs/b_str_pretrained/seed42/outer_0/inner_0/checkpoints/best.pt';state=torch.load(cp,map_location='cpu',weights_only=False)['model_state']
a=ResidualSSLSubject(nt.build_model(c)).eval();d=AdapterResidual(nt.build_model(c)).eval();a.load_state_dict(state);miss,unexpected=d.load_state_dict(state,strict=False)
assert all(k.startswith('adapter.') for k in miss) and not unexpected
a.ssl_features=z;d.raw_features=z
with torch.inference_mode():cpdiff=float((a(x,wm,am,lengths)['logits']-d(x,wm,am,lengths)['logits']).abs().max())
assert cpdiff==0,cpdiff
counts={}
for name,model in [('frozen',frozen),('adapter',adapter),('gate',gate),('activity_gate',act)]:
 counts[name]={'total':sum(p.numel() for p in model.parameters()),'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad)}
assert counts['adapter']['trainable']-counts['frozen']['trainable']==132160
assert counts['gate']['trainable']-counts['frozen']['trainable']==6240
assert counts['activity_gate']['trainable']-counts['gate']['trainable']==11
out={'initial_logit_max_diff_vs_str':init,'adapter_phase3b_checkpoint_logit_max_diff':cpdiff,'parameter_counts':counts,'ssl_cache_requires_grad':z.requires_grad,'outer_access':False}
(BASE/'analysis/consistency.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
