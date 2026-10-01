import json,sys,torch
from pathlib import Path
ROOT=Path('/home/zyt/deep_final/foundation_validation');sys.path.insert(0,str(ROOT))
from src.utils.config import load_config
from src.engine.nested_training import build_model
from transfer_hooks import FrozenEmbeddingCache,ResidualSSLSubject
from src.engine.checkpoint import load_checkpoint
config=load_config(str(ROOT/'configs/str01_seed42.yaml'))
torch.manual_seed(42)
base=build_model(config).eval().cuda()
load_checkpoint('/home/zyt/deep_final/artifacts/phase2_str_training_20260926/runs/baseline/seed42/outer_0/inner_0/checkpoints/best.pt',model=base,map_location='cuda')
x=torch.zeros((1,11,2,6,2000),device='cuda')
wm=torch.ones((1,11,2),device='cuda',dtype=torch.bool)
am=torch.ones((1,11),device='cuda',dtype=torch.bool)
lengths=torch.tensor([[976,976,976,976,976,976,976,976,2000,2000,2000]],device='cuda')
with torch.no_grad(): before=base(x,wm,am,lengths)['logits'].clone()
wrapped=ResidualSSLSubject(base).eval().cuda();cache=FrozenEmbeddingCache()
wrapped.ssl_features=cache.batch([cache.ids[0]],'pretrained',torch.device('cuda'))
with torch.no_grad():after=wrapped(x,wm,am,lengths)['logits']
maxdiff=(after-before).abs().max().item()
assert maxdiff<1e-7,(before,after,maxdiff)
wrapped.train();out=wrapped(x,wm,am,lengths)['logits'];out.sum().backward()
assert wrapped.wrist_projection.weight.grad is not None
assert not wrapped.ssl_features.requires_grad
print(json.dumps({'max_abs_logit_difference':maxdiff,'frozen_cache_requires_grad':wrapped.ssl_features.requires_grad,'projection_gradient_exists':True,'feature_shape':list(wrapped.ssl_features.shape)}))
