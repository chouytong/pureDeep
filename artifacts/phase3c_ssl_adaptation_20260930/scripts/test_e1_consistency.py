import json,sys,torch
from pathlib import Path
ROOT=Path('/home/zyt/deep_final/foundation_validation');P3B=Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928')
sys.path[:0]=[str(ROOT),str(P3B/'scripts'),str(Path(__file__).resolve().parent)]
from src.utils.config import load_config
from src.engine.nested_training import build_model
from src.engine.checkpoint import load_checkpoint
from transfer_hooks import FrozenEmbeddingCache,ResidualSSLSubject
from phase3c_hooks import Stage4Cache,AdaptedResidual
c=load_config(str(ROOT/'configs/str01_seed42.yaml'));torch.manual_seed(42)
original=ResidualSSLSubject(build_model(c)).eval().cuda()
adapt=AdaptedResidual(build_model(c)).eval().cuda()
path=P3B/'runs/b_str_pretrained/seed42/outer_0/inner_0/checkpoints/best.pt'
load_checkpoint(path,model=original,map_location='cuda');load_checkpoint(path,model=adapt.base,map_location='cuda')
frozen=FrozenEmbeddingCache();stage4=Stage4Cache();sid=frozen.ids[0]
x=torch.zeros((1,11,2,6,2000),device='cuda');wm=torch.ones((1,11,2),device='cuda',dtype=torch.bool);am=torch.ones((1,11),device='cuda',dtype=torch.bool)
lengths=torch.tensor([[976]*8+[2000]*3],device='cuda')
original.ssl_features=frozen.batch([sid],'pretrained',torch.device('cuda'))
adapt.stage4_windows,adapt.window_counts=stage4.batch([sid],torch.device('cuda'))
with torch.no_grad():
 a=original(x,wm,am,lengths)['logits'];b=adapt(x,wm,am,lengths)['logits']
diff=float((a-b).abs().max());assert diff<1e-6,diff
adapt.train();bn=adapt.layer5[1].running_mean.detach().clone()
out=adapt(x,wm,am,lengths)['logits'];out.sum().backward()
assert adapt.layer5[0].weight.grad is not None
assert not adapt.stage4_windows.requires_grad
assert torch.equal(bn,adapt.layer5[1].running_mean)
print(json.dumps({'phase3b_checkpoint_logit_max_diff':diff,'layer5_conv_has_gradient':True,'stage4_cache_requires_grad':False,'layer5_bn_running_mean_unchanged':True,'layer5_parameters':sum(p.numel() for p in adapt.layer5.parameters()),'trainable_total':sum(p.numel() for p in adapt.parameters() if p.requires_grad)}))
