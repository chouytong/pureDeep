import sys,json,torch
from pathlib import Path
ROOT=Path('/home/zyt/deep_final/foundation_validation');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
from src.utils.config import load_config
from src.models import build_model
from phase3_hooks import ActivityAuxModel
from torch import nn
config=load_config(str(ROOT/'configs/str01_seed42.yaml'))
base=build_model(config)
checkpoint=torch.load('/home/zyt/deep_final/artifacts/phase2_str_training_20260926/runs/baseline/seed42/outer_0/inner_0/checkpoints/best.pt',map_location='cpu',weights_only=False)
base.load_state_dict(checkpoint['model_state'])
wrapped=ActivityAuxModel(build_model(config),32)
wrapped.backbone.load_state_dict(checkpoint['model_state'])
assert not any(isinstance(m,nn.modules.batchnorm._BatchNorm) for m in base.modules())
base.eval();wrapped.eval()
torch.manual_seed(20260928)
x=torch.randn(2,11,2,6,128);w=torch.ones(2,11,2,dtype=torch.bool);a=torch.ones(2,11,dtype=torch.bool);lengths=torch.full((2,11),128,dtype=torch.long)
with torch.no_grad():
 y0=base(x,w,a,lengths)['logits'];y1=wrapped(x,w,a,lengths)['logits']
assert torch.equal(y0,y1)
result={'checkpoint':'Phase-2 exact STR baseline seed42 outer0 inner0','max_abs_final_logit_delta':float((y0-y1).abs().max()),'bitwise_equal':True,'base_parameters':sum(p.numel() for p in base.parameters()),'aux_parameters':sum(p.numel() for p in wrapped.activity_aux.parameters()),'no_batchnorm':True,'aux_disabled_in_eval':True}
out=Path('/home/zyt/deep_final/artifacts/phase3a_puredeep_20260928/analysis/logits_consistency.json');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
