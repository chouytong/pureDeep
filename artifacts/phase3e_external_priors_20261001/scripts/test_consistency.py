import sys,copy,json
from pathlib import Path
import torch
B=Path(__file__).resolve().parents[1];R=Path('/home/zyt/deep_final/foundation_validation');P=Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928');sys.path[:0]=[str(B/'scripts'),str(R),str(P/'scripts')]
from src.utils.config import load_config
from src.engine import nested_training as nt
import phase3e_hooks as h
from transfer_hooks import ResidualSSLSubject
c=load_config(str(R/'configs/str01_seed42.yaml'));torch.manual_seed(42);b=nt.build_model(c);x=torch.randn(1,11,2,6,64);wm=torch.ones(1,11,2,dtype=torch.bool);am=torch.ones(1,11,dtype=torch.bool);le=torch.full((1,11),64,dtype=torch.long);b.eval();out={};counts={}
for dim in (512,384):
 h.FEATURE_DIM=dim;m=h.ResidualSSLSubject(copy.deepcopy(b)).eval();m.ssl_features=torch.randn(1,11,2,dim)
 with torch.inference_mode():d=(m(x,wm,am,le)['logits']-b(x,wm,am,le)['logits']).abs().max().item()
 assert d==0;out[str(dim)]=d;counts[str(dim)]=sum(p.numel() for p in m.parameters());sm=h.SSLOnlySubject();counts['ssl_only_'+str(dim)]=sum(p.numel() for p in sm.parameters())
h.FEATURE_DIM=1024;a=ResidualSSLSubject(nt.build_model(c)).eval();m=h.ResidualSSLSubject(nt.build_model(c)).eval();state=torch.load(P/'runs/b_str_pretrained/seed42/outer_0/inner_0/checkpoints/best.pt',map_location='cpu',weights_only=False)['model_state'];a.load_state_dict(state);m.load_state_dict(state);a.ssl_features=m.ssl_features=torch.randn(1,11,2,1024)
with torch.inference_mode():d=(a(x,wm,am,le)['logits']-m(x,wm,am,le)['logits']).abs().max().item()
assert d==0;out['loaded_frozen_checkpoint']=d
z={'logit_max_differences':out,'classifier_trainable_counts':counts,'status':'PASS'};(B/'analysis/consistency.json').write_text(json.dumps(z,indent=2)+'\n');print(z)
