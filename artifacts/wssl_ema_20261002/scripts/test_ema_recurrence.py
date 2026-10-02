import sys,json
from pathlib import Path
import torch
H=Path(__file__).resolve().parent.parent;F=Path('/home/zyt/deep_final/foundation_validation');B=Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928');sys.path[:0]=[str(F),str(B/'scripts')]
from src.utils.config import load_config
from independent_wssl import IndependentWSSL
from ema_hooks import EMA
cfg=load_config(str(F/'configs/str01_seed42.yaml'));model=IndependentWSSL(cfg).cuda();initial={n:p.detach().clone() for n,p in model.named_parameters()};ema=EMA(model,cfg,26)
with torch.no_grad():
 for p in model.parameters():p.add_(.1)
 for _ in range(26):ema.update(model)
 errors=[float((p-(initial[n]+.05)).abs().max()) for n,p in ema.model.named_parameters()]
assert max(errors)<1e-5
assert all(p.grad is None and not p.requires_grad for p in ema.model.parameters())
assert all(p.grad is None for p in model.parameters())
result=dict(status='PASS',steps=26,alpha=ema.alpha,one_epoch_half_life_max_error=max(errors),no_grad_feedback=True,ema_parameters_frozen=True)
(H/'analysis/ema_recurrence_test.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
