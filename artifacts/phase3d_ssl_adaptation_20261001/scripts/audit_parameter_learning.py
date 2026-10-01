"""Read-only audit that adapter and gate parameters changed in completed models."""
import json
from pathlib import Path
import numpy as np,pandas as pd,torch
BASE=Path('/home/zyt/deep_final/artifacts/phase3d_ssl_adaptation_20261001')
rows=[]
for variant in ('adapter','gate'):
 for seed in (42,43,44):
  for outer in range(5):
   for inner in range(3):
    stage=BASE/f'runs/{variant}/seed{seed}/outer_{outer}/inner_{inner}'
    status=json.loads((stage/'stage_status.json').read_text());assert status['status']=='complete' and status['outer_test_loader_created'] is False
    s=torch.load(stage/'checkpoints/best.pt',map_location='cpu',weights_only=False)['model_state']
    if variant=='adapter':
     weight=float(s['adapter.2.weight'].norm());bias=float(s['adapter.2.bias'].norm())
    else:
     weight=float(s['gate_net.2.weight'].norm());bias=float((s['gate_net.2.bias']-np.log(9.)).norm())
    rows.append(dict(variant=variant,seed=seed,outer=outer,inner=inner,best_epoch=status['summary']['best_epoch'],new_weight_l2=weight,new_bias_delta_l2=bias))
assert len(rows)==90 and all(r['new_weight_l2']>0 for r in rows)
pd.DataFrame(rows).to_csv(BASE/'analysis/parameter_learning_audit.csv',index=False)
print(json.dumps({'runs_verified':90,'all_new_parameter_weights_changed':True,'minimum_weight_l2':min(r['new_weight_l2'] for r in rows),'outer_test_loader_created':False},indent=2))
