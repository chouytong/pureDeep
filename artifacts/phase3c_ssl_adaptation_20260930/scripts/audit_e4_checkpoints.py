"""Read-only checkpoint and loader audit for E4 new runs."""
import hashlib,json
from pathlib import Path
import pandas as pd
BASE=Path('/home/zyt/deep_final/artifacts/phase3c_ssl_adaptation_20260930')
rows=[]
for frac in (25,50,75):
 for variant in ('str','frozen'):
  for seed in (42,43,44):
   for outer in range(5):
    for inner in range(3):
     stage=BASE/f'runs/{variant}/p{frac}/seed{seed}/outer_{outer}/inner_{inner}'
     status=json.loads((stage/'stage_status.json').read_text())
     assert status['status']=='complete' and status['outer_test_loader_created'] is False
     cp=stage/'checkpoints/best.pt'
     assert hashlib.sha256(cp.read_bytes()).hexdigest()==status['summary']['checkpoint_sha256']
     rows.append(dict(fraction=frac,variant=variant,seed=seed,outer=outer,inner=inner,checkpoint_sha256=status['summary']['checkpoint_sha256'],outer_test_loader_created=False))
assert len(rows)==270
pd.DataFrame(rows).to_csv(BASE/'analysis/e4_checkpoint_integrity.csv',index=False)
print(json.dumps({'new_runs_verified':270,'checkpoint_sha_match':True,'outer_test_loader_created':False},indent=2))
