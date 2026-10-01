"""Prespecified Phase-3D E1 adapter then E2 gate full 45×2 matrix."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
progress=ROOT/'e1_e2_progress.json'
for variant in ('adapter','gate'):
 for seed in (42,43,44):
  label=f'{variant}_seed{seed}';progress.write_text(json.dumps({'status':'running','label':label,'time':time.time()}))
  with (ROOT/f'{label}.log').open('a') as log:
   rc=subprocess.run([sys.executable,str(ROOT/'scripts/run_inner.py'),'--variant',variant,'--seed',str(seed),'--resume'],stdout=log,stderr=subprocess.STDOUT).returncode
  if rc:
   progress.write_text(json.dumps({'status':'failed','label':label,'code':rc}));raise SystemExit(rc)
progress.write_text(json.dumps({'status':'complete','time':time.time()}))
