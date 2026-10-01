"""Conditional E4 full 45-unit masked adaptation + original WSSL-STR fitting."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
p=ROOT/'domain_progress.json'
for seed in (42,43,44):
 label=f'domain_seed{seed}';p.write_text(json.dumps({'status':'running','label':label,'time':time.time()}))
 with (ROOT/f'{label}.log').open('a') as log:
  rc=subprocess.run([sys.executable,str(ROOT/'scripts/run_domain.py'),'--seed',str(seed),'--resume'],stdout=log,stderr=subprocess.STDOUT).returncode
 if rc:
  p.write_text(json.dumps({'status':'failed','label':label,'code':rc}));raise SystemExit(rc)
p.write_text(json.dumps({'status':'complete','time':time.time()}))
