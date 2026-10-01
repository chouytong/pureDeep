import json,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent.parent
for seed in (42,43,44):
 (root/'e1_progress.json').write_text(json.dumps({'status':'running','seed':seed,'time':time.time()}))
 with (root/f'e1_seed{seed}.log').open('a') as log:
  rc=subprocess.run([sys.executable,str(root/'scripts/run_inner.py'),'--variant','adapt','--seed',str(seed),'--resume'],stdout=log,stderr=subprocess.STDOUT).returncode
 if rc:
  (root/'e1_progress.json').write_text(json.dumps({'status':'failed','seed':seed,'code':rc}));raise SystemExit(rc)
(root/'e1_progress.json').write_text(json.dumps({'status':'complete','time':time.time()}))
