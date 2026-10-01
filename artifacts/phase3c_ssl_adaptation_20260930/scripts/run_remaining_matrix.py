"""Prespecified sequential E2, E3, E4 GPU workload after completed E1."""
import json,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent.parent
assert json.loads((root/'e1_progress.json').read_text())['status']=='complete'
progress=root/'remaining_progress.json'
def run(cmd,log,label):
 progress.write_text(json.dumps({'status':'running','label':label,'time':time.time()}))
 with (root/log).open('a') as f:rc=subprocess.run([sys.executable,str(root/'scripts'/cmd[0])]+cmd[1:],stdout=f,stderr=subprocess.STDOUT).returncode
 if rc:
  progress.write_text(json.dumps({'status':'failed','label':label,'code':rc}));raise SystemExit(rc)
for seed in (42,43,44):run(['run_oof.py','--seed',str(seed),'--resume'],f'e2_oof_seed{seed}.log',f'e2_seed{seed}')
for seed in (42,43,44):run(['run_inner.py','--variant','multi','--seed',str(seed),'--resume'],f'e3_multi_seed{seed}.log',f'e3_seed{seed}')
for frac in (25,50,75):
 for variant in ('str','frozen'):
  for seed in (42,43,44):run(['run_inner.py','--variant',variant,'--fraction',str(frac),'--seed',str(seed),'--resume'],f'e4_{variant}_p{frac}_seed{seed}.log',f'e4_{variant}_p{frac}_seed{seed}')
progress.write_text(json.dumps({'status':'complete','time':time.time()}))
