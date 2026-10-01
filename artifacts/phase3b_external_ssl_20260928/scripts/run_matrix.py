#!/usr/bin/env python3
"""Run only the prespecified Phase-3B inner development matrix."""
import json,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent.parent
progress=root/'matrix_progress.json'
for variant in ('a_ssl_only_pretrained','b_str_pretrained','c_str_random'):
 for seed in (42,43,44):
  cmd=[sys.executable,str(root/'scripts/run_inner.py'),'--manifest',str(root/'manifest.json'),'--variant',variant,'--seed',str(seed),'--resume']
  log=root/f'run_{variant}_seed{seed}.log'
  progress.write_text(json.dumps({'status':'running','variant':variant,'seed':seed,'started_unix':time.time()}))
  with log.open('a') as stream: code=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT).returncode
  if code:
   progress.write_text(json.dumps({'status':'failed','variant':variant,'seed':seed,'exit_code':code}))
   raise SystemExit(code)
  progress.write_text(json.dumps({'status':'completed_run','variant':variant,'seed':seed,'completed_unix':time.time()}))
progress.write_text(json.dumps({'status':'complete','completed_unix':time.time()}))
