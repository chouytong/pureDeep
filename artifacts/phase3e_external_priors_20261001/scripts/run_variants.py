import sys,subprocess,json,time,os
from pathlib import Path
B=Path(__file__).resolve().parents[1];env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8');log=B/f'driver_{sys.argv[1]}.jsonl'
for variant in sys.argv[1:]:
 for seed in (42,43,44):
  p=B/f'train_{variant}_{seed}.log';t=time.time()
  with p.open('a') as f:r=subprocess.run([sys.executable,str(B/'scripts/run_inner.py'),'--variant',variant,'--seed',str(seed),'--resume'],stdout=f,stderr=subprocess.STDOUT,env=env)
  record={'variant':variant,'seed':seed,'seconds':time.time()-t,'exit':r.returncode};print(record,flush=True)
  with log.open('a') as f:f.write(json.dumps(record)+'\n')
  if r.returncode:raise SystemExit(r.returncode)
