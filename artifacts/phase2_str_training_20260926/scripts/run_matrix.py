#!/usr/bin/env python3
"""Execute registered Phase-2 inner variants, at most three seed processes per variant."""
import argparse,concurrent.futures,json,subprocess,sys,time
from pathlib import Path
B=Path('/home/zyt/deep_final/artifacts/phase2_str_training_20260926');RUN=B/'scripts/run_inner.py';PY='/home/zyt/envs/mfam/bin/python';MAN=B/'manifest.json'
def one(variant,seed):
 log=B/'logs'/variant/f'seed{seed}.log';log.parent.mkdir(parents=True,exist_ok=True)
 with log.open('a') as stream:
  p=subprocess.run([PY,str(RUN),'--manifest',str(MAN),'--variant',variant,'--seed',str(seed),'--resume'],cwd='/home/zyt/deep_final/foundation_validation',stdout=stream,stderr=subprocess.STDOUT)
 return variant,seed,p.returncode,str(log)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('variants',nargs='+');args=ap.parse_args();registered=json.loads(MAN.read_text())['variants']
 for variant in args.variants:
  assert variant in registered and variant!='baseline';t=time.time();print('START',variant,flush=True)
  with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:results=list(ex.map(lambda s:one(variant,s),(42,43,44)))
  print('END',variant,'seconds',round(time.time()-t,1),'results',results,flush=True)
  if any(r[2] for r in results):raise SystemExit(f'variant failed: {variant}')
if __name__=='__main__':main()
