"""Stage attempts and failures persist; never starts extra splits."""
from common_p3 import *
import argparse,subprocess,resource
MAP={'audit':'audit_p3.py','export':'export_p3.py','correctness-test':'tests_p3.py','train3':'train_p3.py','analyze':'analyze_p3.py','integrity':'integrity_p3.py','publish-dry-run':'publish_preview.py'}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=MAP);a=ap.parse_args();ledger=PRIVATE/'manifests/stage_ledger.json';history=load(ledger) if ledger.exists() else [];attempt=1+sum(r['stage']==a.stage for r in history);path=HERE/'scripts'/MAP[a.stage];log=HERE/'logs'/f'{a.stage}_attempt{attempt}.log';log.parent.mkdir(exist_ok=True);before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.perf_counter()
 with log.open('x') as f:
  p=subprocess.Popen([sys.executable,'-B',str(path)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
  for line in p.stdout:f.write(line);f.flush();print(line,end='',flush=True)
  code=p.wait()
 after=resource.getrusage(resource.RUSAGE_CHILDREN);record=dict(stage=a.stage,attempt=attempt,exit_code=code,CPU_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,wall_seconds=time.perf_counter()-start,source_sha256=sha(path),log=str(log.relative_to(HERE)));history.append(record);write(ledger,history);print(json.dumps(record),flush=True)
 if code:raise SystemExit(code)
if __name__=='__main__':main()
