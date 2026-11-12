"""Stage wrapper with process-tree CPU accounting and immutable attempt logs."""
from common_l2 import *
import argparse,subprocess,resource,os,signal,threading

MAP={'audit':'audit_l2.py','solver-test':'solver_tests_v2.py','pilot':'pilot_l2.py','build-data':'build_data.py','fit-normalization':'fit_norm.py','train-wssl':'train_l2.py','train-str':'train_l2.py','analyze':'analyze_l2.py','integrity-check':'finalize_l2.py','publish-dry-run':'publish_preview_l2.py'}
CHARGED={'solver-test','pilot','build-data'}
def tree_cpu(pid):
 total=0.
 try:
  raw=Path(f'/proc/{pid}/stat').read_text();v=raw[raw.rfind(')')+2:].split();total=sum(float(v[j]) for j in [11,12,13,14])/os.sysconf('SC_CLK_TCK');children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
  for child in children:total+=tree_cpu(int(child))
 except (FileNotFoundError,ProcessLookupError):pass
 return total
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=MAP);ap.add_argument('arguments',nargs=argparse.REMAINDER);args=ap.parse_args();ledger=PRIVATE/'manifests/cpu_ledger.json';history=load(ledger)['attempts'] if ledger.exists() else [];spent=sum(v['charged_CPU_seconds'] for v in history);charged=args.stage in CHARGED
 if args.stage=='build-data' and (PRIVATE/'manifests/data_ready.json').exists():
  started=time.process_time();ready=load(PRIVATE/'manifests/data_ready.json');manifest=Path(ready['manifest_path']) if 'manifest_path' in ready else PRIVATE/'manifests'/f'{VERSION}.json';assert sha(manifest)==ready['manifest_sha256'];data=load(manifest);assert len(data['records'])==8580 and data['axis_count']==25740 and data['status']=='COMPLETE_ALL_CERTIFICATES'
  assert sha(HERE/'scripts/numeric_core.py')==data['solver_source_sha256'] and sha(HERE/'configs/solver_production_v1.json')==data['solver_config_sha256']
  for r in data['records']:
   for pk,hk in [('raw_path','raw_sha256'),('output_path','output_sha256'),('certificate_arrays_path','certificate_arrays_sha256')]:assert sha(r[pk])==r[hk]
   assert r['status']=='PASS' and all(c['passed'] for c in r['axis_certificates']) and load(r['certificate_json_path'])['version']==VERSION
  cpu=time.process_time()-started;history.append(dict(stage='build-data',attempt=1+sum(v['stage']=='build-data' for v in history),status='VERIFIED_COMPLETE_REUSE',CPU_seconds=cpu,charged_CPU_seconds=cpu,exit_code=0,source_sha256=sha(Path(__file__)),command=sys.argv));write(ledger,dict(attempts=history,total_solver_CPU_seconds=sum(v['charged_CPU_seconds'] for v in history),limit_CPU_seconds=21600));print('PASS_VERIFIED_COMPLETE_DATA_REUSE');return
 if charged and spent>=21600:raise RuntimeError('BLOCKED_COMPUTE_BUDGET')
 path=HERE/'scripts'/MAP[args.stage]
 if args.stage=='solver-test' and '--verify-data' in args.arguments:path=HERE/'scripts/data_verify_l2.py';args.arguments.remove('--verify-data')
 if args.stage=='solver-test' and '--quantization-impact' in args.arguments:path=HERE/'scripts/quantization_l2.py';args.arguments.remove('--quantization-impact')
 if args.stage=='analyze' and '--resume-reports' in args.arguments:path=HERE/'scripts/finish_analysis_l2.py';args.arguments.remove('--resume-reports')
 assert path.exists();attempt=1+sum(v['stage']==args.stage for v in history);log=HERE/'logs'/f'{args.stage}_attempt{attempt}.log';assert not log.exists();before=resource.getrusage(resource.RUSAGE_CHILDREN);selfcpu=time.process_time();beg=time.perf_counter();env=dict(os.environ)
 if charged:env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 extra=(['--model-type','wssl' if args.stage=='train-wssl' else 'str'] if args.stage.startswith('train-') else []);cmd=[sys.executable,'-B',str(path)]+extra+args.arguments
 with log.open('x') as f:
  child=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env,start_new_session=True)
  def pump():
   for line in child.stdout:f.write(line);f.flush();print(line,end='',flush=True)
  thread=threading.Thread(target=pump);thread.start();blocked=False
  while child.poll() is None:
   if charged and spent+tree_cpu(child.pid)+(time.process_time()-selfcpu)>=21598:os.killpg(child.pid,signal.SIGTERM);blocked=True;break
   time.sleep(.5)
  code=child.wait();thread.join()
 after=resource.getrusage(resource.RUSAGE_CHILDREN);cpu=(after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime)+(time.process_time()-selfcpu);history.append(dict(stage=args.stage,attempt=attempt,command=cmd,source_sha256=sha(path),exit_code=code,CPU_seconds=cpu,charged_CPU_seconds=cpu if charged else 0.,wall_seconds=time.perf_counter()-beg,status='BLOCKED_COMPUTE_BUDGET' if blocked else 'COMPLETE' if code==0 else 'FAILED',log=str(log.relative_to(HERE))));write(ledger,dict(attempts=history,total_solver_CPU_seconds=sum(v['charged_CPU_seconds'] for v in history),limit_CPU_seconds=21600));print(json.dumps(dict(stage=args.stage,exit_code=code,charged_CPU_total=sum(v['charged_CPU_seconds'] for v in history),CPU_seconds=cpu,wall_seconds=time.perf_counter()-beg)))
 if code!=0 or blocked:raise SystemExit(code or 2)
if __name__=='__main__':main()
