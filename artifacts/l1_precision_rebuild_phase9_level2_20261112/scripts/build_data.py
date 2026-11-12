from common_l2 import *
from numeric_core import solve_many,certificate
from concurrent.futures import ProcessPoolExecutor
import os,resource,datetime

def atomic_bytes(path,producer):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():raise FileExistsError('Refuse overwrite '+str(path))
 temporary=path.with_name(path.name+f'.partial.{os.getpid()}')
 with temporary.open('xb') as f:producer(f);f.flush();os.fsync(f.fileno())
 temporary.rename(path)
def atomic_json(path,data):atomic_bytes(path,lambda f:f.write((json.dumps(clean(data),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()))

def build_one(task):
 r,config,core_sha,config_sha,producer_sha,raw_sha=task;beg=time.process_time();sid=r['subject_id'];activity=r['activity'];wrist=r['wrist'];root=PRIVATE/'processed_signals'/VERSION;certroot=PRIVATE/'numerical_certificates'/VERSION;out=root/activity/sid/f'{wrist}.npy';certfile=certroot/activity/sid/f'{wrist}.npz';marker=certroot/activity/sid/f'{wrist}.json';rawpath=Path(r['raw_path']);assert sha(rawpath)==raw_sha
 if marker.exists():
  saved=load(marker);assert saved['status']=='PASS' and saved['version']==VERSION and saved['raw_sha256']==raw_sha and saved['solver_source_sha256']==core_sha and saved['solver_config_sha256']==config_sha and saved['producer_source_sha256']==producer_sha and sha(out)==saved['output_sha256'] and sha(certfile)==saved['certificate_arrays_sha256'];saved['reused_complete_verified']=True;return saved
 if out.exists() or certfile.exists():raise RuntimeError('BLOCKED_PARTIAL_UNCOMMITTED_RECORD; preserve existing files')
 raw=np.loadtxt(rawpath,delimiter=',',dtype=float);assert raw.ndim==2 and raw.shape[1]==7 and np.isfinite(raw).all() and np.all(np.diff(raw[:,0])>0) and len(raw)==r['metadata_rows'];y=raw[:,1:4].T;opts={k:config[k] for k in ['max_newton_iterations','max_barrier_stages','barrier_reduction','line_search_steps','step_fraction']};trend,dual,certs=solve_many(y,opts)
 if not all(c['passed'] for c in certs):
  failed=PRIVATE/'manifests/failures'/f"{sid}_{activity}_{wrist}_{os.getpid()}.json";write(failed,dict(status='FAIL_CERTIFICATE',record=r,certificates=certs,source_sha256=core_sha,config_sha256=config_sha));raise RuntimeError('BLOCKED_AXIS_CERTIFICATE')
 residual=y-trend;identity=abs(raw[:,1:4].T-trend-residual).max();assert identity<=5e-13*(1+abs(y).max());processed=np.vstack([residual[:,48:],raw[48:,4:7].T]).astype(np.float32);assert processed.shape==(6,len(raw)-48) and np.isfinite(processed).all();assert np.array_equal(processed[3:],raw[48:,4:7].T.astype(np.float32))
 quant=abs(processed[:3].astype(float)-residual[:,48:]);bound=abs(np.spacing(processed[:3])).astype(float)+8*np.finfo(float).eps*(1+abs(residual).max());assert np.all(quant<=bound)
 # Independently recompute each certificate from the arrays that will be saved.
 for j in range(3):
  check=certificate(y[j],trend[j],dual[j],certs[j]['iterations'],'SAVED_ARRAY_RECHECK');assert check['passed'];certs[j].update(solver_source_sha256=core_sha,solver_config_sha256=config_sha,channel=CHANNELS[j],full_raw_length=len(raw))
 atomic_bytes(certfile,lambda f:np.savez_compressed(f,trend=trend,dual=np.stack(dual),solver_source_sha256=core_sha,solver_config_sha256=config_sha,raw_sha256=raw_sha,version=VERSION));atomic_bytes(out,lambda f:np.save(f,processed,allow_pickle=False));assert np.array_equal(np.load(out,allow_pickle=False),processed)
 result=dict(status='PASS',version=VERSION,subject_id=sid,activity=activity,wrist=wrist,label=r['label'],source_condition=r['source_condition'],raw_path=str(rawpath),raw_sha256=raw_sha,output_path=str(out),output_sha256=sha(out),input_length=len(raw),output_length=len(raw)-48,solver_source_sha256=core_sha,solver_config_sha256=config_sha,producer_source_sha256=producer_sha,certificate_arrays_path=str(certfile),certificate_arrays_sha256=sha(certfile),certificate_json_path=str(marker),axis_certificates=certs,quantization_max_abs=float(quant.max()),quantization_bound_pass=True,raw_trend_residual_identity_max=float(identity),gyro_exact=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),record_CPU_seconds=time.process_time()-beg,worker_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,reused_complete_verified=False)
 atomic_json(marker,result);return result

def main():
 pilot=load(HERE/'analysis/pilot_integrity.json');assert pilot['status']=='PASS_PILOT_AND_BUDGET';source=HERE/'scripts/numeric_core.py';cfgpath=HERE/'configs/solver_production_v1.json';assert sha(source)==pilot['production_source_sha256'] and sha(cfgpath)==pilot['production_config_sha256'];config=load(cfgpath);producer=sha(Path(__file__));freezefile=HERE/'configs/FULL_DATA_BUILD_FREEZE.json'
 if not freezefile.exists():freeze('FULL_DATA_BUILD',[Path(__file__),HERE/'scripts/common_l2.py',source,cfgpath,HERE/'PROTOCOL.md'],data_version=VERSION,expected_wrist_records=8580,classification_results_seen=False)
 else:
  for p,h in load(freezefile)['files'].items():assert sha(HERE/p)==h
 ledger=load(PRIVATE/'manifests/cpu_ledger.json');assert ledger['total_solver_CPU_seconds']+pilot['CPU_projection_seconds']<21600
 allowed=load(PRIVATE/'manifests/raw_allowlist.json');rs=allowed['records'];assert len(rs)==8580 and len({r['subject_id'] for r in rs})==390;protected=load(PRIVATE/'manifests/protected_assets.json')['sha256'];tasks=[(r,config,sha(source),sha(cfgpath),producer,protected[r['raw_path']]) for r in rs];rows=[];axis=[];summary=[];begin=time.perf_counter()
 with ProcessPoolExecutor(max_workers=4) as pool:
  for k,result in enumerate(pool.map(build_one,tasks,chunksize=4)):
   rows.append(result)
   for c in result['axis_certificates']:axis.append(dict(subject_id=result['subject_id'],activity=result['activity'],wrist=result['wrist'],**c))
   summary.append(dict(subject_id=result['subject_id'],activity=result['activity'],wrist=result['wrist'],record_CPU_seconds=result['record_CPU_seconds'],quantization_max_abs=result['quantization_max_abs'],worker_peak_RSS_KiB=result['worker_peak_RSS_KiB']))
   with (PRIVATE/'manifests/build_progress.jsonl').open('a') as f:f.write(json.dumps(dict(index=k,output_sha256=result['output_sha256'],certificate_json_path=result['certificate_json_path'],reused=result['reused_complete_verified']))+'\n')
   if (k+1)%220==0:print(json.dumps(dict(stage='build-data',records=k+1,total=8580,certificates=len(axis),wall_seconds=time.perf_counter()-begin)),flush=True)
 assert len(rows)==8580 and len(axis)==25740 and len({(r['subject_id'],r['activity'],r['wrist']) for r in rows})==8580 and all(c['passed'] for c in axis)
 manifest=PRIVATE/'manifests'/f'{VERSION}.json';content=dict(version=VERSION,status='COMPLETE_ALL_CERTIFICATES',subject_count=390,record_count=8580,axis_count=25740,solver_source_sha256=sha(source),solver_config_sha256=sha(cfgpath),producer_source_sha256=producer,records=rows)
 if manifest.exists():assert load(manifest)==clean(content)
 else:atomic_json(manifest,content)
 frame=pd.DataFrame(axis);frame.to_csv(PRIVATE/'numerical_certificates/all_axis_certificate_scalars.csv',index=False);pd.DataFrame(summary).to_csv(PRIVATE/'subject_diagnostics/build_record_costs.csv',index=False)
 public=frame.groupby(['activity','channel']).agg(axis_count=('passed','size'),certificate_pass_count=('passed','sum'),maximum_raw_gap=('raw_gap','max'),minimum_raw_gap=('raw_gap','min'),mean_CPU_seconds=('cpu_seconds','mean'),maximum_CPU_seconds=('cpu_seconds','max'),maximum_stationarity_residual=('stationarity_residual','max'),maximum_dual_violation=('dual_feasibility_violation','max'),negative_gap_roundoff_count=('negative_gap_in_roundoff','sum')).reset_index();public['data_version']=VERSION;public.to_csv(HERE/'analysis/numerical_certificate_summary.csv',index=False)
 status=dict(status='PASS_COMPLETE_CERTIFIED_DATA',version=VERSION,manifest_path=str(manifest),manifest_sha256=sha(manifest),wrist_records=8580,Acc_axes=25740,all_axis_certificates_pass=True,all_source_hashes_verified=True,all_output_shapes_and_gyro_exact=True,maximum_quantization_error=max(r['quantization_max_abs'] for r in rows),new_data_not_mixed_with_legacy=True,raw_target_full_length_before_trim=True,source_sha256=sha(source),config_sha256=sha(cfgpath),producer_sha256=producer,record_independent_stop=True,partial_records_reused=int(sum(r['reused_complete_verified'] for r in rows)),wall_seconds=time.perf_counter()-begin)
 write(HERE/'analysis/data_version_integrity.json',status);write(PRIVATE/'manifests/data_ready.json',status);print(json.dumps(clean(status)))
if __name__=='__main__':main()
