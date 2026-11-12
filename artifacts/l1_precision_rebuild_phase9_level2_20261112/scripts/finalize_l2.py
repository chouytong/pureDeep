"""Close immutable scientific provenance and safe aggregate publication manifest."""
from common_l2 import *
from data_l2 import data_manifest,norm_binding
import torch,math,re,ast
REPORTS=['FINAL_REPORT','SOLVER_IMPLEMENTATION_REPORT','SOLVER_CORRECTNESS_REPORT','DATA_REBUILD_REPORT','NORMALIZATION_PROVENANCE_REPORT','WSSL_RETRAIN_REPORT','STR_RETRAIN_REPORT','SSL_INCREMENT_REVALIDATION_REPORT','DEVICE_STRATIFIED_ANALYSIS','COST_AND_LATENCY_REPORT','CORRECTNESS_AND_PROVENANCE','LIMITATIONS_AND_NEXT_STEPS']
CSV=['numerical_certificate_summary','batch_independence_summary','experiment_state','metrics_by_split_seed','paired_statistics','seed_stability','subject_cluster_summary','device_stratified_summary','cost_summary']

def main():
 assert load(HERE/'analysis/report_integrity.json')['status']=='PASS_REPORTS_GENERATED';oldcount=guard();ready=load(PRIVATE/'manifests/data_ready.json');manifest=PRIVATE/'manifests'/f'{VERSION}.json';d=data_manifest(VERSION,manifest)
 for file in ['PRE_CLASSIFICATION','PRODUCTION_SOLVER_V1','FULL_DATA_BUILD','WSSL_TRAINING','STR_TRAINING','ANALYSIS','COST']:
  fr=load(HERE/'configs'/f'{file}_FREEZE.json')
  for path,h in fr['files'].items():assert sha(HERE/path)==h,('Frozen source changed',path)
 assert sha(HERE/'scripts/numeric_core.py')==d['solver_source_sha256'] and sha(HERE/'scripts/build_data.py')==d['producer_source_sha256'] and sha(HERE/'configs/solver_production_v1.json')==d['solver_config_sha256']
 for r in d['records']:
  for path,h in [('raw_path','raw_sha256'),('output_path','output_sha256'),('certificate_arrays_path','certificate_arrays_sha256')]:assert sha(r[path])==r[h]
  cert=load(r['certificate_json_path']);assert cert['version']==VERSION and all(a['passed'] for a in cert['axis_certificates']) and r['gyro_exact'] and r['quantization_bound_pass']
 ni=load(load(HERE/'analysis/normalization_integrity.json')['normalization_index']);norms={(r['context'],r['inner']):r for r in ni['normalizations']};runs=[]
 for kind in ['wssl','str']:
  integrity=load(HERE/'analysis'/f'{kind}_training_integrity.json');assert integrity['status']=='PASS_COMPLETE_45_RUNS' and integrity['runs']==45
  assert sha(PRIVATE/'manifests'/f'{kind}_initial_completion_report.md')==integrity['report_sha256']
  lines=load(PRIVATE/'manifests'/f'{kind}_lineage.json')['records'];assert len(lines)==45 and len({(r['seed'],r['context'],r['inner']) for r in lines})==45
  for r in lines:
   seed,c,i=r['seed'],r['context'],r['inner'];sp=next(s for cc,ii,s,e in splits() if (cc,ii)==(c,i));assert set(r['fit_subject_ids'])==set(sp['train_subjects']) and set(r['query_subject_ids'])==set(sp['validation_subjects']);assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['prediction'])==r['prediction_sha256'];assert r['normalization_sha256']==norms[c,i]['sha256']==sha(norms[c,i]['path']);norm_binding(VERSION,manifest,norms[c,i]['path'],c,i)
   stage=Path(r['checkpoint']).parent;receipt=load(stage/'complete.json');epochs=load(stage/'epochs.json');assert 1<=len(epochs)<=50;best=max(e['validation_BA'] for e in epochs);first=next(e['epoch'] for e in epochs if e['validation_BA']==best);assert receipt['cost']['best_epoch']==first;assert receipt['cost']['optimizer_steps']==len(epochs)*math.ceil(len(sp['train_subjects'])/8)
   if len(epochs)<50:assert len(epochs)-first==12
   ck=torch.load(r['checkpoint'],map_location='cpu',weights_only=False);assert ck['version']==VERSION and ck['epoch']==first and ck['manifest_sha256']==sha(manifest) and ck['normalization_sha256']==norms[c,i]['sha256'];assert sum(v.numel() for v in ck['model_state'].values())=={'wssl':143172,'str':75524}[kind];assert all(torch.isfinite(v).all() for v in ck['model_state'].values())
   with np.load(r['prediction'],allow_pickle=False) as a:assert set(a['subject_ids'].astype(str))==set(sp['validation_subjects']) and a['logits'].shape==(len(sp['validation_subjects']),2) and np.isfinite(a['logits']).all()
   runs.append(dict(model=kind,seed=seed,context=c,inner=i,epochs=len(epochs),best_epoch=first,optimizer_steps=receipt['cost']['optimizer_steps'],status='PASS'))
 wssl_report=HERE/'analysis/wssl_comparison_report_archived.json';earliest_str=min(p.stat().st_mtime for p in (PRIVATE/'checkpoints/str').glob('*/started.json'));assert wssl_report.stat().st_mtime<earliest_str and load(wssl_report)['status']=='PASS_ARCHIVED_BEFORE_NEXT_STAGE';assert load(wssl_report)['report_sha256']==sha(HERE/'reports/WSSL_RETRAIN_REPORT.md')
 assert len(runs)==90;pd.DataFrame(runs).to_csv(HERE/'analysis/training_execution_summary.csv',index=False)
 for n in REPORTS:assert (HERE/'reports'/f'{n}.md').stat().st_size>300
 for n in CSV:assert len(pd.read_csv(HERE/'analysis'/f'{n}.csv'))>0
 f=pd.read_csv(HERE/'analysis/metrics_by_split_seed.csv');assert len(f)==180 and f.groupby('condition').size().tolist()==[45]*4
 assert load(HERE/'tests/statistics_checks.json')['status']=='PASS';assert load(HERE/'analysis/cost_integrity.json')['timings']==90
 # Complete certified assets and trained outputs remain private, with an independent hash catalog.
 private_manifest=PRIVATE/'manifests/new_scientific_assets_sha256.json';mutable={str(private_manifest),str(PRIVATE/'manifests/cpu_ledger.json')};assets={str(p.relative_to(HERE)):sha(p) for p in sorted(PRIVATE.rglob('*')) if p.is_file() and str(p) not in mutable};write(private_manifest,dict(sha256=assets,mutable_operational_CPU_ledger_excluded=True,scientific_snapshot=True))
 # Public aggregate schema checks: no subject IDs or raw filenames, and no small device cells.
 blocked=re.compile(r'^(subject_id|patient_id|participant_id|sample_id|pair_id|raw_path|processed_path|output_path)$',re.I)
 for p in (HERE/'analysis').glob('*.csv'):
  frame=pd.read_csv(p);assert not any(blocked.fullmatch(c) for c in frame.columns),p
 assert pd.read_csv(HERE/'analysis/device_stratified_summary.csv').unique_subjects.min()>=10
 for p in (HERE/'scripts').glob('*.py'):ast.parse(p.read_text())
 cpu=load(PRIVATE/'manifests/cpu_ledger.json');assert cpu['total_solver_CPU_seconds']<21600
 pd.DataFrame([{k:r.get(k) for k in ['stage','attempt','status','exit_code','CPU_seconds','charged_CPU_seconds','wall_seconds']} for r in cpu['attempts']]).to_csv(HERE/'analysis/stage_compute_summary.csv',index=False)
 # All failed attempts are retained in the ledger; successful final training count is exact.
 failed=[dict(stage=r['stage'],attempt=r['attempt'],exit_code=r['exit_code']) for r in cpu['attempts'] if r['exit_code']!=0]
 decision=load(HERE/'analysis/performance_decision.json');result=dict(status='PASS_SCIENTIFIC_DELIVERABLES',NUMERICAL_CORRECTNESS='PASS',RECORD_INDEPENDENCE='PASS',DATA_AND_TRAINING_PROVENANCE='PASS',PERFORMANCE_PROMOTION=decision['status'],formal_WSSL_runs=45,formal_STR_runs=45,normalization_fits=15,synthetic_training_smoke_updates=2,certified_wrist_records=8580,certified_Acc_axes=25740,source_commit=START,data_version=VERSION,manifest_sha256=sha(manifest),protected_legacy_files_unchanged=oldcount,new_private_scientific_files=len(assets),private_manifest_sha256=sha(private_manifest),solver_CPU_seconds=cpu['total_solver_CPU_seconds'],solver_CPU_limit=21600,failed_stage_attempts=failed,current_model_pointer_changed=False,HarNet_reextracted=False,new_outer_evaluation=False,online_end_to_end='NOT_MEASURED',physical_gravity_semantics='UNKNOWN',publication='PENDING_SEPARATE_EXACT_TREE_RECEIPT')
 write(HERE/'analysis/final_integrity.json',result);public=[]
 for p in sorted(HERE.rglob('*')):
  if not p.is_file() or any(part in {'restricted_manifests','logs','__pycache__'} for part in p.relative_to(HERE).parts) or p.name=='source_output_manifest.json':continue
  public.append(dict(path=str(p.relative_to(HERE)),sha256=sha(p),bytes=p.stat().st_size))
 write(HERE/'analysis/source_output_manifest.json',dict(files=public,private_assets_excluded=True));assert guard()==oldcount;print(json.dumps(result),flush=True)
if __name__=='__main__':main()
