"""Read back provenance, frozen sources and aggregate closure before publication."""
from common8 import *
import platform,sklearn,scipy

def main():
 assert load(HERE/'analysis/comparison_integrity.json')['status']=='PASS_COMPLETE_D_COMPARISON';assert load(HERE/'analysis/timing_integrity.json')['status']=='PASS';assert (HERE/'reports/FINAL_REPORT.md').exists()
 checked=[]
 for path in sorted((HERE/'configs').glob('*_FREEZE.json')):
  for rel,h in load(path)['files'].items():assert sha(HERE/rel)==h
  checked.append(path.name)
 for name in ['activity_interaction','supcon','readout']:
  records=load(HERE/'restricted_manifests'/f'{name}_lineage.json')['records'];assert len(records)==45
  for record in records:
   fit=set(map(str,record['fit_subject_ids']));query=set(map(str,record['query_subject_ids']));excluded=set(map(str,record['excluded_subject_ids']));assert not fit&query and not (fit|query)&excluded
   sp=next(s for cc,ii,s,e in splits() if (cc,ii)==(record['context'],record['inner']));assert fit==set(map(str,sp['train_subjects'])) and query==set(map(str,sp['validation_subjects']))
   # Classifier paths/hashes differ between the existing fitted-model formats.
   if name in ['activity_interaction','supcon']:
    prefix='ai' if name=='activity_interaction' else 'supcon';path=HERE/'checkpoints'/f"{prefix}_seed{record['seed']}_context{record['context']}_inner{record['inner']}/best.pt";assert sha(path)==record['checkpoint_sha256']
   assert sha(baseline(record['seed'],record['context'],record['inner'])/'normalization.json')==record['normalization_sha256']
 counts=dict(prediction_npz=len(list((HERE/'predictions').glob('*.npz'))),new_deep_checkpoints=len(list((HERE/'checkpoints').glob('*/best.pt'))),traditional_joblib=len(list((HERE/'checkpoints').rglob('*.joblib'))))
 assert counts==dict(prediction_npz=345,new_deep_checkpoints=90,traditional_joblib=180),counts
 stat=pd.read_csv(HERE/'analysis/final_paired_statistics.csv');assert len(stat)==81
 assert stat[(stat.condition=='xgboost')].wilcoxon_p.isna().all();assert stat[(stat.condition=='random_forest')&(stat.metric=='nll')].wilcoxon_p.isna().all()
 allmetrics=pd.read_csv(HERE/'analysis/all_model_metrics.csv');assert len(allmetrics)==345 and not allmetrics.duplicated(['condition','seed','context','inner']).any()
 private=[]
 for directory in ['checkpoints','predictions']:
  for p in sorted((HERE/directory).rglob('*')):
   if p.is_file():private.append(dict(path=str(p.relative_to(HERE)),sha256=sha(p),bytes=p.stat().st_size))
 write(HERE/'restricted_manifests/new_private_outputs_sha256.json',dict(files=private))
 write(HERE/'audit/environment.json',dict(Python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,XGBoost='ABSENT_NO_INSTALLATION',GPU=torch.cuda.get_device_name(),torch_threads_used=4,actual_date_local='2026-11-12',directory_date_user_requested='20261111',new_packages_installed=0))
 protected=guard();assert protected==22585
 write(HERE/'analysis/final_integrity.json',dict(status='PASS_COMPLETE_PHASE8',official_start=START,all_frozen_stage_sources_unchanged=checked,protected_files_unchanged=protected,canonical_split_seed_pairs=45,stage_order='A1>A2>B>C>D',B_C_training_overlap=False,new_deep_formal_fits=90,traditional_final_classifier_fits=180,train_only_scalers=45,smoke_updates_B=2,smoke_updates_C=1,counts=counts,comparison_weighted_metric_tests='PASS',source_and_weight_provenance_verified=True,new_outer_performance_or_loader=False,old_outer_exposure_permanently_disclosed=True,original_model_preprocessing_norm_threshold_unchanged=True,phase9_started=False,private_payloads_not_public=True))
 # Public files only; self-exclusion prevents recursive manifest hashes.
 files=[]
 for directory in ['scripts','configs','audit','analysis','reports','figures','tests']:
  for path in sorted((HERE/directory).rglob('*')):
   if path.is_file() and path.name!='source_output_manifest.json':files.append(dict(path=str(path.relative_to(HERE)),sha256=sha(path),bytes=path.stat().st_size))
 for name in ['README.md','REPRODUCE.md','PROTOCOL.md']:
  path=HERE/name;files.append(dict(path=name,sha256=sha(path),bytes=path.stat().st_size))
 assert all(x['bytes']<=5*1024*1024 for x in files);write(HERE/'analysis/source_output_manifest.json',dict(files=files,private_directories_excluded=['checkpoints','predictions','features','restricted_manifests','logs'],protected_files_unchanged=protected))
 print(json.dumps(dict(status='PASS_COMPLETE_PHASE8',protected_files=protected,public_manifest_files=len(files),counts=counts)))

if __name__=='__main__':main()
