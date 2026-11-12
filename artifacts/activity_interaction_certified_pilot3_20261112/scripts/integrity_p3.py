"""Close the fixed three-run pilot without re-reading excluded scientific arrays."""
from common_p3 import *
import ast,re,math

def main():
 assert load(HERE/'analysis/analysis_integrity.json')['status']=='PASS';count=guard();torch.set_num_threads(4)
 for name in ['PROTOCOL','EXPORT','CORRECTNESS','TRAIN3','ANALYSIS']:
  for p,h in load(HERE/'configs'/f'{name}_FREEZE.json')['files'].items():assert sha(p)==h
 assert load(HERE/'configs/PROTOCOL_FREEZE.json')['seeds']==SEEDS
 assert (HERE/'configs/PROTOCOL_FREEZE.json').stat().st_mtime<min(p.stat().st_mtime for p in (PRIVATE/'checkpoints').glob('seed*/started.json'))
 lines=load(PRIVATE/'manifests/training_lineage.json')['records'];assert len(lines)==3 and sorted(r['seed'] for r in lines)==SEEDS;params=pd.read_csv(HERE/'analysis/parameter_counts.csv');assert len(params)==3 and (params.new_trainable_parameters==67522).all() and (params.original_frozen_classifier_parameters==143172).all() and (params.unique_deployment_parameters==10668102).all()
 manifest=load(PRIVATE/'manifests/token_index.json')['packages'];assert len(manifest)==6;assert len(list((PRIVATE/'checkpoints').glob('seed*/started.json')))==3
 for r in lines:
  seed=r['seed'];assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['prediction'])==r['prediction_sha256'];stage=Path(r['checkpoint']).parent;assert sha(stage/'epochs.json')==r['epochs_sha256'];epochs=load(stage/'epochs.json');best=max(e['validation_BA'] for e in epochs);first=next(e['epoch'] for e in epochs if e['validation_BA']==best);cost=params[params.seed==seed].iloc[0];assert first==cost.best_epoch and len(epochs)==cost.training_epochs and 1<=len(epochs)<=50 and (len(epochs)==50 or len(epochs)-first==12);assert cost.optimizer_steps==len(epochs)*math.ceil(len(binding()['fit_subject_ids'])/8)
  ck=torch.load(r['checkpoint'],map_location='cpu',weights_only=False);base=torch.load(binding()['seeds'][str(seed)]['checkpoint'],map_location='cpu',weights_only=False);assert ck['data_version']==VERSION and ck['base_checkpoint_sha256']==binding()['seeds'][str(seed)]['checkpoint_sha256'] and ck['normalization_sha256']==binding()['seeds'][str(seed)]['normalization_sha256']
  assert all(torch.equal(v,ck['model_state']['base.'+k]) for k,v in base['model_state'].items());assert sum(v.numel() for k,v in ck['model_state'].items() if not k.startswith('base.'))==67522
  for role in ['train','validation']:d=token(seed,role);assert str(d['role'])==role
  d=token(seed,'validation')
  with np.load(r['prediction'],allow_pickle=False) as z:assert np.array_equal(z['subject_ids'],d['subject_ids']) and np.array_equal(z['target'],d['target']) and np.isfinite(z['logits']).all()
 for n in ['metrics_by_seed','paired_deltas_by_seed','mean_std_summary','parameter_counts','experiment_state']:
  frame=pd.read_csv(HERE/'analysis'/f'{n}.csv');assert len(frame)>0 and not any(re.match(r'^(subject_id|patient_id|participant_id|pair_id|raw_path|output_path)$',c) for c in frame.columns)
 frame=pd.read_csv(HERE/'analysis/metrics_by_seed.csv');assert len(frame)==6 and frame.groupby('condition').size().tolist()==[3,3]
 for n in ['FINAL_REPORT','CORRECTNESS_AND_PROVENANCE','COST_AND_LATENCY']:assert (HERE/'reports'/f'{n}.md').stat().st_size>500
 assert load(HERE/'tests/correctness_tests.json')['status']=='PASS'
 for p in (HERE/'scripts').glob('*.py'):ast.parse(p.read_text())
 ledger=load(PRIVATE/'manifests/stage_ledger.json');pd.DataFrame(ledger).drop(columns=['source_sha256','log']).to_csv(HERE/'analysis/stage_compute_summary.csv',index=False);failed=[dict(stage=r['stage'],attempt=r['attempt'],exit_code=r['exit_code']) for r in ledger if r['exit_code']!=0]
 private_manifest=PRIVATE/'manifests/new_assets_sha256.json';exclude={str(private_manifest),str(PRIVATE/'manifests/stage_ledger.json')};assets={str(p.relative_to(HERE)):sha(p) for p in sorted(PRIVATE.rglob('*')) if p.is_file() and str(p) not in exclude};write(private_manifest,dict(sha256=assets,mutable_operational_ledger_excluded=True))
 j=load(HERE/'analysis/pilot_judgment.json');result=dict(CORRECTNESS='PASS',PILOT_STATUS='COMPLETED',PERFORMANCE_PATTERN=j['PERFORMANCE_PATTERN'],formal_new_runs=3,context=0,inner=0,seeds=SEEDS,remaining42_runs_started=False,base_training=0,L1_solves=0,normalization_refits=0,HarNet_extraction=0,legacy_AI_weights_loaded=False,legacy_token_arrays_read=0,signal_forward_instances=load(HERE/'analysis/export_integrity.json')['subject_forward_instances'],token_role_packages=6,unique_train_subjects=len(binding()['fit_subject_ids']),unique_validation_subjects=len(binding()['query_subject_ids']),necessary_protected_files_unchanged=count,full_historical_audit_repeated=False,excluded_signal_cache_prediction_arrays_read=0,base_checkpoint_tensors_bitwise_unchanged_in_all3_new_checkpoints=True,new_private_asset_count=len(assets),private_manifest_sha256=sha(private_manifest),current_model_pointer_changed=False,ensemble_candidates=0,failed_stage_attempts=failed,publication='PENDING_EXTERNAL_EXACT_TREE_RECEIPT')
 write(HERE/'analysis/integrity.json',result);public=[]
 for p in sorted(HERE.rglob('*')):
  if p.is_file() and not set(p.relative_to(HERE).parts)&{'restricted_manifests','logs','__pycache__'} and p.name!='source_output_manifest.json':public.append(dict(path=str(p.relative_to(HERE)),sha256=sha(p),bytes=p.stat().st_size))
 write(HERE/'analysis/source_output_manifest.json',dict(files=public));assert guard()==count;print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
