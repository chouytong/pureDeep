from common_l2 import *
import subprocess,ast,platform,torch,yaml,scipy

def main():
 for name in ['reports','tests','configs','audit','analysis','logs','figures','restricted_manifests/manifests','restricted_manifests/numerical_certificates','restricted_manifests/processed_signals','restricted_manifests/normalization','restricted_manifests/checkpoints','restricted_manifests/predictions','restricted_manifests/subject_diagnostics']:(HERE/name).mkdir(parents=True,exist_ok=True)
 assert not (PRIVATE/'manifests/protected_assets.json').exists()
 def git(*args):return subprocess.check_output(['git','-C','/home/zyt/pureDeep-github-upload',*args],text=True).strip()
 current=git('rev-parse','HEAD');remote=git('rev-parse','origin/main');assert not git('status','--porcelain') and current==remote
 if current!=START:
  changed=git('diff','--name-only',START,current).splitlines();write(HERE/'audit/later_commits.json',dict(reference=START,current=current,changed=changed));raise RuntimeError('BLOCKED_VERSION_REVIEW: later commits require source-specific review')
 protected=dict(load(P9/'restricted_manifests/protected_assets.json')['sha256'])
 for p,h in protected.items():assert sha(p)==h
 docs=[ROOT/'AGENTS.md',MAIN/'README.md',ROOT/'CONTEXT_HANDOFF.md',ROOT/'conclusions/README.md',ROOT/'conclusions/CURRENT_MODEL.json',ROOT/'conclusions/EXPERIMENT_REPORT_INDEX.csv']+[P9/n for n in ['FINAL_REPORT.md','L1_SOLVER_CORRECTNESS_REPORT.md','LEVEL2_EXPERIMENT_PROTOCOL.md','DATA_PIPELINE_MAP.md','CORRECTNESS_AND_PROVENANCE.md','ACCELERATION_SEMANTICS_ADDENDUM.md','SENSOR_COORDINATE_AND_UNIT_AUDIT.md','FILTER_AND_SPECTRAL_AUDIT.md']]
 read=[]
 for p in docs:assert p.exists();p.read_text();protected[str(p)]=sha(p);read.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
 pd.DataFrame(read).to_csv(HERE/'audit/required_materials.csv',index=False)
 for directory in [P9,MAIN]:
  for p in directory.rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts:protected[str(p)]=sha(p)
 sources=[MAIN/'src/datasets/preprocessing.py',ROOT/'foundation_validation/scripts/prepare_multi_activity_data_v2.py',MAIN/'src/datasets/builders.py',MAIN/'src/datasets/folds.py',MAIN/'src/datasets/manifest.py',MAIN/'src/datasets/subject_activity.py',MAIN/'src/models/pure_deep.py',MAIN/'src/models/wssl.py',MAIN/'src/engine/nested_training.py',MAIN/'src/engine/runner.py',MAIN/'src/utils/seed.py',P6/'scripts/independent_wssl.py',P6/'scripts/external_inventory.py']
 src=[]
 for p in sources:
  text=p.read_text();ast.parse(text);protected[str(p)]=sha(p);src.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
 pd.DataFrame(src).to_csv(HERE/'audit/current_source_inputs.csv',index=False)
 cfg=yaml.safe_load((MAIN/'configs/str01_seed42.yaml').read_text());oldprocessing=load(OLD/'processing_config.json');assert oldprocessing['accelerometer']['lambda']==50 and oldprocessing['accelerometer']['rho']==40 and oldprocessing['accelerometer']['max_iterations']==5000 and oldprocessing['trim_start_samples']==48
 allids=set();splitrows=[]
 for c,i,sp,ex in splits():
  fit=set(map(str,sp['train_subjects']));query=set(map(str,sp['validation_subjects']));assert not fit&query and not (fit|query)&set(ex);allids|=fit|query;splitrows.append(dict(context=c,inner=i,train_count=len(fit),query_count=len(query),excluded_count=len(ex),within_split_overlap=0))
 assert len(allids)==390;rs=records();assert {r['subject_id'] for r in rs}==allids and len(rs)==8580
 assert all(r['metadata_units']==['s']+['g']*3+['rad/s']*3 and r['metadata_channels']==['Time','Accelerometer_X','Accelerometer_Y','Accelerometer_Z','Gyroscope_X','Gyroscope_Y','Gyroscope_Z'] for r in rs)
 legacy=load(P9/'analysis/solver_objective_witness_integrity.json');assert legacy['flagged_axis_records']==20173 and legacy['real_input_output_creation_hashes_exact']==8580
 recipe=[]
 for model in ['wssl','str']:
  for c,i,sp,ex in splits():
   for seed in [42,43,44]:
    base=baseline(model,seed,c,i);ck=torch.load(base/'checkpoints/best.pt',map_location='cpu',weights_only=False);oldcfg=ck['config'];newcfg=yaml.safe_load((MAIN/f'configs/str01_seed{seed}.yaml').read_text());assert oldcfg['model']==newcfg['model'] and oldcfg['training']==newcfg['training'] and oldcfg['evaluation']==newcfg['evaluation'];assert oldcfg['loss']['label_smoothing']==0
    count=sum(v.numel() for v in ck['model_state'].values());assert count in [75524,143172] # no running-state buffers in this retained architecture
    for name in ['checkpoints/best.pt','normalization.json','split.json','provenance.json','config.yaml','predictions/validation.csv']:
     p=base/name
     if p.exists():protected[str(p)]=sha(p)
    norm=load(base/'normalization.json');assert set(norm['subject_ids'])==set(sp['train_subjects']);recipe.append(dict(model=model,seed=seed,context=c,inner=i,parameter_state_elements=count,architecture_training_evaluation_match=True,AMP=oldcfg['training']['mixed_precision'],batch_size=oldcfg['training']['batch_size'],epochs=oldcfg['training']['epochs'],normalization_old_fit_membership_exact=True))
 pd.DataFrame(recipe).to_csv(HERE/'audit/legacy_recipe_equivalence.csv',index=False);pd.DataFrame(splitrows).to_csv(HERE/'audit/subject_role_counts.csv',index=False)
 # Official fixed exports, metadata first. Never decode a global cohort cache here.
 for sid in sorted(allids):
  p=P6/'features/external_by_subject'/f'{sid}.npz';assert str(p) in protected and sha(p)==protected[str(p)]
  with np.load(p,allow_pickle=False) as z:assert str(z['subject_id'])==sid and z['activities'].tolist()==ACTS and z['wrist_order'].tolist()==SIDES and 'layer5' in z.files
 write(PRIVATE/'manifests/protected_assets.json',dict(sha256=protected,official_reference=START,current_commit=current));write(PRIVATE/'manifests/raw_allowlist.json',dict(subject_ids=sorted(allids),records=rs))
 environment=dict(Python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,GPU=torch.cuda.get_device_name(),cuda=torch.version.cuda,matmul_TF32=torch.backends.cuda.matmul.allow_tf32,cudnn_TF32=torch.backends.cudnn.allow_tf32,initial_cudnn_deterministic=torch.backends.cudnn.deterministic,original_seed_code='deterministicTrue/warn_only; cuDNN deterministicTrue/benchmarkFalse; CUBLAS:4096:8',historical_TF32_runtime_receipt_available=False,TF32_policy='same installed torch2.8 defaults; no tuning or TF32 changes',new_dependencies_installed=False)
 write(HERE/'audit/environment.json',environment);write(HERE/'analysis/audit_integrity.json',dict(status='PASS_SOURCE_AND_AUTHORIZATION',reference=START,current=current,protected_count=len(protected),legacy_checkpoint_pairs=90,allowed_subject_count=390,raw_records=8580,actual_legacy_L1_settings=oldprocessing['accelerometer'],current_L1_function_defaults_not_production_recipe=True,normalization_refits_so_far=0,new_classification_results_seen=False,original_HarNet_per_subject_cache_metadata_verified=True,gravity_semantics='UNKNOWN_ALLOWED_FIXED_TARGET_AMENDMENT'))
 freeze('PRE_CLASSIFICATION',[HERE/'PROTOCOL.md',HERE/'PROTOCOL_AMENDMENT.md',HERE/'configs/USER_REQUEST.md'],WSSL_formal_runs=45,STR_formal_runs=45,new_classification_results_seen=False,C1_families=[2,7],C2_families=[4,14],DID_families=[2,7]);print(json.dumps(dict(status='PASS_AUDIT',protected=len(protected),current=current,environment=environment)))
if __name__=='__main__':main()
