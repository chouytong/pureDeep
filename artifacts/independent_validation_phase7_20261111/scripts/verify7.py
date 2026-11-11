"""Final audit of completed, explicitly limited Phase7 deliverables."""
from common7 import *
import ast,datetime,importlib.util
def main():
 count=guard();assert count==22529
 freeze=load(HERE/'configs/SYNTHETIC_SOURCE_FREEZE.json')
 for rel,d in freeze['files'].items():assert sha(HERE/rel)==d
 startup=load(HERE/'analysis/startup_integrity.json');assert startup['git_main']==START and startup['locked_assets_all_match'] and startup['missing_documents']==0
 actual=pd.read_csv(HERE/'analysis/model_lock_hash_audit.csv');assert len(actual)==5 and actual.expected_sha256.eq(actual.actual_sha256).all()
 for row in actual.itertuples():assert sha(row.path)==row.expected_sha256
 sources=pd.read_csv(HERE/'analysis/mainline_source_audit.csv');assert len(sources)==45
 for row in sources.itertuples():assert sha(ROOT/row.path)==row.sha256 and sha(MIRROR/row.path)==row.sha256
 h=pd.read_csv(HERE/'analysis/history_review.csv');assert len(h)==19 and h.status.eq('CURRENT_FILE_READ_AND_HASHED').all()
 for row in h.itertuples():assert sha(row.path)==row.sha256
 syn=load(HERE/'analysis/synthetic_integrity.json');tests=pd.read_csv(HERE/'tests/synthetic_correctness.csv')
 assert syn['status']=='PASS_SYNTHETIC_ONLY' and len(tests)==syn['tests']==25 and tests.status.str.startswith('PASS').all()
 assert syn['determinism_max_error']==0 and syn['official_model_and_BN_unchanged'] and not syn['real_external_adapter_validated'] and syn['clinical_sensor_files_opened']==0
 for k in ['new_training','optimizer_updates','new_fits','external_data_downloads']:assert syn[k]==0
 ref=load(HERE/'analysis/reference_provenance_integrity.json');assert ref['units']==15 and ref['reference_algorithm_source_matches_creation_freeze'] and ref['reference_new_fits']==ref['reference_new_scores']==0
 assert ref['two_score_fusion_scaler_states_present']==45 and ref['reference_feature_scaler_full_state']=='MISSING' and ref['PCA_full_state']=='MISSING' and ref['transformed_bank_vectors']=='MISSING'
 assert ref['serialized_files_headers_inventoried']==630 and ref['exact_external_reference_scoring']=='BLOCKED_MISSING_FITTED_STATES'
 assert len(load(HERE/'restricted_manifests/reference_lineage.json')['records'])==15
 assert len(load(HERE/'restricted_manifests/fusion_scaler_scope_audit.json')['records'])==45
 state=load(HERE/'restricted_manifests/model_ids.json');assert len(state['fit_subject_ids'])==205 and state['external_query_ids']==state['new_fit_ids']==state['clinical_sensor_files_opened']==[]
 data=pd.read_csv(HERE/'DATASET_COMPATIBILITY_MATRIX.csv');assert len(data)==20 and data.dataset.is_unique and not data.direct_model_ready.any()
 calculated=data[['label_score','sensor_activity_score','independence_score','access_score','sample_subtype_score']].to_numpy()@np.array([30,25,20,15,10])/4
 assert np.allclose(calculated,data.priority_score,atol=1e-12,rtol=0)
 assert (data.loc[~data.target_category.str.startswith('A'),'label_score']==0).all()
 assert data.loc[data.dataset.str.startswith('PADS'),'compatibility'].eq('Blocked').all()
 assert data['user_data_grant'].str.startswith('NONE_VERIFIED').all()
 decision=load(HERE/'decision.json');assert decision['external_metrics'] is None and decision['reference_external_interaction'] is None and not decision['start_phase8'] and not decision['model_changed']
 # Only active new scripts are checked as executable workflow. Failed audit v1
 # remains as a preserved history snapshot and is never called again.
 active=['common7.py','bootstrap7.py','external_adapter.py','synthetic_tests.py','reference_audit.py','dataset_discovery.py','build_reports7.py','verify7.py']
 for name in active:
  for node in ast.walk(ast.parse((HERE/'scripts'/name).read_text())):
   if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):assert node.func.attr not in {'fit','fit_transform','backward','step','train'},(name,node.lineno)
 required=['FINAL_REPORT.md','DATASET_DISCOVERY_REPORT.md','DATASET_COMPATIBILITY_MATRIX.csv','MODEL_DEPLOYMENT_AUDIT.md','EXTERNAL_INPUT_ADAPTER_REPORT.md','INDEPENDENT_VALIDATION_PROTOCOL.md','REFERENCE_PIPELINE_PROVENANCE.md','NEXT_STAGE_RESEARCH_PLAN.md','CORRECTNESS_AND_PROVENANCE.md','REPRODUCE.md','DATA_ACCESS_PREPARATION.md','SEARCH_LOG.md']
 for name in required:assert (HERE/name).is_file() and (HERE/name).stat().st_size
 lock=load(HERE/'analysis/deployment_lock.json')
 write(HERE/'analysis/phase7_provenance.json',dict(checkpoint_sha256=lock['checkpoint_sha256'],normalization_sha256=lock['normalization_file_sha256'],canonical_config_sha256=lock['canonical_config_sha256'],preprocessing_sha256=lock['preprocessing_config_sha256'],official_HarNet_sha256=lock['official_HarNet10_checkpoint_sha256'],new_experiment_data_source='programmatically generated synthetic fixtures only; public study-level metadata; existing authorized score/ID audit',new_external_data_authorization='NONE_ACQUIRED; no download/access application authorized this round',external_subject_identity_verification='NOT_COMPLETED',private_fit_query_excluded_lineage='restricted_manifests/model_ids.json + reference_lineage.json',runtime_logs='logs (server-private)',correctness='tests/synthetic_correctness.csv',all_training_and_fit_budgets=0))
 spec=importlib.util.spec_from_file_location('publisher',ROOT/'tools/git_publish/publish_to_github.py');pub=importlib.util.module_from_spec(spec);spec.loader.exec_module(pub)
 public=[]
 for p in sorted(HERE.rglob('*')):
  if not p.is_file() or any(k in {'restricted_manifests','logs','__pycache__'} for k in p.relative_to(HERE).parts):continue
  if p.name in {'source_output_manifest.json','phase7_integrity.json','final_verification.json'}:continue
  assert pub.reason(p,p.relative_to(ROOT)) is None,('Publication rule rejects artifact',str(p))
  assert not pub.SECRET.search(p.read_bytes())
  public.append(dict(path=str(p.relative_to(HERE)),sha256=sha(p),bytes=p.stat().st_size))
 write(HERE/'analysis/source_output_manifest.json',dict(files=public,private_directories_excluded=['restricted_manifests','logs'],recursive_final_audit_excluded=True))
 guard();result=dict(status='PASS_WITH_BLOCKED_CLINICAL_AND_REFERENCE_EXECUTION',timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),git_start=START,protected_files_unchanged=count,history_currently_read=19,mainline_matches_published_files=45,model_asset_hashes_passed=5,synthetic_correctness_tests_passed=25,synthetic_repeat_logit_max_error=0,real_clinical_input_adapter_validated=False,dataset_candidates=20,compatible_authorized_independent_cohorts=0,original_reference_source_and_query_scores='COMPLETE',original_reference_exact_external_pipeline='BLOCKED_MISSING_FITTED_STATES',saved_fusion_scalers_are_not_reference_feature_scalers=True,new_deep_training=0,optimizer_updates=0,new_classifier_training=0,new_temperature_threshold_reference_fits=0,external_individual_data_downloads=0,clinical_sensor_files_opened=0,new_data_applications_contacts_or_agreements=0,dependency_installs=0,external_classification_metrics=None,external_class_coverage_interaction=None,mainline_and_model_unchanged=True,historical_disclosures_preserved=True,OS_physical_isolation_claimed=False,phase8_started=False,source_output_manifest_sha256=sha(HERE/'analysis/source_output_manifest.json'),public_artifact_count_excluding_final_audit=len(public))
 write(HERE/'analysis/phase7_integrity.json',result);write(HERE/'tests/final_verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
