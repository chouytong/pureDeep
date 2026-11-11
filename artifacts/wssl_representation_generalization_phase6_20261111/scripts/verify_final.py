"""Final read-only scientific/asset checks, followed by new audit output writes."""
from common import *
import ast,datetime,importlib.util

def main():
 protected=guard();assert protected['git_start_sha']=='161b904462e754a892c3d3d2e4f9ea68b33d18c0'
 assert len(protected['sha256'])==21942
 for name in ['ANALYSIS_FREEZE.json','STATISTICS_SOURCE_FREEZE.json','SUPPLEMENT_SOURCE_RECORD.json']:
  for rel,digest in load(HERE/'configs'/name)['files'].items():assert sha(HERE/rel)==digest,(name,rel)
 freeze=load(HERE/'configs/ANALYSIS_FREEZE.json');assert freeze['BH_families']==dict(A_primary=3,A_exploratory=18,A_group=6,B_interaction=3)
 asset=pd.read_csv(HERE/'analysis/paired_representation_asset_audit.csv');identity=pd.read_csv(HERE/'analysis/paired_inference_identity.csv');merged=pd.read_csv(HERE/'analysis/paired_representation_integrity.csv')
 assert len(asset)==len(identity)==len(merged)==45 and asset.asset_status.eq('PASS_PRE_INFERENCE').all()
 assert identity[['frozen_probability_max_error','adapted_probability_max_error']].max().max()<=freeze['probability_identity_tolerance']
 assert identity.parameter_buffers_unchanged.all()
 lineage=load(HERE/'restricted_manifests/extraction_lineage.json');assert len(lineage['records'])==45 and lineage['only_validation_subjects_in_model_inputs']
 split={(c,i):(sp,ex) for c,i,sp,ex in splits()};n=0
 for r in lineage['records']:
  sp,ex=split[r['context'],r['inner']];query=set(r['query_subject_ids']);fit=set(r['fit_subject_ids']);excluded=set(r['excluded_subject_ids'])
  assert query==set(map(str,sp['validation_subjects'])) and excluded==set(ex)
  assert query.isdisjoint(fit) and query.isdisjoint(excluded) and fit.isdisjoint(excluded)
  assert sha(r['source_feature_path'])==r['feature_sha256'] and sha(r['shared_layer4_path'])==r['shared_layer4_sha256']
  for kind in ['frozen','adapted']:assert sha(stage(kind,r['seed'],r['context'],r['inner'])/'checkpoints/best.pt')==r[kind+'_checkpoint_sha256']
  assert r['source_code_sha256']==sha(HERE/'scripts/paired_inference.py')
  with np.load(r['source_feature_path'],allow_pickle=False) as z:
   ids=check_ids(z['subject_ids'],query,True);assert np.isfinite(z['frozen_layer5']).all() and np.isfinite(z['adapted_layer5']).all()
   for kind in ['frozen','adapted']:
    pred=prediction(stage(kind,r['seed'],r['context'],r['inner'])/'predictions/validation.csv',query).loc[ids]
    p=z[kind+'_probabilities'];assert np.max(abs(p-pred[['probability_pd','probability_dd']].to_numpy()))<=1e-12
    assert np.array_equal(p.argmax(1),pred.prediction.to_numpy())
   n+=len(ids)
 files=load(HERE/'restricted_manifests/all_exported_feature_hashes.json')['files'];assert len(files)==495
 assert set(x['path'] for x in files)==set(str(x) for x in (HERE/'features').rglob('*.npz'))
 for record in files:assert sha(record['path'])==record['sha256']
 tests=load(HERE/'tests/paired_forward_correctness.json');assert len(tests['cases'])==45 and all(all(r[k] for k in ['parts_equal_explicit_forward','inputs_not_mutated','forbidden_IO_rejected']) for r in tests['cases'])
 assert tests['official_encoder_unchanged'] and tests['BN_buffers_unchanged'] and tests['all_gradients_disabled'] and tests['optimizer_steps']==0
 layer=pd.read_csv(HERE/'analysis/representation_layer_summary.csv');zero=layer[layer.layer=='layer4'];assert (zero[['cosine_distance_mean','L2_mean','relative_displacement_mean']].to_numpy()==0).all()
 a=pd.read_csv(HERE/'analysis/representation_paired_tests.csv');sens=pd.read_csv(HERE/'analysis/representation_tests_planned_BH_sensitivity.csv')
 for family,count,undefined in [('A_primary',3,0),('A_exploratory',18,3),('A_group',6,3)]:
  t=a[a.family==family];assert len(t)==count and t.wilcoxon_p.isna().sum()==undefined
 assert np.allclose(a.wilcoxon_p,sens.wilcoxon_p,equal_nan=True) and np.allclose(a.bh_q,sens.bh_q,equal_nan=True)
 for family,ix in sens.groupby('family').groups.items():
  p=sens.loc[ix,'wilcoxon_p'].to_numpy();q=bh(np.where(np.isfinite(p),p,1.));q[~np.isfinite(p)]=np.nan
  assert np.allclose(q,sens.loc[ix,'bh_q_planned_family_sensitivity'],equal_nan=True)
 b=load(HERE/'analysis/phase6b_integrity.json');assert b['old_scores_and_cutoffs_unchanged'] and b['new_bank_scaler_PCA_fits']==0 and b['per_activity_reference_distance_decomposition']=='BLOCKED_NOT_SAVED'
 assert len(pd.read_csv(HERE/'analysis/coverage_interaction_tests.csv'))==3
 supp=load(HERE/'tests/supplement_integrity.json');assert supp['old_patient_groups_reproduced_43_46_301']
 lock=load(HERE/'analysis/external_validation_lock.json');assert sha(lock['checkpoint_path'])==lock['checkpoint_sha256']
 assert lock['protocol_sha256']==sha(HERE/'INDEPENDENT_VALIDATION_PROTOCOL.md') and lock['external_metrics'] is None and not lock['external_data_ready']
 assert lock['calibrator'].startswith('NONE') and lock['threshold']==.5
 env=load(HERE/'analysis/runtime_environment.json');assert env['new_deep_training']==env['new_fits']==env['dependency_installs']==0
 # Check active scientific scripts for fitting/training call sites, not prose mentions.
 science=['bootstrap_audit.py','paired_inference.py','independent_wssl.py','drift_analysis.py','coverage_analysis.py','supplement_checks.py','external_inventory.py','stability_check.py']
 for name in science:
  tree=ast.parse((HERE/'scripts'/name).read_text())
  for node in ast.walk(tree):
   if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
    assert node.func.attr not in {'fit','fit_transform','backward','step'},(name,node.lineno)
    if node.func.attr=='train':
     # Protected explicit-model source contains an UNUSED copy helper that mirrors
     # mode. No active script invokes it; preserve this source rather than edit it.
     assert name=='independent_wssl.py' and ast.unparse(node)=='other.train(model.training)'
   if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):assert node.func.id!='independent_copy'
 required=['ADAPTATION_REPRESENTATION_REPORT.md','CLASS_CONDITIONAL_COVERAGE_REPORT.md','INDEPENDENT_VALIDATION_PROTOCOL.md','DATASET_COMPATIBILITY_AUDIT.md','FINAL_REPORT.md','CORRECTNESS_AND_PROVENANCE.md','STATISTICAL_APPENDIX.md','REPRODUCE.md','NEXT_STAGE_CANDIDATES.md','decision.json']
 assert all((HERE/p).is_file() and (HERE/p).stat().st_size for p in required)
 # Apply the unchanged publisher's actual privacy/size logic to the public manifest.
 spec=importlib.util.spec_from_file_location('publisher',ROOT/'tools/git_publish/publish_to_github.py');pub=importlib.util.module_from_spec(spec);spec.loader.exec_module(pub)
 public=[]
 for p in sorted(HERE.rglob('*')):
  if not p.is_file() or any(x in {'features','restricted_manifests','logs','__pycache__'} for x in p.relative_to(HERE).parts):continue
  if p.name in {'source_output_manifest.json','final_integrity.json','final_correctness.json'}:continue
  assert pub.reason(p,p.relative_to(ROOT)) is None,('Public artifact rejected by unchanged publication rule',str(p))
  assert not pub.SECRET.search(p.read_bytes())
  public.append(dict(path=str(p.relative_to(HERE)),sha256=sha(p),bytes=p.stat().st_size))
 assert len(public)>50
 write_json(HERE/'analysis/source_output_manifest.json',dict(files=public,private_directories_excluded=['features','restricted_manifests','logs'],self_and_final_audit_excluded_to_avoid_recursive_hash=True))
 guard()
 result=dict(status='PASS_WITH_EXPLICIT_SCOPE_LIMITATIONS',timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protected_files_unchanged=len(protected['sha256']),all45_pairs=45,model_validation_evaluations=90,validated_query_appearances_including_seeds=n,max_probability_error=float(identity[['frozen_probability_max_error','adapted_probability_max_error']].max().max()),all495_private_exports_hash_verified=True,all_original_source_freezes_unchanged=True,all_decisions_match=True,training_and_new_fits=0,old_patient_groups_reproduced=True,layer4_shared_zero_control=True,original_BH_and_declared_sensitivity_preserved=True,per_activity_reference_scores='BLOCKED_NOT_SAVED',external_validation='PROTOCOL_ONLY_NO_ELIGIBLE_DATA',historical_outer_exposure_permanent=True,OS_isolation_claimed=False,phase7_started=False,public_manifest_sha256=sha(HERE/'analysis/source_output_manifest.json'))
 write_json(HERE/'analysis/final_integrity.json',result);write_json(HERE/'tests/final_correctness.json',result)
 print(json.dumps(result,indent=2))

if __name__=='__main__':main()
