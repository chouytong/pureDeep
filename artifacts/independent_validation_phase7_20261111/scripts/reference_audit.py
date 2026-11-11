"""Read-only recovery audit. Never imports or runs original fitting functions."""
from common7 import *
import re,csv,zipfile,ast

def main():
 guard();code=REF/'scripts/typicality.py';text=code.read_text();tree=ast.parse(text)
 methods={n.name:dict(start=n.lineno,end=n.end_lineno) for n in ast.walk(tree) if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
 freeze=load(REF/'configs/B_freeze.json');hashes=freeze['files'];known={Path(k).name:v for k,v in hashes.items()};assert sha(code)==known['typicality.py']
 source_rows=[]
 for p in [code,REF/'scripts/common.py',REF/'scripts/calibration_fusion.py',REF/'scripts/audit_bank_norms.py',P5/'scripts/analyze.py',P5/'scripts/audit_lineage_supplement.py']:
  body=p.read_text();ast.parse(body);source_rows.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,all_source_text_read=True,executed=False))
 pd.DataFrame(source_rows).to_csv(HERE/'analysis/reference_source_audit.csv',index=False)
 canonical=load(MAIN/'splits/pads_classification/v3_nested_cv/seed42/nested_cv_splits.json');units=[]
 for outer in canonical['outer']:
  for inner in outer['inner_folds']:units.append((int(outer['outer_fold']),int(inner['inner_fold']),inner,list(map(str,outer['test_subjects']))))
 # Existing final query scores are distinct from missing reference training-OOF scores.
 p5=pd.read_csv(P5/'restricted_manifests/paired_appearances_private.csv',dtype={'subject_id':str});private=[];rows=[]
 for c,i,sp,excluded in units:
  fit=set(map(str,sp['train_subjects']));query=set(map(str,sp['validation_subjects']));ex=set(excluded);assert not fit&query and not query&ex
  manifest=REF/'restricted_manifests'/f'B_common_{c}_{i}.json';m=load(manifest);banks=load(REF/'restricted_manifests'/f'B_banks_{c}_{i}.json')
  assert set(m['train_subject_ids'])==fit and set(m['query_subject_ids'])==query
  assert all(set(banks[k])<=fit and not set(banks[k])&query for k in ['pd_subject_ids','dd_subject_ids'])
  assert len(banks['pd_subject_ids'])==len(banks['dd_subject_ids'])==min(m['full_bank_sizes'])
  path=REF/'predictions'/f'B_context{c}_inner{i}.csv';score=pd.read_csv(path,dtype={'subject_id':str});assert score.subject_id.is_unique and set(score.subject_id)==query
  assert np.allclose(score.min_distance,np.minimum(score.d_pd,score.d_dd),atol=1e-15,rtol=0) and np.allclose(score.relative_score,score.d_pd-score.d_dd,atol=1e-15,rtol=0)
  # Recover exactly the old unlabeled validation quantiles; not new thresholds.
  lo,hi=np.quantile(score.min_distance,[.25,.75],method='linear');f=p5[(p5.context==c)&(p5.inner==i)]
  for seed in [42,43,44]:
   s=f[f.seed==seed].set_index('subject_id').loc[score.subject_id];assert np.max(abs(s.u.to_numpy()-score.min_distance.to_numpy()))<1e-12
   expected=np.where(score.min_distance<=lo,'low',np.where(score.min_distance>=hi,'high','middle'));assert np.array_equal(s.coverage_group,expected)
  partition=[]
  for k in range(3):
   q=REF/'restricted_manifests'/f'Bfusion_OOF_{c}_{i}_fold{k}_B_common_{c}_{i}.json';r=load(q);a=set(r['train_subject_ids']);b=set(r['query_subject_ids']);assert not a&b and a|b==fit and not (a|b)&query
   partition.append(dict(fit_subject_ids=sorted(a),query_subject_ids=sorted(b),source_sha256=sha(q)))
  rows.append(dict(context=c,inner=i,fit_count=len(fit),query_count=len(query),excluded_count=len(ex),balanced_bank_per_class=len(banks['pd_subject_ids']),score_count=len(score),stored_query_scores='COMPLETE',fit_query_isolation='COMPLETE',reference_OOF_partitions='COMPLETE',whole_reference_OOF_scalar_matrix='MISSING',fitted_scaler_PCA_bank_vectors='MISSING_IN_AUDITED_SCOPE',q25=float(lo),q75=float(hi),cutoff_source='Phase5 validation-u quantiles; NOT train-derived OOF cutoffs',query_score_sha256=sha(path),fit_manifest_sha256=sha(manifest)))
  private.append(dict(context=c,inner=i,fit_subject_ids=sorted(fit),query_subject_ids=sorted(query),excluded_subject_ids=excluded,banks=banks,reference_crossfit_partitions=partition,original_B_freeze_sha256=sha(REF/'configs/B_freeze.json')))
 pd.DataFrame(rows).to_csv(HERE/'analysis/reference_asset_inventory.csv',index=False);write(HERE/'restricted_manifests/reference_lineage.json',dict(records=private))
 # Search names across active project artifacts, avoiding sealed outer namespaces.
 candidates=[];serialized=[]
 for base in [REF,P5,P6]:
  for p in base.rglob('*'):
   if not p.is_file() or '__pycache__' in p.parts or any('quarantin' in x or 'outer_final' in x for x in p.parts):continue
   if re.search(r'scaler|pca|bank|reference|typical|oof|transform|space',p.name,re.I):candidates.append(dict(path=str(p),suffix=p.suffix,bytes=p.stat().st_size,sha256=sha(p)))
   if p.suffix in {'.npz','.pkl','.pickle','.joblib'}:
    keys=[]
    if p.suffix=='.npz':
     with zipfile.ZipFile(p) as z:keys=z.namelist() # directory metadata only; no array decoded
    serialized.append(dict(path=str(p),suffix=p.suffix,keys=keys,sha256=sha(p)))
 write(HERE/'restricted_manifests/reference_state_search.json',dict(scope=[str(x) for x in [REF,P5,P6]],name_candidates=candidates,serialized_inventory=serialized,NPZ_arrays_decoded=False,unsearched_outside_scope_not_proven_absent=True))
 unexpected=[r for r in serialized if r['suffix']!='.npz' or any(re.search(r'scaler|pca|bank_vector|reference_oof|components|singular_values',k,re.I) for k in r['keys'])]
 # The search correctly flags45 saved TWO-SCORE fusion scalers. They are not
 # the22 independent1024-D reference feature scalers/PCA/bank transforms.
 classified=[]
 for r in unexpected:
  p=Path(r['path']);assert p.parent==REF/'checkpoints' and p.name.startswith('fusion_')
  assert set(r['keys'])=={'scaler_mean.npy','scaler_scale.npy','coef.npy','intercept.npy','temperature.npy'}
  shapes={}
  with zipfile.ZipFile(p) as z:
   for k in r['keys']:
    with z.open(k) as stream:
     version=np.lib.format.read_magic(stream)
     shape,fortran,dtype=np.lib.format._read_array_header(stream,version)
     shapes[k]=shape
  assert shapes['scaler_mean.npy']==shapes['scaler_scale.npy']==(2,) and shapes['coef.npy']==(1,2)
  classified.append(dict(path=str(p),status='COMPLETE_TWO_SCORE_FUSION_SCALER_NOT_REFERENCE_FEATURE_SCALER',shapes=shapes,sha256=r['sha256']))
 assert len(classified)==45
 write(HERE/'restricted_manifests/fusion_scaler_scope_audit.json',dict(records=classified,NPZ_array_values_decoded=False))
 hashes=pd.read_csv(REF/'analysis/B_bank_norm_hash_audit.csv');assert len(hashes)==5280 and all(hashes[k].astype(str).str.fullmatch('[0-9a-f]{64}').all() for k in ['scaler_mean_sha256','PCA_components_sha256'])
 # Original helper may re-fit5280 spaces; we do NOT execute it or reproduce its fitted states.
 status=dict(status='PARTIAL_SOURCE_AND_QUERY_SCORES_COMPLETE_EXACT_EXTERNAL_PIPELINE_BLOCKED',units=15,reference_algorithm_source_matches_creation_freeze=True,reference_transform_hash_rows=5280,transform_hashes_are_not_transform_values=True,stored_final_query_scores_present=True,training_reference_OOF_score_matrix='MISSING',reference_OOF_partitions_present=True,deep_WSSL_OOF_not_reference_OOF=True,reference_feature_scaler_full_state='MISSING',PCA_full_state='MISSING',transformed_bank_vectors='MISSING',two_score_fusion_scaler_states_present=45,fusion_scaler_does_not_restore_reference_geometry=True,per_activity_query_distances='MISSING',bank_membership_ids_present=True,exact_external_reference_scoring='BLOCKED_MISSING_FITTED_STATES',missing_state_search_scope=[str(x) for x in [REF,P5,P6]],global_absence_claimed=False,reference_new_fits=0,reference_new_scores=0,serialized_files_headers_inventoried=len(serialized),private_ID_records=len(private),source_functions=methods,source_sha256=sha(code))
 write(HERE/'analysis/reference_provenance_integrity.json',status);guard();print('REFERENCE_RECOVERY_AUDIT_PARTIAL_NO_REFIT')
if __name__=='__main__':main()
