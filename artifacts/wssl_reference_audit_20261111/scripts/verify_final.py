"""Verify saved results and immutable history. No fitting or inference."""
from common import *
import ast,datetime

def main():
    lock=guard();checks={}
    required=['AUDIT_REPORT.md','EXPERIMENT_LEDGER.csv','EXPERIMENT_LEDGER.md','NEGATIVE_RESULTS_DO_NOT_REPEAT.md','CANDIDATE_PRIORITY.md','PROTOCOL.md','BASELINE_REPRODUCTION.md','LOSS_NUMERICAL_AUDIT.md','MODALITY_ERROR_DIAGNOSTICS.md','CALIBRATION_DIAGNOSTICS.md','TYPICALITY_REPORT.md','FINAL_REPORT.md','decision.json','LITERATURE_VERIFICATION.md','REPRODUCE.md','STATISTICAL_APPENDIX.md']
    for n in required:assert (HERE/n).stat().st_size>0,n
    checks['required_deliverables_present']=True
    for p in (HERE/'scripts').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    checks['all_script_syntax_valid']=True
    freezes=list((HERE/'configs').glob('*_freeze.json'));frozen=0
    for p in freezes:
        record=json.loads(p.read_text())
        for name,digest in record['files'].items():assert sha(HERE/name)==digest,(p,name);frozen+=1
    checks['all_entry_freeze_source_hashes_unchanged']=True
    repro=json.loads((HERE/'analysis/baseline_reproduction.json').read_text());assert repro['checkpoints']==45 and repro['max_probability_difference']<=1e-12 and not repro['archived_logits_available']
    baseline_metrics=pd.read_csv(HERE/'analysis/baseline_metrics.csv');assert len(baseline_metrics)==45
    oof=json.loads((HERE/'analysis/OOF_provenance.json').read_text());assert oof['models_actually_available']==135 and oof['prediction_sets_available']==135 and oof['entire_query_training_chain_disjoint']
    checks['baseline_45_and_historical_OOF_135_provenance']=True
    for c,i,sp in splits():
      allowed=set(map(str,sp['train_subjects']));valid=set(map(str,sp['validation_subjects']));assert not allowed&valid
      for seed in [42,43,44]:
        with np.load(HERE/'features'/f'seed{seed}_context{c}_inner{i}.npz',allow_pickle=False) as z:
          ids=list(map(str,z['subject_ids']));q=z['is_validation'];assert len(ids)==len(set(ids));assert set(np.array(ids)[q])==valid and set(np.array(ids)[~q])==allowed
        with np.load(HERE/'features'/f'OOF_seed{seed}_context{c}_inner{i}.npz',allow_pickle=False) as z:assert set(map(str,z['subject_ids']))==allowed
    checks['saved_feature_fit_query_boundary_matches_development_splits']=True
    # Synthetic negative sentinel demonstrates fit membership checking; not a historical system-wide I/O guard.
    sp=splits()[0][2]
    try:fit_ids('MUST_NOT_WRITE',['__forbidden_fit_subject__'],[],sp['train_subjects'])
    except AssertionError:pass
    else:raise AssertionError('Out-of-allowed fit ID was accepted')
    assert not (HERE/'restricted_manifests/MUST_NOT_WRITE.json').exists()
    checks['negative_fit_ID_membership_test']=True
    assert json.loads((HERE/'analysis/A0_gate.json').read_text())['training_allowed'] is False
    assert json.loads((HERE/'analysis/A0_precision_assessment.json').read_text())['failed_actual_runtime_seeds']==[44]
    assert json.loads((HERE/'analysis/B_bank_norm_summary.json').read_text())['selected_bank_zero_vectors']==0
    d=json.loads((HERE/'decision.json').read_text());assert d['new_deep_training_runs']==0 and not d['retained_native_model_changed'] and d['historical_outer_summary_text_exposed'] and not d['strict_outer_no_access_certified']
    checks['decisions_preserve_failed_tests_and_protocol_deviation']=True
    ledger=pd.read_csv(HERE/'EXPERIMENT_LEDGER.csv').set_index('experiment_id')
    for n,k in [('V8',45),('V8-GN',45),('STR-01',45),('WSR-01',45),('prototype',90),('strong-backbone',135),('phase3c-deep-OOF',135)]:assert int(ledger.loc[n,'actual_completed_inner_stages'])==k,(n,k)
    files={}
    for p in HERE.rglob('*'):
        if not p.is_file() or any(x in p.parts for x in ['features','predictions','checkpoints','restricted_manifests','logs','__pycache__']):continue
        if p.name in ['source_output_manifest.json','final_integrity.json']:continue
        files[str(p.relative_to(HERE))]={'sha256':sha(p),'bytes':p.stat().st_size}
    write_json(HERE/'analysis/source_output_manifest.json',dict(files=files,contains_private_arrays=False,git_start_sha=lock['git_sha']))
    guard()
    private=json.loads(Path(lock['private_data_ledger']['path']).read_text()) if 'private_data_ledger' in lock else None
    write_json(HERE/'analysis/final_integrity.json',dict(status='ASSET_INTEGRITY_PASS_WITH_DISCLOSED_PROTOCOL_DEVIATION',timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,protected_file_count=len(lock['sha256']),protected_private_data_files=len(private['files']) if private else None,entry_freeze_files=len(freezes),frozen_source_entries=frozen,all_protected_hashes_unchanged=True,all_correctness_pass=False,actual_TF32_parameter_gradient_precision_pass=False,strict_outer_no_access_pass=False,historical_outer_summary_text_exposed=True,new_deep_training_runs=0,retained_native_model_changed=False,no_new_candidate_after_deviation_identification=True,publication_receipt='Stored outside deep_final after exact-tree GitHub verification; no self-referential source commit'))
    print(json.dumps(dict(status='FINAL_INTEGRITY_RECORDED',asset_hashes_pass=True,all_correctness_pass=False,outer_no_access_pass=False,new_deep_runs=0)),flush=True)

if __name__=='__main__':main()
