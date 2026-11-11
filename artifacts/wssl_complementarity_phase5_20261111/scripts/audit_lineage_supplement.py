"""Complete the saved fusion-fit lineage audit, without refitting its coefficients."""
from common import *

def main():
    lock=guard();records=[];partition_checks=0
    for c,i,sp,excluded in splits():
      fit=set(sp['train_subjects']);query=set(sp['validation_subjects'])
      for seed in [42,43,44]:
        path=OLD/'restricted_manifests'/f'Bfusion_fit_seed{seed}_{c}_{i}.json';record=load(path)
        assert set(record['train_subject_ids'])==fit and set(record['query_subject_ids'])==query
        assert record['features']==['WSSL deepOOF logit','whole-chainOOF relative typicality'] and record['C']==1
        refs=[];deep=[]
        for k in range(3):
            a=load(OLD/'restricted_manifests'/f'Bfusion_OOF_{c}_{i}_fold{k}_B_common_{c}_{i}.json')
            b=load(ROOT/f'artifacts/phase3c_ssl_adaptation_20260930/oof/seed{seed}/outer_{c}/inner_{i}/crossfit_{k}/split.json')
            assert set(a['train_subject_ids'])==set(b['train_subjects']) and set(a['query_subject_ids'])==set(b['validation_subjects'])
            assert not set(a['query_subject_ids'])&set(a['train_subject_ids']) and not (set(a['query_subject_ids'])|set(a['train_subject_ids']))&query
            refs.append(sha(OLD/'restricted_manifests'/f'Bfusion_OOF_{c}_{i}_fold{k}_B_common_{c}_{i}.json'));deep.append(lock['sha256'][str(ROOT/f'artifacts/phase3c_ssl_adaptation_20260930/oof/seed{seed}/outer_{c}/inner_{i}/crossfit_{k}/checkpoints/best.pt')]);partition_checks+=1
        records.append(dict(seed=seed,context=c,inner=i,fit_subject_ids=sorted(fit),query_subject_ids=sorted(query),excluded_subject_ids=excluded,source_fit_manifest_sha256=sha(path),source_fusion_state_sha256=sha(OLD/'checkpoints'/f'fusion_seed{seed}_context{c}_inner{i}.npz'),source_WSSL_OOF_feature_sha256=sha(OLD/'features'/f'OOF_seed{seed}_context{c}_inner{i}.npz'),source_reference_OOF_individual_feature_sha256=None,source_reference_OOF_individual_feature_status='MISSING',source_reference_fit_manifest_sha256=refs,source_WSSL_OOF_checkpoint_sha256=deep,source_validation_WSSL_feature_sha256=sha(OLD/'features'/f'seed{seed}_context{c}_inner{i}.npz'),source_reference_validation_score_sha256=sha(OLD/'predictions'/f'B_context{c}_inner{i}.csv'),normalization_file_sha256=sha(stage('baseline',seed,c,i)/'normalization.json'),split_id=f'context{c}_inner{i}'))
    write_json(HERE/'restricted_manifests/fusion_fit_lineage.json',dict(records=records,private_do_not_publish=True))
    write_json(HERE/'analysis/fusion_lineage_integrity.json',dict(status='SAVED_FIT_IDS_AND_SOURCE_PARTITIONS_PASS_WITH_REFERENCE_STATE_GAPS',fusion_fit_records=len(records),matched_reference_deep_OOF_partitions=partition_checks,all_target_validation_subjects_excluded=True,whole_reference_training_feature_matrix_available=False,full_reference_transform_state_available=False,no_refit=True,known_phase5_results_already_seen=True,purpose='complete lineage verification, no change of protocol or scientific analyses'))
    guard();print('FUSION_LINEAGE_RECORDS_VERIFIED')

if __name__=='__main__':main()
