"""Hash, ID and metadata audit only; no checkpoint/model inference or feature replay."""
from common import *

def main():
    lock=guard();digests=lock['sha256'];rows=[];lineage=[];reference=[]
    prior=load(OLD/'analysis/OOF_provenance.json');assert prior['models_actually_available']==135 and prior['strict_state_loads']==135
    inventory=pd.read_csv(OLD/'analysis/OOF_inventory_135.csv').set_index(['seed','context','inner','crossfit'])
    transforms=pd.read_csv(OLD/'analysis/B_bank_norm_hash_audit.csv')
    def h(p):
        p=str(p);assert p in digests,'Unprotected asset: '+p
        return digests[p]
    for c,i,sp,excluded in splits():
        train=set(sp['train_subjects']);query=set(sp['validation_subjects']);assert not train&query and not (train|query)&set(excluded)
        feature=feature_scores(42,c,i,sp);train_labels=feature[~feature.is_validation].target.to_dict()
        for seed in [42,43,44]:
            norms=[]
            for kind in ['baseline','adapted']:
                p=stage(kind,seed,c,i);s=load(p/'split.json');n=load(p/'normalization.json');prov=load(p/'provenance.json')
                assert set(s['train_subjects'])==train and set(s['validation_subjects'])==query and not s['test_subjects']
                assert set(n['subject_ids'])==train and n['normalization_sha256']==prov['normalization_sha256']
                assert prov['label_mapping']=={'PD':0,'DD':1}
                f=prediction(p/'predictions/validation.csv',query)
                fv=feature_scores(seed,c,i,sp).loc[f.index];assert np.array_equal(f.target,fv.target)
                if kind=='baseline':assert np.max(abs(f[['probability_pd','probability_dd']].to_numpy()-fv[['stored_probability_pd','stored_probability_dd']].to_numpy()))<=1e-12
                norms.append(n)
                row=dict(kind=kind,seed=seed,context=c,inner=i,status='VERIFIED_HASH_IDS_LABELS_PRIOR_LOAD',checkpoint_sha256=h(p/'checkpoints/best.pt'),prediction_sha256=h(p/'predictions/validation.csv'),normalization_file_sha256=h(p/'normalization.json'),normalization_values_sha256=n['normalization_sha256'],split_file_sha256=h(p/'split.json'),config_file_sha256=h(p/'config.yaml'),train_n=len(train),query_n=len(query),excluded_n=len(excluded),feature_status='EXISTING_POST_MODEL_FEATURE_PACKET' if kind=='baseline' else 'MISSING_POST_ADAPTATION_FEATURE_PACKET')
                rows.append(row);lineage.append(dict(**row,fit_subject_ids=sorted(train),query_subject_ids=sorted(query),excluded_subject_ids=excluded,source_checkpoint=str(p/'checkpoints/best.pt'),source_feature_sha256=h(OLD/'features'/f'seed{seed}_context{c}_inner{i}.npz') if kind=='baseline' else None,source_feature_creation_time_manifest='not saved; current hash plus prior source/probability audit, not independent feature re-inference'))
            assert norms[0]['normalization_sha256']==norms[1]['normalization_sha256']
            assert np.array_equal(norms[0]['mean'],norms[1]['mean']) and np.array_equal(norms[0]['std'],norms[1]['std'])
            for k in range(3):
                p=ROOT/f'artifacts/phase3c_ssl_adaptation_20260930/oof/seed{seed}/outer_{c}/inner_{i}/crossfit_{k}'
                s=load(p/'split.json');n=load(p/'normalization.json');fit=set(s['train_subjects']);qs=set(s['validation_subjects'])
                assert fit|qs==train and not fit&qs and not (fit|qs)&(query|set(excluded)) and not s['test_subjects'] and set(n['subject_ids'])==fit
                evidence=inventory.loc[(seed,c,i,k)];assert h(p/'checkpoints/best.pt')==evidence.checkpoint_sha256 and h(p/'predictions/validation.csv')==evidence.prediction_sha256
                pf=prediction(p/'predictions/validation.csv',qs);assert all(int(y)==train_labels[sid] for sid,y in pf.target.items())
                row=dict(kind='historical_deep_OOF',seed=seed,context=c,inner=i,crossfit=k,status='VERIFIED_HASH_IDS_LABELS_PRIOR_STRICT_LOAD',checkpoint_sha256=h(p/'checkpoints/best.pt'),prediction_sha256=h(p/'predictions/validation.csv'),normalization_file_sha256=h(p/'normalization.json'),normalization_values_sha256=n['normalization_sha256'],split_file_sha256=h(p/'split.json'),train_n=len(fit),query_n=len(qs),excluded_n=len(query|set(excluded)))
                rows.append(row);lineage.append(dict(**row,fit_subject_ids=sorted(fit),query_subject_ids=sorted(qs),excluded_subject_ids=sorted(query|set(excluded)),source_feature_sha256=h(OLD/'features'/f'OOF_seed{seed}_context{c}_inner{i}.npz'),checkpoint_selection='fixed epoch10, prior audit strictly loaded these same SHA states'))
        for fold in [-1,0,1,2]:
            prefix='' if fold==-1 else f'Bfusion_OOF_{c}_{i}_fold{fold}_'
            a=OLD/'restricted_manifests'/f'{prefix}B_common_{c}_{i}.json';b=OLD/'restricted_manifests'/f'{prefix}B_banks_{c}_{i}.json'
            fitrecord=load(a);bank=load(b);fit=set(fitrecord['train_subject_ids']);qs=set(fitrecord['query_subject_ids'])
            if fold==-1:assert fit==train and qs==query
            else:assert fit|qs==train and not fit&qs and not (fit|qs)&query
            assert fit<=train and not fit&qs
            for label,key in [(0,'pd_subject_ids'),(1,'dd_subject_ids')]:
                require_ids(bank[key],fit);assert all(train_labels[s]==label for s in bank[key])
            assert len(bank['pd_subject_ids'])==len(bank['dd_subject_ids'])>=5
            t=transforms[(transforms.context==c)&(transforms.inner==i)&(transforms.fit_namespace==a.stem)]
            assert len(t)==22 and t['rank'].min()==16 and t.zero_selected_bank_vectors.sum()==0
            row=dict(context=c,inner=i,reference_fold=fold,fit_n=len(fit),query_n=len(qs),bank_per_class=len(bank['pd_subject_ids']),fit_ID_record_sha256=h(a),bank_ID_record_sha256=h(b),external_encoder_cache_sha256=h(ROOT/'data/ssl/harnet10_embeddings.npz'),transform_hash_record_sha256=h(OLD/'analysis/B_bank_norm_hash_audit.csv'),lineage_status='ID_AND_HASH_RECORDS_VERIFIED',serialized_scaler_PCA_and_bank_vectors='MISSING',numeric_bank_replay='BLOCKED_NO_REFIT_AUTHORIZED')
            reference.append(row);lineage.append(dict(**row,seed='reference deterministic; OOF source partition seed42 shared',fit_subject_ids=sorted(fit),query_subject_ids=sorted(qs),excluded_subject_ids=sorted((train|query|set(excluded))-fit-qs)))
        print(json.dumps(dict(stage='asset_audit',context=c,inner=i,status='HASH_AND_IDS_VERIFIED')),flush=True)
    pd.DataFrame(rows).to_csv(HERE/'analysis/asset_inventory.csv',index=False)
    pd.DataFrame(reference).to_csv(HERE/'analysis/reference_asset_lineage_summary.csv',index=False)
    write_json(HERE/'restricted_manifests/all_model_reference_lineage.json',dict(records=lineage,private_ID_records_do_not_publish=True))
    discovered=[]
    for root in [ROOT/'artifacts/phase3c_ssl_adaptation_20260930',OLD]:
        for p in root.rglob('*'):
            if p.is_file() and p.suffix in ['.npz','.npy','.pkl','.h5','.parquet']:
                discovered.append(dict(path=str(p),bytes=p.stat().st_size,sha256=h(p) if str(p) in digests else sha(p)))
    write_json(HERE/'restricted_manifests/feature_discovery_inventory.json',dict(files=discovered,search_scope='phase3c artifact and previous reference audit; source scripts persist no adapted post-layer5 features and no row-level reference OOF score'))
    missing=[dict(asset='saved reference OOF row-level score vector',status='MISSING',blocked='joint/class-conditional OOF reference distribution and exact training-fusion score replay'),dict(asset='serialized fitted reference Scaler/PCA and bank vectors',status='MISSING',blocked='independent numeric replay of reference fit/transform chain; ID/hash records only'),dict(asset='post-adaptation final HarNet/local/subject features',status='MISSING',blocked='cosine/L2/norm representation drift'),dict(asset='adapted raw two-class logits',status='MISSING',blocked='raw DD logit displacement; probability-implied log odds remain available')]
    write_json(HERE/'analysis/asset_audit.json',dict(status='AVAILABLE_SCORE_ASSETS_VERIFIED_WITH_MISSING_STATE_SUBANALYSES',model_assets=len(rows),baseline=45,adapted=45,historical_OOF=135,reference_fit_ID_records=len(reference),protected_assets=len(digests),missing=missing,original_probability_reproduction_reused=True,new_checkpoint_inference=False,new_feature_extraction=False,new_fits=0,historical_outer_exposure_inherited=True,independent_reference_fit_replay_pass=False))
    guard();print('ASSET_AUDIT_COMPLETE_WITH_EXPLICIT_MISSING_STATES',flush=True)

if __name__=='__main__':main()
