"""Replay fixed training transforms to complete bank norm/hash metadata, no scores."""
from common import *
from typicality import Space
from threadpoolctl import threadpool_limits

def main():
    guard();rows=[];total_fits=0
    with threadpool_limits(limits=4):
      for c,i,sp in splits():
        allowed=list(map(str,sp['train_subjects']));lookup={s:j for j,s in enumerate(allowed)};raw=cache(allowed).reshape(len(allowed),22,1024)
        folder=HERE/'restricted_manifests'
        paths=list(folder.glob(f'B_strict_{c}_{i}_*.json'))+[folder/f'B_common_{c}_{i}.json']+list(folder.glob(f'Bfusion_OOF_{c}_{i}_fold*_B_common_{c}_{i}.json'))
        assert len(paths)==16
        for p in paths:
            record=json.loads(p.read_text());ids=record['train_subject_ids'];assert set(ids)<=set(allowed)
            x=raw[[lookup[s] for s in ids]]
            if 'B_common_' in p.name:
                bank=json.loads(p.with_name(p.name.replace('B_common_','B_banks_')).read_text());positions={s:j for j,s in enumerate(ids)};banks=[[positions[s] for s in bank[k]] for k in ['pd_subject_ids','dd_subject_ids']]
            else:banks=[list(range(len(ids)))]
            for ctx in range(22):
                space=Space().fit(x[:,ctx]);z=space.transform(x[:,ctx]);zero=np.linalg.norm(z,axis=1)<=1e-12
                rows.append(dict(context=c,inner=i,fit_namespace=p.stem,activity_wrist=ctx,fit_n=len(ids),rank=int(space.keep.sum()),zero_fitted_vectors=int(sum(zero)),zero_selected_bank_vectors=int(sum(zero[b].sum() for b in banks)),scaler_mean_sha256=hashlib.sha256(space.scaler.mean_.tobytes()).hexdigest(),PCA_components_sha256=hashlib.sha256(space.pca.components_.tobytes()).hexdigest()))
                total_fits+=1
        print(json.dumps(dict(stage='bank_norm_hash_audit',context=c,inner=i,status='COMPLETE')),flush=True)
    pd.DataFrame(rows).to_csv(HERE/'analysis/B_bank_norm_hash_audit.csv',index=False)
    write_json(HERE/'analysis/B_bank_norm_summary.json',dict(status='PASS',replayed_fixed_transform_fits=total_fits,selected_bank_zero_vectors=sum(x['zero_selected_bank_vectors'] for x in rows),fitted_training_zero_vectors=sum(x['zero_fitted_vectors'] for x in rows),minimum_rank=min(x['rank'] for x in rows),PCA_whitening=False,new_validation_performance_computed=False,new_candidate_models=False))
    guard();print('BANK_NORM_AND_TRANSFORM_HASH_AUDIT_PASS',flush=True)

if __name__=='__main__':main()
