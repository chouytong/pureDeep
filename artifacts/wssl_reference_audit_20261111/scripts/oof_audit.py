"""Inventory135 actual deep-OOF states and full source isolation, no retraining."""
from common import *
import torch,gc
from independent_wssl import IndependentWSSL
from src.datasets import folds as fd
from src.utils.config import load_config
from src.utils.provenance import sha256_json

def main():
    torch.set_num_threads(4);guard()
    freeze('OOF_audit',[HERE/'PROTOCOL.md',Path(__file__),HERE/'scripts/common.py'])
    root=ROOT/'artifacts/phase3c_ssl_adaptation_20260930/oof'
    rows=[];extremes=0;norm_refits=0
    for c,i,target in splits():
        target_train=set(target['train_subjects']);target_val=set(target['validation_subjects']);normalizers={}
        for seed in [42,43,44]:
            pieces=[]
            for k in range(3):
                stage=root/f'seed{seed}/outer_{c}/inner_{i}/crossfit_{k}'
                required=['checkpoints/best.pt','predictions/validation.csv','split.json','normalization.json','config.yaml','logs/epochs.jsonl','stage_status.json']
                assert all((stage/p).is_file() for p in required),'Missing real OOF asset'
                ck=torch.load(stage/'checkpoints/best.pt',map_location='cpu',weights_only=False)
                split=json.loads((stage/'split.json').read_text());normal=json.loads((stage/'normalization.json').read_text());status=json.loads((stage/'stage_status.json').read_text())
                train=set(split['train_subjects']);query=set(split['validation_subjects'])
                assert train|query==target_train and not train&query and not (train|query)&target_val
                assert not split['test_subjects'] and status['outer_test_loader_created'] is False
                assert set(normal['subject_ids'])==train and normal['subject_ids_sha256']==sha256_json(sorted(train))
                assert normal['fitted_on']=='explicit_training_subjects_only'
                for key in ['mean','std']:assert np.array_equal(ck['normalization'][key].numpy(),np.array(normal[key]))
                history=[json.loads(s) for s in (stage/'logs/epochs.jsonl').read_text().splitlines()]
                assert ck['epoch']==9 and len(history)==10 and status['summary']['best_epoch']==10
                assert all(record['improved'] for record in history),'OOF must use fixed final epoch, not query-selected best'
                assert ck['config']['training']['epochs']==10 and ck['config']['training']['early_stopping']['enabled'] is False
                assert ck['scheduler_state']['T_max']==50
                assert status['summary']['checkpoint_sha256']==sha(stage/'checkpoints/best.pt')
                cfg=load_config(MAIN/f'configs/str01_seed{seed}.yaml')
                if k not in normalizers:
                    allowed=train|query;original=fd.load_configured_records
                    fd.load_configured_records=lambda *a,**kw:[r for r in original(*a,**kw) if r.subject_id in allowed]
                    try:bundle=fd.build_subject_fold_datasets(cfg,train_subject_ids=sorted(train),validation_subject_ids=sorted(query),test_subject_ids=(),fold_id=f'OOF_{c}_{i}_{k}')
                    finally:fd.load_configured_records=original
                    assert bundle.test is None
                    normalizers[k]=(bundle.mean.clone(),bundle.std.clone(),sorted(train));norm_refits+=1
                    del bundle
                assert all(torch.equal(normalizers[k][j],ck['normalization'][key]) for j,key in enumerate(['mean','std']))
                assert normalizers[k][2]==sorted(train)
                model=IndependentWSSL(cfg);model.load_state_dict(ck['model_state'],strict=True)
                assert sum(p.numel() for p in model.parameters())==143172
                assert all(torch.isfinite(v).all() for v in model.state_dict().values())
                f=pred(stage/'predictions/validation.csv')
                assert set(f.index)==query and np.array_equal(f.target,labels(f.index))
                probability=f[['probability_pd','probability_dd']].to_numpy()
                extreme=int(np.any((probability<=0)|(probability>=1),axis=1).sum());extremes+=extreme
                if extreme:
                    raise RuntimeError('Archived OOF endpoint probability requires frozen-forward score recovery before calibration; no clipping allowed')
                logit=np.log(probability[:,1])-np.log(probability[:,0])
                pieces.append(pd.DataFrame(dict(subject_id=f.index,target=f.target.to_numpy(),probability_dd=f.probability_dd.to_numpy(),logit=logit,crossfit=k)))
                fit_ids(f'OOF_seed{seed}_{c}_{i}_fold{k}',sorted(train),sorted(query),target_train,
                        target_validation_subject_ids=sorted(target_val),checkpoint_sha256=sha(stage/'checkpoints/best.pt'),normalization_sha256=status['summary']['normalization_sha256'],fixed_epoch=10,query_not_used_for_selection=True)
                rows.append(dict(seed=seed,context=c,inner=i,crossfit=k,train_n=len(train),query_n=len(query),checkpoint_exists=True,strict_state_load=True,query_encoder_train_disjoint=True,query_normalizer_fit_disjoint=True,target_validation_excluded=True,selection='fixed_epoch10',pca_bank_calibrator_at_original_stage='not present',normalization_refit_exact=True,checkpoint_sha256=sha(stage/'checkpoints/best.pt'),prediction_sha256=sha(stage/'predictions/validation.csv'),endpoint_probability_rows=extreme))
                del model,ck;gc.collect()
            f=pd.concat(pieces).sort_values('subject_id');assert f.subject_id.is_unique and set(f.subject_id)==target_train
            archived=pd.read_csv(root/f'seed{seed}/outer_{c}/inner_{i}/oof_predictions.csv',dtype={'subject_id':str}).set_index('subject_id').loc[f.subject_id]
            assert np.array_equal(archived.target.to_numpy(),f.target.to_numpy())
            assert np.allclose(archived.probability_dd.to_numpy(),f.probability_dd.to_numpy(),atol=1e-12,rtol=0)
            np.savez_compressed(HERE/'features'/f'OOF_seed{seed}_context{c}_inner{i}.npz',subject_ids=f.subject_id.to_numpy(dtype=str),target=f.target.to_numpy(),logit=f.logit.to_numpy(),crossfit=f.crossfit.to_numpy())
        print(json.dumps(dict(stage='OOF_audit',context=c,inner=i,status='PASS')),flush=True)
    assert len(rows)==135
    pd.DataFrame(rows).to_csv(HERE/'analysis/OOF_inventory_135.csv',index=False)
    guard();write_json(HERE/'analysis/OOF_provenance.json',dict(status='PASS',models_actually_available=135,prediction_sets_available=135,strict_state_loads=135,normalization_refits=norm_refits,normalization_refits_exact=True,entire_query_training_chain_disjoint=True,all_target_validation_subjects_excluded=True,selection='fixed epoch10, forced last rather than query-metric choice',scope='3 folds within each target inner-train only',endpoint_probability_rows=extremes,new_deep_models=0,temperature_eligible=True,reference_fusion_source_eligible=True,score_transfer_limitation='OOF small train pool fixed epoch10 vs full train pool best-epoch; not identical score distribution'))
    print('OOF135_SOURCE_CHAIN_PASS',flush=True)

if __name__=='__main__':main()
