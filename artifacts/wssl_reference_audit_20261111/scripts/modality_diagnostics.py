from common import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

def project(train,test):
    scale=StandardScaler().fit(np.asarray(train,np.float64));a=scale.transform(train);b=scale.transform(test)
    pca=PCA(n_components=min(16,len(train)-1,train.shape[1]),whiten=False,svd_solver='randomized',iterated_power=4,random_state=SEED).fit(a)
    return pca.transform(a),pca.transform(b)

def main():
    guard();freeze('A1',[HERE/'PROTOCOL.md',Path(__file__),HERE/'scripts/common.py'])
    predictions=[];runs=[];corrections=[];adapted_rows=[];probe_predictions=[]
    with threadpool_limits(limits=4):
      for c,i,split in splits():
        for seed in [42,43,44]:
            f=pred(baseline(seed,c,i)/'predictions/validation.csv').copy();f['context']=c;f['inner']=i;f['seed']=seed
            predictions.append(f.reset_index()[['subject_id','target','prediction','probability_pd','probability_dd','context','inner','seed']])
            adapted=ROOT/f'artifacts/phase3c_ssl_adaptation_20260930/runs/adapt/p100/seed{seed}/outer_{c}/inner_{i}'
            af=pred(adapted/'predictions/validation.csv').loc[f.index]
            assert np.array_equal(af.target,f.target)
            normal=json.loads((adapted/'normalization.json').read_text())
            base_normal=json.loads((baseline(seed,c,i)/'normalization.json').read_text())
            assert normal['normalization_sha256']==base_normal['normalization_sha256']
            ap=af[['probability_pd','probability_dd']].to_numpy()
            adapted_rows.append(dict(context=c,inner=i,seed=seed,**metric(af.target,af.probability_dd,prob=ap),score_spearman=float(spearmanr(af.probability_dd,f.probability_dd).statistic),prediction_agreement=float((af.prediction==f.prediction).mean())))
            for label in [0,1]:
                mask=f.target.to_numpy()==label;old=f.prediction.to_numpy()[mask]==label;new=af.prediction.to_numpy()[mask]==label
                corrections.append(dict(context=c,inner=i,seed=seed,label=label,corrected=int(sum(new&~old)),newly_wrong=int(sum(~new&old)),net=int(sum(new)-sum(old)),total=int(sum(mask))))
            with np.load(HERE/'features'/f'seed{seed}_context{c}_inner{i}.npz') as z:
                ids=z['subject_ids'].tolist();y=z['target'].copy();validation=z['is_validation'].copy();local=z['local'].reshape(len(ids),-1).copy();score=np.diff(z['logits'].astype(np.float64),axis=1).ravel()
            assert set(np.array(ids)[~validation])==set(split['train_subjects']) and set(np.array(ids)[validation])==set(split['validation_subjects'])
            assert np.array_equal(y,labels(ids)) and local.shape[1]==1408
            fit_ids(f'A1_probe_seed{seed}_{c}_{i}',np.array(ids)[~validation],np.array(ids)[validation],split['train_subjects'],PCA_dim=16,logistic_C=1,underlying_encoder_saw_fit_subjects=True,whole_chain_OOF=False)
            ssl=cache(ids).reshape(len(ids),-1)
            st,sv=project(ssl[~validation],ssl[validation]);lt,lv=project(local[~validation],local[validation])
            for condition,train,test in [
                ('score_ssl_control',np.column_stack([score[~validation],st]),np.column_stack([score[validation],sv])),
                ('score_ssl_plus_local',np.column_stack([score[~validation],st,lt]),np.column_stack([score[validation],sv,lv]))]:
                transform=StandardScaler().fit(train);model=LogisticRegression(C=1,penalty='l2',solver='lbfgs',max_iter=1000,random_state=SEED).fit(transform.transform(train),y[~validation])
                p=model.predict_proba(transform.transform(test));decision=p.argmax(1)
                runs.append(dict(context=c,inner=i,seed=seed,condition=condition,**metric(y[validation],p[:,1],decision,prob=p)))
                for sid,target,pdscore,ddscore,pr in zip(np.array(ids)[validation],y[validation],p[:,0],p[:,1],decision):probe_predictions.append(dict(subject_id=sid,target=int(target),probability_pd=float(pdscore),probability_dd=float(ddscore),prediction=int(pr),context=c,inner=i,seed=seed,condition=condition))
        print(json.dumps(dict(stage='A1',context=c,inner=i,status='COMPLETE')),flush=True)
    f=pd.concat(predictions,ignore_index=True);assert not f.duplicated(['subject_id','context','inner','seed']).any()
    appearance=f.groupby(['subject_id','context','inner'],as_index=False).agg(target=('target','first'),probability_pd=('probability_pd','mean'),probability_dd=('probability_dd','mean'),seed_sd=('probability_dd','std'))
    appearance['prediction']=appearance[['probability_pd','probability_dd']].to_numpy().argmax(1);appearance['wrong']=(appearance.prediction!=appearance.target).astype(int)
    subject=appearance.groupby('subject_id').agg(target=('target','first'),appearances=('wrong','size'),error_fraction=('wrong','mean'),mean_probability=('probability_dd','mean'),split_probability_sd=('probability_dd','std'),within_appearance_seed_sd=('seed_sd','mean'))
    assert (subject.appearances==4).all() and len(subject)==390
    subject['error_group']=np.where(subject.error_fraction>=.75,'stable-error',np.where(subject.error_fraction<=.25,'stable-correct','unstable'))
    manifest=pd.read_csv(ROOT/'data/processed/pads_multi_activity/v2_l1_full_length/manifests/CrossArms.csv',dtype={'subject_id':str}).set_index('subject_id')
    subject['source_category']=manifest.loc[subject.index,'source_condition']
    subject['all4_wrong']=subject.error_fraction==1
    strict=f.groupby('subject_id').apply(lambda q:bool((q.prediction!=q.target).all()),include_groups=False)
    subject['all12_wrong']=strict
    subject['absolute_probability_margin']=abs(subject.mean_probability-.5)
    subject.to_csv(HERE/'predictions/A1_subject_error_groups.csv')
    appearance.to_csv(HERE/'predictions/A1_appearance_predictions.csv',index=False)
    counts=subject.groupby(['target','error_group']).size().unstack(fill_value=0)
    counts.to_csv(HERE/'analysis/A1_error_group_counts.csv')
    summary=subject.groupby(['target','source_category']).agg(n=('mean_probability','size'),mean_probability=('mean_probability','mean'),mean_absolute_margin=('absolute_probability_margin','mean'),mean_error_fraction=('error_fraction','mean'),stable_error_n=('error_group',lambda q:int((q=='stable-error').sum())),stable_correct_n=('error_group',lambda q:int((q=='stable-correct').sum())),split_probability_sd=('split_probability_sd','mean'),within_seed_sd=('within_appearance_seed_sd','mean'))
    summary.to_csv(HERE/'analysis/A1_category_error_summary.csv')
    write_json(HERE/'analysis/A1_error_definitions.json',dict(primary_stable_error=int((subject.error_group=='stable-error').sum()),primary_PD=int(((subject.error_group=='stable-error')&(subject.target==0)).sum()),primary_DD=int(((subject.error_group=='stable-error')&(subject.target==1)).sum()),all4_wrong=int(subject.all4_wrong.sum()),all4_PD=int(subject.loc[subject.target==0,'all4_wrong'].sum()),all4_DD=int(subject.loc[subject.target==1,'all4_wrong'].sum()),all12_wrong=int(subject.all12_wrong.sum()),definition='seed-probability average per appearance, then4appearance error proportion>=.75',validation_defined_groups_not_used_for_any_fit=True))
    pd.DataFrame(adapted_rows).to_csv(HERE/'analysis/A1_adapted_metrics.csv',index=False)
    pd.DataFrame(corrections).to_csv(HERE/'analysis/A1_adapted_corrections.csv',index=False)
    raw=pd.DataFrame(runs);raw.to_csv(HERE/'analysis/A1_probe_metrics.csv',index=False)
    comparison=paired(raw[raw.condition=='score_ssl_plus_local'],raw[raw.condition=='score_ssl_control']);comparison.to_csv(HERE/'analysis/A1_probe_paired.csv',index=False)
    pd.DataFrame(probe_predictions).to_csv(HERE/'predictions/A1_probe_predictions.csv',index=False)
    adapted_cmp=paired(pd.DataFrame(adapted_rows),pd.read_csv(HERE/'analysis/baseline_metrics.csv'));adapted_cmp.to_csv(HERE/'analysis/A1_adapted_paired.csv',index=False)
    guard();write_json(HERE/'analysis/A1_complete.json',dict(status='COMPLETE',deep_training_runs=0,new_logistic_fits=90,local_representation='ordered pre-fusion11x2x64 mixed Acc+Gyro',Gyro_exclusive_increment_confirmed=False,underlying_encoder_fit_subjects_seen=True,whole_chain_OOF=False,contribution_matrix_repeated=False,adapted_checkpoints_retrained=False))
    print('A1_COMPLETE',flush=True)

if __name__=='__main__':main()
