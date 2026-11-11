"""One OOF temperature diagnostic and at most one gated two-score pipeline."""
from common import *
from scipy.special import expit
from scipy.optimize import minimize_scalar,minimize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import typicality as typ
import time

def nll(y,s):
    return float(np.mean(np.logaddexp(0,s)-np.asarray(y)*s))

def calibration(y,s):
    y=np.asarray(y);p=expit(s);decision=(s>0).astype(int);confidence=np.maximum(p,1-p)
    top=np.minimum((confidence*10).astype(int),9);bins=[];ece=0.;class_ece=0.
    for b in range(10):
        keep=top==b;n=int(keep.sum())
        acc=float(np.mean(decision[keep]==y[keep])) if n else None;conf=float(confidence[keep].mean()) if n else None
        if n:ece+=n/len(y)*abs(acc-conf)
        bins.append(dict(bin=b,lower=b/10,upper=(b+1)/10,n=n,accuracy=acc,confidence=conf))
    for b in range(10):
        keep=np.minimum((p*10).astype(int),9)==b
        if keep.any():class_ece+=keep.mean()*abs(float(y[keep].mean()-p[keep].mean()))
    def objective(beta):
        z=beta[0]+beta[1]*s;e=expit(z)-y
        return nll(y,z)+.5e-6*beta[1]**2,np.array([e.mean(),np.mean(e*s)+1e-6*beta[1]])
    fit=minimize(objective,np.array([0.,1.]),jac=True,method='BFGS',options={'gtol':1e-8,'maxiter':1000})
    assert np.isfinite(fit.x).all()
    return dict(nll=nll(y,s),brier=float(2*np.mean((p-y)**2)),ece=float(ece),classwise_ece=float(class_ece),calibration_intercept=float(fit.x[0]),calibration_slope=float(fit.x[1]),calibration_fit_gradient_norm=float(np.linalg.norm(fit.jac))),bins

def test_methods():
    y=np.array([0,1,0,1]);s=np.array([-2.,1.,-.4,2.])
    assert abs(nll(y,s)-np.mean(-np.log(np.where(y==1,expit(s),expit(-s)))))<1e-12
    for T in [.05,1.,20.]:
        assert np.array_equal(s>0,s/T>0) and roc_auc_score(y,s)==roc_auc_score(y,s/T)
    stats,bins=calibration(y,s);assert stats['nll']>0 and sum(b['n'] for b in bins)==4
    write_json(HERE/'tests/calibration_fusion_methods.json',dict(status='PASS',positive_temperature_monotone=True,fixed_decisions_preserved=True,hand_nll=True,probability_mapping_reference_not_used=True))

def reference_oof(c,i,target):
    # Same frozen reference function; only its output namespace is redirected.
    target_train=set(target['train_subjects']);target_val=set(target['validation_subjects']);scores={};bank_sizes=[]
    old_fit,old_write=typ.fit_ids,typ.write_json
    for k in range(3):
        stage=ROOT/f'artifacts/phase3c_ssl_adaptation_20260930/oof/seed42/outer_{c}/inner_{i}/crossfit_{k}'
        sp=json.loads((stage/'split.json').read_text());fit=list(map(str,sp['train_subjects']));query=list(map(str,sp['validation_subjects']))
        assert set(fit)|set(query)==target_train and not set(fit)&set(query) and not (set(fit)|set(query))&target_val
        namespace=f'Bfusion_OOF_{c}_{i}_fold{k}_'
        typ.fit_ids=lambda name,*a,**kw:old_fit(namespace+name,*a,**kw)
        typ.write_json=lambda path,obj:old_write(HERE/'restricted_manifests'/(namespace+Path(path).name),obj)
        try:
            x=cache(fit).reshape(len(fit),22,1024);q=cache(query).reshape(len(query),22,1024)
            y=labels(fit);out,thresholds,diagnostics=typ.common_reference(x,y,fit,q,query,c,i)
        finally:typ.fit_ids,typ.write_json=old_fit,old_write
        score=out['pd']-out['dd']
        for sid,v in zip(query,score):
            assert sid not in scores;scores[sid]=float(v)
        bank_sizes.append(int(np.bincount(y,minlength=2).min()))
    assert set(scores)==target_train
    return scores,bank_sizes

def main():
    guard();test_methods()
    oof=json.loads((HERE/'analysis/OOF_provenance.json').read_text());assert oof['status']=='PASS'
    b_gate=json.loads((HERE/'analysis/B_gate.json').read_text());fusion_allowed=b_gate['fusion_screen_pass'] and oof['reference_fusion_source_eligible']
    p=HERE/'configs/A2_fusion_freeze.json';assert not p.exists()
    write_json(p,dict(date='2026-11-11',files={str(p.relative_to(HERE)):sha(p) for p in [HERE/'PROTOCOL.md',Path(__file__),HERE/'scripts/common.py',HERE/'scripts/typicality.py',HERE/'tests/calibration_fusion_methods.json']},B_screen_results_already_known=True,OOF_provenance_already_known=True,temperature_or_fusion_validation_performance_seen=False,fusion_allowed=bool(fusion_allowed)))
    calibration_rows=[];reliability=[];transfer=[];fusion_rows=[];private=[];temperature=[];start=time.perf_counter()
    baseline_metrics=pd.read_csv(HERE/'analysis/baseline_metrics.csv').set_index(['seed','context','inner'])
    with threadpool_limits(limits=4):
      for c,i,target in splits():
        refs,bank_sizes=reference_oof(c,i,target) if fusion_allowed else ({},[])
        bval=pd.read_csv(HERE/'predictions'/f'B_context{c}_inner{i}.csv',dtype={'subject_id':str}).set_index('subject_id')
        for seed in [42,43,44]:
            with np.load(HERE/'features'/f'OOF_seed{seed}_context{c}_inner{i}.npz') as z:
                tids=z['subject_ids'].tolist();ty=z['target'].copy();ts=z['logit'].copy()
            assert set(tids)==set(target['train_subjects'])
            with np.load(HERE/'features'/f'seed{seed}_context{c}_inner{i}.npz') as z:
                keep=z['is_validation'];ids=z['subject_ids'][keep].tolist();y=z['target'][keep].copy();s=np.diff(z['logits'][keep].astype(np.float64),axis=1).ravel();train_s=np.diff(z['logits'][~keep].astype(np.float64),axis=1).ravel()
            base=pred(baseline(seed,c,i)/'predictions/validation.csv').loc[ids]
            assert np.array_equal((s>0).astype(int),base.prediction)
            fit_ids(f'A2_temperature_seed{seed}_{c}_{i}',tids,ids,target['train_subjects'],source='qualified deep OOF fixed epoch10')
            result=minimize_scalar(lambda logT:nll(ty,ts/np.exp(logT)),bounds=(np.log(.05),np.log(20)),method='bounded',options={'xatol':1e-10,'maxiter':1000})
            assert result.success and np.isfinite(result.fun);T=float(np.exp(result.x));assert T>0
            calibrated=s/T
            assert np.array_equal((calibrated>0).astype(int),base.prediction) and roc_auc_score(y,s)==roc_auc_score(y,calibrated)
            temperature.append(dict(context=c,inner=i,seed=seed,temperature=T,oof_nll_before=nll(ty,ts),oof_nll_after=nll(ty,ts/T),boundary_optimum=bool(T<.0501 or T>19.99)))
            for condition,score in [('raw',s),('positive_temperature',calibrated)]:
                values,bins=calibration(y,score);calibration_rows.append(dict(context=c,inner=i,seed=seed,condition=condition,**values))
                reliability.extend(dict(context=c,inner=i,seed=seed,condition=condition,**b) for b in bins)
            write_json(HERE/'restricted_manifests'/f'A2_descriptive_calibration_seed{seed}_{c}_{i}.json',dict(validation_subject_ids=ids,used_for_deployment=False,purpose='validation intercept/slope descriptive fit only; never temperature fit'))
            status=json.loads((baseline(seed,c,i)/'stage_status.json').read_text())
            row=dict(context=c,inner=i,seed=seed,OOF_train_pool_mean=float(np.mean([len(json.loads((ROOT/f'artifacts/phase3c_ssl_adaptation_20260930/oof/seed{seed}/outer_{c}/inner_{i}/crossfit_{k}/split.json').read_text())['train_subjects']) for k in range(3)])),full_train_pool=len(tids),OOF_epochs=10,full_best_epoch=status['summary']['best_epoch'],OOF_logit_mean=float(ts.mean()),OOF_logit_sd=float(ts.std(ddof=1)),full_train_in_sample_logit_sd=float(train_s.std(ddof=1)),validation_logit_mean=float(s.mean()),validation_logit_sd=float(s.std(ddof=1)))
            if fusion_allowed:
                rs=np.array([refs[sid] for sid in tids]);rv=bval.loc[ids,'relative_score'].to_numpy();design=np.column_stack([ts,rs]);validation=np.column_stack([s,rv])
                transform=StandardScaler().fit(design)
                fit_ids(f'Bfusion_fit_seed{seed}_{c}_{i}',tids,ids,target['train_subjects'],features=['WSSL deepOOF logit','whole-chainOOF relative typicality'],C=1,encoder_and_reference_entire_query_chain_out_of_fold=True)
                model=LogisticRegression(C=1,penalty='l2',solver='lbfgs',max_iter=1000,random_state=SEED).fit(transform.transform(design),ty)
                prob=model.predict_proba(transform.transform(validation));decision=prob.argmax(1)
                fusion_rows.append(dict(context=c,inner=i,seed=seed,**metric(y,prob[:,1],decision,prob=prob)))
                for sid,target_y,pdscore,ddscore,prediction in zip(ids,y,prob[:,0],prob[:,1],decision):private.append(dict(subject_id=sid,target=int(target_y),probability_pd=float(pdscore),probability_dd=float(ddscore),prediction=int(prediction),context=c,inner=i,seed=seed))
                out=HERE/'checkpoints';out.mkdir(exist_ok=True)
                np.savez_compressed(out/f'fusion_seed{seed}_context{c}_inner{i}.npz',scaler_mean=transform.mean_,scaler_scale=transform.scale_,coef=model.coef_,intercept=model.intercept_,temperature=T)
                row.update(reference_OOF_bank_min=min(bank_sizes),reference_OOF_bank_max=max(bank_sizes),reference_full_bank_min=int(np.bincount(ty,minlength=2).min()),reference_OOF_mean=float(rs.mean()),reference_OOF_sd=float(rs.std(ddof=1)),reference_validation_mean=float(rv.mean()),reference_validation_sd=float(rv.std(ddof=1)))
            transfer.append(row)
        print(json.dumps(dict(stage='A2_Bfusion',context=c,inner=i,status='COMPLETE',fusion_executed=bool(fusion_allowed))),flush=True)
    cals=pd.DataFrame(calibration_rows);cals.to_csv(HERE/'analysis/A2_calibration_metrics.csv',index=False)
    pd.DataFrame(reliability).to_csv(HERE/'analysis/A2_reliability_bins.csv',index=False)
    pd.DataFrame(temperature).to_csv(HERE/'analysis/A2_temperatures.csv',index=False)
    pd.DataFrame(transfer).to_csv(HERE/'analysis/A2_Bfusion_score_transfer.csv',index=False)
    cp=paired(cals[cals.condition=='positive_temperature'],cals[cals.condition=='raw'],['nll','brier','ece']);cp.to_csv(HERE/'analysis/A2_calibration_paired.csv',index=False)
    calibration_retained=bool(cp.set_index('metric').loc['nll','mean_delta']<0 and cp.set_index('metric').loc['brier','mean_delta']<=0)
    fusion_decision='NOT_RUN';gate={}
    if fusion_allowed:
        frame=pd.DataFrame(fusion_rows);assert len(frame)==45
        frame.to_csv(HERE/'analysis/B_fusion_metrics.csv',index=False);pd.DataFrame(private).to_csv(HERE/'predictions/B_fusion_predictions.csv',index=False)
        bp=paired(frame,pd.read_csv(HERE/'analysis/baseline_metrics.csv'));bp.to_csv(HERE/'analysis/B_fusion_paired.csv',index=False)
        p=bp.set_index('metric');gate=dict(BA_positive=bool(p.loc['ba','mean_delta']>0),BA_10wins=bool(p.loc['ba','wins']>=10),BA_CI_positive=bool(p.loc['ba','ci_low']>0),AUROC_nondecrease=bool(p.loc['auroc','mean_delta']>=0),F1_nondecrease=bool(p.loc['macro_f1','mean_delta']>=0),PD_recall_guard=bool(p.loc['pd_recall','mean_delta']>=-.01),DD_recall_guard=bool(p.loc['dd_recall','mean_delta']>=-.01))
        base=pd.read_csv(HERE/'analysis/baseline_metrics.csv')
        for m in ['ba','auroc']:
            bs=base.groupby(['context','inner'])[m].std(ddof=1).mean();fs=frame.groupby(['context','inner'])[m].std(ddof=1).mean();gate[m+'_seed_stability']=bool(fs<=1.25*bs)
        fusion_decision='RETAIN' if all(gate.values()) else ('REJECT' if not gate['BA_positive'] or not gate['PD_recall_guard'] or not gate['DD_recall_guard'] else 'INCONCLUSIVE')
    guard();write_json(HERE/'analysis/A2_Bfusion_decision.json',dict(status='COMPLETE',temperature_diagnostic_fits=45,descriptive_calibration_fits=90,temperature_sidecar_decision='RETAIN' if calibration_retained else 'REJECT',temperature_changes_AUROC_or_fixed_decisions=False,fusion_source_eligible=True,fusion_screen_pass=bool(b_gate['fusion_screen_pass']),fusion_pipeline_fits=45 if fusion_allowed else 0,fusion_decision=fusion_decision,fusion_gate=gate,retained_native_baseline_changed=False,new_deep_training_runs=0,elapsed_seconds=time.perf_counter()-start))
    print(json.dumps(dict(stage='A2_Bfusion',fusion_decision=fusion_decision,calibration_sidecar='RETAIN' if calibration_retained else 'REJECT')),flush=True)

if __name__=='__main__':main()
