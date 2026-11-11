"""New descriptive statistics from saved scores only; all fit/re-inference forbidden."""
from common import *
from scipy.special import expit
from scipy.stats import ks_2samp,wasserstein_distance
from itertools import combinations
from threadpoolctl import threadpool_limits
import time

CORE=['wssl_logit','d_pd','dd_aligned','relative']

def kernel(dd,pd):
    dd=np.asarray(dd);pd=np.asarray(pd)
    return (dd[:,None]>pd[None,:]).astype(float)+.5*(dd[:,None]==pd[None,:])

def macro_kernel(f,prefix):
    y=f.target.to_numpy();dd=f[y==1];pd=f[y==0]
    if prefix=='relative':return kernel(dd.relative,pd.relative)
    return .5*(kernel(dd[prefix+'_pdd'],pd[prefix+'_pdd'])+kernel(-dd[prefix+'_ppd'],-pd[prefix+'_ppd']))

def weighted_metrics(f,prefix,a,b):
    y=f.target.to_numpy();pr=f[prefix+'_prediction'].to_numpy();n0=a.sum(1);n1=b.sum(1)
    tn=a@(pr[y==0]==0);tp=b@(pr[y==1]==1);fp=n0-tn;fn=n1-tp
    auc=np.sum((b@macro_kernel(f,prefix))*a,axis=1)/(n0*n1)
    return np.column_stack([(tn+tp)/(n0+n1),.5*(tn/n0+tp/n1),auc,.5*(2*tn/(2*tn+fp+fn)+2*tp/(2*tp+fp+fn)),tn/n0,tp/n1])

def transitions(f,c,i,seed,scope,kind='relative'):
    rows=[]
    for group in ['all','stable-error','stable-correct','unstable']:
      for label in [-1,0,1]:
        m=np.ones(len(f),bool)
        if group!='all':m &= f.error_group.to_numpy()==group
        if label!=-1:m &= f.target.to_numpy()==label
        t=f[m];a=t.baseline_prediction.to_numpy()==t.target.to_numpy();b=t[kind+'_prediction'].to_numpy()==t.target.to_numpy();n=len(t)
        rows.append(dict(context=c,inner=i,seed=seed,aggregation=scope,error_group=group,label='all' if label==-1 else ('PD' if label==0 else 'DD'),n=n,both_correct=int(sum(a&b)),baseline_correct_candidate_wrong=int(sum(a&~b)),baseline_wrong_candidate_correct=int(sum(~a&b)),both_wrong=int(sum(~a&~b)),gross_rescue=int(sum(~a&b)),newly_broken=int(sum(a&~b)),net_rescue=int(sum(b)-sum(a)),net_rescue_rate=float((sum(b)-sum(a))/n) if n else np.nan))
    return rows

def synthetic_tests():
    x=np.array([[1.,-1.],[0.,0.],[-1.,1.]])
    mu=np.array([.2,-.3]);scale=np.array([2.,.4]);coef=np.array([1.2,.8]);intercept=-.7
    f=intercept+(x-mu)/scale@coef
    assert np.allclose(expit(f),1/(1+np.exp(-f)),atol=1e-15)
    assert int(np.array([.5,.5]).argmax())==0
    for bad,allowed in [(['a','a'],['a']),(['z'],['a'])]:
        try:require_ids(bad,allowed)
        except AssertionError:pass
        else:raise AssertionError('ID test failed')
    assert kernel([0.,1.],[0.,1.]).tolist()==[[.5,0.],[1.,.5]]
    assert not np.isfinite(correlation([1,1,1],[1,2,3]))
    toy=pd.DataFrame(dict(target=[0,0,1,1],baseline_prediction=[0,1,0,1],baseline_pdd=[.1,.7,.3,.9],baseline_ppd=[.9,.3,.7,.1]))
    got=weighted_metrics(toy,'baseline',np.ones((1,2)),np.ones((1,2)))[0];expected=metrics(toy.target,toy.baseline_prediction,toy.baseline_pdd,toy[['baseline_ppd','baseline_pdd']].to_numpy())
    assert np.max(abs(got-np.array([expected[m] for m in METRICS])))<1e-12
    write_json(HERE/'tests/synthetic_correctness.json',dict(status='PASS',explicit_ID_allowlist=True,duplicate_rejected=True,probability_formula=True,tie_to_PD=True,weighted_metrics_match_canonical=True,rank_tie_half=True,degenerate_correlation_NA=True,no_fit_calls=True))

def main():
    guard();synthetic_tests();started=time.perf_counter();asset=load(HERE/'analysis/asset_audit.json');assert asset['baseline']==45 and asset['adapted']==45 and asset['historical_OOF']==135
    groupmap={}
    allowed_global=set(s for c,i,sp,ex in splits() for s in sp['validation_subjects'])
    with (OLD/'predictions/A1_subject_error_groups.csv').open() as stream:
      for row in csv.DictReader(stream):
        sid=row['subject_id'];assert sid in allowed_global and sid not in groupmap;groupmap[sid]=row
    assert len(groupmap)==390
    score_rows=[];corr_rows=[];coverage_rows=[];coverage_corr=[];comp=[];adapt=[];pairs=[];bias=[];parameters=[];shifts=[];same_subject=[];drift=[];allframes=[];ensframes=[];formal=[]
    prior_transfer=pd.read_csv(OLD/'analysis/A2_Bfusion_score_transfer.csv').set_index(['seed','context','inner'])
    prior_metrics={k:pd.read_csv(OLD/'analysis'/n) for k,n in [('baseline','baseline_metrics.csv'),('fusion','B_fusion_metrics.csv'),('adapted','A1_adapted_metrics.csv')]}
    prior_relative=pd.read_csv(OLD/'analysis/B_metrics.csv');prior_relative=prior_relative[prior_relative.condition=='relative'].set_index(['context','inner'])
    maxdiff=0.;mean_scale_error=0.
    with threadpool_limits(limits=4):
      for c,i,sp,excluded in splits():
        allowed=set(sp['validation_subjects']);r=pd.read_csv(OLD/'predictions'/f'B_context{c}_inner{i}.csv',dtype={'subject_id':str}).set_index('subject_id').sort_index();require_ids(r.index,allowed,allowed)
        assert np.max(abs(r.relative_score-(r.d_pd-r.d_dd)))<=1e-12 and np.max(abs(r.min_distance-np.minimum(r.d_pd,r.d_dd)))<=1e-12
        assert np.array_equal(r.relative_prediction,(r.relative_score>0).astype(int))
        lo,hi=np.quantile(r.min_distance,[.25,.75]);assert hi>lo
        unit=[]
        for seed in [42,43,44]:
            base=prediction(stage('baseline',seed,c,i)/'predictions/validation.csv',allowed)
            ad=prediction(stage('adapted',seed,c,i)/'predictions/validation.csv',allowed)
            fu=fusion_rows(c,i,seed,allowed);fs=feature_scores(seed,c,i,sp);oo=oof_scores(seed,c,i,sp)
            require_ids(base.index,allowed,allowed);ids=base.index
            assert np.array_equal(base.target,ad.target) and np.array_equal(base.target,fu.target) and np.array_equal(base.target,r.loc[ids].target) and np.array_equal(base.target,fs.loc[ids].target)
            tr=fs[~fs.is_validation].loc[oo.index];assert np.array_equal(tr.target,oo.target)
            assert all(int(t)==int(groupmap[s]['target']) for s,t in base.target.items())
            state=OLD/'checkpoints'/f'fusion_seed{seed}_context{c}_inner{i}.npz'
            with np.load(state,allow_pickle=False) as z:mu=z['scaler_mean'].copy();scale=z['scaler_scale'].copy();w=z['coef'].ravel().copy();b=float(z['intercept'][0])
            assert mu.shape==scale.shape==w.shape==(2,) and np.isfinite(np.r_[mu,scale,w,b]).all() and (scale>0).all()
            pp=prior_transfer.loc[(seed,c,i)]
            assert abs(mu[0]-oo.wssl_logit.mean())<=1e-12 and abs(scale[0]-oo.wssl_logit.std(ddof=0))<=1e-12
            n=len(oo);err=max(abs(mu[1]-pp.reference_OOF_mean),abs(scale[1]-pp.reference_OOF_sd*np.sqrt((n-1)/n)));assert err<=1e-12;mean_scale_error=max(mean_scale_error,err)
            design=np.column_stack([fs.loc[ids].wssl_logit,r.loc[ids].relative_score]);standard=(design-mu)/scale;terms=standard*w;fl=b+terms.sum(1);prob=expit(fl);d=float(np.max(abs(prob-fu.probability_dd)));maxdiff=max(maxdiff,d);assert d<=1e-12 and np.array_equal((fl>0).astype(int),fu.prediction)
            assert np.all(base.probability_pd>0) and np.all(base.probability_dd>0) and np.all(ad.probability_pd>0) and np.all(ad.probability_dd>0),'Probability endpoints block log-odds, do not clip'
            f=pd.DataFrame(dict(subject_id=ids,target=base.target.to_numpy(),context=c,inner=i,seed=seed,error_group=[groupmap[s]['error_group'] for s in ids],baseline_prediction=base.prediction.to_numpy(),baseline_pdd=base.probability_dd.to_numpy(),baseline_ppd=base.probability_pd.to_numpy(),wssl_logit=design[:,0],wssl_raw_dd_logit=fs.loc[ids].raw_dd_logit.to_numpy(),d_pd=r.loc[ids].d_pd.to_numpy(),d_dd=r.loc[ids].d_dd.to_numpy(),dd_aligned=-r.loc[ids].d_dd.to_numpy(),relative=design[:,1],relative_prediction=r.loc[ids].relative_prediction.to_numpy(),u=r.loc[ids].min_distance.to_numpy(),fusion_pdd=fu.probability_dd.to_numpy(),fusion_ppd=fu.probability_pd.to_numpy(),fusion_prediction=fu.prediction.to_numpy(),fusion_logit=fl,wssl_standardized=standard[:,0],reference_standardized=standard[:,1],W_contribution=terms[:,0],R_contribution=terms[:,1],fusion_intercept=b,adapted_prediction=ad.prediction.to_numpy(),adapted_pdd=ad.probability_dd.to_numpy(),adapted_ppd=ad.probability_pd.to_numpy()))
            f['wssl_probability_margin']=abs(f.baseline_pdd-.5);f['relative_absolute_margin']=abs(f.relative)
            f['baseline_probability_implied_logodds']=np.log(f.baseline_pdd/f.baseline_ppd);f['adapted_probability_implied_logodds']=np.log(f.adapted_pdd/f.adapted_ppd)
            f['adapted_logodds_change']=f.adapted_probability_implied_logodds-f.baseline_probability_implied_logodds
            f['adapted_absolute_margin_change']=abs(f.adapted_pdd-.5)-abs(f.baseline_pdd-.5)
            f['adapted_true_signed_margin_change']=(2*f.target-1)*(f.adapted_probability_implied_logodds-f.baseline_probability_implied_logodds)
            f['coverage_group']=np.where(f.u<=lo,'low',np.where(f.u>=hi,'high','middle'))
            allframes.append(f);unit.append(f)
            for kind in ['baseline','relative','fusion','adapted']:
                probs=None if kind=='relative' else f[[kind+'_ppd',kind+'_pdd']].to_numpy();score=f.relative if kind=='relative' else f[kind+'_pdd']
                m=metrics(f.target,f[kind+'_prediction'],score,probs)
                expected=prior_relative.loc[(c,i)] if kind=='relative' else prior_metrics[kind].set_index(['seed','context','inner']).loc[(seed,c,i)]
                assert all(abs(m[k]-expected[k])<=1e-12 for k in METRICS),(kind,c,i,seed)
                formal.append(dict(condition=kind,seed=seed,context=c,inner=i,**m,interpretation='archived predictions metric identity check; no rerun of historical models'))
            for label in [-1,0,1]:
                t=f if label==-1 else f[f.target==label];lab='all' if label==-1 else ('PD' if label==0 else 'DD')
                for name in ['wssl_logit','wssl_raw_dd_logit','baseline_pdd','d_pd','d_dd','dd_aligned','relative','u','wssl_probability_margin','relative_absolute_margin']:
                    score_rows.append(dict(context=c,inner=i,seed=seed,label=lab,score=name,**descript(t[name])))
                for a,bb in combinations(CORE,2):
                  for method in ['pearson','spearman']:corr_rows.append(dict(context=c,inner=i,seed=seed,label=lab,left=a,right=bb,method=method,correlation=correlation(t[a],t[bb],method)))
                for name in ['wssl_probability_margin','relative','relative_absolute_margin']:coverage_corr.append(dict(context=c,inner=i,seed=seed,label=lab,left='u',right=name,spearman=correlation(t.u,t[name])))
                low=t[t.coverage_group=='low'];high=t[t.coverage_group=='high'];le=float((low.baseline_prediction!=low.target).mean()) if len(low) else np.nan;he=float((high.baseline_prediction!=high.target).mean()) if len(high) else np.nan
                coverage_rows.append(dict(context=c,inner=i,seed=seed,label=lab,q25=lo,q75=hi,low_n=len(low),high_n=len(high),low_error_rate=le,high_error_rate=he,high_minus_low_error_rate=he-le,scope='validation-defined quartiles, descriptive not deployment'))
                yd=t.target.to_numpy();mb=t.baseline_prediction.to_numpy()==yd;mf=t.fusion_prediction.to_numpy()==yd
                for event,keep in [('all',np.ones(len(t),bool)),('both_correct',mb&mf),('newly_broken',mb&~mf),('rescued',~mb&mf),('both_wrong',~mb&~mf)]:
                    q=t[keep];bias.append(dict(context=c,inner=i,seed=seed,label=lab,event=event,n=len(q),baseline_DD_prediction_rate=float((q.baseline_prediction==1).mean()),fusion_DD_prediction_rate=float((q.fusion_prediction==1).mean()),mean_baseline_pDD=q.baseline_pdd.mean(),mean_fusion_pDD=q.fusion_pdd.mean(),mean_W_input=q.wssl_logit.mean(),mean_reference=q.relative.mean(),mean_W_standardized=q.wssl_standardized.mean(),mean_reference_standardized=q.reference_standardized.mean(),mean_W_contribution=q.W_contribution.mean(),mean_R_contribution=q.R_contribution.mean(),intercept=b,mean_fusion_logit=q.fusion_logit.mean()))
                for score in ['wssl_logit']:
                    qo=oo if label==-1 else oo[oo.target==label];qv=fs.loc[ids] if label==-1 else fs.loc[ids][fs.loc[ids].target==label]
                    so=qo[score].to_numpy();sv=qv[score].to_numpy();sd=so.std(ddof=0)
                    shifts.append(dict(context=c,inner=i,seed=seed,label=lab,score=score,OOF_n=len(so),full_validation_n=len(sv),OOF_mean=so.mean(),OOF_sd=so.std(ddof=1),OOF_q05=np.quantile(so,.05),OOF_median=np.median(so),OOF_q95=np.quantile(so,.95),full_validation_mean=sv.mean(),full_validation_sd=sv.std(ddof=1),full_q05=np.quantile(sv,.05),full_median=np.median(sv),full_q95=np.quantile(sv,.95),standardized_mean_shift=(sv.mean()-so.mean())/sd,KS_statistic=ks_2samp(so,sv).statistic,Wasserstein_in_OOF_SD=wasserstein_distance(so,sv)/sd,source='saved individual OOF and full validation scores',per_class_joint_reference_available=False))
            shifts.append(dict(context=c,inner=i,seed=seed,label='all',score='relative',OOF_n=n,full_validation_n=len(f),OOF_mean=pp.reference_OOF_mean,OOF_sd=pp.reference_OOF_sd,full_validation_mean=f.relative.mean(),full_validation_sd=f.relative.std(ddof=1),standardized_mean_shift=f.reference_standardized.mean(),source='saved OOF aggregate/Scaler only; OOF rows MISSING',KS_statistic=np.nan,Wasserstein_in_OOF_SD=np.nan,per_class_joint_reference_available=False,reference_OOF_bank_min=pp.reference_OOF_bank_min,reference_OOF_bank_max=pp.reference_OOF_bank_max,reference_full_bank_min=pp.reference_full_bank_min))
            same_subject.append(dict(context=c,inner=i,seed=seed,n=n,OOF_training_pool_mean=pp.OOF_train_pool_mean,full_training_pool=pp.full_train_pool,OOF_epochs=pp.OOF_epochs,full_best_epoch=pp.full_best_epoch,paired_full_train_in_sample_minus_OOF_mean=float((tr.wssl_logit-oo.wssl_logit).mean()),paired_absolute_change=float(abs(tr.wssl_logit-oo.wssl_logit).mean()),OOF_sd=float(oo.wssl_logit.std(ddof=1)),full_train_in_sample_sd=float(tr.wssl_logit.std(ddof=1)),warning='same-subject diagnostic; full model saw these train subjects, not validation'))
            ratio=float((oo.target==1).mean());prior_logit=float(np.log(ratio/(1-ratio)))
            parameters.append(dict(context=c,inner=i,seed=seed,W_coefficient=w[0],R_coefficient=w[1],intercept=b,W_scaler_mean=mu[0],R_scaler_mean=mu[1],W_scaler_scale=scale[0],R_scaler_scale=scale[1],training_n=n,training_DD_prevalence=ratio,training_prior_log_odds=prior_logit,intercept_minus_prior_logodds=b-prior_logit,validation_DD_prevalence=float(f.target.mean()),W_standardized_validation_mean=standard[:,0].mean(),R_standardized_validation_mean=standard[:,1].mean(),W_mean_contribution=terms[:,0].mean(),R_mean_contribution=terms[:,1].mean(),validation_mean_fusion_logit=fl.mean(),OOF_mean_fusion_logit_implied_by_centering=b,validation_mean_logit_shift=fl.mean()-b,W_mean_absolute_contribution=abs(terms[:,0]).mean(),R_mean_absolute_contribution=abs(terms[:,1]).mean(),W_minus_R_absolute_contribution=abs(terms[:,0]).mean()-abs(terms[:,1]).mean(),W_variance=terms[:,0].var(ddof=1),R_variance=terms[:,1].var(ddof=1),twice_covariance=2*np.cov(terms.T,ddof=1)[0,1],fusion_logit_variance=fl.var(ddof=1),reconstruction_max_probability_error=d,input_order='[actual stored WSSL zDD-zPD, relative dPD-dDD]',positive_label=1,class_order_evidence='source + exact probability recovery; classes_ not serialized',OOF_reference_rows='MISSING',full_fitted_reference_state='MISSING'))
            assert abs(parameters[-1]['fusion_logit_variance']-(parameters[-1]['W_variance']+parameters[-1]['R_variance']+parameters[-1]['twice_covariance']))<=1e-12
            comp+=transitions(f,c,i,seed,'single_seed')
            adapt+=transitions(f,c,i,seed,'single_seed',kind='adapted')
            y=f.target.to_numpy();sw=f.wssl_logit.to_numpy();sr=f.relative.to_numpy();wb=sw[y==1,None]-sw[y==0][None,:];rr=sr[y==1,None]-sr[y==0][None,:];wk=(wb>0);rk=(rr>0);bad=wb<0
            pairs.append(dict(context=c,inner=i,seed=seed,class_pairs=wb.size,WSSL_tied_pairs=int(sum((wb==0).ravel())),relative_tied_pairs=int(sum((rr==0).ravel())),both_strict_correct=int(sum((wk&rk).ravel())),WSSL_wrong_relative_correct=int(sum((bad&rk).ravel())),WSSL_correct_relative_wrong=int(sum((wk&(rr<0)).ravel())),both_strict_wrong=int(sum((bad&(rr<0)).ravel())),conditional_relative_AUC_on_WSSL_strict_wrong=float(kernel(sr[y==1],sr[y==0])[bad].mean()) if bad.any() else np.nan,score_AUC_difference=float((kernel(sr[y==1],sr[y==0])-kernel(sw[y==1],sw[y==0])).mean()),pair_unit_warning='pairs not independent patients; tie contributions separately accounted'))
            for layer in ['local','ssl_residual','bilateral','subject','structured','adapted_raw_DD_logit']:
                drift.append(dict(context=c,inner=i,seed=seed,representation=layer,status='BLOCKED_ADAPTED_REPRESENTATION_MISSING' if layer!='adapted_raw_DD_logit' else 'BLOCKED_RAW_LOGITS_NOT_SAVED',cosine_similarity=np.nan,L2_displacement=np.nan,frozen_norm=np.nan,adapted_norm=np.nan,feature_reextracted=False,source_normalization_equal=True,mean_probability_implied_logodds_change=f.adapted_logodds_change.mean(),mean_absolute_probability_implied_logodds_change=abs(f.adapted_logodds_change).mean(),prediction_flip_rate=float((f.baseline_prediction!=f.adapted_prediction).mean()),mean_absolute_probability_margin_change=f.adapted_absolute_margin_change.mean()))
        stack=pd.concat(unit);e=stack.groupby('subject_id').agg(target=('target','first'),baseline_pdd=('baseline_pdd','mean'),relative=('relative','first'),relative_prediction=('relative_prediction','first'),adapted_pdd=('adapted_pdd','mean'),u=('u','first')).reset_index()
        e['error_group']=e.subject_id.map(lambda s:groupmap[s]['error_group']);e['baseline_prediction']=(e.baseline_pdd>.5).astype(int);e['adapted_prediction']=(e.adapted_pdd>.5).astype(int);e['context']=c;e['inner']=i
        ensframes.append(e);comp+=transitions(e,c,i,'seed_probability_mean','seed_probability_mean');adapt+=transitions(e,c,i,'seed_probability_mean','seed_probability_mean',kind='adapted')
        print(json.dumps(dict(stage='saved_score_analysis',context=c,inner=i,status='COMPLETE',probability_recovery_max=maxdiff)),flush=True)
    allf=pd.concat(allframes,ignore_index=True);ens=pd.concat(ensframes,ignore_index=True)
    assert len(ens)==1560 and all(ens.groupby('subject_id').size()==4)
    for sid,t in ens.groupby('subject_id'):
        error=float((t.baseline_prediction!=t.target).mean());assert abs(error-float(groupmap[sid]['error_fraction']))<1e-12
        expected='stable-error' if error>=.75 else 'stable-correct' if error<=.25 else 'unstable';assert expected==groupmap[sid]['error_group']
    allf.to_csv(HERE/'restricted_manifests/paired_appearances_private.csv',index=False);ens.to_csv(HERE/'restricted_manifests/seed_probability_mean_appearances_private.csv',index=False)
    outputs={'score_distribution.csv':score_rows,'score_correlations.csv':corr_rows,'subject_complementarity.csv':comp,'reference_coverage.csv':coverage_rows,'coverage_correlations.csv':coverage_corr,'ranking_complementarity.csv':pairs,'fusion_parameter_audit.csv':parameters,'oof_full_distribution_shift.csv':shifts,'fusion_class_bias.csv':bias,'adaptation_error_transitions.csv':adapt,'representation_drift.csv':drift,'same_subject_OOF_full_train_shift.csv':same_subject,'formal_metric_identity.csv':formal}
    for n,rows in outputs.items():pd.DataFrame(rows).to_csv(HERE/'analysis'/n,index=False)
    tests=[]
    cmp=pd.DataFrame(comp);cmp=cmp[(cmp.aggregation=='single_seed')&(cmp.error_group=='all')]
    for label in ['PD','DD']:
        t=cmp[cmp.label==label].groupby(['context','inner']).net_rescue_rate.mean();tests.append(test_units(t,'relative_minus_WSSL_'+label+'_correct_rate','A_primary'))
    form=pd.DataFrame(formal);base=form[form.condition=='baseline'].set_index(['seed','context','inner']);ref=form[form.condition=='relative'].set_index(['seed','context','inner']);d=ref.ba-base.ba
    tests.append(test_units(d.groupby(['context','inner']).mean(),'relative_minus_WSSL_BA','A_primary'))
    pr=pd.DataFrame(pairs).groupby(['context','inner']).conditional_relative_AUC_on_WSSL_strict_wrong.mean()-.5;tests.append(test_units(pr,'conditional_relative_AUC_on_WSSL_wrong_minus_half','A_primary'))
    cor=pd.DataFrame(corr_rows)
    for key,t in cor.groupby(['label','left','right','method']):tests.append(test_units(t.groupby(['context','inner']).correlation.mean(),'|'.join(key),'A_correlations'))
    cv=pd.DataFrame(coverage_rows)
    for label,t in cv.groupby('label'):tests.append(test_units(t.groupby(['context','inner']).high_minus_low_error_rate.mean(),'coverage_high_minus_low_error_'+label,'A_coverage'))
    par=pd.DataFrame(parameters)
    for name in ['W_standardized_validation_mean','R_standardized_validation_mean','validation_mean_logit_shift','W_minus_R_absolute_contribution']:tests.append(test_units(par.groupby(['context','inner'])[name].mean(),name,'B_mechanism'))
    save_tests(tests,'predeclared_tests.csv')
    # Prediction drift association is descriptive; no feature drift is imputed.
    drift_score=[]
    for key,t in allf.groupby(['context','inner','seed']):
      for label in [-1,0,1]:
        q=t if label==-1 else t[t.target==label];a=q.baseline_prediction==q.target;b=q.adapted_prediction==q.target
        for event,keep in [('all',np.ones(len(q),bool)),('both_correct',a&b),('newly_broken',a&~b),('rescued',~a&b),('both_wrong',~a&~b)]:
            z=q[keep];drift_score.append(dict(context=key[0],inner=key[1],seed=key[2],label='all' if label==-1 else ('PD' if label==0 else 'DD'),event=event,n=len(z),mean_logodds_change=z.adapted_logodds_change.mean(),mean_absolute_logodds_change=abs(z.adapted_logodds_change).mean(),mean_true_signed_margin_change=z.adapted_true_signed_margin_change.mean(),mean_absolute_probability_margin_change=z.adapted_absolute_margin_change.mean(),baseline_pDD=z.baseline_pdd.mean(),adapted_pDD=z.adapted_pdd.mean(),interpretation='archived probability-implied scores only; no embedding mechanism identified'))
    pd.DataFrame(drift_score).to_csv(HERE/'analysis/adaptation_prediction_drift.csv',index=False)
    write_json(HERE/'analysis/phase5a_integrity.json',dict(status='SAVED_SCORE_ANALYSIS_PASS_WITH_STATE_GAPS',validation_units=45,subjects=390,seed_probability_mean_appearances=1560,appearances_per_subject=4,stable_error=68,stable_correct=279,unstable=43,full_saved_fusion_probability_max_error=maxdiff,OOF_aggregate_scaler_consistency_max_error=mean_scale_error,ID_label_alignment=True,formal_metric_identity_tolerance=1e-12,baseline_reference_adapted_fusion_archived_metrics_identical=True,new_fits=0,new_neural_inference=0,historical_outer_exposure_inherited=True,whole_global_cache_decoded=False,raw_IMU_read=False,missing_OOF_reference_scores=True,missing_reference_fit_state=True,missing_adapted_features=True,strict_global_no_outer_exposure_claim=False,elapsed_seconds=time.perf_counter()-started))
    guard();print('PHASE5_SCORE_ANALYSIS_COMPLETE',flush=True)

if __name__=='__main__':main()
