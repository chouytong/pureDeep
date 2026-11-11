"""Disclosed conditional uncertainty and descriptive summaries; no model selection."""
from common import *
from scipy.stats import spearmanr
from scipy.special import expit
from threadpoolctl import threadpool_limits
import importlib.util,os,subprocess

def summary(table,condition):
    rows=[]
    for m in METRICS:
        units=table.groupby(['context','inner'])[m].mean()
        stochastic='seed' in table and np.issubdtype(table.seed.dtype,np.number)
        rows.append(dict(condition=condition,metric=m,mean=float(units.mean()),split_sd=float(units.std(ddof=1)),seed_mean_sd=float(table.groupby('seed')[m].mean().std(ddof=1)) if stochastic else None,mean_within_split_seed_sd=float(table.groupby(['context','inner'])[m].std(ddof=1).mean()) if stochastic else None,seed_interpretation='three initialized trained classifiers' if stochastic else 'deterministic external-cache method; seed SD not applicable'))
    return rows

def kernel(dd,pd):return (dd[:,None]>pd[None,:]).astype(float)+.5*(dd[:,None]==pd[None,:])

def probability_kernel(f,y):
    a=f.loc[y==1];b=f.loc[y==0]
    return .5*(kernel(a.probability_dd.to_numpy(),b.probability_dd.to_numpy())+kernel(b.probability_pd.to_numpy(),a.probability_pd.to_numpy()).T)

def cluster_bootstrap(kind):
    # Same subject multiplicity in every appearance and every seed/fitted model.
    appearances=pd.read_csv(HERE/'predictions/A1_appearance_predictions.csv',dtype={'subject_id':str})
    subjects=appearances[['subject_id','target']].drop_duplicates().sort_values('subject_id')
    ids0=subjects[subjects.target==0].subject_id.tolist();ids1=subjects[subjects.target==1].subject_id.tolist();map0={s:j for j,s in enumerate(ids0)};map1={s:j for j,s in enumerate(ids1)}
    rng=np.random.default_rng(SEED);w0=rng.multinomial(len(ids0),np.full(len(ids0),1/len(ids0)),size=10000);w1=rng.multinomial(len(ids1),np.full(len(ids1),1/len(ids1)),size=10000)
    bsba=np.zeros(10000);bsauc=np.zeros(10000);pointba=pointauc=0.
    fusion=pd.read_csv(HERE/'predictions/B_fusion_predictions.csv',dtype={'subject_id':str}) if kind=='fusion' else None
    for c,i,sp in splits():
        ids=sorted(sp['validation_subjects']);y=labels(ids);s0=[map0[s] for s,t in zip(ids,y) if t==0];s1=[map1[s] for s,t in zip(ids,y) if t==1];a=w0[:,s0];b=w1[:,s1];den0=a.sum(1);den1=b.sum(1);assert (den0>0).all() and (den1>0).all()
        auc_delta=np.zeros((sum(y==1),sum(y==0)));correct_delta=np.zeros(len(ids))
        raw_b=pd.read_csv(HERE/'predictions'/f'B_context{c}_inner{i}.csv',dtype={'subject_id':str}).set_index('subject_id').loc[ids] if kind!='fusion' else None
        for seed in [42,43,44]:
            base=pred(baseline(seed,c,i)/'predictions/validation.csv').loc[ids]
            if kind=='fusion':
                f=fusion[(fusion.context==c)&(fusion.inner==i)&(fusion.seed==seed)].set_index('subject_id').loc[ids]
                kk=probability_kernel(f,y);pr=f.prediction.to_numpy()
            else:
                score=raw_b.relative_score.to_numpy();kk=kernel(score[y==1],score[y==0]);pr=raw_b.relative_prediction.to_numpy()
            auc_delta+=(kk-probability_kernel(base,y))/3
            correct_delta+=((pr==y).astype(float)-(base.prediction.to_numpy()==y))/3
        bsauc+=np.sum((b@auc_delta)*a,axis=1)/(den0*den1)/15
        bsba+=.5*((a@correct_delta[y==0])/den0+(b@correct_delta[y==1])/den1)/15
        pointauc+=auc_delta.mean()/15;pointba+=.5*(correct_delta[y==0].mean()+correct_delta[y==1].mean())/15
    rows=[]
    expected=pd.read_csv(HERE/'analysis'/('B_fusion_paired.csv' if kind=='fusion' else 'B_relative_vs_WSSL_paired.csv')).set_index('metric')
    for name,point,draws in [('ba',pointba,bsba),('auroc',pointauc,bsauc)]:
        assert abs(point-expected.loc[name,'mean_delta'])<1e-12,'Subject-cluster point estimate must match seed-first paired estimate'
        ci=np.quantile(draws,[.025,.975]);rows.append(dict(comparison=kind+'-baseline',metric=name,mean_delta=point,ci_low=float(ci[0]),ci_high=float(ci[1]),bootstrap_draws=10000,PD_subjects=len(ids0),DD_subjects=len(ids1),scope='conditional fixed fitted models; all appearances resampled jointly; not retraining/selection uncertainty'))
    return rows

def simple_calibration(y,s):
    p=expit(s);confidence=np.maximum(p,1-p);correct=(s>0)==y;bins=np.minimum((confidence*10).astype(int),9);ece=sum(np.mean(bins==b)*abs(np.mean(correct[bins==b])-np.mean(confidence[bins==b])) for b in range(10) if np.any(bins==b))
    return dict(nll=float(np.mean(np.logaddexp(0,s)-y*s)),brier=float(2*np.mean((p-y)**2)),ece=float(ece))

def main():
    guard();all_summaries=[];contexts=[];agreements=[];corrections=[]
    base=pd.read_csv(HERE/'analysis/baseline_metrics.csv')
    tables={'baseline':base,'lastblock_adapted':pd.read_csv(HERE/'analysis/A1_adapted_metrics.csv'),'fusion':pd.read_csv(HERE/'analysis/B_fusion_metrics.csv')}
    probe=pd.read_csv(HERE/'analysis/A1_probe_metrics.csv')
    for name,table in probe.groupby('condition'):tables[name]=table
    b=pd.read_csv(HERE/'analysis/B_metrics.csv')
    for name,table in b.groupby('condition'):tables[name]=table
    for name,table in tables.items():
        all_summaries+=summary(table,name)
        units=table.groupby(['context','inner'])[METRICS].mean().reset_index()
        for c,t in units.groupby('context'):contexts.append(dict(condition=name,context=int(c),**t[METRICS].mean().to_dict()))
    pd.DataFrame(all_summaries).to_csv(HERE/'analysis/ALL_stability_summary.csv',index=False)
    pd.DataFrame(contexts).to_csv(HERE/'analysis/five_context_sensitivity.csv',index=False)
    paired(tables['relative'],base).to_csv(HERE/'analysis/B_relative_vs_WSSL_paired.csv',index=False)
    fusion_predictions=pd.read_csv(HERE/'predictions/B_fusion_predictions.csv',dtype={'subject_id':str})
    for c,i,sp in splits():
        bb=pd.read_csv(HERE/'predictions'/f'B_context{c}_inner{i}.csv',dtype={'subject_id':str}).set_index('subject_id')
        for seed in [42,43,44]:
            f=pred(baseline(seed,c,i)/'predictions/validation.csv');cand=fusion_predictions[(fusion_predictions.context==c)&(fusion_predictions.inner==i)&(fusion_predictions.seed==seed)].set_index('subject_id').loc[f.index];ref=bb.loc[f.index]
            for kind,score,decision in [('relative',ref.relative_score.to_numpy(),ref.relative_prediction.to_numpy()),('fusion',cand.probability_dd.to_numpy(),cand.prediction.to_numpy())]:
                agreements.append(dict(condition=kind,context=c,inner=i,seed=seed,score_spearman=float(spearmanr(score,f.probability_dd).statistic),prediction_agreement=float((decision==f.prediction.to_numpy()).mean())))
                for label in [0,1]:
                    keep=f.target.to_numpy()==label;a=decision[keep]==label;old=f.prediction.to_numpy()[keep]==label
                    corrections.append(dict(condition=kind,context=c,inner=i,seed=seed,label=label,corrected=int(sum(a&~old)),newly_wrong=int(sum(~a&old)),net=int(sum(a)-sum(old)),n=int(sum(keep))))
    pd.DataFrame(agreements).to_csv(HERE/'analysis/B_agreement_with_WSSL.csv',index=False)
    pd.DataFrame(corrections).to_csv(HERE/'analysis/B_all_subject_corrections.csv',index=False)
    with threadpool_limits(limits=4):cluster=cluster_bootstrap('relative')+cluster_bootstrap('fusion')
    pd.DataFrame(cluster).to_csv(HERE/'analysis/subject_cluster_bootstrap.csv',index=False)
    # Grouped calibration uses no group-informed fit or patient routing.
    groups=[];dd_bins=[];temperature=pd.read_csv(HERE/'analysis/A2_temperatures.csv').set_index(['seed','context','inner'])
    manifest=pd.read_csv(ROOT/'data/processed/pads_multi_activity/v2_l1_full_length/manifests/CrossArms.csv',dtype={'subject_id':str}).set_index('subject_id')
    for c,i,sp in splits():
      for seed in [42,43,44]:
        with np.load(HERE/'features'/f'seed{seed}_context{c}_inner{i}.npz') as z:
            keep=z['is_validation'];ids=z['subject_ids'][keep].tolist();y=z['target'][keep].copy();score=np.diff(z['logits'][keep].astype(np.float64),axis=1).ravel()
        source=manifest.loc[ids,'source_condition'].to_numpy();T=temperature.loc[(seed,c,i),'temperature']
        for condition,s in [('raw',score),('positive_temperature',score/T)]:
            p=expit(s);bins=np.minimum((p*10).astype(int),9)
            for bindex in range(10):
                m=bins==bindex;dd_bins.append(dict(context=c,inner=i,seed=seed,condition=condition,bin=bindex,n=int(sum(m)),mean_probability=float(p[m].mean()) if m.any() else None,DD_fraction=float(y[m].mean()) if m.any() else None))
            for group,mask in [('PD',y==0),('DD',y==1)]+[(str(g),source==g) for g in sorted(set(source[y==1]))]:
                if mask.any():groups.append(dict(context=c,inner=i,seed=seed,condition=condition,group=group,n=int(sum(mask)),**simple_calibration(y[mask],s[mask])))
    pd.DataFrame(groups).to_csv(HERE/'analysis/A2_grouped_calibration.csv',index=False)
    pd.DataFrame(dd_bins).to_csv(HERE/'analysis/A2_DD_reliability_bins.csv',index=False)
    cp=pd.read_csv(HERE/'analysis/A2_calibration_paired.csv');cp['improved_count']=cp.losses;cp['worsened_count']=cp.wins;cp['direction_note']='lower NLL/Brier/ECE is better; original wins column counts positive delta, not improvement';cp.to_csv(HERE/'analysis/A2_calibration_directional.csv',index=False)
    # Preserve failed precision checks as failures, regardless of collection status.
    grad=json.loads((HERE/'analysis/A0_real_gradients.json').read_text())['cases'];failed=[x['seed'] for x in grad if not x['actual_precision_tolerance_pass']]
    write_json(HERE/'analysis/A0_precision_assessment.json',dict(status='PRECISION_LIMITED_ACTUAL_FP32' if failed else 'PASS',failed_actual_runtime_seeds=failed,parameter_gradient_relative_tolerance=2e-5,tolerance_relaxed=False,IEEE_FP32_shadow_passed=True,formal_TF32_settings_changed=False,interpretation='The mathematical common-scalar identity and strict floating-point parameter-gradient equivalence are distinct. TF32-enabled convolution backward has scaling roundoff; not evidence of a PyTorch loss bug or DD-error causation. The original strict gate remains failed.'))
    import importlib.metadata
    info=dict(versions={p:importlib.metadata.version(p) for p in ['torch','numpy','pandas','scipy','scikit-learn','PyYAML']},torch_version=__import__('torch').__version__,python=sys.version,CPU_count=os.cpu_count(),GPU=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout.strip(),threads=4,new_deep_training_runs=0,optimizer_updates=0,model_peak_memory='not instrumented; not inferred from a later process',matplotlib_available=importlib.util.find_spec('matplotlib') is not None)
    write_json(HERE/'analysis/runtime_environment.json',info)
    guard();write_json(HERE/'analysis/statistics_complete.json',dict(status='PASS',seed_first=True,paired_bootstrap_seed=SEED,paired_bootstrap_draws=10000,subject_cluster_joint_appearances=True,subject_cluster_conditional_on_fitted_models=True,five_context_sensitivity_not_outer_test=True))
    print(json.dumps(dict(stage='statistics',status='PASS',matplotlib_available=info['matplotlib_available'])),flush=True)

if __name__=='__main__':main()
