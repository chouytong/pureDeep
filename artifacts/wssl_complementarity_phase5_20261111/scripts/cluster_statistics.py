"""Conditional subject-cluster bootstrap, no model or data-fit operations."""
from common import *
from analyze import kernel,weighted_metrics
from threadpoolctl import threadpool_limits
import time

def interval(draws,point,**extra):
    ok=np.isfinite(draws);assert ok.sum()>=9500,'Insufficient valid bootstrap draws; do not change estimand silently'
    lo,hi=np.quantile(draws[ok],[.025,.975])
    return dict(mean_delta=point,ci_low=lo,ci_high=hi,valid_draws=int(ok.sum()),invalid_draws=int(sum(~ok)),requested_draws=10000,scope='class-stratified subject resampling; all appearances/seeds jointly; conditional fixed fitted models only',**extra)

def main():
    guard();started=time.perf_counter()
    allf=pd.read_csv(HERE/'restricted_manifests/paired_appearances_private.csv',dtype={'subject_id':str})
    ens=pd.read_csv(HERE/'restricted_manifests/seed_probability_mean_appearances_private.csv',dtype={'subject_id':str})
    subjects=allf[['subject_id','target']].drop_duplicates().sort_values('subject_id');assert len(subjects)==390 and subjects.subject_id.is_unique
    ids0=subjects[subjects.target==0].subject_id.tolist();ids1=subjects[subjects.target==1].subject_id.tolist();assert len(ids0)==276 and len(ids1)==114
    map0={s:j for j,s in enumerate(ids0)};map1={s:j for j,s in enumerate(ids1)}
    rng=np.random.default_rng(SEED);w0=rng.multinomial(276,np.full(276,1/276),size=10000);w1=rng.multinomial(114,np.full(114,1/114),size=10000)
    totals={k:np.zeros((10000,6)) for k in ['relative','fusion','adapted']};points={k:np.zeros(6) for k in totals};rank_draws=np.zeros(10000);rank_point=0.;units=0
    with threadpool_limits(limits=4):
      for (c,i,seed),f in allf.groupby(['context','inner','seed']):
        f=f.sort_values('subject_id');y=f.target.to_numpy();s0=[map0[s] for s in f[y==0].subject_id];s1=[map1[s] for s in f[y==1].subject_id];a=w0[:,s0];b=w1[:,s1]
        onea=np.ones((1,len(s0)));oneb=np.ones((1,len(s1)));base=weighted_metrics(f,'baseline',a,b);base_point=weighted_metrics(f,'baseline',onea,oneb)[0]
        for kind in totals:
            d=weighted_metrics(f,kind,a,b)-base;p=weighted_metrics(f,kind,onea,oneb)[0]-base_point
            totals[kind]+=d/45;points[kind]+=p/45
        sw=f.wssl_logit.to_numpy();sr=f.relative.to_numpy();bad=sw[y==1,None]<sw[y==0][None,:];rk=kernel(sr[y==1],sr[y==0]);den=np.sum((b@bad.astype(float))*a,axis=1);num=np.sum((b@(rk*bad))*a,axis=1)
        value=np.divide(num,den,out=np.full(10000,np.nan),where=den>0)-.5;rank_draws+=value/45;rank_point+=(float(rk[bad].mean())-.5)/45;units+=1
      assert units==45
      formal=pd.read_csv(HERE/'analysis/formal_metric_identity.csv')
      base_mean=formal[formal.condition=='baseline'].groupby(['context','inner'])[METRICS].mean().mean()
      out=[]
      for kind in totals:
        expected=formal[formal.condition==kind].groupby(['context','inner'])[METRICS].mean().mean()-base_mean
        assert np.max(abs(expected.to_numpy()-points[kind]))<=1e-12,(kind,points[kind],expected.to_numpy())
        for j,m in enumerate(METRICS):out.append(interval(totals[kind][:,j],points[kind][j],comparison=kind+'-baseline',metric=m,estimand='seed-first 15split metric difference'))
      out.append(interval(rank_draws,rank_point,comparison='conditional relative ranking on WSSL strict wrong pairs',metric='conditional_AUC_minus_half',estimand='45unit mean, equivalent to seed-first 15split'))
      pd.DataFrame(out).to_csv(HERE/'analysis/subject_cluster_bootstrap.csv',index=False)
      # Same subject multiplicities for fixed validation-defined coverage groups.
      ids=ids0+ids1;allw=np.column_stack([w0,w1]);mapping={s:j for j,s in enumerate(ids)}
      cv=[]
      for label in [-1,0,1]:
        f=allf if label==-1 else allf[allf.target==label];vectors={}
        for group in ['low','high']:
            q=f[f.coverage_group==group];den=np.zeros(390);num=np.zeros(390)
            for sid,t in q.groupby('subject_id'):
                j=mapping[sid];den[j]=len(t);num[j]=sum(t.baseline_prediction!=t.target)
            d=allw@den;n=allw@num;vectors[group]=np.divide(n,d,out=np.full(10000,np.nan),where=d>0)
            vectors[group+'_point']=num.sum()/den.sum()
        cv.append(interval(vectors['high']-vectors['low'],vectors['high_point']-vectors['low_point'],label='all' if label==-1 else ('PD' if label==0 else 'DD'),metric='high_minus_low_error',estimand='pooled appearance/seed exposure rates; NOT 15split equal weight'))
      pd.DataFrame(cv).to_csv(HERE/'analysis/coverage_subject_cluster_bootstrap.csv',index=False)
      # Subject means of 4 appearances, using the predefined 3seed probability-mean decisions.
      ens['baseline_correct']=(ens.baseline_prediction==ens.target).astype(float);ens['relative_correct']=(ens.relative_prediction==ens.target).astype(float);ens['adapted_correct']=(ens.adapted_prediction==ens.target).astype(float)
      ens['relative_gross']=(1-ens.baseline_correct)*ens.relative_correct;ens['relative_broken']=ens.baseline_correct*(1-ens.relative_correct)
      ens['adapted_gross']=(1-ens.baseline_correct)*ens.adapted_correct;ens['adapted_broken']=ens.baseline_correct*(1-ens.adapted_correct)
      subject=ens.groupby('subject_id').agg(target=('target','first'),error_group=('error_group','first'),baseline_correct=('baseline_correct','mean'),relative_correct=('relative_correct','mean'),adapted_correct=('adapted_correct','mean'),relative_gross=('relative_gross','mean'),relative_broken=('relative_broken','mean'),adapted_gross=('adapted_gross','mean'),adapted_broken=('adapted_broken','mean')).loc[ids]
      subject.to_csv(HERE/'restricted_manifests/subject_4appearance_summary_private.csv')
      sg=[];patterns=[]
      for group in ['all','stable-error','stable-correct','unstable']:
       for label in [-1,0,1]:
        keep=np.ones(390,bool)
        if group!='all':keep &= subject.error_group.to_numpy()==group
        if label!=-1:keep &= subject.target.to_numpy()==label
        for kind in ['relative','adapted']:
            v=subject[kind+'_correct'].to_numpy()-subject.baseline_correct.to_numpy();n=keep.sum();den=allw@keep;num=allw@(v*keep);draw=np.divide(num,den,out=np.full(10000,np.nan),where=den>0)
            sg.append(interval(draw,float(v[keep].mean()),condition=kind,error_group=group,label='all' if label==-1 else ('PD' if label==0 else 'DD'),subjects=int(n),gross_rescue_appearance_fraction=float(subject.loc[keep,kind+'_gross'].mean()),newly_broken_appearance_fraction=float(subject.loc[keep,kind+'_broken'].mean()),estimand='patient mean correctness change over 4 seed-probability-mean appearances'))
        v=subject.adapted_correct.to_numpy()-subject.baseline_correct.to_numpy();p=subject.loc[keep]
        patterns.append(dict(error_group=group,label='all' if label==-1 else ('PD' if label==0 else 'DD'),subjects=int(n),improved_subjects=int(sum(v[keep]>0)),unchanged_subjects=int(sum(v[keep]==0)),worsened_subjects=int(sum(v[keep]<0)),subjects_any_rescue=int(sum(p.adapted_gross>0)),subjects_any_new_error=int(sum(p.adapted_broken>0)),subjects_both_rescue_and_harm=int(sum((p.adapted_gross>0)&(p.adapted_broken>0))),interpretation='4 appearances, seed probability mean; not a new deployed classifier'))
      pd.DataFrame(sg).to_csv(HERE/'analysis/error_group_subject_cluster_bootstrap.csv',index=False)
      pd.DataFrame(patterns).to_csv(HERE/'analysis/adaptation_subject_consistency.csv',index=False)
    # Seed and split uncertainty kept separate; relative is deterministic and not three trained seeds.
    stability=[];seedmeans=[]
    for kind,f in formal.groupby('condition'):
      for m in METRICS:
        t=f.groupby(['context','inner'])[m].mean();stability.append(dict(condition=kind,metric=m,mean=t.mean(),split_SD=t.std(ddof=1),within_split_seed_SD=f.groupby(['context','inner'])[m].std(ddof=1).mean() if kind!='relative' else np.nan,SD_of_three_seed_means=f.groupby('seed')[m].mean().std(ddof=1) if kind!='relative' else np.nan,independent_external_validation=False))
      if kind!='relative':
        v=f.groupby('seed')[METRICS].mean().reset_index();v.insert(0,'condition',kind);seedmeans.extend(v.to_dict('records'))
    pd.DataFrame(stability).to_csv(HERE/'analysis/seed_split_stability.csv',index=False);pd.DataFrame(seedmeans).to_csv(HERE/'analysis/per_seed_metrics.csv',index=False)
    contexts=formal.groupby(['condition','context','inner'])[METRICS].mean().groupby(['condition','context']).mean().reset_index();contexts.to_csv(HERE/'analysis/context_sensitivity.csv',index=False)
    # Historical metric p/q are copied as evidence, never silently replaced with another family.
    import shutil
    for n in ['B_relative_vs_WSSL_paired.csv','B_fusion_paired.csv','A1_adapted_paired.csv','B_conditional_auc.csv']:
        target=HERE/'analysis'/('reused_'+n);assert not target.exists();shutil.copy2(OLD/'analysis'/n,target)
    guard();write_json(HERE/'analysis/cluster_statistics_integrity.json',dict(status='PASS',bootstrap_seed=SEED,draws=10000,PD_subjects=276,DD_subjects=114,joint_all_appearances_and_seeds=True,point_estimates_match_formal_seed_first=True,coverage_estimand_explicitly_pooled=True,zero_denominator_draws_reported=True,independent_external_generalization_inferred=False,elapsed_seconds=time.perf_counter()-started))
    print('SUBJECT_CLUSTER_STATISTICS_COMPLETE',flush=True)

if __name__=='__main__':main()
