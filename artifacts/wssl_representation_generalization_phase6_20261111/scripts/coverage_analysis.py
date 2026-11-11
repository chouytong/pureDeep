"""Direct class interaction from Phase5 fixed scores/cutoffs; no bank fit or rescoring."""
from common import *
from threadpoolctl import threadpool_limits
import time

def saved_scores(c,i,seed,allowed):
 rows=[]
 with (P5/'restricted_manifests/paired_appearances_private.csv').open() as f:
  for row in csv.DictReader(f):
   if (int(row['context']),int(row['inner']),int(row['seed']))!=(c,i,seed):continue
   assert row['subject_id'] in allowed
   rows.append({k:row[k] for k in ['subject_id','target','baseline_prediction','baseline_pdd','u','d_pd','d_dd','relative','coverage_group']})
 f=pd.DataFrame(rows);check_ids(f.subject_id,allowed,True)
 for k in ['target','baseline_prediction']:f[k]=f[k].astype(int)
 for k in ['baseline_pdd','u','d_pd','d_dd','relative']:f[k]=f[k].astype(float)
 return f

def main():
 guard();start=time.perf_counter();cuts=pd.read_csv(P5/'analysis/reference_coverage.csv');banks=pd.read_csv(P5/'analysis/reference_asset_lineage_summary.csv');banks=banks[banks.reference_fold==-1].set_index(['context','inner'])
 rows=[];inter=[];subtypes=[];frames=[]
 for c,i,sp,excluded in splits():
  allowed=set(sp['validation_subjects']);categories={r.subject_id:r.source_condition for r in records(allowed)}
  for seed in [42,43,44]:
   f=saved_scores(c,i,seed,allowed);cu=cuts[(cuts.context==c)&(cuts.inner==i)&(cuts.seed==seed)&(cuts.label=='all')].iloc[0];lo,hi=cu.q25,cu.q75
   assert np.max(abs(f.u-np.minimum(f.d_pd,f.d_dd)))<=1e-12 and np.max(abs(f.relative-(f.d_pd-f.d_dd)))<=1e-12
   expected=np.where(f.u<=lo,'low',np.where(f.u>=hi,'high','middle'));assert np.array_equal(expected,f.coverage_group),'Do not alter Phase5 groups'
   f['context']=c;f['inner']=i;f['seed']=seed;f['source_category']=f.subject_id.map(categories);f['error']=(f.baseline_prediction!=f.target).astype(float);f['absolute_margin']=abs(f.baseline_pdd-.5);f['true_signed_margin']=(2*f.target-1)*(f.baseline_pdd-.5)
   frames.append(f);effects={}
   for label in [0,1]:
    t=f[f.target==label];lab='PD' if label==0 else 'DD';a=t[t.coverage_group=='high'];b=t[t.coverage_group=='low'];assert len(a) and len(b)
    effects[label]={m:a[m].mean()-b[m].mean() for m in ['error','absolute_margin','true_signed_margin','relative']}
    for group in ['low','high']:
     q=t[t.coverage_group==group];rows.append(dict(context=c,inner=i,seed=seed,label=lab,coverage_group=group,subject_count=len(q),error_rate=q.error.mean(),correct_rate=1-q.error.mean(),absolute_margin=q.absolute_margin.mean(),true_signed_margin=q.true_signed_margin.mean(),relative_score=q.relative.mean(),q25=lo,q75=hi,reference_bank_per_class=banks.loc[(c,i),'bank_per_class']))
   inter.append(dict(context=c,inner=i,seed=seed,error_interaction_PD_minus_DD=effects[0]['error']-effects[1]['error'],true_margin_interaction_DD_minus_PD=effects[1]['true_signed_margin']-effects[0]['true_signed_margin'],absolute_margin_interaction_DD_minus_PD=effects[1]['absolute_margin']-effects[0]['absolute_margin'],reference_relative_interaction_DD_minus_PD=effects[1]['relative']-effects[0]['relative'],reference_bank_per_class=banks.loc[(c,i),'bank_per_class']))
   for cat,t in f[f.target==1].groupby('source_category'):
    for group in ['low','high']:
     q=t[t.coverage_group==group];subtypes.append(dict(context=c,inner=i,seed=seed,source_category=cat,coverage_group=group,subject_count=len(q),error_count=int(q.error.sum()),error_rate=q.error.mean() if len(q) else np.nan,descriptive_only=True))
 pd.DataFrame(rows).to_csv(HERE/'analysis/coverage_group_metrics.csv',index=False);ir=pd.DataFrame(inter);ir.to_csv(HERE/'analysis/coverage_class_interaction.csv',index=False);pd.DataFrame(subtypes).to_csv(HERE/'analysis/coverage_DD_subtype_description.csv',index=False)
 ir.groupby(['context','inner']).mean(numeric_only=True).groupby('context').mean().reset_index().to_csv(HERE/'analysis/coverage_context_sensitivity.csv',index=False)
 tests=[]
 for m in ['error_interaction_PD_minus_DD','true_margin_interaction_DD_minus_PD','absolute_margin_interaction_DD_minus_PD']:tests.append(unit_test(ir.groupby(['context','inner'])[m].mean(),m,'B_interaction'))
 tf=pd.DataFrame(tests);tf['bh_q']=bh(tf.wilcoxon_p);tf.to_csv(HERE/'analysis/coverage_interaction_tests.csv',index=False)
 allf=pd.concat(frames,ignore_index=True);allf.to_csv(HERE/'restricted_manifests/coverage_appearances_private.csv',index=False)
 summaries=[];subject=allf.groupby('subject_id').agg(target=('target','first'),source_category=('source_category','first'),exposures=('seed','size'),high_count=('coverage_group',lambda x:sum(x=='high')),low_count=('coverage_group',lambda x:sum(x=='low')),middle_count=('coverage_group',lambda x:sum(x=='middle')),mean_error=('error','mean'),mean_absolute_margin=('absolute_margin','mean'),mean_true_margin=('true_signed_margin','mean')).sort_index()
 assert len(subject)==390 and (subject.exposures==12).all();assert (allf.groupby(['subject_id','context','inner']).size()==3).all()
 subject.to_csv(HERE/'restricted_manifests/coverage_subject_summary_private.csv')
 for label,t in subject.groupby('target'):summaries.append(dict(label='PD' if label==0 else 'DD',subject_count=len(t),minimum_validation_appearances=int(t.exposures.min()/3),maximum_validation_appearances=int(t.exposures.max()/3),seed_count=3,patients_ever_high=int(sum(t.high_count>0)),patients_ever_low=int(sum(t.low_count>0)),mean_high_appearance_count=float(t.high_count.mean()/3),mean_low_appearance_count=float(t.low_count.mean()/3),mean_error=t.mean_error.mean(),mean_absolute_margin=t.mean_absolute_margin.mean(),mean_true_margin=t.mean_true_margin.mean(),scope='aggregate only; full per-subject table private'))
 pd.DataFrame(summaries).to_csv(HERE/'analysis/coverage_subject_summary.csv',index=False)
 ids0=subject[subject.target==0].index.tolist();ids1=subject[subject.target==1].index.tolist();ids=ids0+ids1;mapping={s:j for j,s in enumerate(ids)};rng=np.random.default_rng(SEED)
 w=np.column_stack([rng.multinomial(len(ids0),np.full(len(ids0),1/len(ids0)),10000),rng.multinomial(len(ids1),np.full(len(ids1),1/len(ids1)),10000)]);effects={};point={}
 with threadpool_limits(limits=4):
  for label in [0,1]:
   for group in ['high','low']:
    q=allf[(allf.target==label)&(allf.coverage_group==group)];den=np.zeros(390);nums={m:np.zeros(390) for m in ['error','absolute_margin','true_signed_margin']}
    for sid,t in q.groupby('subject_id'):
     j=mapping[sid];den[j]=len(t)
     for m in nums:nums[m][j]=t[m].sum()
    dd=w@den
    for m,num in nums.items():effects[label,group,m]=np.divide(w@num,dd,out=np.full(10000,np.nan),where=dd>0);point[label,group,m]=num.sum()/den.sum()
 clusters=[]
 for m,direction in [('error','PD_minus_DD'),('true_signed_margin','DD_minus_PD'),('absolute_margin','DD_minus_PD')]:
  pd_delta=effects[0,'high',m]-effects[0,'low',m];dd_delta=effects[1,'high',m]-effects[1,'low',m];value=pd_delta-dd_delta if direction=='PD_minus_DD' else dd_delta-pd_delta
  pdd=point[0,'high',m]-point[0,'low',m];ddd=point[1,'high',m]-point[1,'low',m];estimate=pdd-ddd if direction=='PD_minus_DD' else ddd-pdd;valid=np.isfinite(value);assert sum(valid)>=9500;ci=np.quantile(value[valid],[.025,.975])
  clusters.append(dict(metric=m,direction=direction,interaction=estimate,PD_high_minus_low=pdd,DD_high_minus_low=ddd,ci_low=ci[0],ci_high=ci[1],valid_draws=int(sum(valid)),estimand='class-specific pooled exposure high-low difference-in-differences, not equal15split interaction',conditional_on_fixed_fits_and_Phase5_cutoffs=True))
 pd.DataFrame(clusters).to_csv(HERE/'analysis/coverage_interaction_subject_cluster.csv',index=False)
 availability=pd.read_csv(HERE/'analysis/activity_wrist_availability.csv');assert (availability.available_records==availability.query_count).all() and (availability.activity_present==availability.query_count).all()
 confound=pd.DataFrame([dict(factor='activity and wrist availability',finding='all11 activities and both wrists exist and are finite for every eligible query; no missingness variation',status='OBSERVED_CONSTANT'),dict(factor='task execution quality',finding='not captured by finite/mask checks',status='UNRESOLVED'),dict(factor='reference bank sizes',finding=f'balanced within each split; min {banks.bank_per_class.min()} max {banks.bank_per_class.max()}; no within-split class bank-count contrast',status='COUNTS_CONTROLLED_NOT_GEOMETRIC_COVERAGE'),dict(factor='per-activity/wrist reference distances',finding='not stored in Phase5; no reference refit/rescoring authorized',status='BLOCKED'),dict(factor='repeat appearances',finding='4 validation appearances and3seed each; subject joint resampling used',status='CONTROLLED_ESTIMAND'),dict(factor='DD clinical heterogeneity',finding='existing source-category descriptions only; small groups and no mechanism identification',status='EXPLORATORY_ONLY')]);confound.to_csv(HERE/'analysis/coverage_confounder_audit.csv',index=False)
 write_json(HERE/'analysis/phase6b_integrity.json',dict(status='PASS_EXISTING_SCORE_INTERACTION_WITH_LIMITED_CONFOUNDER_IDENTIFICATION',old_scores_and_cutoffs_unchanged=True,new_bank_scaler_PCA_fits=0,direct_class_interaction_tested=True,family_size=3,subject_count=390,appearance_count=4,seeds=3,activity_wrist_missingness_constant=True,per_activity_reference_distance_decomposition='BLOCKED_NOT_SAVED',clinical_causal_explanation_proven=False,new_reject_or_anomaly_gate=False,elapsed_seconds=time.perf_counter()-start))
 guard();print('COVERAGE_DIRECT_INTERACTION_COMPLETE',flush=True)

if __name__=='__main__':main()
