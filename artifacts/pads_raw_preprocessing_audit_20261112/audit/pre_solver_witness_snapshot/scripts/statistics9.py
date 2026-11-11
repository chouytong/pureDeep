"""Patient-level fixed-family association diagnostics, with no classifier fitting."""
from common9 import *
CLASS_METRICS=['duration_seconds','acc_dynamic_rms','gyro_rms','acc_jerk_rms','gyro_jerk_rms','acc_fraction_3_12','gyro_fraction_3_12','gap_fraction','constant_fraction','clip3g_fraction','start1_acc_rms_ratio','end1_acc_rms_ratio']

def compare(a,b,paired=False):
 a=np.asarray(a,float);b=np.asarray(b,float)
 if paired:valid=np.isfinite(a)&np.isfinite(b);a=a[valid];b=b[valid]
 else:a=a[np.isfinite(a)];b=b[np.isfinite(b)]
 if len(a)<3 or len(b)<3:return dict(status='NA_TOO_FEW',n_a=len(a),n_b=len(b),p=np.nan)
 rng=np.random.default_rng(SEED)
 if paired:
  d=a-b;bootstrap=d[rng.integers(0,len(d),(2000,len(d)))].mean(1);p=stats.wilcoxon(d,zero_method='wilcox').pvalue if np.any(abs(d)>1e-12) else 1.;effect=float(np.mean(np.sign(d)))
 else:
  u,p=stats.mannwhitneyu(a,b,method='asymptotic');effect=2*u/(len(a)*len(b))-1;bootstrap=a[rng.integers(0,len(a),(2000,len(a)))].mean(1)-b[rng.integers(0,len(b),(2000,len(b)))].mean(1)
 ci=np.quantile(bootstrap,[.025,.975]);return dict(status='REPEATED_COHORT_ASSOCIATION_ONLY',n_a=len(a),n_b=len(b),mean_a=a.mean(),mean_b=b.mean(),mean_difference=a.mean()-b.mean(),median_a=np.median(a),median_b=np.median(b),CI_low=ci[0],CI_high=ci[1],p=p,effect_size=effect,effect_definition='mean(sign(a-b)); paired sign-balance' if paired else 'rank biserial a minus b',patient_bootstrap_draws=2000)

def main():
 assert load(HERE/'analysis/raw_diagnostics_integrity.json')['status']=='PASS_COMPLETE_RAW_DIAGNOSTICS';freeze('STATISTICS',[HERE/'PROTOCOL.md',Path(__file__),HERE/'scripts/common9.py'],families=[132,440,880],RMS_for_boundary_tests='centered vector dynamic_RMS; raw RMS also descriptively reported',new_fits=0)
 frame=pd.read_csv(HERE/'restricted_manifests/analysis/record_metrics.csv',dtype={'subject_id':str});temporal=pd.read_csv(HERE/'restricted_manifests/analysis/temporal_boundary_metrics.csv',dtype={'subject_id':str});meta=pd.DataFrame(records()).drop_duplicates('subject_id')[['subject_id','device']];frame=frame.merge(meta,on='subject_id',validate='many_to_one');per=frame.groupby(['subject_id','activity','label','source_condition','device'])[CLASS_METRICS].mean().reset_index();tests=[]
 for activity in ACTS:
  g=per[per.activity==activity]
  for metric in CLASS_METRICS:tests.append(dict(activity=activity,measurement=metric,a='DD',b='PD',**compare(g[g.label=='DD'][metric],g[g.label=='PD'][metric])))
 tests=pd.DataFrame(tests);assert len(tests)==132;tests['BH132_q']=bh(tests.p);csv(tests,'class_conditional_data_quality.csv')
 desc=[]
 for keys,g in per.groupby(['activity','label','source_condition','device']):
  for m in CLASS_METRICS:
   v=g[m].dropna();desc.append(dict(zip(['activity','label','source_condition','device'],keys),measurement=m,subject_count=len(g),valid_subject_count=len(v),mean=v.mean(),median=v.median(),p25=v.quantile(.25),p75=v.quantile(.75),statistical_unit='unique subject; wrists averaged',inference='DESCRIPTIVE_ONLY_NO_NEW_DEVICE_TESTS'))
 csv(pd.DataFrame(desc),'class_quality_by_device_and_source.csv')
 t=temporal.groupby(['subject_id','activity','label','sensor','window'])[['dynamic_rms','jerk_rms']].mean().reset_index();cross=[];within=[]
 for activity in ACTS:
  for sensor in ['acc','gyro']:
   subset=t[(t.activity==activity)&(t.sensor==sensor)]
   for side in ['start','end']:
    for sec in [.5,1.,2.,3.,5.]:
     window=f'{side}{sec:g}'
     for m in ['dynamic_rms','jerk_rms']:
      g=subset[subset.window==window].set_index('subject_id');mid=subset[subset.window=='mid'].set_index('subject_id');g=g.join(mid[[m]],rsuffix='_mid',validate='one_to_one');g['difference']=g[m]-g[m+'_mid']
      cross.append(dict(activity=activity,sensor=sensor,window=window,measurement=m,comparison='DD boundary_minus_mid versus PD boundary_minus_mid',**compare(g[g.label=='DD'].difference,g[g.label=='PD'].difference)))
      for label in ['PD','DD']:
       k=g[g.label==label];within.append(dict(activity=activity,sensor=sensor,window=window,measurement=m,label=label,comparison='same patient boundary minus mid',**compare(k[m],k[m+'_mid'],True)))
 cross=pd.DataFrame(cross);within=pd.DataFrame(within);assert len(cross)==440 and len(within)==880;cross['BH440_q']=bh(cross.p);within['BH880_q']=bh(within.p);csv(cross,'temporal_boundary_class_associations.csv');csv(within,'temporal_boundary_within_class.csv')
 write(HERE/'analysis/statistics_integrity.json',dict(status='PASS_FIXED_PATIENT_FAMILIES',class_tests=132,temporal_class_tests=440,temporal_paired_tests=880,patient_count=390,PD_count=276,DD_count=114,wrists_not_independent=True,NA_tests_adjustment_p=1,own_NA_p_q=None,bootstrap_draws=2000,class_quality_BH05_count=int(sum(tests.BH132_q<.05)),temporal_class_BH05_count=int(sum(cross.BH440_q<.05)),temporal_paired_BH05_count=int(sum(within.BH880_q<.05)),association_not_causal=True,validation_scores_read=False,classifiers_fitted=0))
 print('PASS_FIXED_PATIENT_STATISTICS')

if __name__=='__main__':main()
