"""Descriptive evidence context for anomaly flags and paired current-filter changes."""
from common9 import *

def main():
 assert load(HERE/'analysis/raw_diagnostics_integrity.json')['status']=='PASS_COMPLETE_RAW_DIAGNOSTICS'
 rs=records();record={(r['subject_id'],r['activity'],r['wrist']):r for r in rs};a=pd.read_csv(HERE/'restricted_manifests/analysis/signal_anomaly_candidates.csv',dtype={'subject_id':str});result=[]
 for key,g in a.groupby(['subject_id','activity','wrist']):
  t,x=read_raw(record[key]);dif=np.diff(x,axis=0);mad=np.median(abs(dif-np.median(dif,axis=0)),axis=0)*1.4826;flags=np.divide(abs(dif-np.median(dif,axis=0)),mad,out=np.zeros_like(dif),where=mad>1e-15)>20;concurrent=flags.sum(1);gaps=np.diff(t)>10*np.median(np.diff(t));spec=spectral(x[:,:3],1/np.median(np.diff(t)))
  for kind,gg in g.groupby('anomaly_type'):result.append(dict(subject_id=key[0],activity=key[1],wrist=key[2],label=record[key]['label'],anomaly_type=kind,flagged_channel_count=gg.channel.nunique(),candidate_count=gg.candidate_count.sum(),max_channels_with_concurrent_derivative_flag=int(concurrent.max()),timestamp_large_gap_present=bool(gaps.any()),acc_fraction_3_12=spec['fraction_3_12'],acc_magnitude_max=float(np.linalg.norm(x[:,:3],axis=1).max()),gyro_magnitude_max=float(np.linalg.norm(x[:,3:],axis=1).max()),confirmed_sensor_error=False,adjudication='unresolved candidate: no hardware range/stationary reference; concurrent motion can be real'))
 evidence=pd.DataFrame(result);csv(evidence,'anomaly_evidence_private.csv',True);summary=evidence.groupby(['activity','anomaly_type']).agg(affected_subject_count=('subject_id','nunique'),affected_wrist_record_count=('subject_id','size'),concurrent3plus_channel_record_count=('max_channels_with_concurrent_derivative_flag',lambda v:int((v>=3).sum())),large_gap_also_present_count=('timestamp_large_gap_present','sum')).reset_index();summary['interpretation']='descriptive flags; cannot diagnose noise from jerk or amplitude alone';csv(summary,'signal_anomaly_evidence.csv')
 sp=pd.read_csv(HERE/'restricted_manifests/analysis/spectral_quality_by_activity.csv',dtype={'subject_id':str});pairs=sp.pivot(index=['subject_id','activity','wrist','label','sensor'],columns='version',values=['band_0_3','band_3_12','band_15_50','dynamic_rms']).reset_index();pairs.columns=['_'.join(v).rstrip('_') if isinstance(v,tuple) else v for v in pairs.columns];rows=[]
 for m in ['band_0_3','band_3_12','band_15_50','dynamic_rms']:pairs[m+'_ratio']=np.divide(pairs[m+'_current_processed'],pairs[m+'_raw_trim48'],out=np.full(len(pairs),np.nan),where=pairs[m+'_raw_trim48']>1e-15)
 per=pairs.groupby(['subject_id','activity','label','sensor'])[[m+'_ratio' for m in ['band_0_3','band_3_12','band_15_50','dynamic_rms']]].mean().reset_index()
 for keys,g in per.groupby(['activity','label','sensor']):
  for m in ['band_0_3','band_3_12','band_15_50','dynamic_rms']:
   v=g[m+'_ratio'].dropna();rows.append(dict(zip(['activity','label','sensor'],keys),measurement=m+'_current_over_rawtrim48',subject_count=len(g),valid_subject_count=len(v),mean=v.mean(),median=v.median(),p05=v.quantile(.05),p95=v.quantile(.95),fraction_below_0_9=float(np.mean(v<.9)),fraction_above1_1=float(np.mean(v>1.1)),interpretation='within-record nonlinear signal diagnostic; not disease-information loss or fixed transfer'))
 csv(pd.DataFrame(rows),'current_filter_paired_effects.csv')
 # Device-stratified acquisition quality is descriptive; no new result-selected tests.
 fm=pd.read_csv(HERE/'restricted_manifests/analysis/record_metrics.csv',dtype={'subject_id':str});meta=pd.DataFrame(rs).drop_duplicates('subject_id')[['subject_id','device']];fm=fm.merge(meta,on='subject_id');ts=pd.read_csv(HERE/'restricted_manifests/analysis/timestamp_quality.csv',dtype={'subject_id':str});fm=fm.merge(ts[['subject_id','activity','wrist','median_fs','maximum_dt','dt_jitter_SD']],on=['subject_id','activity','wrist'],validate='one_to_one');cols=['median_fs','maximum_dt','dt_jitter_SD','acc_dynamic_rms','gyro_rms','acc_jerk_rms','gyro_jerk_rms','gap_fraction'];patient=fm.groupby(['subject_id','device','label'])[cols].mean().reset_index();ds=[]
 for keys,g in patient.groupby(['device','label']):
  for m in cols:ds.append(dict(device=keys[0],label=keys[1],measurement=m,subject_count=len(g),mean=g[m].mean(),median=g[m].median(),p25=g[m].quantile(.25),p75=g[m].quantile(.75),inference='DESCRIPTIVE_ONLY_NO_CAUSAL_IDENTIFICATION'))
 csv(pd.DataFrame(ds),'device_quality_descriptive.csv');write(HERE/'analysis/supplement_integrity.json',dict(status='PASS_DESCRIPTIVE_SUPPLEMENT',anomaly_context_rows=len(evidence),new_classification_or_training=0,no_anomaly_auto_deletion=True,device_diagnosis_confounding_not_resolved=True))
 print('PASS_DESCRIPTIVE_EVIDENCE')

if __name__=='__main__':main()
