"""Schema v2: metadata includes Time; original v1 and freeze retained."""
"""Whole-task-cohort quality diagnostics; no model, fitted normalization or outer roles."""
from common9 import *
from concurrent.futures import ProcessPoolExecutor
from collections import Counter,defaultdict

def constant_runs(x):
 eq=np.diff(x)==0;edges=np.r_[0,np.flatnonzero(~eq)+1,len(x)];lengths=np.diff(edges);long=lengths>=50
 return int(lengths.max()),float(lengths[long].sum()/len(x)),list(zip(edges[:-1][long].tolist(),(edges[1:][long]-1).tolist()))

def sensor_metrics(t,x):
 if len(x)<4 or not np.isfinite(x).all():return {k:np.nan for k in ['magnitude_mean','rms','dynamic_rms','variance','mad','jerk_rms','zcr','short_time_energy','autocorrelation_lag1',*spectral(np.zeros(4),100)]}
 fs=1/np.median(np.diff(t));mag=np.linalg.norm(x,axis=1);center=x-x.mean(0);delta=np.diff(t);jerk=np.diff(x,axis=0)/delta[:,None];zcr=np.mean((center[1:]>0)!=(center[:-1]>0));v=mag-mag.mean();corr=np.corrcoef(v[:-1],v[1:])[0,1] if v.std()>1e-15 else np.nan;window=max(1,int(round(fs*.25)));power=np.sum(center**2,axis=1);energy=np.convolve(power,np.ones(window)/window,mode='valid') if len(power)>=window else power
 return dict(magnitude_mean=float(mag.mean()),rms=float(np.sqrt(np.mean(np.sum(x*x,axis=1)))),dynamic_rms=float(np.sqrt(np.mean(np.sum(center*center,axis=1)))),variance=float(np.sum(np.var(x,axis=0))),mad=float(np.median(np.abs(mag-np.median(mag)))),jerk_rms=float(np.sqrt(np.mean(np.sum(jerk*jerk,axis=1)))),zcr=float(zcr),short_time_energy=float(energy.mean()),autocorrelation_lag1=float(corr),**spectral(x,fs))

def temporal_windows(t):
 relative=t-t[0];duration=relative[-1];windows=[]
 for length in [.5,1.,2.,3.,5.]:
  for side in ['start','end']:
   if duration<length:keep=np.zeros(len(t),bool)
   else:keep=(relative<length) if side=='start' else (relative>duration-length)
   windows.append((f'{side}{length:g}',keep))
 mid=(relative>=5)&(relative<=duration-5)
 if mid.sum()<4 or (t[mid][-1]-t[mid][0] if mid.any() else 0)<1:mid=(relative>=duration*.4)&(relative<=duration*.6)
 if mid.sum()<4 or t[mid][-1]-t[mid][0]<.5:mid[:]=False
 return windows+[('mid',mid)]

def one(r):
 base={k:r[k] for k in ['subject_id','activity','wrist','label','source_condition']};quality=[];anomaly=[];temporal=[];spectra=[];record=dict(base);timestamp=dict(base);t,x=read_raw(r);n=len(t);d=np.diff(t);finite=np.isfinite(x).all();validtime=np.isfinite(t).all() and np.all(d>0);assert n>48 and finite and validtime
 fs=1/np.median(d);large=d>10*np.median(d);packet=np.maximum(np.round(d/np.median(d))-1,0);timestamp.update(sample_count=n,nominal_fs=r['nominal_fs'],mean_fs=(n-1)/(t[-1]-t[0]),median_fs=fs,median_dt=float(np.median(d)),mean_dt=float(d.mean()),maximum_dt=float(d.max()),dt_jitter_SD=float(d.std()),duplicate_timestamp_count=int(sum(d==0)),nonmonotonic_count=int(sum(d<0)),large_gap_count=int(sum(large)),estimated_missing_grid_points=int(packet.sum()),estimated_gap_fraction=float(packet.sum()/(n+packet.sum())),first_time=float(t[0]),last_time=float(t[-1]))
 proc=np.load(r['processed_path'],allow_pickle=False);assert proc.shape==(6,n-48) and np.isfinite(proc).all();gyro_exact=np.array_equal(proc[3:],x[48:,3:].T.astype(np.float32));assert gyro_exact
 record.update(raw_samples=n,processed_samples=n-48,duration_seconds=float(t[-1]-t[0]),trim48_elapsed_seconds=float(t[48]-t[0]),gap_fraction=float(packet.sum()/(n+packet.sum())),clip3g_fraction=float(np.mean(np.abs(x[48:,:3])>3)),gyro_processed_exact=True,metadata_rows_equal=n==r['metadata_rows'],units_match=r['metadata_units']==['s']+['g']*3+['rad/s']*3,channel_count=6)
 for j,name in enumerate(CHANNELS):
  y=x[:,j];med=np.median(y);mad=np.median(abs(y-med));mx,constant,segments=constant_runs(y);extreme=(y==y.max())|(y==y.min());sat=extreme & (np.r_[False,np.diff(y)==0]|np.r_[np.diff(y)==0,False]);level=np.flatnonzero(abs(y-med)>20*1.4826*mad) if mad>1e-15 else np.array([],int);dy=np.diff(y);dm=np.median(dy);ds=np.median(abs(dy-dm));jump=np.flatnonzero(abs(dy-dm)>20*1.4826*ds) if ds>1e-15 else np.array([],int)
  quality.append(dict(base,channel=name,sensor='Acc' if j<3 else 'Gyro',sample_count=n,mean=y.mean(),median=med,std=y.std(),variance=y.var(),minimum=y.min(),maximum=y.max(),range=np.ptp(y),iqr=np.quantile(y,.75)-np.quantile(y,.25),rms=np.sqrt(np.mean(y*y)),mad=mad,zero_fraction=np.mean(y==0),nan_fraction=np.mean(np.isnan(y)),inf_fraction=np.mean(np.isinf(y)),constant_segment_ratio=constant,maximum_constant_samples=mx,unique_value_count=len(np.unique(y)),extreme_repeat_saturation_candidate_ratio=float(sat.mean()),hardware_saturation_confirmed=False))
  for typ,ix in [('robust_level_candidate',level),('robust_derivative_candidate',jump)]:
   if len(ix):anomaly.append(dict(base,channel=name,anomaly_type=typ,candidate_count=len(ix),sample_positions_json=json.dumps(ix.tolist()),time_positions_json=json.dumps(t[ix].tolist()),confirmed_sensor_error=False,automatic_action='NONE',interpretation='extreme movement or sensor artifact unresolved'))
  for a,b in segments:anomaly.append(dict(base,channel=name,anomaly_type='constant_run_candidate',candidate_count=b-a+1,sample_positions_json=json.dumps([a,b]),time_positions_json=json.dumps([t[a],t[b]]),confirmed_sensor_error=False,automatic_action='NONE',interpretation='quantization/true low motion/device issue unresolved'))
 for k in np.flatnonzero(large):anomaly.append(dict(base,channel='timestamp',anomaly_type='gap_over10median',candidate_count=1,sample_positions_json=json.dumps([int(k),int(k+1)]),time_positions_json=json.dumps([float(t[k]),float(t[k+1])]),confirmed_sensor_error=False,automatic_action='NONE',interpretation='observed timestamp gap; actual missing samples versus clock jump unresolved'))
 repeated=np.flatnonzero(np.all(np.diff(x,axis=0)==0,axis=1));record['full6channel_duplicate_adjacent_count']=len(repeated)
 constants=[]
 for sensor,sl in [('acc',slice(0,3)),('gyro',slice(3,6))]:
  m=sensor_metrics(t,x[:,sl]);record.update({sensor+'_'+k:v for k,v in m.items()});constants.extend(q['constant_segment_ratio'] for q in quality if q['sensor'].lower()==sensor)
  for window,keep in temporal_windows(t):
   value=sensor_metrics(t[keep],x[keep,sl]) if keep.sum()>=4 else sensor_metrics(np.array([]),np.empty((0,3)))
   temporal.append(dict(base,sensor=sensor,window=window,window_sample_count=int(keep.sum()),status='OK' if keep.sum()>=4 else 'NA_INSUFFICIENT_DURATION',**value))
  for kind,y in [('raw_trim48',x[48:,sl]),('current_processed',proc[sl].T)]:spectra.append(dict(base,sensor=sensor,version=kind,**spectral(y,fs),rms=np.sqrt(np.mean(np.sum(y*y,axis=1))),dynamic_rms=np.sqrt(np.mean(np.sum((y-y.mean(0))**2,axis=1)))))
 record['constant_fraction']=float(np.mean(constants));mid={z['sensor']:z for z in temporal if z['window']=='mid'}
 for side in ['start1','end1']:
  m=next(z for z in temporal if z['sensor']=='acc' and z['window']==side);den=mid['acc']['dynamic_rms'];record[side+'_acc_rms_ratio']=m['dynamic_rms']/den if den>1e-15 else np.nan
 return record,timestamp,quality,anomaly,temporal,spectra,t

def aggregate(frame,groups,columns):
 # First reduce repeated wrists/channels inside each subject; release quantiles only.
 per=frame.groupby(['subject_id']+groups,dropna=False)[columns].mean().reset_index();rows=[]
 for keys,g in per.groupby(groups,dropna=False):
  if not isinstance(keys,tuple):keys=(keys,)
  key=dict(zip(groups,keys))
  for name in columns:
   values=g[name].dropna().to_numpy();rows.append(dict(key,measurement=name,subject_count=g.subject_id.nunique(),valid_subject_count=len(values),mean=values.mean() if len(values) else np.nan,median=np.median(values) if len(values) else np.nan,p05=np.quantile(values,.05) if len(values) else np.nan,p25=np.quantile(values,.25) if len(values) else np.nan,p75=np.quantile(values,.75) if len(values) else np.nan,p95=np.quantile(values,.95) if len(values) else np.nan,statistical_unit='subject; wrists/channels averaged within subject'))
 return pd.DataFrame(rows)

def main():
 assert load(HERE/'analysis/startup_integrity.json')['status']=='PASS_CODE_FIRST_BEFORE_SIGNAL_DIAGNOSTICS';protected=load(HERE/'restricted_manifests/protected_assets.json')['sha256'];rows=[];timestamps=[];quality=[];anomalies=[];temp=[];spectra=[];times={};beg=time.perf_counter();rs=records()
 with ProcessPoolExecutor(max_workers=8) as pool:
  for k,(r,result) in enumerate(zip(rs,pool.map(one,rs,chunksize=16))):
   row,stamp,q,a,tmp,s,t=result;rows.append(row);timestamps.append(stamp);quality+=q;anomalies+=a;temp+=tmp;spectra+=s;times[(row['subject_id'],row['activity'],row['wrist'])]=t
   if (k+1)%220==0:print(json.dumps(dict(stage='raw_quality',records=k+1,total=8580,elapsed_seconds=time.perf_counter()-beg)),flush=True)
 frame=pd.DataFrame(rows);ts=pd.DataFrame(timestamps);qual=pd.DataFrame(quality);an=pd.DataFrame(anomalies);temporal=pd.DataFrame(temp);spec=pd.DataFrame(spectra)
 for f,name in [(frame,'record_metrics.csv'),(ts,'timestamp_quality.csv'),(ts,'sampling_rate_audit.csv'),(qual,'signal_quality_summary.csv'),(an,'signal_anomaly_candidates.csv'),(temporal,'temporal_boundary_metrics.csv'),(spec,'spectral_quality_by_activity.csv')]:csv(f,name,True)
 inventory=[]
 for p in sorted(RAW.rglob('*')):
  if not p.is_file():continue
  inventory.append(dict(path=str(p.relative_to(RAW)),format=p.suffix,bytes=p.stat().st_size,sha256=protected[str(p)],empty=p.stat().st_size==0,parse_status='PARSED_METADATA_OR_SENSOR_BY_THIS_AUDIT'))
 inv=pd.DataFrame(inventory);csv(inv,'raw_file_inventory.csv',True);dupes=inv.groupby('sha256').filter(lambda z:len(z)>1);csv(dupes,'raw_exact_duplicate_files.csv',True)
 publicinv=inv.groupby('format').agg(file_count=('path','count'),total_bytes=('bytes','sum'),minimum_bytes=('bytes','min'),median_bytes=('bytes','median'),maximum_bytes=('bytes','max'),empty_count=('empty','sum')).reset_index();publicinv['exact_duplicate_group_count']=int((inv.groupby('sha256').size()>1).sum());csv(publicinv,'raw_file_inventory.csv')
 complete=pd.read_csv(HERE/'restricted_manifests/analysis/subject_activity_completeness_schema_v2.csv',dtype={'subject_id':str});comp=complete.groupby(['activity','label']).agg(subject_count=('subject_id','nunique'),missing_left=('left_count',lambda v:int((v==0).sum())),missing_right=('right_count',lambda v:int((v==0).sum())),duplicate_session=('session_count',lambda v:int((v!=1).sum())),invalid_channel_definitions=('channels_6',lambda v:int((~v).sum()))).reset_index();csv(comp,'subject_activity_completeness.csv')
 # Timestamp differences use acquisition-relative clocks, not proven absolute sync.
 sync=[]
 for sid,activity in sorted({(s,a) for s,a,w in times}):
  left=times[(sid,activity,'left')];right=times[(sid,activity,'right')];assert len(left)==len(right);offset=left-right;sync.append(dict(subject_id=sid,activity=activity,label=frame[(frame.subject_id==sid)&(frame.activity==activity)].label.iloc[0],mean_offset=offset.mean(),median_offset=np.median(offset),offset_SD=offset.std(),maximum_absolute_offset=abs(offset).max(),first_offset=offset[0],last_offset=offset[-1],linear_relative_clock_drift_slope=np.polyfit((left+right)/2,offset,1)[0],absolute_physical_synchrony_known=False))
 sy=pd.DataFrame(sync);csv(sy,'bilateral_sync_report.csv',True);csv(aggregate(sy,['activity','label'],['mean_offset','median_offset','offset_SD','maximum_absolute_offset','first_offset','last_offset','linear_relative_clock_drift_slope']),'bilateral_sync_report.csv')
 measures=['mean','median','std','variance','minimum','maximum','range','iqr','rms','mad','zero_fraction','nan_fraction','inf_fraction','constant_segment_ratio','maximum_constant_samples','unique_value_count','extreme_repeat_saturation_candidate_ratio']
 csv(aggregate(qual,['activity','sensor','channel'],measures),'signal_quality_summary.csv');csv(aggregate(qual,['activity','sensor'],measures),'signal_quality_by_activity.csv');csv(aggregate(qual,['label','source_condition','sensor'],measures),'signal_quality_by_class.csv')
 ac=an.groupby(['activity','wrist','channel','anomaly_type']).agg(candidate_record_count=('subject_id','count'),affected_subject_count=('subject_id','nunique'),candidate_sample_count=('candidate_count','sum')).reset_index();ac['confirmed_sensor_error']=False;ac['sample_positions']='SERVER_PRIVATE_ONLY';csv(ac,'signal_anomaly_candidates.csv')
 csv(aggregate(ts,['activity','label'],['mean_fs','median_fs','median_dt','maximum_dt','dt_jitter_SD','large_gap_count','estimated_missing_grid_points','estimated_gap_fraction','duplicate_timestamp_count','nonmonotonic_count']),'sampling_rate_audit.csv');csv(aggregate(ts,['activity','label'],['median_dt','maximum_dt','dt_jitter_SD','large_gap_count','estimated_missing_grid_points','duplicate_timestamp_count','nonmonotonic_count']),'timestamp_quality.csv')
 cols=['magnitude_mean','rms','dynamic_rms','variance','mad','jerk_rms','spectral_energy','dominant_frequency','zcr','short_time_energy','autocorrelation_lag1','high_to_low_proxy'];csv(aggregate(temporal,['activity','sensor','window','label'],cols),'temporal_boundary_metrics.csv');csv(aggregate(spec,['activity','sensor','version'],['spectral_energy','dominant_frequency','centroid','bandwidth','spectral_entropy','band_0_3','band_3_12','band_12_15','band_15_50','fraction_3_12','rms','dynamic_rms']),'spectral_quality_by_activity.csv')
 csv(aggregate(frame,['activity','label','source_condition'],['duration_seconds','raw_samples','processed_samples','trim48_elapsed_seconds']),'activity_duration_report.csv')
 pads=[]
 for a,g in frame.groupby('activity'):
  n=int(g.processed_samples.iloc[0]);pads.append(dict(activity=a,record_count=len(g),true_samples=n,STR_fixed_length=False,STR_crop_after48_samples=0,container_padding_to2000_samples=2000-n,container_padding_fraction=(2000-n)/2000,padding_passed_into_wrist_encoder=False,HarNet_padded_samples=max(0,1000-n),HarNet_contexts=1 if n<1000 else 2,HarNet_distinct_original_samples_covered=min(n,2000),HarNet_coverage_fraction=min(n,2000)/n,HarNet_first_last_context_overlap_samples=max(0,2000-n) if n>=1000 else 0,post48_tail_lost_samples=max(0,n-2000)))
 padding=pd.DataFrame(pads);csv(padding,'padding_mask_audit.csv');csv(padding,'sequence_truncation_report.csv');csv(padding,'activity_duration_report_by_branch.csv')
 unit=pd.DataFrame([dict(sensor='Acc',unit='g',record_count=len(rs),metadata_consistent=bool(frame.units_match.all()),conversion_applied='none; native g retained',coordinate_axes='UNKNOWN device-axis orientation and bilateral physical equivalence',hardware_full_scale='UNKNOWN'),dict(sensor='Gyro',unit='rad/s',record_count=len(rs),metadata_consistent=bool(frame.units_match.all()),conversion_applied='none; native rad/s retained',coordinate_axes='UNKNOWN device-axis orientation and bilateral physical equivalence',hardware_full_scale='UNKNOWN')]);csv(unit,'sensor_unit_audit.csv')
 expected={Path(r['raw_path']).resolve() for r in rs};actual=set((RAW/'movement/timeseries').glob('*.txt'));assert expected==actual;assert frame.metadata_rows_equal.all() and frame.units_match.all()
 write(HERE/'analysis/raw_diagnostics_integrity.json',dict(status='PASS_COMPLETE_RAW_DIAGNOSTICS',records=8580,channel_records=len(qual),temporal_rows=len(temporal),task_subject_count=390,all_raw_expected_paths_exact=True,raw_metadata_channel_units_consistent=bool(frame.units_match.all()),all_processed_gyro_exact=True,all_processed_shapes_finite=True,empty_raw_files=int((inv.bytes==0).sum()),exact_duplicate_hash_groups=int((inv.groupby('sha256').size()>1).sum()),large_gap_records=int((ts.large_gap_count>0).sum()),largest_gap_seconds=float(ts.maximum_dt.max()),median_fs_min=float(ts.median_fs.min()),median_fs_max=float(ts.median_fs.max()),mean_fs_min=float(ts.mean_fs.min()),mean_fs_max=float(ts.mean_fs.max()),full6channel_adjacent_repeat_records=int((frame.full6channel_duplicate_adjacent_count>0).sum()),incomplete_subject_activity_rows=int(((complete.left_count!=1)|(complete.right_count!=1)|(complete.session_count!=1)).sum()),new_fits=0,elapsed_seconds=time.perf_counter()-beg))
 print(json.dumps(dict(status='PASS_RAW_DIAGNOSTICS',records=8580,seconds=time.perf_counter()-beg)))

if __name__=='__main__':main()
