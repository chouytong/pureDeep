"""Complete sample-location evidence for extrema-repeat candidates and naming checks."""
from common9 import *
def main():
 qual=pd.read_csv(HERE/'restricted_manifests/analysis/signal_quality_summary.csv',dtype={'subject_id':str});sel=qual[qual.extreme_repeat_saturation_candidate_ratio>0];lookup={(r['subject_id'],r['activity'],r['wrist']):r for r in records()};rows=[];previous=None
 for row in sel.itertuples():
  key=(row.subject_id,row.activity,row.wrist)
  if key!=previous:t,x=read_raw(lookup[key]);previous=key
  j=CHANNELS.index(row.channel);y=x[:,j];flag=((y==y.min())|(y==y.max()))&(np.r_[False,np.diff(y)==0]|np.r_[np.diff(y)==0,False]);ix=np.flatnonzero(flag);rows.append(dict(subject_id=row.subject_id,activity=row.activity,wrist=row.wrist,channel=row.channel,candidate_count=len(ix),sample_positions_json=json.dumps(ix.tolist()),time_positions_json=json.dumps(t[ix].tolist()),hardware_full_scale='UNKNOWN',confirmed_saturation=False))
 frame=pd.DataFrame(rows,columns=['subject_id','activity','wrist','channel','candidate_count','sample_positions_json','time_positions_json','hardware_full_scale','confirmed_saturation']);csv(frame,'saturation_candidate_positions.csv',True)
 if len(frame):out=frame.groupby(['activity','channel']).agg(candidate_axis_record_count=('subject_id','size'),affected_subject_count=('subject_id','nunique'),flagged_samples=('candidate_count','sum')).reset_index()
 else:out=pd.DataFrame(columns=['activity','channel','candidate_axis_record_count','affected_subject_count','flagged_samples'])
 csv(out,'saturation_candidate_counts.csv')
 rs=records();expected=['Time','Accelerometer_X','Accelerometer_Y','Accelerometer_Z','Gyroscope_X','Gyroscope_Y','Gyroscope_Z'];assert all(r['metadata_channels']==expected for r in rs)
 for r in rs:assert Path(r['raw_path']).name==f"{r['subject_id']}_{r['activity']}_{'LeftWrist' if r['wrist']=='left' else 'RightWrist'}.txt"
 for p in (RAW/'patients').glob('*.json'):assert p.name==f"patient_{load(p)['id']}.json"
 for p in (RAW/'movement').glob('*.json'):assert p.name==f"observation_{load(p)['subject_id']}.json"
 write(HERE/'analysis/additional_integrity.json',dict(status='PASS',raw_naming_and_metadata_correspondence=True,all_metadata_sensor_axis_order_exact=True,all_saturation_candidate_positions_private=True,saturation_candidate_axis_records=len(rows),saturation_hardware_confirmation=False,task_execution_clinical_completion_known=False,inter_sensor_latency_calibration_available=False))
 print('PASS_NAMING_AND_CANDIDATE_POSITIONS')
if __name__=='__main__':main()
