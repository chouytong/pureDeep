"""Pre-signal schema correction. Keep startup/v1 outputs unchanged for provenance."""
from common9 import *
def main():
 assert load(HERE/'analysis/startup_integrity.json')['raw_arrays_decoded_by_startup']==0
 rs=records();expected=['Time','Accelerometer_X','Accelerometer_Y','Accelerometer_Z','Gyroscope_X','Gyroscope_Y','Gyroscope_Z'];assert all(r['metadata_channels']==expected and r['metadata_units']==['s']+['g']*3+['rad/s']*3 for r in rs)
 old=pd.read_csv(HERE/'restricted_manifests/analysis/subject_activity_completeness.csv',dtype={'subject_id':str});new=old.copy();new['channels_6']=True;csv(new,'subject_activity_completeness_schema_v2.csv',True)
 m=pd.DataFrame(rs).drop_duplicates('subject_id');counts=m.groupby(['device','label']).size().unstack(fill_value=0).reset_index();counts['association_only_no_causal_claim']=True;csv(counts,'device_class_distribution.csv')
 write(HERE/'analysis/metadata_schema_correction.json',dict(status='PASS_BEFORE_SIGNAL_STATISTICS',reason='Time is explicit metadata channel0; 7 metadata columns correspond to timestamp plus6 sensors',initial_schema_check_was_incorrect=True,formal_data_or_code_changed=False,original_v1_source_and_outputs_preserved=True,original_DIAGNOSTICS_freeze_preserved=True,source_v2='scripts/diagnostics9_schema_v2.py',raw_signal_arrays_decoded_by_correction=0,device_series3_metadata_subject_count=int(sum(m.device=='Apple Watch Series 3')),device_series4_metadata_subject_count=int(sum(m.device=='Apple Watch Series 4')),device_validation_status='metadata value; hardware logs not available',new_fit_calls=0))
 freeze('DIAGNOSTICS_SCHEMA_V2',[Path(__file__),HERE/'scripts/diagnostics9_schema_v2.py',HERE/'scripts/common9.py',HERE/'PROTOCOL.md'],reason='correct metadata schema before any raw signal diagnostic; originalv1 retained',rules_thresholds_changed=False,additional_device_analysis='descriptive only; no new significance family')
 print('PASS_METADATA_TIME_PLUS_SIX_SENSOR_SCHEMA')
if __name__=='__main__':main()
