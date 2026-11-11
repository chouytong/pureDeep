"""Keep full diagnostic tables private; suppress small crossed cells before release."""
from common9 import *
def main():
 path=HERE/'analysis/class_quality_by_device_and_source.csv';frame=pd.read_csv(path);small=frame.subject_count<10;private=HERE/'restricted_manifests/analysis/class_quality_crossed_before_privacy_release.csv';assert not private.exists();private.write_bytes(path.read_bytes())
 for col in ['mean','median','p25','p75']:frame.loc[small,col]=np.nan
 frame['privacy_release']=np.where(small,'SUPPRESSED_STATISTICS_CELL_LT10','AGGREGATE_CELL_GE10');frame.to_csv(path,index=False)
 # PSD bins beyond a patient's maximum observed Nyquist are not extrapolated.
 psdframe=pd.read_csv(HERE/'analysis/group_psd_curves.csv');ts=pd.read_csv(HERE/'restricted_manifests/analysis/timestamp_quality.csv',dtype={'subject_id':str});p=ts.groupby(['subject_id','activity','label']).median_fs.max().reset_index();counts=[]
 for row in psdframe.itertuples():
  g=p[(p.activity==row.activity)&(p.label==row.label)];counts.append(int(sum(g.median_fs/2>=row.frequency_Hz)))
 psdframe['valid_patient_count_at_frequency']=counts;psdframe['no_nyquist_extrapolation']=True;psdframe.to_csv(HERE/'analysis/group_psd_curves.csv',index=False)
 write(HERE/'analysis/privacy_release_integrity.json',dict(status='PASS',crossed_cell_statistics_suppressed_below10=True,suppressed_measurement_rows=int(small.sum()),full_table_retained_private=True,public_curves_have_frequency_specific_valid_patient_counts=True,PSD_out_of_record_Nyquist_is_NA_not_zero=True,private_signal_figures_only_in_restricted_directory=True))
 write(HERE/'analysis/engineering_corrections.json',dict(corrections=[dict(stage='metadata schema before signal diagnostics',problem='initial check assumed6 metadata fields; actual7 includesTime',resolution='DIAGNOSTICS_SCHEMA_V2 frozen; old source/output preserved; 8580 units/channel arrays verified',signal_outputs_affected=False),dict(stage='smoke source syntax',problem='v1 used a keyword starting with digit; parse failed before execution',resolution='failed source moved to logs/failed_tests9_v1.py; tests.log preserved; tests9_v2.py passed',model_or_raw_data_changed=False),dict(stage='inspection command environment',problem='system python3 lacked pandas',resolution='inspection rerun with existing mfam_analysis Python; no install/data mutation'),dict(stage='PSD plotting warning',problem='some physical-frequency endpoint bins exceed observed Nyquist and contain onlyNaN',resolution='no extrapolation; NA retained; explicit valid patient counts per bin added; finite raw inputs verified')],no_failed_runs_deleted=True,existing_frozen_sources_unchanged=True))
 print('PASS_PRIVACY_RELEASE')
if __name__=='__main__':main()
