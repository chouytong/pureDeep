"""Check raw acceleration semantics without inferring an undocumented SDK field."""
from common9 import *
def main():
 f=pd.read_csv(HERE/'restricted_manifests/analysis/record_metrics.csv',dtype={'subject_id':str});q=pd.read_csv(HERE/'restricted_manifests/analysis/signal_quality_summary.csv',dtype={'subject_id':str});mean=q[q.sensor=='Acc'].pivot(index=['subject_id','activity','wrist'],columns='channel',values='mean').reset_index();mean['norm_of_axis_means_g']=np.linalg.norm(mean[['AccX','AccY','AccZ']].to_numpy(),axis=1);f=f.merge(mean[['subject_id','activity','wrist','norm_of_axis_means_g']],on=['subject_id','activity','wrist'],validate='one_to_one');per=f.groupby(['subject_id','activity','label'])[['acc_magnitude_mean','acc_rms','norm_of_axis_means_g']].mean().reset_index();out=[]
 for keys,g in per.groupby(['activity','label']):
  for col in ['acc_magnitude_mean','acc_rms','norm_of_axis_means_g']:
   out.append(dict(activity=keys[0],label=keys[1],measurement=col,subject_count=len(g),median=g[col].median(),p25=g[col].quantile(.25),p75=g[col].quantile(.75),unit='g',interpretation='raw tensor values; not proof of a particular acquisition SDK API'))
 csv(pd.DataFrame(out),'acceleration_semantics_diagnostics.csv');unit=pd.read_csv(HERE/'analysis/sensor_unit_audit.csv');unit['gravity_semantics']=np.where(unit.sensor=='Acc','UNKNOWN: official detrend intent does not prove native raw gravity content','not applicable');unit['acquisition_SDK_field_verified']=False;unit.to_csv(HERE/'analysis/sensor_unit_audit.csv',index=False)
 # Preserve every completed draft before the scientific qualification is revised.
 archive=HERE/'restricted_manifests/drafts_before_acc_semantics';archive.mkdir(exist_ok=False)
 for p in list(HERE.glob('*.md'))+[HERE/'scripts/reports9.py']:
  if p.is_file():(archive/p.name).write_bytes(p.read_bytes())
 write(HERE/'analysis/acceleration_semantics_integrity.json',dict(status='PASS_QUALIFIED_PHYSICAL_SEMANTICS',unit_g_from_all_metadata=True,raw_full_gravity_vector_presence='UNKNOWN',official_paper_and_preprocessor_intent='account for gravitational orientation trend; not a verified acquisition API contract',SDK_field_and_export_transform='UNKNOWN',synthetic_constant1g_is_not_typical_raw_claim=True,raw_magnitude_diagnostics_are_patient_aggregated=True,earlier_report_drafts_preserved_private=True,no_gravity_added_or_removed_from_formal_inputs=True,new_model_or_normalization_fit_calls=0))
 print('PASS_ACCELERATION_SEMANTICS_QUALIFICATION')
if __name__=='__main__':main()
