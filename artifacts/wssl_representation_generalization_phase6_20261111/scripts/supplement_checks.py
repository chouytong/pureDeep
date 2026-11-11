"""Complete planned group cluster summaries and disclose BH accounting for undefined tests."""
from common import *
from threadpoolctl import threadpool_limits
import importlib.metadata,subprocess,datetime

def main():
 guard();rows=[];part=[]
 # Recover the already-defined Phase5 patient grouping from fresh probabilities.
 for c,i,sp,excluded in splits():
  for seed in [42,43,44]:
   path=HERE/'features'/f'paired_seed{seed}_context{c}_inner{i}.npz'
   with np.load(path,allow_pickle=False) as z:
    ids=check_ids(z['subject_ids'],sp['validation_subjects'],True)
    part.extend(dict(subject_id=sid,target=int(y),context=c,inner=i,seed=seed,frozen_pDD=float(p),adapted_pDD=float(a)) for sid,y,p,a in zip(ids,z['target'],z['frozen_probabilities'][:,1],z['adapted_probabilities'][:,1]))
 f=pd.DataFrame(part).groupby(['subject_id','context','inner']).agg(target=('target','first'),frozen_pDD=('frozen_pDD','mean'),adapted_pDD=('adapted_pDD','mean')).reset_index();f['frozen_correct']=((f.frozen_pDD>.5).astype(int)==f.target).astype(float);f['adapted_correct']=((f.adapted_pDD>.5).astype(int)==f.target).astype(float)
 g=f.groupby('subject_id').agg(target=('target','first'),frozen_correct=('frozen_correct','mean'),adapted_correct=('adapted_correct','mean'));d=g.adapted_correct-g.frozen_correct;g['patient_event']=np.where(d>0,'improved',np.where(d<0,'worsened','unchanged'))
 assert g.patient_event.value_counts().to_dict()=={'unchanged':301,'worsened':46,'improved':43}
 subject=pd.read_csv(HERE/'restricted_manifests/subject_mean_drift_private.csv',dtype={'subject_id':str}).merge(g[['patient_event']],left_on='subject_id',right_index=True,validate='many_to_one')
 subject.to_csv(HERE/'restricted_manifests/subject_mean_drift_with_patient_group_private.csv',index=False)
 rng=np.random.default_rng(SEED);cluster=[];desc=[]
 with threadpool_limits(limits=4):
  for layer,t in subject[subject.layer!='layer4'].groupby('layer'):
   t=t.sort_values('subject_id');ids0=np.flatnonzero(t.target.to_numpy()==0);ids1=np.flatnonzero(t.target.to_numpy()==1);w=np.zeros((10000,len(t)),np.int16);w[:,ids0]=rng.multinomial(len(ids0),np.full(len(ids0),1/len(ids0)),10000);w[:,ids1]=rng.multinomial(len(ids1),np.full(len(ids1),1/len(ids1)),10000)
   x=t.relative_displacement.to_numpy()
   for name,plus,minus in [('patient_improved_minus_worsened',t.patient_event.to_numpy()=='improved',t.patient_event.to_numpy()=='worsened'),('DD_minus_PD',t.target.to_numpy()==1,t.target.to_numpy()==0)]:
    dp=w@plus;dm=w@minus;v=(w@(x*plus))/dp-(w@(x*minus))/dm;valid=np.isfinite(v);ci=np.quantile(v[valid],[.025,.975]);cluster.append(dict(layer=layer,comparison=name,mean_difference=x[plus].mean()-x[minus].mean(),ci_low=ci[0],ci_high=ci[1],valid_draws=int(valid.sum()),estimand='subject12-appearance scalar means; fixed4-appearance seed-probability-mean patient groups or disease classes',not_equivalent_to_per_appearance_event_contrast=True))
   for event,q in t.groupby('patient_event'):desc.append(dict(layer=layer,patient_event=event,subject_count=len(q),relative_displacement_mean=q.relative_displacement.mean(),cosine_distance_mean=q.cosine_distance.mean(),margin_change_mean=q.margin_change.mean()))
 pd.DataFrame(cluster).to_csv(HERE/'analysis/representation_group_subject_cluster.csv',index=False);pd.DataFrame(desc).to_csv(HERE/'analysis/representation_patient_group_summary.csv',index=False)
 # Original frozen analysis q-values remain unchanged. Explicit conservative sensitivity.
 tests=pd.read_csv(HERE/'analysis/representation_paired_tests.csv');tests['bh_q_planned_family_sensitivity']=np.nan
 for family,ix in tests.groupby('family').groups.items():
  p=tests.loc[ix,'wilcoxon_p'].to_numpy();q=bh(np.where(np.isfinite(p),p,1.));q[~np.isfinite(p)]=np.nan;tests.loc[ix,'bh_q_planned_family_sensitivity']=q
 tests.to_csv(HERE/'analysis/representation_tests_planned_BH_sensitivity.csv',index=False)
 write_json(HERE/'analysis/undefined_test_accounting.json',dict(original_BH='applied to finite test p-values; original output preserved',planned_family_sensitivity='additional conservative audit: unavailable preregistered tests treated as p=1 only inside adjustment; their own p and q stay NA',post_result_conservative_sensitivity=True,raw_p_or_bootstrap_CI_changed=False,retention_decision_changed=False,counts=[dict(family=family,planned=len(t),testable=int(t.wilcoxon_p.notna().sum()),undefined=int(t.wilcoxon_p.isna().sum())) for family,t in tests.groupby('family')]))
 # Every exported private feature gets a content hash and source/context references.
 exported=[];extraction=load(HERE/'restricted_manifests/extraction_lineage.json')
 for p in (HERE/'features').rglob('*.npz'):
  exported.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,official_checkpoint_sha256=sha(OFFICIAL),source_code_sha256=sha(HERE/'scripts/paired_inference.py'),mainline_source_registration='configs/ANALYSIS_FREEZE.json + protected_assets',scope='query-only exports; fit/query/excluded/checkpoint/normalization/preprocess/split are in extraction_lineage records'))
 write_json(HERE/'restricted_manifests/all_exported_feature_hashes.json',dict(files=exported,source_lineage=extraction,external_encoder_PADS_fit_subject_ids=[],external_pretraining_subject_membership='unknown, official checkpoint provenance only'))
 write_json(HERE/'analysis/runtime_environment.json',dict(python=sys.version,versions={n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','scipy','scikit-learn']},GPU=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout.strip(),threads=4,dependency_installs=0,formal_environment_changed=False,inference_validation_evaluations=90,new_deep_training=0,new_fits=0,global_legacy_sensor_feature_arrays_decoded=False,model_peak_memory_not_instrumented=True))
 write_json(HERE/'tests/supplement_integrity.json',dict(status='PASS',old_patient_groups_reproduced_43_46_301=True,planned_group_cluster_summaries_completed=True,all_new_private_feature_files_hashed=len(exported),original_raw_statistical_outputs_preserved=True,posthoc_conservative_BH_sensitivity_disclosed=True))
 guard();print('SUPPLEMENTAL_PROVENANCE_AND_CLUSTER_CHECKS_COMPLETE')

if __name__=='__main__':main()
