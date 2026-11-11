"""Inspect existing normalization, canonical ID sets and duplicate-content links; no refit."""
from common9 import *
import torch

def main():
 d=load(MAIN/'splits/pads_classification/v3_nested_cv/seed42/nested_cv_splits.json');allids=set(load(HERE/'restricted_manifests/subject_metadata.json')['subjects']);overlap=[];private=[];statsrows=[];appearance={};normcount=0
 for outer in d['outer']:
  c=int(outer['outer_fold']);excluded=set(map(str,outer['test_subjects']))
  for inner in outer['inner_folds']:
   i=int(inner['inner_fold']);fit=set(map(str,inner['train_subjects']));query=set(map(str,inner['validation_subjects']));assert not fit&query and not fit&excluded and not query&excluded and fit|query|excluded==allids
   overlap.append(dict(context=c,inner=i,train_count=len(fit),validation_count=len(query),excluded_count=len(excluded),train_validation_overlap=0,train_excluded_overlap=0,validation_excluded_overlap=0,canonical_union_count=390,split_unit='subject; all activities/wrists inherited',outer_data_loader_created=False))
   for sid in query:appearance[sid]=appearance.get(sid,0)+1
   previous=None
   for seed in [42,43,44]:
    p=ROOT/f'artifacts/phase3b_external_ssl_20260928/runs/b_str_pretrained/seed{seed}/outer_{c}/inner_{i}';norm=load(p/'normalization.json');split=load(p/'split.json');assert set(norm['subject_ids'])==fit and set(split['train_subjects'])==fit and set(split['validation_subjects'])==query
    mean=np.asarray(norm['mean'],np.float32);std=np.asarray(norm['std'],np.float32);assert mean.shape==std.shape==(11,2,6,1) and np.isfinite(mean).all() and np.isfinite(std).all() and (std>=1e-6).all();ck=torch.load(p/'checkpoints/best.pt',map_location='cpu',weights_only=False);assert np.array_equal(mean,ck['normalization']['mean'].numpy()) and np.array_equal(std,ck['normalization']['std'].numpy())
    if previous is not None:assert np.array_equal(mean,previous[0]) and np.array_equal(std,previous[1])
    previous=(mean.copy(),std.copy());normcount+=1;private.append(dict(seed=seed,context=c,inner=i,fit_subject_ids=sorted(fit),query_subject_ids=sorted(query),excluded_subject_ids=sorted(excluded),normalization_sha256=sha(p/'normalization.json'),checkpoint_sha256=sha(p/'checkpoints/best.pt'),normalization_path=str(p/'normalization.json'),metadata_train_membership_pass=True,checkpoint_values_exact=True,new_refit=False))
    for a,activity in enumerate(ACTS):
     for w,wrist in enumerate(SIDES):
      for j,channel in enumerate(CHANNELS):statsrows.append(dict(context=c,inner=i,seed=seed,activity=activity,wrist=wrist,channel=channel,stored_mean=float(mean[a,w,j,0]),stored_std=float(std[a,w,j,0])))
 frame=pd.DataFrame(statsrows);summary=frame.groupby(['activity','wrist','channel']).agg(stored_norm_count=('stored_mean','size'),mean_of_stored_means=('stored_mean','mean'),minimum_stored_mean=('stored_mean','min'),maximum_stored_mean=('stored_mean','max'),mean_of_stored_stds=('stored_std','mean'),minimum_stored_std=('stored_std','min'),maximum_stored_std=('stored_std','max')).reset_index();summary['new_normalization_fit']=False;csv(summary,'normalization_statistics.csv');csv(pd.DataFrame(overlap),'subject_overlap_check.csv');write(HERE/'restricted_manifests/normalization_lineage.json',dict(records=private,validation_appearance_counts=appearance));csv(frame,'normalization_statistics_private45.csv',True)
 inv=pd.read_csv(HERE/'restricted_manifests/analysis/raw_file_inventory.csv');dupes=inv[inv.format=='.txt'].groupby('sha256').filter(lambda x:len(x)>1);cross_subject=0
 for digest,g in dupes.groupby('sha256'):
  if len({Path(x).name.split('_')[0] for x in g.path})>1:cross_subject+=1
 write(HERE/'analysis/normalization_split_integrity.json',dict(status='PASS_EXISTING_METADATA_NO_REFIT',canonical_splits=15,checkpoint_normalization_pairs=normcount,normalization_new_fit_calls=0,normalization_shape=[11,2,6,1],train_membership_exact=True,stored_checkpoint_values_exact=True,seed_norm_values_same_within_split=True,minimum_stored_std=float(frame.stored_std.min()),unit_affine_transform_invertible_at_input=True,per_subject_normalization=False,per_activity_wrist_channel=True,all_windows_same_patient_role=True,cross_subject_exact_duplicate_signal_groups=cross_subject,validation_appearance_min=min(appearance.values()),validation_appearance_max=max(appearance.values()),overlapping_splits_not_independent=True,outer_signals_or_performance_read=False,global_harnet_cache_decoded_by_phase9=False))
 print('PASS_NORMALIZATION_METADATA_NO_REFIT')

if __name__=='__main__':main()
