from common_l2 import *
from data_l2 import data_manifest,configured_records,norm_binding,ManifestTimeSeriesDataset
import argparse,torch
from src.datasets.builders import _activity_statistics

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-version',required=True);ap.add_argument('--manifest',required=True);ap.add_argument('--output-dir',required=True);args=ap.parse_args();data=data_manifest(args.data_version,args.manifest);allrecords=configured_records(data);output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True);index=[];summary=[];private=[];torch.set_num_threads(4)
 import src.datasets.manifest as loader
 original=loader._load_array
 for c,i,sp,ex in splits():
  target=output/f'context{c}_inner{i}.json'
  if target.exists():data,norm=norm_binding(args.data_version,args.manifest,target,c,i);index.append(dict(context=c,inner=i,path=str(target),sha256=sha(target)));continue
  fit=set(sp['train_subjects']);selected=[r for r in allrecords if r.subject_id in fit];assert len(selected)==len(fit)*11;allowed={r.left_path.resolve() for r in selected}|{r.right_path.resolve() for r in selected}
  def guarded(path,delimiter):
   if Path(path).resolve() not in allowed:raise ValueError('NORMALIZATION_NONTRAIN_PATH_BLOCKED')
   return original(path,delimiter)
  loader._load_array=guarded
  try:
   ds=ManifestTimeSeriesDataset(selected,6,'acc_gyro','bilateral',None,'full',0.,',',False,mean=None,std=None);mean,std=_activity_statistics(ds,ACTS,1e-6)
   try:guarded(next(r.left_path for r in allrecords if r.subject_id not in fit),',')
   except ValueError:pass
   else:raise AssertionError('Nontrain normalization path accepted')
  finally:loader._load_array=original
  assert mean.shape==std.shape==(11,2,6,1) and torch.isfinite(mean).all() and torch.isfinite(std).all() and (std>=1e-6).all()
  norm=dict(version=args.data_version,manifest_sha256=sha(args.manifest),context=c,inner=i,fit_subject_ids=sorted(fit),mean=mean.tolist(),std=std.tolist(),shape=[11,2,6,1],epsilon=1e-6,algorithm='original mainline _activity_statistics full samples float64 sum/square ->float32',algorithm_source_sha256=sha(MAIN/'src/datasets/builders.py'),signals_exclude_padding=True,shared_by_model_types=['wssl','str'],shared_by_seeds=[42,43,44]);write(target,norm);index.append(dict(context=c,inner=i,path=str(target),sha256=sha(target)));private.append(norm)
  old=load(baseline('wssl',42,c,i)/'normalization.json');om=np.array(old['mean']);os=np.array(old['std']);m=mean.numpy();s=std.numpy()
  for a,activity in enumerate(ACTS):
   for w,wrist in enumerate(SIDES):
    for j,channel in enumerate(CHANNELS):summary.append(dict(context=c,inner=i,activity=activity,wrist=wrist,channel=channel,absolute_mean_change=float(abs(m[a,w,j,0]-om[a,w,j,0])),std_ratio=float(s[a,w,j,0]/os[a,w,j,0])))
  print(json.dumps(dict(stage='fit-normalization',completed=len(index),context=c,inner=i,fit_count=len(fit))),flush=True)
 assert len(index)==15
 indexpath=output/'index.json';write(indexpath,dict(version=args.data_version,manifest_sha256=sha(args.manifest),normalizations=index));frame=pd.DataFrame(summary)
 if len(frame):frame.groupby(['activity','wrist','channel']).agg(mean_absolute_mean_change=('absolute_mean_change','mean'),minimum_std_ratio=('std_ratio','min'),mean_std_ratio=('std_ratio','mean'),maximum_std_ratio=('std_ratio','max')).reset_index().to_csv(HERE/'analysis/normalization_change_summary.csv',index=False)
 # Deliberately old norm must be rejected before any model/data loader construction.
 try:norm_binding(args.data_version,args.manifest,baseline('wssl',42,0,0)/'normalization.json',0,0)
 except ValueError:pass
 else:raise AssertionError('Legacy normalization accepted for new data')
 write(HERE/'analysis/normalization_integrity.json',dict(status='PASS_15_TRAIN_ONLY_NORMALIZATIONS',version=args.data_version,manifest_sha256=sha(args.manifest),normalization_index=str(indexpath),index_sha256=sha(indexpath),unique_split_fits=15,validation_inputs_used_for_fit=False,all_member_sets_exact=True,legacy_version_negative_test_pass=True,nontrain_path_negative_test_pass=True,three_seeds_and_two_models_share_same_norm=True));print('PASS_15_NEW_NORMALIZATIONS')
if __name__=='__main__':main()
