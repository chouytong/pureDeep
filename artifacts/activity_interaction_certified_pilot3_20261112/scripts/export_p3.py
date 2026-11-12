"""Fresh supervised bilateral tokens from each Certified WSSL; no legacy token reuse."""
from common_p3 import *
from src.datasets.subject_activity import SubjectActivityDataset,collate_subject_activities
from torch.utils.data import DataLoader
import src.datasets.manifest as manifest_loader
import gc

def main():
 assert load(HERE/'analysis/audit_integrity.json')['status']=='PASS';guard();freeze('EXPORT',[Path(__file__),HERE/'scripts/common_p3.py'],new_training=False,base_eval=True,roles=['train','validation'],logit_abs=1e-5,probability_abs=2e-7)
 torch.set_num_threads(4);b=binding();sp,excluded=split();allowed=set(b['fit_subject_ids'])|set(b['query_subject_ids']);first=b['seeds']['42'];data,norm=norm_binding(VERSION,first['manifest'],first['normalization'],0,0);records_=[r for r in configured_records(data) if r.subject_id in allowed];paths={r.left_path.resolve() for r in records_}|{r.right_path.resolve() for r in records_};original_load=manifest_loader._load_array
 def guarded(path,delimiter):
  if Path(path).resolve() not in paths:raise PermissionError('Excluded/unbound signal rejected')
  return original_load(path,delimiter)
 manifest_loader._load_array=guarded;index=[];summary=[];total_subjects=0;total_batches=0;start=time.perf_counter()
 try:
  try:guarded('/forbidden/excluded_signal.npy',',')
  except PermissionError:pass
  else:raise AssertionError('No input allowlist')
  ext=external(allowed)
  for seed in SEEDS:
   seed_everything(seed,True);r=b['seeds'][str(seed)];base=model_for('wssl',cfg(seed)).cuda().eval();ck=torch.load(r['checkpoint'],map_location='cuda',weights_only=False);base.load_state_dict(ck['model_state'],strict=True)
   for p in base.parameters():p.requires_grad_(False)
   before=fingerprint(base);captured={}
   def hook(module,args):
    assert len(args)==2;captured['bilateral']=args[0].detach().clone();captured['mask']=args[1].detach().clone();captured['calls']=captured.get('calls',0)+1
   handle=base.backbone.activity_aggregator.register_forward_pre_hook(hook)
   for role,ids in [('train',set(b['fit_subject_ids'])),('validation',set(b['query_subject_ids']))]:
    ds=SubjectActivityDataset([v for v in records_ if v.subject_id in ids],ACTS,cfg(seed)['data'],torch.tensor(norm['mean']),torch.tensor(norm['std']),'validation');assert set(ds.subject_ids)==ids and ds.activity_dropout==ds.rotation_degrees==0
    loader=DataLoader(ds,batch_size=8,shuffle=False,num_workers=4,pin_memory=True,persistent_workers=True,generator=torch.Generator().manual_seed(seed+1),collate_fn=collate_subject_activities);values={k:[] for k in ['subject_ids','target','bilateral','mask','logits','probabilities']};role_start=time.perf_counter();maxhead=0.;maxrepeat=0.
    with torch.inference_mode():
     for batch in loader:
      current=list(map(str,batch['subject_id']));assert set(current)<=ids;base.ssl_features=torch.from_numpy(np.stack([ext[s] for s in current])).cuda();x=[batch[k].cuda(non_blocking=True) for k in ['x','wrist_mask','activity_mask','activity_lengths']];captured.clear();out=base(*x);assert captured['calls']==1 and torch.equal(captured['mask'],batch['activity_mask'].cuda());h=captured['bilateral'];mask=captured['mask'];assert h.shape==(len(current),11,258) and mask.shape==(len(current),11) and mask.all();assert torch.isfinite(h).all()
      # Calling the aggregator to reconstruct the head is observed separately from full forward.
      a=base.backbone.activity_aggregator(h,mask);s=base.backbone.structured_token_residual(h,mask);head_logits=base.backbone.classifier(a['subject_embedding'])+s['structured_residual_logits'];err=float((head_logits-out['logits']).abs().max());maxhead=max(maxhead,err);assert err<=1e-5
      values['subject_ids']+=current;values['target']+=batch['y'].tolist()
      for key,value in [('bilateral',h),('mask',mask),('logits',out['logits']),('probabilities',out['probabilities'])]:values[key].append(value.cpu().numpy())
      total_batches+=1;total_subjects+=len(current)
    assert len(values['subject_ids'])==len(ids) and set(values['subject_ids'])==ids;arrays={k:np.asarray(v) if k in ['subject_ids','target'] else np.concatenate(v) for k,v in values.items()};archive_logits_err=None;archive_prob_err=None;decisions_equal=None;logits_bitwise=None
    if role=='validation':
     with np.load(r['archived_prediction'],allow_pickle=False) as z:lookup={str(s):j for j,s in enumerate(z['subject_ids'])};ix=[lookup[s] for s in arrays['subject_ids']];assert np.array_equal(arrays['target'],z['target'][ix]);archive_logits_err=float(np.max(abs(arrays['logits']-z['logits'][ix])));archive_prob_err=float(np.max(abs(arrays['probabilities']-z['probabilities'][ix])));decisions_equal=bool(np.array_equal(arrays['probabilities'][:,1]>.5,z['probabilities'][ix,1]>.5));logits_bitwise=bool(np.array_equal(arrays['logits'],z['logits'][ix]))
     assert archive_logits_err<=1e-5 and archive_prob_err<=2e-7 and decisions_equal
    p=PRIVATE/'tokens'/f'seed{seed}_{role}.npz';assert not p.exists();np.savez_compressed(p,**arrays,activities=np.asarray(ACTS),wrist_order=np.asarray(['left','right']),version=np.asarray(VERSION),base_checkpoint_sha256=np.asarray(r['checkpoint_sha256']),normalization_sha256=np.asarray(r['normalization_sha256']),manifest_sha256=np.asarray(r['manifest_sha256']),role=np.asarray(role));index.append(dict(seed=seed,role=role,path=str(p),sha256=sha(p),base_checkpoint_sha256=r['checkpoint_sha256'],normalization_sha256=r['normalization_sha256'],subject_count=len(ids)))
    summary.append(dict(seed=seed,role=role,subjects=len(ids),tokens_per_subject=11,token_dimension=258,all_activity_masks_valid=True,head_vs_full_max_logit_error=maxhead,archived_max_logit_error=archive_logits_err,archived_max_probability_error=archive_prob_err,archived_logits_bitwise=logits_bitwise,archived_decisions_equal=decisions_equal,export_wall_seconds=time.perf_counter()-role_start));print(json.dumps(clean(summary[-1])),flush=True);del loader,ds
   handle.remove();assert fingerprint(base)==before and not base.training and all(p.grad is None for p in base.parameters());del base,ck;gc.collect();torch.cuda.empty_cache()
 finally:manifest_loader._load_array=original_load
 write(PRIVATE/'manifests/token_index.json',dict(packages=index));pd.DataFrame(summary).to_csv(HERE/'analysis/token_export_summary.csv',index=False);write(HERE/'analysis/export_integrity.json',dict(status='PASS',checkpoints=3,role_packages=6,subject_forward_instances=total_subjects,signal_forward_batches=total_batches,unique_allowed_subjects=len(allowed),unique_validation_subjects=len(b['query_subject_ids']),no_training=True,baseline_states_unchanged=True,real_lengths_and_masks_used=True,old_supervised_token_packages_read=0,excluded_signal_cache_prediction_arrays_read=0,normalization_refits=0,L1_solves=0,HarNet_reextractions=0,elapsed_wall_seconds=time.perf_counter()-start,protected_files_unchanged=guard()))
if __name__=='__main__':main()
