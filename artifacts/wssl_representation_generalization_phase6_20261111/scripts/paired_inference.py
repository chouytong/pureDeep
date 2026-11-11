"""Validation-only raw-file inference; no decoding of historical sensor feature caches."""
from common import *
import os,copy,gc,time,torch
from scipy.signal import resample_poly
from torch.utils.data import DataLoader
from src.datasets.subject_activity import SubjectActivityDataset,collate_subject_activities
from src.utils.seed import seed_everything
from independent_wssl import IndependentWSSL

def fingerprint(m):return {k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in m.state_dict().items()}
def legacy_flags():
 torch.use_deterministic_algorithms(False);torch.backends.cudnn.deterministic=False;torch.backends.cudnn.benchmark=False
def classify_flags(seed):seed_everything(seed,True)

def parts(model,inputs,ssl):
 x,wm,am,lengths=inputs;b=model.backbone
 _,local=b._encode_activities_with_wrists(x,wm,am,lengths);valid=wm.bool()&am.bool()[...,None]
 added=model.wrist_projection(model.norm(ssl))*valid[...,None];fused=local+added;n=len(x)
 keep=am.reshape(-1).nonzero(as_tuple=False).squeeze(1)
 f,_=b.wrist_fusion.fusion_features(fused.reshape(-1,2,64).index_select(0,keep),valid.reshape(-1,2).index_select(0,keep))
 bilateral=fused.new_zeros((n*11,258)).index_copy(0,keep,f).reshape(n,11,258)
 a=b.activity_aggregator(bilateral,am);s=b.structured_token_residual(bilateral,am)
 logits=b.classifier(a['subject_embedding'])+s['structured_residual_logits']
 return dict(logits=logits,projected=added,final_wrist=fused,local=local)

class RawExternal:
 def __init__(self,official):
  self.first4=torch.nn.Sequential(*list(official.feature_extractor.children())[:4]).eval().cuda();self.last=official.feature_extractor.layer5.eval().cuda()
  self.raw_keys,self.flat_keys,self.count_map=layout();self.index={k:j for j,k in enumerate(self.raw_keys)};self.pool=HERE/'features/external_by_subject';self.pool.mkdir(exist_ok=True)
  self.io=[];self.reconstructed_subjects=[]
 def ensure(self,allowed):
  check_ids(sorted(allowed),allowed,True);missing=[s for s in sorted(allowed) if not (self.pool/f'{s}.npz').exists()]
  if not missing:return
  rec=records(set(missing));window={};h4={s:np.zeros((11,2,2,512,3),np.float32) for s in missing};h5={s:np.zeros((11,2,2,1024),np.float32) for s in missing};cts={s:np.zeros((11,2),np.int16) for s in missing}
  for r in rec:
   assert r.subject_id in allowed
   for w in range(2):
    path=raw_path(r,w);assert path.is_file();raw=np.loadtxt(path,delimiter=',',dtype=np.float32)
    acc=raw[48:,1:4];assert raw.shape[1]==7 and len(acc)==(r.left_length if w==0 else r.right_length) and np.isfinite(acc).all();acc=np.clip(acc,-3,3)
    if len(acc)<1000:
     before=(1000-len(acc))//2;segments=[np.pad(acc,((before,1000-len(acc)-before),(0,0)),mode='edge')]
    else:segments=[acc[:1000],acc[-1000:]]
    a=ACTS.index(r.activity);cts[r.subject_id][a,w]=len(segments)
    for k,segment in enumerate(segments):
     key=(r.subject_id,a,w,k);assert key in self.index;x=resample_poly(segment,3,10,axis=0).T.astype(np.float32);assert x.shape==(3,300) and np.isfinite(x).all();window[key]=x
    self.io.append(dict(subject_id=r.subject_id,activity=r.activity,wrist=w,raw_path=str(path),raw_sha256=sha(path),window_count=len(segments)))
  legacy_flags()
  with torch.inference_mode():
   for lo in range(0,len(self.raw_keys),64):
    keys=self.raw_keys[lo:lo+64];selected=[j for j,k in enumerate(keys) if k in window]
    if not selected:continue
    x=np.zeros((len(keys),3,300),np.float32)
    for j in selected:x[j]=window[keys[j]]
    out=self.first4(torch.from_numpy(x).cuda()).cpu().numpy();assert out.shape==(len(keys),512,3)
    for j in selected:
     sid,a,w,k=keys[j];h4[sid][a,w,k]=out[j]
   for lo in range(0,len(self.flat_keys),64):
    keys=self.flat_keys[lo:lo+64];selected=[j for j,k in enumerate(keys) if k[0] in h4]
    if not selected:continue
    x=np.zeros((len(keys),512,3),np.float32)
    for j in selected:
     sid,a,w,k=keys[j];x[j]=h4[sid][a,w,k]
    out=self.last(torch.from_numpy(x).cuda()).flatten(1).cpu().numpy();assert out.shape==(len(keys),1024)
    for j in selected:
     sid,a,w,k=keys[j];h5[sid][a,w,k]=out[j]
  for sid in missing:
   assert np.isfinite(h4[sid]).all() and np.isfinite(h5[sid]).all()
   mean=h5[sid].sum(axis=2)/cts[sid][...,None];mid=h4[sid].mean(axis=-1).sum(axis=2)/cts[sid][...,None]
   dest=self.pool/f'{sid}.npz';assert not dest.exists();np.savez_compressed(dest,subject_id=np.asarray(sid),activities=np.asarray(ACTS),wrist_order=np.asarray(['left','right']),layer4_windows=h4[sid],layer4_mean=mid.astype(np.float32),layer5=mean.astype(np.float32),window_counts=cts[sid])
  self.reconstructed_subjects.extend(missing)
 def batch(self,ids,allowed):
  check_ids(ids,allowed);out={k:[] for k in ['layer4_windows','layer4_mean','layer5','window_counts']}
  for sid in ids:
   assert sid in allowed
   with np.load(self.pool/f'{sid}.npz',allow_pickle=False) as z:
    assert str(z['subject_id'])==sid and z['activities'].tolist()==ACTS and z['wrist_order'].tolist()==['left','right']
    for k in out:out[k].append(z[k].copy())
  return {k:np.stack(v) for k,v in out.items()}

def main():
 assert load(HERE/'analysis/pre_inference_asset_gate.json')['status']=='PASS_ALL45'
 os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.set_num_threads(4);guard();start=time.perf_counter()
 sys.path.insert(0,str(VENDOR));import hubconf
 official=hubconf.harnet10(pretrained=True,my_device='cpu',class_num=2).eval();rawstate=torch.load(OFFICIAL,map_location='cpu',weights_only=False)
 for k,v in official.feature_extractor.state_dict().items():assert torch.equal(v,rawstate['module.feature_extractor.'+k]),k
 for p in official.parameters():p.requires_grad_(False)
 ext=RawExternal(official);official_before=fingerprint(official.feature_extractor);rows=[];lineage=[];tests=[];composition=[]
 import src.datasets.manifest as manifest
 original_load=manifest._load_array
 try:
  for c,i,sp,excluded in splits():
   allowed=set(sp['validation_subjects']);rec=records(allowed);paths={r.left_path.resolve() for r in rec}|{r.right_path.resolve() for r in rec}
   def guarded_load(path,delimiter):
    if Path(path).resolve() not in paths:raise PermissionError('Signal outside current validation allowlist')
    return original_load(path,delimiter)
   manifest._load_array=guarded_load
   # Negative access check must fail before I/O.
   try:guarded_load(Path('/forbidden/outer_signal.npy'),',')
   except PermissionError:pass
   else:raise AssertionError('Allowlist did not fail closed')
   ext.ensure(allowed);ordered=sorted(allowed);shared=ext.batch(ordered,allowed)
   layer4_path=HERE/'features'/f'layer4_context{c}_inner{i}.npz';assert not layer4_path.exists();np.savez_compressed(layer4_path,subject_ids=np.asarray(ordered),activities=np.asarray(ACTS),wrist_order=np.asarray(['left','right']),layer4_windows=shared['layer4_windows'],layer4_mean=shared['layer4_mean'],window_counts=shared['window_counts'])
   for seed in [42,43,44]:
    classify_flags(seed);frozen_ck=torch.load(stage('frozen',seed,c,i)/'checkpoints/best.pt',map_location='cpu',weights_only=False);adapted_ck=torch.load(stage('adapted',seed,c,i)/'checkpoints/best.pt',map_location='cpu',weights_only=False)
    config=cfg(seed);frozen=IndependentWSSL(config).eval().cuda();adapted=IndependentWSSL(config).eval().cuda();frozen.load_state_dict(frozen_ck['model_state'],strict=True);adapted.load_state_dict({k[5:]:v for k,v in adapted_ck['model_state'].items() if k.startswith('base.')},strict=True)
    last=copy.deepcopy(ext.last).eval().cuda();last.load_state_dict({k[7:]:v for k,v in adapted_ck['model_state'].items() if k.startswith('layer5.')},strict=True)
    for name,v in last.state_dict().items():
     if any(k in name for k in ['running_mean','running_var','num_batches_tracked']):assert torch.equal(v,ext.last.state_dict()[name]),name
    for m in [frozen,adapted,last]:
     for p in m.parameters():p.requires_grad_(False)
    initial=[fingerprint(m) for m in [frozen,adapted,last]]
    ds=SubjectActivityDataset(rec,ACTS,config['data'],frozen_ck['normalization']['mean'],frozen_ck['normalization']['std'],'validation');assert set(ds.subject_ids)==allowed
    loader=DataLoader(ds,batch_size=8,shuffle=False,num_workers=0,collate_fn=collate_subject_activities)
    output={k:[] for k in ['frozen_layer5','adapted_layer5','frozen_projected','adapted_projected','frozen_final_wrist','adapted_final_wrist','frozen_local','adapted_local','frozen_logits','adapted_logits','frozen_probabilities','adapted_probabilities','wrist_mask','activity_mask','activity_lengths']};ids=[];labels=[]
    with torch.inference_mode():
     for number,batch in enumerate(loader):
      current=list(batch['subject_id']);check_ids(current,allowed);external=ext.batch(current,allowed);h4=torch.from_numpy(external['layer4_windows']).cuda();count=torch.from_numpy(external['window_counts']).cuda();ssl=torch.from_numpy(external['layer5']).cuda()
      x=[batch[k].cuda() for k in ['x','wrist_mask','activity_mask','activity_lengths']];before=x[0].clone();flat=h4.reshape(-1,512,3);valid=(torch.arange(2,device='cuda')<count[...,None]).reshape(-1);ix=valid.nonzero(as_tuple=False).squeeze(1)
      z=last(flat.index_select(0,ix)).flatten(1);adssl=z.new_zeros((len(current)*22,1024)).index_add(0,ix//2,z);adssl=(adssl/count.reshape(-1,1)).reshape(len(current),11,2,1024)
      a=parts(frozen,x,ssl);b=parts(adapted,x,adssl)
      assert torch.equal(before,x[0]) and not any(m.training for m in [frozen,adapted,last])
      if number==0:
       assert torch.equal(a['logits'],frozen(*x,ssl_features=ssl)['logits']) and torch.equal(b['logits'],adapted(*x,ssl_features=adssl)['logits'])
       tests.append(dict(context=c,inner=i,seed=seed,parts_equal_explicit_forward=True,inputs_not_mutated=True,forbidden_IO_rejected=True))
      for name,tensor in [('frozen_layer5',ssl),('adapted_layer5',adssl)]:output[name].append(tensor.cpu().numpy())
      for prefix,v in [('frozen',a),('adapted',b)]:
       for k in ['projected','final_wrist','local','logits']:output[prefix+'_'+k].append(v[k].cpu().numpy())
       output[prefix+'_probabilities'].append(torch.softmax(v['logits'],1).cpu().numpy())
      for k in ['wrist_mask','activity_mask','activity_lengths']:output[k].append(batch[k].numpy())
      ids.extend(current);labels.extend(batch['y'].tolist())
    values={k:np.concatenate(v) for k,v in output.items()};check_ids(ids,allowed,True);assert ids==ordered;y=np.asarray(labels);errors={}
    for kind in ['frozen','adapted']:
     saved=prediction(stage(kind,seed,c,i)/'predictions/validation.csv',allowed).loc[ids];assert np.array_equal(y,saved.target)
     p=values[kind+'_probabilities'];err=float(abs(p-saved[['probability_pd','probability_dd']].to_numpy()).max());errors[kind]=err
     assert err<=1e-12, f'{kind} probability identity failed {seed}/{c}/{i}: {err}'
     assert np.array_equal(p.argmax(1),saved.prediction)
    for m,prior in zip([frozen,adapted,last],initial):assert fingerprint(m)==prior
    path=HERE/'features'/f'paired_seed{seed}_context{c}_inner{i}.npz';assert not path.exists();np.savez_compressed(path,subject_ids=np.asarray(ids),target=y,activities=np.asarray(ACTS),wrist_order=np.asarray(['left','right']),window_counts=shared['window_counts'],**values)
    row=dict(context=c,inner=i,seed=seed,status='PASS_PAIRED_INFERENCE',query_count=len(ids),frozen_probability_max_error=errors['frozen'],adapted_probability_max_error=errors['adapted'],feature_sha256=sha(path),shared_layer4_sha256=sha(layer4_path),frozen_checkpoint_sha256=sha(stage('frozen',seed,c,i)/'checkpoints/best.pt'),adapted_checkpoint_sha256=sha(stage('adapted',seed,c,i)/'checkpoints/best.pt'),normalization_file_sha256=sha(stage('frozen',seed,c,i)/'normalization.json'),official_checkpoint_sha256=sha(OFFICIAL),parameter_buffers_unchanged=True)
    rows.append(row);lineage.append(dict(**row,query_subject_ids=ids,fit_subject_ids=sp['train_subjects'],excluded_subject_ids=excluded,source_feature_path=str(path),shared_layer4_path=str(layer4_path),source_code_sha256=sha(Path(__file__)),preprocessing_sha256=sha(ROOT/'data/processed/pads_multi_activity/v2_l1_full_length/processing_config.json'),split_sha256=sha(stage('frozen',seed,c,i)/'split.json')))
    for aa,name in enumerate(ACTS):
     for ww in range(2):composition.append(dict(context=c,inner=i,seed=seed,activity=name,wrist=ww,query_count=len(ids),available_records=int(values['wrist_mask'][:,aa,ww].sum()),activity_present=int(values['activity_mask'][:,aa].sum()),minimum_true_length=int(values['activity_lengths'][:,aa].min()),maximum_true_length=int(values['activity_lengths'][:,aa].max())))
    pd.DataFrame(rows).to_csv(HERE/'analysis/inference_progress.csv',index=False)
    print(json.dumps(dict(stage='paired_inference',context=c,inner=i,seed=seed,status='PASS',max_error=errors)),flush=True)
    del frozen,adapted,last,ds,loader,frozen_ck,adapted_ck,values;gc.collect()
   del shared
 finally:manifest._load_array=original_load
 assert fingerprint(official.feature_extractor)==official_before and len(rows)==45
 pd.DataFrame(rows).to_csv(HERE/'analysis/paired_inference_identity.csv',index=False);pd.DataFrame(composition).to_csv(HERE/'analysis/activity_wrist_availability.csv',index=False)
 write_json(HERE/'restricted_manifests/extraction_lineage.json',dict(records=lineage,raw_reads=ext.io,only_validation_subjects_in_model_inputs=True,new_external_cache_reconstructed_subject_count=len(ext.reconstructed_subjects)))
 write_json(HERE/'tests/paired_forward_correctness.json',dict(cases=tests,official_encoder_unchanged=True,BN_buffers_unchanged=True,all_gradients_disabled=True,optimizer_steps=0,legacy_global_feature_arrays_decoded=False))
 write_json(HERE/'analysis/phase6a_extraction.json',dict(status='PASS_ALL45',pairs=45,model_validation_evaluations=90,new_deep_training=0,feature_arrays_preserve_11_activities_2_wrists=True,layer4_shared_by_design=True,max_probability_error=max(max(r['frozen_probability_max_error'],r['adapted_probability_max_error']) for r in rows),elapsed_seconds=time.perf_counter()-start))
 guard();print('PAIRED_INFERENCE_ALL45_PASS',flush=True)

if __name__=='__main__':main()
