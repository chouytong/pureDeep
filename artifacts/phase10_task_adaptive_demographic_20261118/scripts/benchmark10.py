"""Timing-only online inference on a fixed validation subset, never a new fit."""
from common10 import *
from train10 import config
from models10 import PartialHarNet,DemographicResidual,official_encoder,fingerprint
from src.datasets.subject_activity import SubjectActivityDataset,collate_subject_activities
from src.utils.seed import seed_everything
from scipy.signal import resample_poly
import torch,gc

def main():
 seed_everything(42,True);torch.set_num_threads(4)
 for phase in ['A','B','C','D']:checkfreeze('PHASE10'+phase)
 assert (HERE/'reports/PHASE10D_DD_LOSS_REPORT.md').exists()
 sp,excluded=role(0,0);chosen=sorted(map(str,sp['validation_subjects']))[:8];allow=set(chosen)
 assert len(chosen)==8 and not allow&excluded
 data,n,entry=norm(0,0);rec=[r for r in configured_records(data) if r.subject_id in allow]
 assert len(rec)==88
 expected={r['output_path']:r['output_sha256'] for r in data['records'] if r['subject_id'] in allow}
 for r in rec:
  for p in [r.left_path,r.right_path]:assert sha(p)==expected[str(p)]
 cfg_=config(42);ds=SubjectActivityDataset(rec,ACTS,cfg_['data'],torch.tensor(n['mean']),torch.tensor(n['std']),'validation')
 assert set(ds.subject_ids)==allow
 rawpaths={}
 for r in rec:
  for w in range(2):
   p=ROOT/'data/raw/pads/movement/timeseries'/f'{r.subject_id}_{r.record_name}_{["LeftWrist","RightWrist"][w]}.txt'
   rawpaths[r.subject_id,ACTS.index(r.activity),w]=p
 rawhash={str(p):sha(p) for p in rawpaths.values()}
 def inputs():
  # All I/O is restricted to eight preselected validation subjects. No labels
  # are used for subset selection or timing. No global cache is decoded.
  batch=collate_subject_activities([ds[j] for j in range(len(ds))]);ids=list(batch['subject_id']);assert set(ids)==allow
  windows=np.zeros((8,11,2,2,3,300),np.float32);counts=np.zeros((8,11,2),np.int16)
  for b,s in enumerate(ids):
   for a in range(11):
    for w in range(2):
     raw=np.loadtxt(rawpaths[s,a,w],delimiter=',',dtype=np.float32);assert raw.shape[1]==7
     acc=np.clip(raw[48:,1:4],-3,3);assert np.isfinite(acc).all()
     if len(acc)<1000:
      before=(1000-len(acc))//2;segments=[np.pad(acc,((before,1000-len(acc)-before),(0,0)),mode='edge')]
     else:segments=[acc[:1000],acc[-1000:]]
     counts[b,a,w]=len(segments)
     for k,seg in enumerate(segments):windows[b,a,w,k]=resample_poly(seg,3,10,axis=0).T.astype(np.float32)
  return batch,windows,counts
 official=official_encoder().eval().cuda();official_before=fingerprint(official)
 first4=torch.nn.Sequential(*list(official.children())[:4])
 pre=load(PRIVATE/'demographics/context0_inner0/preprocessor.json');meta=metadata()
 rows=[];private=[]
 for kind in ['WSSL-CERTIFIED-L1','FT-L1','DEMO','DD']:
  if kind=='WSSL-CERTIFIED-L1':
   path=Path(line('wssl',42,0,0)['checkpoint']);ck=torch.load(path,map_location='cpu',weights_only=False);model=model_for('wssl',cfg_);model.load_state_dict(ck['model_state'])
  else:
   path=PRIVATE/'runs'/kind/'seed42_context0_inner0/best.pt';ck=torch.load(path,map_location='cpu',weights_only=False)
   model=PartialHarNet(cfg_,1) if kind=='FT-L1' else DemographicResidual(model_for('wssl',cfg_)) if kind=='DEMO' else model_for('wssl',cfg_)
   model.load_state_dict(ck['model_state'])
  model=model.eval().cuda()
  for p in model.parameters():p.requires_grad_(False)
  prior=fingerprint(model);timings=[];errors=[];peaks=[]
  for repeat in range(4):
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
   batch,windows,counts=inputs();cpu_seconds=time.perf_counter()-start;ids=batch['subject_id']
   valid=(np.arange(2)<counts[...,None]).reshape(-1);ix=np.flatnonzero(valid)
   tensor=torch.from_numpy(windows.reshape(-1,3,300)[valid]).cuda()
   xs=[batch[k].cuda() for k in ['x','wrist_mask','activity_mask','activity_lengths']]
   with torch.inference_mode():
    h4=torch.cat([first4(x) for x in tensor.split(64)],0)
    if kind=='FT-L1':
     padded=h4.new_zeros((8*44,512,3));padded.index_copy_(0,torch.from_numpy(ix).cuda(),h4)
     model.prefix=padded.reshape(8,11,2,2,512,3);model.counts=torch.from_numpy(counts).cuda();out=model(*xs)
    else:
     z=torch.cat([official.layer5(x).flatten(1) for x in h4.split(64)],0)
     ssl=z.new_zeros((8*22,1024)).index_add_(0,torch.from_numpy(ix//2).cuda(),z)
     ssl=(ssl/torch.from_numpy(counts.reshape(-1,1)).cuda()).reshape(8,11,2,1024)
     base=model.base if kind=='DEMO' else model;base.ssl_features=ssl;out=base(*xs)
     if kind=='DEMO':
      demo=(demographic_matrix(meta,ids)-np.asarray(pre['mean']))/np.asarray(pre['scale'])
      out=model(torch.tensor(demo,dtype=torch.float32,device='cuda'),out['logits'])
    prob=out['probabilities'].cpu().numpy();assert np.isfinite(prob).all()
   torch.cuda.synchronize();seconds=time.perf_counter()-start
   if kind=='WSSL-CERTIFIED-L1':oi,oy,op,oz=certified('wssl',42,0,0)
   else:
    with np.load(PRIVATE/'runs'/kind/'seed42_context0_inner0/predictions.npz',allow_pickle=False) as z:oi=z['subject_ids'].astype(str);op=z['probabilities']
   loc={s:j for j,s in enumerate(oi)};error=float(abs(prob-op[[loc[s] for s in ids]]).max())
   if repeat:
    timings.append(seconds);errors.append(error);peaks.append(torch.cuda.max_memory_allocated())
    rows.append(dict(condition=kind,seed=42,context=0,inner=0,repeat=repeat,subjects=8,warmup_repeats=1,
     timing_scope='ONLINE_RAW_ACC_HARNET_PLUS_CERTIFIED_STR_IO_TRANSFORM_GPU_FORWARD',batch_size=8,seconds=seconds,seconds_per_subject=seconds/8,
     CPU_IO_and_transform_seconds=cpu_seconds,peak_GPU_allocated_bytes=peaks[-1],canonical_cached_probability_max_abs_error=error,
     arithmetic_identity_required=False,performance_metrics_replaced=False))
  assert fingerprint(model)==prior
  private.append(dict(condition=kind,subject_ids=chosen,checkpoint=str(path),checkpoint_sha256=sha(path),seconds=timings,probability_max_errors=errors))
  del model,ck;gc.collect();torch.cuda.empty_cache()
 assert fingerprint(official)==official_before
 for p,h in rawhash.items():assert sha(p)==h
 csvout('inference_benchmark.csv',rows)
 write(PRIVATE/'inference_benchmark_lineage.json',dict(records=private,raw_file_hashes=rawhash,normalization_sha256=entry['sha256'],
  source_sha256=sha(Path(__file__)),selection='first eight lexicographically sorted validation subjects, fixed independent of labels/results'),new=True)
 write(HERE/'tests/inference_benchmark_integrity.json',dict(status='PASS_TIMING_ONLY',new_fits=0,optimizer_updates=0,only_eight_validation_subjects=True,
  frozen_states_unchanged=True,canonical_predictions_not_replaced=True,maximum_probability_drift=max(x['canonical_cached_probability_max_abs_error'] for x in rows),
  timing_repetitions=3,warmup=1,protected_files_verified=guard()),new=True)
 print('PASS timing-only online inference',flush=True)

if __name__=='__main__':main()
