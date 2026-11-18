"""Serial, source-bound Phase10 pilots/full matrices with no automatic rescue search."""
from common10 import *
from models10 import PartialHarNet,DemographicResidual,parameter_groups,prefix4,fingerprint
import argparse,gc,yaml,torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from src.datasets.subject_activity import SubjectActivityDataset,collate_subject_activities
from src.engine.runner import train_epoch,evaluate_epoch
from src.losses import build_loss
from src.utils.seed import seed_everything
from train_l2 import BoundLoader,raw_prediction

KINDS={'FT-L1':'B','FT-L2':'B','DEMO':'C','DD':'D'}
def config(seed):
 c=copy.deepcopy(cfg(seed));c['data']['version']=VERSION
 c['data']['root']=str(P9/'restricted_manifests/processed_signals'/VERSION)
 c['data'].pop('activity_manifests',None)
 c['evaluation']['run_test_after_training']=False
 return c

class TraceAdamW(torch.optim.AdamW):
 def __init__(self,groups):
  super().__init__(groups,lr=2e-4,weight_decay=1e-4,betas=(.9,.999));self.trace=[]
 def step(self,closure=None):
  for group in self.param_groups:
   gradients=[p.grad.detach() for p in group['params'] if p.grad is not None]
   assert all(torch.isfinite(g).all() for g in gradients)
   total=sum(float(torch.sum(g.double()**2)) for g in gradients)
   self.trace.append(dict(group=group['name'],gradient_L2=math.sqrt(total),learning_rate=group['lr']))
  return super().step(closure)

class FTLoader:
 def __init__(self,loader,model,allowed,prefix):self.loader=loader;self.model=model;self.allowed=allowed;self.prefix=prefix
 def __len__(self):return len(self.loader)
 def __iter__(self):
  for b in self.loader:
   ids=ids_ok(b['subject_id'],self.allowed,False)
   self.model.prefix=torch.from_numpy(np.stack([self.prefix[s][0] for s in ids])).cuda(non_blocking=True)
   self.model.counts=torch.from_numpy(np.stack([self.prefix[s][1] for s in ids])).cuda(non_blocking=True)
   yield b

def fixed_scope(c,i):
 sp,excluded=role(c,i);fit=set(map(str,sp['train_subjects']));query=set(map(str,sp['validation_subjects']))
 assert not fit&query and not (fit|query)&excluded
 data,n,entry=norm(c,i);records=[r for r in configured_records(data) if r.subject_id in fit|query]
 # File content identities are checked before any DataLoader can read a signal.
 expected={r['output_path']:r['output_sha256'] for r in data['records'] if r['subject_id'] in fit|query}
 paths={v.left_path.resolve() for v in records}|{v.right_path.resolve() for v in records}
 assert len(paths)==len(fit|query)*22
 for path in paths:assert sha(path)==expected[str(path)]
 import src.datasets.manifest as lm
 original=lm._load_array
 def guarded(path,delimiter):
  p=Path(path).resolve()
  if p not in paths:raise PermissionError('Outside current fit/query allowlist')
  return original(path,delimiter)
 lm._load_array=guarded
 return sp,excluded,fit,query,records,n,entry,expected,original

def loaders(seed,records,n,cfg_,fit,query):
 ds=[]
 for mode,allow in [('train',fit),('validation',query)]:
  d=SubjectActivityDataset([r for r in records if r.subject_id in allow],ACTS,cfg_['data'],torch.tensor(n['mean']),torch.tensor(n['std']),mode)
  ids_ok(d.subject_ids,allow);assert d.activity_dropout==0 and d.rotation_degrees==0
  ds.append(d)
 tl=DataLoader(ds[0],batch_size=8,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True,generator=torch.Generator().manual_seed(seed),collate_fn=collate_subject_activities)
 vl=DataLoader(ds[1],batch_size=8,shuffle=False,num_workers=4,pin_memory=True,persistent_workers=True,generator=torch.Generator().manual_seed(seed+1),collate_fn=collate_subject_activities)
 return ds,tl,vl

def demo_export(seed,c,i,cfg_,fit,query,records,n,entry):
 dest=PRIVATE/'demo_exports'/f'seed{seed}_context{c}_inner{i}'
 if dest.exists():
  receipt=load(dest/'complete.json');assert receipt['normalization_sha256']==entry['sha256']
  for path,digest in receipt['sha256'].items():assert sha(path)==digest
  return dest
 dest.mkdir(parents=True);r=line('wssl',seed,c,i)
 assert sha(r['checkpoint'])==r['checkpoint_sha256']
 base=model_for('wssl',cfg_).cuda().eval();ck=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
 assert ck['version']==VERSION and ck['normalization_sha256']==entry['sha256']
 base.load_state_dict(ck['model_state'],strict=True)
 for p in base.parameters():p.requires_grad_(False)
 initial=fingerprint(base);ext=external(fit|query);files={};meta=metadata()
 pre=load(PRIVATE/'demographics'/f'context{c}_inner{i}/preprocessor.json')
 # Only the already verified train-only four-column preprocessing is reused.
 write(dest/'demo_preprocessor.json',{k:pre[k] for k in ['fit_subject_ids','query_subject_ids','continuous_median','gender_mode','mean','scale','features']},new=True)
 error=0.;times=[]
 for name,allowed in [('train',fit),('validation',query)]:
  d=SubjectActivityDataset([r for r in records if r.subject_id in allowed],ACTS,cfg_['data'],torch.tensor(n['mean']),torch.tensor(n['std']),'validation')
  loader=DataLoader(d,batch_size=8,shuffle=False,num_workers=4,pin_memory=True,persistent_workers=True,collate_fn=collate_subject_activities)
  bound=BoundLoader(loader,base,allowed,ext);start=time.perf_counter();ids,y,z,p=raw_prediction(base,bound,allowed);times.append(time.perf_counter()-start)
  demo=(demographic_matrix(meta,ids)-np.asarray(pre['mean']))/np.asarray(pre['scale'])
  assert np.isfinite(demo).all() and np.array_equal(y,np.asarray([meta[s]['label'] for s in ids]))
  if name=='validation':
   oi,oy,op,oz=certified('wssl',seed,c,i);ix={s:j for j,s in enumerate(oi)};order=[ix[s] for s in ids]
   assert np.array_equal(y,oy[order]);error=float(np.max(abs(p-op[order])));assert error<=1e-6
  path=dest/(name+'.npz');np.savez_compressed(path,subject_ids=ids,target=y,baseline_logits=z,baseline_probabilities=p,demographics=demo.astype(np.float32));files[str(path)]=sha(path)
  del loader,bound,d;gc.collect()
 assert fingerprint(base)==initial
 write(dest/'complete.json',dict(normalization_sha256=entry['sha256'],baseline_checkpoint_sha256=r['checkpoint_sha256'],sha256=files,
                                validation_probability_max_error=error,train_validation_forward_seconds=sum(times),frozen_base_unchanged=True,
                                source_sha256=sha(Path(__file__)),no_legacy_supervised_embedding=True),new=True)
 del base,ck,ext;gc.collect();torch.cuda.empty_cache()
 return dest

def evaluate_demo(model,loader,criterion):
 model.eval();ys=[];zs=[];ps=[];total=0.;seen=0
 with torch.no_grad():
  for x,z,y in loader:
   x,z,y=x.cuda(),z.cuda(),y.cuda();o=model(x,z);loss=criterion(o,y)['loss'];total+=float(loss)*len(y);seen+=len(y)
   ys.append(y.cpu().numpy());zs.append(o['logits'].cpu().numpy());ps.append(o['probabilities'].cpu().numpy())
 return dict(loss=total/seen,**pred_metrics(np.concatenate(ys),np.concatenate(ps),np.concatenate(zs))),np.concatenate(ys),np.concatenate(zs),np.concatenate(ps)

def run_one(kind,seed,c,i):
 phase=KINDS[kind];assert load(HERE/'analysis/phase10a_integrity.json')['status']=='PASS_PHASE10A_COMPLETE'
 if phase=='C':assert (HERE/'reports/PHASE10B_UNFREEZING_REPORT.md').exists() and load(HERE/'analysis/phase10a_integrity.json')['demographic_fields_trusted']
 if phase=='D':assert (HERE/'reports/PHASE10C_DEMOGRAPHIC_REPORT.md').exists() and load(HERE/'analysis/phase10d_start_gate.json')['status']=='ELIGIBLE_FIXED_1_10_PILOT'
 checkfreeze('PHASE10'+phase)
 stage=PRIVATE/'runs'/kind/f'seed{seed}_context{c}_inner{i}'
 if stage.exists():
  if (stage/'complete.json').exists():
   d=load(stage/'complete.json');assert d['source_sha256']==sha(Path(__file__))
   for p,h in d['outputs'].items():assert sha(p)==h
   return d
  raise RuntimeError('Incomplete run preserved; investigate instead of overwriting '+str(stage))
 import src.datasets.manifest as lm
 seed_everything(seed,True);torch.set_num_threads(4)
 sp,excluded,fit,query,records,n,entry,expected,original=fixed_scope(c,i)
 config_=config(seed);count=np.bincount([r.label for r in records if r.activity==ACTS[0] and r.subject_id in fit],minlength=2)
 assert int(count.sum())==len(fit)
 weights=len(fit)/(2*count)
 if kind=='DD':
  weights=weights*np.array([1.,1.1]);weights=weights/(count@weights/len(fit))
 config_['loss']['class_weights']=weights.tolist();start=time.perf_counter();cpu0=time.process_time()
 try:
  if kind=='DEMO':
   export=demo_export(seed,c,i,config_,fit,query,records,n,entry)
   seed_everything(seed,True);base=model_for('wssl',config_);r=line('wssl',seed,c,i);base.load_state_dict(torch.load(r['checkpoint'],map_location='cpu',weights_only=False)['model_state'])
   model=DemographicResidual(base).cuda();base_before=fingerprint(model.base);packages={}
   for name,allow in [('train',fit),('validation',query)]:
    with np.load(export/(name+'.npz'),allow_pickle=False) as z:packages[name]={k:z[k].copy() for k in z.files}
    ids_ok(packages[name]['subject_ids'].astype(str),allow)
   ds=[TensorDataset(torch.from_numpy(packages[name]['demographics']),torch.from_numpy(packages[name]['baseline_logits']),torch.from_numpy(packages[name]['target']).long()) for name in ['train','validation']]
   tl=DataLoader(ds[0],batch_size=8,shuffle=True,num_workers=4,persistent_workers=True,generator=torch.Generator().manual_seed(seed));vl=DataLoader(ds[1],batch_size=8,shuffle=False,num_workers=4,persistent_workers=True,generator=torch.Generator().manual_seed(seed+1))
   with torch.no_grad():
    x,z,y=next(iter(vl));assert torch.equal(model(x.cuda(),z.cuda())['logits'].cpu(),z)
   bridge=None
  else:
   ds,tl,vl=loaders(seed,records,n,config_,fit,query)
   if kind.startswith('FT-'):
    assert kind=='FT-L1','FT-L2 requires a separately justified raw-prefix3 export; not silently substituted'
    model=PartialHarNet(config_,depth=1).cuda();prefix=prefix4(fit|query)
    bt=FTLoader(tl,model,fit,prefix);bv=FTLoader(vl,model,query,prefix)
   else:
    model=model_for('wssl',config_).cuda();ext=external(fit|query);bt=BoundLoader(tl,model,fit,ext);bv=BoundLoader(vl,model,query,ext)
   bridge=(bt,bv)
  groups=parameter_groups(model,kind);opt=TraceAdamW(groups);criterion=build_loss(config_).cuda()
  factor=lambda epoch:.005+.995*.5*(1+math.cos(math.pi*min(epoch,50)/50))
  scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda=[factor]*len(groups))
  initial={k:p.detach().cpu().clone() for k,p in model.named_parameters() if p.requires_grad}
  frozen_before={k:v for k,v in fingerprint(model).items() if k not in initial}
  params=sum(p.numel() for p in model.parameters());trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
  total=params+(10457408 if kind in ['DEMO','DD'] else 0);frozen=total-trainable
  stage.mkdir(parents=True);torch.cuda.reset_peak_memory_stats()
  source_commit=subprocess_head()
  public_cfg=dict(candidate=kind,seed=seed,context=c,inner=i,data_version=VERSION,
                  data_manifest_sha256=load(P9/'restricted_manifests/manifests/data_ready.json')['manifest_sha256'],normalization_sha256=entry['sha256'],
                  train_class_counts=count.tolist(),train_class_weights=weights.tolist(),optimizer='AdamW',weight_decay=1e-4,betas=[.9,.999],batch_size=8,workers=4,
                  epochs=50,patience=12,selection='strict_validation_BA_first_best',threshold=.5,tie='PD',AMP=False,
                  groups=[dict(name=g['name'],learning_rate=g['lr'],parameter_count=sum(p.numel() for p in g['params'])) for g in groups],
                  source_commit=source_commit,source_sha256=sha(Path(__file__)),models_source_sha256=sha(HERE/'scripts/models10.py'),BN_running_and_affine_frozen=True)
  configpath=HERE/'configs'/f'{kind}_seed{seed}_context{c}_inner{i}.json';write(configpath,public_cfg,new=True)
  write(stage/'started.json',dict(**public_cfg,fit_subject_ids=sorted(fit),query_subject_ids=sorted(query),excluded_subject_ids=sorted(excluded)),new=True)
  best=-math.inf;wait=0;epochs=[];bestepoch=None
  for epoch in range(50):
   if kind=='DEMO':
    model.train();seen=0;loss_sum=0.
    for x,z,y in tl:
     x,z,y=x.cuda(),z.cuda(),y.cuda();opt.zero_grad(set_to_none=True);o=model(x,z);loss=criterion(o,y)['loss'];assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.);opt.step();seen+=len(y);loss_sum+=float(loss.detach())*len(y)
    tm=dict(loss=loss_sum/seen);vm,_,_,_=evaluate_demo(model,vl,criterion);ba=vm['ba']
   else:
    tm=train_epoch(model,bt,criterion,opt,torch.device('cuda'),mixed_precision=False,gradient_clip_norm=5.)
    vm,_=evaluate_epoch(model,bv,criterion,torch.device('cuda'),mixed_precision=False);ba=vm['balanced_accuracy']
   assert math.isfinite(tm['loss']) and math.isfinite(vm['loss'])
   improved=ba>best
   if improved:
    best=ba;bestepoch=epoch+1;wait=0
   else:wait+=1
   epochs.append(dict(epoch=epoch+1,train_weighted_CE=tm['loss'],validation_weighted_CE=vm['loss'],validation_BA=ba,improved=improved,
                      learning_rates={g['name']:g['lr'] for g in opt.param_groups}))
   scheduler.step();write(stage/'epochs.json',epochs)
   if improved:
    torch.save(dict(model_state=model.state_dict(),optimizer_state=opt.state_dict(),scheduler_state=scheduler.state_dict(),epoch=bestepoch,
                    seed=seed,context=c,inner=i,version=VERSION,normalization_sha256=entry['sha256'],source_sha256=sha(Path(__file__)),
                    config_sha256=sha(configpath),source_commit=source_commit,manifest_sha256=public_cfg['data_manifest_sha256'],candidate=kind,
                    torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state_all(),
                    train_loader_rng_state=tl.generator.get_state(),scheduler_position='after_completed_epoch_step'),stage/'best.pt')
   if (epoch+1)%5==0:print(json.dumps(dict(candidate=kind,seed=seed,context=c,inner=i,epoch=epoch+1,best_BA=best,wall_seconds=time.perf_counter()-start)),flush=True)
   if wait>=12:break
  saved=torch.load(stage/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(saved['model_state']);model.eval();torch.cuda.synchronize();inference_start=time.perf_counter()
  if kind=='DEMO':
   _,y,z,p=evaluate_demo(model,vl,criterion);ids=packages['validation']['subject_ids'].astype(str);assert fingerprint(model.base)==base_before
  else:ids,y,z,p=raw_prediction(model,bv,query)
  torch.cuda.synchronize();inference_seconds=time.perf_counter()-inference_start;ids_ok(ids,query)
  metric=dict(condition=kind,seed=seed,context=c,inner=i,source_commit=source_commit,**pred_metrics(y,p,z))
  np.savez_compressed(stage/'predictions.npz',subject_ids=ids,target=y,logits=z,probabilities=p)
  after=fingerprint(model)
  assert all(after[k]==v for k,v in frozen_before.items()),'Frozen parameter/buffer mutation'
  paramrows=[]
  for group in opt.param_groups:
   keys=[k for k,p in model.named_parameters() if any(p is q for q in group['params'])]
   norm0=math.sqrt(sum(float(torch.sum(initial[k].double()**2)) for k in keys));norm1=math.sqrt(sum(float(torch.sum(dict(model.named_parameters())[k].detach().double()**2)) for k in keys))
   delta=math.sqrt(sum(float(torch.sum((dict(model.named_parameters())[k].detach().cpu().double()-initial[k].double())**2)) for k in keys))
   grads=[x['gradient_L2'] for x in opt.trace if x['group']==group['name']]
   paramrows.append(dict(condition=kind,seed=seed,context=c,inner=i,group=group['name'],parameters=sum(p.numel() for p in group['params']),initial_norm=norm0,best_norm=norm1,update_L2=delta,mean_gradient_L2=np.mean(grads),max_gradient_L2=max(grads),all_finite=True))
   assert delta>0 and max(grads)>0
  cost=dict(condition=kind,seed=seed,context=c,inner=i,total_parameters=total,trainable_parameters=trainable,frozen_parameters=frozen,
            epochs=len(epochs),best_epoch=bestepoch,optimizer_steps=len(opt.trace)//len(groups),wall_seconds=time.perf_counter()-start,
            parent_CPU_seconds=time.process_time()-cpu0,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),cached_path_validation_inference_seconds=inference_seconds)
  outputpaths=[stage/'best.pt',stage/'predictions.npz',stage/'epochs.json',configpath]
  receipt=dict(metric=metric,cost=cost,parameter_groups=paramrows,source_sha256=sha(Path(__file__)),models_source_sha256=sha(HERE/'scripts/models10.py'),
               outputs={str(p):sha(p) for p in outputpaths},normalization_sha256=entry['sha256'],config_sha256=sha(configpath),checkpoint_sha256=sha(stage/'best.pt'),
               frozen_state_unchanged=True,optimizer_state_saved=True,source_commit=source_commit,actual_fit_subject_ids=sorted(fit),actual_query_subject_ids=sorted(query))
  write(stage/'complete.json',receipt,new=True);print(json.dumps(dict(candidate=kind,seed=seed,context=c,inner=i,status='COMPLETE',metric=metric,cost=cost)),flush=True)
  return receipt
 finally:
  lm._load_array=original
  gc.collect();torch.cuda.empty_cache()

def subprocess_head():
 import subprocess
 return subprocess.check_output(['git','-C','/home/zyt/pureDeep-github-upload','rev-parse','HEAD'],text=True).strip()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--candidate',choices=list(KINDS),required=True);ap.add_argument('--scope',choices=['pilot','full'],required=True);a=ap.parse_args()
 if a.scope=='full':assert load(HERE/'analysis'/f'{a.candidate}_pilot_gate.json')['expand']
 targets=[(c,i) for c,i,sp,ex in splits()] if a.scope=='full' else [(0,0)]
 rows=[];costs=[];params=[]
 for c,i in targets:
  for seed in [42,43,44]:
   result=run_one(a.candidate,seed,c,i);rows.append(result['metric']);costs.append(result['cost']);params.extend(result['parameter_groups'])
   gc.collect();torch.cuda.empty_cache()
   # Progress is provisional and preserved in the private run receipts; only
   # completed stage aggregates are published, never partial-seed conclusions.
 phase=KINDS[a.candidate];suffix='pilot' if a.scope=='pilot' else 'full'
 csvout(f'{a.candidate}_{suffix}_metrics.csv',rows);csvout(f'{a.candidate}_{suffix}_costs.csv',costs);csvout(f'{a.candidate}_{suffix}_parameter_groups.csv',params)
 guard();checkfreeze('PHASE10'+phase)
 print(json.dumps(dict(candidate=a.candidate,scope=a.scope,complete=len(rows),status='ALL_REQUESTED_RUNS_COMPLETE')),flush=True)

if __name__=='__main__':main()
