"""Exact mainline classifiers/runner; explicit certified-data and train-only norm binding."""
from common_l2 import *
from data_l2 import data_manifest,configured_records,norm_binding
from metrics_l2 import pred_metrics
import argparse,os,copy,gc,io,resource,random,yaml,torch
from torch import nn
from torch.utils.data import DataLoader
from src.models import build_model
from src.models.wssl import ResidualSSLSubject
from src.datasets.subject_activity import SubjectActivityDataset,collate_subject_activities
from src.utils.seed import seed_everything
from src.engine.runner import train_epoch,evaluate_epoch
from src.engine.optim import build_optimizer,build_scheduler
from src.losses import build_loss

def cfg(seed):return yaml.safe_load((MAIN/f'configs/str01_seed{seed}.yaml').read_text())
def model_for(kind,config):return ResidualSSLSubject(build_model(config)) if kind=='wssl' else build_model(config)
def ids_ok(ids,allowed,exact=True):
 ids=list(map(str,ids));assert len(ids)==len(set(ids)) and set(ids)<=set(allowed)
 if exact:assert set(ids)==set(allowed)
 return ids

def external(allowed):
 result={};protected=load(PRIVATE/'manifests/protected_assets.json')['sha256']
 for sid in sorted(allowed):
  p=P6/'features/external_by_subject'/f'{sid}.npz';assert sha(p)==protected[str(p)]
  with np.load(p,allow_pickle=False) as z:
   assert str(z['subject_id'])==sid and z['activities'].tolist()==ACTS and z['wrist_order'].tolist()==SIDES
   result[sid]=z['layer5'].copy();assert result[sid].shape==(11,2,1024) and np.isfinite(result[sid]).all()
 return result
class BoundLoader:
 def __init__(self,loader,model,allowed,ext):self.loader=loader;self.model=model;self.allowed=allowed;self.ext=ext
 def __len__(self):return len(self.loader)
 def __iter__(self):
  for batch in self.loader:
   current=ids_ok(batch['subject_id'],self.allowed,False)
   if self.ext is not None:self.model.ssl_features=torch.from_numpy(np.stack([self.ext[s] for s in current])).cuda(non_blocking=True)
   yield batch

def smoke(kind):
 seed_everything(42,True);config=cfg(42);model=model_for(kind,config).cuda();assert sum(p.numel() for p in model.parameters())=={'wssl':143172,'str':75524}[kind]
 # Only synthetic values; no cohort or validation inputs are used for this optimizer update.
 x=torch.randn(4,11,2,6,2000,device='cuda');wm=torch.ones(4,11,2,dtype=torch.bool,device='cuda');am=torch.ones(4,11,dtype=torch.bool,device='cuda');lengths=torch.tensor([976,2000,976,2000],device='cuda')[:,None].expand(4,11).clone();y=torch.tensor([0,1,0,1],device='cuda');ssl=torch.randn(4,11,2,1024,device='cuda') if kind=='wssl' else None
 if ssl is not None:model.ssl_features=ssl
 for j in range(4):x[j,:,:,:,lengths[j,0]:]=0
 original=x.clone();opt=build_optimizer(model,config);model.train();out=model(x,wm,am,lengths);loss=nn.functional.cross_entropy(out['logits'],y);assert out['logits'].shape==(4,2) and torch.isfinite(loss);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();model.eval()
 with torch.no_grad():
  expected=model(x,wm,am,lengths)['logits'];changed=x.clone()
  for j in range(4):changed[j,:,:,:,lengths[j,0]:]=999
  assert torch.equal(expected,model(changed,wm,am,lengths)['logits']);assert torch.equal(x,original)
 buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0);fresh=model_for(kind,config).cuda().eval();fresh.load_state_dict(torch.load(buffer,weights_only=True,map_location='cuda'))
 if ssl is not None:fresh.ssl_features=ssl
 with torch.no_grad():assert torch.equal(expected,fresh(x,wm,am,lengths)['logits'])
 assert all(p.requires_grad for p in model.parameters()) and (ssl is None or not ssl.requires_grad)
 write(HERE/'tests'/f'{kind}_interface_smoke.json',dict(status='PASS',synthetic_only=True,smoke_optimizer_updates=1,formal_seed_reset=True,checkpoint_reload_bitwise=True,padding_invariance_bitwise=True,input_unchanged=True,logits_finite=True,all_classifier_parameters_trainable=True,external_HarNet_not_loaded_or_optimized=True,parameters=sum(p.numel() for p in model.parameters())))
 del model,fresh,opt;gc.collect();torch.cuda.empty_cache()

def raw_prediction(model,loader,allowed):
 model.eval();ids=[];ys=[];zs=[];ps=[]
 with torch.inference_mode():
  for b in loader:
   ids+=ids_ok(b['subject_id'],allowed,False);x=[b[k].cuda(non_blocking=True) for k in ['x','wrist_mask','activity_mask','activity_lengths']];o=model(*x);assert torch.isfinite(o['logits']).all();zs.append(o['logits'].cpu().numpy());ps.append(o['probabilities'].cpu().numpy());ys.extend(b['y'].tolist())
 ids_ok(ids,allowed);return np.asarray(ids),np.asarray(ys),np.concatenate(zs),np.concatenate(ps)

def main():
 ap=argparse.ArgumentParser()
 for n in ['model-type','data-version','manifest','normalization','split','seed','output-dir']:ap.add_argument('--'+n,required=True)
 a=ap.parse_args();kind=a.model_type;assert kind in ['wssl','str'];assert a.seed=='all' or int(a.seed) in [42,43,44];assert Path(a.output_dir).resolve().is_relative_to(PRIVATE.resolve())
 data=data_manifest(a.data_version,a.manifest);ni=load(HERE/'analysis/normalization_integrity.json');assert ni['status']=='PASS_15_TRAIN_ONLY_NORMALIZATIONS' and ni['index_sha256']==sha(a.normalization)
 normindex=load(a.normalization);assert normindex['version']==a.data_version and normindex['manifest_sha256']==sha(a.manifest);norms={(r['context'],r['inner']):r for r in normindex['normalizations']};allrecords=configured_records(data)
 if kind=='str':assert load(HERE/'analysis/wssl_training_integrity.json')['status']=='PASS_COMPLETE_45_RUNS' and (HERE/'reports/WSSL_RETRAIN_REPORT.md').exists()
 guard();torch.set_num_threads(4);seed_everything(42,True)
 frozen=HERE/'configs'/f'{kind.upper()}_TRAINING_FREEZE.json'
 if not frozen.exists():freeze(f'{kind.upper()}_TRAINING',[Path(__file__),HERE/'scripts/data_l2.py',HERE/'scripts/metrics_l2.py',HERE/'PROTOCOL.md',HERE/'PROTOCOL_AMENDMENT.md'],runs=45,formal_results_seen=False,HarNet_frozen_classifier_trainable=True)
 else:
  for p,h in load(frozen)['files'].items():assert sha(HERE/p)==h
 for c,i,sp,ex in splits():
  norm_binding(a.data_version,a.manifest,norms[c,i]['path'],c,i)
 try:data_manifest('legacy',a.manifest)
 except ValueError:pass
 else:raise AssertionError('Bad version accepted')
 try:norm_binding(a.data_version,a.manifest,baseline(kind,42,0,0)/'normalization.json',0,0)
 except ValueError:pass
 else:raise AssertionError('Old norm accepted')
 smoke(kind);rows=[];costs=[];lineage=[];begall=time.perf_counter();outroot=Path(a.output_dir);outroot.mkdir(parents=True,exist_ok=True)
 import src.datasets.manifest as loader_module
 original=loader_module._load_array
 try:
  for c,i,sp,ex in splits():
   if a.split!='all' and a.split!=f'{c}:{i}':continue
   normentry=norms[c,i];assert sha(normentry['path'])==normentry['sha256'];_,norm=norm_binding(a.data_version,a.manifest,normentry['path'],c,i);fit=set(map(str,sp['train_subjects']));query=set(map(str,sp['validation_subjects']));assert not fit&query and not (fit|query)&set(ex)
   for seed in [42,43,44] if a.seed=='all' else [int(a.seed)]:
    stage=outroot/f'seed{seed}_context{c}_inner{i}';receipt=stage/'complete.json'
    if receipt.exists():
     saved=load(receipt);assert saved['source_sha256']==sha(Path(__file__)) and saved['manifest_sha256']==sha(a.manifest) and saved['normalization_sha256']==normentry['sha256']
     for p,h in saved['outputs'].items():assert sha(p)==h
     rows.append(saved['metric']);costs.append(saved['cost']);lineage.append(saved['lineage']);continue
    if stage.exists():raise RuntimeError('BLOCKED_INCOMPLETE_RUN_PRESERVED '+str(stage))
    seed_everything(seed,True);config=cfg(seed);r=[v for v in allrecords if v.subject_id in fit|query];paths={v.left_path.resolve() for v in r}|{v.right_path.resolve() for v in r}
    def guarded(path,delimiter):
     if Path(path).resolve() not in paths:raise PermissionError('Excluded signal path rejected')
     return original(path,delimiter)
    loader_module._load_array=guarded
    try:guarded('/forbidden/excluded.npy',',')
    except PermissionError:pass
    else:raise AssertionError('I/O allowlist failed')
    ds=[]
    for name,allowed in [('train',fit),('validation',query)]:
     d=SubjectActivityDataset([v for v in r if v.subject_id in allowed],ACTS,config['data'],torch.tensor(norm['mean']),torch.tensor(norm['std']),name);ids_ok(d.subject_ids,allowed);assert d.activity_dropout==0 and d.rotation_degrees==0;ds.append(d)
    ext=external(fit|query) if kind=='wssl' else None
    gen=torch.Generator().manual_seed(seed);vg=torch.Generator().manual_seed(seed+1);tl=DataLoader(ds[0],batch_size=8,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True,generator=gen,collate_fn=collate_subject_activities);vl=DataLoader(ds[1],batch_size=8,shuffle=False,num_workers=4,pin_memory=True,persistent_workers=True,generator=vg,collate_fn=collate_subject_activities)
    model=model_for(kind,config).cuda();nparams=sum(p.numel() for p in model.parameters());assert nparams=={'wssl':143172,'str':75524}[kind] and all(p.requires_grad for p in model.parameters());bt=BoundLoader(tl,model,fit,ext);bv=BoundLoader(vl,model,query,ext)
    count=np.bincount([ds[0].labels[s] for s in ds[0].subject_ids],minlength=2);config=copy.deepcopy(config);config['loss']['class_weights']=(len(ds[0])/(2*count)).tolist();criterion=build_loss(config).cuda();opt=build_optimizer(model,config);scheduler=build_scheduler(opt,config);stage.mkdir();write(stage/'started.json',dict(model_type=kind,seed=seed,context=c,inner=i,source_sha256=sha(Path(__file__)),manifest_sha256=sha(a.manifest),normalization_sha256=normentry['sha256'],formal_run_index=len(rows)+1,original_initialization=True))
    best=-np.inf;wait=0;epochs=[];start=time.perf_counter();cpubeg=time.process_time();steps=0
    for epoch in range(50):
     tm=train_epoch(model,bt,criterion,opt,torch.device('cuda'),mixed_precision=False,gradient_clip_norm=5.);vm,_=evaluate_epoch(model,bv,criterion,torch.device('cuda'),mixed_precision=False);steps+=len(tl);improved=vm['balanced_accuracy']>best
     if improved:
      best=vm['balanced_accuracy'];bestepoch=epoch+1;wait=0;torch.save(dict(model_state=model.state_dict(),epoch=bestepoch,seed=seed,context=c,inner=i,version=a.data_version,manifest_sha256=sha(a.manifest),normalization_sha256=normentry['sha256'],source_sha256=sha(Path(__file__)),HarNet_not_in_optimizer=True),stage/'best.pt')
     else:wait+=1
     scheduler.step();epochs.append(dict(epoch=epoch+1,train_loss=tm['loss'],validation_BA=vm['balanced_accuracy'],learning_rate=tm['learning_rate'],improved=improved,optimizer_steps=steps));write(stage/'epochs.json',epochs)
     if (epoch+1)%5==0:print(json.dumps(dict(model=kind,context=c,inner=i,seed=seed,epoch=epoch+1,best_BA=best,elapsed_seconds=time.perf_counter()-begall)),flush=True)
     if wait>=12:break
    ck=torch.load(stage/'best.pt',map_location='cuda',weights_only=False);assert ck['manifest_sha256']==sha(a.manifest) and ck['normalization_sha256']==normentry['sha256'];model.load_state_dict(ck['model_state']);ids,y,z,p=raw_prediction(model,bv,query);metric=dict(condition=f'{kind.upper()}-CERTIFIED-L1',seed=seed,context=c,inner=i,**pred_metrics(y,p,z));pred=stage/'predictions.npz';np.savez_compressed(pred,subject_ids=ids,target=y,logits=z,probabilities=p)
    cost=dict(model=kind,seed=seed,context=c,inner=i,classifier_parameters=nparams,trainable_parameters=nparams,frozen_external_parameters=10457408 if kind=='wssl' else 0,epochs=len(epochs),best_epoch=bestepoch,optimizer_steps=steps,training_validation_wall_seconds=time.perf_counter()-start,parent_CPU_seconds=time.process_time()-cpubeg,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated())
    record=dict(seed=seed,context=c,inner=i,model=kind,fit_subject_ids=sorted(fit),query_subject_ids=sorted(query),excluded_subject_ids=ex,checkpoint=str(stage/'best.pt'),checkpoint_sha256=sha(stage/'best.pt'),prediction=str(pred),prediction_sha256=sha(pred),normalization=normentry['path'],normalization_sha256=normentry['sha256'],manifest_sha256=sha(a.manifest),cache_paths=[str(P6/'features/external_by_subject'/f'{s}.npz') for s in sorted(fit|query)] if ext is not None else [],raw_signal_role_boundary='fit trainloader; query validationloader; no excluded loader')
    write(receipt,dict(source_sha256=sha(Path(__file__)),manifest_sha256=sha(a.manifest),normalization_sha256=normentry['sha256'],metric=metric,cost=cost,lineage=record,outputs={str(stage/'best.pt'):sha(stage/'best.pt'),str(pred):sha(pred),str(stage/'epochs.json'):sha(stage/'epochs.json')}));rows.append(metric);costs.append(cost);lineage.append(record);pd.DataFrame(rows).to_csv(HERE/'analysis'/f'{kind}_training_progress.csv',index=False);print(json.dumps(dict(model=kind,completed=len(rows),epochs=len(epochs),BA=metric['ba'],elapsed_seconds=time.perf_counter()-begall)),flush=True)
    del model,opt,criterion,scheduler,bt,bv,tl,vl,ds,ext,ck;gc.collect();torch.cuda.empty_cache()
 finally:loader_module._load_array=original
 frame=pd.DataFrame(rows);frame.to_csv(HERE/'analysis'/f'{kind}_metrics.csv',index=False);pd.DataFrame(costs).to_csv(HERE/'analysis'/f'{kind}_training_costs.csv',index=False);write(PRIVATE/'manifests'/f'{kind}_lineage.json',dict(records=lineage))
 if a.split=='all' and a.seed=='all':
  assert len(rows)==45;report=HERE/'reports'/f'{kind.upper()}_RETRAIN_REPORT.md';report.write_text(f'# {kind.upper()} certified L1 retraining\n\n45/45 runs completed. Original seed initialization, exact mainline classifier and training runner, balanced CE, original strict validation BA selection, max50/patience12. HarNet frozen; classifier trained. No performance-dependent experiment stopping. New train-only norms; private per-run predictions and checkpoints.\n\n'+table(frame.groupby('condition')[METRICS].mean().reset_index())+'\n\nThis report is archived before the next stage. Final paired legacy comparison is in the final analysis.\n')
  write(HERE/'analysis'/f'{kind}_training_integrity.json',dict(status='PASS_COMPLETE_45_RUNS',runs=45,smoke='PASS_SYNTHETIC_ONLY',formal_optimizer_updates=sum(c['optimizer_steps'] for c in costs),data_version=a.data_version,manifest_sha256=sha(a.manifest),normalization_index_sha256=sha(a.normalization),report_sha256=sha(report),classifier_parameters={'wssl':143172,'str':75524}[kind],AMP=False,TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32,cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),workers=4,epochs_max=50,patience=12,failed_formal_runs=0,elapsed_seconds=time.perf_counter()-begall))
 guard();print(kind.upper()+'_COMPLETE')
if __name__=='__main__':main()
