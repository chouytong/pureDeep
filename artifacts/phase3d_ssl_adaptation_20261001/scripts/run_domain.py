#!/usr/bin/env python3
"""Conditional fold-local masked HarNet layer5 adaptation, then unchanged WSSL-STR."""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np,torch
from torch import nn
BASE=Path('/home/zyt/deep_final/artifacts/phase3d_ssl_adaptation_20261001')
ROOT=Path('/home/zyt/deep_final/foundation_validation')
P3B=Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928')
VENDOR=P3B/'vendor/ssl-wearables'
sys.path[:0]=[str(ROOT),str(P3B/'scripts'),str(BASE/'scripts'),str(VENDOR)]
import hubconf
from src.utils.config import load_config,require_pads_classification_config
from src.engine import nested_training as nt
from src.engine.nested_training import _load_frozen_split,_run_inner_fold
from src.utils.device import select_device
from src.utils.provenance import sha256_file
from transfer_hooks import ResidualSSLSubject,FrozenEmbeddingCache
from phase3d_hooks import SSLLoader
from restricted_manifests import restrict
from prepare_domain_raw import prepare

class LocalCache:
 def __init__(self,path):
  z=np.load(path,allow_pickle=False)
  self.ids=[str(s) for s in z['subject_ids']];self.index={s:i for i,s in enumerate(self.ids)}
  self.activities=[str(a) for a in z['activities']]
  self.pretrained=torch.from_numpy(z['features'].copy())
  assert self.pretrained.shape==(len(self.ids),11,2,1024) and len(self.index)==len(self.ids)
 def batch(self,ids,kind,device):
  assert kind=='pretrained'
  return self.pretrained[[self.index[str(s)] for s in ids]].to(device,non_blocking=True)

def feature_extractor():
 model=hubconf.harnet10(pretrained=True,my_device='cpu',class_num=2).feature_extractor
 first4=nn.Sequential(*list(model.children())[:4]).eval().cuda();first4.requires_grad_(False)
 layer5=model.layer5.eval().cuda()
 return first4,layer5

def encode(first4,layer5,x,key,nsubjects,batchsize=64):
 out=np.zeros((nsubjects,11,2,1024),np.float32);counts=np.zeros((nsubjects,11,2),np.int16)
 with torch.inference_mode():
  for lo in range(0,len(x),batchsize):
   hi=min(lo+batchsize,len(x));a=torch.from_numpy(x[lo:hi]).cuda()
   z=layer5(first4(a)).flatten(1).cpu().numpy()
   for v,k in zip(z,key[lo:hi]):
    ki=tuple(map(int,k));out[ki]+=v;counts[ki]+=1
 assert np.all(counts>=1) and np.all(counts<=2)
 out/=counts[...,None]
 return out,counts

def adapt(rawdir,seed,outer,inner,smoke=False):
 name='smoke_domain' if smoke else 'domain_adaptation'
 dest=BASE/name/f'seed{seed}'/f'outer_{outer}'/f'inner_{inner}';dest.mkdir(parents=True,exist_ok=True)
 statusfile=dest/'adapt_status.json';featurefile=dest/'features.npz'
 if statusfile.is_file() and featurefile.is_file() and not smoke:
  status=json.loads(statusfile.read_text())
  if status['status']=='complete' and hashlib.sha256(featurefile.read_bytes()).hexdigest()==status['feature_sha256']:
   return featurefile
 train=np.load(rawdir/'train.npz',allow_pickle=False);val=np.load(rawdir/'validation.npz',allow_pickle=False)
 xtr=train['x'];ktr=train['key'];ids_tr=[str(x) for x in train['subject_ids']]
 xval=val['x'];kval=val['key'];ids_val=[str(x) for x in val['subject_ids']]
 assert not set(ids_tr)&set(ids_val)
 torch.manual_seed(20261001+1000*outer+100*inner+seed);np.random.seed(20261001+1000*outer+100*inner+seed)
 torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 rng=np.random.default_rng(20261001+1000*outer+100*inner+seed)
 first4,layer5=feature_extractor()
 original={k:v.detach().cpu().clone() for k,v in layer5.state_dict().items()}
 if smoke:
  unadapted_train,_=encode(first4,layer5,xtr,ktr,len(ids_tr))
  old=FrozenEmbeddingCache();idx=[old.index[s] for s in ids_tr]
  delta=np.abs(unadapted_train-old.pretrained[idx].numpy())
  maxdiff=float(delta.max());mean_diff=float(delta.mean());p99_diff=float(np.quantile(delta,.99))
  assert maxdiff<.005 and mean_diff<1e-5 and p99_diff<1e-4,(maxdiff,mean_diff,p99_diff)
 else:maxdiff=None;mean_diff=None;p99_diff=None
 decoder=nn.Linear(1024,900).cuda()
 opt=torch.optim.AdamW([{'params':layer5.parameters(),'lr':1e-5},{'params':decoder.parameters(),'lr':1e-3}],weight_decay=1e-4)
 losses=[];epochs=1 if smoke else 5
 for epoch in range(epochs):
  order=rng.permutation(len(xtr));batchloss=[]
  for lo in range(0,len(order),64):
   if smoke and lo>=128:break
   ix=order[lo:lo+64];x=torch.from_numpy(xtr[ix]).cuda()
   block=np.zeros((len(ix),20),dtype=bool)
   selected=np.argsort(rng.random((len(ix),20)),axis=1)[:,:3]
   np.put_along_axis(block,selected,True,axis=1)
   mask=torch.from_numpy(np.repeat(block,15,axis=1)).cuda()[:,None,:].expand(-1,3,-1)
   masked=x.masked_fill(mask,0)
   with torch.no_grad():h=first4(masked)
   pred=decoder(layer5(h).flatten(1)).reshape(-1,3,300)
   loss=((pred-x).square()*mask).sum()/mask.sum()
   opt.zero_grad(set_to_none=True);loss.backward();opt.step();batchloss.append(float(loss.detach().cpu()))
  losses.append(float(np.mean(batchloss)));print(json.dumps({'phase':'masked_adapt','seed':seed,'outer':outer,'inner':inner,'epoch':epoch+1,'loss':losses[-1]}),flush=True)
 assert all(torch.equal(layer5.state_dict()[k].cpu(),v) for k,v in original.items() if 'running_' in k or 'num_batches_tracked' in k)
 param_delta=sum(float((layer5.state_dict()[k].cpu()-v).square().sum()) for k,v in original.items() if layer5.state_dict()[k].dtype.is_floating_point)**.5
 assert param_delta>0
 adapted_train,counts_tr=encode(first4,layer5,xtr,ktr,len(ids_tr))
 adapted_val,counts_val=encode(first4,layer5,xval,kval,len(ids_val))
 features=np.concatenate([adapted_train,adapted_val]);ids=ids_tr+ids_val
 np.savez_compressed(featurefile,features=features,subject_ids=np.asarray(ids),activities=train['activities'],train_subject_ids=np.asarray(ids_tr),validation_subject_ids=np.asarray(ids_val),train_counts=counts_tr,validation_counts=counts_val)
 torch.save({'layer5_state':layer5.state_dict(),'decoder_discarded':True,'train_subject_ids':ids_tr,'validation_subject_ids_excluded':ids_val,'losses':losses},dest/'adapted_layer5.pt')
 status={'status':'complete','seed':seed,'outer':outer,'inner':inner,'train_n':len(ids_tr),'validation_n':len(ids_val),'train_windows':len(xtr),'validation_windows_in_adaptation':0,'epochs':epochs,'last_loss':losses[-1],'layer5_parameter_delta_l2':param_delta,'bn_running_stats_unchanged':True,'unadapted_feature_maxdiff_vs_phase3b':maxdiff,'unadapted_feature_meanabs_vs_phase3b':mean_diff,'unadapted_feature_p99abs_vs_phase3b':p99_diff,'feature_sha256':hashlib.sha256(featurefile.read_bytes()).hexdigest(),'layer5_sha256':hashlib.sha256((dest/'adapted_layer5.pt').read_bytes()).hexdigest(),'outer_raw_access':False}
 statusfile.write_text(json.dumps(status,indent=2)+'\n')
 return featurefile

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--outer',type=int);ap.add_argument('--inner',type=int);ap.add_argument('--smoke',action='store_true');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
 assert args.seed in (42,43,44)
 manifest=json.loads((BASE/'manifest.json').read_text())
 assert sha256_file(P3B/'manifest.json')==manifest['phase3b_manifest_sha256']
 c=load_config(str(ROOT/'configs'/f'str01_seed{args.seed}.yaml'));require_pads_classification_config(c)
 c.setdefault('development',{})['phase3d_variant']='domain_masked'
 c['experiment']['name']=f'phase3d_domain_masked_seed{args.seed}'
 c['experiment']['output_root']=str(BASE/('smoke' if args.smoke else 'runs')/'domain_masked'/f'seed{args.seed}')
 split,_,_,sha=_load_frozen_split(c);assert sha==manifest['split_sha256']
 device=select_device('cuda')
 original_build=nt.build_model;original_train=nt.train_epoch;original_eval=nt.evaluate_epoch
 for outer in split['outer']:
  oi=int(outer['outer_fold'])
  if args.outer is not None and oi!=args.outer:continue
  for orig in outer['inner_folds']:
   ii=int(orig['inner_fold'])
   if args.inner is not None and ii!=args.inner:continue
   rawdir=prepare(oi,ii);featurefile=adapt(rawdir,args.seed,oi,ii,args.smoke)
   cache=LocalCache(featurefile);assert cache.activities==[str(a) for a in c['data']['activities']]
   assert set(cache.ids)==set(orig['train_subjects'])|set(orig['validation_subjects'])
   nt.build_model=lambda cfg:ResidualSSLSubject(original_build(cfg))
   nt.train_epoch=lambda model,loader,*a,**k:original_train(model,SSLLoader(loader,model,cache,'domain_masked'),*a,**k)
   nt.evaluate_epoch=lambda model,loader,*a,**k:original_eval(model,SSLLoader(loader,model,cache,'domain_masked'),*a,**k)
   cc=copy.deepcopy(c);restrict(cc,set(orig['train_subjects'])|set(orig['validation_subjects']),BASE/'restricted_manifests'/f'outer_{oi}'/f'inner_{ii}')
   oo=copy.deepcopy(outer);oo['test_subjects']=[]
   stage=Path(c['experiment']['output_root'])/f'outer_{oi}'/f'inner_{ii}'
   result=_run_inner_fold(cc,oo,copy.deepcopy(orig),stage,device,resume=args.resume,smoke=args.smoke)
   print(json.dumps({'variant':'domain_masked','seed':args.seed,'outer':oi,'inner':ii,'train_n':len(orig['train_subjects']),'best_epoch':result['best_epoch'],'ba':result['validation_metrics']['balanced_accuracy'],'auroc':result['validation_metrics']['macro_auroc'],'outer_test_loader_created':False}),flush=True)
 print(json.dumps({'status':'COMPLETE','variant':'domain_masked','seed':args.seed,'outer_test_evaluated':False}),flush=True)
if __name__=='__main__':main()
