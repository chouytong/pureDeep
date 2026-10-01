"""Isolated Phase-2 train-only sampling/augmentation/loss hooks; no production code edits."""
from pathlib import Path
import json,math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from src.datasets.subject_activity import collate_subject_activities
from src.losses.mfam_loss import ClassificationLoss
from src.engine import nested_training as nt

class AugDataset(Dataset):
 def __init__(self,base,kind,strategy):
  self.base=base;self.kind=kind;self.strategy=strategy
  self.class_indices={c:[i for i,s in enumerate(base.subject_ids) if base.labels[s]==c] for c in (0,1)}
 def __len__(self):return len(self.base)
 def __getitem__(self,i):
  item=dict(self.base[i]);y=int(item['y']);p=.5 if self.strategy=='equal' else (1.0 if y==1 else 0.0)
  changed=bool(torch.rand(())<p)
  if not changed:item['phase2_augmented']=torch.tensor(0);return item
  x=item['x'].clone();lengths=item['activity_lengths'];wrist=item['wrist_mask']
  if self.kind=='jitter':
   for a in range(x.shape[0]):
    n=int(lengths[a]);
    if n<=0:continue
    for w in range(2):
     if wrist[a,w]:x[a,w,:,:n]+=torch.randn_like(x[a,w,:,:n])*.01
  elif self.kind=='scaling':
   for a in range(x.shape[0]):
    n=int(lengths[a]);
    if n<=0:continue
    for w in range(2):
     if not wrist[a,w]:continue
     factor=(1+torch.randn(())*.03).clamp(.94,1.06);mean=self.base.mean[a,w,:,0];std=self.base.std[a,w,:,0]
     raw=x[a,w,:,:n]*std[:,None]+mean[:,None]
     x[a,w,:,:n]=(raw*factor-mean[:,None])/std[:,None]
  elif self.kind=='temporal':
   for a in range(x.shape[0]):
    n=int(lengths[a]);
    if n<=1:continue
    t=torch.arange(n,dtype=x.dtype);disp=(torch.rand(())*2-1)*.01*(n-1);source=(t+disp*torch.sin(math.pi*t/(n-1))).clamp(0,n-1);left=source.floor().long();right=(left+1).clamp(max=n-1);frac=(source-left).view(1,1,n)
    x[a,:,:,:n]=x[a,:,:,left]*(1-frac)+x[a,:,:,right]*frac
  elif self.kind=='mixup':
   candidates=self.class_indices[y]
   if len(candidates)<2:changed=False
   else:
    j=candidates[int(torch.randint(len(candidates),(1,)))];other=self.base[j]
    if j==i or not torch.equal(other['activity_mask'],item['activity_mask']) or not torch.equal(other['wrist_mask'],item['wrist_mask']) or not torch.equal(other['activity_lengths'],item['activity_lengths']):changed=False
    else:
     lam=float(.9+.1*torch.rand(()));x=lam*x+(1-lam)*other['x']
  else:raise ValueError(self.kind)
  item['x']=x;item['phase2_augmented']=torch.tensor(int(changed));return item

def activate(config,variant):
 kind=variant.get('augmentation');strategy=variant.get('strategy','equal');sampling=variant.get('sampling','original')
 if kind is None and sampling=='original' and variant.get('loss_kind') is None:return
 if kind is not None:assert kind in ('jitter','scaling','temporal','mixup') and strategy in ('equal','dd_only')
 assert sampling in ('original','balanced')
 original_loader=nt._loader;original_train_epoch=nt.train_epoch;original_build_loss=nt.build_loss
 def phase2_loader(dataset,cfg,*,train,seed_offset):
  if not train:return original_loader(dataset,cfg,train=train,seed_offset=seed_offset)
  if kind is None and sampling=='original':return original_loader(dataset,cfg,train=train,seed_offset=seed_offset)
  wrapped=AugDataset(dataset,kind,strategy) if kind else dataset
  gen=torch.Generator().manual_seed(int(cfg['experiment']['seed'])+seed_offset)
  sampler=None
  if sampling=='balanced':
   labels=[int(dataset.labels[s]) for s in dataset.subject_ids];counts=[labels.count(0),labels.count(1)];weights=torch.tensor([1/counts[y] for y in labels],dtype=torch.double)
   sampler=WeightedRandomSampler(weights,num_samples=len(labels),replacement=True,generator=gen)
  return DataLoader(wrapped,batch_size=int(cfg['training']['batch_size']),shuffle=sampler is None,sampler=sampler,num_workers=int(cfg['data'].get('num_workers',0)),pin_memory=bool(cfg['data'].get('pin_memory',False)),persistent_workers=int(cfg['data'].get('num_workers',0))>0,generator=gen,collate_fn=collate_subject_activities)
 class CountingLoader:
  def __init__(self,base):self.base=base;self.counts={'pd_exposures':0,'dd_exposures':0,'augmented_pd_exposures':0,'augmented_dd_exposures':0}
  def __iter__(self):
   for batch in self.base:
    y=batch['y'];aug=batch.get('phase2_augmented',torch.zeros_like(y))
    for c,name in ((0,'pd'),(1,'dd')):
     idx=y==c;self.counts[f'{name}_exposures']+=int(idx.sum());self.counts[f'augmented_{name}_exposures']+=int(aug[idx].sum())
    yield batch
  def __len__(self):return len(self.base)
 def phase2_train_epoch(model,loader,criterion,optimizer,device,**kwargs):
  if kind is None and sampling=='original':return original_train_epoch(model,loader,criterion,optimizer,device,**kwargs)
  counted=CountingLoader(loader);result=original_train_epoch(model,counted,criterion,optimizer,device,**kwargs);c=counted.counts;result['phase2_exposure']=c;result['phase2_augmented_original_ratio']=sum(c[k] for k in ('augmented_pd_exposures','augmented_dd_exposures'))/max(1,sum(c[k] for k in ('pd_exposures','dd_exposures'))-sum(c[k] for k in ('augmented_pd_exposures','augmented_dd_exposures')));return result
 nt._loader=phase2_loader;nt.train_epoch=phase2_train_epoch
 loss_kind=variant.get('loss_kind')
 if loss_kind:
  class FocalLoss(nn.Module):
   def __init__(self,weights):super().__init__();self.register_buffer('weights',torch.tensor(weights,dtype=torch.float32))
   def forward(self,outputs,targets):
    logits=outputs['logits'];p=F.softmax(logits,dim=-1).gather(1,targets[:,None]).squeeze(1);ce=F.cross_entropy(logits,targets,weight=self.weights,reduction='none');v=((1-p)**2*ce).mean();return {'loss':v,'classification_loss':v}
  def custom_build(cfg):
   if loss_kind=='focal':return FocalLoss(cfg['loss']['class_weights'])
   counts=cfg['loss']['class_weight_train_counts'];n=torch.tensor([counts['PD'],counts['DD']],dtype=torch.float64)
   if loss_kind=='class_balanced':w=(1-.99)/(1-torch.pow(torch.tensor(.99),n))
   elif loss_kind=='sqrt_weighted':w=torch.sqrt(n.sum()/(2*n))
   else:raise ValueError(loss_kind)
   w=w/w.mean();return ClassificationLoss(label_smoothing=0,class_weights=w.tolist())
  nt.build_loss=custom_build
