"""Isolated Phase-3C model hooks. Production STR and Phase-3B code stay unchanged."""
import math,sys,types
from pathlib import Path
import numpy as np,torch
from torch import nn
from src.engine import nested_training as nt
P3B=Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928')
VENDOR=P3B/'vendor/ssl-wearables'
sys.path.insert(0,str(VENDOR))
import hubconf
from transfer_hooks import FrozenEmbeddingCache,ResidualSSLSubject
ROOT=Path('/home/zyt/deep_final/artifacts/phase3c_ssl_adaptation_20260930')

class Stage4Cache:
 def __init__(self):
  z=np.load(ROOT/'analysis/stage4_windows.npz',allow_pickle=False)
  self.ids=[str(x) for x in z['subject_ids']];self.index={s:i for i,s in enumerate(self.ids)}
  self.activities=[str(x) for x in z['activities']]
  self.stage4=torch.from_numpy(z['stage4'].copy());self.counts=torch.from_numpy(z['counts'].copy())
  self.mid=torch.from_numpy(z['mid'].copy())
  assert self.stage4.shape==(390,11,2,2,512,3) and self.counts.shape==(390,11,2)
  assert len(self.index)==390 and bool(torch.all((self.counts==1)|(self.counts==2)))
 def batch(self,ids,device):
  idx=[self.index[str(s)] for s in ids]
  return self.stage4[idx].to(device,non_blocking=True),self.counts[idx].to(device,non_blocking=True)

class CombinedCache:
 def __init__(self,stage4,final):
  assert stage4.ids==final.ids and stage4.activities==final.activities
  self.ids=stage4.ids;self.index=stage4.index;self.activities=stage4.activities
  self.feature=torch.cat([stage4.mid,final.pretrained],dim=-1)
  assert self.feature.shape==(390,11,2,1536)
 def batch(self,ids,device):
  return self.feature[[self.index[str(s)] for s in ids]].to(device,non_blocking=True)

class AdaptedResidual(nn.Module):
 def __init__(self,backbone):
  super().__init__()
  self.base=ResidualSSLSubject(backbone)
  official=hubconf.harnet10(pretrained=True,my_device='cpu',class_num=2)
  self.layer5=official.feature_extractor.layer5
  self.layer5.eval()
  self.num_classes=backbone.num_classes
  self.stage4_windows=None;self.window_counts=None
 def train(self,mode=True):
  super().train(mode)
  self.layer5.eval()  # fixed pretrained BN running statistics; affine remains trainable
  return self
 def forward(self,inputs,wrist_mask,activity_mask=None,activity_lengths=None):
  w=self.stage4_windows;c=self.window_counts
  if w is None or c is None or w.shape[:4]!=(inputs.shape[0],11,2,2):
   raise RuntimeError('Subject-matched stage4 windows missing')
  flat=w.reshape(-1,512,3)
  valid=(torch.arange(2,device=w.device)<c[...,None]).reshape(-1)
  ix=valid.nonzero(as_tuple=False).squeeze(1)
  encoded=self.layer5(flat.index_select(0,ix)).flatten(1)
  out=encoded.new_zeros((inputs.shape[0]*11*2,1024)).index_add(0,ix//2,encoded)
  self.base.ssl_features=(out/c.reshape(-1,1).to(out.dtype)).reshape(inputs.shape[0],11,2,1024)
  return self.base(inputs,wrist_mask,activity_mask,activity_lengths)

class MultiLevelResidual(nn.Module):
 def __init__(self,backbone):
  super().__init__()
  assert backbone.activity_count==11 and backbone.wrist_encoder.feature_dim==64 and backbone.bilateral_fusion_mode=='full'
  self.backbone=backbone;self.num_classes=backbone.num_classes
  self.norm=nn.LayerNorm(1536);self.wrist_projection=nn.Linear(1536,64)
  nn.init.zeros_(self.wrist_projection.weight);nn.init.zeros_(self.wrist_projection.bias)
  self.ssl_features=None
  original=backbone._encode_activities_with_wrists;wrapper=self
  def fused_encoding(_backbone,inputs,wrist_mask,activity_mask,activity_lengths):
   _,wrist=original(inputs,wrist_mask,activity_mask,activity_lengths)
   ssl=wrapper.ssl_features
   if ssl is None or ssl.shape[:3]!=wrist.shape[:3] or ssl.shape[-1]!=1536:
    raise RuntimeError('Subject-aligned middle+final SSL features missing')
   wm=wrist_mask.to(device=wrist.device,dtype=torch.bool)&activity_mask[...,None].to(torch.bool)
   added=wrapper.wrist_projection(wrapper.norm(ssl))*wm[...,None]
   fused_wrist=wrist+added
   batch,activities=fused_wrist.shape[:2]
   valid=activity_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
   selected_wrist=fused_wrist.reshape(-1,2,64).index_select(0,valid)
   selected_mask=wm.reshape(-1,2).index_select(0,valid)
   features,_=_backbone.wrist_fusion.fusion_features(selected_wrist,selected_mask)
   result=fused_wrist.new_zeros((batch*activities,258)).index_copy(0,valid,features)
   return result.reshape(batch,activities,258),fused_wrist
  backbone._encode_activities_with_wrists=types.MethodType(fused_encoding,backbone)
 def forward(self,inputs,wrist_mask,activity_mask=None,activity_lengths=None):
  return self.backbone(inputs,wrist_mask,activity_mask,activity_lengths)

class CacheLoader:
 def __init__(self,base,model,cache,variant):self.base=base;self.model=model;self.cache=cache;self.variant=variant
 def __iter__(self):
  for batch in self.base:
   device=next(self.model.parameters()).device
   if self.variant=='adapt':
    self.model.stage4_windows,self.model.window_counts=self.cache.batch(batch['subject_id'],device)
   elif self.variant=='frozen':self.model.ssl_features=self.cache.batch(batch['subject_id'],'pretrained',device)
   else:self.model.ssl_features=self.cache.batch(batch['subject_id'],device)
   yield batch
 def __len__(self):return len(self.base)

def activate(config,variant):
 """Patch only this process's inner runner, once per process."""
 if variant=='str':return None
 original_build=nt.build_model;original_train=nt.train_epoch;original_eval=nt.evaluate_epoch
 if variant=='adapt':cache=Stage4Cache()
 elif variant=='multi':cache=CombinedCache(Stage4Cache(),FrozenEmbeddingCache())
 elif variant=='frozen':cache=FrozenEmbeddingCache()
 else:raise ValueError(variant)
 assert cache.activities==[str(a) for a in config['data']['activities']]
 if variant=='adapt':nt.build_model=lambda cfg:AdaptedResidual(original_build(cfg))
 elif variant=='multi':nt.build_model=lambda cfg:MultiLevelResidual(original_build(cfg))
 else:nt.build_model=lambda cfg:ResidualSSLSubject(original_build(cfg))
 nt.train_epoch=lambda model,loader,*a,**k:original_train(model,CacheLoader(loader,model,cache,variant),*a,**k)
 nt.evaluate_epoch=lambda model,loader,*a,**k:original_eval(model,CacheLoader(loader,model,cache,variant),*a,**k)
 if variant=='adapt':
  def optimizer(model,cfg):
   tr=cfg['training'];lr=float(tr['learning_rate']);wd=float(tr['weight_decay']);betas=tuple(float(x) for x in tr.get('adam_betas',[.9,.999]))
   assert tr['optimizer'].lower()=='adamw' and abs(lr-2e-4)<1e-12
   return torch.optim.AdamW([{'params':model.base.parameters(),'lr':lr},{'params':model.layer5.parameters(),'lr':lr/10}],weight_decay=wd,betas=betas)
  def scheduler(opt,cfg):
   assert cfg['training']['scheduler']['type']=='cosine'
   epochs=int(cfg['training']['epochs']);eta=float(cfg['training']['scheduler']['minimum_lr']);lr=float(cfg['training']['learning_rate'])
   floor=eta/lr
   factor=lambda e:floor+(1-floor)*.5*(1+math.cos(math.pi*min(e,epochs)/epochs))
   return torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda=[factor,factor])
  nt.build_optimizer=optimizer;nt.build_scheduler=scheduler
 return cache
