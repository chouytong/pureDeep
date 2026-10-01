"""Isolated Phase-3D adapter and gate hooks; production model and Phase-3B stay untouched."""
import math,types
from pathlib import Path
import torch
from torch import nn
from src.engine import nested_training as nt
from transfer_hooks import FrozenEmbeddingCache,ResidualSSLSubject

class AdapterResidual(ResidualSSLSubject):
 def __init__(self,backbone):
  super().__init__(backbone)
  self.adapter=nn.Sequential(nn.Linear(1024,64),nn.GELU(),nn.Linear(64,1024))
  nn.init.zeros_(self.adapter[-1].weight);nn.init.zeros_(self.adapter[-1].bias)
  self.raw_features=None
 def forward(self,inputs,wrist_mask,activity_mask=None,activity_lengths=None):
  raw=self.raw_features
  if raw is None or raw.shape[:3]!=wrist_mask.shape or raw.shape[-1]!=1024:
   raise RuntimeError('Subject-matched SSL features missing')
  self.ssl_features=raw+self.adapter(raw)
  return super().forward(inputs,wrist_mask,activity_mask,activity_lengths)

class GatedResidual(nn.Module):
 def __init__(self,backbone,activity_gate=False):
  super().__init__()
  assert backbone.activity_count==11 and backbone.wrist_encoder.feature_dim==64 and backbone.bilateral_fusion_mode=='full'
  self.backbone=backbone;self.num_classes=backbone.num_classes
  self.norm=nn.LayerNorm(1024);self.wrist_projection=nn.Linear(1024,64)
  nn.init.zeros_(self.wrist_projection.weight);nn.init.zeros_(self.wrist_projection.bias)
  self.gate_net=nn.Sequential(nn.Linear(128,32),nn.GELU(),nn.Linear(32,64))
  nn.init.zeros_(self.gate_net[-1].weight);nn.init.constant_(self.gate_net[-1].bias,math.log(9.0))
  self.activity_gate=activity_gate
  if activity_gate:self.activity_bias=nn.Parameter(torch.zeros(11))
  self.ssl_features=None
  original=backbone._encode_activities_with_wrists;wrapper=self
  def fused_encoding(_backbone,inputs,wrist_mask,activity_mask,activity_lengths):
   _,wrist=original(inputs,wrist_mask,activity_mask,activity_lengths)
   ssl=wrapper.ssl_features
   if ssl is None or ssl.shape[:3]!=wrist.shape[:3] or ssl.shape[-1]!=1024:
    raise RuntimeError('Subject-matched SSL features missing')
   wm=wrist_mask.to(device=wrist.device,dtype=torch.bool)&activity_mask[...,None].to(torch.bool)
   added=wrapper.wrist_projection(wrapper.norm(ssl))
   logits=wrapper.gate_net(torch.cat([wrist,added],dim=-1))
   if wrapper.activity_gate:logits=logits+wrapper.activity_bias[None,:,None,None]
   gate=torch.sigmoid(logits)
   fused_wrist=wrist+gate*added*wm[...,None]
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

class SSLLoader:
 def __init__(self,base,model,cache,variant):self.base=base;self.model=model;self.cache=cache;self.variant=variant
 def __iter__(self):
  for batch in self.base:
   device=next(self.model.parameters()).device
   feature=self.cache.batch(batch['subject_id'],'pretrained',device)
   if self.variant=='adapter':self.model.raw_features=feature
   else:self.model.ssl_features=feature
   yield batch
 def __len__(self):return len(self.base)

def activate(config,variant):
 assert variant in ('adapter','gate','activity_gate')
 cache=FrozenEmbeddingCache();assert cache.activities==[str(a) for a in config['data']['activities']]
 original_build=nt.build_model;original_train=nt.train_epoch;original_eval=nt.evaluate_epoch
 if variant=='adapter':nt.build_model=lambda cfg:AdapterResidual(original_build(cfg))
 else:nt.build_model=lambda cfg:GatedResidual(original_build(cfg),activity_gate=variant=='activity_gate')
 nt.train_epoch=lambda model,loader,*a,**k:original_train(model,SSLLoader(loader,model,cache,variant),*a,**k)
 nt.evaluate_epoch=lambda model,loader,*a,**k:original_eval(model,SSLLoader(loader,model,cache,variant),*a,**k)
 return cache
