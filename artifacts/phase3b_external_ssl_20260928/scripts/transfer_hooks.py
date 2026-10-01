"""Isolated Phase-3B frozen-feature transfer models; no production edits."""
import types
from pathlib import Path
import numpy as np
import torch
from torch import nn
from src.engine import nested_training as nt

CACHE_PATH=Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928/analysis/ssl_embeddings_all.npz')

class FrozenEmbeddingCache:
    def __init__(self):
        z=np.load(CACHE_PATH,allow_pickle=False)
        self.ids=[str(s) for s in z['subject_ids']]
        self.index={s:i for i,s in enumerate(self.ids)}
        self.activities=[str(a) for a in z['activities']]
        self.pretrained=torch.from_numpy(z['pretrained'].copy())
        self.random=torch.from_numpy(z['random'].copy())
        self.counts=z['window_counts']
        assert len(self.ids)==390 and len(self.index)==390
        assert self.pretrained.shape==self.random.shape==(390,11,2,1024)
        assert self.counts.shape==(390,11,2) and np.all(self.counts>=1)
    def batch(self,subject_ids,kind,device):
        index=[self.index[str(s)] for s in subject_ids]
        x=self.pretrained if kind=='pretrained' else self.random
        return x[index].to(device=device,non_blocking=True)

class SSLOnlySubject(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes=2
        self.norm=nn.LayerNorm(1024)
        self.wrist_projection=nn.Linear(1024,64)
        self.activity_projection=nn.Linear(258,16)
        self.classifier=nn.Linear(176,2)
        self.ssl_features=None
    def forward(self,inputs,wrist_mask,activity_mask=None,activity_lengths=None):
        ssl=self.ssl_features
        if ssl is None or ssl.shape[:3]!=wrist_mask.shape or ssl.shape[-1]!=1024:
            raise RuntimeError('Subject-aligned frozen SSL features missing')
        if activity_mask is None:
            activity_mask=torch.ones(ssl.shape[:2],device=ssl.device,dtype=torch.bool)
        wm=wrist_mask.to(device=ssl.device,dtype=torch.bool)&activity_mask[...,None].to(torch.bool)
        w=self.wrist_projection(self.norm(ssl))*wm[...,None]
        left,right=w[:,:,0],w[:,:,1]
        count=wm.sum(dim=2,keepdim=True).clamp_min(1)
        mean=w.sum(dim=2)/count
        diff=(left-right).abs()*(wm[:,:,0]&wm[:,:,1])[...,None]
        fusion=torch.cat([left,right,mean,diff,wm.to(w.dtype)],dim=-1)
        ordered=torch.relu(self.activity_projection(fusion))*activity_mask[...,None]
        logits=self.classifier(ordered.reshape(len(ssl),-1))
        return {'logits':logits,'probabilities':torch.softmax(logits,dim=-1),'bag_embedding':ordered.reshape(len(ssl),-1),'activity_mask':activity_mask}

class ResidualSSLSubject(nn.Module):
    def __init__(self,backbone):
        super().__init__()
        assert backbone.activity_count==11 and backbone.wrist_encoder.feature_dim==64
        assert backbone.bilateral_fusion_mode=='full'
        self.backbone=backbone
        self.num_classes=backbone.num_classes
        self.norm=nn.LayerNorm(1024)
        self.wrist_projection=nn.Linear(1024,64)
        nn.init.zeros_(self.wrist_projection.weight)
        nn.init.zeros_(self.wrist_projection.bias)
        self.ssl_features=None
        original=backbone._encode_activities_with_wrists
        wrapper=self
        def fused_encoding(_backbone,inputs,wrist_mask,activity_mask,activity_lengths):
            _,wrist=original(inputs,wrist_mask,activity_mask,activity_lengths)
            ssl=wrapper.ssl_features
            if ssl is None or ssl.shape[:3]!=wrist.shape[:3] or ssl.shape[-1]!=1024:
                raise RuntimeError('Subject-aligned frozen SSL features missing')
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

class SSLLoader:
    def __init__(self,base,model,cache,kind):
        self.base=base;self.model=model;self.cache=cache;self.kind=kind
    def __iter__(self):
        for batch in self.base:
            ids=batch['subject_id']
            device=next(self.model.parameters()).device
            self.model.ssl_features=self.cache.batch(ids,self.kind,device)
            assert self.model.ssl_features.shape[0]==len(ids)
            yield batch
    def __len__(self):return len(self.base)

def activate(config,variant,cache):
    kind=variant['embedding_kind']
    assert kind in ('pretrained','random')
    assert cache.activities==[str(a) for a in config['data']['activities']]
    original_build=nt.build_model
    original_train=nt.train_epoch
    original_eval=nt.evaluate_epoch
    if variant['model_kind']=='ssl_only':
        assert kind=='pretrained'
        nt.build_model=lambda cfg:SSLOnlySubject()
    elif variant['model_kind']=='str_residual':
        nt.build_model=lambda cfg:ResidualSSLSubject(original_build(cfg))
    else:raise ValueError(variant['model_kind'])
    def train_wrapper(model,loader,*args,**kwargs):
        return original_train(model,SSLLoader(loader,model,cache,kind),*args,**kwargs)
    def eval_wrapper(model,loader,*args,**kwargs):
        return original_eval(model,SSLLoader(loader,model,cache,kind),*args,**kwargs)
    nt.train_epoch=train_wrapper
    nt.evaluate_epoch=eval_wrapper
