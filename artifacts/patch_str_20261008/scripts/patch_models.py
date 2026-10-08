"""Zero-init per-wrist patch residual, original STR forward unchanged."""
import torch
from torch import nn
from common import require
from src.models import build_model as build_str
from patch_extractor import extract_patches
from patch_encoder import PatchEncoder

COUNTS={'P1':80340,'P2':80661}

class PatchBranch(nn.Module):
    def __init__(self,condition):
        super().__init__()
        require(condition in COUNTS,'Only registered conditions')
        self.condition=condition
        self.encoder=PatchEncoder()
        self.bottleneck=nn.Sequential(nn.Linear(64,16),nn.GELU())
        self.residual=nn.Linear(16,64)
        nn.init.zeros_(self.residual.weight);nn.init.zeros_(self.residual.bias)
        # Common encoder/head initialize before P2-only tensors, preserving matching.
        if condition=='P2':
            self.order_conv=nn.Conv1d(64,64,3,padding=1,groups=64,bias=True)
            self.scorer=nn.Linear(64,1)
            self.activation=nn.GELU()
    def aggregate(self,z,mask):
        z=z.masked_fill(~mask[...,None],0)
        if self.condition=='P1':
            attention=mask.to(z.dtype)/mask.sum(1,keepdim=True).clamp_min(1)
        else:
            z=self.activation(self.order_conv(z.transpose(1,2))).transpose(1,2)
            z=z.masked_fill(~mask[...,None],0)
            scores=self.scorer(z).squeeze(-1).masked_fill(~mask,-torch.finfo(z.dtype).max)
            attention=torch.softmax(scores,dim=1)*mask.to(z.dtype)
            attention=attention/attention.sum(1,keepdim=True).clamp_min(torch.finfo(z.dtype).tiny)
        return (z*attention[...,None]).sum(1),attention
    def from_embeddings(self,z,mask):
        summary,attention=self.aggregate(z,mask)
        residual=self.residual(self.bottleneck(summary)).masked_fill(~mask.any(1)[:,None],0)
        return dict(patch_residual=residual,patch_attention=attention,patch_summary=summary)
    def from_patches(self,patches,mask):
        return self.from_embeddings(self.encoder(patches,mask),mask)
    def forward(self,signal):
        patches,mask,starts=extract_patches(signal)
        return self.from_patches(patches,mask)

class PatchWrist(nn.Module):
    def __init__(self,original,condition):
        super().__init__();self.original=original;self.feature_dim=original.feature_dim
        require(self.feature_dim==64,'Original wrist changed')
        with torch.random.fork_rng(devices=[]):self.branch=PatchBranch(condition)
    def forward(self,signal):
        original=self.original(signal);patch=self.branch(signal)
        return {**original,**patch,'bag_embedding':original['bag_embedding']+patch['patch_residual']}

def build_patch(config,condition):
    model=build_str(config);require(sum(p.numel() for p in model.parameters())==75524,'STR architecture changed')
    model.wrist_encoder=PatchWrist(model.wrist_encoder,condition)
    require(sum(p.numel() for p in model.parameters())==COUNTS[condition],'Patch capacity changed')
    return model

def load_str_state(model,state):
    mapped={'wrist_encoder.original.'+k[len('wrist_encoder.'):] if k.startswith('wrist_encoder.') else k:v for k,v in state.items()}
    missing,extra=model.load_state_dict(mapped,strict=False)
    require(not extra and all(k.startswith('wrist_encoder.branch.') for k in missing),'Missing original tensors')
    require(len(missing)==len(model.wrist_encoder.branch.state_dict()),'Wrong branch keys')

def original_state(model):
    return {('wrist_encoder.'+k[len('wrist_encoder.original.'):] if k.startswith('wrist_encoder.original.') else k):v for k,v in model.state_dict().items() if not k.startswith('wrist_encoder.branch.')}
