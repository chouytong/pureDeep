"""State-compatible WSSL with no captured model/caches in method closures."""
import random
import numpy as np
import torch
from torch import nn
from src.models import build_model

class IndependentWSSL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone=build_model(config)
        assert self.backbone.activity_count==11 and self.backbone.wrist_encoder.feature_dim==64
        assert self.backbone.bilateral_fusion_mode=='full'
        self.num_classes=self.backbone.num_classes
        self.norm=nn.LayerNorm(1024)
        self.wrist_projection=nn.Linear(1024,64)
        nn.init.zeros_(self.wrist_projection.weight);nn.init.zeros_(self.wrist_projection.bias)
        self.ssl_features=None  # Instance-local bridge for unchanged foundation loader.
    def forward(self,inputs,wrist_mask,activity_mask=None,activity_lengths=None,ssl_features=None):
        b=self.backbone
        if inputs.ndim!=5 or inputs.shape[1]!=b.activity_count:raise ValueError('Expected subject activity input')
        batch=inputs.shape[0]
        if activity_mask is None:activity_mask=torch.ones((batch,b.activity_count),dtype=torch.bool,device=inputs.device)
        activity_mask=activity_mask.to(device=inputs.device,dtype=torch.bool)
        if activity_lengths is None:activity_lengths=activity_mask.long()*inputs.shape[-1]
        activity_lengths=activity_lengths.to(device=inputs.device,dtype=torch.long)
        if not torch.isfinite(inputs[activity_mask]).all():raise ValueError('Nonfinite valid input')
        _,wrist=b._encode_activities_with_wrists(inputs,wrist_mask,activity_mask,activity_lengths)
        ssl=self.ssl_features if ssl_features is None else ssl_features
        if ssl is None or ssl.shape[:3]!=wrist.shape[:3] or ssl.shape[-1]!=1024:raise RuntimeError('Subject-aligned frozen SSL input missing')
        wm=wrist_mask.to(device=wrist.device,dtype=torch.bool)&activity_mask[...,None]
        added=self.wrist_projection(self.norm(ssl))*wm[...,None]
        fused_wrist=wrist+added
        batch,activities=fused_wrist.shape[:2]
        valid=activity_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
        selected_wrist=fused_wrist.reshape(-1,2,64).index_select(0,valid)
        selected_mask=wm.reshape(-1,2).index_select(0,valid)
        features,_=b.wrist_fusion.fusion_features(selected_wrist,selected_mask)
        result=fused_wrist.new_zeros((batch*activities,258)).index_copy(0,valid,features)
        activity_features=result.reshape(batch,activities,258)
        aggregated=b.activity_aggregator(activity_features,activity_mask)
        base_logits=b.classifier(aggregated['subject_embedding']);logits=base_logits
        residual={};wrist_residual={}
        if b.structured_token_residual is not None:
            residual=b.structured_token_residual(activity_features,activity_mask)
            logits=logits+residual['structured_residual_logits']
        if b.wrist_structured_residual is not None:
            wrist_residual=b.wrist_structured_residual(fused_wrist,wrist_mask,activity_mask)
            logits=logits+wrist_residual['wrist_structured_residual_logits']
        return dict(logits=logits,base_logits=base_logits,probabilities=torch.softmax(logits,dim=-1),bag_embedding=aggregated['subject_embedding'],activity_mask=activity_mask,**residual,**wrist_residual,**aggregated)

def independent_copy(model, config):
    """Fresh instance/state storage, without consuming ordinary training RNG."""
    py=random.getstate();npstate=np.random.get_state();cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all()
    try:
        other=IndependentWSSL(config).to(next(model.parameters()).device)
        other.load_state_dict(model.state_dict(),strict=True)
        if model.ssl_features is not None:other.ssl_features=model.ssl_features.detach().clone()
        other.train(model.training)
        return other
    finally:
        random.setstate(py);np.random.set_state(npstate);torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(cuda)
