"""AI-WSSL-STR: exact Phase8 interaction math, independently registered.

The core below is copied verbatim from Phase8 ActivityInteraction (class rename only).
Adapters add input validation and a frozen signal-to-token path, not network layers.
Checkpoint state keys remain exactly base.*, projection.*, identity.*, attention.*,
norm1.*, norm2.*, ffn.*, output.*; all existing checkpoints strict-load unchanged.
"""
from pathlib import Path
import json
import torch
from torch import nn
from .pure_deep import build_pure_deep_subject
from .wssl import ResidualSSLSubject

class _ActivityInteractionCore(nn.Module):
 def __init__(self,base):
  super().__init__();self.base=base
  for p in self.base.parameters():p.requires_grad_(False)
  self.base.eval();self.projection=nn.Linear(258,64);self.identity=nn.Embedding(11,64);self.attention=nn.MultiheadAttention(64,4,dropout=.1,batch_first=True)
  self.norm1=nn.LayerNorm(64);self.norm2=nn.LayerNorm(64);self.drop=nn.Dropout(.1);self.ffn=nn.Sequential(nn.Linear(64,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,64));self.output=nn.Linear(64,258);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
 def train(self,mode=True):super().train(mode);self.base.eval();return self
 def forward(self,h,mask,attention_off=False):
  if h.ndim!=3 or h.shape[1:]!=(11,258) or mask.shape!=h.shape[:2] or not mask.any(1).all():raise ValueError('Expected11 ordered valid activity tokens')
  h=h.masked_fill(~mask[...,None],0);b=self.base.backbone;a=b.activity_aggregator(h,mask);s=b.structured_token_residual(h,mask);baseline_logits=b.classifier(a['subject_embedding'])+s['structured_residual_logits']
  token=self.projection(h)+self.identity(torch.arange(11,device=h.device))[None];mixed,weights=self.attention(token,token,token,key_padding_mask=~mask,need_weights=True,average_attn_weights=False)
  if attention_off:mixed=torch.zeros_like(mixed)
  token=self.norm1(token+self.drop(mixed));token=self.norm2(token+self.drop(self.ffn(token)));pooled=(token*mask[...,None]).sum(1)/mask.sum(1,keepdim=True);delta=self.output(pooled)
  logits=baseline_logits+torch.nn.functional.linear(delta,b.classifier[1].weight,bias=None)
  return dict(logits=logits,baseline_logits=baseline_logits,activity_attention=weights,subject_embedding=a['subject_embedding']+delta)

FIXED_INTERACTION = dict(token_dim=258, projection_dim=64, activities=11,
                         heads=4, layers=1, ffn_hidden_dim=128, dropout=0.1,
                         output_dim=258, output_init="zeros", residual_bias=False)


def validate_ai_structure(config):
    path = Path(__file__).resolve().parents[2] / "models" / "ai_wssl_str_structure.json"
    expected = json.loads(path.read_text())
    model = dict(config["model"])
    model.pop("name", None)
    declared = model.pop("ai_wssl_str", FIXED_INTERACTION)
    if declared != FIXED_INTERACTION or model != expected["backbone_model"]:
        raise ValueError("AI-WSSL-STR requires the registered unchanged Phase8 architecture")
    for key, value in expected["data_structure"].items():
        if config["data"].get(key) != value:
            raise ValueError(f"AI-WSSL-STR data structure mismatch: {key}")


class AIWSSLSTR(_ActivityInteractionCore):
    """Experimental Candidate. Freeze all WSSL; train only the original AI module."""
    model_id = "ai_wssl_str"
    model_status = "Experimental Candidate"
    architecture_version = "phase8_activity_interaction_v1"

    def __init__(self, config):
        validate_ai_structure(config)
        super().__init__(ResidualSSLSubject(build_pure_deep_subject(config)))
        self.num_classes = self.base.num_classes
        self.activities = tuple(config["data"]["activities"])

    def encode_tokens(self, inputs, wrist_mask, activity_mask=None,
                      activity_lengths=None, *, ssl_features):
        if inputs.ndim != 5 or tuple(inputs.shape[1:4]) != (11, 2, 6):
            raise ValueError("Expected signals [B,11,2,6,T]")
        batch, length = inputs.shape[0], inputs.shape[-1]
        if wrist_mask.shape != (batch, 11, 2):
            raise ValueError("Expected wrist_mask [B,11,2]")
        mask = (torch.ones((batch, 11), dtype=torch.bool, device=inputs.device)
                if activity_mask is None else activity_mask.to(inputs.device, dtype=torch.bool))
        if mask.shape != (batch, 11) or not mask.any(1).all():
            raise ValueError("Every subject needs at least one valid activity")
        lengths = (mask.long() * length if activity_lengths is None
                   else activity_lengths.to(inputs.device, dtype=torch.long))
        if lengths.shape != (batch, 11) or ((lengths[mask] <= 0) | (lengths[mask] > length)).any():
            raise ValueError("Invalid true activity lengths")
        if not torch.isfinite(inputs[mask]).all():
            raise ValueError("Nonfinite valid activity signal")
        if ssl_features is None or ssl_features.shape != (batch, 11, 2, 1024):
            raise ValueError("Explicit subject-aligned frozen HarNet [B,11,2,1024] required")
        if ssl_features.device != inputs.device or not torch.isfinite(ssl_features[mask]).all():
            raise ValueError("HarNet feature device/nonfinite mismatch")
        self.base.eval()
        previous = self.base.ssl_features
        try:
            self.base.ssl_features = ssl_features.detach()
            with torch.no_grad():
                tokens, _ = self.base.backbone._encode_activities_with_wrists(
                    inputs, wrist_mask.to(inputs.device, dtype=torch.bool), mask, lengths)
        finally:
            self.base.ssl_features = previous
        return tokens, mask

    def forward_tokens(self, bilateral_tokens, activity_mask, attention_off=False):
        if activity_mask.dtype != torch.bool or activity_mask.device != bilateral_tokens.device:
            raise ValueError("Token mask must be boolean on the token device")
        if bilateral_tokens.ndim == 3 and activity_mask.shape == bilateral_tokens.shape[:2]:
            if not torch.isfinite(bilateral_tokens[activity_mask]).all():
                raise ValueError("Nonfinite valid token")
        result = super().forward(bilateral_tokens, activity_mask, attention_off)
        result["probabilities"] = torch.softmax(result["logits"], dim=-1)
        return result

    def forward(self, inputs, wrist_mask=None, activity_mask=None,
                activity_lengths=None, *, ssl_features=None, attention_off=False):
        if inputs.ndim == 3:
            # Historical token call model(h, mask) retains its exact meaning.
            mask = activity_mask if activity_mask is not None else wrist_mask
            if mask is None:
                raise ValueError("Explicit token activity_mask required")
            return self.forward_tokens(inputs, mask, attention_off)
        if wrist_mask is None:
            raise ValueError("Explicit signal wrist_mask required")
        tokens, mask = self.encode_tokens(inputs, wrist_mask, activity_mask,
                                          activity_lengths, ssl_features=ssl_features)
        return self.forward_tokens(tokens, mask, attention_off)
