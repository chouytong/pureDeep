"""Independent per-wrist residual; unchanged original subject forward and activity identity."""
import torch
from torch import nn
from common import require
from src.models import build_model as build_str
from capacity_control import CapacityControl
from frequency_branch import FrequencyPrior


class PriorWrist(nn.Module):
    def __init__(self, original, condition):
        super().__init__()
        self.original = original
        self.feature_dim = original.feature_dim
        require(self.feature_dim == 64, 'Unexpected STR wrist dimension')
        # Only the new CPU-initialized branch consumes this temporary RNG stream.
        # Base parameters and the post-factory RNG stream remain exactly original.
        with torch.random.fork_rng(devices=[]):
            if condition == 'F1':self.branch = CapacityControl()
            elif condition == 'F2':self.branch = FrequencyPrior()
            else:raise ValueError('Only registered F1/F2 conditions are allowed')

    def forward(self, signal):
        original = self.original(signal)
        prior = self.branch(signal,original['bag_embedding'])
        return {**original,**prior,'bag_embedding':original['bag_embedding']+prior['prior_residual']}


def build_prior(config, condition):
    model = build_str(config)
    require(sum(p.numel() for p in model.parameters()) == 75524, 'Base model differs')
    model.wrist_encoder = PriorWrist(model.wrist_encoder, condition)
    require(sum(p.numel() for p in model.parameters()) == {'F1':78005,'F2':77998}[condition], 'Prior capacity differs')
    return model


def load_str_state(model, state):
    renamed = {'wrist_encoder.original.'+k[len('wrist_encoder.'):] if k.startswith('wrist_encoder.') else k:v
               for k,v in state.items()}
    missing,extra = model.load_state_dict(renamed,strict=False)
    require(not extra and all(k.startswith('wrist_encoder.branch.') for k in missing), 'Wrong frozen STR state')
    require(len(missing) == len(model.wrist_encoder.branch.state_dict()), 'Missing original state tensor')


def original_state(model):
    return {('wrist_encoder.'+k[len('wrist_encoder.original.'):] if k.startswith('wrist_encoder.original.') else k):v
            for k,v in model.state_dict().items() if not k.startswith('wrist_encoder.branch.')}
