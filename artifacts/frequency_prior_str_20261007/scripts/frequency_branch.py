"""F2: five small learned time-domain band encoders and a 16-D wrist residual."""
import torch
from torch import nn
from filter_bank import FixedBandBank


class FrequencyPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.bank = FixedBandBank()
        self.band_encoders = nn.ModuleList([nn.Sequential(
            nn.Conv1d(6,6,15,padding=7,groups=6,bias=False),
            nn.Conv1d(6,8,1,bias=True),nn.GELU(),nn.AdaptiveAvgPool1d(1)) for _ in range(5)])
        self.project = nn.Sequential(nn.Linear(40,16),nn.GELU())
        self.residual = nn.Linear(16,64)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, signal, original_embedding):
        bands = self.bank(signal)
        features = torch.stack([encoder(bands[:,i]).squeeze(-1)
                                for i,encoder in enumerate(self.band_encoders)],dim=1)
        feature = self.project(features.flatten(1))
        return {'band_embeddings':features,'prior_embedding':feature,
                'prior_residual':self.residual(feature)}
