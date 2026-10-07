"""F1: near-capacity-matched residual from the current original wrist feature."""
from torch import nn


class CapacityControl(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(64,17),nn.GELU(),nn.Linear(17,16),nn.GELU())
        self.residual = nn.Linear(16,64)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, signal, original_embedding):
        feature = self.encode(original_embedding)
        return {'prior_embedding':feature,'prior_residual':self.residual(feature)}
