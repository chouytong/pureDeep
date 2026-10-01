from __future__ import annotations

import torch
from torch import nn


class ChannelAttention1d(nn.Module):
    """结合全局平均池化与最大池化的一维通道注意力。"""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if reduction <= 0:
            raise ValueError("reduction must be positive")
        reduced = max(1, channels // reduction)
        self.shared_mlp = nn.Sequential(
            nn.Conv1d(channels, reduced, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduced, channels, kernel_size=1, bias=True),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3:
            raise ValueError(f"Expected a 3D tensor, got {tuple(inputs.shape)}")
        average = inputs.mean(dim=-1, keepdim=True)
        maximum = inputs.amax(dim=-1, keepdim=True)
        weights = torch.sigmoid(
            self.shared_mlp(average) + self.shared_mlp(maximum)
        )
        return inputs * weights, weights


class TanhAttentionScorer(nn.Module):
    """使用两层感知机和 tanh 激活计算每个时间实例的注意力分数。"""

    def __init__(self, input_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, instances: torch.Tensor) -> torch.Tensor:
        return self.network(instances).squeeze(-1)


class GatedAttentionScorer(nn.Module):
    """可配置的门控注意力打分器。"""

    def __init__(self, input_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.tanh_branch = nn.Linear(input_dim, attention_dim)
        self.sigmoid_branch = nn.Linear(input_dim, attention_dim)
        self.output = nn.Linear(attention_dim, 1)

    def forward(self, instances: torch.Tensor) -> torch.Tensor:
        gated = torch.tanh(self.tanh_branch(instances)) * torch.sigmoid(
            self.sigmoid_branch(instances)
        )
        return self.output(gated).squeeze(-1)
