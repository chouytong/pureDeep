from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .attention import ChannelAttention1d


def _same_padding(kernel_size: int, dilation: int) -> int:
    effective = dilation * (kernel_size - 1) + 1
    if effective % 2 == 0:
        raise ValueError(
            "Only odd effective kernels are supported for exact same padding"
        )
    return (effective - 1) // 2


class ConvBnSilu(nn.Sequential):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=_same_padding(kernel_size, dilation),
                bias=bias,
            ),
            nn.BatchNorm1d(output_channels),
            nn.SiLU(inplace=True),
        )


class MultiScaleChannelAttentionEncoder(nn.Module):
    """通过不同扩张率卷积和通道注意力编码多尺度时序特征。"""

    def __init__(
        self,
        input_channels: int,
        branch_channels: int,
        hidden_dim: int,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4),
        channel_reduction: int = 8,
        conv_bias: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not dilations:
            raise ValueError("At least one dilation is required")
        self.branches = nn.ModuleList(
            [
                ConvBnSilu(
                    input_channels,
                    branch_channels,
                    kernel_size,
                    dilation,
                    conv_bias,
                )
                for dilation in dilations
            ]
        )
        self.fusion = ConvBnSilu(
            branch_channels * len(dilations),
            hidden_dim,
            kernel_size=1,
            dilation=1,
            bias=conv_bias,
        )
        self.channel_attention = ChannelAttention1d(
            hidden_dim, reduction=channel_reduction
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.hidden_dim = hidden_dim

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        branch_outputs = [branch(inputs) for branch in self.branches]
        time_lengths = {output.shape[-1] for output in branch_outputs}
        if len(time_lengths) != 1:
            raise RuntimeError("Multi-scale branches produced different time lengths")
        fused = self.fusion(torch.cat(branch_outputs, dim=1))
        attended, channel_weights = self.channel_attention(fused)
        return self.dropout(attended), channel_weights
