from __future__ import annotations

import math

import torch
from torch import nn

from .attention import GatedAttentionScorer, TanhAttentionScorer


class AttentionMIL(nn.Module):
    """将时序特征切成实例，并通过注意力和可选 Top-K 门控聚合。"""

    def __init__(
        self,
        feature_dim: int,
        window_size: int,
        stride: int,
        attention_dim: int | None = None,
        retention_ratio: float = 0.3,
        epsilon: float = 1e-8,
        hard_gating: bool = True,
        hard_gating_train_only: bool = False,
        attention_type: str = "tanh",
    ) -> None:
        super().__init__()
        if window_size <= 0 or stride <= 0:
            raise ValueError("window_size and stride must be positive")
        if not 0 < retention_ratio <= 1:
            raise ValueError("retention_ratio must be in (0, 1]")
        attention_dim = attention_dim or max(1, feature_dim // 2)
        if attention_type == "tanh":
            self.scorer = TanhAttentionScorer(feature_dim, attention_dim)
        elif attention_type == "gated":
            self.scorer = GatedAttentionScorer(feature_dim, attention_dim)
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.retention_ratio = float(retention_ratio)
        self.epsilon = float(epsilon)
        self.hard_gating = bool(hard_gating)
        self.hard_gating_train_only = bool(hard_gating_train_only)

    def _instances(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] < self.window_size:
            return features.mean(dim=-1).unsqueeze(1)
        windows = features.unfold(
            dimension=-1,
            size=self.window_size,
            step=self.stride,
        )
        return windows.mean(dim=-1).transpose(1, 2).contiguous()

    def _apply_topk(self, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        instance_count = weights.shape[-1]
        keep_count = max(1, int(math.ceil(self.retention_ratio * instance_count)))
        _, indices = torch.topk(weights, k=keep_count, dim=-1)
        mask = torch.zeros_like(weights, dtype=torch.bool)
        mask.scatter_(dim=-1, index=indices, value=True)
        sparse = torch.where(mask, weights, torch.zeros_like(weights))
        sparse = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(self.epsilon)
        return sparse, mask

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError(
                f"Expected [batch, feature, time], got {tuple(features.shape)}"
            )
        instances = self._instances(features)
        scores = self.scorer(instances)
        raw_weights = torch.softmax(scores, dim=-1)

        use_hard_gating = self.hard_gating and (
            self.training or not self.hard_gating_train_only
        )
        if use_hard_gating:
            weights, topk_mask = self._apply_topk(raw_weights)
        else:
            weights = raw_weights
            topk_mask = torch.ones_like(weights, dtype=torch.bool)

        bag = torch.sum(instances * weights.unsqueeze(-1), dim=1)
        return {
            "bag_embedding": bag,
            "instances": instances,
            "attention_scores": scores,
            "raw_attention": raw_weights,
            "attention": weights,
            "topk_mask": topk_mask,
        }
