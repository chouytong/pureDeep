from __future__ import annotations

import math

import torch
from torch import nn


class ActivityAttentionAggregator(nn.Module):
    """Masked lightweight attention over fixed-order activity embeddings."""

    def __init__(
        self, feature_dim: int, attention_dim: int, activity_count: int, dropout: float
    ) -> None:
        super().__init__()
        self.activity_embedding = nn.Embedding(activity_count, feature_dim)
        self.normalization = nn.LayerNorm(feature_dim)
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(attention_dim, 1),
        )

    def forward(
        self, features: torch.Tensor, activity_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError(f"Expected [B,A,D], got {tuple(features.shape)}")
        if activity_mask.shape != features.shape[:2]:
            raise ValueError(
                f"Expected activity_mask {tuple(features.shape[:2])}, "
                f"got {tuple(activity_mask.shape)}"
            )
        mask = activity_mask.to(device=features.device, dtype=torch.bool)
        if torch.any(mask.sum(dim=1) == 0):
            raise ValueError("Every subject must contain at least one valid activity")
        indices = torch.arange(features.shape[1], device=features.device)
        enriched = self.normalization(features + self.activity_embedding(indices)[None])
        scores = self.scorer(enriched).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1).masked_fill(~mask, 0.0)
        subject = torch.sum(enriched * attention.unsqueeze(-1), dim=1)
        return {
            "subject_embedding": subject,
            "activity_embeddings": enriched,
            "activity_attention": attention,
        }


class MaskedMeanActivityAggregator(nn.Module):
    """Parameter-free aggregation used to test activity-attention necessity."""

    def forward(
        self, features: torch.Tensor, activity_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError(f"Expected [B,A,D], got {tuple(features.shape)}")
        if activity_mask.shape != features.shape[:2]:
            raise ValueError(
                f"Expected activity_mask {tuple(features.shape[:2])}, "
                f"got {tuple(activity_mask.shape)}"
            )
        mask = activity_mask.to(device=features.device, dtype=torch.bool)
        counts = mask.sum(dim=1, keepdim=True)
        if torch.any(counts == 0):
            raise ValueError("Every subject must contain at least one valid activity")
        weights = mask.to(features.dtype) / counts.to(features.dtype)
        subject = torch.sum(features * weights.unsqueeze(-1), dim=1)
        return {
            "subject_embedding": subject,
            "activity_embeddings": features,
            "activity_attention": weights,
        }


class MeanAttentionResidualAggregator(ActivityAttentionAggregator):
    """Stable mean anchor plus a gated residual toward learned attention."""

    def __init__(
        self,
        feature_dim: int,
        attention_dim: int,
        activity_count: int,
        dropout: float,
        initial_attention_weight: float = 0.25,
    ) -> None:
        super().__init__(feature_dim, attention_dim, activity_count, dropout)
        if not 0.0 < initial_attention_weight < 1.0:
            raise ValueError("initial_attention_weight must be in (0, 1)")
        initial_logit = math.log(
            initial_attention_weight / (1.0 - initial_attention_weight)
        )
        self.attention_residual_logit = nn.Parameter(
            torch.tensor(initial_logit, dtype=torch.float32)
        )

    def forward(
        self, features: torch.Tensor, activity_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        attention_output = super().forward(features, activity_mask)
        enriched = attention_output["activity_embeddings"]
        mask = activity_mask.to(device=features.device, dtype=torch.bool)
        weights = mask.to(features.dtype) / mask.sum(
            dim=1, keepdim=True
        ).to(features.dtype)
        mean_subject = torch.sum(enriched * weights.unsqueeze(-1), dim=1)
        gate = torch.sigmoid(self.attention_residual_logit)
        subject = mean_subject + gate * (
            attention_output["subject_embedding"] - mean_subject
        )
        return {
            **attention_output,
            "subject_embedding": subject,
            "activity_mean_weights": weights,
            "attention_residual_gate": gate.expand(features.shape[0]),
        }
