from __future__ import annotations

import torch
from torch import nn


class BilateralFusionHead(nn.Module):
    """掩码感知双腕融合：输入 [B,2,H] 和 [B,2]，输出 [B,K]。"""

    def __init__(self, feature_dim: int, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.fusion_dim = 4 * self.feature_dim + 2
        self.classifier = nn.Sequential(
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(self.fusion_dim, self.num_classes),
        )

    def fusion_features(
        self, embeddings: torch.Tensor, wrist_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if embeddings.ndim != 3 or embeddings.shape[1] != 2:
            raise ValueError(
                f"Expected embeddings [B,2,H], got {tuple(embeddings.shape)}"
            )
        if wrist_mask.shape != embeddings.shape[:2]:
            raise ValueError(
                f"Expected wrist_mask {tuple(embeddings.shape[:2])}, "
                f"got {tuple(wrist_mask.shape)}"
            )
        mask = wrist_mask.to(device=embeddings.device, dtype=embeddings.dtype)
        if torch.any((mask != 0) & (mask != 1)):
            raise ValueError("wrist_mask must contain only 0 or 1")
        if torch.any(mask.sum(dim=1) < 1):
            raise ValueError("Every sample must contain at least one valid wrist")
        masked = embeddings * mask.unsqueeze(-1)
        left, right = masked[:, 0], masked[:, 1]
        mean = masked.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        both = (mask[:, 0] * mask[:, 1]).unsqueeze(-1)
        difference = (embeddings[:, 0] - embeddings[:, 1]).abs() * both
        fusion = torch.cat((left, right, mean, difference, mask), dim=-1)
        return fusion, masked

    def forward(
        self, embeddings: torch.Tensor, wrist_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        fusion, masked = self.fusion_features(embeddings, wrist_mask)
        logits = self.classifier(fusion)
        return {
            "logits": logits,
            "probabilities": torch.softmax(logits, dim=-1),
            "fusion_features": fusion,
            "wrist_embeddings": masked,
        }
