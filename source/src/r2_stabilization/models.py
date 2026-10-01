from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from src.targeted_ablation.models import TargetedSubjectMFAM


VARIANTS = ("S1", "S2", "S3")
BRANCH_MODES = ("both", "deep_disabled", "statistical_disabled")


class StabilizedR2(nn.Module):
    """R2 with one of the three preregistered low-capacity fusion paths."""

    deep_dim = 514
    statistical_dim = 32
    gated_dim = 64

    def __init__(self, config: Mapping[str, Any], variant: str) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown R2 stabilization variant: {variant}")
        self.variant = variant
        self.branch_mode = "both"
        self.core = TargetedSubjectMFAM(config, ("R2",), statistical_dim=32)

        # Extract the exact R2 classifier so core encoding can be reused without
        # performing an unused stochastic Dropout forward first.
        self.r2_classifier = self.core.classifier
        self.core.classifier = nn.Identity()
        dropout = float(
            config["model"].get("activity_fusion", {}).get("classifier_dropout", 0.2)
        )

        if variant == "S2":
            self.deep_norm = nn.LayerNorm(self.deep_dim, elementwise_affine=True)
            self.statistical_norm = nn.LayerNorm(
                self.statistical_dim, elementwise_affine=True
            )
        elif variant == "S3":
            self.deep_projection = nn.Linear(self.deep_dim, self.gated_dim)
            self.statistical_projection = nn.Linear(
                self.statistical_dim, self.gated_dim
            )
            self.gate_layer = nn.Linear(self.gated_dim * 2, self.gated_dim)
            self.gated_classifier = nn.Sequential(
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(self.gated_dim, 2),
            )

    def set_branch_mode(self, mode: str) -> None:
        if mode not in BRANCH_MODES:
            raise ValueError(f"Unknown branch diagnostic mode: {mode}")
        self.branch_mode = mode

    def forward(
        self,
        inputs: torch.Tensor,
        wrist_mask: torch.Tensor,
        activity_mask: torch.Tensor | None = None,
        activity_lengths: torch.Tensor | None = None,
        statistical_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if statistical_features is None:
            raise ValueError("R2 stabilization requires statistical features")
        encoded = self.core(
            inputs,
            wrist_mask,
            activity_mask,
            activity_lengths,
            statistical_features,
        )
        deep_raw = encoded["bag_embedding"]
        statistical_raw = statistical_features
        if deep_raw.shape[-1] != self.deep_dim:
            raise ValueError(f"Expected deep embedding dimension {self.deep_dim}")
        if statistical_raw.shape[-1] != self.statistical_dim:
            raise ValueError(
                f"Expected statistical embedding dimension {self.statistical_dim}"
            )

        if self.variant == "S1":
            deep_fusion = deep_raw
            statistical_fusion = statistical_raw
            gate = deep_raw.new_full((deep_raw.shape[0], 1), 0.5)
            if self.branch_mode == "deep_disabled":
                deep_fusion = torch.zeros_like(deep_fusion)
            elif self.branch_mode == "statistical_disabled":
                statistical_fusion = torch.zeros_like(statistical_fusion)
            classifier_input = torch.cat((deep_fusion, statistical_fusion), dim=-1)
            fused = classifier_input
            logits = self.r2_classifier(classifier_input)
        elif self.variant == "S2":
            deep_fusion = self.deep_norm(deep_raw)
            statistical_fusion = self.statistical_norm(statistical_raw)
            gate = deep_raw.new_full((deep_raw.shape[0], 1), 0.5)
            if self.branch_mode == "deep_disabled":
                deep_fusion = torch.zeros_like(deep_fusion)
            elif self.branch_mode == "statistical_disabled":
                statistical_fusion = torch.zeros_like(statistical_fusion)
            classifier_input = torch.cat((deep_fusion, statistical_fusion), dim=-1)
            fused = classifier_input
            logits = self.r2_classifier(classifier_input)
        else:
            deep_projected = self.deep_projection(deep_raw)
            statistical_projected = self.statistical_projection(statistical_raw)
            gate = torch.sigmoid(
                self.gate_layer(torch.cat((deep_projected, statistical_projected), dim=-1))
            )
            deep_fusion = deep_projected
            statistical_fusion = statistical_projected
            if self.branch_mode == "deep_disabled":
                deep_fusion = torch.zeros_like(deep_fusion)
            elif self.branch_mode == "statistical_disabled":
                statistical_fusion = torch.zeros_like(statistical_fusion)
            fused = gate * deep_fusion + (1.0 - gate) * statistical_fusion
            classifier_input = fused
            logits = self.gated_classifier(classifier_input)

        return {
            **encoded,
            "logits": logits,
            "probabilities": torch.softmax(logits, dim=-1),
            "deep_embedding_raw": deep_raw,
            "statistical_embedding_raw": statistical_raw,
            "deep_embedding_fusion": deep_fusion,
            "statistical_embedding_fusion": statistical_fusion,
            "fused_embedding": fused,
            "classifier_input": classifier_input,
            "gate": gate,
        }


def build_stabilized_r2(
    config: Mapping[str, Any], variant: str
) -> StabilizedR2:
    return StabilizedR2(config, variant)
