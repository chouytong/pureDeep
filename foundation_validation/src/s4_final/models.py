from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from src.models.mfam import _scatter_valid
from src.targeted_ablation.models import FullBandResidualWristEncoder, TargetedSubjectMFAM


BRANCH_MODES = ("both", "frequency_disabled", "fullband_disabled", "statistical_disabled")


class S4SubjectMFAM(nn.Module):
    """Preregistered frequency + R1 full-band + PCA32 direct-concat model."""

    frequency_dim = 514
    fullband_dim = 514
    statistical_dim = 32
    classifier_input_dim = 1060

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.core = TargetedSubjectMFAM(config, ("R1",), statistical_dim=32)
        wrist = self.core.base.activity_encoder.wrist_encoder
        if not isinstance(wrist, FullBandResidualWristEncoder):
            raise TypeError("S4 requires the validated R1 wrist encoder")
        # The preregistered S4 fusion is subject-level direct concatenation, so
        # R1's frequency/full-band projection is intentionally not registered.
        wrist.branch_projection = nn.Identity()
        self.core.classifier = nn.Identity()
        dropout = float(config["model"].get("activity_fusion", {}).get("classifier_dropout", 0.2))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(self.classifier_input_dim, 2),
        )
        self.branch_mode = "both"

    def set_branch_mode(self, mode: str) -> None:
        if mode not in BRANCH_MODES:
            raise ValueError(f"Unknown S4 branch mode: {mode}")
        self.branch_mode = mode

    def _activity_branches(
        self, inputs: torch.Tensor, wrist_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = inputs.shape[0]
        flat = inputs.reshape(batch * 2, inputs.shape[2], inputs.shape[3])
        valid = wrist_mask.reshape(-1).bool()
        if not bool(valid.any()):
            raise ValueError("Batch contains no valid wrist")
        indices = valid.nonzero(as_tuple=False).squeeze(1)
        selected = flat.index_select(0, indices)
        wrist = self.core.base.activity_encoder.wrist_encoder
        frequency = wrist.frequency_path(selected)["bag_embedding"]
        full_encoded, _ = wrist.full_band_encoder(selected)
        fullband = wrist.full_band_mil(full_encoded)["bag_embedding"]
        frequency = _scatter_valid(frequency, indices, batch * 2).reshape(batch, 2, -1)
        fullband = _scatter_valid(fullband, indices, batch * 2).reshape(batch, 2, -1)
        fusion = self.core.base.activity_encoder.fusion
        frequency_features, _ = fusion.fusion_features(frequency, wrist_mask)
        fullband_features, _ = fusion.fusion_features(fullband, wrist_mask)
        return frequency_features, fullband_features

    def forward(
        self,
        inputs: torch.Tensor,
        wrist_mask: torch.Tensor,
        activity_mask: torch.Tensor | None = None,
        activity_lengths: torch.Tensor | None = None,
        statistical_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if inputs.ndim != 5 or inputs.shape[1] != self.core.activity_count:
            raise ValueError("S4 expected [B,A,2,C,T]")
        batch = inputs.shape[0]
        if wrist_mask.shape != (batch, self.core.activity_count, 2):
            raise ValueError("S4 wrist_mask shape mismatch")
        if statistical_features is None or statistical_features.shape != (batch, 32):
            raise ValueError("S4 requires PCA32 statistical features")
        if not torch.isfinite(statistical_features).all():
            raise ValueError("S4 statistical features are non-finite")
        if activity_mask is None:
            activity_mask = torch.ones(
                (batch, self.core.activity_count), dtype=torch.bool, device=inputs.device
            )
        activity_mask = activity_mask.to(device=inputs.device, dtype=torch.bool)
        if activity_lengths is None:
            activity_lengths = torch.full(
                (batch, self.core.activity_count), inputs.shape[-1],
                dtype=torch.long, device=inputs.device,
            ) * activity_mask.long()
        activity_lengths = activity_lengths.to(device=inputs.device, dtype=torch.long)
        if not torch.isfinite(inputs[activity_mask]).all():
            raise ValueError("S4 valid input is non-finite")

        flat_inputs = inputs.reshape(
            batch * self.core.activity_count, 2, inputs.shape[3], inputs.shape[4]
        )
        flat_wrist = wrist_mask.reshape(batch * self.core.activity_count, 2)
        valid_indices = activity_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
        flat_lengths = activity_lengths.reshape(-1)
        frequency = inputs.new_zeros((batch * self.core.activity_count, 514))
        fullband = inputs.new_zeros((batch * self.core.activity_count, 514))
        for length in torch.unique(flat_lengths.index_select(0, valid_indices)).tolist():
            indices = valid_indices[
                flat_lengths.index_select(0, valid_indices) == int(length)
            ]
            f, b = self._activity_branches(
                flat_inputs.index_select(0, indices)[..., : int(length)],
                flat_wrist.index_select(0, indices),
            )
            frequency = frequency.index_copy(0, indices, f)
            fullband = fullband.index_copy(0, indices, b)
        frequency = frequency.reshape(batch, self.core.activity_count, 514)
        fullband = fullband.reshape(batch, self.core.activity_count, 514)
        aggregator = self.core.base.activity_aggregator
        frequency_subject = aggregator(frequency, activity_mask)["subject_embedding"]
        fullband_subject = aggregator(fullband, activity_mask)["subject_embedding"]
        statistical = statistical_features
        cosine = nn.functional.cosine_similarity(
            frequency_subject, fullband_subject, dim=-1, eps=1e-8
        ).unsqueeze(-1)

        frequency_used = frequency_subject
        fullband_used = fullband_subject
        statistical_used = statistical
        if self.branch_mode == "frequency_disabled":
            frequency_used = torch.zeros_like(frequency_used)
        elif self.branch_mode == "fullband_disabled":
            fullband_used = torch.zeros_like(fullband_used)
        elif self.branch_mode == "statistical_disabled":
            statistical_used = torch.zeros_like(statistical_used)
        classifier_input = torch.cat(
            (frequency_used, fullband_used, statistical_used), dim=-1
        )
        logits = self.classifier(classifier_input)
        return {
            "logits": logits,
            "probabilities": torch.softmax(logits, dim=-1),
            "bag_embedding": frequency_subject,
            "frequency_embedding_raw": frequency_subject,
            "fullband_embedding_raw": fullband_subject,
            "statistical_embedding_raw": statistical,
            "frequency_embedding_fusion": frequency_used,
            "fullband_embedding_fusion": fullband_used,
            "statistical_embedding_fusion": statistical_used,
            "frequency_fullband_cosine": cosine,
            "deep_embedding_raw": frequency_subject,
            "deep_embedding_fusion": frequency_used,
            "fused_embedding": classifier_input,
            "classifier_input": classifier_input,
            "gate": logits.new_full((batch, 1), 0.5),
        }


def build_s4(config: Mapping[str, Any]) -> S4SubjectMFAM:
    return S4SubjectMFAM(config)
