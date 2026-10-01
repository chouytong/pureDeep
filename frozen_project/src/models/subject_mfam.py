from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .activity_fusion import ActivityAttentionAggregator
from .mfam import MFAM


class SubjectMFAM(nn.Module):
    """Shared MFAM activity encoder followed by masked subject-level attention."""

    def __init__(
        self,
        activities: Sequence[str],
        input_channels: int,
        num_classes: int,
        sample_rate: float,
        frequency_config: Mapping[str, Any],
        encoder_config: Mapping[str, Any],
        mil_config: Mapping[str, Any],
        activity_fusion_config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.activities = tuple(str(value) for value in activities)
        self.activity_count = len(self.activities)
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.activity_encoder = MFAM(
            input_channels=self.input_channels,
            num_classes=self.num_classes,
            sample_rate=sample_rate,
            frequency_config=frequency_config,
            encoder_config=encoder_config,
            mil_config=mil_config,
            classifier_config={},
        )
        self.activity_encoder.fusion.classifier = nn.Identity()
        feature_dim = self.activity_encoder.fusion.fusion_dim
        self.activity_aggregator = ActivityAttentionAggregator(
            feature_dim=feature_dim,
            attention_dim=int(activity_fusion_config.get("attention_dim", 128)),
            activity_count=self.activity_count,
            dropout=float(activity_fusion_config.get("dropout", 0.0)),
        )
        classifier_dropout = float(
            activity_fusion_config.get("classifier_dropout", 0.2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout)
            if classifier_dropout > 0
            else nn.Identity(),
            nn.Linear(feature_dim, self.num_classes),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        wrist_mask: torch.Tensor,
        activity_mask: torch.Tensor | None = None,
        activity_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if inputs.ndim != 5 or inputs.shape[1] != self.activity_count:
            raise ValueError(
                f"Expected [B,{self.activity_count},2,C,T], got {tuple(inputs.shape)}"
            )
        if inputs.shape[2] != 2 or inputs.shape[3] != self.input_channels:
            raise ValueError("SubjectMFAM wrist or channel dimension is invalid")
        batch_size = inputs.shape[0]
        if wrist_mask.shape != (batch_size, self.activity_count, 2):
            raise ValueError("wrist_mask must have shape [B,A,2]")
        if activity_mask is None:
            activity_mask = torch.ones(
                (batch_size, self.activity_count), dtype=torch.bool, device=inputs.device
            )
        activity_mask = activity_mask.to(device=inputs.device, dtype=torch.bool)
        if activity_mask.shape != (batch_size, self.activity_count):
            raise ValueError("activity_mask must have shape [B,A]")
        if not torch.isfinite(inputs[activity_mask]).all():
            raise ValueError("Valid activity input contains NaN or infinite values")
        if activity_lengths is None:
            activity_lengths = torch.full(
                (batch_size, self.activity_count),
                inputs.shape[-1],
                dtype=torch.long,
                device=inputs.device,
            )
            activity_lengths = activity_lengths * activity_mask.to(torch.long)
        activity_lengths = activity_lengths.to(device=inputs.device, dtype=torch.long)
        if activity_lengths.shape != (batch_size, self.activity_count):
            raise ValueError("activity_lengths must have shape [B,A]")
        if bool((activity_lengths[activity_mask] < 2).any()):
            raise ValueError("Every valid activity requires at least 2 samples")
        if bool((activity_lengths[activity_mask] > inputs.shape[-1]).any()):
            raise ValueError("activity_lengths exceed the padded input length")
        if bool((activity_lengths[~activity_mask] != 0).any()):
            raise ValueError("Missing activities must have length 0")
        flat_inputs = inputs.reshape(
            batch_size * self.activity_count, 2, inputs.shape[3], inputs.shape[4]
        )
        flat_wrist_mask = wrist_mask.reshape(batch_size * self.activity_count, 2)
        valid_indices = activity_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
        flat_lengths = activity_lengths.reshape(-1)
        feature_dim = self.activity_encoder.fusion.fusion_dim
        features = inputs.new_zeros(
            (batch_size * self.activity_count, feature_dim)
        )
        # Padding never enters FFT/Conv/MIL: activities are grouped by true T and
        # sliced before they pass through the shared MFAM activity encoder.
        for length in torch.unique(flat_lengths.index_select(0, valid_indices)).tolist():
            length_indices = valid_indices[
                flat_lengths.index_select(0, valid_indices) == int(length)
            ]
            encoded = self.activity_encoder.encode_activity(
                flat_inputs.index_select(0, length_indices)[..., : int(length)],
                flat_wrist_mask.index_select(0, length_indices),
            )
            features = features.index_copy(
                0, length_indices, encoded["fusion_features"]
            )
        features = features.reshape(batch_size, self.activity_count, feature_dim)
        aggregated = self.activity_aggregator(features, activity_mask)
        logits = self.classifier(aggregated["subject_embedding"])
        return {
            "logits": logits,
            "probabilities": torch.softmax(logits, dim=-1),
            "bag_embedding": aggregated["subject_embedding"],
            **aggregated,
        }


def build_subject_mfam(config: Mapping[str, Any]) -> SubjectMFAM:
    model = config["model"]
    return SubjectMFAM(
        activities=config["data"]["activities"],
        input_channels=int(model["input_channels"]),
        num_classes=int(model["num_classes"]),
        sample_rate=float(model["sample_rate"]),
        frequency_config=model["frequency"],
        encoder_config=model["encoder"],
        mil_config=model["mil"],
        activity_fusion_config=model.get("activity_fusion", {}),
    )
