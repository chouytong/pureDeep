from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from src.models.encoder import MultiScaleChannelAttentionEncoder
from src.models.mfam import WristMFAMEncoder, _scatter_valid
from src.models.mil import AttentionMIL
from src.models.subject_mfam import SubjectMFAM


class MaskedMeanActivityAggregator(nn.Module):
    """Parameter-free arithmetic mean over valid raw activity embeddings."""

    def forward(
        self, features: torch.Tensor, activity_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError(f"Expected [B,A,D], got {tuple(features.shape)}")
        if activity_mask.shape != features.shape[:2]:
            raise ValueError("activity_mask shape does not match features")
        mask = activity_mask.to(device=features.device, dtype=torch.bool)
        counts = mask.sum(dim=1, keepdim=True)
        if bool((counts == 0).any()):
            raise ValueError("Every subject requires at least one valid activity")
        weights = mask.to(features.dtype) / counts.to(features.dtype)
        subject = torch.sum(features * weights.unsqueeze(-1), dim=1)
        return {
            "subject_embedding": subject,
            "activity_embeddings": features,
            "activity_attention": weights,
        }


def group_count(channels: int, maximum_groups: int = 8) -> int:
    for groups in range(min(int(maximum_groups), int(channels)), 0, -1):
        if channels % groups == 0:
            return groups
    raise AssertionError("Every positive integer is divisible by one")


def replace_temporal_batchnorm_with_groupnorm(module: nn.Module) -> int:
    """Deterministically replace BatchNorm1d descendants, returning count."""

    replacements = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm1d):
            groups = group_count(child.num_features, 8)
            replacement = nn.GroupNorm(
                num_groups=groups,
                num_channels=child.num_features,
                eps=child.eps,
                affine=child.affine,
            )
            setattr(module, name, replacement)
            replacements += 1
        else:
            replacements += replace_temporal_batchnorm_with_groupnorm(child)
    return replacements


def normalized_attention_entropy(weights: torch.Tensor) -> torch.Tensor:
    count = int(weights.shape[-1])
    if count <= 1:
        return weights.new_zeros(weights.shape[:-1])
    safe = weights.clamp_min(torch.finfo(weights.dtype).tiny)
    entropy = -(weights * safe.log()).sum(dim=-1)
    return entropy / math.log(count)


class FullBandResidualWristEncoder(nn.Module):
    """M0 frequency wrist path plus one preregistered full-band residual path."""

    def __init__(
        self,
        frequency_path: WristMFAMEncoder,
        *,
        input_channels: int,
        sample_rate: float,
        encoder_config: Mapping[str, Any],
        mil_config: Mapping[str, Any],
        full_branch_channels: int = 32,
        full_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.frequency_path = frequency_path
        self.input_channels = int(input_channels)
        self.feature_dim = int(frequency_path.feature_dim)
        if int(full_hidden_dim) != self.feature_dim:
            raise ValueError("R1 requires equal 128-dimensional branch embeddings")
        self.full_band_encoder = MultiScaleChannelAttentionEncoder(
            input_channels=self.input_channels,
            branch_channels=int(full_branch_channels),
            hidden_dim=int(full_hidden_dim),
            kernel_size=int(encoder_config.get("kernel_size", 3)),
            dilations=tuple(int(v) for v in encoder_config.get("dilations", [1, 2, 4])),
            channel_reduction=int(encoder_config.get("channel_reduction", 8)),
            conv_bias=bool(encoder_config.get("conv_bias", False)),
            dropout=float(encoder_config.get("dropout", 0.0)),
        )
        window_size = max(
            1, int(round(float(mil_config["window_seconds"]) * float(sample_rate)))
        )
        stride = max(
            1, int(round(window_size * (1.0 - float(mil_config["overlap"]))))
        )
        self.full_band_mil = AttentionMIL(
            feature_dim=int(full_hidden_dim),
            window_size=window_size,
            stride=stride,
            attention_dim=(
                None
                if mil_config.get("attention_dim") is None
                else int(mil_config["attention_dim"])
            ),
            retention_ratio=float(mil_config.get("retention_ratio", 0.3)),
            epsilon=float(mil_config.get("epsilon", 1e-8)),
            hard_gating=bool(mil_config.get("hard_gating", True)),
            hard_gating_train_only=bool(
                mil_config.get("hard_gating_train_only", False)
            ),
            attention_type=str(mil_config.get("attention_type", "tanh")),
        )
        self.branch_projection = nn.Linear(self.feature_dim * 2, self.feature_dim)
        self.branch_mode = "both"

    def set_branch_mode(self, mode: str) -> None:
        if mode not in {"both", "frequency_only", "fullband_only"}:
            raise ValueError(f"Unknown R1 branch mode: {mode}")
        self.branch_mode = mode

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        frequency = self.frequency_path(inputs)
        full_encoded, full_channel_weights = self.full_band_encoder(inputs)
        full_mil = self.full_band_mil(full_encoded)
        frequency_bag = frequency["bag_embedding"]
        full_bag = full_mil["bag_embedding"]
        frequency_used = frequency_bag
        full_used = full_bag
        if self.branch_mode == "frequency_only":
            full_used = torch.zeros_like(full_used)
        elif self.branch_mode == "fullband_only":
            frequency_used = torch.zeros_like(frequency_used)
        combined = self.branch_projection(torch.cat((frequency_used, full_used), dim=-1))
        return {
            **frequency,
            "bag_embedding": combined,
            "frequency_bag_embedding": frequency_bag,
            "full_band_bag_embedding": full_bag,
            "full_band_encoded_features": full_encoded,
            "full_band_channel_weights": full_channel_weights,
            "full_band_raw_attention": full_mil["raw_attention"],
            "full_band_attention": full_mil["attention"],
            "full_band_topk_mask": full_mil["topk_mask"],
        }


class TargetedSubjectMFAM(nn.Module):
    """Composable wrapper used only by the preregistered development ablations."""

    def __init__(
        self,
        config: Mapping[str, Any],
        modifications: Sequence[str],
        statistical_dim: int = 32,
    ) -> None:
        super().__init__()
        self.modifications = tuple(sorted(str(value) for value in modifications))
        allowed = {"R1", "R2", "A1", "A2", "N1"}
        if not set(self.modifications) <= allowed:
            raise ValueError(f"Unknown modifications: {self.modifications}")
        if len(set(self.modifications) & {"R1", "R2"}) > 1:
            raise ValueError("At most one representation modification is allowed")
        if len(self.modifications) > 2:
            raise ValueError("At most two modifications are allowed")
        model = config["model"]
        self.base = SubjectMFAM(
            activities=config["data"]["activities"],
            input_channels=int(model["input_channels"]),
            num_classes=int(model["num_classes"]),
            sample_rate=float(model["sample_rate"]),
            frequency_config=deepcopy(model["frequency"]),
            encoder_config=deepcopy(model["encoder"]),
            mil_config=deepcopy(model["mil"]),
            activity_fusion_config=deepcopy(model.get("activity_fusion", {})),
        )
        self.activities = self.base.activities
        self.activity_count = self.base.activity_count
        self.input_channels = self.base.input_channels
        self.num_classes = self.base.num_classes
        self.statistical_dim = int(statistical_dim) if "R2" in self.modifications else 0
        self.groupnorm_replacements = 0

        if "R1" in self.modifications:
            original = self.base.activity_encoder.wrist_encoder
            self.base.activity_encoder.wrist_encoder = FullBandResidualWristEncoder(
                original,
                input_channels=self.input_channels,
                sample_rate=float(model["sample_rate"]),
                encoder_config=model["encoder"],
                mil_config=model["mil"],
                full_branch_channels=32,
                full_hidden_dim=128,
            )
        if "A1" in self.modifications:
            for module in self.base.modules():
                if isinstance(module, AttentionMIL):
                    module.hard_gating = False
        if "A2" in self.modifications:
            self.base.activity_aggregator = MaskedMeanActivityAggregator()
        if "N1" in self.modifications:
            self.groupnorm_replacements = replace_temporal_batchnorm_with_groupnorm(
                self.base.activity_encoder.wrist_encoder
            )
            if self.groupnorm_replacements == 0:
                raise RuntimeError("N1 found no temporal BatchNorm1d modules")

        feature_dim = self.base.activity_encoder.fusion.fusion_dim
        classifier_dropout = float(model.get("activity_fusion", {}).get("classifier_dropout", 0.2))
        if self.statistical_dim:
            self.classifier = nn.Sequential(
                nn.Dropout(classifier_dropout) if classifier_dropout > 0 else nn.Identity(),
                nn.Linear(feature_dim + self.statistical_dim, self.num_classes),
            )
        else:
            self.classifier = self.base.classifier

    def set_r1_branch_mode(self, mode: str) -> None:
        wrist = self.base.activity_encoder.wrist_encoder
        if not isinstance(wrist, FullBandResidualWristEncoder):
            if mode != "both":
                raise ValueError("Branch ablation is only available for R1")
            return
        wrist.set_branch_mode(mode)

    def _encode_activity(
        self, inputs: torch.Tensor, wrist_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Equivalent to MFAM.encode_activity while preserving R1 diagnostics."""

        batch_size = inputs.shape[0]
        flat = inputs.reshape(batch_size * 2, inputs.shape[2], inputs.shape[3])
        valid = wrist_mask.reshape(-1).bool()
        if not bool(valid.any()):
            raise ValueError("Batch contains no valid wrist")
        valid_indices = valid.nonzero(as_tuple=False).squeeze(1)
        encoded = self.base.activity_encoder.wrist_encoder(
            flat.index_select(0, valid_indices)
        )
        scattered = {
            key: _scatter_valid(value, valid_indices, batch_size * 2).reshape(
                batch_size, 2, *value.shape[1:]
            )
            for key, value in encoded.items()
        }
        fusion_features, wrist_embeddings = (
            self.base.activity_encoder.fusion.fusion_features(
                scattered["bag_embedding"], wrist_mask
            )
        )
        return {
            "fusion_features": fusion_features,
            "wrist_embeddings": wrist_embeddings,
            **scattered,
        }

    def forward(
        self,
        inputs: torch.Tensor,
        wrist_mask: torch.Tensor,
        activity_mask: torch.Tensor | None = None,
        activity_lengths: torch.Tensor | None = None,
        statistical_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if inputs.ndim != 5 or inputs.shape[1] != self.activity_count:
            raise ValueError(
                f"Expected [B,{self.activity_count},2,C,T], got {tuple(inputs.shape)}"
            )
        batch_size = inputs.shape[0]
        if wrist_mask.shape != (batch_size, self.activity_count, 2):
            raise ValueError("wrist_mask must have shape [B,A,2]")
        if activity_mask is None:
            activity_mask = torch.ones(
                (batch_size, self.activity_count), dtype=torch.bool, device=inputs.device
            )
        activity_mask = activity_mask.to(device=inputs.device, dtype=torch.bool)
        if activity_lengths is None:
            activity_lengths = torch.full(
                (batch_size, self.activity_count),
                inputs.shape[-1],
                dtype=torch.long,
                device=inputs.device,
            ) * activity_mask.to(torch.long)
        activity_lengths = activity_lengths.to(device=inputs.device, dtype=torch.long)
        if not torch.isfinite(inputs[activity_mask]).all():
            raise ValueError("Valid activity input contains non-finite values")

        flat_inputs = inputs.reshape(
            batch_size * self.activity_count, 2, inputs.shape[3], inputs.shape[4]
        )
        flat_wrist_mask = wrist_mask.reshape(batch_size * self.activity_count, 2)
        valid_indices = activity_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
        flat_lengths = activity_lengths.reshape(-1)
        feature_dim = self.base.activity_encoder.fusion.fusion_dim
        features = inputs.new_zeros((batch_size * self.activity_count, feature_dim))
        entropy = inputs.new_zeros(batch_size * self.activity_count)
        utilization = inputs.new_zeros(batch_size * self.activity_count)
        frequency_l2 = inputs.new_zeros(batch_size * self.activity_count)
        full_band_l2 = inputs.new_zeros(batch_size * self.activity_count)
        branch_cosine = inputs.new_zeros(batch_size * self.activity_count)

        for length in torch.unique(flat_lengths.index_select(0, valid_indices)).tolist():
            length_indices = valid_indices[
                flat_lengths.index_select(0, valid_indices) == int(length)
            ]
            selected_wrist_mask = flat_wrist_mask.index_select(0, length_indices)
            encoded = self._encode_activity(
                flat_inputs.index_select(0, length_indices)[..., : int(length)],
                selected_wrist_mask,
            )
            features = features.index_copy(0, length_indices, encoded["fusion_features"])
            raw_attention = encoded["raw_attention"]
            topk_mask = encoded["topk_mask"]
            wrist_weights = selected_wrist_mask.to(raw_attention.dtype)
            denominator = wrist_weights.sum(dim=1).clamp_min(1.0)
            entropy_value = (
                normalized_attention_entropy(raw_attention) * wrist_weights
            ).sum(dim=1) / denominator
            utilization_value = (
                topk_mask.to(raw_attention.dtype).mean(dim=-1) * wrist_weights
            ).sum(dim=1) / denominator
            entropy = entropy.index_copy(0, length_indices, entropy_value)
            utilization = utilization.index_copy(0, length_indices, utilization_value)

            if "frequency_bag_embedding" in encoded:
                frequency_bag = encoded["frequency_bag_embedding"]
                full_bag = encoded["full_band_bag_embedding"]
                frequency_value = frequency_bag.norm(dim=-1)
                full_value = full_bag.norm(dim=-1)
                cosine_value = nn.functional.cosine_similarity(
                    frequency_bag, full_bag, dim=-1, eps=1e-8
                )
                frequency_l2 = frequency_l2.index_copy(
                    0,
                    length_indices,
                    (frequency_value * wrist_weights).sum(dim=1) / denominator,
                )
                full_band_l2 = full_band_l2.index_copy(
                    0,
                    length_indices,
                    (full_value * wrist_weights).sum(dim=1) / denominator,
                )
                branch_cosine = branch_cosine.index_copy(
                    0,
                    length_indices,
                    (cosine_value * wrist_weights).sum(dim=1) / denominator,
                )

        features = features.reshape(batch_size, self.activity_count, feature_dim)
        aggregated = self.base.activity_aggregator(features, activity_mask)
        subject_embedding = aggregated["subject_embedding"]
        if self.statistical_dim:
            if statistical_features is None:
                raise ValueError("R2 requires fold-transformed statistical_features")
            if statistical_features.shape != (batch_size, self.statistical_dim):
                raise ValueError(
                    f"Expected statistical_features [{batch_size},{self.statistical_dim}]"
                )
            if not torch.isfinite(statistical_features).all():
                raise ValueError("statistical_features contain non-finite values")
            classifier_features = torch.cat((subject_embedding, statistical_features), dim=-1)
        else:
            if statistical_features is not None:
                raise ValueError("Statistical features supplied to a non-R2 model")
            classifier_features = subject_embedding
        logits = self.classifier(classifier_features)
        return {
            "logits": logits,
            "probabilities": torch.softmax(logits, dim=-1),
            "bag_embedding": subject_embedding,
            "classifier_features": classifier_features,
            **aggregated,
            "mil_attention_entropy": entropy.reshape(batch_size, self.activity_count),
            "mil_instance_utilization": utilization.reshape(batch_size, self.activity_count),
            "r1_frequency_bag_l2": frequency_l2.reshape(batch_size, self.activity_count),
            "r1_full_band_bag_l2": full_band_l2.reshape(batch_size, self.activity_count),
            "r1_branch_cosine": branch_cosine.reshape(batch_size, self.activity_count),
        }


def build_targeted_model(
    config: Mapping[str, Any], modifications: Sequence[str]
) -> TargetedSubjectMFAM:
    return TargetedSubjectMFAM(config, modifications, statistical_dim=32)
