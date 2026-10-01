from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from .bilateral import BilateralFusionHead
from .encoder import MultiScaleChannelAttentionEncoder
from .frequency import FrequencyDecomposition
from .mil import AttentionMIL


class WristMFAMEncoder(nn.Module):
    """单腕六轴（或传感器消融后的三轴）共享编码器。"""

    def __init__(
        self,
        input_channels: int,
        sample_rate: float,
        frequency_config: Mapping[str, Any],
        encoder_config: Mapping[str, Any],
        mil_config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.feature_dim = int(encoder_config["hidden_dim"])
        self.frequency = FrequencyDecomposition(
            sample_rate=float(sample_rate),
            bands=frequency_config["bands"],
            boundary=frequency_config.get("boundary", "half_open"),
            include_last_high=frequency_config.get("include_last_high", True),
            include_original=frequency_config.get("include_original", False),
        )
        frequency_channels = self.input_channels * self.frequency.output_multiplier
        self.encoder = MultiScaleChannelAttentionEncoder(
            input_channels=frequency_channels,
            branch_channels=int(encoder_config["branch_channels"]),
            hidden_dim=self.feature_dim,
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
        self.mil = AttentionMIL(
            feature_dim=self.feature_dim,
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

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        # inputs=[有效腕数量,C,T]；左右腕调用的是同一个实例，因此参数真正共享。
        decomposed = self.frequency(inputs)
        encoded, channel_weights = self.encoder(decomposed)
        mil_output = self.mil(encoded)
        return {
            "bag_embedding": mil_output["bag_embedding"],
            "encoded_features": encoded,
            "channel_weights": channel_weights,
            "instances": mil_output["instances"],
            "attention_scores": mil_output["attention_scores"],
            "raw_attention": mil_output["raw_attention"],
            "attention": mil_output["attention"],
            "topk_mask": mil_output["topk_mask"],
        }


def _scatter_valid(
    values: torch.Tensor, valid_indices: torch.Tensor, total: int
) -> torch.Tensor:
    target = values.new_zeros((total, *values.shape[1:]))
    return target.index_copy(0, valid_indices, values)


class MFAM(nn.Module):
    """双腕 PADS 分类模型。

    输入 x=[B,2,C,T]、wrist_mask=[B,2]。两腕通过同一个 WristMFAMEncoder，
    每腕先做频率分解、MS-CAE 和 Attention-MIL，再融合左右/均值/绝对差异。
    这是普通 PADS 分类模型，不表示论文复现。
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        sample_rate: float,
        frequency_config: Mapping[str, Any],
        encoder_config: Mapping[str, Any],
        mil_config: Mapping[str, Any],
        classifier_config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.sample_rate = float(sample_rate)
        self.wrist_encoder = WristMFAMEncoder(
            input_channels=self.input_channels,
            sample_rate=self.sample_rate,
            frequency_config=frequency_config,
            encoder_config=encoder_config,
            mil_config=mil_config,
        )
        self.fusion = BilateralFusionHead(
            feature_dim=self.wrist_encoder.feature_dim,
            num_classes=self.num_classes,
            dropout=float(classifier_config.get("dropout", 0.0)),
        )

    def encode_activity(
        self, inputs: torch.Tensor, wrist_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if inputs.ndim != 4 or inputs.shape[1] != 2:
            raise ValueError(
                f"Expected [batch,2,channels,time], got {tuple(inputs.shape)}"
            )
        if inputs.shape[2] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} channels per wrist, got {inputs.shape[2]}"
            )
        if not torch.isfinite(inputs).all():
            raise ValueError("Input contains NaN or infinite values")
        batch_size = inputs.shape[0]
        if wrist_mask is None:
            wrist_mask = inputs.new_ones((batch_size, 2))
        wrist_mask = wrist_mask.to(device=inputs.device)
        if wrist_mask.shape != (batch_size, 2):
            raise ValueError(f"Expected wrist_mask [{batch_size},2]")
        flat = inputs.reshape(batch_size * 2, inputs.shape[2], inputs.shape[3])
        valid = wrist_mask.reshape(-1).bool()
        if not bool(valid.any()):
            raise ValueError("Batch contains no valid wrist")
        valid_indices = valid.nonzero(as_tuple=False).squeeze(1)
        encoded = self.wrist_encoder(flat.index_select(0, valid_indices))
        scattered = {
            key: _scatter_valid(value, valid_indices, batch_size * 2).reshape(
                batch_size, 2, *value.shape[1:]
            )
            for key, value in encoded.items()
        }
        fusion_features, wrist_embeddings = self.fusion.fusion_features(
            scattered["bag_embedding"], wrist_mask
        )
        return {
            "fusion_features": fusion_features,
            "wrist_embeddings": wrist_embeddings,
            "bag_embedding": fusion_features,
            "encoded_features": scattered["encoded_features"],
            "channel_weights": scattered["channel_weights"],
            "instances": scattered["instances"],
            "attention_scores": scattered["attention_scores"],
            "raw_attention": scattered["raw_attention"],
            "attention": scattered["attention"],
            "topk_mask": scattered["topk_mask"],
        }

    def forward(
        self, inputs: torch.Tensor, wrist_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        encoded = self.encode_activity(inputs, wrist_mask)
        logits = self.fusion.classifier(encoded["fusion_features"])
        return {
            "logits": logits,
            "probabilities": torch.softmax(logits, dim=-1),
            **encoded,
        }


def build_model(config: Mapping[str, Any]) -> nn.Module:
    model_config = config["model"]
    model_name = str(model_config.get("name", "mfam")).lower()
    if model_name == "pure_deep_subject":
        from .pure_deep import build_pure_deep_subject

        return build_pure_deep_subject(config)
    if model_name in {"subject_mfam", "multi_activity_mfam"}:
        from .subject_mfam import build_subject_mfam

        return build_subject_mfam(config)
    if model_name != "mfam":
        raise ValueError(f"Unsupported model.name: {model_name!r}")
    return MFAM(
        input_channels=int(model_config["input_channels"]),
        num_classes=int(model_config["num_classes"]),
        sample_rate=float(model_config["sample_rate"]),
        frequency_config=model_config["frequency"],
        encoder_config=model_config["encoder"],
        mil_config=model_config["mil"],
        classifier_config=model_config.get("classifier", {}),
    )
