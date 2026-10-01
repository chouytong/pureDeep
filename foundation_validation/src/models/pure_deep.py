from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as functional

from .activity_fusion import (
    ActivityAttentionAggregator,
    MaskedMeanActivityAggregator,
    MeanAttentionResidualAggregator,
)
from .bilateral import BilateralFusionHead


def make_temporal_normalization(
    channels: int, normalization: str, group_count: int
) -> nn.Module:
    """Build normalization for the temporal path.

    GroupNorm is the batch-statistics-independent comparison used by NR-01.
    """
    mode = str(normalization).lower()
    if mode == "batch_norm":
        return nn.BatchNorm1d(channels)
    if mode == "group_norm":
        groups = min(int(group_count), int(channels))
        while groups > 1 and channels % groups:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unsupported temporal normalization: {normalization!r}")


class SeparableResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int,
                 dropout: float, normalization: str = "batch_norm",
                 group_count: int = 8) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding,
                      dilation=dilation, groups=channels, bias=False),
            make_temporal_normalization(channels, normalization, group_count),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1, bias=False),
            make_temporal_normalization(channels, normalization, group_count),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.block(inputs)


class FeatureStatisticsPool(nn.Module):
    """Learnable attention plus moments of learned feature maps.

    The moments are not handcrafted signal features: they are computed only after
    the end-to-end temporal encoder and their gradients propagate through it.
    """

    def __init__(self, feature_dim: int, mode: str = "attention_mean_std") -> None:
        super().__init__()
        if mode not in {"attention_mean", "attention_mean_std"}:
            raise ValueError(f"Unsupported feature pooling mode: {mode!r}")
        self.mode = mode
        self.attention = nn.Conv1d(feature_dim, 1, kernel_size=1)
        multiplier = 3 if mode == "attention_mean_std" else 2
        self.projection = nn.Sequential(
            nn.Linear(feature_dim * multiplier, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        weights = torch.softmax(self.attention(features), dim=-1)
        attention_mean = torch.sum(features * weights, dim=-1)
        mean = features.mean(dim=-1)
        pooled = [attention_mean, mean]
        if self.mode == "attention_mean_std":
            variance = torch.mean((features - mean.unsqueeze(-1)).square(), dim=-1)
            pooled.append(torch.sqrt(variance + 1.0e-5))
        embedding = self.projection(torch.cat(pooled, dim=-1))
        return {"embedding": embedding, "temporal_attention": weights.squeeze(1)}


class LearnableSpectrumEncoder(nn.Module):
    """Encode a differentiable full-spectrum representation without fixed bands."""

    def __init__(
        self,
        input_channels: int,
        feature_dim: int,
        sample_rate: float,
        frequency_bins: int,
        maximum_hz: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if frequency_bins < 16:
            raise ValueError("frequency_bins must be at least 16")
        if not 0.0 < maximum_hz <= sample_rate / 2.0:
            raise ValueError("maximum_hz must be in (0, Nyquist]")
        self.sample_rate = float(sample_rate)
        self.frequency_bins = int(frequency_bins)
        self.maximum_hz = float(maximum_hz)
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, feature_dim, kernel_size=9, stride=2,
                      padding=4, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.GELU(),
        )
        self.encoder = nn.Sequential(
            SeparableResidualBlock(feature_dim, 5, 1, dropout),
            SeparableResidualBlock(feature_dim, 5, 2, dropout),
        )
        self.pool = FeatureStatisticsPool(feature_dim, "attention_mean_std")

    def spectrum(self, inputs: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(inputs, dim=-1, norm="forward")
        frequencies = torch.fft.rfftfreq(
            inputs.shape[-1], d=1.0 / self.sample_rate, device=inputs.device
        )
        keep = frequencies <= self.maximum_hz
        magnitude = torch.log1p(torch.abs(spectrum[..., keep]))
        return functional.interpolate(
            magnitude,
            size=self.frequency_bins,
            mode="linear",
            align_corners=False,
        )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        representation = self.spectrum(inputs)
        features = self.encoder(self.stem(representation))
        pooled = self.pool(features)
        return {
            "embedding": pooled["embedding"],
            "spectral_features": features,
            "spectral_attention": pooled["temporal_attention"],
        }


class NormFreeMomentEncoder(nn.Module):
    """Preserve subject-level distribution cues in learned feature maps.

    This branch contains no predefined signal statistics or fixed filters. Its
    temporal filters are learned end to end and deliberately omit normalization
    before global mean/std pooling, so between-subject amplitude information is
    not erased by per-batch feature normalization.
    """

    def __init__(self, input_channels: int, feature_dim: int,
                 kernels: Sequence[int]) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        kernel_values = tuple(int(value) for value in kernels)
        if not kernel_values or any(value < 1 or value % 2 == 0 for value in kernel_values):
            raise ValueError("moment kernels must be non-empty positive odd integers")
        branch_dim = max(8, feature_dim // 2)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(input_channels, branch_dim, kernel_size=kernel,
                              padding=kernel // 2, bias=True),
                    nn.GELU(),
                )
                for kernel in kernel_values
            ]
        )
        self.mix = nn.Sequential(
            nn.Conv1d(branch_dim * len(kernel_values), feature_dim, 1, bias=True),
            nn.GELU(),
        )
        self.projection = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.mix(torch.cat([branch(inputs) for branch in self.branches], dim=1))
        mean = features.mean(dim=-1)
        variance = torch.mean((features - mean.unsqueeze(-1)).square(), dim=-1)
        std = torch.sqrt(variance + 1.0e-5)
        return {
            "embedding": self.projection(torch.cat([mean, std], dim=-1)),
            "moment_features": features,
        }


class LearnableRelativeEnergyEncoder(nn.Module):
    """Learn shared temporal filters and pool only their relative energies.

    Unlike handcrafted band powers, the filter coefficients are optimized end to
    end. Unit-norm, zero-mean filters and within-channel log-energy normalization
    prevent this branch from duplicating the absolute-amplitude cues already
    represented by the normalization-free moment branch.
    """

    def __init__(
        self,
        input_channels: int,
        feature_dim: int,
        filter_count: int,
        kernel_size: int,
        sample_rate: float,
        initial_minimum_hz: float,
        initial_maximum_hz: float,
    ) -> None:
        super().__init__()
        if filter_count < 2:
            raise ValueError("relative-energy filter_count must be at least two")
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("relative-energy kernel_size must be odd and >= 3")
        nyquist = float(sample_rate) / 2.0
        if not 0.0 < initial_minimum_hz < initial_maximum_hz <= nyquist:
            raise ValueError("Invalid relative-energy initialization frequencies")
        self.input_channels = int(input_channels)
        self.feature_dim = int(feature_dim)
        self.filter_count = int(filter_count)
        self.kernel_size = int(kernel_size)
        self.weight = nn.Parameter(torch.empty(filter_count, 1, kernel_size))
        self.projection = nn.Sequential(
            nn.Linear(input_channels * filter_count, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )
        time = (
            torch.arange(kernel_size, dtype=torch.float32)
            - (kernel_size - 1) / 2.0
        ) / float(sample_rate)
        frequencies = torch.linspace(
            float(initial_minimum_hz), float(initial_maximum_hz), filter_count
        )
        window = torch.hann_window(kernel_size, periodic=False)
        initialized = torch.cos(2.0 * torch.pi * frequencies[:, None] * time[None, :])
        initialized = initialized * window
        initialized = initialized - initialized.mean(dim=-1, keepdim=True)
        initialized = initialized / initialized.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        with torch.no_grad():
            self.weight.copy_(initialized.unsqueeze(1))

    def normalized_filters(self) -> torch.Tensor:
        centered = self.weight - self.weight.mean(dim=-1, keepdim=True)
        return centered / centered.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if inputs.ndim != 3 or inputs.shape[1] != self.input_channels:
            raise ValueError("Relative-energy encoder expects [B,C,T]")
        batch_size, channels, length = inputs.shape
        responses = functional.conv1d(
            inputs.reshape(batch_size * channels, 1, length),
            self.normalized_filters(),
            padding=self.kernel_size // 2,
        )
        log_energy = torch.log(
            responses.square().mean(dim=-1).clamp_min(1.0e-8)
        ).reshape(batch_size, channels, self.filter_count)
        relative_log_energy = log_energy - torch.logsumexp(
            log_energy, dim=-1, keepdim=True
        )
        embedding = self.projection(relative_log_energy.flatten(start_dim=1))
        return {
            "embedding": embedding,
            "relative_log_energy": relative_log_energy,
        }


class MomentOnlyWristEncoder(nn.Module):
    """Adapter exposing the learned-moment branch as the complete wrist encoder."""

    def __init__(self, input_channels: int, feature_dim: int,
                 kernels: Sequence[int]) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.encoder = NormFreeMomentEncoder(input_channels, feature_dim, kernels)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.encoder(inputs)
        return {
            "bag_embedding": output["embedding"],
            "encoded_features": output["moment_features"],
            **output,
        }


class PureDeepWristEncoder(nn.Module):
    def __init__(self, input_channels: int, feature_dim: int, dropout: float,
                 pooling: str, stem_mode: str = "shared", *,
                 sample_rate: float = 100.0,
                 spectral_config: Mapping[str, Any] | None = None,
                 moment_config: Mapping[str, Any] | None = None,
                 relative_energy_config: Mapping[str, Any] | None = None,
                 temporal_config: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.stem_mode = str(stem_mode)
        temporal = dict(temporal_config or {})
        temporal_normalization = str(temporal.get("normalization", "batch_norm"))
        temporal_norm_groups = int(temporal.get("normalization_groups", 8))
        if temporal_norm_groups < 1:
            raise ValueError("pure_deep.temporal.normalization_groups must be positive")

        def make_stem(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=15, stride=2,
                          padding=7, bias=False),
                make_temporal_normalization(
                    out_channels, temporal_normalization, temporal_norm_groups
                ),
                nn.GELU(),
            )

        if self.stem_mode == "shared":
            self.stem = make_stem(input_channels, feature_dim)
        elif self.stem_mode == "split_acc_gyro":
            if input_channels != 6 or feature_dim % 2:
                raise ValueError("split_acc_gyro requires 6 inputs and an even feature_dim")
            self.acc_stem = make_stem(3, feature_dim // 2)
            self.gyro_stem = make_stem(3, feature_dim // 2)
        else:
            raise ValueError(f"Unsupported stem_mode: {self.stem_mode!r}")
        temporal_kernel = int(temporal.get("kernel_size", 7))
        temporal_dilations = tuple(
            int(value) for value in temporal.get("dilations", [1, 2, 4])
        )
        if temporal_kernel < 1 or temporal_kernel % 2 == 0:
            raise ValueError("pure_deep.temporal.kernel_size must be a positive odd integer")
        if not temporal_dilations or any(value < 1 for value in temporal_dilations):
            raise ValueError("pure_deep.temporal.dilations must be positive integers")
        self.temporal_kernel = temporal_kernel
        self.temporal_dilations = temporal_dilations
        self.encoder = nn.Sequential(
            *[
                SeparableResidualBlock(
                    feature_dim, temporal_kernel, dilation, dropout,
                    temporal_normalization, temporal_norm_groups
                )
                for dilation in temporal_dilations
            ]
        )
        self.pool = FeatureStatisticsPool(feature_dim, pooling)
        spectral = dict(spectral_config or {})
        self.spectral_encoder: LearnableSpectrumEncoder | None = None
        if bool(spectral.get("enabled", False)):
            spectral_dim = int(spectral.get("feature_dim", feature_dim // 2))
            self.spectral_encoder = LearnableSpectrumEncoder(
                input_channels=input_channels,
                feature_dim=spectral_dim,
                sample_rate=sample_rate,
                frequency_bins=int(spectral.get("frequency_bins", 256)),
                maximum_hz=float(spectral.get("maximum_hz", 20.0)),
                dropout=float(spectral.get("dropout", dropout)),
            )
            self.time_spectral_fusion = nn.Sequential(
                nn.Linear(feature_dim + spectral_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
            )
        moment = dict(moment_config or {})
        self.moment_encoder: NormFreeMomentEncoder | None = None
        if bool(moment.get("enabled", False)):
            moment_dim = int(moment.get("feature_dim", feature_dim // 2))
            self.moment_encoder = NormFreeMomentEncoder(
                input_channels=input_channels,
                feature_dim=moment_dim,
                kernels=moment.get("kernels", [1, 15, 63]),
            )
            self.time_moment_fusion = nn.Sequential(
                nn.Linear(feature_dim + moment_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
            )
        relative = dict(relative_energy_config or {})
        self.relative_energy_encoder: LearnableRelativeEnergyEncoder | None = None
        if bool(relative.get("enabled", False)):
            relative_dim = int(relative.get("feature_dim", feature_dim // 2))
            self.relative_energy_encoder = LearnableRelativeEnergyEncoder(
                input_channels=input_channels,
                feature_dim=relative_dim,
                filter_count=int(relative.get("filter_count", 8)),
                kernel_size=int(relative.get("kernel_size", 101)),
                sample_rate=sample_rate,
                initial_minimum_hz=float(relative.get("initial_minimum_hz", 0.5)),
                initial_maximum_hz=float(relative.get("initial_maximum_hz", 20.0)),
            )
            self.time_relative_energy_fusion = nn.Sequential(
                nn.Linear(feature_dim + relative_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
            )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.stem_mode == "shared":
            stem_features = self.stem(inputs)
        else:
            stem_features = torch.cat(
                [self.acc_stem(inputs[:, :3]), self.gyro_stem(inputs[:, 3:])], dim=1
            )
        features = self.encoder(stem_features)
        pooled = self.pool(features)
        embedding = pooled["embedding"]
        output = {"encoded_features": features, **pooled}
        if self.spectral_encoder is not None:
            spectral = self.spectral_encoder(inputs)
            embedding = self.time_spectral_fusion(
                torch.cat([embedding, spectral["embedding"]], dim=-1)
            )
            output.update(spectral)
        if self.moment_encoder is not None:
            moment = self.moment_encoder(inputs)
            embedding = self.time_moment_fusion(
                torch.cat([embedding, moment["embedding"]], dim=-1)
            )
            output.update(moment)
        if self.relative_energy_encoder is not None:
            relative = self.relative_energy_encoder(inputs)
            embedding = self.time_relative_energy_fusion(
                torch.cat([embedding, relative["embedding"]], dim=-1)
            )
            output.update({
                "relative_energy_embedding": relative["embedding"],
                "relative_log_energy": relative["relative_log_energy"],
            })
        return {"bag_embedding": embedding, **output}


class ResidualBasicBlock1D(nn.Module):
    """Standard two-convolution residual block adapted to 1-D sensor data."""

    def __init__(self, in_channels: int, out_channels: int, stride: int,
                 kernel_size: int, dropout: float, group_count: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                      padding=padding, bias=False),
            make_temporal_normalization(out_channels, "group_norm", group_count),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding,
                      bias=False),
            make_temporal_normalization(out_channels, "group_norm", group_count),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                make_temporal_normalization(out_channels, "group_norm", group_count),
            )
        )
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.block(inputs) + self.shortcut(inputs))


class ResNet1DWristEncoder(nn.Module):
    """Compact, conventional 1-D ResNet wrist encoder with global average pooling."""

    def __init__(self, input_channels: int, feature_dim: int,
                 config: Mapping[str, Any], dropout: float) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        channels = tuple(int(value) for value in config.get("stage_channels", [32, 64, 64]))
        blocks = tuple(int(value) for value in config.get("blocks_per_stage", [2, 2, 2]))
        if len(channels) != len(blocks) or not channels or any(value < 1 for value in blocks):
            raise ValueError("resnet1d stages and blocks_per_stage must be aligned and positive")
        kernel_size = int(config.get("kernel_size", 7))
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("resnet1d.kernel_size must be odd and >=3")
        groups = int(config.get("normalization_groups", 8))
        stem_kernel = int(config.get("stem_kernel_size", 15))
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, channels[0], stem_kernel, stride=2,
                      padding=stem_kernel // 2, bias=False),
            make_temporal_normalization(channels[0], "group_norm", groups),
            nn.GELU(),
        )
        stages = []
        current = channels[0]
        for stage_index, (out_channels, block_count) in enumerate(zip(channels, blocks)):
            for block_index in range(block_count):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                stages.append(ResidualBasicBlock1D(
                    current, out_channels, stride, kernel_size, dropout, groups
                ))
                current = out_channels
        self.encoder = nn.Sequential(*stages)
        self.projection = nn.Sequential(
            nn.Linear(current, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(self.stem(inputs))
        embedding = self.projection(features.mean(dim=-1))
        return {"bag_embedding": embedding, "encoded_features": features}


class InceptionModule1D(nn.Module):
    """InceptionTime-style multi-scale temporal module."""

    def __init__(self, in_channels: int, bottleneck_channels: int,
                 branch_channels: int, kernel_sizes: Sequence[int],
                 dropout: float, group_count: int) -> None:
        super().__init__()
        self.bottleneck = nn.Conv1d(
            in_channels, bottleneck_channels, 1, bias=False
        ) if in_channels > 1 else nn.Identity()
        bottleneck_output = bottleneck_channels if in_channels > 1 else in_channels
        self.convolutions = nn.ModuleList([
            nn.Conv1d(bottleneck_output, branch_channels, int(kernel),
                      padding=int(kernel) // 2, bias=False)
            for kernel in kernel_sizes
        ])
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, 1, bias=False),
        )
        out_channels = branch_channels * (len(kernel_sizes) + 1)
        self.normalization = make_temporal_normalization(
            out_channels, "group_norm", group_count
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out_channels = out_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        bottleneck = self.bottleneck(inputs)
        branches = [convolution(bottleneck) for convolution in self.convolutions]
        branches.append(self.pool_branch(inputs))
        return self.dropout(self.activation(self.normalization(torch.cat(branches, dim=1))))


class InceptionResidualStack1D(nn.Module):
    def __init__(self, in_channels: int, module_count: int,
                 bottleneck_channels: int, branch_channels: int,
                 kernel_sizes: Sequence[int], dropout: float,
                 group_count: int) -> None:
        super().__init__()
        modules = []
        current = in_channels
        for _ in range(module_count):
            module = InceptionModule1D(
                current, bottleneck_channels, branch_channels, kernel_sizes,
                dropout, group_count,
            )
            modules.append(module)
            current = module.out_channels
        self.modules_list = nn.ModuleList(modules)
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, current, 1, bias=False),
            make_temporal_normalization(current, "group_norm", group_count),
        ) if in_channels != current else nn.Identity()
        self.activation = nn.GELU()
        self.out_channels = current

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = inputs
        for module in self.modules_list:
            features = module(features)
        return self.activation(features + self.shortcut(inputs))


class InceptionTimeWristEncoder(nn.Module):
    """Two residual stacks of three InceptionTime modules each."""

    def __init__(self, input_channels: int, feature_dim: int,
                 config: Mapping[str, Any], dropout: float) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        bottleneck = int(config.get("bottleneck_channels", 32))
        branch = int(config.get("branch_channels", 32))
        kernels = tuple(int(value) for value in config.get("kernel_sizes", [39, 19, 9]))
        if len(kernels) != 3 or any(value < 3 or value % 2 == 0 for value in kernels):
            raise ValueError("inception_time.kernel_sizes must contain three odd values >=3")
        groups = int(config.get("normalization_groups", 8))
        stack_count = int(config.get("residual_stack_count", 2))
        modules_per_stack = int(config.get("modules_per_stack", 3))
        stacks = []
        current = input_channels
        for _ in range(stack_count):
            stack = InceptionResidualStack1D(
                current, modules_per_stack, bottleneck, branch, kernels,
                dropout, groups,
            )
            stacks.append(stack)
            current = stack.out_channels
        self.encoder = nn.Sequential(*stacks)
        self.projection = nn.Sequential(
            nn.Linear(current, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(inputs)
        embedding = self.projection(features.mean(dim=-1))
        return {"bag_embedding": embedding, "encoded_features": features}


class GyroLearnableSpectrumEncoder(nn.Module):
    """Lightweight differentiable spectrum encoder restricted to Gyro XYZ."""

    def __init__(self, sample_rate: float, frequency_bins: int,
                 maximum_hz: float, hidden_dim: int, embedding_dim: int,
                 dropout: float, group_count: int) -> None:
        super().__init__()
        if frequency_bins < 32:
            raise ValueError("gyro frequency_bins must be >=32")
        if not 0.0 < maximum_hz <= sample_rate / 2.0:
            raise ValueError("gyro maximum_hz must be in (0, Nyquist]")
        self.sample_rate = float(sample_rate)
        self.frequency_bins = int(frequency_bins)
        self.maximum_hz = float(maximum_hz)
        self.encoder = nn.Sequential(
            nn.Conv1d(3, hidden_dim, 7, padding=3, bias=False),
            make_temporal_normalization(hidden_dim, "group_norm", group_count),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 5, stride=2, padding=2,
                      groups=hidden_dim, bias=False),
            make_temporal_normalization(hidden_dim, "group_norm", group_count),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1, bias=False),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

    def spectrum(self, gyro: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(gyro, dim=-1, norm="forward")
        frequencies = torch.fft.rfftfreq(
            gyro.shape[-1], d=1.0 / self.sample_rate, device=gyro.device
        )
        magnitude = torch.log1p(torch.abs(spectrum[..., frequencies <= self.maximum_hz]))
        return functional.interpolate(
            magnitude, size=self.frequency_bins, mode="linear", align_corners=False
        )

    def forward(self, gyro: torch.Tensor) -> dict[str, torch.Tensor]:
        if gyro.ndim != 3 or gyro.shape[1] != 3:
            raise ValueError("GyroLearnableSpectrumEncoder expects [B,3,T]")
        features = self.encoder(self.spectrum(gyro))
        mean = features.mean(dim=-1)
        variance = (features - mean.unsqueeze(-1)).square().mean(dim=-1)
        embedding = self.projection(torch.cat([mean, torch.sqrt(variance + 1.0e-5)], dim=-1))
        return {"embedding": embedding, "gyro_spectral_features": features}


class FrequencyAugmentedWristEncoder(nn.Module):
    """Add one preregistered Gyro-only spectral branch to a temporal encoder."""

    def __init__(self, temporal_encoder: nn.Module, feature_dim: int,
                 sample_rate: float, config: Mapping[str, Any],
                 dropout: float) -> None:
        super().__init__()
        self.temporal_encoder = temporal_encoder
        self.feature_dim = int(feature_dim)
        spectral_dim = int(config.get("embedding_dim", 16))
        self.spectral_encoder = GyroLearnableSpectrumEncoder(
            sample_rate=sample_rate,
            frequency_bins=int(config.get("frequency_bins", 128)),
            maximum_hz=float(config.get("maximum_hz", 12.0)),
            hidden_dim=int(config.get("hidden_dim", 24)),
            embedding_dim=spectral_dim,
            dropout=float(config.get("dropout", dropout)),
            group_count=int(config.get("normalization_groups", 8)),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.feature_dim + spectral_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        temporal = self.temporal_encoder(inputs)
        spectral = self.spectral_encoder(inputs[:, 3:6])
        embedding = self.fusion(torch.cat([
            temporal["bag_embedding"], spectral["embedding"]
        ], dim=-1))
        return {**temporal, **spectral, "bag_embedding": embedding}


class StructuredTokenResidualReadout(nn.Module):
    """Lightweight residual decision path over ordered activity tokens."""

    def __init__(self, activity_dim: int, activity_count: int,
                 projection_dim: int, num_classes: int) -> None:
        super().__init__()
        self.activity_count = int(activity_count)
        self.projection_dim = int(projection_dim)
        self.projection = nn.Sequential(
            nn.Linear(activity_dim, projection_dim),
            nn.GELU(),
        )
        self.residual_head = nn.Linear(
            self.activity_count * self.projection_dim, num_classes
        )
        # STR-01 starts from the matched V8 decision function. The residual
        # path learns departures without perturbing the initial logits.
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, activity_features: torch.Tensor,
                activity_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if activity_features.ndim != 3:
            raise ValueError("Expected activity features shaped [B,A,D]")
        if activity_features.shape[1] != self.activity_count:
            raise ValueError(
                f"Expected {self.activity_count} ordered activity tokens"
            )
        structured_tokens = self.projection(activity_features)
        structured_tokens = structured_tokens * activity_mask.to(
            device=structured_tokens.device, dtype=structured_tokens.dtype
        ).unsqueeze(-1)
        structured_embedding = structured_tokens.reshape(
            structured_tokens.shape[0], -1
        )
        return {
            "structured_residual_logits": self.residual_head(
                structured_embedding
            ),
            "structured_activity_tokens": structured_tokens,
            "structured_residual_embedding": structured_embedding,
        }


class WristStructuredResidualReadout(nn.Module):
    """Residual decision path over fixed activity-by-wrist token structure."""

    def __init__(self, wrist_dim: int, activity_count: int,
                 projection_dim: int, num_classes: int) -> None:
        super().__init__()
        self.activity_count = int(activity_count)
        self.wrist_count = 2
        self.projection_dim = int(projection_dim)
        self.projection = nn.Sequential(
            nn.Linear(wrist_dim, projection_dim),
            nn.GELU(),
        )
        self.residual_head = nn.Linear(
            self.activity_count * self.wrist_count * self.projection_dim,
            num_classes,
        )
        # WSR-01 starts from the complete STR-01 decision function.
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, wrist_embeddings: torch.Tensor,
                wrist_mask: torch.Tensor,
                activity_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if wrist_embeddings.ndim != 4:
            raise ValueError("Expected wrist embeddings shaped [B,A,2,D]")
        if wrist_embeddings.shape[1:3] != (self.activity_count, self.wrist_count):
            raise ValueError(
                f"Expected {self.activity_count} activities and two wrists"
            )
        mask = wrist_mask.to(device=wrist_embeddings.device, dtype=torch.bool)
        mask = mask & activity_mask.to(
            device=wrist_embeddings.device, dtype=torch.bool
        ).unsqueeze(-1)
        tokens = self.projection(wrist_embeddings)
        tokens = tokens * mask.to(dtype=tokens.dtype).unsqueeze(-1)
        embedding = tokens.reshape(tokens.shape[0], -1)
        return {
            "wrist_structured_residual_logits": self.residual_head(embedding),
            "wrist_structured_tokens": tokens,
            "wrist_structured_embedding": embedding,
        }


class PureDeepSubjectModel(nn.Module):
    """Pure neural bilateral multi-activity PADS classifier."""

    def __init__(self, activities: Sequence[str], input_channels: int,
                 num_classes: int, model_config: Mapping[str, Any]) -> None:
        super().__init__()
        self.activities = tuple(str(value) for value in activities)
        self.activity_count = len(self.activities)
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        pure = model_config.get("pure_deep", {})
        feature_dim = int(pure.get("feature_dim", 128))
        encoder_mode = str(pure.get("encoder_mode", "time"))
        if encoder_mode == "moment_only":
            moment = dict(pure.get("moment_branch", {}))
            if not bool(moment.get("enabled", False)):
                raise ValueError("moment_only requires pure_deep.moment_branch.enabled")
            self.wrist_encoder = MomentOnlyWristEncoder(
                input_channels=input_channels,
                feature_dim=feature_dim,
                kernels=moment.get("kernels", [1, 15, 63]),
            )
        elif encoder_mode == "time":
            self.wrist_encoder = PureDeepWristEncoder(
                input_channels, feature_dim, float(pure.get("dropout", 0.1)),
                str(pure.get("pooling", "attention_mean_std")),
                str(pure.get("stem_mode", "shared")),
                sample_rate=float(model_config.get("sample_rate", 100.0)),
                spectral_config=pure.get("spectral"),
                moment_config=pure.get("moment_branch"),
                relative_energy_config=pure.get("relative_energy_branch"),
                temporal_config=pure.get("temporal"),
            )
        elif encoder_mode == "resnet1d":
            self.wrist_encoder = ResNet1DWristEncoder(
                input_channels, feature_dim, pure.get("resnet1d", {}),
                float(pure.get("dropout", 0.1)),
            )
        elif encoder_mode == "inception_time":
            self.wrist_encoder = InceptionTimeWristEncoder(
                input_channels, feature_dim, pure.get("inception_time", {}),
                float(pure.get("dropout", 0.1)),
            )
        else:
            raise ValueError(f"Unsupported pure_deep.encoder_mode: {encoder_mode!r}")
        gyro_frequency = dict(pure.get("gyro_frequency_branch", {}))
        if bool(gyro_frequency.get("enabled", False)):
            if input_channels != 6:
                raise ValueError("Gyro frequency branch requires Acc XYZ + Gyro XYZ")
            self.wrist_encoder = FrequencyAugmentedWristEncoder(
                self.wrist_encoder, feature_dim,
                float(model_config.get("sample_rate", 100.0)),
                gyro_frequency, float(pure.get("dropout", 0.1)),
            )
        self.wrist_fusion = BilateralFusionHead(feature_dim, num_classes, dropout=0.0)
        self.wrist_fusion.classifier = nn.Identity()
        bilateral = dict(model_config.get("bilateral_fusion", {}))
        self.bilateral_fusion_mode = str(bilateral.get("mode", "full"))
        if self.bilateral_fusion_mode == "full":
            fusion_dim = self.wrist_fusion.fusion_dim
        elif self.bilateral_fusion_mode == "mean_only":
            fusion_dim = feature_dim
        else:
            raise ValueError(
                "model.bilateral_fusion.mode must be 'full' or 'mean_only'"
            )
        self.activity_feature_dim = int(fusion_dim)
        activity = model_config.get("activity_fusion", {})
        aggregation = str(activity.get("mode", "attention"))
        if aggregation == "attention":
            self.activity_aggregator = ActivityAttentionAggregator(
                feature_dim=fusion_dim,
                attention_dim=int(activity.get("attention_dim", 128)),
                activity_count=self.activity_count,
                dropout=float(activity.get("dropout", 0.1)),
            )
        elif aggregation == "masked_mean":
            self.activity_aggregator = MaskedMeanActivityAggregator()
        elif aggregation == "mean_attention_residual":
            self.activity_aggregator = MeanAttentionResidualAggregator(
                feature_dim=fusion_dim,
                attention_dim=int(activity.get("attention_dim", 128)),
                activity_count=self.activity_count,
                dropout=float(activity.get("dropout", 0.1)),
                initial_attention_weight=float(
                    activity.get("initial_attention_weight", 0.25)
                ),
            )
        else:
            raise ValueError(f"Unsupported activity_fusion.mode: {aggregation!r}")
        classifier_dropout = float(activity.get("classifier_dropout", 0.2))
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout), nn.Linear(fusion_dim, num_classes)
        )
        structured = dict(model_config.get("structured_token_residual", {}))
        self.structured_token_residual = None
        if bool(structured.get("enabled", False)):
            self.structured_token_residual = StructuredTokenResidualReadout(
                activity_dim=fusion_dim,
                activity_count=self.activity_count,
                projection_dim=int(structured.get("projection_dim", 16)),
                num_classes=num_classes,
            )
        wrist_structured = dict(
            model_config.get("wrist_structured_residual", {})
        )
        self.wrist_structured_residual = None
        if bool(wrist_structured.get("enabled", False)):
            self.wrist_structured_residual = WristStructuredResidualReadout(
                wrist_dim=self.wrist_encoder.feature_dim,
                activity_count=self.activity_count,
                projection_dim=int(wrist_structured.get("projection_dim", 4)),
                num_classes=num_classes,
            )

    def _encode_activities_with_wrists(
        self, inputs: torch.Tensor, wrist_mask: torch.Tensor,
        activity_mask: torch.Tensor, activity_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = inputs.shape[0]
        flat_inputs = inputs.reshape(batch_size * self.activity_count, 2,
                                     inputs.shape[3], inputs.shape[4])
        flat_wrist = wrist_mask.reshape(-1, 2)
        flat_lengths = activity_lengths.reshape(-1)
        valid_activities = activity_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
        result = inputs.new_zeros((batch_size * self.activity_count,
                                  self.activity_feature_dim))
        wrist_result = inputs.new_zeros((
            batch_size * self.activity_count, 2,
            self.wrist_encoder.feature_dim,
        ))
        for length in torch.unique(flat_lengths.index_select(0, valid_activities)).tolist():
            activity_indices = valid_activities[
                flat_lengths.index_select(0, valid_activities) == int(length)
            ]
            selected = flat_inputs.index_select(0, activity_indices)[..., : int(length)]
            selected_wrist = flat_wrist.index_select(0, activity_indices)
            wrist_flat = selected.reshape(-1, selected.shape[2], selected.shape[3])
            valid_wrist = selected_wrist.reshape(-1).bool()
            wrist_indices = valid_wrist.nonzero(as_tuple=False).squeeze(1)
            encoded = self.wrist_encoder(wrist_flat.index_select(0, wrist_indices))
            wrist_embeddings = selected.new_zeros((selected.shape[0] * 2,
                                                   self.wrist_encoder.feature_dim))
            wrist_embeddings = wrist_embeddings.index_copy(
                0, wrist_indices, encoded["bag_embedding"]
            ).reshape(selected.shape[0], 2, -1)
            wrist_result = wrist_result.index_copy(
                0, activity_indices, wrist_embeddings
            )
            if self.bilateral_fusion_mode == "full":
                fusion, _ = self.wrist_fusion.fusion_features(
                    wrist_embeddings, selected_wrist
                )
            else:
                mask = selected_wrist.to(
                    device=wrist_embeddings.device, dtype=wrist_embeddings.dtype
                )
                masked = wrist_embeddings * mask.unsqueeze(-1)
                fusion = masked.sum(dim=1) / mask.sum(
                    dim=1, keepdim=True
                ).clamp_min(1.0)
            result = result.index_copy(0, activity_indices, fusion)
        return (
            result.reshape(batch_size, self.activity_count, -1),
            wrist_result.reshape(batch_size, self.activity_count, 2, -1),
        )

    def _encode_activities(self, inputs: torch.Tensor, wrist_mask: torch.Tensor,
                           activity_mask: torch.Tensor,
                           activity_lengths: torch.Tensor) -> torch.Tensor:
        return self._encode_activities_with_wrists(
            inputs, wrist_mask, activity_mask, activity_lengths
        )[0]

    def forward(self, inputs: torch.Tensor, wrist_mask: torch.Tensor,
                activity_mask: torch.Tensor | None = None,
                activity_lengths: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if inputs.ndim != 5 or inputs.shape[1] != self.activity_count:
            raise ValueError(f"Expected [B,{self.activity_count},2,C,T]")
        batch_size = inputs.shape[0]
        if activity_mask is None:
            activity_mask = torch.ones((batch_size, self.activity_count),
                                       dtype=torch.bool, device=inputs.device)
        activity_mask = activity_mask.to(device=inputs.device, dtype=torch.bool)
        if activity_lengths is None:
            activity_lengths = activity_mask.to(torch.long) * inputs.shape[-1]
        activity_lengths = activity_lengths.to(device=inputs.device, dtype=torch.long)
        if not torch.isfinite(inputs[activity_mask]).all():
            raise ValueError("Valid activity input contains NaN or infinite values")
        activity_features, pre_fusion_wrists = self._encode_activities_with_wrists(
            inputs, wrist_mask, activity_mask, activity_lengths
        )
        aggregated = self.activity_aggregator(activity_features, activity_mask)
        base_logits = self.classifier(aggregated["subject_embedding"])
        residual = {}
        logits = base_logits
        if self.structured_token_residual is not None:
            residual = self.structured_token_residual(
                activity_features, activity_mask
            )
            logits = logits + residual["structured_residual_logits"]
        wrist_residual = {}
        if self.wrist_structured_residual is not None:
            wrist_residual = self.wrist_structured_residual(
                pre_fusion_wrists, wrist_mask, activity_mask
            )
            logits = logits + wrist_residual[
                "wrist_structured_residual_logits"
            ]
        return {
            "logits": logits,
            "base_logits": base_logits,
            "probabilities": torch.softmax(logits, dim=-1),
            "bag_embedding": aggregated["subject_embedding"],
            "activity_mask": activity_mask,
            **residual,
            **wrist_residual,
            **aggregated,
        }


def build_pure_deep_subject(config: Mapping[str, Any]) -> PureDeepSubjectModel:
    model = config["model"]
    return PureDeepSubjectModel(
        activities=config["data"]["activities"],
        input_channels=int(model["input_channels"]),
        num_classes=int(model["num_classes"]),
        model_config=model,
    )
