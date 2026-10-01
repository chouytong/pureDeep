from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class FrequencyDecomposition(nn.Module):
    """基于可配置频带的可微频域分解。

    Input shape: ``[batch, channels, time]``.
    Output shape: ``[batch, channels * number_of_streams, time]``.
    """

    def __init__(
        self,
        sample_rate: float,
        bands: Sequence[Sequence[float]],
        boundary: str = "half_open",
        include_last_high: bool = True,
        include_original: bool = False,
    ) -> None:
        super().__init__()
        self.sample_rate = float(sample_rate)
        self.bands = tuple((float(low), float(high)) for low, high in bands)
        self.boundary = boundary
        self.include_last_high = bool(include_last_high)
        self.include_original = bool(include_original)
        self._validate()

    @property
    def output_multiplier(self) -> int:
        return len(self.bands) + int(self.include_original)

    def _validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.boundary not in {"half_open", "closed"}:
            raise ValueError("boundary must be 'half_open' or 'closed'")
        nyquist = self.sample_rate / 2.0
        for low, high in self.bands:
            if not 0 <= low < high <= nyquist:
                raise ValueError(
                    f"Invalid band [{low}, {high}] for Nyquist {nyquist}"
                )

    def _mask(
        self,
        frequencies: torch.Tensor,
        low: float,
        high: float,
        is_last: bool,
    ) -> torch.Tensor:
        if self.boundary == "closed":
            return (frequencies >= low) & (frequencies <= high)
        include_high = is_last and self.include_last_high
        if include_high:
            return (frequencies >= low) & (frequencies <= high)
        return (frequencies >= low) & (frequencies < high)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(
                f"Expected [batch, channels, time], got {tuple(inputs.shape)}"
            )
        time_length = inputs.shape[-1]
        if time_length < 2:
            raise ValueError("Frequency decomposition requires at least 2 time points")
        spectrum = torch.fft.rfft(inputs, dim=-1)
        frequencies = torch.fft.rfftfreq(
            time_length,
            d=1.0 / self.sample_rate,
            device=inputs.device,
        )

        streams: list[torch.Tensor] = []
        if self.include_original:
            streams.append(inputs)
        for index, (low, high) in enumerate(self.bands):
            mask = self._mask(
                frequencies,
                low,
                high,
                is_last=index == len(self.bands) - 1,
            )
            filtered = spectrum * mask.to(dtype=spectrum.dtype).view(1, 1, -1)
            reconstructed = torch.fft.irfft(filtered, n=time_length, dim=-1)
            streams.append(reconstructed)

        if not streams:
            raise RuntimeError("Frequency decomposition produced no streams")
        return torch.cat(streams, dim=1)
