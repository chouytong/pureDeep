import math

import torch

from src.models.frequency import FrequencyDecomposition


def test_frequency_decomposition_shape_and_band_selectivity() -> None:
    sample_rate = 64.0
    length = 256
    time = torch.arange(length) / sample_rate
    signal = torch.sin(2 * math.pi * 2.0 * time) + 0.5 * torch.sin(
        2 * math.pi * 9.0 * time
    )
    module = FrequencyDecomposition(
        sample_rate=sample_rate,
        bands=[[0.5, 3.0], [3.0, 7.0], [7.0, 12.0]],
    )
    output = module(signal.view(1, 1, -1))
    assert output.shape == (1, 3, length)
    rms = output.square().mean(dim=-1).sqrt().squeeze(0)
    assert rms[0] > rms[1] * 20
    assert rms[2] > rms[1] * 10


def test_frequency_decomposition_rejects_bad_shape() -> None:
    module = FrequencyDecomposition(64.0, [[0.5, 3.0]])
    try:
        module(torch.zeros(2, 64))
    except ValueError as error:
        assert "Expected" in str(error)
    else:
        raise AssertionError("Expected a shape validation error")
