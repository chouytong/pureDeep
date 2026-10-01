from __future__ import annotations

import numpy as np

from src.datasets.preprocessing import (
    l1_trend_filter_batch,
    remove_acceleration_trend,
)


def test_l1_trend_filter_batch_preserves_shape_and_converges() -> None:
    time = np.linspace(0.0, 1.0, 128)
    signals = np.stack((time, 2.0 * time + 1.0))
    trend, report = l1_trend_filter_batch(
        signals,
        regularization=50.0,
        max_iterations=2000,
        absolute_tolerance=1e-5,
        relative_tolerance=1e-5,
    )
    assert report.converged
    assert trend.shape == signals.shape
    np.testing.assert_allclose(trend, signals, atol=2e-4)


def test_acceleration_is_detrended_and_gyroscope_is_unchanged() -> None:
    time = np.linspace(0.0, 1.0, 128)
    acceleration = np.stack((time, 2.0 * time, -time))
    gyroscope = np.stack((np.sin(time), np.cos(time), time**2))
    signal = np.concatenate((acceleration, gyroscope), axis=0)
    processed, report = remove_acceleration_trend(
        signal,
        regularization=50.0,
        max_iterations=2000,
        absolute_tolerance=1e-5,
        relative_tolerance=1e-5,
    )
    assert report.converged
    np.testing.assert_allclose(processed[:3], 0.0, atol=2e-4)
    np.testing.assert_array_equal(processed[3:], gyroscope)
