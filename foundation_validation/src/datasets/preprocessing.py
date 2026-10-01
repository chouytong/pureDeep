from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized


@dataclass(frozen=True)
class TrendFilterReport:
    converged: bool
    iterations: int
    primal_residual: float
    dual_residual: float
    primal_tolerance: float
    dual_tolerance: float


def _second_difference(length: int) -> sparse.csc_matrix:
    if length < 3:
        raise ValueError("L1 trend filtering requires at least 3 samples")
    return sparse.diags(
        diagonals=(np.ones(length - 2), -2.0 * np.ones(length - 2), np.ones(length - 2)),
        offsets=(0, 1, 2),
        shape=(length - 2, length),
        format="csc",
    )


@lru_cache(maxsize=16)
def _linear_system(length: int, rho: float):
    difference = _second_difference(length)
    system = sparse.eye(length, format="csc") + float(rho) * (
        difference.T @ difference
    )
    return difference, factorized(system)


def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - float(threshold), 0.0)


def l1_trend_filter_batch(
    signals: np.ndarray,
    regularization: float = 50.0,
    *,
    rho: float = 1.0,
    max_iterations: int = 2000,
    absolute_tolerance: float = 1e-4,
    relative_tolerance: float = 1e-4,
) -> tuple[np.ndarray, TrendFilterReport]:
    """Solve the PADS L1 trend-filter objective for a batch of 1-D signals.

    The optimized objective is ``0.5*||y-x||_2^2 + lambda*||D2*x||_1``,
    matching the official PADS CVXPY formulation.  Rows are independent signals;
    batching only reuses the same sparse factorization.
    """
    values = np.asarray(signals, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError(f"signals must be [series,time], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("signals contain NaN or infinite values")
    if regularization < 0 or rho <= 0:
        raise ValueError("regularization must be non-negative and rho must be positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    series_count, length = values.shape
    difference, solve = _linear_system(length, float(rho))
    observations = values.T
    transformed_count = length - 2
    auxiliary = np.zeros((transformed_count, series_count), dtype=np.float64)
    scaled_dual = np.zeros_like(auxiliary)
    trend = observations.copy()
    report = TrendFilterReport(False, 0, np.inf, np.inf, np.inf, np.inf)

    for iteration in range(1, max_iterations + 1):
        right_hand_side = observations + float(rho) * (
            difference.T @ (auxiliary - scaled_dual)
        )
        trend = solve(right_hand_side)
        second_difference = difference @ trend
        previous_auxiliary = auxiliary
        auxiliary = _soft_threshold(
            second_difference + scaled_dual, float(regularization) / float(rho)
        )
        scaled_dual = scaled_dual + second_difference - auxiliary

        primal = float(np.linalg.norm(second_difference - auxiliary))
        dual = float(
            float(rho)
            * np.linalg.norm(difference.T @ (auxiliary - previous_auxiliary))
        )
        primal_tolerance = float(
            np.sqrt(transformed_count * series_count) * absolute_tolerance
            + relative_tolerance
            * max(np.linalg.norm(second_difference), np.linalg.norm(auxiliary))
        )
        dual_tolerance = float(
            np.sqrt(length * series_count) * absolute_tolerance
            + relative_tolerance
            * np.linalg.norm(float(rho) * (difference.T @ scaled_dual))
        )
        report = TrendFilterReport(
            primal <= primal_tolerance and dual <= dual_tolerance,
            iteration,
            primal,
            dual,
            primal_tolerance,
            dual_tolerance,
        )
        if report.converged:
            break

    return np.ascontiguousarray(trend.T), report


def remove_acceleration_trend(
    signal: np.ndarray,
    regularization: float = 50.0,
    **solver_options: float | int,
) -> tuple[np.ndarray, TrendFilterReport]:
    """Detrend Acc XYZ and preserve Gyro XYZ for a ``[6,time]`` PADS signal."""
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 6:
        raise ValueError(f"Expected [6,time], got {values.shape}")
    trend, report = l1_trend_filter_batch(
        values[:3], regularization=regularization, **solver_options
    )
    output = values.copy()
    output[:3] = values[:3] - trend
    return np.ascontiguousarray(output), report
