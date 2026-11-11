from __future__ import annotations

import math
from statistics import median
from typing import Any, Sequence

import numpy as np

from .classification import classification_metrics


def select_binary_threshold(
    targets: Sequence[int] | np.ndarray,
    dd_probabilities: Sequence[float] | np.ndarray,
    *,
    metric: str = "balanced_accuracy",
    zero_division: float = 0.0,
) -> dict[str, Any]:
    """Select a PD/DD decision threshold from inner OOF predictions only.

    Class index 0 is PD and index 1 is DD. A sample is predicted DD when
    p(DD) >= threshold.
    """
    if metric not in {"balanced_accuracy", "macro_f1"}:
        raise ValueError(
            "threshold metric must be balanced_accuracy or macro_f1"
        )
    target_array = np.asarray(targets, dtype=np.int64).reshape(-1)
    score_array = np.asarray(dd_probabilities, dtype=np.float64).reshape(-1)
    if target_array.shape != score_array.shape or target_array.size == 0:
        raise ValueError("targets and dd_probabilities must be non-empty and aligned")
    if not np.isin(target_array, [0, 1]).all():
        raise ValueError("targets must contain only PD=0 and DD=1")
    if not np.isfinite(score_array).all() or np.any(score_array < 0) or np.any(
        score_array > 1
    ):
        raise ValueError("DD probabilities must be finite values in [0, 1]")

    unique_scores = np.unique(score_array)
    midpoints = (
        (unique_scores[:-1] + unique_scores[1:]) / 2.0
        if unique_scores.size > 1
        else np.asarray([], dtype=np.float64)
    )
    candidates = np.unique(
        np.concatenate(
            [
                np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
                unique_scores,
                midpoints,
            ]
        )
    )
    rows: list[dict[str, float]] = []
    for threshold in candidates:
        predictions = (score_array >= threshold).astype(np.int64)
        metrics = classification_metrics(
            target_array,
            predictions,
            num_classes=2,
            zero_division=zero_division,
        )
        rows.append(
            {
                "threshold": float(threshold),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
            }
        )
    secondary = "macro_f1" if metric == "balanced_accuracy" else "balanced_accuracy"
    best = max(
        rows,
        key=lambda row: (
            row[metric],
            row[secondary],
            -abs(row["threshold"] - 0.5),
            -row["threshold"],
        ),
    )
    return {
        "source": "inner_oof_only",
        "positive_class": "DD",
        "decision_rule": "predict DD when p(DD) >= threshold",
        "metric": metric,
        "threshold": best["threshold"],
        "balanced_accuracy": best["balanced_accuracy"],
        "macro_f1": best["macro_f1"],
        "candidate_count": len(rows),
        "sample_count": int(target_array.size),
    }


def final_epoch_from_inner_best(
    best_epochs: Sequence[int],
    *,
    strategy: str = "median_inner_best_epoch",
) -> dict[str, Any]:
    if strategy != "median_inner_best_epoch":
        raise ValueError("Only median_inner_best_epoch is supported")
    epochs = [int(value) for value in best_epochs]
    if not epochs or any(value < 1 for value in epochs):
        raise ValueError("best_epochs must contain positive one-based epochs")
    raw_median = float(median(epochs))
    selected = max(1, int(math.floor(raw_median + 0.5)))
    return {
        "strategy": strategy,
        "inner_best_epochs": epochs,
        "median": raw_median,
        "selected_epoch_count": selected,
    }
