from __future__ import annotations

from typing import Any

import numpy as np


def confusion_matrix(
    targets: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
    if targets.shape != predictions.shape:
        raise ValueError("targets and predictions must have matching shapes")
    if targets.size == 0:
        raise ValueError("Cannot compute metrics for an empty input")
    if np.any(targets < 0) or np.any(targets >= num_classes):
        raise ValueError("targets contain out-of-range labels")
    if np.any(predictions < 0) or np.any(predictions >= num_classes):
        raise ValueError("predictions contain out-of-range labels")
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (targets, predictions), 1)
    return matrix


def _safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    zero_division: float,
) -> np.ndarray:
    result = np.full_like(numerator, float(zero_division), dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


def classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
    zero_division: float = 0.0,
    probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    """计算分类指标。

    precision/recall 同时报告逐类、macro 和 weighted 口径；类别不平衡时应优先
    联合查看 macro-F1、balanced accuracy 与 confusion_matrix。AUROC 仅在传入
    [samples, classes] 概率时计算，不能由离散预测可靠恢复。
    """
    matrix = confusion_matrix(targets, predictions, num_classes)
    true_positive = np.diag(matrix).astype(np.float64)
    predicted_positive = matrix.sum(axis=0).astype(np.float64)
    actual_positive = matrix.sum(axis=1).astype(np.float64)

    precision = _safe_divide(true_positive, predicted_positive, zero_division)
    recall = _safe_divide(true_positive, actual_positive, zero_division)
    f1 = _safe_divide(
        2.0 * precision * recall,
        precision + recall,
        zero_division,
    )
    support = actual_positive.astype(np.int64)
    total = int(matrix.sum())
    accuracy = float(true_positive.sum() / total)
    weights = support / max(total, 1)

    result: dict[str, Any] = {
        "accuracy": accuracy,
        "balanced_accuracy": float(recall.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_precision": float((precision * weights).sum()),
        "weighted_recall": float((recall * weights).sum()),
        "weighted_f1": float((f1 * weights).sum()),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "support": support.tolist(),
        "confusion_matrix": matrix.tolist(),
        "sample_count": total,
    }
    if probabilities is not None:
        scores = np.asarray(probabilities, dtype=np.float64)
        targets_flat = np.asarray(targets, dtype=np.int64).reshape(-1)
        if scores.shape != (targets_flat.size, num_classes):
            raise ValueError(
                "probabilities must have shape [samples, num_classes], got "
                f"{scores.shape}"
            )
        if not np.isfinite(scores).all():
            raise ValueError("probabilities contain NaN or infinite values")
        if np.any(scores < 0):
            raise ValueError("probabilities contain negative values")
        row_sums = scores.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-4):
            raise ValueError("probability rows must sum to one")
        aucs = [
            _binary_roc_auc((targets_flat == index).astype(np.int64), scores[:, index])
            for index in range(num_classes)
        ]
        valid_aucs = [value for value in aucs if value is not None]
        one_hot = np.eye(num_classes, dtype=np.float64)[targets_flat]
        result.update(
            {
                "macro_auroc": (
                    float(np.mean(valid_aucs)) if valid_aucs else None
                ),
                "per_class_auroc": aucs,
                "brier_score": float(np.mean(np.sum((scores - one_hot) ** 2, axis=1))),
                "negative_log_likelihood": float(
                    -np.log(scores[np.arange(targets_flat.size), targets_flat].clip(1e-12)).mean()
                ),
            }
        )
    return result


def _binary_roc_auc(targets: np.ndarray, scores: np.ndarray) -> float | None:
    """Tie-aware Mann-Whitney AUROC without an optional sklearn dependency."""
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive = int(targets.sum())
    negative = int(targets.size - positive)
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(targets.size, dtype=np.float64)
    start = 0
    while start < targets.size:
        stop = start + 1
        while stop < targets.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    rank_sum = ranks[targets == 1].sum()
    return float((rank_sum - positive * (positive + 1) / 2.0) / (positive * negative))
