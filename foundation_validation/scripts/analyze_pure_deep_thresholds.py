from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.metrics.threshold import select_binary_threshold


def _read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    targets: list[int] = []
    probabilities: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            targets.append(int(row["target"]))
            probabilities.append(float(row["probability_dd"]))
    return np.asarray(targets, dtype=np.int64), np.asarray(probabilities, dtype=np.float64)


def _metrics(targets: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(np.int64)
    tp = int(np.sum((targets == 1) & (predictions == 1)))
    tn = int(np.sum((targets == 0) & (predictions == 0)))
    fp = int(np.sum((targets == 0) & (predictions == 1)))
    fn = int(np.sum((targets == 1) & (predictions == 0)))
    pd_recall = tn / (tn + fp) if tn + fp else 0.0
    dd_recall = tp / (tp + fn) if tp + fn else 0.0
    precision_pd = tn / (tn + fn) if tn + fn else 0.0
    precision_dd = tp / (tp + fp) if tp + fp else 0.0
    f1_pd = 2 * precision_pd * pd_recall / (precision_pd + pd_recall) if precision_pd + pd_recall else 0.0
    f1_dd = 2 * precision_dd * dd_recall / (precision_dd + dd_recall) if precision_dd + dd_recall else 0.0
    return {
        "sample_count": int(targets.size),
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / targets.size),
        "balanced_accuracy": float((pd_recall + dd_recall) / 2),
        "macro_precision": float((precision_pd + precision_dd) / 2),
        "macro_f1": float((f1_pd + f1_dd) / 2),
        "pd_recall": float(pd_recall),
        "dd_recall": float(dd_recall),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-fit DD thresholds across inner validation folds without outer-test access."
    )
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    predictions: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for outer in range(5):
        for inner in range(3):
            path = run_dir / f"outer_{outer}" / f"inner_{inner}" / "predictions" / "validation.csv"
            predictions[(outer, inner)] = _read_predictions(path)

    rows: list[dict[str, float | int]] = []
    for outer in range(5):
        for held_inner in range(3):
            source = [predictions[(outer, inner)] for inner in range(3) if inner != held_inner]
            source_targets = np.concatenate([item[0] for item in source])
            source_probabilities = np.concatenate([item[1] for item in source])
            selected = select_binary_threshold(source_targets, source_probabilities)
            held_targets, held_probabilities = predictions[(outer, held_inner)]
            row = _metrics(held_targets, held_probabilities, float(selected["threshold"]))
            row.update(
                {
                    "outer_fold": outer,
                    "held_inner_fold": held_inner,
                    "threshold_source": "other_two_inner_validation_folds",
                    "threshold_fit_sample_count": int(source_targets.size),
                    "threshold_fit_balanced_accuracy": float(selected["balanced_accuracy"]),
                }
            )
            rows.append(row)

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_f1",
        "pd_recall",
        "dd_recall",
        "threshold",
    ]
    summary = {
        "protocol": {
            "scope": "15 inner-validation folds only",
            "outer_test_accessed": False,
            "selection_rule": "For each held inner fold, optimize balanced accuracy using the other two inner-validation folds in the same outer context.",
            "positive_class": "DD",
            "decision_rule": "predict DD when p(DD) >= threshold",
        },
        "fold_count": len(rows),
        "fold_mean": {name: float(np.mean([float(row[name]) for row in rows])) for name in metric_names},
        "fold_std": {name: float(np.std([float(row[name]) for row in rows])) for name in metric_names},
        "notes": [
            "AUROC is threshold-independent and is therefore unchanged from the source run.",
            "The same subjects can occur in different outer contexts; fold dispersion is descriptive, not an independent-subject confidence interval.",
        ],
    }

    output_dir = run_dir / "threshold_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "outer_fold",
        "held_inner_fold",
        "threshold_source",
        "threshold_fit_sample_count",
        "threshold_fit_balanced_accuracy",
        "sample_count",
        "threshold",
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_f1",
        "pd_recall",
        "dd_recall",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    with (output_dir / "cross_fitted_threshold_folds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "cross_fitted_threshold_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
