from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a subject-level mean-probability ensemble"
    )
    parser.add_argument(
        "--prediction-csv", action="append", required=True, type=Path
    )
    parser.add_argument(
        "--threshold",
        required=True,
        help="Use 'auto' on validation, or a fixed numeric threshold on test",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=20260821)
    return parser.parse_args()


def _load(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            probabilities = ast.literal_eval(row["probabilities"])
            rows.append(
                {
                    "subject_id": row["subject_id"],
                    "target": int(row["target"]),
                    "probability_dd": float(probabilities[1]),
                }
            )
    return rows


def _metrics(target: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    prediction = (probability >= threshold).astype(np.int64)
    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    matrix = np.zeros((2, 2), dtype=np.int64)
    for truth, predicted in zip(target, prediction):
        matrix[int(truth), int(predicted)] += 1
    true_positive = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted_count = matrix.sum(axis=0).astype(np.float64)
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive),
        where=support > 0,
    )
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros_like(true_positive),
        where=predicted_count > 0,
    )
    per_class_f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    result = {
        "threshold": float(threshold),
        "sample_count": int(target.size),
        "accuracy": float(np.mean(target == prediction)),
        "balanced_accuracy": float(np.mean(recall)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(per_class_f1)),
        "confusion_matrix": matrix.tolist(),
        "brier_score": float(np.mean((probability - target) ** 2)),
        "negative_log_likelihood": float(
            -np.mean(target * np.log(clipped) + (1 - target) * np.log(1 - clipped))
        ),
    }
    negative = probability[target == 0]
    positive = probability[target == 1]
    if negative.size == 0 or positive.size == 0:
        result["auroc"] = None
    else:
        comparisons = positive[:, None] - negative[None, :]
        result["auroc"] = float(
            (np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0))
            / comparisons.size
        )
    return result


def _select_threshold(target: np.ndarray, probability: np.ndarray) -> tuple[float, dict]:
    unique = np.unique(probability)
    candidates = np.unique(
        np.concatenate(
            ([0.0], unique, (unique[:-1] + unique[1:]) / 2.0, [1.0])
        )
    )
    scored = [(_metrics(target, probability, float(value)), float(value)) for value in candidates]
    best, threshold = max(
        scored,
        key=lambda item: (
            item[0]["macro_f1"],
            item[0]["balanced_accuracy"],
            -abs(item[1] - 0.5),
        ),
    )
    return threshold, best


def _bootstrap(
    target: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    resamples: int,
    seed: int,
) -> dict:
    if resamples <= 0:
        return {}
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(target == label) for label in (0, 1)]
    keys = ["accuracy", "balanced_accuracy", "macro_f1", "auroc", "brier_score"]
    values: dict[str, list[float]] = {key: [] for key in keys}
    for _ in range(resamples):
        sampled = np.concatenate(
            [rng.choice(indices, size=indices.size, replace=True) for indices in class_indices]
        )
        result = _metrics(target[sampled], probability[sampled], threshold)
        for key in keys:
            value = result[key]
            if value is not None:
                values[key].append(float(value))
    return {
        "method": "stratified_subject_bootstrap_percentile",
        "resamples": int(resamples),
        "seed": int(seed),
        "confidence_level": 0.95,
        "intervals": {
            key: [float(np.percentile(item, 2.5)), float(np.percentile(item, 97.5))]
            for key, item in values.items()
        },
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    runs = [_load(path) for path in args.prediction_csv]
    reference = [(row["subject_id"], row["target"]) for row in runs[0]]
    for path, rows in zip(args.prediction_csv[1:], runs[1:]):
        current = [(row["subject_id"], row["target"]) for row in rows]
        if current != reference:
            raise ValueError(f"Prediction rows do not align: {path}")
    target = np.asarray([row["target"] for row in runs[0]], dtype=np.int64)
    run_probabilities = np.asarray(
        [[row["probability_dd"] for row in rows] for rows in runs], dtype=np.float64
    )
    probability = run_probabilities.mean(axis=0)
    if args.threshold == "auto":
        threshold, metrics = _select_threshold(target, probability)
        threshold_source = "selected_on_input_predictions"
    else:
        threshold = float(args.threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        metrics = _metrics(target, probability, threshold)
        threshold_source = "fixed_cli_value"
    default_metrics = _metrics(target, probability, 0.5)
    result = {
        "prediction_files": [str(path.resolve()) for path in args.prediction_csv],
        "ensemble": "arithmetic_mean_of_dd_probabilities",
        "threshold_source": threshold_source,
        "selected_threshold": float(threshold),
        "metrics": metrics,
        "default_threshold_metrics": default_metrics,
        "per_model_probability_correlation": np.corrcoef(run_probabilities).tolist(),
        "bootstrap": _bootstrap(
            target,
            probability,
            threshold,
            args.bootstrap_resamples,
            args.bootstrap_seed,
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    with (args.output_dir / "predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["subject_id", "target", "probability_dd", "prediction"],
        )
        writer.writeheader()
        for (subject_id, _), label, score in zip(reference, target, probability):
            writer.writerow(
                {
                    "subject_id": subject_id,
                    "target": int(label),
                    "probability_dd": float(score),
                    "prediction": int(score >= threshold),
                }
            )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
