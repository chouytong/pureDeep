#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-freeze paired analysis of two completed nested-CV runs"
    )
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260916)
    return parser.parse_args()


def load_predictions(run_dir: Path) -> list[dict[str, Any]]:
    summary = json.loads((run_dir / "nested_cv_summary.json").read_text())
    result: list[dict[str, Any]] = []
    for outer, fold in enumerate(summary["outer_folds"]):
        threshold = float(fold["threshold"])
        path = run_dir / f"outer_{outer}" / "outer_final" / "outer_test" / "predictions.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                probability = float(row["probability_dd"])
                result.append({
                    "subject_id": row["subject_id"],
                    "target": int(row["target"]),
                    "probability_dd": probability,
                    "prediction": int(probability >= threshold),
                    "prediction_default": int(probability >= 0.5),
                    "outer_fold": outer,
                    "threshold": threshold,
                })
    if len(result) != 390 or len({row["subject_id"] for row in result}) != 390:
        raise ValueError("Expected exactly 390 unique outer-test subjects")
    return result


def metric_set(
    targets: np.ndarray, predictions: np.ndarray, probability: np.ndarray
) -> dict[str, float]:
    probabilities = np.column_stack([1.0 - probability, probability])
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "macro_precision": float(precision_score(targets, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(targets, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(targets, probability)),
        "negative_log_likelihood": float(log_loss(targets, probabilities, labels=[0, 1])),
        "two_class_brier": float(np.mean(np.sum((probabilities - np.eye(2)[targets]) ** 2, axis=1))),
        "pd_recall": float(recall_score(targets, predictions, pos_label=0)),
        "dd_recall": float(recall_score(targets, predictions, pos_label=1)),
    }


def interval(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "lower_95": float(np.percentile(values, 2.5)),
        "upper_95": float(np.percentile(values, 97.5)),
    }


def main() -> None:
    args = parse_args()
    candidate_rows = load_predictions(args.candidate_dir.resolve())
    reference_rows = load_predictions(args.reference_dir.resolve())
    candidate = {row["subject_id"]: row for row in candidate_rows}
    reference = {row["subject_id"]: row for row in reference_rows}
    if set(candidate) != set(reference):
        raise ValueError("Candidate/reference subjects do not match")
    subject_ids = sorted(candidate)
    if any(candidate[s]["target"] != reference[s]["target"] for s in subject_ids):
        raise ValueError("Candidate/reference labels do not match")
    targets = np.asarray([candidate[s]["target"] for s in subject_ids], dtype=np.int64)
    candidate_probability = np.asarray([candidate[s]["probability_dd"] for s in subject_ids])
    candidate_prediction = np.asarray([candidate[s]["prediction"] for s in subject_ids])
    candidate_default = np.asarray([candidate[s]["prediction_default"] for s in subject_ids])
    reference_probability = np.asarray([reference[s]["probability_dd"] for s in subject_ids])
    reference_prediction = np.asarray([reference[s]["prediction"] for s in subject_ids])

    rng = np.random.default_rng(args.seed)
    class_indices = [np.flatnonzero(targets == label) for label in (0, 1)]
    metric_names = ("balanced_accuracy", "macro_f1", "auroc")
    candidate_bootstrap = {name: [] for name in metric_names}
    reference_bootstrap = {name: [] for name in metric_names}
    delta_bootstrap = {name: [] for name in metric_names}
    for _ in range(args.bootstrap_replicates):
        sampled = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True)
            for indices in class_indices
        ])
        candidate_metrics = metric_set(
            targets[sampled], candidate_prediction[sampled], candidate_probability[sampled]
        )
        reference_metrics = metric_set(
            targets[sampled], reference_prediction[sampled], reference_probability[sampled]
        )
        for name in metric_names:
            candidate_bootstrap[name].append(candidate_metrics[name])
            reference_bootstrap[name].append(reference_metrics[name])
            delta_bootstrap[name].append(candidate_metrics[name] - reference_metrics[name])

    output = {
        "analysis_scope": "post_freeze_outer_test_descriptive_only",
        "model_selection_performed": False,
        "candidate": str(args.candidate_dir.resolve()),
        "reference": str(args.reference_dir.resolve()),
        "subject_count": len(subject_ids),
        "candidate_fold_threshold_metrics": metric_set(
            targets, candidate_prediction, candidate_probability
        ),
        "candidate_default_0p5_metrics": metric_set(
            targets, candidate_default, candidate_probability
        ),
        "reference_fold_threshold_metrics": metric_set(
            targets, reference_prediction, reference_probability
        ),
        "stratified_subject_bootstrap": {
            "replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "candidate": {
                name: interval(np.asarray(values))
                for name, values in candidate_bootstrap.items()
            },
            "reference": {
                name: interval(np.asarray(values))
                for name, values in reference_bootstrap.items()
            },
            "candidate_minus_reference": {
                name: interval(np.asarray(values))
                for name, values in delta_bootstrap.items()
            },
        },
        "guardrail": (
            "Outer-test results are final descriptive evidence. They must not be "
            "used to modify architecture, thresholds, epochs, or hyperparameters."
        ),
    }
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "final_paired_analysis.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
