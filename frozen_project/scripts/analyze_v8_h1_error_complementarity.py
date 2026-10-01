#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.metrics.classification import classification_metrics
from src.utils.artifacts import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Development-only paired error audit for V8 and H1"
    )
    parser.add_argument("--v8-dir", required=True)
    parser.add_argument("--h1-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def metrics(targets: np.ndarray, probability_dd: np.ndarray) -> dict[str, Any]:
    probabilities = np.column_stack([1.0 - probability_dd, probability_dd])
    predictions = (probability_dd >= 0.5).astype(np.int64)
    return classification_metrics(
        targets, predictions, num_classes=2, zero_division=0.0,
        probabilities=probabilities,
    )


def main() -> None:
    args = parse_args()
    v8_dir = Path(args.v8_dir).resolve()
    h1_dir = Path(args.h1_dir).resolve()
    records: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []
    for outer in range(5):
        for inner in range(3):
            fold_id = f"outer_{outer}/inner_{inner}"
            v8_rows = read_csv(
                v8_dir / f"outer_{outer}" / f"inner_{inner}"
                / "predictions" / "validation.csv"
            )
            h1_rows = read_csv(
                h1_dir / f"outer_{outer}" / f"inner_{inner}" / "predictions.csv"
            )
            v8_by_id = {row["subject_id"]: row for row in v8_rows}
            h1_by_id = {row["subject_id"]: row for row in h1_rows}
            if set(v8_by_id) != set(h1_by_id):
                raise ValueError(f"Subject mismatch in {fold_id}")
            fold_records = []
            for subject_id in sorted(v8_by_id):
                v8 = v8_by_id[subject_id]
                h1 = h1_by_id[subject_id]
                target = int(v8["target"])
                if target != int(h1["binary_label"]):
                    raise ValueError(f"Label mismatch for {subject_id} in {fold_id}")
                record = {
                    "fold_id": fold_id,
                    "outer": outer,
                    "inner": inner,
                    "subject_id": subject_id,
                    "target": target,
                    "diagnosis": h1["diagnosis"],
                    "dd_subtype": h1["dd_subtype"],
                    "v8_probability_dd": float(v8["probability_dd"]),
                    "h1_probability_dd": float(h1["probability_dd"]),
                }
                record["v8_prediction"] = int(record["v8_probability_dd"] >= 0.5)
                record["h1_prediction"] = int(record["h1_probability_dd"] >= 0.5)
                record["v8_correct"] = record["v8_prediction"] == target
                record["h1_correct"] = record["h1_prediction"] == target
                fold_records.append(record)
                records.append(record)
            targets = np.asarray([row["target"] for row in fold_records])
            v8_probability = np.asarray([row["v8_probability_dd"] for row in fold_records])
            h1_probability = np.asarray([row["h1_probability_dd"] for row in fold_records])
            fold_metrics.append({
                "fold_id": fold_id,
                "v8": metrics(targets, v8_probability),
                "h1": metrics(targets, h1_probability),
                "fixed_equal_probability_average": metrics(
                    targets, (v8_probability + h1_probability) / 2.0
                ),
            })

    targets = np.asarray([row["target"] for row in records])
    v8_probability = np.asarray([row["v8_probability_dd"] for row in records])
    h1_probability = np.asarray([row["h1_probability_dd"] for row in records])
    correlation = float(np.corrcoef(v8_probability, h1_probability)[0, 1])
    overlap: dict[str, Any] = {}
    for label, target_value in (("PD", 0), ("DD", 1), ("all", None)):
        selected = [
            row for row in records
            if target_value is None or row["target"] == target_value
        ]
        overlap[label] = {
            "row_count": len(selected),
            "both_correct": sum(row["v8_correct"] and row["h1_correct"] for row in selected),
            "v8_only_correct": sum(row["v8_correct"] and not row["h1_correct"] for row in selected),
            "h1_only_correct": sum(not row["v8_correct"] and row["h1_correct"] for row in selected),
            "both_wrong": sum(not row["v8_correct"] and not row["h1_correct"] for row in selected),
        }

    subtype: dict[str, Any] = {}
    for name in sorted({row["dd_subtype"] for row in records if row["target"] == 1}):
        selected = [row for row in records if row["target"] == 1 and row["dd_subtype"] == name]
        subtype[name] = {
            "rows": len(selected),
            "unique_subjects": len({row["subject_id"] for row in selected}),
            "v8_recall": float(np.mean([row["v8_correct"] for row in selected])),
            "h1_recall": float(np.mean([row["h1_correct"] for row in selected])),
        }

    by_subject: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_subject.setdefault(row["subject_id"], []).append(row)
    if any(len(rows) != 4 for rows in by_subject.values()):
        raise ValueError("Expected every subject in four inner-development contexts")
    persistence: dict[str, Any] = {}
    for model in ("v8", "h1"):
        counts = [sum(row[f"{model}_correct"] for row in rows) for rows in by_subject.values()]
        persistence[model] = {
            "subjects_by_correct_context_count": {
                str(value): counts.count(value) for value in range(5)
            },
            "consistently_wrong_subjects": [
                subject_id for subject_id, rows in by_subject.items()
                if not any(row[f"{model}_correct"] for row in rows)
            ],
        }

    metric_names = (
        "accuracy", "balanced_accuracy", "macro_f1", "macro_auroc",
        "negative_log_likelihood", "brier_score",
    )
    mean_fold_metrics: dict[str, Any] = {}
    for model in ("v8", "h1", "fixed_equal_probability_average"):
        mean_fold_metrics[model] = {
            name: float(np.mean([fold[model][name] for fold in fold_metrics]))
            for name in metric_names
        }
    output = {
        "scope": "development_inner_validation_only",
        "outer_test_accessed": False,
        "row_count": len(records),
        "unique_subject_count": len(by_subject),
        "probability_pearson_correlation": correlation,
        "mean_fold_metrics": mean_fold_metrics,
        "correctness_overlap": overlap,
        "dd_subtype_recall": subtype,
        "cross_context_error_persistence": persistence,
        "interpretation_guardrail": (
            "The fixed equal-probability average and oracle-like correctness overlap "
            "are diagnostics only; H1 is not eligible for the pure-deep candidate."
        ),
    }
    write_json(Path(args.output), output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
