from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.analysis import ANALYSIS_VERSION


CLASS_NAMES = ("PD", "DD")
DD_CONDITIONS = (
    "Other Movement Disorders",
    "Essential Tremor",
    "Atypical Parkinsonism",
    "Multiple Sclerosis",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_patients(patient_root: str | Path, selected: set[str] | None = None) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(patient_root).glob("patient_*.json")):
        patient = read_json(path)
        subject_id = str(patient["id"])
        if selected is None or subject_id in selected:
            result[subject_id] = patient
    return result


def subject_condition_map(manifest_root: str | Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(Path(manifest_root).glob("*.csv")):
        for row in read_csv(path):
            subject = row["subject_id"]
            condition = row["source_condition"]
            previous = result.setdefault(subject, condition)
            if previous != condition:
                raise ValueError(f"Condition mismatch for subject {subject}")
    return result


def frozen_outer_assignments(split_payload: Mapping[str, Any]) -> dict[str, int]:
    assignments: dict[str, int] = {}
    for outer in split_payload["outer"]:
        fold = int(outer["outer_fold"])
        for subject in outer["test_subjects"]:
            subject_id = str(subject)
            if subject_id in assignments:
                raise ValueError(f"Subject {subject_id} appears in multiple outer tests")
            assignments[subject_id] = fold
    if len(assignments) != 390:
        raise ValueError(f"Expected 390 outer assignments, found {len(assignments)}")
    return assignments


def binary_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
    *,
    probability_dd: Sequence[float] | None = None,
    decision_score: Sequence[float] | None = None,
) -> dict[str, Any]:
    y = np.asarray(targets, dtype=np.int64)
    pred = np.asarray(predictions, dtype=np.int64)
    if y.shape != pred.shape or y.ndim != 1 or y.size == 0:
        raise ValueError("targets/predictions must be matching non-empty vectors")
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    result: dict[str, Any] = {
        "sample_count": int(y.size),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "pd_recall": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "dd_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "dd_support": int(np.sum(y == 1)),
        "pd_support": int(np.sum(y == 0)),
    }
    score = None
    if probability_dd is not None:
        probability = np.asarray(probability_dd, dtype=np.float64)
        if probability.shape != y.shape or not np.isfinite(probability).all():
            raise ValueError("probability_dd must be finite and match targets")
        if np.any((probability < 0) | (probability > 1)):
            raise ValueError("probability_dd must be in [0,1]")
        score = probability
        result.update(
            {
                "negative_log_likelihood": float(
                    log_loss(y, np.column_stack([1.0 - probability, probability]), labels=[0, 1])
                ),
                # Required unified binary definition, not two-class one-hot sum.
                "binary_dd_brier": float(brier_score_loss(y, probability)),
            }
        )
    elif decision_score is not None:
        score = np.asarray(decision_score, dtype=np.float64)
        if score.shape != y.shape or not np.isfinite(score).all():
            raise ValueError("decision_score must be finite and match targets")
        result["negative_log_likelihood"] = None
        result["binary_dd_brier"] = None
    if score is not None and len(np.unique(y)) == 2:
        result["auroc"] = float(roc_auc_score(y, score))
        result["dd_pr_auc"] = float(average_precision_score(y, score))
    else:
        result["auroc"] = None
        result["dd_pr_auc"] = None
    return result


def calibration_table(
    targets: Sequence[int], probability_dd: Sequence[float], bins: int = 10
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    y = np.asarray(targets, dtype=np.float64)
    p = np.asarray(probability_dd, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(bins):
        mask = (p >= edges[index]) & (
            p <= edges[index + 1] if index == bins - 1 else p < edges[index + 1]
        )
        count = int(mask.sum())
        if count == 0:
            continue
        mean_probability = float(p[mask].mean())
        observed = float(y[mask].mean())
        gap = abs(mean_probability - observed)
        ece += count / max(len(y), 1) * gap
        rows.append(
            {
                "bin": index,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "mean_probability_dd": mean_probability,
                "observed_dd_fraction": observed,
                "absolute_gap": gap,
            }
        )
    return rows, {
        "ece_equal_width_10": float(ece),
        "binary_dd_brier": float(np.mean((p - y) ** 2)),
        "negative_log_likelihood": float(
            -np.mean(y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1)))
        ),
    }


def summarize_fold_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "balanced_accuracy",
        "macro_f1",
        "accuracy",
        "auroc",
        "pd_recall",
        "dd_recall",
        "negative_log_likelihood",
        "binary_dd_brier",
        "dd_pr_auc",
    )
    result: dict[str, Any] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        result[key] = {
            "values": values,
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
        }
    return result


def build_demographic_preprocessor(
    numeric_fields: Sequence[str], categorical_fields: Sequence[str]
) -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent", missing_values=None)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, list(numeric_fields)), ("categorical", categorical, list(categorical_fields))],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def prediction_row(
    *,
    subject_id: str,
    condition: str,
    target: int,
    prediction: int,
    outer_context: int,
    inner_fold: int,
    probability_dd: float | None,
    decision_score: float | None,
    threshold: float | None,
) -> dict[str, Any]:
    return {
        "subject_id": subject_id,
        "diagnosis": "DD" if target == 1 else "PD",
        "binary_label": int(target),
        "dd_subtype": condition if target == 1 else "Parkinson's",
        "outer_context": int(outer_context),
        "inner_fold": int(inner_fold),
        "probability_pd": None if probability_dd is None else float(1.0 - probability_dd),
        "probability_dd": None if probability_dd is None else float(probability_dd),
        "decision_score": None if decision_score is None else float(decision_score),
        "predicted_label": "DD" if prediction == 1 else "PD",
        "threshold": threshold,
        "analysis_scope": "development_inner_cv",
    }


def provenance_base() -> dict[str, Any]:
    return {
        "analysis_version": ANALYSIS_VERSION,
        "scope": "development_inner_cv_and_frozen_outer_descriptive_diagnostics",
        "outer_feedback_used_for_model_selection": False,
    }
