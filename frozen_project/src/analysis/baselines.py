from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.analysis.common import (
    binary_metrics,
    build_demographic_preprocessor,
    canonical_sha256,
    load_patients,
    prediction_row,
    read_csv,
    read_json,
    sha256_file,
    subject_condition_map,
    summarize_fold_metrics,
    write_csv,
    write_json,
)
from src.analysis.features import ACTIVITIES, select_feature_indices


PREDICTION_FIELDS = (
    "subject_id", "diagnosis", "binary_label", "dd_subtype", "outer_context",
    "inner_fold", "probability_pd", "probability_dd", "decision_score",
    "predicted_label", "threshold", "analysis_scope",
)


def _id_hash(subject_ids: Sequence[str]) -> str:
    return canonical_sha256([str(value) for value in subject_ids])


def _feature_pipeline(model: Any) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(threshold=0.0)),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def _run_fold(
    *,
    model_name: str,
    representation: str,
    estimator: Any,
    X_train: Any,
    y_train: np.ndarray,
    X_validation: Any,
    y_validation: np.ndarray,
    train_subjects: Sequence[str],
    validation_subjects: Sequence[str],
    conditions: Mapping[str, str],
    outer_context: int,
    inner_fold: int,
    feature_schema_sha256: str | None,
    config: Mapping[str, Any],
    output_dir: Path,
    save_model: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stage = output_dir / model_name / f"outer_{outer_context}" / f"inner_{inner_fold}"
    stage.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    estimator.fit(X_train, y_train)
    runtime = time.perf_counter() - started
    prediction = estimator.predict(X_validation).astype(np.int64)
    probability_dd: np.ndarray | None = None
    decision: np.ndarray | None = None
    if hasattr(estimator, "predict_proba"):
        probability_dd = np.asarray(estimator.predict_proba(X_validation))[:, 1]
        metrics = binary_metrics(y_validation, prediction, probability_dd=probability_dd)
    else:
        decision = np.asarray(estimator.decision_function(X_validation), dtype=np.float64)
        metrics = binary_metrics(y_validation, prediction, decision_score=decision)
    predictions = [
        prediction_row(
            subject_id=str(subject), condition=conditions[str(subject)], target=int(target),
            prediction=int(predicted), outer_context=outer_context, inner_fold=inner_fold,
            probability_dd=None if probability_dd is None else float(probability_dd[index]),
            decision_score=None if decision is None else float(decision[index]),
            threshold=0.5 if probability_dd is not None else 0.0,
        )
        for index, (subject, target, predicted) in enumerate(zip(validation_subjects, y_validation, prediction))
    ]
    model_sha = None
    if save_model:
        model_path = stage / "model.joblib"
        joblib.dump(estimator, model_path)
        model_sha = sha256_file(model_path)
    metadata = {
        "status": "complete", "scope": "development_inner_cv",
        "outer_test_accessed": False, "model_name": model_name,
        "representation": representation, "candidate_parameters": dict(config),
        "outer_context": outer_context, "inner_fold": inner_fold,
        "train_subject_count": len(train_subjects), "validation_subject_count": len(validation_subjects),
        "train_subject_ids_sha256": _id_hash(train_subjects),
        "validation_subject_ids_sha256": _id_hash(validation_subjects),
        "feature_schema_sha256": feature_schema_sha256,
        "seed": 42, "runtime_seconds": runtime, "model_artifact_sha256": model_sha,
        "metrics": metrics,
    }
    write_json(stage / "config.json", dict(config))
    write_json(stage / "metrics.json", metrics)
    write_json(stage / "metadata.json", metadata)
    write_csv(stage / "predictions.csv", predictions, PREDICTION_FIELDS)
    return metadata, predictions


def _aggregate_model(model_name: str, records: list[dict[str, Any]], predictions: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    metrics = [record["metrics"] for record in records]
    y = np.asarray([row["binary_label"] for row in predictions], dtype=np.int64)
    pred = np.asarray([int(row["predicted_label"] == "DD") for row in predictions])
    probabilities = [row["probability_dd"] for row in predictions]
    if all(value is not None for value in probabilities):
        pooled = binary_metrics(y, pred, probability_dd=np.asarray(probabilities, dtype=float))
    else:
        pooled = binary_metrics(y, pred, decision_score=np.asarray([row["decision_score"] for row in predictions], dtype=float))
    summary = {
        "model_name": model_name, "scope": "development_inner_cv_estimate_not_outer_test",
        "fold_count": len(records), "fold_metric_summary": summarize_fold_metrics(metrics),
        "pooled_repeated_validation_metrics": pooled,
        "prediction_rows": len(predictions),
        "unique_subjects": len({row["subject_id"] for row in predictions}),
        "outer_test_accessed": False,
    }
    write_csv(output_dir / model_name / "development_predictions_all.csv", predictions, PREDICTION_FIELDS)
    write_json(output_dir / model_name / "development_summary.json", summary)
    return summary


def frozen_subject_mfam_inner_reference(
    frozen_dir: Path, conditions: Mapping[str, str], output_dir: Path
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for outer in range(5):
        for inner in range(3):
            path = frozen_dir / f"outer_{outer}" / f"inner_{inner}" / "predictions" / "validation.csv"
            rows = read_csv(path)
            fold_predictions: list[dict[str, Any]] = []
            for row in rows:
                pdd = float(row["probability_dd"])
                target = int(row["target"])
                predicted = int(row["prediction"])
                fold_predictions.append(
                    prediction_row(
                        subject_id=row["subject_id"], condition=conditions[row["subject_id"]],
                        target=target, prediction=predicted, outer_context=outer, inner_fold=inner,
                        probability_dd=pdd, decision_score=None, threshold=0.5,
                    )
                )
            metrics = binary_metrics(
                [row["binary_label"] for row in fold_predictions],
                [int(row["predicted_label"] == "DD") for row in fold_predictions],
                probability_dd=[row["probability_dd"] for row in fold_predictions],
            )
            records.append({"outer_context": outer, "inner_fold": inner, "metrics": metrics})
            predictions.extend(fold_predictions)
    return _aggregate_model("M0_subject_mfam_frozen_inner_reference", records, predictions, output_dir)


def _demographic_frame(subjects: Sequence[str], patients: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "age": patients[subject].get("age"), "height": patients[subject].get("height"),
                "weight": patients[subject].get("weight"), "gender": patients[subject].get("gender"),
                "handedness": patients[subject].get("handedness"),
            }
            for subject in subjects
        ]
    )


def run_core_linear_baselines(
    *,
    feature_matrix: np.ndarray,
    subjects: Sequence[str],
    labels: np.ndarray,
    schema: Mapping[str, Any],
    split: Mapping[str, Any],
    conditions: Mapping[str, str],
    patients: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    index = {str(subject): position for position, subject in enumerate(subjects)}
    model_specs = {
        "H1_handcrafted_logistic": lambda: _feature_pipeline(LogisticRegression(C=1.0, class_weight=None, max_iter=5000, solver="liblinear", random_state=42)),
        "H1b_handcrafted_logistic_balanced": lambda: _feature_pipeline(LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, solver="liblinear", random_state=42)),
        "H2_handcrafted_linear_svm": lambda: _feature_pipeline(LinearSVC(C=1.0, class_weight=None, max_iter=20000, random_state=42)),
    }
    configs = {
        "H1_handcrafted_logistic": {"C": 1.0, "class_weight": None, "solver": "liblinear", "primary_metric": "balanced_accuracy"},
        "H1b_handcrafted_logistic_balanced": {"C": 1.0, "class_weight": "balanced", "solver": "liblinear", "scope": "secondary_sensitivity"},
        "H2_handcrafted_linear_svm": {"C": 1.0, "class_weight": None, "probability_calibration": None, "nll": "N/A", "brier": "N/A"},
    }
    all_records: dict[str, list[dict[str, Any]]] = {name: [] for name in model_specs}
    all_predictions: dict[str, list[dict[str, Any]]] = {name: [] for name in model_specs}
    d0_records: list[dict[str, Any]] = []
    d0_predictions: list[dict[str, Any]] = []
    for outer_def in split["outer"]:
        outer = int(outer_def["outer_fold"])
        for inner_def in outer_def["inner_folds"]:
            inner = int(inner_def["inner_fold"])
            train_subjects = [str(value) for value in inner_def["train_subjects"]]
            validation_subjects = [str(value) for value in inner_def["validation_subjects"]]
            train_indices = [index[value] for value in train_subjects]
            validation_indices = [index[value] for value in validation_subjects]
            y_train = labels[train_indices]; y_validation = labels[validation_indices]
            for name, factory in model_specs.items():
                record, prediction = _run_fold(
                    model_name=name, representation="deterministic_handcrafted_processed_v2",
                    estimator=factory(), X_train=feature_matrix[train_indices], y_train=y_train,
                    X_validation=feature_matrix[validation_indices], y_validation=y_validation,
                    train_subjects=train_subjects, validation_subjects=validation_subjects,
                    conditions=conditions, outer_context=outer, inner_fold=inner,
                    feature_schema_sha256=str(schema["schema_sha256"]), config=configs[name], output_dir=output_dir,
                )
                all_records[name].append(record); all_predictions[name].extend(prediction)
            demographic = Pipeline(
                [
                    ("preprocessor", build_demographic_preprocessor(["age", "height", "weight"], ["gender", "handedness"])),
                    ("model", LogisticRegression(C=1.0, class_weight=None, max_iter=5000, solver="liblinear", random_state=42)),
                ]
            )
            record, prediction = _run_fold(
                model_name="D0_demographic_logistic", representation="allowed_basic_demographics_only",
                estimator=demographic, X_train=_demographic_frame(train_subjects, patients), y_train=y_train,
                X_validation=_demographic_frame(validation_subjects, patients), y_validation=y_validation,
                train_subjects=train_subjects, validation_subjects=validation_subjects,
                conditions=conditions, outer_context=outer, inner_fold=inner, feature_schema_sha256=None,
                config={"numeric": ["age", "height", "weight"], "categorical": ["gender", "handedness"], "C": 1.0, "class_weight": None},
                output_dir=output_dir,
            )
            d0_records.append(record); d0_predictions.extend(prediction)
    summaries = {name: _aggregate_model(name, all_records[name], all_predictions[name], output_dir) for name in model_specs}
    summaries["D0_demographic_logistic"] = _aggregate_model("D0_demographic_logistic", d0_records, d0_predictions, output_dir)
    return summaries


def run_feature_information_analysis(
    *, feature_matrix: np.ndarray, subjects: Sequence[str], labels: np.ndarray,
    schema: Mapping[str, Any], split: Mapping[str, Any], conditions: Mapping[str, str], output_dir: Path
) -> dict[str, Any]:
    index = {str(subject): position for position, subject in enumerate(subjects)}
    variants: dict[str, np.ndarray] = {}
    for activity in ACTIVITIES:
        variants[f"single_activity::{activity}"] = select_feature_indices(dict(schema), include_activities=[activity])
        variants[f"leave_one_activity_out::{activity}"] = select_feature_indices(dict(schema), exclude_activities=[activity])
    variants.update(
        {
            "sensor::acc_only": select_feature_indices(dict(schema), sensors=["Acc"]),
            "sensor::gyro_only": select_feature_indices(dict(schema), sensors=["Gyro"]),
            "sensor::acc_gyro": select_feature_indices(dict(schema), sensors=["Acc", "Gyro"]),
            "wrist::left_only": select_feature_indices(dict(schema), wrists=["left"]),
            "wrist::right_only": select_feature_indices(dict(schema), wrists=["right"]),
            "wrist::bilateral": select_feature_indices(dict(schema), wrists=["left", "right"]),
        }
    )
    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for variant, selected in variants.items():
        for outer_def in split["outer"]:
            outer = int(outer_def["outer_fold"])
            for inner_def in outer_def["inner_folds"]:
                inner = int(inner_def["inner_fold"])
                train_subjects = [str(value) for value in inner_def["train_subjects"]]
                validation_subjects = [str(value) for value in inner_def["validation_subjects"]]
                train_indices = [index[value] for value in train_subjects]; validation_indices = [index[value] for value in validation_subjects]
                estimator = _feature_pipeline(LogisticRegression(C=1.0, class_weight=None, max_iter=5000, solver="liblinear", random_state=42))
                estimator.fit(feature_matrix[np.ix_(train_indices, selected)], labels[train_indices])
                probability = estimator.predict_proba(feature_matrix[np.ix_(validation_indices, selected)])[:, 1]
                prediction = (probability >= 0.5).astype(int)
                metrics = binary_metrics(labels[validation_indices], prediction, probability_dd=probability)
                metrics_rows.append({"variant": variant, "outer_context": outer, "inner_fold": inner, "feature_count": int(len(selected)), **metrics})
                for subject, target, predicted, pdd in zip(validation_subjects, labels[validation_indices], prediction, probability):
                    prediction_rows.append({"variant": variant, **prediction_row(subject_id=subject, condition=conditions[subject], target=int(target), prediction=int(predicted), outer_context=outer, inner_fold=inner, probability_dd=float(pdd), decision_score=None, threshold=0.5)})
    summary_rows: list[dict[str, Any]] = []
    for variant in variants:
        subset = [row for row in metrics_rows if row["variant"] == variant]
        aggregate = summarize_fold_metrics(subset)
        summary_rows.append(
            {
                "variant": variant, "fold_count": len(subset), "feature_count": subset[0]["feature_count"],
                **{f"{metric}_mean": values["mean"] for metric, values in aggregate.items()},
                **{f"{metric}_std": values["std"] for metric, values in aggregate.items()},
                "scope": "exploratory_development_inner_cv",
            }
        )
    analysis_dir = output_dir / "feature_information_analysis"; analysis_dir.mkdir(parents=True, exist_ok=False)
    write_csv(analysis_dir / "fold_metrics.csv", metrics_rows, list(metrics_rows[0]))
    write_csv(analysis_dir / "predictions.csv", prediction_rows, list(prediction_rows[0]))
    write_csv(analysis_dir / "summary.csv", summary_rows, list(summary_rows[0]))
    write_json(analysis_dir / "config.json", {"model": "handcrafted logistic", "C": 1.0, "class_weight": None, "variants_fixed_before_run": list(variants), "outer_test_accessed": False})
    return {"variant_count": len(variants), "summary": summary_rows}


def baseline_subtype_analysis(output_dir: Path, model_names: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name in model_names:
        predictions = read_csv(output_dir / model_name / "development_predictions_all.csv")
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in predictions:
            if row["diagnosis"] != "DD":
                continue
            grouped.setdefault(row["dd_subtype"], []).append(row)
        for subtype, values in sorted(grouped.items()):
            pdd_values = [float(row["probability_dd"]) for row in values if row["probability_dd"] not in ("", "None")]
            predicted_dd = [row["predicted_label"] == "DD" for row in values]
            rows.append(
                {
                    "model": model_name, "subtype": subtype, "prediction_rows": len(values),
                    "unique_subjects": len({row["subject_id"] for row in values}),
                    "mean_probability_dd": float(np.mean(pdd_values)) if pdd_values else None,
                    "std_probability_dd": float(np.std(pdd_values, ddof=1)) if len(pdd_values) > 1 else None,
                    "predicted_dd_fraction": float(np.mean(predicted_dd)),
                    "scope": "development_inner_cv_repeated_subject_predictions",
                }
            )
    write_csv(output_dir / "cross_model_dd_subtype_development.csv", rows, list(rows[0]))
    return rows
