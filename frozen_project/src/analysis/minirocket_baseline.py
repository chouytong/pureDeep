from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from aeon.transformations.collection.convolution_based import MiniRocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis.baselines import PREDICTION_FIELDS
from src.analysis.common import (
    binary_metrics,
    canonical_sha256,
    prediction_row,
    sha256_file,
    summarize_fold_metrics,
    write_csv,
    write_json,
)


def run_minirocket(
    *,
    tensor: np.ndarray,
    subjects: Sequence[str],
    labels: np.ndarray,
    split: Mapping[str, Any],
    conditions: Mapping[str, str],
    output_dir: Path,
) -> dict[str, Any]:
    if tensor.ndim != 3 or tensor.shape[0] != len(subjects):
        raise ValueError("MiniRocket input must be [subject,channel,time]")
    if len(set(subjects)) != len(subjects):
        raise ValueError("MiniRocket input contains duplicate subject samples")
    model_name = "MR1_minirocket_multivariate_ridge"
    root = output_dir / model_name
    index = {str(subject): position for position, subject in enumerate(subjects)}
    alphas = np.logspace(-3, 3, 10)
    fold_records: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    for outer_def in split["outer"]:
        outer = int(outer_def["outer_fold"])
        for inner_def in outer_def["inner_folds"]:
            inner = int(inner_def["inner_fold"])
            stage = root / f"outer_{outer}" / f"inner_{inner}"
            stage.mkdir(parents=True, exist_ok=False)
            train_subjects = [str(value) for value in inner_def["train_subjects"]]
            validation_subjects = [str(value) for value in inner_def["validation_subjects"]]
            train_indices = [index[value] for value in train_subjects]
            validation_indices = [index[value] for value in validation_subjects]
            X_train = tensor[train_indices]
            X_validation = tensor[validation_indices]
            y_train = labels[train_indices]
            y_validation = labels[validation_indices]
            transformer = MiniRocket(
                n_kernels=10_000,
                max_dilations_per_kernel=32,
                n_jobs=16,
                random_state=42,
            )
            started = time.perf_counter()
            transformed_train = np.asarray(transformer.fit_transform(X_train), dtype=np.float32)
            transformed_validation = np.asarray(transformer.transform(X_validation), dtype=np.float32)
            classifier = Pipeline(
                [
                    ("scaler", StandardScaler(with_mean=False)),
                    ("ridge", RidgeClassifierCV(alphas=alphas, class_weight=None)),
                ]
            )
            classifier.fit(transformed_train, y_train)
            prediction = classifier.predict(transformed_validation).astype(np.int64)
            decision = np.asarray(classifier.decision_function(transformed_validation), dtype=np.float64)
            runtime = time.perf_counter() - started
            metrics = binary_metrics(y_validation, prediction, decision_score=decision)
            fold_predictions = [
                prediction_row(
                    subject_id=subject, condition=conditions[subject], target=int(target),
                    prediction=int(predicted), outer_context=outer, inner_fold=inner,
                    probability_dd=None, decision_score=float(score), threshold=0.0,
                )
                for subject, target, predicted, score in zip(
                    validation_subjects, y_validation, prediction, decision
                )
            ]
            model_path = stage / "model.joblib"
            joblib.dump({"transformer": transformer, "classifier": classifier}, model_path)
            selected_alpha = float(classifier.named_steps["ridge"].alpha_)
            metadata = {
                "status": "complete", "scope": "development_inner_cv",
                "outer_test_accessed": False, "model_name": model_name,
                "representation": "subject-level 132 channels = 11 activity x 2 wrist x 6 sensor channels; each activity linearly normalized to 976 samples",
                "outer_context": outer, "inner_fold": inner,
                "train_subject_count": len(train_subjects), "validation_subject_count": len(validation_subjects),
                "train_subject_ids_sha256": canonical_sha256(train_subjects),
                "validation_subject_ids_sha256": canonical_sha256(validation_subjects),
                "seed": 42, "n_kernels_requested": 10_000,
                "transformed_feature_count": int(transformed_train.shape[1]),
                "ridge_alpha_grid": alphas.tolist(), "selected_alpha_train_only": selected_alpha,
                "probability_calibration": None, "negative_log_likelihood": "N/A", "brier": "N/A",
                "runtime_seconds": runtime, "model_artifact_sha256": sha256_file(model_path),
                "trainable_parameter_count": int(transformed_train.shape[1] + 1),
                "metrics": metrics,
            }
            write_json(stage / "config.json", {
                "MiniRocket": {"n_kernels": 10_000, "max_dilations_per_kernel": 32, "n_jobs": 16, "random_state": 42},
                "RidgeClassifierCV": {"alphas": alphas.tolist(), "class_weight": None, "selection_scope": "inner_train_only"},
            })
            write_json(stage / "metrics.json", metrics)
            write_json(stage / "metadata.json", metadata)
            write_csv(stage / "predictions.csv", fold_predictions, PREDICTION_FIELDS)
            fold_records.append(metadata)
            all_predictions.extend(fold_predictions)
    metrics = [record["metrics"] for record in fold_records]
    y = np.asarray([row["binary_label"] for row in all_predictions], dtype=np.int64)
    pred = np.asarray([row["predicted_label"] == "DD" for row in all_predictions], dtype=np.int64)
    score = np.asarray([row["decision_score"] for row in all_predictions], dtype=np.float64)
    summary = {
        "model_name": model_name, "scope": "development_inner_cv_estimate_not_outer_test",
        "fold_count": 15, "fold_metric_summary": summarize_fold_metrics(metrics),
        "pooled_repeated_validation_metrics": binary_metrics(y, pred, decision_score=score),
        "prediction_rows": len(all_predictions), "unique_subjects": len(set(subjects)),
        "outer_test_accessed": False,
        "representation_shape": list(tensor.shape),
    }
    write_csv(root / "development_predictions_all.csv", all_predictions, PREDICTION_FIELDS)
    write_json(root / "development_summary.json", summary)
    return summary
