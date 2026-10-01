from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS = PROJECT_ROOT / "outputs/pads_classification/v3_analysis_and_baselines/analysis_and_baselines_20260828"
DEFAULT_SPLIT = PROJECT_ROOT / "outputs/pads_classification/v3_nested_cv/formal_subject_mfam_seed42_20260827/frozen_split.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Development-only H1 feature-group diagnostic")
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def estimator() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(threshold=0.0)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight=None,
                    max_iter=5000,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def metrics(y: np.ndarray, pred: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(y, probability)),
        "pd_recall": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "dd_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "nll": float(log_loss(y, np.column_stack((1.0 - probability, probability)), labels=[0, 1])),
        "binary_brier": float(np.mean((probability - y) ** 2)),
    }


def feature_groups(columns: list[str]) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    time_location_scale = {
        "mean", "std", "variance", "rms", "median", "mad", "iqr",
        "minimum", "maximum", "peak_to_peak", "signal_energy",
    }
    time_shape = {"skewness", "kurtosis_excess", "zero_crossing_rate_centered"}
    derivative = {"derivative_rms", "derivative_std"}
    spectral_global = {
        "dominant_frequency_hz_0p5_20", "dominant_power_0p5_20",
        "spectral_entropy_0p5_20", "total_power_0p5_20",
    }
    band_absolute = {
        "band_power_0p5_3", "band_power_3_7", "band_power_7_12", "band_power_12_20",
    }
    band_fraction = {
        "band_fraction_0p5_3", "band_fraction_3_7", "band_fraction_7_12", "band_fraction_12_20",
    }
    definitions = {
        "time_location_scale": time_location_scale,
        "time_shape": time_shape,
        "derivative": derivative,
        "spectral_global": spectral_global,
        "band_absolute": band_absolute,
        "band_fraction": band_fraction,
    }
    parsed = [column.split("|", 3) for column in columns]
    groups: dict[str, np.ndarray] = {"all": np.arange(len(columns), dtype=np.int64)}
    descriptions: dict[str, str] = {"all": "All H1 features; exact reproduction control"}

    for name, names in definitions.items():
        groups[f"only::{name}"] = np.asarray(
            [index for index, (_, _, _, feature) in enumerate(parsed) if feature in names], dtype=np.int64
        )
        groups[f"without::{name}"] = np.asarray(
            [index for index, (_, _, _, feature) in enumerate(parsed) if feature not in names], dtype=np.int64
        )
        descriptions[f"only::{name}"] = f"Only {name} features"
        descriptions[f"without::{name}"] = f"All features except {name}"

    axis = np.asarray([i for i, (_, _, signal, _) in enumerate(parsed) if not signal.endswith("Mag")], dtype=np.int64)
    magnitude = np.asarray([i for i, (_, _, signal, _) in enumerate(parsed) if signal.endswith("Mag")], dtype=np.int64)
    groups["signal::axes_only"] = axis
    groups["signal::magnitude_only"] = magnitude
    descriptions["signal::axes_only"] = "Six measured axes; excludes derived vector magnitudes"
    descriptions["signal::magnitude_only"] = "Only AccMag and GyroMag derived signals"

    time_names = time_location_scale | time_shape | derivative
    spectral_names = spectral_global | band_absolute | band_fraction
    groups["domain::time_all"] = np.asarray(
        [i for i, (_, _, _, feature) in enumerate(parsed) if feature in time_names], dtype=np.int64
    )
    groups["domain::spectral_all"] = np.asarray(
        [i for i, (_, _, _, feature) in enumerate(parsed) if feature in spectral_names], dtype=np.int64
    )
    descriptions["domain::time_all"] = "All time-domain features"
    descriptions["domain::spectral_all"] = "All spectral features"
    return groups, descriptions


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    payload = np.load(args.analysis_dir / "features/handcrafted_features.npz", allow_pickle=False)
    X = payload["X"]
    subjects = payload["subject_ids"].astype(str).tolist()
    y = payload["labels"].astype(np.int64)
    schema = json.loads((args.analysis_dir / "features/feature_schema.json").read_text())
    split = json.loads(args.split.read_text())
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    groups, descriptions = feature_groups(schema["columns"])

    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for variant, selected in groups.items():
        variant_metrics: list[dict[str, float]] = []
        for outer_def in split["outer"]:
            outer = int(outer_def["outer_fold"])
            for inner_def in outer_def["inner_folds"]:
                inner = int(inner_def["inner_fold"])
                train = np.asarray([subject_index[str(s)] for s in inner_def["train_subjects"]], dtype=np.int64)
                validation = np.asarray([subject_index[str(s)] for s in inner_def["validation_subjects"]], dtype=np.int64)
                model = estimator()
                model.fit(X[np.ix_(train, selected)], y[train])
                probability = model.predict_proba(X[np.ix_(validation, selected)])[:, 1]
                pred = (probability >= 0.5).astype(np.int64)
                result = metrics(y[validation], pred, probability)
                variant_metrics.append(result)
                fold_rows.append(
                    {
                        "variant": variant,
                        "outer_context": outer,
                        "inner_fold": inner,
                        "feature_count": int(selected.size),
                        **result,
                    }
                )
        summary: dict[str, Any] = {
            "variant": variant,
            "description": descriptions[variant],
            "feature_count": int(selected.size),
            "fold_count": len(variant_metrics),
        }
        for key in variant_metrics[0]:
            values = np.asarray([row[key] for row in variant_metrics])
            summary[f"{key}_mean"] = float(values.mean())
            summary[f"{key}_std"] = float(values.std(ddof=1))
        summary_rows.append(summary)

    all_ba = next(row["balanced_accuracy_mean"] for row in summary_rows if row["variant"] == "all")
    all_auc = next(row["auroc_mean"] for row in summary_rows if row["variant"] == "all")
    for row in summary_rows:
        row["balanced_accuracy_delta_vs_all"] = row["balanced_accuracy_mean"] - all_ba
        row["auroc_delta_vs_all"] = row["auroc_mean"] - all_auc
    summary_rows.sort(key=lambda row: row["balanced_accuracy_mean"], reverse=True)

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(args.output_dir / "fold_metrics.csv", fold_rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    manifest = {
        "scope": "development_inner_cv_diagnostic_only",
        "outer_test_accessed": False,
        "subject_count": len(subjects),
        "fold_count": 15,
        "split_path": str(args.split),
        "feature_matrix_path": str(args.analysis_dir / "features/handcrafted_features.npz"),
        "pipeline": "median imputation -> zero-variance removal -> train-only StandardScaler -> LogisticRegression(C=1, liblinear)",
        "full_h1_reproduction_balanced_accuracy": all_ba,
        "full_h1_reproduction_auroc": all_auc,
        "variants": len(summary_rows),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    for row in summary_rows:
        print(
            f"{row['variant']:<34} n={row['feature_count']:4d} "
            f"BA={row['balanced_accuracy_mean']:.4f} "
            f"AUROC={row['auroc_mean']:.4f} "
            f"DD={row['dd_recall_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
