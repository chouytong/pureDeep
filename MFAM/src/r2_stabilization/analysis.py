from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)

from src.r2_stabilization.training import write_csv, write_json
from src.targeted_ablation.analysis import (
    H1_NAME,
    M0_NAME,
    cross_fitted_threshold_metrics,
    load_candidate_predictions,
    normalize_reference_predictions,
    subtype_summary,
)


S0_ROOT = Path(
    "/home/zyt/MFAM/outputs/pads_classification/v3_targeted_ablation/"
    "targeted_ablation_20260828"
)
MODELS = ("S1", "S2", "S3")


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_s0_predictions() -> list[dict[str, Any]]:
    rows = load_candidate_predictions(S0_ROOT, "R2")
    return [{**row, "model": "S0"} for row in rows]


def load_new_predictions(output_root: Path, model: str) -> list[dict[str, Any]]:
    rows = _read_csv(output_root / "models" / model / "development_predictions_all.csv")
    return [
        {
            **row,
            "outer_context": int(row["outer_context"]),
            "inner_fold": int(row["inner_fold"]),
            "target": int(row["target"]),
            "probability_pd": float(row["probability_pd"]),
            "probability_dd": float(row["probability_dd"]),
            "prediction": int(row["prediction"]),
            "threshold": float(row["threshold"]),
            "gate_mean": float(row["gate_mean"]),
        }
        for row in rows
    ]


def metric_values(
    rows: Sequence[Mapping[str, Any]], threshold: float = 0.5
) -> dict[str, float]:
    target = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    probability = np.asarray([float(row["probability_dd"]) for row in rows])
    prediction = (probability >= float(threshold)).astype(np.int64)
    probabilities = np.column_stack((1.0 - probability, probability))
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "macro_f1": float(
            f1_score(target, prediction, average="macro", zero_division=0)
        ),
        "auroc": float(roc_auc_score(target, probability)),
        "pd_recall": float(
            recall_score(target, prediction, pos_label=0, zero_division=0)
        ),
        "dd_recall": float(
            recall_score(target, prediction, pos_label=1, zero_division=0)
        ),
        "negative_log_likelihood": float(
            log_loss(target, probabilities, labels=[0, 1])
        ),
        "brier_score": float(brier_score_loss(target, probability)),
    }


def fold_metrics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for outer in range(5):
        for inner in range(3):
            selected = [
                row
                for row in rows
                if int(row["outer_context"]) == outer
                and int(row["inner_fold"]) == inner
            ]
            if not selected:
                raise ValueError(f"Missing outer={outer}, inner={inner}")
            result.append(
                {
                    "outer_context": outer,
                    "inner_fold": inner,
                    "validation_rows": len(selected),
                    **metric_values(selected),
                }
            )
    return result


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    folds = fold_metrics(rows)
    result: dict[str, Any] = {"fold_count": len(folds), "folds": folds}
    for metric in (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "auroc",
        "pd_recall",
        "dd_recall",
        "negative_log_likelihood",
        "brier_score",
    ):
        values = np.asarray([float(row[metric]) for row in folds])
        result[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "median": float(np.median(values)),
            "values": values.tolist(),
        }
    result["pooled_repeated_validation"] = metric_values(rows)
    return result


def paired_comparison(
    candidate: str,
    reference: str,
    candidate_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidate_folds = {
        (row["outer_context"], row["inner_fold"]): row
        for row in fold_metrics(candidate_rows)
    }
    reference_folds = {
        (row["outer_context"], row["inner_fold"]): row
        for row in fold_metrics(reference_rows)
    }
    if set(candidate_folds) != set(reference_folds):
        raise ValueError(f"Fold keys differ: {candidate} vs {reference}")
    details: list[dict[str, Any]] = []
    for key in sorted(reference_folds):
        row: dict[str, Any] = {
            "candidate": candidate,
            "reference": reference,
            "outer_context": key[0],
            "inner_fold": key[1],
        }
        for metric in ("balanced_accuracy", "auroc", "macro_f1", "dd_recall"):
            row[f"candidate_{metric}"] = candidate_folds[key][metric]
            row[f"reference_{metric}"] = reference_folds[key][metric]
            row[f"delta_{metric}"] = (
                candidate_folds[key][metric] - reference_folds[key][metric]
            )
        details.append(row)
    summary: dict[str, Any] = {
        "candidate": candidate,
        "reference": reference,
        "fold_count": 15,
    }
    for metric in ("balanced_accuracy", "auroc", "macro_f1", "dd_recall"):
        values = np.asarray([float(row[f"delta_{metric}"]) for row in details])
        summary[f"delta_{metric}_mean"] = float(values.mean())
        summary[f"delta_{metric}_median"] = float(np.median(values))
        summary[f"folds_delta_{metric}_ge_0"] = int((values >= 0).sum())
        summary[f"folds_delta_{metric}_gt_0"] = int((values > 0).sum())
    return summary, details


def subject_cluster_bootstrap(
    candidate: str,
    reference: str,
    candidate_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int = 2000,
    seed: int = 20260831,
) -> dict[str, Any]:
    candidate_map = {
        (int(row["outer_context"]), int(row["inner_fold"]), str(row["subject_id"])): row
        for row in candidate_rows
    }
    reference_map = {
        (int(row["outer_context"]), int(row["inner_fold"]), str(row["subject_id"])): row
        for row in reference_rows
    }
    if set(candidate_map) != set(reference_map):
        raise ValueError(f"Validation keys differ: {candidate} vs {reference}")
    by_subject: dict[str, list[tuple[int, int, str]]] = {}
    for key in candidate_map:
        by_subject.setdefault(key[2], []).append(key)
    subjects = np.asarray(sorted(by_subject))
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sampled = rng.choice(subjects, size=subjects.size, replace=True)
        candidate_sample: list[Mapping[str, Any]] = []
        reference_sample: list[Mapping[str, Any]] = []
        for subject in sampled:
            for key in by_subject[str(subject)]:
                candidate_sample.append(candidate_map[key])
                reference_sample.append(reference_map[key])
        deltas[index] = (
            metric_values(candidate_sample)["balanced_accuracy"]
            - metric_values(reference_sample)["balanced_accuracy"]
        )
    observed = (
        metric_values(candidate_rows)["balanced_accuracy"]
        - metric_values(reference_rows)["balanced_accuracy"]
    )
    return {
        "candidate": candidate,
        "reference": reference,
        "metric": "pooled_repeated_validation_balanced_accuracy_delta",
        "observed_delta": observed,
        "cluster": "subject_id",
        "unique_subjects": int(subjects.size),
        "validation_rows": len(candidate_rows),
        "iterations": iterations,
        "seed": seed,
        "ci95_low": float(np.percentile(deltas, 2.5)),
        "ci95_high": float(np.percentile(deltas, 97.5)),
        "bootstrap_mean": float(deltas.mean()),
        "independent_row_inference_used": False,
    }


def training_stability(output_root: Path, model: str) -> dict[str, Any]:
    summary = json.loads(
        (output_root / "models" / model / "development_summary.json").read_text()
    )
    folds = summary["fold_summaries"]
    epochs = np.asarray([int(row["best_epoch"]) for row in folds])
    gaps = np.asarray(
        [float(row["train_validation_balanced_accuracy_gap"]) for row in folds]
    )
    runtimes = np.asarray([float(row["runtime_seconds"]) for row in folds])
    return {
        "model": model,
        "precision": "FP32",
        "fold_count": len(folds),
        "failed_runs": int(summary["failed_fold_count"]),
        "best_epoch_minimum": int(epochs.min()),
        "best_epoch_maximum": int(epochs.max()),
        "best_epoch_mean": float(epochs.mean()),
        "best_epoch_median": float(np.median(epochs)),
        "train_validation_ba_gap_mean": float(gaps.mean()),
        "train_validation_ba_gap_median": float(np.median(gaps)),
        "train_validation_ba_gap_maximum": float(gaps.max()),
        "runtime_seconds_total": float(runtimes.sum()),
        "runtime_seconds_mean_fold": float(runtimes.mean()),
        "nonfinite_gradient_batches": int(
            sum(row["numerics"]["nonfinite_gradient_batches"] for row in folds)
        ),
        "nonfinite_loss_batches": int(
            sum(row["numerics"]["nonfinite_loss_batches"] for row in folds)
        ),
        "skipped_optimizer_steps": int(
            sum(row["numerics"]["skipped_optimizer_steps"] for row in folds)
        ),
        "maximum_gradient_norm_before_clipping": float(
            max(
                row["numerics"]["global_gradient_preclip"]["maximum"]
                for row in folds
            )
        ),
        "parameter_count_total": int(
            folds[0]["parameter_count"]["total_parameters"]
        ),
        "parameter_count_trainable": int(
            folds[0]["parameter_count"]["trainable_parameters"]
        ),
    }


def _branch_norm_rows(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _read_csv(output_root / "s0_checkpoint_activation_statistics.csv"):
        rows.append(dict(row))
    for model in MODELS:
        summary = json.loads(
            (output_root / "models" / model / "development_summary.json").read_text()
        )
        for fold in summary["fold_summaries"]:
            for partition in ("train", "validation"):
                diagnostics = fold[f"{partition}_diagnostics"]["activation_statistics"]
                for activation, values in diagnostics.items():
                    rows.append(
                        {
                            "model": model,
                            "partition": partition,
                            "outer_context": fold["outer_context"],
                            "inner_fold": fold["inner_fold"],
                            "activation": activation,
                            "dimension": values["dimension"],
                            **{f"l2_{key}": value for key, value in values["l2"].items()},
                            "maximum_absolute": values["maximum_absolute"],
                            "outer_test_accessed": False,
                        }
                    )
            validation = fold["validation_diagnostics"]["activation_statistics"]
            deep = float(validation["deep_embedding_raw"]["l2"]["mean"])
            statistical = float(
                validation["statistical_embedding_raw"]["l2"]["mean"]
            )
            rows.append(
                {
                    "model": model,
                    "partition": "validation",
                    "outer_context": fold["outer_context"],
                    "inner_fold": fold["inner_fold"],
                    "activation": "deep_to_statistical_raw_mean_l2_ratio",
                    "dimension": 1,
                    "l2_count": 1,
                    "l2_finite_count": 1,
                    "l2_mean": deep / statistical,
                    "l2_std": 0.0,
                    "l2_median": deep / statistical,
                    "l2_p5": deep / statistical,
                    "l2_p95": deep / statistical,
                    "l2_minimum": deep / statistical,
                    "l2_maximum": deep / statistical,
                    "maximum_absolute": deep / statistical,
                    "outer_test_accessed": False,
                }
            )
    return rows


def _training_numerics_rows(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        summary = json.loads(
            (output_root / "models" / model / "development_summary.json").read_text()
        )
        for fold in summary["fold_summaries"]:
            numerics = fold["numerics"]
            rows.append(
                {
                    "model": model,
                    "outer_context": fold["outer_context"],
                    "inner_fold": fold["inner_fold"],
                    "best_epoch": fold["best_epoch"],
                    "runtime_seconds": fold["runtime_seconds"],
                    "nonfinite_gradient_batches": numerics[
                        "nonfinite_gradient_batches"
                    ],
                    "nonfinite_loss_batches": numerics["nonfinite_loss_batches"],
                    "skipped_optimizer_steps": numerics["skipped_optimizer_steps"],
                    "gradient_preclip_mean": numerics[
                        "global_gradient_preclip"
                    ]["mean"],
                    "gradient_preclip_p95": numerics[
                        "global_gradient_preclip"
                    ]["p95"],
                    "gradient_preclip_maximum": numerics[
                        "global_gradient_preclip"
                    ]["maximum"],
                    "gradient_postclip_maximum": numerics[
                        "global_gradient_postclip"
                    ]["maximum"],
                    "loss_maximum": numerics["loss"]["maximum"],
                    "outer_test_accessed": False,
                }
            )
    return rows


def _branch_ablation_rows(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in ("S2", "S3"):
        summary = json.loads(
            (output_root / "models" / model / "development_summary.json").read_text()
        )
        for fold in summary["fold_summaries"]:
            both = fold["validation_metrics"]
            for mode, values in fold["branch_ablation"].items():
                metrics = values["metrics"]
                rows.append(
                    {
                        "model": model,
                        "outer_context": fold["outer_context"],
                        "inner_fold": fold["inner_fold"],
                        "mode": mode,
                        "both_ba": both["balanced_accuracy"],
                        "mode_ba": metrics["balanced_accuracy"],
                        "ba_drop_when_disabled": both["balanced_accuracy"]
                        - metrics["balanced_accuracy"],
                        "both_auroc": both["macro_auroc"],
                        "mode_auroc": metrics["macro_auroc"],
                        "auroc_drop_when_disabled": both["macro_auroc"]
                        - metrics["macro_auroc"],
                        "both_dd_recall": both["dd_recall"],
                        "mode_dd_recall": metrics["dd_recall"],
                        "outer_test_accessed": False,
                    }
                )
    return rows


def _gate_rows(predictions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    groups = [("all", list(predictions))]
    groups.extend(
        (f"diagnosis:{name}", [row for row in predictions if row["diagnosis"] == name])
        for name in ("PD", "DD")
    )
    groups.extend(
        (
            f"subtype:{name}",
            [row for row in predictions if row["dd_subtype"] == name],
        )
        for name in (
            "Other Movement Disorders",
            "Essential Tremor",
            "Atypical Parkinsonism",
            "Multiple Sclerosis",
        )
    )
    for group, rows in groups:
        values = np.asarray([float(row["gate_mean"]) for row in rows])
        result.append(
            {
                "model": "S3",
                "group": group,
                "rows": len(rows),
                "unique_subjects": len({str(row["subject_id"]) for row in rows}),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "median": float(np.median(values)),
                "p5": float(np.percentile(values, 5)),
                "p95": float(np.percentile(values, 95)),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "interpretation": "gate weights deep branch; 1=deep, 0=statistical",
            }
        )
    return result


def _retention_decision(
    paired: Mapping[tuple[str, str], Mapping[str, Any]],
    stability: Mapping[str, Mapping[str, Any]],
    branch_rows: Sequence[Mapping[str, Any]],
    gate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    stable = {
        model: (
            stability[model]["nonfinite_gradient_batches"] == 0
            and stability[model]["nonfinite_loss_batches"] == 0
            and stability[model]["skipped_optimizer_steps"] == 0
        )
        for model in MODELS
    }
    s2 = paired[("S2", "S1")]
    s2_gap_reduction = (
        stability["S1"]["train_validation_ba_gap_mean"]
        - stability["S2"]["train_validation_ba_gap_mean"]
    )
    s2_metric = (
        s2["delta_balanced_accuracy_mean"] >= 0.01
        and s2["folds_delta_balanced_accuracy_ge_0"] >= 9
    ) or (
        s2["delta_auroc_mean"] >= 0.01
        and s2["folds_delta_auroc_ge_0"] >= 9
    )
    s2_gap = (
        s2_gap_reduction >= 0.03
        and s2["delta_balanced_accuracy_mean"] >= -0.01
        and s2["delta_auroc_mean"] >= -0.01
    )
    retain_s2 = bool(stable["S2"] and (s2_metric or s2_gap))
    simpler = "S2" if retain_s2 else "S1"
    s3 = paired[("S3", simpler)]
    s3_metric = (
        s3["delta_balanced_accuracy_mean"] >= 0.01
        and s3["folds_delta_balanced_accuracy_ge_0"] >= 9
    ) or (
        s3["delta_auroc_mean"] >= 0.01
        and s3["folds_delta_auroc_ge_0"] >= 9
    )
    gate_all = next(row for row in gate_rows if row["group"] == "all")
    gate_ok = 0.10 <= float(gate_all["mean"]) <= 0.90
    s3_deep = [
        row
        for row in branch_rows
        if row["model"] == "S3" and row["mode"] == "deep_disabled"
    ]
    deep_ba_drop = float(
        np.mean([float(row["ba_drop_when_disabled"]) for row in s3_deep])
    )
    deep_auc_drop = float(
        np.mean([float(row["auroc_drop_when_disabled"]) for row in s3_deep])
    )
    deep_contributes = deep_ba_drop >= 0.01 or deep_auc_drop >= 0.01
    retain_s3 = bool(
        stable["S3"] and s3_metric and gate_ok and deep_contributes
    )
    if retain_s3:
        selected = "S3"
    elif retain_s2:
        selected = "S2"
    elif stable["S1"]:
        selected = "S1"
    else:
        stable_models = [model for model in MODELS if stable[model]]
        selected = stable_models[0] if stable_models else "none"
    return {
        "selected_candidate": selected,
        "freeze_scope": "development_candidate_specification_only",
        "stable": stable,
        "retain_S2": retain_s2,
        "S2_metric_rule": bool(s2_metric),
        "S2_gap_rule": bool(s2_gap),
        "S2_gap_reduction": s2_gap_reduction,
        "S3_reference_simpler_candidate": simpler,
        "retain_S3": retain_s3,
        "S3_metric_rule": bool(s3_metric),
        "S3_gate_mean": float(gate_all["mean"]),
        "S3_gate_noncollapsed": bool(gate_ok),
        "S3_deep_disabled_mean_ba_drop": deep_ba_drop,
        "S3_deep_disabled_mean_auroc_drop": deep_auc_drop,
        "S3_deep_contributes": bool(deep_contributes),
        "outer_test_accessed": False,
    }


def analyze_all(output_root: Path) -> dict[str, Any]:
    predictions = {
        "M0": [{**row, "model": "M0"} for row in normalize_reference_predictions(M0_NAME)],
        "H1": [{**row, "model": "H1"} for row in normalize_reference_predictions(H1_NAME)],
        "S0": load_s0_predictions(),
        **{model: load_new_predictions(output_root, model) for model in MODELS},
    }
    summaries = {model: summarize(rows) for model, rows in predictions.items()}
    stability = {model: training_stability(output_root, model) for model in MODELS}

    leaderboard: list[dict[str, Any]] = []
    for model in ("M0", "S0", "S1", "S2", "S3", "H1"):
        summary = summaries[model]
        if model in MODELS:
            precision = "FP32"
            params: Any = stability[model]["parameter_count_total"]
            nonfinite = stability[model]["nonfinite_gradient_batches"]
            skipped = stability[model]["skipped_optimizer_steps"]
        elif model == "S0":
            precision = "AMP"
            params = 123000
            historical = json.loads(
                (S0_ROOT / "models/R2/development_summary.json").read_text()
            )
            nonfinite = sum(
                int(row["nonfinite_gradient_batches"])
                for row in historical["fold_summaries"]
            )
            skipped = nonfinite
        else:
            precision = "reference"
            params = "reference"
            nonfinite = "not_applicable"
            skipped = "not_applicable"
        leaderboard.append(
            {
                "model": model,
                "precision": precision,
                "fusion": {
                    "M0": "deep_only",
                    "S0": "raw_concat",
                    "S1": "raw_concat",
                    "S2": "dual_LayerNorm_concat",
                    "S3": "64d_vector_gated_sum",
                    "H1": "handcrafted_logistic_reference",
                }[model],
                "balanced_accuracy": summary["balanced_accuracy"]["mean"],
                "balanced_accuracy_sd": summary["balanced_accuracy"]["std"],
                "auroc": summary["auroc"]["mean"],
                "macro_f1": summary["macro_f1"]["mean"],
                "accuracy": summary["accuracy"]["mean"],
                "pd_recall": summary["pd_recall"]["mean"],
                "dd_recall": summary["dd_recall"]["mean"],
                "negative_log_likelihood": summary[
                    "negative_log_likelihood"
                ]["mean"],
                "brier_score": summary["brier_score"]["mean"],
                "params": params,
                "nonfinite_gradient_batches": nonfinite,
                "skipped_optimizer_steps": skipped,
                "scope": "development_inner_cv_not_outer_test",
            }
        )
    write_csv(output_root / "development_leaderboard.csv", leaderboard)

    paired_summaries: list[dict[str, Any]] = []
    paired_details: list[dict[str, Any]] = []
    bootstrap: list[dict[str, Any]] = []
    paired_map: dict[tuple[str, str], Mapping[str, Any]] = {}
    references = ("S0", "M0", "H1")
    for candidate in MODELS:
        for reference in references:
            summary, details = paired_comparison(
                candidate,
                reference,
                predictions[candidate],
                predictions[reference],
            )
            paired_summaries.append(summary)
            paired_details.extend(details)
            paired_map[(candidate, reference)] = summary
            bootstrap.append(
                subject_cluster_bootstrap(
                    candidate,
                    reference,
                    predictions[candidate],
                    predictions[reference],
                )
            )
    for candidate, reference in (("S2", "S1"), ("S3", "S1"), ("S3", "S2")):
        summary, details = paired_comparison(
            candidate,
            reference,
            predictions[candidate],
            predictions[reference],
        )
        paired_summaries.append(summary)
        paired_details.extend(details)
        paired_map[(candidate, reference)] = summary
    write_csv(output_root / "paired_comparison_summary.csv", paired_summaries)
    write_csv(output_root / "paired_fold_differences.csv", paired_details)
    write_csv(output_root / "subject_cluster_bootstrap_delta_ba.csv", bootstrap)

    threshold_summary: list[dict[str, Any]] = []
    threshold_folds: list[dict[str, Any]] = []
    for model, rows in predictions.items():
        summary, folds = cross_fitted_threshold_metrics(rows)
        threshold_summary.append(
            {
                "model": model,
                "balanced_accuracy_mean": summary["balanced_accuracy"]["mean"],
                "macro_f1_mean": summary["macro_f1"]["mean"],
                "auroc_mean": summary["auroc"]["mean"],
                "pd_recall_mean": summary["pd_recall"]["mean"],
                "dd_recall_mean": summary["dd_recall"]["mean"],
                "threshold_mean": summary["threshold"]["mean"],
                "threshold_std": summary["threshold"]["std"],
                "outer_test_accessed": False,
            }
        )
        threshold_folds.extend({"model": model, **row} for row in folds)
    write_csv(output_root / "cross_fitted_threshold_summary.csv", threshold_summary)
    write_csv(output_root / "cross_fitted_threshold_folds.csv", threshold_folds)

    subtype_rows: list[dict[str, Any]] = []
    for model, rows in predictions.items():
        subtype_rows.extend(
            subtype_summary(model, rows, iterations=2000, seed=20260831)
        )
    write_csv(output_root / "dd_subtype_summary.csv", subtype_rows)

    stability_rows = list(stability.values())
    write_csv(output_root / "training_stability.csv", stability_rows)
    numerics_rows = _training_numerics_rows(output_root)
    write_csv(output_root / "training_numerics.csv", numerics_rows)
    branch_norm_rows = _branch_norm_rows(output_root)
    write_csv(output_root / "branch_norm_statistics.csv", branch_norm_rows)
    branch_rows = _branch_ablation_rows(output_root)
    write_csv(output_root / "branch_ablation_diagnostics.csv", branch_rows)
    gate_rows = _gate_rows(predictions["S3"])
    write_csv(output_root / "gate_statistics.csv", gate_rows)

    diagnostic = json.loads(
        (output_root / "s0_amp_diagnostic_summary.json").read_text()
    )
    historical = json.loads(
        (S0_ROOT / "models/R2/development_summary.json").read_text()
    )
    overflow_rows = [
        {
            "model": "S0_historical",
            "precision": "AMP",
            "folds": 15,
            "nonfinite_gradient_batches": sum(
                int(row["nonfinite_gradient_batches"])
                for row in historical["fold_summaries"]
            ),
            "nonfinite_loss_batches": 0,
            "skipped_optimizer_steps": sum(
                int(row["nonfinite_gradient_batches"])
                for row in historical["fold_summaries"]
            ),
            "scope": "read_only_historical_artifact",
        },
        {
            "model": "S0_diagnostic_first_epoch",
            "precision": "AMP",
            "folds": diagnostic["fold_count"],
            "nonfinite_gradient_batches": diagnostic[
                "total_nonfinite_gradient_batches"
            ],
            "nonfinite_loss_batches": 0,
            "skipped_optimizer_steps": diagnostic[
                "total_skipped_optimizer_steps"
            ],
            "scope": "bounded_diagnostic_replay_not_candidate",
        },
    ]
    overflow_rows.extend(
        {
            "model": model,
            "precision": "FP32",
            "folds": 15,
            "nonfinite_gradient_batches": stability[model][
                "nonfinite_gradient_batches"
            ],
            "nonfinite_loss_batches": stability[model]["nonfinite_loss_batches"],
            "skipped_optimizer_steps": stability[model]["skipped_optimizer_steps"],
            "scope": "development_inner_cv_not_outer_test",
        }
        for model in MODELS
    )
    write_csv(output_root / "gradient_overflow_summary.csv", overflow_rows)

    decision = _retention_decision(
        paired_map, stability, branch_rows, gate_rows
    )
    selected = decision["selected_candidate"]
    candidate_spec = {
        **decision,
        "model_specification": (
            json.loads(
                (output_root / "models" / selected / "outer_0/inner_0/model.json").read_text()
            )
            if selected != "none"
            else None
        ),
        "training_precision": "FP32" if selected != "none" else None,
        "plan_sha256": json.loads(
            (output_root / "run_manifest.json").read_text()
        )["plan_sha256"],
        "not_a_final_outer_model": True,
    }
    write_json(output_root / "selected_development_candidate.json", candidate_spec)
    result = {
        "scope": "development_inner_cv_not_outer_test",
        "models": list(predictions),
        "validation_rows_per_model": 1560,
        "unique_subjects": 390,
        "validation_rows_are_independent_subjects": False,
        "leaderboard": leaderboard,
        "paired_comparisons": paired_summaries,
        "subject_cluster_bootstrap": bootstrap,
        "candidate_decision": candidate_spec,
        "outer_test_accessed": False,
    }
    write_json(output_root / "analysis_summary.json", result)
    return result
