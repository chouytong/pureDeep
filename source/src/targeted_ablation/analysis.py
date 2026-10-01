from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)

from src.targeted_ablation.training import write_json


PREVIOUS_ROOT = Path(
    "/home/zyt/MFAM/outputs/pads_classification/v3_analysis_and_baselines/analysis_and_baselines_20260828"
)
M0_NAME = "M0_subject_mfam_frozen_inner_reference"
H1_NAME = "H1_handcrafted_logistic"
MR_NAME = "MR1_minirocket_multivariate_ridge"


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalize_reference_predictions(model: str) -> list[dict[str, Any]]:
    path = PREVIOUS_ROOT / "baselines" / model / "development_predictions_all.csv"
    result: list[dict[str, Any]] = []
    for row in read_csv(path):
        if row["probability_dd"] not in {"", "None"}:
            probability_dd = float(row["probability_dd"])
            score_source = "probability"
        else:
            decision = float(row["decision_score"])
            probability_dd = (
                1.0 / (1.0 + math.exp(-decision))
                if decision >= 0
                else math.exp(decision) / (1.0 + math.exp(decision))
            )
            score_source = "sigmoid_of_uncalibrated_decision_score"
        result.append(
            {
                "model": model,
                "outer_context": int(row["outer_context"]),
                "inner_fold": int(row["inner_fold"]),
                "subject_id": row["subject_id"],
                "target": int(row["binary_label"]),
                "diagnosis": row["diagnosis"],
                "dd_subtype": row["dd_subtype"],
                "probability_pd": 1.0 - probability_dd,
                "probability_dd": probability_dd,
                "prediction": 1 if row["predicted_label"] == "DD" else 0,
                "prediction_label": row["predicted_label"],
                "threshold": float(row["threshold"]),
                "analysis_scope": "development_inner_cv_not_outer_test",
                "score_source": score_source,
            }
        )
    return result


def load_candidate_predictions(output_root: Path, variant: str) -> list[dict[str, Any]]:
    directory = output_root / "models" / variant.replace("+", "_")
    rows = read_csv(directory / "development_predictions_all.csv")
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
        }
        for row in rows
    ]


def metric_values(rows: Sequence[Mapping[str, Any]], threshold: float = 0.5) -> dict[str, float]:
    target = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    probability = np.asarray([float(row["probability_dd"]) for row in rows])
    prediction = (probability >= float(threshold)).astype(np.int64)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "macro_f1": float(f1_score(target, prediction, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(target, probability)),
        "pd_recall": float(recall_score(target, prediction, pos_label=0, zero_division=0)),
        "dd_recall": float(recall_score(target, prediction, pos_label=1, zero_division=0)),
    }


def fold_metrics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for outer in range(5):
        for inner in range(3):
            selected = [
                row
                for row in rows
                if int(row["outer_context"]) == outer and int(row["inner_fold"]) == inner
            ]
            if not selected:
                raise ValueError(f"Missing predictions for outer={outer}, inner={inner}")
            result.append(
                {
                    "outer_context": outer,
                    "inner_fold": inner,
                    "validation_subjects": len(selected),
                    **metric_values(selected),
                }
            )
    return result


def summarize_folds(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    folds = fold_metrics(rows)
    result: dict[str, Any] = {"fold_count": len(folds), "folds": folds}
    for metric in ("balanced_accuracy", "macro_f1", "auroc", "pd_recall", "dd_recall"):
        values = np.asarray([float(row[metric]) for row in folds])
        result[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "median": float(np.median(values)),
            "values": values.tolist(),
        }
    result["pooled_repeated_validation"] = metric_values(rows)
    return result


def select_threshold(rows: Sequence[Mapping[str, Any]]) -> float:
    probabilities = np.asarray([float(row["probability_dd"]) for row in rows])
    unique = np.unique(probabilities)
    candidates = np.unique(
        np.concatenate(
            (
                np.asarray([0.0, 0.5, 1.0]),
                unique,
                (unique[:-1] + unique[1:]) / 2.0 if unique.size > 1 else unique,
            )
        )
    )
    scored = [
        (metric_values(rows, float(threshold))["balanced_accuracy"], float(threshold))
        for threshold in candidates
    ]
    best_score = max(value[0] for value in scored)
    tied = [value[1] for value in scored if abs(value[0] - best_score) <= 1e-12]
    return min(tied, key=lambda value: (abs(value - 0.5), value))


def cross_fitted_threshold_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fold_rows: list[dict[str, Any]] = []
    for outer in range(5):
        for held_inner in range(3):
            source = [
                row
                for row in rows
                if int(row["outer_context"]) == outer
                and int(row["inner_fold"]) != held_inner
            ]
            held = [
                row
                for row in rows
                if int(row["outer_context"]) == outer
                and int(row["inner_fold"]) == held_inner
            ]
            threshold = select_threshold(source)
            fold_rows.append(
                {
                    "outer_context": outer,
                    "held_inner_fold": held_inner,
                    "threshold_source": "other_two_disjoint_inner_validation_folds_same_outer",
                    "threshold_source_rows": len(source),
                    "held_rows": len(held),
                    "threshold": threshold,
                    **metric_values(held, threshold),
                }
            )
    summary: dict[str, Any] = {"fold_count": 15}
    for metric in ("balanced_accuracy", "macro_f1", "auroc", "pd_recall", "dd_recall"):
        values = np.asarray([float(row[metric]) for row in fold_rows])
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "median": float(np.median(values)),
        }
    thresholds = np.asarray([float(row["threshold"]) for row in fold_rows])
    summary["threshold"] = {
        "mean": float(thresholds.mean()),
        "std": float(thresholds.std(ddof=1)),
        "minimum": float(thresholds.min()),
        "maximum": float(thresholds.max()),
    }
    summary["outer_test_accessed"] = False
    return summary, fold_rows


def paired_fold_comparison(
    candidate: str,
    candidate_rows: Sequence[Mapping[str, Any]],
    m0_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidate_folds = {
        (row["outer_context"], row["inner_fold"]): row
        for row in fold_metrics(candidate_rows)
    }
    m0_folds = {
        (row["outer_context"], row["inner_fold"]): row
        for row in fold_metrics(m0_rows)
    }
    details = []
    for key in sorted(m0_folds):
        row = {"model": candidate, "outer_context": key[0], "inner_fold": key[1]}
        for metric in ("balanced_accuracy", "auroc", "macro_f1", "dd_recall"):
            row[f"candidate_{metric}"] = candidate_folds[key][metric]
            row[f"M0_{metric}"] = m0_folds[key][metric]
            row[f"delta_{metric}"] = candidate_folds[key][metric] - m0_folds[key][metric]
        details.append(row)
    summary: dict[str, Any] = {"model": candidate, "fold_count": 15}
    for metric in ("balanced_accuracy", "auroc", "macro_f1", "dd_recall"):
        values = np.asarray([float(row[f"delta_{metric}"]) for row in details])
        summary[f"delta_{metric}_mean"] = float(values.mean())
        summary[f"delta_{metric}_median"] = float(np.median(values))
        summary[f"folds_delta_{metric}_ge_0"] = int((values >= 0).sum())
        summary[f"folds_delta_{metric}_gt_0"] = int((values > 0).sum())
    return summary, details


def subject_cluster_bootstrap_delta_ba(
    candidate: str,
    candidate_rows: Sequence[Mapping[str, Any]],
    m0_rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int = 2000,
    seed: int = 20260828,
) -> dict[str, Any]:
    candidate_map = {
        (int(row["outer_context"]), int(row["inner_fold"]), str(row["subject_id"])): row
        for row in candidate_rows
    }
    m0_map = {
        (int(row["outer_context"]), int(row["inner_fold"]), str(row["subject_id"])): row
        for row in m0_rows
    }
    if set(candidate_map) != set(m0_map):
        raise ValueError(f"Candidate/M0 validation keys differ for {candidate}")
    by_subject: dict[str, list[tuple[int, int, str]]] = {}
    for key in candidate_map:
        by_subject.setdefault(key[2], []).append(key)
    subjects = np.asarray(sorted(by_subject))
    rng = np.random.default_rng(int(seed))
    deltas = np.empty(int(iterations), dtype=np.float64)
    for iteration in range(int(iterations)):
        sampled = rng.choice(subjects, size=subjects.size, replace=True)
        candidate_sample: list[Mapping[str, Any]] = []
        m0_sample: list[Mapping[str, Any]] = []
        for subject in sampled:
            for key in by_subject[str(subject)]:
                candidate_sample.append(candidate_map[key])
                m0_sample.append(m0_map[key])
        deltas[iteration] = (
            metric_values(candidate_sample)["balanced_accuracy"]
            - metric_values(m0_sample)["balanced_accuracy"]
        )
    observed = (
        metric_values(candidate_rows)["balanced_accuracy"]
        - metric_values(m0_rows)["balanced_accuracy"]
    )
    return {
        "model": candidate,
        "metric": "pooled_repeated_validation_balanced_accuracy_delta",
        "observed_delta": observed,
        "cluster": "subject_id",
        "unique_subjects": int(subjects.size),
        "validation_rows": len(candidate_rows),
        "iterations": int(iterations),
        "seed": int(seed),
        "ci95_low": float(np.percentile(deltas, 2.5)),
        "ci95_high": float(np.percentile(deltas, 97.5)),
        "bootstrap_mean": float(deltas.mean()),
        "independent_row_inference_used": False,
    }


def subtype_summary(
    model: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int = 2000,
    seed: int = 20260828,
) -> list[dict[str, Any]]:
    result = []
    for subtype in (
        "Other Movement Disorders",
        "Essential Tremor",
        "Atypical Parkinsonism",
        "Multiple Sclerosis",
    ):
        selected = [row for row in rows if row["dd_subtype"] == subtype]
        by_subject: dict[str, list[Mapping[str, Any]]] = {}
        for row in selected:
            by_subject.setdefault(str(row["subject_id"]), []).append(row)
        subjects = np.asarray(sorted(by_subject))
        predicted_by_subject = np.asarray(
            [
                np.mean([float(row["probability_dd"]) >= 0.5 for row in by_subject[s]])
                for s in subjects
            ]
        )
        probability_by_subject = np.asarray(
            [np.mean([float(row["probability_dd"]) for row in by_subject[s]]) for s in subjects]
        )
        rng = np.random.default_rng(int(seed) + len(result) * 1009)
        predicted_bootstrap = np.empty(iterations)
        probability_bootstrap = np.empty(iterations)
        for index in range(iterations):
            sampled = rng.integers(0, subjects.size, size=subjects.size)
            predicted_bootstrap[index] = predicted_by_subject[sampled].mean()
            probability_bootstrap[index] = probability_by_subject[sampled].mean()
        result.append(
            {
                "model": model,
                "subtype": subtype,
                "unique_subjects": int(subjects.size),
                "prediction_rows": len(selected),
                "predicted_dd_fraction": float(predicted_by_subject.mean()),
                "predicted_dd_ci95_low": float(np.percentile(predicted_bootstrap, 2.5)),
                "predicted_dd_ci95_high": float(np.percentile(predicted_bootstrap, 97.5)),
                "mean_probability_dd": float(probability_by_subject.mean()),
                "mean_probability_dd_ci95_low": float(np.percentile(probability_bootstrap, 2.5)),
                "mean_probability_dd_ci95_high": float(np.percentile(probability_bootstrap, 97.5)),
                "cluster": "subject_id",
                "iterations": iterations,
                "scope": "exploratory_development_inner_cv",
            }
        )
    return result


def select_combinations(output_root: Path) -> dict[str, Any]:
    m0_rows = normalize_reference_predictions(M0_NAME)
    m0 = summarize_folds(m0_rows)
    candidates = {}
    for variant in ("R1", "R2", "A1", "A2", "N1"):
        rows = load_candidate_predictions(output_root, variant)
        summary = summarize_folds(rows)
        paired, _ = paired_fold_comparison(variant, rows, m0_rows)
        eligible = (
            paired["delta_balanced_accuracy_mean"] >= 0.015
            and paired["delta_auroc_mean"] >= -0.01
            and paired["folds_delta_balanced_accuracy_ge_0"] >= 9
        )
        candidates[variant] = {
            "balanced_accuracy": summary["balanced_accuracy"]["mean"],
            "auroc": summary["auroc"]["mean"],
            **paired,
            "eligible": bool(eligible),
        }
    ranking = lambda name: (
        -candidates[name]["balanced_accuracy"],
        -candidates[name]["auroc"],
        name,
    )
    representations = sorted(
        [name for name in ("R1", "R2") if candidates[name]["eligible"]],
        key=ranking,
    )
    complementary = sorted(
        [name for name in ("A1", "A2", "N1") if candidates[name]["eligible"]],
        key=ranking,
    )
    combinations: list[str] = []
    if representations and complementary:
        combinations.append(f"{representations[0]}+{complementary[0]}")
        if len(complementary) > 1:
            combinations.append(f"{representations[0]}+{complementary[1]}")
    result = {
        "selection_scope": "development_inner_cv_only",
        "criteria": {
            "delta_balanced_accuracy_minimum": 0.015,
            "delta_auroc_minimum": -0.01,
            "folds_delta_balanced_accuracy_ge_0_minimum": 9,
        },
        "candidates": candidates,
        "eligible_representations_ranked": representations,
        "eligible_complementary_ranked": complementary,
        "selected_combinations": combinations[:2],
        "thresholds_relaxed": False,
        "outer_test_accessed": False,
    }
    write_json(output_root / "combination_selection.json", result)
    return result


def training_stability(output_root: Path, variant: str) -> dict[str, Any]:
    summary_path = output_root / "models" / variant.replace("+", "_") / "development_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    folds = summary["fold_summaries"]
    best_epochs = np.asarray([int(row["best_epoch"]) for row in folds])
    gaps = np.asarray([float(row["train_validation_balanced_accuracy_gap"]) for row in folds])
    runtimes = np.asarray([float(row["runtime_seconds"]) for row in folds])
    gradients = np.asarray([float(row["maximum_gradient_norm_before_clipping"]) for row in folds])
    entropy = np.asarray(
        [float(row["validation_diagnostics"]["mil_attention_entropy"]["mean"]) for row in folds]
    )
    utilization = np.asarray(
        [float(row["validation_diagnostics"]["mil_instance_utilization"]["mean"]) for row in folds]
    )
    return {
        "model": variant,
        "fold_count": len(folds),
        "failed_runs": int(summary["failed_fold_count"]),
        "best_epoch_minimum": int(best_epochs.min()),
        "best_epoch_maximum": int(best_epochs.max()),
        "best_epoch_mean": float(best_epochs.mean()),
        "best_epoch_median": float(np.median(best_epochs)),
        "train_validation_ba_gap_mean": float(gaps.mean()),
        "train_validation_ba_gap_median": float(np.median(gaps)),
        "train_validation_ba_gap_maximum": float(gaps.max()),
        "runtime_seconds_total": float(runtimes.sum()),
        "runtime_seconds_mean_fold": float(runtimes.mean()),
        "maximum_gradient_norm_before_clipping": float(gradients.max()),
        "nonfinite_runs": int(sum(bool(row["nonfinite_loss_or_gradient"]) for row in folds)),
        "attention_entropy_mean": float(entropy.mean()),
        "instance_utilization_mean": float(utilization.mean()),
        "parameter_count_total": int(folds[0]["parameter_count"]["total_parameters"]),
        "parameter_count_trainable": int(folds[0]["parameter_count"]["trainable_parameters"]),
    }


def analyze_all(output_root: Path) -> dict[str, Any]:
    selection = json.loads((output_root / "combination_selection.json").read_text())
    variants = ["R1", "R2", "A1", "A2", "N1", *selection["selected_combinations"]]
    models = [M0_NAME, H1_NAME, MR_NAME, *variants]
    predictions = {
        M0_NAME: normalize_reference_predictions(M0_NAME),
        H1_NAME: normalize_reference_predictions(H1_NAME),
        MR_NAME: normalize_reference_predictions(MR_NAME),
        **{variant: load_candidate_predictions(output_root, variant) for variant in variants},
    }
    summaries = {model: summarize_folds(rows) for model, rows in predictions.items()}
    leaderboard = []
    for model in models:
        summary = summaries[model]
        params: Any = "reference"
        if model in variants:
            params = training_stability(output_root, model)["parameter_count_total"]
        leaderboard.append(
            {
                "model": model,
                "modification": (
                    "frozen M0"
                    if model == M0_NAME
                    else "reference"
                    if model in {H1_NAME, MR_NAME}
                    else model
                ),
                "balanced_accuracy": summary["balanced_accuracy"]["mean"],
                "delta_ba_vs_M0": summary["balanced_accuracy"]["mean"] - summaries[M0_NAME]["balanced_accuracy"]["mean"],
                "auroc": summary["auroc"]["mean"],
                "delta_auroc_vs_M0": summary["auroc"]["mean"] - summaries[M0_NAME]["auroc"]["mean"],
                "macro_f1": summary["macro_f1"]["mean"],
                "pd_recall": summary["pd_recall"]["mean"],
                "dd_recall": summary["dd_recall"]["mean"],
                "params": params,
                "scope": "development_inner_cv_not_outer_test",
            }
        )
    leaderboard.sort(key=lambda row: float(row["balanced_accuracy"]), reverse=True)
    write_csv(output_root / "final_development_leaderboard.csv", leaderboard)

    paired_summaries = []
    paired_details = []
    bootstrap = []
    for variant in variants:
        paired, details = paired_fold_comparison(
            variant, predictions[variant], predictions[M0_NAME]
        )
        paired_summaries.append(paired)
        paired_details.extend(details)
        bootstrap.append(
            subject_cluster_bootstrap_delta_ba(
                variant, predictions[variant], predictions[M0_NAME]
            )
        )
    write_csv(output_root / "paired_fold_differences.csv", paired_details)
    write_csv(output_root / "paired_comparison_summary.csv", paired_summaries)
    write_csv(output_root / "subject_cluster_bootstrap_delta_ba.csv", bootstrap)

    threshold_summaries = []
    threshold_details = []
    for model in models:
        summary, details = cross_fitted_threshold_metrics(predictions[model])
        threshold_summaries.append(
            {
                "model": model,
                **{
                    f"{metric}_mean": summary[metric]["mean"]
                    for metric in ("balanced_accuracy", "macro_f1", "auroc", "pd_recall", "dd_recall")
                },
                "threshold_mean": summary["threshold"]["mean"],
                "threshold_std": summary["threshold"]["std"],
                "outer_test_accessed": False,
            }
        )
        threshold_details.extend({"model": model, **row} for row in details)
    write_csv(output_root / "cross_fitted_threshold_summary.csv", threshold_summaries)
    write_csv(output_root / "cross_fitted_threshold_folds.csv", threshold_details)

    subtype_rows = []
    for model in models:
        subtype_rows.extend(subtype_summary(model, predictions[model]))
    write_csv(output_root / "dd_subtype_subject_cluster_summary.csv", subtype_rows)

    stability_rows = [training_stability(output_root, variant) for variant in variants]
    write_csv(output_root / "training_stability.csv", stability_rows)

    r1_rows = [row for row in stability_rows if "R1" in row["model"].split("+")]
    r1_branch = []
    for variant in [row["model"] for row in r1_rows]:
        summary = json.loads(
            (
                output_root
                / "models"
                / variant.replace("+", "_")
                / "development_summary.json"
            ).read_text()
        )
        for fold in summary["fold_summaries"]:
            if fold["branch_ablation"] is None:
                continue
            r1_branch.append(
                {
                    "model": variant,
                    "outer_context": fold["outer_context"],
                    "inner_fold": fold["inner_fold"],
                    "frequency_only_ba": fold["branch_ablation"]["frequency_only"]["balanced_accuracy"],
                    "frequency_only_auroc": fold["branch_ablation"]["frequency_only"]["macro_auroc"],
                    "fullband_only_ba": fold["branch_ablation"]["fullband_only"]["balanced_accuracy"],
                    "fullband_only_auroc": fold["branch_ablation"]["fullband_only"]["macro_auroc"],
                    "frequency_bag_l2": fold["validation_diagnostics"]["r1_frequency_bag_l2"]["mean"],
                    "full_band_bag_l2": fold["validation_diagnostics"]["r1_full_band_bag_l2"]["mean"],
                    "branch_cosine": fold["validation_diagnostics"]["r1_branch_cosine"]["mean"],
                }
            )
    if r1_branch:
        write_csv(output_root / "r1_branch_diagnostics.csv", r1_branch)

    result = {
        "scope": "development_inner_cv_only",
        "models": models,
        "leaderboard": leaderboard,
        "paired_comparisons": paired_summaries,
        "subject_cluster_bootstrap": bootstrap,
        "combination_selection": selection,
        "outer_test_accessed": False,
        "validation_rows_are_independent_subjects": False,
        "unique_subjects": 390,
        "repeated_validation_rows": 1560,
    }
    write_json(output_root / "analysis_summary.json", result)
    return result
