from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.r2_stabilization.analysis import (
    load_new_predictions,
    paired_comparison,
    subject_cluster_bootstrap,
    summarize,
    training_stability,
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


PROJECT_ROOT = Path("/home/zyt/MFAM")
TARGETED_ROOT = PROJECT_ROOT / (
    "outputs/pads_classification/v3_targeted_ablation/targeted_ablation_20260828"
)
STABILIZATION_ROOT = PROJECT_ROOT / (
    "outputs/pads_classification/v3_r2_stabilization/stabilization_20260831"
)
MODELS = ("M0", "R1", "S1", "S4", "H1")


def _predictions(output_root: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        "M0": [{**row, "model": "M0"} for row in normalize_reference_predictions(M0_NAME)],
        "R1": [{**row, "model": "R1"} for row in load_candidate_predictions(TARGETED_ROOT, "R1")],
        "S1": [{**row, "model": "S1"} for row in load_new_predictions(STABILIZATION_ROOT, "S1")],
        "S4": [{**row, "model": "S4"} for row in load_new_predictions(output_root, "S4")],
        "H1": [{**row, "model": "H1"} for row in normalize_reference_predictions(H1_NAME)],
    }


def _leaderboard(
    summaries: Mapping[str, Mapping[str, Any]],
    s4_stability: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        summary = summaries[model]
        rows.append(
            {
                "model": model,
                "evidence_type": "development_inner_cv",
                "precision": "FP32" if model in {"S1", "S4"} else "frozen_reference",
                "balanced_accuracy": summary["balanced_accuracy"]["mean"],
                "balanced_accuracy_sd": summary["balanced_accuracy"]["std"],
                "auroc": summary["auroc"]["mean"],
                "auroc_sd": summary["auroc"]["std"],
                "macro_f1": summary["macro_f1"]["mean"],
                "accuracy": summary["accuracy"]["mean"],
                "pd_recall": summary["pd_recall"]["mean"],
                "dd_recall": summary["dd_recall"]["mean"],
                "negative_log_likelihood": summary["negative_log_likelihood"]["mean"],
                "binary_dd_brier": summary["brier_score"]["mean"],
                "parameter_count": (
                    s4_stability["parameter_count_total"] if model == "S4" else "read_only_reference"
                ),
                "scope": "development_inner_cv_not_outer_test",
                "outer_test_accessed": False,
            }
        )
    return rows


def _training_numerics(output_root: Path) -> list[dict[str, Any]]:
    summary = json.loads((output_root / "models/S4/development_summary.json").read_text())
    rows: list[dict[str, Any]] = []
    for fold in summary["fold_summaries"]:
        numerics = fold["numerics"]
        rows.append(
            {
                "model": "S4",
                "outer_context": fold["outer_context"],
                "inner_fold": fold["inner_fold"],
                "best_epoch": fold["best_epoch"],
                "runtime_seconds": fold["runtime_seconds"],
                "train_validation_ba_gap": fold["train_validation_balanced_accuracy_gap"],
                "nonfinite_loss_batches": numerics["nonfinite_loss_batches"],
                "nonfinite_gradient_batches": numerics["nonfinite_gradient_batches"],
                "skipped_optimizer_steps": numerics["skipped_optimizer_steps"],
                "gradient_preclip_mean": numerics["global_gradient_preclip"]["mean"],
                "gradient_preclip_p95": numerics["global_gradient_preclip"]["p95"],
                "gradient_preclip_maximum": numerics["global_gradient_preclip"]["maximum"],
                "gradient_postclip_maximum": numerics["global_gradient_postclip"]["maximum"],
                "loss_maximum": numerics["loss"]["maximum"],
                "outer_test_accessed": False,
            }
        )
    return rows


def _branch_rows(output_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary = json.loads((output_root / "models/S4/development_summary.json").read_text())
    contribution: list[dict[str, Any]] = []
    representation: list[dict[str, Any]] = []
    for fold in summary["fold_summaries"]:
        outer, inner = fold["outer_context"], fold["inner_fold"]
        diagnostics = fold["branch_ablation"]
        both = diagnostics["both"]["metrics"]
        contribution.append(
            {
                "model": "S4", "outer_context": outer, "inner_fold": inner,
                "diagnostic": "D0", "mode": "both",
                "balanced_accuracy": both["balanced_accuracy"],
                "auroc": both["macro_auroc"], "dd_recall": both["dd_recall"],
                "delta_ba_from_D0": 0.0, "delta_auroc_from_D0": 0.0,
                "mean_absolute_probability_dd_change": 0.0,
                "mean_logit_l2_change": 0.0, "retrained": False,
                "outer_test_accessed": False,
            }
        )
        for diagnostic, mode in (
            ("D1", "frequency_disabled"),
            ("D2", "fullband_disabled"),
            ("D3", "statistical_disabled"),
        ):
            values = diagnostics[mode]
            metrics = values["metrics"]
            contribution.append(
                {
                    "model": "S4", "outer_context": outer, "inner_fold": inner,
                    "diagnostic": diagnostic, "mode": mode,
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "auroc": metrics["macro_auroc"], "dd_recall": metrics["dd_recall"],
                    "delta_ba_from_D0": values["delta_ba_from_both"],
                    "delta_auroc_from_D0": values["delta_auroc_from_both"],
                    "mean_absolute_probability_dd_change": values["mean_absolute_probability_dd_change"],
                    "mean_logit_l2_change": values["mean_logit_l2_change"],
                    "retrained": False, "outer_test_accessed": False,
                }
            )
        activations = fold["validation_diagnostics"]["activation_statistics"]
        for branch, activation in (
            ("frequency", "frequency_embedding_raw"),
            ("fullband", "fullband_embedding_raw"),
            ("statistical", "statistical_embedding_raw"),
        ):
            values = activations[activation]
            representation.append(
                {
                    "model": "S4", "outer_context": outer, "inner_fold": inner,
                    "representation": branch, "dimension": values["dimension"],
                    "l2_mean": values["l2"]["mean"], "l2_std": values["l2"]["std"],
                    "l2_median": values["l2"]["median"],
                    "maximum_absolute": values["maximum_absolute"],
                    "frequency_fullband_cosine_mean": (
                        diagnostics["frequency_fullband_cosine"]["mean"] if branch == "frequency" else ""
                    ),
                    "pairwise_cosine_note": (
                        "frequency_vs_fullband_direct" if branch == "frequency"
                        else "frequency/statistical_and_fullband/statistical_undefined_without_forbidden_514_to_32_projection"
                    ),
                    "outer_test_accessed": False,
                }
            )
    return contribution, representation


def _aggregate_branch(contribution: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for diagnostic, mode in (
        ("D0", "both"), ("D1", "frequency_disabled"),
        ("D2", "fullband_disabled"), ("D3", "statistical_disabled"),
    ):
        rows = [row for row in contribution if row["mode"] == mode]
        result.append(
            {
                "diagnostic": diagnostic, "mode": mode, "fold_count": len(rows),
                "balanced_accuracy_mean": float(np.mean([row["balanced_accuracy"] for row in rows])),
                "auroc_mean": float(np.mean([row["auroc"] for row in rows])),
                "dd_recall_mean": float(np.mean([row["dd_recall"] for row in rows])),
                "delta_ba_from_D0_mean": float(np.mean([row["delta_ba_from_D0"] for row in rows])),
                "delta_auroc_from_D0_mean": float(np.mean([row["delta_auroc_from_D0"] for row in rows])),
                "mean_absolute_probability_dd_change": float(np.mean([row["mean_absolute_probability_dd_change"] for row in rows])),
                "mean_logit_l2_change": float(np.mean([row["mean_logit_l2_change"] for row in rows])),
                "interpretation_limit": "inference-time contribution_to_trained_model_not_causal_synergy",
            }
        )
    return result


def analyze_s4(output_root: Path) -> dict[str, Any]:
    predictions = _predictions(output_root)
    summaries = {model: summarize(rows) for model, rows in predictions.items()}
    s4_stability = training_stability(output_root, "S4")
    s1_stability = training_stability(STABILIZATION_ROOT, "S1")

    leaderboard = _leaderboard(summaries, s4_stability)
    write_csv(output_root / "development_leaderboard.csv", leaderboard)

    paired_summaries: list[dict[str, Any]] = []
    paired_details: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    paired_map: dict[str, Mapping[str, Any]] = {}
    for reference in ("M0", "R1", "S1", "H1"):
        summary, details = paired_comparison(
            "S4", reference, predictions["S4"], predictions[reference]
        )
        paired_map[reference] = summary
        paired_summaries.append(summary)
        paired_details.extend(details)
        bootstrap_rows.append(
            subject_cluster_bootstrap(
                "S4", reference, predictions["S4"], predictions[reference],
                iterations=2000, seed=20260901,
            )
        )
    write_csv(output_root / "paired_comparison_summary.csv", paired_summaries)
    write_csv(output_root / "paired_fold_differences.csv", paired_details)
    write_csv(output_root / "subject_cluster_bootstrap_delta_ba.csv", bootstrap_rows)

    threshold_summary: list[dict[str, Any]] = []
    threshold_folds: list[dict[str, Any]] = []
    for model in MODELS:
        threshold, folds = cross_fitted_threshold_metrics(predictions[model])
        threshold_summary.append(
            {
                "model": model,
                "balanced_accuracy_mean": threshold["balanced_accuracy"]["mean"],
                "macro_f1_mean": threshold["macro_f1"]["mean"],
                "auroc_mean": threshold["auroc"]["mean"],
                "pd_recall_mean": threshold["pd_recall"]["mean"],
                "dd_recall_mean": threshold["dd_recall"]["mean"],
                "threshold_mean": threshold["threshold"]["mean"],
                "threshold_std": threshold["threshold"]["std"],
                "protocol": "other_two_inner_validation_folds_same_outer",
                "outer_test_accessed": False,
            }
        )
        threshold_folds.extend({"model": model, **row} for row in folds)
    write_csv(output_root / "cross_fitted_threshold_summary.csv", threshold_summary)
    write_csv(output_root / "cross_fitted_threshold_folds.csv", threshold_folds)

    subtype_rows: list[dict[str, Any]] = []
    for model in ("M0", "H1", "S1", "S4"):
        subtype_rows.extend(
            subtype_summary(model, predictions[model], iterations=2000, seed=20260901)
        )
    write_csv(output_root / "dd_subtype_summary.csv", subtype_rows)

    write_csv(output_root / "training_stability.csv", [s1_stability, s4_stability])
    numerics = _training_numerics(output_root)
    write_csv(output_root / "training_numerics.csv", numerics)
    contribution, representation = _branch_rows(output_root)
    branch_summary = _aggregate_branch(contribution)
    write_csv(output_root / "branch_contribution_folds.csv", contribution)
    write_csv(output_root / "branch_contribution_summary.csv", branch_summary)
    write_csv(output_root / "branch_representation_diagnostics.csv", representation)

    s4s1 = paired_map["S1"]
    criteria = {
        "mean_delta_ba_ge_0_015": {
            "threshold": 0.015,
            "observed": s4s1["delta_balanced_accuracy_mean"],
            "pass": s4s1["delta_balanced_accuracy_mean"] >= 0.015,
        },
        "mean_delta_auroc_ge_0_010": {
            "threshold": 0.010,
            "observed": s4s1["delta_auroc_mean"],
            "pass": s4s1["delta_auroc_mean"] >= 0.010,
        },
        "ba_nonnegative_folds_ge_9": {
            "threshold": 9,
            "observed": s4s1["folds_delta_balanced_accuracy_ge_0"],
            "pass": s4s1["folds_delta_balanced_accuracy_ge_0"] >= 9,
        },
        "mean_delta_dd_recall_ge_minus_0_02": {
            "threshold": -0.02,
            "observed": s4s1["delta_dd_recall_mean"],
            "pass": s4s1["delta_dd_recall_mean"] >= -0.02,
        },
        "zero_nonfinite_and_skipped": {
            "threshold": 0,
            "observed": {
                "nonfinite_loss_batches": s4_stability["nonfinite_loss_batches"],
                "nonfinite_gradient_batches": s4_stability["nonfinite_gradient_batches"],
                "skipped_optimizer_steps": s4_stability["skipped_optimizer_steps"],
            },
            "pass": (
                s4_stability["nonfinite_loss_batches"] == 0
                and s4_stability["nonfinite_gradient_batches"] == 0
                and s4_stability["skipped_optimizer_steps"] == 0
            ),
        },
    }
    passed = all(bool(value["pass"]) for value in criteria.values())
    decision = {
        "s4_passes_all_core_criteria": passed,
        "decision": "S4_replaces_S1" if passed else "S4_rejected_S1_remains_selected",
        "selected_development_candidate": "S4" if passed else "S1",
        "previous_development_candidate": "S1",
        "criteria": criteria,
        "rule_was_mechanically_applied": True,
        "thresholds_were_not_changed": True,
        "retraining_or_tuning_after_decision": False,
        "freeze_scope": "development_candidate_specification_only",
        "not_a_final_outer_model": True,
        "outer_test_accessed": False,
    }
    write_json(output_root / "selected_development_candidate.json", decision)
    result = {
        "scope": "development_inner_cv_not_outer_test",
        "models": list(MODELS),
        "validation_rows_per_model": len(predictions["S4"]),
        "unique_subjects": len({row["subject_id"] for row in predictions["S4"]}),
        "validation_rows_are_independent_subjects": False,
        "leaderboard": leaderboard,
        "summaries": summaries,
        "paired_comparisons": paired_summaries,
        "subject_cluster_bootstrap": bootstrap_rows,
        "branch_contribution": branch_summary,
        "training_stability": {"S1": s1_stability, "S4": s4_stability},
        "candidate_decision": decision,
        "outer_test_loader_created": False,
        "outer_test_signal_accessed": False,
        "outer_test_predictions_accessed": False,
        "outer_test_features_transformed": False,
    }
    write_json(output_root / "analysis_summary.json", result)
    return result
