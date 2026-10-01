#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models import build_model
from src.utils.config import load_config

from run_bilateral_component_diagnosis import extract_components
from run_layerwise_representation_diagnosis import load_fold_datasets


ACTIVITIES = (
    "CrossArms", "DrinkGlas", "Entrainment", "HoldWeight", "LiftHold",
    "PointFinger", "Relaxed", "RelaxedTask", "StretchHold", "TouchIndex",
    "TouchNose",
)
COMPONENTS = ("left", "right", "mean")
GROUPS = ("stable_correct", "stable_error", "unstable")
SUBJECT_METRICS = (
    "activity_true_margin_mean",
    "activity_true_margin_min",
    "activity_wrong_fraction",
    "activity_conflict_rate",
    "activity_margin_std",
    "activity_margin_iqr",
    "activity_abs_margin_mean",
    "wrist_conflict_rate",
    "wrist_disagreement_mean",
    "wrist_disagreement_max",
    "left_true_margin_mean",
    "right_true_margin_mean",
    "bilateral_true_margin_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen V8-GN error-oriented activity/wrist evidence diagnosis"
    )
    parser.add_argument("--run", action="append", required=True, help="SEED:/absolute/run/path")
    parser.add_argument("--stability-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-folds", type=int, default=0)
    return parser.parse_args()


def bh_fdr(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def paired_p(delta: np.ndarray) -> float:
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta) & (np.abs(delta) > 1.0e-12)]
    return float(wilcoxon(delta).pvalue) if len(delta) else 1.0


def fit_margin(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray) -> np.ndarray:
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2500, class_weight="balanced", solver="lbfgs", C=1.0
        ),
    )
    probe.fit(train_x, train_y)
    return np.asarray(probe.decision_function(val_x), dtype=float)


def conflict_rate(margins: np.ndarray) -> np.ndarray:
    positive = (margins > 0).sum(axis=1)
    negative = (margins < 0).sum(axis=1)
    pairs = margins.shape[1] * (margins.shape[1] - 1) / 2
    return positive * negative / max(pairs, 1.0)


def safe_rho(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 4 or len(np.unique(x[valid])) < 2 or len(np.unique(y[valid])) < 2:
        return float("nan")
    return float(spearmanr(x[valid], y[valid]).statistic)


def main() -> None:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Frozen embedding extraction requires CUDA for matched execution")

    stability = pd.read_csv(args.stability_csv, dtype={"subject_id": str})
    stability["subject_id"] = stability.subject_id.astype(str).str.zfill(3)
    stability_map = stability.set_index("subject_id").stability_group.to_dict()
    runs = []
    for item in args.run:
        seed, path = item.split(":", 1)
        runs.append((int(seed), Path(path).expanduser().resolve()))

    subject_rows: list[dict[str, Any]] = []
    activity_rows: list[dict[str, Any]] = []
    equivalence_rows: list[dict[str, Any]] = []
    relationship_rows: list[dict[str, Any]] = []
    processed = 0
    for seed, run in runs:
        for stage in sorted(run.glob("outer_*/inner_*")):
            if args.max_folds and processed >= args.max_folds:
                break
            outer = int(stage.parent.name.split("_")[-1])
            inner = int(stage.name.split("_")[-1])
            config = load_config(stage / "config.yaml")
            model = build_model(config).to(device)
            checkpoint = torch.load(
                stage / "checkpoints/best.pt", map_location=device, weights_only=False
            )
            model.load_state_dict(checkpoint["model_state"], strict=True)
            train_dataset, val_dataset, _ = load_fold_datasets(stage, config)
            train, train_eq = extract_components(model, train_dataset, config, device)
            val, val_eq = extract_components(model, val_dataset, config, device)
            equivalence_rows.append({
                "seed": seed, "outer": outer, "inner": inner,
                **{f"train_{key}": value for key, value in train_eq.items()},
                **{f"validation_{key}": value for key, value in val_eq.items()},
            })

            margins: dict[str, np.ndarray] = {}
            for component in COMPONENTS:
                component_margin = np.zeros((len(val["label"]), len(ACTIVITIES)), dtype=float)
                for activity_index, activity in enumerate(ACTIVITIES):
                    component_margin[:, activity_index] = fit_margin(
                        train["components"][component][:, activity_index],
                        train["label"],
                        val["components"][component][:, activity_index],
                    )
                margins[component] = component_margin

            label_sign = (2 * val["label"] - 1).astype(float)
            mean_margin = margins["mean"]
            mean_true = mean_margin * label_sign[:, None]
            left_true = margins["left"] * label_sign[:, None]
            right_true = margins["right"] * label_sign[:, None]
            wrist_conflict = np.sign(margins["left"]) != np.sign(margins["right"])
            wrist_disagreement = np.abs(margins["left"] - margins["right"])
            head_margin = val["logits"][:, 1] - val["logits"][:, 0]
            final_true_margin = head_margin * label_sign
            error = (final_true_margin < 0).astype(int)

            fold_rows = []
            for index, subject_id in enumerate(val["subject_id"].astype(str)):
                row = {
                    "seed": seed, "outer": outer, "inner": inner,
                    "subject_id": subject_id,
                    "label": int(val["label"][index]),
                    "stability_group": stability_map[subject_id],
                    "error": int(error[index]),
                    "final_true_margin": float(final_true_margin[index]),
                    "activity_true_margin_mean": float(mean_true[index].mean()),
                    "activity_true_margin_min": float(mean_true[index].min()),
                    "activity_wrong_fraction": float((mean_true[index] < 0).mean()),
                    "activity_conflict_rate": float(conflict_rate(mean_margin[index:index + 1])[0]),
                    "activity_margin_std": float(mean_margin[index].std(ddof=0)),
                    "activity_margin_iqr": float(np.subtract(*np.percentile(mean_margin[index], [75, 25]))),
                    "activity_abs_margin_mean": float(np.abs(mean_margin[index]).mean()),
                    "wrist_conflict_rate": float(wrist_conflict[index].mean()),
                    "wrist_disagreement_mean": float(wrist_disagreement[index].mean()),
                    "wrist_disagreement_max": float(wrist_disagreement[index].max()),
                    "left_true_margin_mean": float(left_true[index].mean()),
                    "right_true_margin_mean": float(right_true[index].mean()),
                    "bilateral_true_margin_mean": float((0.5 * (left_true[index] + right_true[index])).mean()),
                }
                for activity_index, activity in enumerate(ACTIVITIES):
                    row[f"mean_margin_{activity}"] = float(mean_margin[index, activity_index])
                    row[f"mean_true_margin_{activity}"] = float(mean_true[index, activity_index])
                    row[f"left_margin_{activity}"] = float(margins["left"][index, activity_index])
                    row[f"right_margin_{activity}"] = float(margins["right"][index, activity_index])
                subject_rows.append(row)
                fold_rows.append(row)

            fold_frame = pd.DataFrame(fold_rows)
            for metric in SUBJECT_METRICS:
                relationship_rows.append({
                    "seed": seed, "outer": outer, "inner": inner,
                    "metric": metric,
                    "rho_with_final_true_margin": safe_rho(
                        fold_frame[metric].to_numpy(), fold_frame.final_true_margin.to_numpy()
                    ),
                    "rho_with_error": safe_rho(
                        fold_frame[metric].to_numpy(), fold_frame.error.to_numpy()
                    ),
                })
            for activity_index, activity in enumerate(ACTIVITIES):
                for component in COMPONENTS:
                    raw_margin = margins[component][:, activity_index]
                    true_margin = raw_margin * label_sign
                    for group in GROUPS:
                        mask = fold_frame.stability_group.to_numpy() == group
                        for label_name, label_value in (("all", None), ("PD", 0), ("DD", 1)):
                            selected = mask if label_value is None else mask & (val["label"] == label_value)
                            if selected.sum() == 0:
                                continue
                            activity_rows.append({
                                "seed": seed, "outer": outer, "inner": inner,
                                "activity": activity, "component": component,
                                "stability_group": group, "label_group": label_name,
                                "n": int(selected.sum()),
                                "raw_dd_margin_mean": float(raw_margin[selected].mean()),
                                "true_margin_mean": float(true_margin[selected].mean()),
                                "wrong_evidence_fraction": float((true_margin[selected] < 0).mean()),
                                "abs_margin_mean": float(np.abs(raw_margin[selected]).mean()),
                            })
            processed += 1
            print(f"processed seed={seed} outer={outer} inner={inner}", flush=True)
        if args.max_folds and processed >= args.max_folds:
            break

    subjects = pd.DataFrame(subject_rows)
    activities = pd.DataFrame(activity_rows)
    relationships = pd.DataFrame(relationship_rows)
    equivalence = pd.DataFrame(equivalence_rows)
    subjects.to_csv(output / "subject_appearances_45_seed_folds.csv", index=False)
    activities.to_csv(output / "activity_group_evidence_45_seed_folds.csv", index=False)
    relationships.to_csv(output / "metric_relationships_45_seed_folds.csv", index=False)
    equivalence.to_csv(output / "extraction_equivalence_checks.csv", index=False)

    numeric_subject = ["error", "final_true_margin", *SUBJECT_METRICS]
    group45 = subjects.groupby(
        ["seed", "outer", "inner", "stability_group"], as_index=False
    )[numeric_subject].mean()
    group15 = group45.groupby(
        ["outer", "inner", "stability_group"], as_index=False
    )[numeric_subject].mean()
    group15.to_csv(output / "subject_group_metrics_15_splits.csv", index=False)

    indexed = group15.set_index(["outer", "inner", "stability_group"])
    split_keys = sorted({(int(row.outer), int(row.inner)) for row in group15.itertuples()})
    comparisons = []
    for metric in SUBJECT_METRICS:
        for left_group, right_group in (
            ("stable_error", "stable_correct"),
            ("stable_error", "unstable"),
        ):
            delta = np.asarray([
                indexed.loc[(outer, inner, left_group), metric]
                - indexed.loc[(outer, inner, right_group), metric]
                for outer, inner in split_keys
            ])
            comparisons.append({
                "metric": metric, "comparison": f"{left_group}-{right_group}",
                "n_splits": len(delta), "mean_delta": float(delta.mean()),
                "positive_splits": int((delta > 0).sum()),
                "negative_splits": int((delta < 0).sum()),
                "wilcoxon_p": paired_p(delta),
            })
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame["bh_fdr_q"] = bh_fdr(comparison_frame.wilcoxon_p.to_numpy())
    comparison_frame.to_csv(output / "subject_group_comparisons_bh.csv", index=False)

    label_group45 = subjects.groupby(
        ["seed", "outer", "inner", "label", "stability_group"], as_index=False
    )[numeric_subject].mean()
    label_group15 = label_group45.groupby(
        ["outer", "inner", "label", "stability_group"], as_index=False
    )[numeric_subject].mean()
    label_group15.to_csv(output / "subject_label_group_metrics_15_splits.csv", index=False)
    label_indexed = label_group15.set_index(
        ["outer", "inner", "label", "stability_group"]
    )
    label_comparisons = []
    for label, label_name in ((0, "PD"), (1, "DD")):
        for metric in SUBJECT_METRICS:
            values = []
            for outer, inner in split_keys:
                try:
                    values.append(
                        label_indexed.loc[(outer, inner, label, "stable_error"), metric]
                        - label_indexed.loc[(outer, inner, label, "stable_correct"), metric]
                    )
                except KeyError:
                    continue
            delta = np.asarray(values, dtype=float)
            label_comparisons.append({
                "label_group": label_name, "metric": metric,
                "n_splits": len(delta),
                "mean_delta_error_minus_correct": float(delta.mean()) if len(delta) else float("nan"),
                "positive_splits": int((delta > 0).sum()),
                "negative_splits": int((delta < 0).sum()),
                "wilcoxon_p": paired_p(delta),
            })
    label_comparison_frame = pd.DataFrame(label_comparisons)
    label_comparison_frame["bh_fdr_q"] = bh_fdr(
        label_comparison_frame.wilcoxon_p.to_numpy()
    )
    label_comparison_frame.to_csv(output / "subject_label_comparisons_bh.csv", index=False)

    activity_numeric = ("n", "raw_dd_margin_mean", "true_margin_mean", "wrong_evidence_fraction", "abs_margin_mean")
    activity15 = activities.groupby(
        ["outer", "inner", "activity", "component", "stability_group", "label_group"],
        as_index=False,
    )[list(activity_numeric)].mean()
    activity15.to_csv(output / "activity_group_evidence_15_splits.csv", index=False)
    activity_indexed = activity15.set_index(
        ["outer", "inner", "activity", "component", "stability_group", "label_group"]
    )
    activity_comparisons = []
    for activity in ACTIVITIES:
        for component in COMPONENTS:
            for label_group in ("all", "PD", "DD"):
                for metric in ("true_margin_mean", "wrong_evidence_fraction", "abs_margin_mean"):
                    values = []
                    for outer, inner in split_keys:
                        try:
                            values.append(
                                activity_indexed.loc[(outer, inner, activity, component, "stable_error", label_group), metric]
                                - activity_indexed.loc[(outer, inner, activity, component, "stable_correct", label_group), metric]
                            )
                        except KeyError:
                            continue
                    delta = np.asarray(values, dtype=float)
                    activity_comparisons.append({
                        "activity": activity, "component": component,
                        "label_group": label_group, "metric": metric,
                        "n_splits": len(delta),
                        "mean_delta_error_minus_correct": float(delta.mean()) if len(delta) else float("nan"),
                        "positive_splits": int((delta > 0).sum()),
                        "negative_splits": int((delta < 0).sum()),
                        "wilcoxon_p": paired_p(delta),
                    })
    activity_comparison_frame = pd.DataFrame(activity_comparisons)
    finite = activity_comparison_frame.wilcoxon_p.notna()
    activity_comparison_frame.loc[finite, "bh_fdr_q"] = bh_fdr(
        activity_comparison_frame.loc[finite, "wilcoxon_p"].to_numpy()
    )
    activity_comparison_frame.to_csv(output / "activity_comparisons_bh.csv", index=False)

    relation15 = relationships.groupby(["outer", "inner", "metric"], as_index=False)[
        ["rho_with_final_true_margin", "rho_with_error"]
    ].mean()
    relation15.to_csv(output / "metric_relationships_15_splits.csv", index=False)
    relation_summary = relation15.groupby("metric", as_index=False)[
        ["rho_with_final_true_margin", "rho_with_error"]
    ].agg(["mean", "std"])
    relation_summary.columns = ["_".join(column) for column in relation_summary.columns]
    relation_summary.to_csv(output / "metric_relationship_summary.csv")

    primary = comparison_frame[
        comparison_frame.comparison.eq("stable_error-stable_correct")
    ].set_index("metric")
    summary = {
        "protocol": {
            "scope": "fixed inner-development train/validation only",
            "outer_test_accessed": False,
            "model_retrained": False,
            "model_parameters_updated": False,
            "probe_fit": "separate train-only logistic probe per activity and wrist/component",
            "raw_seed_fold_rows": 45,
            "primary_independent_splits": 15,
            "seed_aggregation": "mean across seeds 42/43/44 within identical split",
            "multiple_comparison_control": "BH-FDR within subject-level and activity-level test families",
            "stability_group_source": str(args.stability_csv.resolve()),
            "historical_outer_runner_incident_retained": True,
        },
        "stable_error_minus_stable_correct": {
            metric: {
                "mean_delta": float(primary.loc[metric, "mean_delta"]),
                "positive_splits": int(primary.loc[metric, "positive_splits"]),
                "wilcoxon_p": float(primary.loc[metric, "wilcoxon_p"]),
                "bh_fdr_q": float(primary.loc[metric, "bh_fdr_q"]),
            }
            for metric in SUBJECT_METRICS
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "protocol.json").write_text(
        json.dumps(summary["protocol"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
