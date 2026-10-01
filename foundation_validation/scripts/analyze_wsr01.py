#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

FAMILIES = {
    "time_location_scale": {
        "mean", "std", "variance", "rms", "median", "mad", "iqr",
        "minimum", "maximum", "peak_to_peak", "signal_energy",
    },
    "band_fraction": {
        "band_fraction_0p5_3", "band_fraction_3_7", "band_fraction_7_12",
        "band_fraction_12_20",
    },
}
CLASSIFICATION_METRICS = (
    "accuracy", "balanced_accuracy", "macro_f1", "macro_auroc",
    "pd_recall", "dd_recall",
)
RECOVERABILITY_METRICS = (
    "r2_variance_weighted", "median_feature_spearman", "normalized_mae",
)


def arguments():
    parser = argparse.ArgumentParser(description="WSR-01 preregistered evaluation")
    parser.add_argument("--wsr-run", action="append", required=True)
    parser.add_argument("--baseline-run", action="append", required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--baseline-recoverability", type=Path, required=True)
    parser.add_argument("--h1-analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fold_map(summary):
    result = {}
    for row in summary["fold_summaries"]:
        if "outer_context" in row:
            outer, inner = int(row["outer_context"]), int(row["inner_fold"])
        elif "outer_fold" in row:
            outer, inner = int(row["outer_fold"]), int(row["inner_fold"])
        else:
            outer_text, inner_text = row["fold_id"].split("/")
            outer = int(outer_text.removeprefix("outer_"))
            inner = int(inner_text.removeprefix("inner_"))
        result[(outer, inner)] = row["validation_metrics"]
    return result


def seed_from_path(path, fallback):
    match = re.search(r"seed(42|43|44)", str(path))
    return int(match.group(1)) if match else fallback


def paired_p(delta):
    values = np.asarray(delta, dtype=float)
    nonzero = values[np.abs(values) > 1e-12]
    return 1.0 if not len(nonzero) else float(wilcoxon(nonzero).pvalue)


def rank_biserial(delta):
    values = np.asarray(delta, dtype=float)
    values = values[np.abs(values) > 1e-12]
    if not len(values):
        return 0.0
    ranks = rankdata(np.abs(values), method="average")
    return float((ranks[values > 0].sum() - ranks[values < 0].sum()) / ranks.sum())


def bh(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    result = np.empty(len(values), dtype=float)
    running = 1.0
    for position in range(len(values) - 1, -1, -1):
        index = order[position]
        running = min(running, values[index] * len(values) / (position + 1))
        result[index] = min(1.0, running)
    return result


def classification(args, output):
    wsr_maps = [fold_map(read(Path(path) / "development_summary.json"))
                for path in args.wsr_run]
    str_maps = [fold_map(read(Path(path) / "development_summary.json"))
                for path in args.baseline_run]
    keys = sorted(wsr_maps[0])
    rows = []
    for outer, inner in keys:
        row = {"outer": outer, "inner": inner}
        for metric in CLASSIFICATION_METRICS:
            baseline = np.mean([mapping[(outer, inner)][metric] for mapping in str_maps])
            model = np.mean([mapping[(outer, inner)][metric] for mapping in wsr_maps])
            row[f"str01_{metric}"] = baseline
            row[f"wsr01_{metric}"] = model
            row[f"delta_{metric}"] = model - baseline
        rows.append(row)
    splits = pd.DataFrame(rows)
    splits.to_csv(output / "classification_15_paired_splits.csv", index=False)
    inference = []
    for metric in CLASSIFICATION_METRICS:
        delta = splits[f"delta_{metric}"].to_numpy()
        inference.append({
            "metric": metric,
            "str01_mean": splits[f"str01_{metric}"].mean(),
            "wsr01_mean": splits[f"wsr01_{metric}"].mean(),
            "mean_delta": delta.mean(),
            "wins": int((delta > 1e-12).sum()),
            "losses": int((delta < -1e-12).sum()),
            "ties": int((np.abs(delta) <= 1e-12).sum()),
            "wilcoxon_p": paired_p(delta),
            "rank_biserial": rank_biserial(delta),
        })
    inference = pd.DataFrame(inference)
    inference["bh_q_across_6_metrics"] = bh(inference.wilcoxon_p)
    inference.to_csv(output / "classification_paired_inference.csv", index=False)
    seed_rows = []
    for name, paths, maps in (
        ("STR-01", args.baseline_run, str_maps),
        ("WSR-01", args.wsr_run, wsr_maps),
    ):
        for index, (path, mapping) in enumerate(zip(paths, maps)):
            row = {"model": name, "seed": seed_from_path(path, index)}
            for metric in CLASSIFICATION_METRICS:
                row[metric] = np.mean([mapping[key][metric] for key in keys])
            seed_rows.append(row)
    seed_frame = pd.DataFrame(seed_rows)
    seed_frame.to_csv(output / "classification_seed_metrics.csv", index=False)
    seed_frame.groupby("model")[list(CLASSIFICATION_METRICS)].agg(
        ["mean", "std"]
    ).to_csv(output / "classification_seed_summary.csv")
    return splits, inference


def standardize_targets(train, validation):
    mean, scale = train.mean(0), train.std(0)
    scale[scale < 1e-8] = 1.0
    return (train - mean) / scale, (validation - mean) / scale


def regression_metrics(true, prediction):
    r2, rho = [], []
    for index in range(true.shape[1]):
        if np.std(true[:, index]) > 1e-10:
            r2.append(r2_score(true[:, index], prediction[:, index]))
            value = spearmanr(true[:, index], prediction[:, index]).statistic
            if np.isfinite(value):
                rho.append(value)
    return {
        "r2_variance_weighted": r2_score(
            true, prediction, multioutput="variance_weighted"
        ),
        "median_feature_r2": np.median(r2),
        "median_feature_spearman": np.median(rho),
        "normalized_mae": np.mean(np.abs(true - prediction)),
        "target_dimension": true.shape[1],
    }


def fit_probe(train_x, train_y, validation_x, validation_y):
    scaler = StandardScaler().fit(train_x)
    train_y, validation_y = standardize_targets(train_y, validation_y)
    model = Ridge(alpha=1.0).fit(scaler.transform(train_x), train_y)
    return regression_metrics(
        validation_y, model.predict(scaler.transform(validation_x))
    )


def recoverability(args, output):
    feature = np.load(args.h1_analysis / "features/handcrafted_features.npz")
    x = feature["X"]
    subjects = feature["subject_ids"].astype(str)
    schema = read(args.h1_analysis / "features/feature_schema.json")
    columns = schema.get("feature_names", schema.get("columns"))
    parsed = [column.split("|", 3) for column in columns]
    groups = {
        family: np.asarray([
            i for i, parts in enumerate(parsed) if parts[3] in names
        ]) for family, names in FAMILIES.items()
    }
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    representations = {
        "wsr_subject_embedding": "subject_embedding",
        "wsr_activity_structured": "structured_embedding",
        "wsr_wrist_structured": "wrist_structured_embedding",
        "wsr_str_decision_input": "str_decision_input",
        "wsr_decision_input": "decision_input",
    }
    pattern = re.compile(r"seed(\d+)_outer(\d+)_inner(\d+)\.npz")
    rows = []
    files = sorted(args.embedding_dir.glob("seed*_outer*_inner*.npz"))
    for number, path in enumerate(files, 1):
        seed, outer, inner = map(int, pattern.fullmatch(path.name).groups())
        payload = np.load(path)
        train_index = np.asarray([
            subject_index[value] for value in payload["train_subject_id"].astype(str)
        ])
        validation_index = np.asarray([
            subject_index[value]
            for value in payload["validation_subject_id"].astype(str)
        ])
        for family, selected in groups.items():
            train_y = x[np.ix_(train_index, selected)]
            validation_y = x[np.ix_(validation_index, selected)]
            for representation, key in representations.items():
                result = fit_probe(
                    payload[f"train_{key}"], train_y,
                    payload[f"validation_{key}"], validation_y,
                )
                rows.append({
                    "seed": seed, "outer": outer, "inner": inner,
                    "representation": representation, "family": family,
                    **result,
                })
        print(f"recoverability {number}/{len(files)} {path.name}", flush=True)
    raw = pd.DataFrame(rows)
    baseline = pd.read_csv(args.baseline_recoverability)
    baseline = baseline[
        baseline.family.isin(FAMILIES)
        & baseline.representation.isin({
            "str_subject_embedding", "str_structured_residual", "str_decision_input"
        })
    ].copy()
    raw = pd.concat([baseline[raw.columns], raw], ignore_index=True)
    raw.to_csv(output / "recoverability_45_seed_folds.csv", index=False)
    metrics = [
        "r2_variance_weighted", "median_feature_r2",
        "median_feature_spearman", "normalized_mae", "target_dimension",
    ]
    splits = raw.groupby(
        ["outer", "inner", "representation", "family"], as_index=False
    )[metrics].mean()
    splits.to_csv(output / "recoverability_15_splits.csv", index=False)
    summary = splits.groupby(
        ["representation", "family"], as_index=False
    ).agg(
        r2_mean=("r2_variance_weighted", "mean"),
        r2_sd=("r2_variance_weighted", "std"),
        spearman_mean=("median_feature_spearman", "mean"),
        spearman_sd=("median_feature_spearman", "std"),
        normalized_mae_mean=("normalized_mae", "mean"),
        normalized_mae_sd=("normalized_mae", "std"),
    )
    summary.to_csv(output / "recoverability_summary.csv", index=False)
    return splits, summary


def recoverability_inference(splits, output):
    comparisons = (
        ("wrist_vs_str_activity", "wsr_wrist_structured", "str_structured_residual"),
        ("full_decision_vs_str", "wsr_decision_input", "str_decision_input"),
        ("activity_control", "wsr_activity_structured", "str_structured_residual"),
    )
    indexed = splits.set_index(["outer", "inner", "representation", "family"])
    units = sorted({(int(row.outer), int(row.inner)) for row in splits.itertuples()})
    rows = []
    for comparison, model, baseline in comparisons:
        for family in FAMILIES:
            for metric in RECOVERABILITY_METRICS:
                raw_delta = np.asarray([
                    indexed.loc[(outer, inner, model, family), metric]
                    - indexed.loc[(outer, inner, baseline, family), metric]
                    for outer, inner in units
                ])
                oriented = -raw_delta if metric == "normalized_mae" else raw_delta
                rows.append({
                    "comparison": comparison, "model": model,
                    "baseline": baseline, "family": family, "metric": metric,
                    "model_minus_baseline": raw_delta.mean(),
                    "oriented_improvement": oriented.mean(),
                    "wins": int((oriented > 1e-12).sum()),
                    "losses": int((oriented < -1e-12).sum()),
                    "ties": int((np.abs(oriented) <= 1e-12).sum()),
                    "wilcoxon_p": paired_p(oriented),
                    "rank_biserial_improvement": rank_biserial(oriented),
                })
    result = pd.DataFrame(rows)
    result["bh_q_across_18_tests"] = bh(result.wilcoxon_p)
    result.to_csv(output / "recoverability_paired_inference.csv", index=False)
    return result


def mechanism_relationship(classification_splits, recover_splits, output):
    indexed = recover_splits.set_index(
        ["outer", "inner", "representation", "family"]
    )
    class_order = classification_splits.set_index(["outer", "inner"])
    rows = []
    for comparison, model, baseline in (
        ("wrist_vs_str_activity", "wsr_wrist_structured", "str_structured_residual"),
        ("full_decision_vs_str", "wsr_decision_input", "str_decision_input"),
    ):
        for family in FAMILIES:
            for recover_metric in RECOVERABILITY_METRICS:
                recover_delta = np.asarray([
                    indexed.loc[(outer, inner, model, family), recover_metric]
                    - indexed.loc[(outer, inner, baseline, family), recover_metric]
                    for outer, inner in class_order.index
                ])
                if recover_metric == "normalized_mae":
                    recover_delta = -recover_delta
                for class_metric in ("balanced_accuracy", "macro_auroc"):
                    rho, p = spearmanr(
                        recover_delta,
                        class_order[f"delta_{class_metric}"].to_numpy(),
                    )
                    rows.append({
                        "comparison": comparison, "family": family,
                        "recoverability_metric": recover_metric,
                        "classification_metric": class_metric,
                        "spearman_rho": rho, "p_value": p,
                    })
    result = pd.DataFrame(rows)
    result["bh_q_exploratory"] = bh(result.p_value)
    result.to_csv(
        output / "recoverability_classification_relationships.csv", index=False
    )


def main():
    args = arguments()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    classification_splits, inference = classification(args, output)
    recover_splits, recover_summary = recoverability(args, output)
    recover_inference = recoverability_inference(recover_splits, output)
    mechanism_relationship(classification_splits, recover_splits, output)
    protocol = {
        "scope": "fixed 5x3 inner-development splits; three seeds averaged before split inference",
        "outer_information_used": False,
        "classification_independent_units": 15,
        "recoverability_targets": ["time_location_scale", "band_fraction"],
        "recoverability_probe": "fixed Ridge(alpha=1.0), train-only representation and target scaling",
        "multiple_comparison": "BH across 6 classification tests and separately across 18 recoverability tests; mechanism correlations exploratory BH",
        "parameter_count_str01": 75524,
        "parameter_count_wsr01": 75962,
        "parameter_increment": 438,
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(inference.to_string(index=False))
    print(recover_summary.to_string(index=False))
    print(recover_inference.to_string(index=False))
    print(json.dumps(protocol, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
