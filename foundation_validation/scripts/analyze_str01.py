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
    "time_shape": {"skewness", "kurtosis_excess", "zero_crossing_rate_centered"},
    "derivative": {"derivative_rms", "derivative_std"},
    "spectral_global": {
        "dominant_frequency_hz_0p5_20", "dominant_power_0p5_20",
        "spectral_entropy_0p5_20", "total_power_0p5_20",
    },
    "band_absolute": {
        "band_power_0p5_3", "band_power_3_7", "band_power_7_12", "band_power_12_20",
    },
    "band_fraction": {
        "band_fraction_0p5_3", "band_fraction_3_7", "band_fraction_7_12",
        "band_fraction_12_20",
    },
}
CLASSIFICATION_METRICS = (
    "accuracy", "balanced_accuracy", "macro_f1", "macro_auroc", "pd_recall", "dd_recall"
)


def arguments():
    parser = argparse.ArgumentParser(description="STR-01 preregistered evaluation")
    parser.add_argument("--str-run", action="append", required=True)
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
    str_maps = [fold_map(read(Path(path) / "development_summary.json")) for path in args.str_run]
    base_maps = [fold_map(read(Path(path) / "development_summary.json")) for path in args.baseline_run]
    keys = sorted(str_maps[0])
    rows = []
    for outer, inner in keys:
        row = {"outer": outer, "inner": inner}
        for metric in CLASSIFICATION_METRICS:
            base = np.mean([mapping[(outer, inner)][metric] for mapping in base_maps])
            model = np.mean([mapping[(outer, inner)][metric] for mapping in str_maps])
            row[f"baseline_{metric}"] = base
            row[f"str01_{metric}"] = model
            row[f"delta_{metric}"] = model - base
        rows.append(row)
    splits = pd.DataFrame(rows)
    splits.to_csv(output / "classification_15_paired_splits.csv", index=False)
    inference = []
    for metric in CLASSIFICATION_METRICS:
        delta = splits[f"delta_{metric}"].to_numpy()
        inference.append({
            "metric": metric, "baseline_mean": splits[f"baseline_{metric}"].mean(),
            "str01_mean": splits[f"str01_{metric}"].mean(), "mean_delta": delta.mean(),
            "wins": int((delta > 0).sum()), "losses": int((delta < 0).sum()),
            "wilcoxon_p": paired_p(delta), "rank_biserial": rank_biserial(delta),
        })
    inference = pd.DataFrame(inference)
    inference["bh_q_across_6_metrics"] = bh(inference.wilcoxon_p)
    inference.to_csv(output / "classification_paired_inference.csv", index=False)

    seed_rows = []
    for name, maps in (("V8-GN", base_maps), ("STR-01", str_maps)):
        for seed_index, mapping in enumerate(maps):
            row = {"model": name, "seed_index": seed_index}
            for metric in CLASSIFICATION_METRICS:
                row[metric] = np.mean([mapping[key][metric] for key in keys])
            seed_rows.append(row)
    seed_frame = pd.DataFrame(seed_rows)
    seed_frame.to_csv(output / "classification_seed_metrics.csv", index=False)
    summary = seed_frame.groupby("model")[list(CLASSIFICATION_METRICS)].agg(["mean", "std"])
    summary.to_csv(output / "classification_seed_summary.csv")
    return splits, inference, seed_frame


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
        "r2_variance_weighted": r2_score(true, prediction, multioutput="variance_weighted"),
        "median_feature_r2": np.median(r2),
        "median_feature_spearman": np.median(rho),
        "normalized_mae": np.mean(np.abs(true - prediction)),
        "target_dimension": true.shape[1],
    }


def fit_probe(train_x, train_y, validation_x, validation_y):
    scaler = StandardScaler().fit(train_x)
    train_y, validation_y = standardize_targets(train_y, validation_y)
    model = Ridge(alpha=1.0).fit(scaler.transform(train_x), train_y)
    return regression_metrics(validation_y, model.predict(scaler.transform(validation_x)))


def recoverability(args, output):
    feature = np.load(args.h1_analysis / "features/handcrafted_features.npz")
    x = feature["X"]
    subjects = feature["subject_ids"].astype(str)
    schema = read(args.h1_analysis / "features/feature_schema.json")
    columns = schema["feature_names"] if "feature_names" in schema else schema["columns"]
    parsed = [column.split("|", 3) for column in columns]
    groups = {
        family: np.asarray([i for i, parts in enumerate(parsed) if parts[3] in names])
        for family, names in FAMILIES.items()
    }
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    rows = []
    pattern = re.compile(r"seed(\d+)_outer(\d+)_inner(\d+)\.npz")
    files = sorted(args.embedding_dir.glob("seed*_outer*_inner*.npz"))
    representations = {
        "str_subject_embedding": "subject_embedding",
        "str_structured_residual": "structured_embedding",
        "str_decision_input": "decision_input",
    }
    for number, path in enumerate(files, 1):
        seed, outer, inner = map(int, pattern.fullmatch(path.name).groups())
        payload = np.load(path)
        train_index = np.asarray([subject_index[value] for value in payload["train_subject_id"].astype(str)])
        validation_index = np.asarray([
            subject_index[value] for value in payload["validation_subject_id"].astype(str)
        ])
        for family, selected in groups.items():
            train_y, validation_y = x[np.ix_(train_index, selected)], x[np.ix_(validation_index, selected)]
            for representation, key in representations.items():
                result = fit_probe(
                    payload[f"train_{key}"], train_y,
                    payload[f"validation_{key}"], validation_y,
                )
                rows.append({"seed": seed, "outer": outer, "inner": inner,
                             "representation": representation, "family": family, **result})
        print(f"recoverability {number}/{len(files)} {path.name}", flush=True)
    raw = pd.DataFrame(rows)
    baseline = pd.read_csv(args.baseline_recoverability)
    baseline = baseline[baseline.representation == "subject_embedding"].copy()
    baseline["representation"] = "v8_subject_embedding"
    raw = pd.concat([baseline[raw.columns], raw], ignore_index=True)
    raw.to_csv(output / "recoverability_45_seed_folds.csv", index=False)
    metrics = ["r2_variance_weighted", "median_feature_r2", "median_feature_spearman",
               "normalized_mae", "target_dimension"]
    splits = raw.groupby(["outer", "inner", "representation", "family"], as_index=False)[metrics].mean()
    splits.to_csv(output / "recoverability_15_splits.csv", index=False)
    summary = splits.groupby(["representation", "family"], as_index=False).agg(
        r2_mean=("r2_variance_weighted", "mean"), r2_sd=("r2_variance_weighted", "std"),
        spearman_mean=("median_feature_spearman", "mean"),
        spearman_sd=("median_feature_spearman", "std"),
        normalized_mae_mean=("normalized_mae", "mean"),
        normalized_mae_sd=("normalized_mae", "std"),
    )
    summary.to_csv(output / "recoverability_summary.csv", index=False)
    return splits, summary


def mechanism_relationship(classification_splits, recover_splits, output):
    key_families = ("time_location_scale", "band_fraction")
    base = recover_splits[recover_splits.representation == "v8_subject_embedding"].set_index(
        ["outer", "inner", "family"]
    )
    rows = []
    for representation in ("str_structured_residual", "str_decision_input"):
        current = recover_splits[recover_splits.representation == representation].set_index(
            ["outer", "inner", "family"]
        )
        for family in key_families:
            class_order = classification_splits.set_index(["outer", "inner"])
            for recover_metric in ("r2_variance_weighted", "median_feature_spearman", "normalized_mae"):
                recover_delta = np.asarray([
                    current.loc[(outer, inner, family), recover_metric]
                    - base.loc[(outer, inner, family), recover_metric]
                    for outer, inner in class_order.index
                ])
                for class_metric in ("balanced_accuracy", "macro_auroc"):
                    class_delta = class_order[f"delta_{class_metric}"].to_numpy()
                    rho, p = spearmanr(recover_delta, class_delta)
                    rows.append({
                        "representation": representation, "family": family,
                        "recoverability_metric": recover_metric,
                        "classification_metric": class_metric,
                        "spearman_rho": rho, "p_value": p,
                    })
    result = pd.DataFrame(rows)
    result["bh_q_exploratory"] = bh(result.p_value)
    result.to_csv(output / "recoverability_classification_relationships.csv", index=False)
    return result


def main():
    args = arguments()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    classification_splits, inference, seed_frame = classification(args, output)
    recover_splits, recover_summary = recoverability(args, output)
    relationships = mechanism_relationship(classification_splits, recover_splits, output)
    protocol = {
        "scope": "fixed 5x3 inner-development splits; three seeds averaged before split inference",
        "outer_information_used": False,
        "classification_independent_units": 15,
        "recoverability_probe": "fixed Ridge(alpha=1.0), train-only representation and target scaling",
        "multiple_comparison": "BH across six preregistered classification metrics; mechanism correlations exploratory BH",
        "parameter_count_v8": 71026,
        "parameter_count_str01": 75524,
        "parameter_increment": 4498,
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(inference.to_string(index=False))
    print(recover_summary[recover_summary.family.isin(["time_location_scale", "band_fraction"])].to_string(index=False))
    print(json.dumps(protocol, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
