#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata, spearmanr, wilcoxon
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FAMILY_DEFINITIONS = {
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
PRIMARY_METRICS = ("balanced_accuracy", "auroc", "macro_f1", "pd_recall", "dd_recall")
SIGNAL_FEATURES = (
    "motion_rms", "jerk_rms", "robust_outlier_fraction", "bandpower_0p5_3",
    "bandpower_3_7", "dominant_frequency_0p5_12", "spectral_entropy_0p5_25",
    "tremor_peak_ratio_3_7", "zero_lag_magnitude_correlation",
)


def parse_args():
    parser = argparse.ArgumentParser(description="H1 versus frozen V8-GN representation gap diagnosis")
    parser.add_argument("--h1-analysis", type=Path, required=True)
    parser.add_argument("--family-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--stable-audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def bh_adjust(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 1.0
    for position in range(len(values) - 1, -1, -1):
        index = int(order[position])
        running = min(running, values[index] * len(values) / (position + 1))
        adjusted[index] = min(1.0, running)
    return adjusted


def rank_biserial(delta):
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.abs(delta) > 1.0e-12]
    if not len(delta):
        return 0.0
    ranks = rankdata(np.abs(delta), method="average")
    positive = ranks[delta > 0].sum()
    negative = ranks[delta < 0].sum()
    return float((positive - negative) / (positive + negative))


def cliffs_delta(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if not len(left) or not len(right):
        return np.nan
    return float((np.greater(left[:, None], right[None, :]).sum()
                  - np.less(left[:, None], right[None, :]).sum()) / (len(left) * len(right)))


def full_estimator():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=1.0, class_weight=None, max_iter=5000,
                                     solver="liblinear", random_state=42)),
    ])


def family_indices(columns):
    parsed = [column.split("|", 3) for column in columns]
    return {
        family: np.asarray([i for i, parts in enumerate(parsed) if parts[3] in names], dtype=int)
        for family, names in FAMILY_DEFINITIONS.items()
    }, parsed


def paired_family_inference(family_dir, output):
    folds = pd.read_csv(family_dir / "fold_metrics.csv")
    full = folds[folds.variant == "all"].set_index(["outer_context", "inner_fold"])
    rows = []
    for variant in sorted(value for value in folds.variant.unique()
                          if value.startswith("only::") or value.startswith("without::")):
        subset = folds[folds.variant == variant].set_index(["outer_context", "inner_fold"])
        for metric in PRIMARY_METRICS:
            delta = (subset[metric] - full[metric]).to_numpy(dtype=float)
            nonzero = delta[np.abs(delta) > 1.0e-12]
            p_value = float(wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0
            rows.append({
                "variant": variant,
                "family": variant.split("::", 1)[1],
                "analysis": variant.split("::", 1)[0],
                "metric": metric,
                "mean_delta_vs_full": float(delta.mean()),
                "median_delta_vs_full": float(np.median(delta)),
                "wins": int((delta > 0).sum()),
                "paired_rank_biserial": rank_biserial(delta),
                "wilcoxon_p": p_value,
            })
    result = pd.DataFrame(rows)
    result["bh_q_across_family_variant_metrics"] = bh_adjust(result.wilcoxon_p)
    result.to_csv(output / "family_paired_inference.csv", index=False)
    return result


def standardize_targets(train, validation):
    mean = train.mean(axis=0)
    scale = train.std(axis=0, ddof=0)
    scale[scale < 1.0e-8] = 1.0
    return (train - mean) / scale, (validation - mean) / scale


def regression_metrics(y_true, y_pred):
    per_r2 = []
    per_spearman = []
    for index in range(y_true.shape[1]):
        if np.std(y_true[:, index]) > 1.0e-10:
            per_r2.append(r2_score(y_true[:, index], y_pred[:, index]))
            rho = spearmanr(y_true[:, index], y_pred[:, index]).statistic
            if np.isfinite(rho):
                per_spearman.append(float(rho))
    return {
        "r2_variance_weighted": float(r2_score(y_true, y_pred, multioutput="variance_weighted")),
        "median_feature_r2": float(np.median(per_r2)) if per_r2 else np.nan,
        "median_feature_spearman": float(np.median(per_spearman)) if per_spearman else np.nan,
        "normalized_mae": float(np.mean(np.abs(y_true - y_pred))),
        "target_dimension": int(y_true.shape[1]),
        "validation_scalar_count": int(y_true.size),
    }


def fit_recoverability(train_x, train_y, validation_x, validation_y):
    x_scaler = StandardScaler().fit(train_x)
    z_train_x = x_scaler.transform(train_x)
    z_validation_x = x_scaler.transform(validation_x)
    z_train_y, z_validation_y = standardize_targets(train_y, validation_y)
    model = Ridge(alpha=1.0, fit_intercept=True).fit(z_train_x, z_train_y)
    prediction = model.predict(z_validation_x)
    return regression_metrics(z_validation_y, prediction)


def recoverability(args, X, subjects, columns, parsed, groups, output):
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    activities = list(dict.fromkeys(parts[0] for parts in parsed))
    wrists = list(dict.fromkeys(parts[1] for parts in parsed))
    by_stratum = {}
    for family, family_set in FAMILY_DEFINITIONS.items():
        for activity in activities:
            for wrist in wrists:
                by_stratum[(family, activity, wrist)] = np.asarray([
                    i for i, parts in enumerate(parsed)
                    if parts[0] == activity and parts[1] == wrist and parts[3] in family_set
                ], dtype=int)

    rows = []
    files = sorted(args.embedding_dir.glob("seed*_outer*_inner*.npz"))
    pattern = re.compile(r"seed(\d+)_outer(\d+)_inner(\d+)\.npz")
    for file_index, path in enumerate(files, 1):
        match = pattern.fullmatch(path.name)
        seed, outer, inner = map(int, match.groups())
        payload = np.load(path, allow_pickle=False)
        train_ids = payload["train_subject_id"].astype(str)
        validation_ids = payload["validation_subject_id"].astype(str)
        train_index = np.asarray([subject_index[value] for value in train_ids])
        validation_index = np.asarray([subject_index[value] for value in validation_ids])
        for family, selected in groups.items():
            # Subject representation predicts the complete subject-level family vector.
            result = fit_recoverability(
                payload["train_subject_embedding"], X[np.ix_(train_index, selected)],
                payload["validation_subject_embedding"], X[np.ix_(validation_index, selected)],
            )
            rows.append({"seed": seed, "outer": outer, "inner": inner,
                         "representation": "subject_embedding", "family": family, **result})

            # Activity representations predict both wrists' family statistics; targets
            # are standardized separately within each activity using inner-train only.
            for representation, key in (("bilateral_activity", "activity_fused"),
                                        ("activity_context", "activity_context")):
                train_x_parts, validation_x_parts, train_y_parts, validation_y_parts = [], [], [], []
                for activity_index, activity in enumerate(activities):
                    indices = np.concatenate([
                        by_stratum[(family, activity, wrist)] for wrist in wrists
                    ])
                    train_target, validation_target = standardize_targets(
                        X[np.ix_(train_index, indices)], X[np.ix_(validation_index, indices)]
                    )
                    train_x_parts.append(payload[f"train_{key}"][:, activity_index])
                    validation_x_parts.append(payload[f"validation_{key}"][:, activity_index])
                    train_y_parts.append(train_target)
                    validation_y_parts.append(validation_target)
                train_x = np.concatenate(train_x_parts)
                validation_x = np.concatenate(validation_x_parts)
                train_y = np.concatenate(train_y_parts)
                validation_y = np.concatenate(validation_y_parts)
                x_scaler = StandardScaler().fit(train_x)
                model = Ridge(alpha=1.0).fit(x_scaler.transform(train_x), train_y)
                result = regression_metrics(validation_y, model.predict(x_scaler.transform(validation_x)))
                rows.append({"seed": seed, "outer": outer, "inner": inner,
                             "representation": representation, "family": family, **result})

            # Wrist tokens predict same-wrist statistics, pooled across 22 fixed strata.
            train_x_parts, validation_x_parts, train_y_parts, validation_y_parts = [], [], [], []
            for activity_index, activity in enumerate(activities):
                for wrist_index, wrist in enumerate(wrists):
                    indices = by_stratum[(family, activity, wrist)]
                    train_target, validation_target = standardize_targets(
                        X[np.ix_(train_index, indices)], X[np.ix_(validation_index, indices)]
                    )
                    train_x_parts.append(payload["train_wrist_tokens"][:, activity_index, wrist_index])
                    validation_x_parts.append(payload["validation_wrist_tokens"][:, activity_index, wrist_index])
                    train_y_parts.append(train_target)
                    validation_y_parts.append(validation_target)
            train_x = np.concatenate(train_x_parts)
            validation_x = np.concatenate(validation_x_parts)
            train_y = np.concatenate(train_y_parts)
            validation_y = np.concatenate(validation_y_parts)
            x_scaler = StandardScaler().fit(train_x)
            model = Ridge(alpha=1.0).fit(x_scaler.transform(train_x), train_y)
            result = regression_metrics(validation_y, model.predict(x_scaler.transform(validation_x)))
            rows.append({"seed": seed, "outer": outer, "inner": inner,
                         "representation": "wrist_encoder", "family": family, **result})
        print(f"recoverability {file_index}/{len(files)} {path.name}", flush=True)

    raw = pd.DataFrame(rows)
    raw.to_csv(output / "recoverability_45_seed_folds.csv", index=False)
    metrics = ["r2_variance_weighted", "median_feature_r2", "median_feature_spearman",
               "normalized_mae", "target_dimension", "validation_scalar_count"]
    split = raw.groupby(["outer", "inner", "representation", "family"], as_index=False)[metrics].mean()
    split.to_csv(output / "recoverability_15_splits.csv", index=False)
    summary = split.groupby(["representation", "family"], as_index=False).agg(
        r2_mean=("r2_variance_weighted", "mean"),
        r2_sd=("r2_variance_weighted", "std"),
        median_feature_r2_mean=("median_feature_r2", "mean"),
        spearman_mean=("median_feature_spearman", "mean"),
        spearman_sd=("median_feature_spearman", "std"),
        normalized_mae_mean=("normalized_mae", "mean"),
        normalized_mae_sd=("normalized_mae", "std"),
    )
    summary.to_csv(output / "recoverability_summary.csv", index=False)
    return raw, split, summary


def h1_v8_complementarity(args, X, y, subjects, columns, parsed, groups, output):
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    column_family = {}
    for family, indices in groups.items():
        for index in indices:
            column_family[int(index)] = family
    files_by_split = {}
    pattern = re.compile(r"seed(\d+)_outer(\d+)_inner(\d+)\.npz")
    for path in sorted(args.embedding_dir.glob("seed*_outer*_inner*.npz")):
        seed, outer, inner = map(int, pattern.fullmatch(path.name).groups())
        files_by_split.setdefault((outer, inner), {})[seed] = path

    metadata = pd.read_csv(args.stable_audit_dir / "subject_metadata_and_stability.csv",
                           dtype={"subject_id": str})
    metadata.subject_id = metadata.subject_id.str.zfill(3)
    metadata_lookup = metadata.set_index("subject_id")
    records = []
    for (outer, inner), seed_paths in sorted(files_by_split.items()):
        seed_payloads = {seed: np.load(path, allow_pickle=False) for seed, path in seed_paths.items()}
        reference = seed_payloads[42]
        train_ids = reference["train_subject_id"].astype(str)
        validation_ids = reference["validation_subject_id"].astype(str)
        train_index = np.asarray([subject_index[value] for value in train_ids])
        validation_index = np.asarray([subject_index[value] for value in validation_ids])
        model = full_estimator().fit(X[train_index], y[train_index])
        h1_probability = model.predict_proba(X[validation_index])[:, 1]
        transformed = model.named_steps["scaler"].transform(
            model.named_steps["variance"].transform(
                model.named_steps["imputer"].transform(X[validation_index])
            )
        )
        retained = np.flatnonzero(model.named_steps["variance"].get_support())
        contribution = transformed * model.named_steps["model"].coef_[0]
        family_contribution = {family: np.zeros(len(validation_index)) for family in groups}
        activity_contribution = {activity: np.zeros(len(validation_index))
                                 for activity in dict.fromkeys(parts[0] for parts in parsed)}
        for transformed_index, original_index in enumerate(retained):
            family_contribution[column_family[int(original_index)]] += contribution[:, transformed_index]
            activity_contribution[parsed[int(original_index)][0]] += contribution[:, transformed_index]

        probabilities = []
        for seed, payload in seed_payloads.items():
            if not np.array_equal(payload["validation_subject_id"].astype(str), validation_ids):
                raise ValueError(f"Validation ordering mismatch in {outer}.{inner} seed {seed}")
            probabilities.append(payload["validation_probability_dd"])
        v8_probability = np.mean(probabilities, axis=0)
        for row_index, subject_id in enumerate(validation_ids):
            target = int(y[validation_index[row_index]])
            h1_prediction = int(h1_probability[row_index] >= 0.5)
            v8_prediction = int(v8_probability[row_index] >= 0.5)
            if h1_prediction == target and v8_prediction == target:
                group = "both_correct"
            elif h1_prediction == target:
                group = "h1_only_correct"
            elif v8_prediction == target:
                group = "v8_only_correct"
            else:
                group = "both_wrong"
            record = {
                "outer": outer, "inner": inner, "subject_id": subject_id,
                "target": target, "label_name": "DD" if target else "PD",
                "h1_probability_dd": float(h1_probability[row_index]),
                "v8_probability_dd": float(v8_probability[row_index]),
                "h1_correct": h1_prediction == target,
                "v8_correct": v8_prediction == target,
                "correctness_group": group,
                "stability_group": metadata_lookup.loc[subject_id, "stability_group"],
            }
            direction = 1.0 if target == 1 else -1.0
            for family in groups:
                record[f"h1_true_direction_contribution_family::{family}"] = (
                    direction * float(family_contribution[family][row_index])
                )
            for activity in activity_contribution:
                record[f"h1_true_direction_contribution_activity::{activity}"] = (
                    direction * float(activity_contribution[activity][row_index])
                )
            records.append(record)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "h1_v8_validation_records.csv", index=False)

    overlap = frame.groupby(["correctness_group", "label_name"]).size().rename("rows").reset_index()
    overlap.to_csv(output / "correctness_overlap.csv", index=False)
    stable = frame[~frame.v8_correct].groupby(
        ["stability_group", "label_name"], as_index=False
    ).agg(v8_wrong_rows=("subject_id", "size"), h1_correction_rate=("h1_correct", "mean"),
          unique_subjects=("subject_id", "nunique"))
    stable.to_csv(output / "h1_correction_of_v8_errors_by_stability.csv", index=False)

    group_columns = [column for column in frame if "true_direction_contribution" in column]
    contribution = frame.groupby(["correctness_group", "label_name"], as_index=False)[group_columns].mean()
    contribution.to_csv(output / "h1_contribution_profiles.csv", index=False)

    count_table = frame.pivot_table(index="subject_id", columns="correctness_group", values="outer",
                                    aggfunc="count", fill_value=0)
    for name in ("both_correct", "h1_only_correct", "v8_only_correct", "both_wrong"):
        if name not in count_table:
            count_table[name] = 0
    count_table = count_table.reset_index()
    count_table["appearance_count"] = count_table[["both_correct", "h1_only_correct",
                                                    "v8_only_correct", "both_wrong"]].sum(axis=1)
    def persistent_category(row):
        names = ["both_correct", "h1_only_correct", "v8_only_correct", "both_wrong"]
        best = max(names, key=lambda name: row[name])
        return best if row[best] >= 3 else "mixed"
    count_table["persistent_category"] = count_table.apply(persistent_category, axis=1)
    count_table = count_table.merge(metadata[["subject_id", "label_name", "stability_group"]],
                                    on="subject_id", how="left")
    count_table.to_csv(output / "subject_error_complementarity.csv", index=False)

    geometry = pd.read_csv(args.stable_audit_dir / "representation_geometry_subject.csv",
                           dtype={"subject_id": str})
    geometry.subject_id = geometry.subject_id.str.zfill(3)
    geometry = geometry.merge(count_table[["subject_id", "persistent_category"]], on="subject_id")
    geometry.groupby(["persistent_category", "label_name"], as_index=False).mean(numeric_only=True).to_csv(
        output / "representation_geometry_by_complementarity.csv", index=False
    )

    activity = pd.read_csv(args.stable_audit_dir / "activity_pd_like_long.csv",
                           dtype={"subject_id": str})
    activity.subject_id = activity.subject_id.str.zfill(3)
    activity = activity.merge(count_table[["subject_id", "persistent_category"]], on="subject_id")
    activity.groupby(["persistent_category", "label_name", "activity"], as_index=False).agg(
        subject_count=("subject_id", "nunique"), pd_like_score_mean=("pd_like_score", "mean"),
        pd_like_score_sd=("pd_like_score", "std")
    ).to_csv(output / "activity_phenotype_by_complementarity.csv", index=False)

    signals = pd.read_csv(args.stable_audit_dir / "high_risk_activity_signal_features.csv",
                          dtype={"subject_id": str})
    signals.subject_id = signals.subject_id.str.zfill(3)
    signals = signals.merge(count_table[["subject_id", "persistent_category"]], on="subject_id")
    signals.groupby(["persistent_category", "label_name", "activity", "sensor", "scope"],
                    as_index=False)[list(SIGNAL_FEATURES)].mean().to_csv(
        output / "raw_signal_profile_by_complementarity.csv", index=False
    )
    comparisons = []
    for keys, group in signals.groupby(["label_name", "activity", "sensor", "scope"]):
        rescue = group[group.persistent_category == "h1_only_correct"]
        overlap_wrong = group[group.persistent_category == "both_wrong"]
        for feature in SIGNAL_FEATURES:
            left = rescue[feature].dropna().to_numpy()
            right = overlap_wrong[feature].dropna().to_numpy()
            if len(left) >= 3 and len(right) >= 3:
                p_value = float(mannwhitneyu(left, right, alternative="two-sided").pvalue)
                comparisons.append({
                    "label_name": keys[0], "activity": keys[1], "sensor": keys[2],
                    "scope": keys[3], "feature": feature,
                    "h1_rescue_n": len(left), "both_wrong_n": len(right),
                    "h1_rescue_mean": float(left.mean()), "both_wrong_mean": float(right.mean()),
                    "cliffs_delta": cliffs_delta(left, right), "mannwhitney_p": p_value,
                })
    comparison_frame = pd.DataFrame(comparisons)
    if len(comparison_frame):
        comparison_frame["bh_q"] = bh_adjust(comparison_frame.mannwhitney_p)
    comparison_frame.to_csv(output / "raw_signal_rescue_vs_both_wrong_bh.csv", index=False)
    return frame, count_table, stable, contribution


def main():
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    payload = np.load(args.h1_analysis / "features/handcrafted_features.npz", allow_pickle=False)
    X = payload["X"].astype(np.float64)
    subjects = payload["subject_ids"].astype(str)
    y = payload["labels"].astype(int)
    schema = json.loads((args.h1_analysis / "features/feature_schema.json").read_text())
    columns = schema["columns"]
    groups, parsed = family_indices(columns)

    paired = paired_family_inference(args.family_dir, output)
    raw, split, summary = recoverability(args, X, subjects, columns, parsed, groups, output)
    frame, subject_table, stable, contribution = h1_v8_complementarity(
        args, X, y, subjects, columns, parsed, groups, output
    )

    family_summary = pd.read_csv(args.family_dir / "summary.csv")
    full = family_summary[family_summary.variant == "all"].iloc[0]
    gap_rows = []
    for family in groups:
        without = family_summary[family_summary.variant == f"without::{family}"].iloc[0]
        subset = summary[summary.family == family]
        gap_rows.append({
            "family": family,
            "lofo_ba_value": float(full.balanced_accuracy_mean - without.balanced_accuracy_mean),
            "lofo_auroc_value": float(full.auroc_mean - without.auroc_mean),
            "best_representation_r2": float(subset.r2_mean.max()),
            "best_representation": str(subset.loc[subset.r2_mean.idxmax(), "representation"]),
            "subject_embedding_r2": float(subset[subset.representation == "subject_embedding"].r2_mean.iloc[0]),
            "wrist_encoder_r2": float(subset[subset.representation == "wrist_encoder"].r2_mean.iloc[0]),
            "best_spearman": float(subset.spearman_mean.max()),
            "lowest_normalized_mae": float(subset.normalized_mae_mean.min()),
        })
    gap = pd.DataFrame(gap_rows).sort_values(
        ["lofo_auroc_value", "lofo_ba_value"], ascending=False
    )
    gap.to_csv(output / "classification_value_vs_recoverability.csv", index=False)
    protocol = {
        "scope": "fixed inner-development only",
        "outer_information_used": False,
        "backbone_trained_or_modified": False,
        "h1_hyperparameter_search": False,
        "h1_pipeline": "original fixed H1 logistic: median imputation, variance filter, train-only scaling, C=1 liblinear",
        "recoverability_probe": "fixed Ridge(alpha=1.0), train-only X and target scaling",
        "seed_handling": "V8 seed 42/43/44 probe metrics averaged within identical split before inference",
        "independent_statistical_unit": "15 fixed development splits",
        "feature_families": {key: sorted(value) for key, value in FAMILY_DEFINITIONS.items()},
        "feature_count": len(columns),
        "embedding_files": len(list(args.embedding_dir.glob("*.npz"))),
        "validation_rows": len(frame),
        "unique_subjects": int(frame.subject_id.nunique()),
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(protocol, indent=2))
    print("\nCLASSIFICATION VALUE VS RECOVERABILITY")
    print(gap.to_string(index=False))
    print("\nOVERLAP")
    print(frame.groupby(["correctness_group", "label_name"]).size())
    print("\nV8 WRONG -> H1 CORRECTION")
    print(stable.to_string(index=False))


if __name__ == "__main__":
    main()
