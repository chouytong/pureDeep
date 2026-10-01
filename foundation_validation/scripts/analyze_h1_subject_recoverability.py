#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


FAMILIES = {
    "time_location_scale": {"mean", "std", "variance", "rms", "median", "mad", "iqr",
                            "minimum", "maximum", "peak_to_peak", "signal_energy"},
    "time_shape": {"skewness", "kurtosis_excess", "zero_crossing_rate_centered"},
    "derivative": {"derivative_rms", "derivative_std"},
    "spectral_global": {"dominant_frequency_hz_0p5_20", "dominant_power_0p5_20",
                        "spectral_entropy_0p5_20", "total_power_0p5_20"},
    "band_absolute": {"band_power_0p5_3", "band_power_3_7", "band_power_7_12",
                      "band_power_12_20"},
    "band_fraction": {"band_fraction_0p5_3", "band_fraction_3_7", "band_fraction_7_12",
                      "band_fraction_12_20"},
}
REPRESENTATIONS = {
    "wrist_mean": "subject_wrist_encoder",
    "bilateral_activity_mean": "subject_bilateral_activity",
    "activity_context_mean": "subject_activity_context",
    "learned_subject_embedding": "subject_embedding",
}


def bh(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    result = np.empty(len(values))
    running = 1.0
    for position in range(len(values) - 1, -1, -1):
        index = int(order[position])
        running = min(running, values[index] * len(values) / (position + 1))
        result[index] = min(1.0, running)
    return result


def rb(values):
    values = np.asarray(values)
    values = values[np.abs(values) > 1e-12]
    ranks = rankdata(np.abs(values))
    return float((ranks[values > 0].sum() - ranks[values < 0].sum()) / ranks.sum())


def metrics(y, prediction):
    correlations = []
    for index in range(y.shape[1]):
        if np.std(y[:, index]) > 1e-10 and np.std(prediction[:, index]) > 1e-10:
            value = spearmanr(y[:, index], prediction[:, index]).statistic
            if np.isfinite(value):
                correlations.append(value)
    return {
        "r2": float(r2_score(y, prediction, multioutput="variance_weighted")),
        "median_spearman": float(np.median(correlations)),
        "normalized_mae": float(np.mean(np.abs(y - prediction))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h1-analysis", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = np.load(args.h1_analysis / "features/handcrafted_features.npz", allow_pickle=False)
    X = payload["X"].astype(float)
    ids = payload["subject_ids"].astype(str)
    index = {value: position for position, value in enumerate(ids)}
    schema = json.loads((args.h1_analysis / "features/feature_schema.json").read_text())
    features = [column.split("|", 3)[3] for column in schema["columns"]]
    family_indices = {
        family: np.asarray([i for i, feature in enumerate(features) if feature in names])
        for family, names in FAMILIES.items()
    }
    pattern = re.compile(r"seed(\d+)_outer(\d+)_inner(\d+)\.npz")
    rows = []
    for path in sorted(args.embedding_dir.glob("*.npz")):
        seed, outer, inner = map(int, pattern.fullmatch(path.name).groups())
        data = np.load(path, allow_pickle=False)
        train_index = np.asarray([index[value] for value in data["train_subject_id"].astype(str)])
        validation_index = np.asarray([index[value] for value in data["validation_subject_id"].astype(str)])
        for family, selected in family_indices.items():
            target_scaler = StandardScaler().fit(X[np.ix_(train_index, selected)])
            train_target = target_scaler.transform(X[np.ix_(train_index, selected)])
            validation_target = target_scaler.transform(X[np.ix_(validation_index, selected)])
            components = min(16, train_target.shape[0] - 1, train_target.shape[1])
            pca = PCA(n_components=components, svd_solver="randomized", random_state=0).fit(train_target)
            train_pca = pca.transform(train_target)
            validation_pca = pca.transform(validation_target)
            pca_scale = train_pca.std(axis=0)
            pca_scale[pca_scale < 1e-8] = 1.0
            train_pca = train_pca / pca_scale
            validation_pca = validation_pca / pca_scale
            for representation, key in REPRESENTATIONS.items():
                train_x = data[f"train_{key}"]
                validation_x = data[f"validation_{key}"]
                scaler = StandardScaler().fit(train_x)
                model = Ridge(alpha=1.0).fit(scaler.transform(train_x), train_pca)
                result = metrics(validation_pca, model.predict(scaler.transform(validation_x)))
                rows.append({
                    "seed": seed, "outer": outer, "inner": inner,
                    "family": family, "representation": representation,
                    "pca_components": components,
                    "train_pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
                    **result,
                })
    raw = pd.DataFrame(rows)
    raw.to_csv(args.output_dir / "subject_family_pca_recoverability_45_seed_folds.csv", index=False)
    numeric = ["pca_components", "train_pca_variance_explained", "r2", "median_spearman",
               "normalized_mae"]
    split = raw.groupby(["outer", "inner", "family", "representation"], as_index=False)[numeric].mean()
    split.to_csv(args.output_dir / "subject_family_pca_recoverability_15_splits.csv", index=False)
    summary = split.groupby(["family", "representation"], as_index=False).agg(
        r2_mean=("r2", "mean"), r2_sd=("r2", "std"),
        spearman_mean=("median_spearman", "mean"), nmae_mean=("normalized_mae", "mean"),
        pca_variance_mean=("train_pca_variance_explained", "mean"),
    )
    summary.to_csv(args.output_dir / "subject_family_pca_recoverability_summary.csv", index=False)

    paired = []
    for family in FAMILIES:
        table = split[split.family == family].pivot(index=["outer", "inner"],
                                                    columns="representation", values="r2")
        learned = table.learned_subject_embedding
        for source in ("wrist_mean", "bilateral_activity_mean", "activity_context_mean"):
            delta = (table[source] - learned).to_numpy()
            paired.append({
                "family": family, "comparison": f"{source}-learned_subject_embedding",
                "mean_r2_delta": float(delta.mean()), "wins": int((delta > 0).sum()),
                "rank_biserial": rb(delta), "wilcoxon_p": float(wilcoxon(delta).pvalue),
            })
    paired = pd.DataFrame(paired)
    paired["bh_q_18_tests"] = bh(paired.wilcoxon_p)
    paired.to_csv(args.output_dir / "subject_family_pca_representation_comparisons.csv", index=False)
    print(summary.to_string(index=False))
    print("\nPAIRED")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
