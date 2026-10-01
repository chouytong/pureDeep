#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


def bh(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    running = 1.0
    for position in range(len(values) - 1, -1, -1):
        index = order[position]
        running = min(running, values[index] * len(values) / (position + 1))
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_p(values):
    values = np.asarray(values)
    values = values[np.abs(values) > 1e-12]
    return 1.0 if not len(values) else float(wilcoxon(values).pvalue)


def rank_biserial(values):
    values = np.asarray(values)
    values = values[np.abs(values) > 1e-12]
    if not len(values):
        return 0.0
    ranks = rankdata(np.abs(values), method="average")
    return float((ranks[values > 0].sum() - ranks[values < 0].sum()) / ranks.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.analysis_dir.resolve()
    frame = pd.read_csv(root / "recoverability_15_splits.csv")
    indexed = frame.set_index(["outer", "inner", "representation", "family"])
    keys = sorted(frame[["outer", "inner"]].drop_duplicates().itertuples(index=False, name=None))
    families = ("time_location_scale", "band_fraction")
    representations = (
        "str_subject_embedding", "str_structured_residual", "str_decision_input"
    )
    metric_specs = (
        ("r2_variance_weighted", 1.0),
        ("median_feature_spearman", 1.0),
        ("normalized_mae", -1.0),
    )
    rows = []
    for representation in representations:
        for family in families:
            for metric, direction in metric_specs:
                delta = np.asarray([
                    direction * (
                        indexed.loc[(outer, inner, representation, family), metric]
                        - indexed.loc[(outer, inner, "v8_subject_embedding", family), metric]
                    ) for outer, inner in keys
                ])
                rows.append({
                    "representation": representation, "family": family,
                    "metric": metric,
                    "positive_means_improved": True,
                    "mean_oriented_delta": delta.mean(),
                    "wins": int((delta > 0).sum()), "losses": int((delta < 0).sum()),
                    "wilcoxon_p": paired_p(delta),
                    "rank_biserial": rank_biserial(delta),
                })
    result = pd.DataFrame(rows)
    result["bh_q_across_18_key_tests"] = bh(result.wilcoxon_p)
    result.to_csv(root / "recoverability_paired_inference.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
