#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


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


def rank_biserial(values):
    values = np.asarray(values, dtype=float)
    values = values[np.abs(values) > 1.0e-12]
    if not len(values):
        return 0.0
    ranks = rankdata(np.abs(values), method="average")
    positive = ranks[values > 0].sum()
    negative = ranks[values < 0].sum()
    return float((positive - negative) / (positive + negative))


def signed_test(values):
    values = np.asarray(values, dtype=float)
    nonzero = values[np.abs(values) > 1.0e-12]
    return float(wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.analysis_dir / "recoverability_15_splits.csv")

    one_sample = []
    for (representation, family), group in data.groupby(["representation", "family"]):
        values = group.r2_variance_weighted.to_numpy()
        one_sample.append({
            "representation": representation, "family": family,
            "mean_r2": float(values.mean()), "splits_r2_positive": int((values > 0).sum()),
            "rank_biserial_vs_zero": rank_biserial(values),
            "wilcoxon_p_vs_zero": signed_test(values),
        })
    one_sample = pd.DataFrame(one_sample)
    one_sample["bh_q_24_tests"] = bh_adjust(one_sample.wilcoxon_p_vs_zero)
    one_sample.to_csv(args.analysis_dir / "recoverability_r2_inference.csv", index=False)

    paired = []
    for family in sorted(data.family.unique()):
        indexed = data[data.family == family].set_index(["outer", "inner", "representation"])
        for source in ("wrist_encoder", "bilateral_activity", "activity_context"):
            for metric, favorable in (("r2_variance_weighted", 1),
                                      ("median_feature_spearman", 1),
                                      ("normalized_mae", -1)):
                source_values = indexed.xs(source, level="representation")[metric]
                subject_values = indexed.xs("subject_embedding", level="representation")[metric]
                delta = (source_values - subject_values).to_numpy() * favorable
                paired.append({
                    "family": family, "source_representation": source, "metric": metric,
                    "favorable_delta_vs_subject": float(delta.mean()),
                    "wins": int((delta > 0).sum()),
                    "paired_rank_biserial": rank_biserial(delta),
                    "wilcoxon_p": signed_test(delta),
                })
    paired = pd.DataFrame(paired)
    paired["bh_q_54_tests"] = bh_adjust(paired.wilcoxon_p)
    paired.to_csv(args.analysis_dir / "recoverability_representation_loss_inference.csv", index=False)

    records = pd.read_csv(args.analysis_dir / "h1_v8_validation_records.csv",
                          dtype={"subject_id": str})
    summary = []
    for (stability, label), group in records[~records.v8_correct].groupby(
        ["stability_group", "label_name"]
    ):
        summary.append({
            "stability_group": stability, "label_name": label,
            "v8_wrong_rows": len(group), "h1_correct_rows": int(group.h1_correct.sum()),
            "h1_correction_rate": float(group.h1_correct.mean()),
            "unique_subjects": int(group.subject_id.nunique()),
        })
    pd.DataFrame(summary).to_csv(
        args.analysis_dir / "h1_correction_of_v8_errors_by_stability_exact.csv", index=False
    )

    subjects = pd.read_csv(args.analysis_dir / "subject_error_complementarity.csv",
                           dtype={"subject_id": str})
    stable = subjects[subjects.stability_group == "stable_error"]
    stable.groupby(["label_name", "persistent_category"], as_index=False).agg(
        subjects=("subject_id", "nunique")
    ).to_csv(args.analysis_dir / "stable_error_subject_outcomes.csv", index=False)


if __name__ == "__main__":
    main()
