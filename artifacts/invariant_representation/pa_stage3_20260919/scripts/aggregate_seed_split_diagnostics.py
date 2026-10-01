#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


RELATIONSHIP_COLUMNS = (
    "disease_probe_balanced_accuracy",
    "disease_probe_auroc",
    "disease_centroid_ba",
    "disease_silhouette_euclidean",
    "disease_silhouette_cosine",
    "disease_fisher_ratio",
    "subject_retrieval_top1",
    "subject_similarity_gap",
    "activity_probe_accuracy",
    "domain_probe_auc_disease_controlled_mean",
    "normalized_mean_shift",
    "coral_covariance_shift",
    "pd_centroid_drift",
    "dd_centroid_drift",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.fold_csv)
    expected = {(outer, inner) for outer in range(5) for inner in range(3)}
    observed = set(zip(frame.outer.astype(int), frame.inner.astype(int)))
    if observed != expected or len(frame) != 45 or frame.seed.nunique() != 3:
        raise ValueError("Expected 3 seeds on the same 5x3 inner splits")
    counts = frame.groupby(["outer", "inner"]).seed.nunique()
    if not (counts == 3).all():
        raise ValueError("Every independent split must contain all three seeds")
    numeric = [
        column for column in frame.columns
        if column not in {"seed", "outer", "inner"}
    ]
    aggregated = frame.groupby(["outer", "inner"], as_index=False)[numeric].mean()
    relationships = []
    for column in RELATIONSHIP_COLUMNS:
        rho, p_value = spearmanr(
            aggregated[column], aggregated["head_balanced_accuracy"]
        )
        relationships.append({
            "statistical_unit": "seed-aggregated independent inner split",
            "n": 15,
            "variable": column,
            "outcome": "head_balanced_accuracy",
            "spearman_rho": float(rho),
            "p_value": float(p_value),
        })
    relationship_frame = pd.DataFrame(relationships)
    summary = {
        "protocol": {
            "raw_seed_fold_rows": 45,
            "primary_statistical_units": 15,
            "aggregation": "arithmetic mean across seeds 42/43/44 per outer-inner split",
            "outer_test_accessed": False,
        },
        "split_level_metrics": {
            column: {
                "mean": float(aggregated[column].mean()),
                "std_across_15_splits": float(aggregated[column].std(ddof=1)),
                "minimum": float(aggregated[column].min()),
                "maximum": float(aggregated[column].max()),
            }
            for column in numeric
        },
        "relationships_with_ba": relationships,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(args.output_dir / "seed_aggregated_15_splits.csv", index=False)
    relationship_frame.to_csv(
        args.output_dir / "seed_aggregated_relationships.csv", index=False
    )
    (args.output_dir / "seed_aggregated_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(relationship_frame.to_string(index=False))


if __name__ == "__main__":
    main()
