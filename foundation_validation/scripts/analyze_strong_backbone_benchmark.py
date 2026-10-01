#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon


METRICS = ("balanced_accuracy", "macro_auroc", "macro_f1", "dd_recall")


def bh_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 1.0
    for reverse_index in range(count - 1, -1, -1):
        original_index = int(order[reverse_index])
        rank = reverse_index + 1
        running = min(running, p_values[original_index] * count / rank)
        adjusted[original_index] = min(1.0, running)
    return adjusted.tolist()


def paired_rank_biserial(values: np.ndarray) -> float:
    nonzero = values[values != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    denominator = positive + negative
    return (positive - negative) / denominator if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", action="append", nargs=2,
                        metavar=("NAME", "ANALYSIS_JSON"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for comparison, raw_path in args.comparison:
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for metric in METRICS:
            values = np.asarray(
                payload["paired_fold_average_vs_baseline"][metric]["deltas"],
                dtype=float,
            )
            test = wilcoxon(values, alternative="two-sided", zero_method="wilcox",
                            method="auto")
            standard_deviation = float(values.std(ddof=1))
            rows.append({
                "comparison": comparison,
                "metric": metric,
                "independent_split_count": int(values.size),
                "mean_delta": float(values.mean()),
                "median_delta": float(np.median(values)),
                "sample_std_delta": standard_deviation,
                "cohen_dz": float(values.mean() / standard_deviation)
                if standard_deviation > 0 else math.nan,
                "paired_rank_biserial": paired_rank_biserial(values),
                "wins": int((values > 0).sum()),
                "ties": int((values == 0).sum()),
                "wilcoxon_statistic": float(test.statistic),
                "wilcoxon_p_two_sided": float(test.pvalue),
            })

    adjusted = bh_adjust([row["wilcoxon_p_two_sided"] for row in rows])
    for row, q_value in zip(rows, adjusted):
        row["bh_q_across_12_confirmatory_tests"] = q_value

    result = {
        "analysis_unit": "independent development split after averaging seeds 42/43/44",
        "test": "two-sided paired Wilcoxon signed-rank",
        "effect_sizes": ["paired rank-biserial correlation", "Cohen dz"],
        "multiplicity": "Benjamini-Hochberg across 3 comparisons x 4 metrics",
        "outer_information_used": False,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
