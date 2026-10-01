from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_f1",
    "macro_auroc",
    "pd_recall",
    "dd_recall",
    "negative_log_likelihood",
    "brier_score",
)


def _read(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _fold_map(summary: dict) -> dict[tuple[int, int], dict]:
    result = {}
    for row in summary["fold_summaries"]:
        if "outer_context" in row:
            outer = int(row["outer_context"])
            inner = int(row["inner_fold"])
        elif "outer_fold" in row:
            outer = int(row["outer_fold"])
            inner = int(row["inner_fold"])
        else:
            outer_text, inner_text = row["fold_id"].split("/")
            outer = int(outer_text.removeprefix("outer_"))
            inner = int(inner_text.removeprefix("inner_"))
        result[(outer, inner)] = row["validation_metrics"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize pure-deep seed stability on inner validation folds only.")
    parser.add_argument("--seed-run", action="append", required=True, type=Path)
    parser.add_argument("--baseline-summary", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    seed_summaries = [_read(path / "development_summary.json") for path in args.seed_run]
    baseline_summaries = [_read(path) for path in args.baseline_summary]
    if len(baseline_summaries) == 1:
        baseline_summaries = baseline_summaries * len(seed_summaries)
    if len(baseline_summaries) != len(seed_summaries):
        raise ValueError("Provide either one baseline or one matched baseline per seed")
    seed_folds = [_fold_map(summary) for summary in seed_summaries]
    baseline_folds = [_fold_map(summary) for summary in baseline_summaries]
    fold_ids = sorted(baseline_folds[0])
    if any(sorted(folds) != fold_ids for folds in seed_folds):
        raise ValueError("Seed and baseline summaries do not contain the same folds")
    if any(sorted(folds) != fold_ids for folds in baseline_folds):
        raise ValueError("Matched baseline summaries do not contain the same folds")

    per_seed = []
    for path, summary in zip(args.seed_run, seed_summaries):
        per_seed.append({
            "run": str(path.resolve()),
            "metrics": {
                metric: float(summary["metric_summary"][metric]["mean"])
                for metric in METRICS
            },
        })

    seed_level = {}
    for metric in METRICS:
        values = [row["metrics"][metric] for row in per_seed]
        seed_level[metric] = {
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values),
            "minimum": min(values),
            "maximum": max(values),
            "values": values,
        }

    paired = {}
    for metric in ("balanced_accuracy", "macro_auroc", "macro_f1", "dd_recall"):
        deltas = []
        for fold_id in fold_ids:
            seed_average = statistics.mean(folds[fold_id][metric] for folds in seed_folds)
            baseline_average = statistics.mean(
                folds[fold_id][metric] for folds in baseline_folds
            )
            deltas.append(seed_average - baseline_average)
        paired[metric] = {
            "mean_delta_vs_baseline": statistics.mean(deltas),
            "fold_wins": sum(delta > 0 for delta in deltas),
            "fold_ties": sum(delta == 0 for delta in deltas),
            "fold_count": len(deltas),
            "deltas": deltas,
        }

    result = {
        "protocol": {
            "scope": "fixed 5 outer contexts x 3 inner-validation folds",
            "seed_count": len(seed_summaries),
            "baseline_count": len(args.baseline_summary),
            "baseline_paths": [str(path.resolve()) for path in args.baseline_summary],
            "outer_test_accessed": False,
            "note": "Seed-level dispersion is primary; 45 fold-runs are not treated as independent observations.",
        },
        "per_seed": per_seed,
        "seed_level_summary": seed_level,
        "paired_fold_average_vs_baseline": paired,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
