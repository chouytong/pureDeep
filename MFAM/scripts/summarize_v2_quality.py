from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize PADS v2 quality JSONL")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--timestamp-gap-ratio", type=float, default=10.0)
    return parser.parse_args()


def summarize(data_root: Path, timestamp_gap_ratio: float = 10.0) -> dict[str, Any]:
    root = data_root.expanduser().resolve()
    records_path = root / "quality_records.jsonl"
    output = root / "quality_summary.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite quality summary: {output}")
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("Quality record file is empty")
    timestamp_gaps: list[dict[str, Any]] = []
    activity_outliers: Counter[str] = Counter()
    for record in records:
        median_dt = float(record["median_dt_seconds"])
        maximum_dt = median_dt + float(record["max_dt_deviation_seconds"])
        ratio = maximum_dt / median_dt
        if ratio > timestamp_gap_ratio:
            timestamp_gaps.append(
                {
                    "subject_id": record["subject_id"],
                    "activity": record["activity"],
                    "wrist": record["wrist"],
                    "maximum_dt_seconds": maximum_dt,
                    "median_dt_seconds": median_dt,
                    "ratio": ratio,
                }
            )
        if sum(record["processed_channels"]["robust_outlier_count"]) > 0:
            activity_outliers[str(record["activity"])] += 1
    warnings = {
        "timestamp_gap_over_threshold": len(timestamp_gaps),
        "records_with_robust_outliers": sum(activity_outliers.values()),
    }
    summary = {
        "status": "pass_with_warnings" if any(warnings.values()) else "pass",
        "record_count": len(records),
        "finite_records": sum(bool(record["finite"]) for record in records),
        "strictly_increasing_timestamp_records": sum(
            bool(record["timestamp_strictly_increasing"]) for record in records
        ),
        "effective_sampling_rate_range": [
            min(float(record["effective_sampling_rate"]) for record in records),
            max(float(record["effective_sampling_rate"]) for record in records),
        ],
        "maximum_constant_run": max(
            int(record["processed_channels"]["maximum_constant_run"])
            for record in records
        ),
        "timestamp_gap_ratio_threshold": timestamp_gap_ratio,
        "warnings": warnings,
        "timestamp_gap_records": sorted(
            timestamp_gaps, key=lambda value: value["ratio"], reverse=True
        ),
        "records_with_robust_outliers_by_activity": dict(activity_outliers),
        "interpretation": (
            "Warnings are audit flags only. No signal was removed, interpolated, "
            "clipped, or resampled by this summary."
        ),
    }
    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            summarize(args.data_root, args.timestamp_gap_ratio),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
