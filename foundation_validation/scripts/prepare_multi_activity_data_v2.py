from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.datasets.prepare_pads import MANIFEST_FIELDS, _build_pair, condition_to_label
from src.datasets.preprocessing import l1_trend_filter_batch


DEFAULT_ACTIVITIES = (
    "CrossArms",
    "DrinkGlas",
    "Entrainment",
    "HoldWeight",
    "LiftHold",
    "PointFinger",
    "Relaxed",
    "RelaxedTask",
    "StretchHold",
    "TouchIndex",
    "TouchNose",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build non-destructive PADS v2 data with official-style detrending"
    )
    parser.add_argument(
        "--source-root", type=Path, default=Path("data/raw/pads")
    )
    parser.add_argument(
        "--processed-output",
        type=Path,
        default=Path("data/processed/pads_multi_activity/v2_l1_full_length"),
    )
    parser.add_argument("--activity", action="append", dest="activities")
    parser.add_argument("--trim-start-samples", type=int, default=48)
    parser.add_argument("--l1-lambda", type=float, default=50.0)
    parser.add_argument("--solver-rho", type=float, default=40.0)
    parser.add_argument("--solver-max-iterations", type=int, default=5000)
    parser.add_argument("--solver-absolute-tolerance", type=float, default=1e-4)
    parser.add_argument("--solver-relative-tolerance", type=float, default=1e-4)
    parser.add_argument("--solver-batch-files", type=int, default=32)
    parser.add_argument("--sampling-rate-tolerance", type=float, default=0.2)
    parser.add_argument("--robust-outlier-threshold", type=float, default=20.0)
    parser.add_argument("--constant-run-minimum", type=int, default=50)
    parser.add_argument("--max-subjects", type=int, default=0)
    return parser.parse_args()


def _require_empty(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {path}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(
            stream,
            np.ascontiguousarray(array, dtype=np.float32),
            allow_pickle=False,
        )


def _maximum_constant_run(values: np.ndarray, tolerance: float = 1e-12) -> int:
    maximum = 1
    for channel in values:
        equal = np.abs(np.diff(channel)) <= tolerance
        current = 1
        for repeated in equal:
            current = current + 1 if repeated else 1
            maximum = max(maximum, current)
    return maximum


def _channel_quality(values: np.ndarray, outlier_threshold: float) -> dict[str, Any]:
    median = np.median(values, axis=1)
    mad = np.median(np.abs(values - median[:, None]), axis=1)
    scale = 1.4826 * mad
    deviations = np.abs(values - median[:, None])
    outlier_mask = np.zeros_like(values, dtype=bool)
    valid_scale = scale > np.finfo(np.float64).eps
    outlier_mask[valid_scale] = (
        deviations[valid_scale] / scale[valid_scale, None]
    ) > float(outlier_threshold)
    return {
        "min": values.min(axis=1).tolist(),
        "max": values.max(axis=1).tolist(),
        "mean": values.mean(axis=1).tolist(),
        "std": values.std(axis=1).tolist(),
        "median": median.tolist(),
        "mad": mad.tolist(),
        "robust_outlier_count": outlier_mask.sum(axis=1).astype(int).tolist(),
        "zero_difference_fraction": np.mean(
            np.abs(np.diff(values, axis=1)) <= 1e-12, axis=1
        ).tolist(),
        "maximum_constant_run": _maximum_constant_run(values),
    }


def _load_raw_signal(
    path: Path,
    nominal_rate: float,
    sampling_rate_tolerance: float,
    outlier_threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    array = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 7:
        raise ValueError(f"Expected [time,6 channels], got {array.shape}")
    finite = bool(np.isfinite(array).all())
    if not finite:
        raise ValueError("Signal contains NaN or infinite values")
    time = array[:, 0]
    differences = np.diff(time)
    strictly_increasing = bool(differences.size and np.all(differences > 0))
    if not strictly_increasing:
        raise ValueError("Timestamp is not strictly increasing")
    effective_rate = float(1.0 / np.median(differences))
    relative_rate_error = abs(effective_rate - nominal_rate) / nominal_rate
    if relative_rate_error > sampling_rate_tolerance:
        raise ValueError(
            f"Effective rate {effective_rate:.6g} differs from nominal {nominal_rate:.6g}"
        )
    signal = np.ascontiguousarray(array[:, 1:].T)
    quality = {
        "raw_length": int(signal.shape[-1]),
        "finite": finite,
        "timestamp_strictly_increasing": strictly_increasing,
        "duplicate_timestamp_count": int(np.sum(differences == 0)),
        "median_dt_seconds": float(np.median(differences)),
        "max_dt_deviation_seconds": float(
            np.max(np.abs(differences - np.median(differences)))
        ),
        "maximum_dt_ratio": float(
            np.max(differences) / np.median(differences)
        ),
        "effective_sampling_rate": effective_rate,
        "nominal_sampling_rate": float(nominal_rate),
        "relative_sampling_rate_error": float(relative_rate_error),
        "raw_channels": _channel_quality(signal, outlier_threshold),
    }
    return time, signal, quality


def _discover_records(
    source_root: Path, activities: list[str], max_subjects: int
) -> tuple[dict[str, list[dict[str, Any]]], Counter[str]]:
    patients: dict[str, tuple[str, str]] = {}
    condition_counts: Counter[str] = Counter()
    for path in sorted((source_root / "patients").glob("patient_*.json")):
        patient = json.loads(path.read_text(encoding="utf-8"))
        subject_id = str(patient["id"])
        condition = str(patient["condition"])
        condition_counts[condition] += 1
        label = condition_to_label(condition)
        if label is not None:
            patients[subject_id] = (label, condition)
    selected_subjects = set(sorted(patients)[:max_subjects]) if max_subjects else set(patients)
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation_path in sorted((source_root / "movement").glob("observation_*.json")):
        observation = json.loads(observation_path.read_text(encoding="utf-8"))
        subject_id = str(observation["subject_id"])
        if subject_id not in selected_subjects:
            continue
        label, condition = patients[subject_id]
        sessions = observation.get("session", [])
        for activity in activities:
            matches = [
                (index, session)
                for index, session in enumerate(sessions)
                if str(session.get("record_name", "")) == activity
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one {activity} session for subject {subject_id}, found {len(matches)}"
                )
            session_index, session = matches[0]
            row = _build_pair(
                source_root,
                subject_id,
                activity,
                session_index,
                session,
                observation,
                label,
                condition,
            )
            records[activity].append(
                {
                    "row": row,
                    "nominal_rate": float(observation["sampling_rate"]),
                }
            )
    return records, condition_counts


def prepare_v2(
    source_root: Path,
    processed_output: Path,
    activities: list[str],
    *,
    trim_start_samples: int = 48,
    l1_lambda: float = 50.0,
    solver_rho: float = 40.0,
    solver_max_iterations: int = 5000,
    solver_absolute_tolerance: float = 1e-4,
    solver_relative_tolerance: float = 1e-4,
    solver_batch_files: int = 32,
    sampling_rate_tolerance: float = 0.2,
    robust_outlier_threshold: float = 20.0,
    constant_run_minimum: int = 50,
    max_subjects: int = 0,
) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    processed_output = processed_output.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"PADS raw source does not exist: {source_root}")
    if not activities or len(activities) != len(set(activities)):
        raise ValueError("activities must contain unique non-empty values")
    if trim_start_samples < 0 or solver_batch_files < 1:
        raise ValueError("trim_start_samples must be non-negative and batch size positive")
    _require_empty(processed_output)

    discovered, condition_counts = _discover_records(
        source_root, activities, max_subjects
    )
    quality_records: list[dict[str, Any]] = []
    rows_by_activity: dict[str, list[dict[str, str]]] = defaultdict(list)
    solver_reports: list[dict[str, Any]] = []
    processed_lengths: dict[str, Counter[int]] = defaultdict(Counter)
    warning_counts: Counter[str] = Counter()

    for activity in activities:
        entries = sorted(
            discovered[activity], key=lambda item: item["row"]["subject_id"]
        )
        wrist_items: list[dict[str, Any]] = []
        for entry in entries:
            row = dict(entry["row"])
            for side in ("left", "right"):
                source_path = source_root / row[f"{side}_path"]
                time, signal, quality = _load_raw_signal(
                    source_path,
                    entry["nominal_rate"],
                    sampling_rate_tolerance,
                    robust_outlier_threshold,
                )
                if signal.shape[-1] <= trim_start_samples:
                    raise ValueError(f"Signal is too short after trimming: {source_path}")
                wrist_items.append(
                    {
                        "row": row,
                        "side": side,
                        "source_path": source_path,
                        "time": time,
                        "signal": signal,
                        "quality": quality,
                    }
                )

        by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for item in wrist_items:
            by_length[int(item["signal"].shape[-1])].append(item)
        for length, same_length in sorted(by_length.items()):
            for start in range(0, len(same_length), solver_batch_files):
                chunk = same_length[start : start + solver_batch_files]
                acceleration = np.concatenate(
                    [item["signal"][:3] for item in chunk], axis=0
                )
                trend, solver_report = l1_trend_filter_batch(
                    acceleration,
                    regularization=l1_lambda,
                    rho=solver_rho,
                    max_iterations=solver_max_iterations,
                    absolute_tolerance=solver_absolute_tolerance,
                    relative_tolerance=solver_relative_tolerance,
                )
                solver_reports.append(
                    {
                        "activity": activity,
                        "raw_length": length,
                        "file_count": len(chunk),
                        **solver_report.__dict__,
                    }
                )
                if not solver_report.converged:
                    raise RuntimeError(
                        f"L1 trend solver did not converge for {activity}, length={length}: "
                        f"{solver_report}"
                    )
                for index, item in enumerate(chunk):
                    processed = item["signal"].copy()
                    processed[:3] -= trend[index * 3 : (index + 1) * 3]
                    processed = np.ascontiguousarray(processed[:, trim_start_samples:])
                    processed_time = item["time"][trim_start_samples:]
                    destination = (
                        processed_output
                        / "signals"
                        / activity
                        / item["row"]["subject_id"]
                        / f"{item['side']}.npy"
                    )
                    _save_array(destination, processed)
                    item["row"][f"{item['side']}_path"] = str(
                        destination.relative_to(processed_output)
                    )
                    item["row"][f"{item['side']}_length"] = str(processed.shape[-1])
                    processed_lengths[activity][int(processed.shape[-1])] += 1
                    processed_quality = _channel_quality(
                        processed, robust_outlier_threshold
                    )
                    if item["quality"]["maximum_dt_ratio"] > 10.0:
                        warning_counts["timestamp_gap_over_10x_median_dt"] += 1
                    if processed_quality["maximum_constant_run"] >= constant_run_minimum:
                        warning_counts["long_constant_run"] += 1
                    if sum(processed_quality["robust_outlier_count"]) > 0:
                        warning_counts["robust_outlier_detected"] += 1
                    quality_records.append(
                        {
                            "subject_id": item["row"]["subject_id"],
                            "activity": activity,
                            "wrist": item["side"],
                            "source_path": str(item["source_path"]),
                            "processed_path": str(destination),
                            **item["quality"],
                            "trimmed_start_samples": trim_start_samples,
                            "processed_length": int(processed.shape[-1]),
                            "processed_duration_seconds": float(
                                processed_time[-1] - processed_time[0]
                            ),
                            "processed_channels": processed_quality,
                            "source_sha256": _sha256(item["source_path"]),
                            "processed_sha256": _sha256(destination),
                        }
                    )

        unique_rows: dict[str, dict[str, str]] = {}
        for item in wrist_items:
            row = item["row"]
            unique_rows[row["pair_id"]] = row
        rows_by_activity[activity] = sorted(
            unique_rows.values(), key=lambda row: row["subject_id"]
        )

    manifests = processed_output / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    manifest_hashes: dict[str, str] = {}
    for activity in activities:
        manifest = manifests / f"{activity}.csv"
        with manifest.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows_by_activity[activity])
        manifest_hashes[activity] = _sha256(manifest)

    quality_path = processed_output / "quality_records.jsonl"
    with quality_path.open("x", encoding="utf-8") as stream:
        for record in quality_records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    first_activity = activities[0]
    label_counts = Counter(row["label"] for row in rows_by_activity[first_activity])
    processing = {
        "profile": "pads_official_style_l1_full_length_v2",
        "output_dtype": "float32",
        "output_layout": "[channel,time]",
        "channel_order": ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"],
        "time_column_removed_from_tensor": True,
        "accelerometer": {
            "operation": "raw - L1 trend",
            "objective": "0.5*||y-x||_2^2 + lambda*||D2*x||_1",
            "lambda": l1_lambda,
            "solver": "batched ADMM",
            "rho": solver_rho,
            "max_iterations": solver_max_iterations,
            "absolute_tolerance": solver_absolute_tolerance,
            "relative_tolerance": solver_relative_tolerance,
        },
        "gyroscope": "unchanged",
        "trim_start_samples": trim_start_samples,
        "additional_filtering": "none",
        "augmentation": "none",
    }
    report = {
        "status": "pass_with_warnings" if warning_counts else "pass",
        "source_root": str(source_root),
        "processed_output": str(processed_output),
        "activities": activities,
        "subject_count": len(rows_by_activity[first_activity]),
        "activity_pair_counts": {
            activity: len(rows_by_activity[activity]) for activity in activities
        },
        "label_counts": dict(label_counts),
        "source_condition_counts": dict(condition_counts),
        "processed_signal_files": len(quality_records),
        "processed_lengths_per_wrist": {
            activity: {str(k): v for k, v in sorted(counts.items())}
            for activity, counts in processed_lengths.items()
        },
        "processing": processing,
        "quality_audit": {
            "record_file": str(quality_path),
            "record_count": len(quality_records),
            "sampling_rate_tolerance": sampling_rate_tolerance,
            "robust_outlier_threshold_mad": robust_outlier_threshold,
            "constant_run_minimum": constant_run_minimum,
            "warning_counts": dict(warning_counts),
            "nan_or_inf_records": 0,
            "non_monotonic_timestamp_records": 0,
        },
        "solver_batches": len(solver_reports),
        "solver_max_iterations_observed": max(
            report["iterations"] for report in solver_reports
        ),
        "manifest_sha256": manifest_hashes,
    }
    processed_output.mkdir(parents=True, exist_ok=True)
    (processed_output / "processing_config.json").write_text(
        json.dumps(processing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (processed_output / "solver_audit.json").write_text(
        json.dumps(solver_reports, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (processed_output / "audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    report = prepare_v2(
        args.source_root,
        args.processed_output,
        list(args.activities or DEFAULT_ACTIVITIES),
        trim_start_samples=args.trim_start_samples,
        l1_lambda=args.l1_lambda,
        solver_rho=args.solver_rho,
        solver_max_iterations=args.solver_max_iterations,
        solver_absolute_tolerance=args.solver_absolute_tolerance,
        solver_relative_tolerance=args.solver_relative_tolerance,
        solver_batch_files=args.solver_batch_files,
        sampling_rate_tolerance=args.sampling_rate_tolerance,
        robust_outlier_threshold=args.robust_outlier_threshold,
        constant_run_minimum=args.constant_run_minimum,
        max_subjects=args.max_subjects,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
