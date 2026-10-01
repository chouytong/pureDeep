from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


DD_CONDITIONS = {
    "Other Movement Disorders",
    "Essential Tremor",
    "Multiple Sclerosis",
    "Atypical Parkinsonism",
}
EXPECTED_METADATA_CHANNELS = [
    "Time",
    "Accelerometer_X",
    "Accelerometer_Y",
    "Accelerometer_Z",
    "Gyroscope_X",
    "Gyroscope_Y",
    "Gyroscope_Z",
]
MANIFEST_FIELDS = [
    "left_path",
    "right_path",
    "subject_id",
    "label",
    "activity",
    "pair_id",
    "session_index",
    "record_name",
    "fold",
    "left_length",
    "right_length",
    "left_sampling_rate",
    "right_sampling_rate",
    "max_time_offset_seconds",
    "source_condition",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a paired bilateral PADS manifest without modifying data"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--activity", default="CrossArms")
    parser.add_argument(
        "--invalid-pair-policy",
        choices=["error", "exclude"],
        default="error",
        help="error stops on invalid pairs; exclude records every rejected pair",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=None,
        help="Defaults to OUTPUT.audit.json and is never overwritten",
    )
    return parser.parse_args()


def condition_to_label(condition: str) -> str | None:
    """PADS 默认标签映射；Healthy 不属于 PD-vs-DD。"""
    if condition == "Parkinson's":
        return "PD"
    if condition in DD_CONDITIONS:
        return "DD"
    if condition == "Healthy":
        return None
    raise ValueError(f"Unknown PADS condition: {condition!r}")


def _load_patient_labels(
    root: Path,
) -> tuple[dict[str, str], dict[str, str], dict[str, int]]:
    labels: dict[str, str] = {}
    conditions: dict[str, str] = {}
    counts: dict[str, int] = {}
    files = sorted((root / "patients").glob("patient_*.json"))
    if not files:
        raise FileNotFoundError(f"No patient metadata found under {root / 'patients'}")
    for path in files:
        patient = json.loads(path.read_text(encoding="utf-8"))
        subject_id = str(patient["id"])
        condition = str(patient["condition"])
        counts[condition] = counts.get(condition, 0) + 1
        label = condition_to_label(condition)
        if label is not None:
            labels[subject_id] = label
            conditions[subject_id] = condition
    return labels, conditions, counts


def _signal_quality(path: Path) -> dict[str, Any]:
    array = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 7:
        raise ValueError(f"expected [time,7], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("contains NaN or infinite values")
    time_axis = array[:, 0]
    differences = np.diff(time_axis)
    if differences.size == 0 or not (differences > 0).all():
        raise ValueError("time column is not strictly increasing")
    return {
        "length": int(array.shape[0]),
        "time": time_axis,
        "median_dt": float(np.median(differences)),
        "effective_sampling_rate": float(1.0 / np.median(differences)),
    }


def _build_pair(
    root: Path,
    subject_id: str,
    activity: str,
    session_index: int,
    session: dict[str, Any],
    observation: dict[str, Any],
    label: str,
    condition: str,
) -> dict[str, str]:
    records_by_location: dict[str, dict[str, Any]] = {}
    for record in session.get("records", []):
        location = str(record.get("device_location", ""))
        if location in records_by_location:
            raise ValueError(f"duplicate device_location={location!r}")
        records_by_location[location] = record
    required = {"LeftWrist", "RightWrist"}
    if set(records_by_location) != required:
        raise ValueError(
            f"expected exactly {sorted(required)}, got {sorted(records_by_location)}"
        )
    left_record = records_by_location["LeftWrist"]
    right_record = records_by_location["RightWrist"]
    for side, record in (("left", left_record), ("right", right_record)):
        if list(record.get("channels", [])) != EXPECTED_METADATA_CHANNELS:
            raise ValueError(f"{side} channel definition is unexpected")
    left_path = (root / "movement" / str(left_record["file_name"])).resolve()
    right_path = (root / "movement" / str(right_record["file_name"])).resolve()
    if not left_path.is_file() or not right_path.is_file():
        raise FileNotFoundError(
            f"missing signal: left={left_path.is_file()}, right={right_path.is_file()}"
        )
    left = _signal_quality(left_path)
    right = _signal_quality(right_path)
    if left["length"] != right["length"]:
        raise ValueError(
            f"length mismatch: left={left['length']}, right={right['length']}"
        )
    nominal_rate = float(observation.get("sampling_rate", 0.0))
    if nominal_rate <= 0:
        raise ValueError(f"invalid observation sampling_rate={nominal_rate}")
    # 独立手表时间轴存在小漂移；这里只验证有效采样率接近元数据标称值并记录偏差。
    for side, quality in (("left", left), ("right", right)):
        if not 0.8 * nominal_rate <= quality["effective_sampling_rate"] <= 1.2 * nominal_rate:
            raise ValueError(
                f"{side} effective sampling rate {quality['effective_sampling_rate']:.4f} "
                f"does not match nominal {nominal_rate:.4f}"
            )
    max_offset = float(np.max(np.abs(left["time"] - right["time"])))
    pair_id = f"{subject_id}_{activity}_session{session_index:02d}"
    return {
        "left_path": str(left_path.relative_to(root)),
        "right_path": str(right_path.relative_to(root)),
        "subject_id": subject_id,
        "label": label,
        "activity": activity,
        "pair_id": pair_id,
        "session_index": str(session_index),
        "record_name": str(session.get("record_name", "")),
        "fold": "",
        "left_length": str(left["length"]),
        "right_length": str(right["length"]),
        "left_sampling_rate": f"{nominal_rate:.8g}",
        "right_sampling_rate": f"{nominal_rate:.8g}",
        "max_time_offset_seconds": f"{max_offset:.10g}",
        "source_condition": condition,
    }


def generate_manifest(
    root: Path,
    output: Path,
    activity: str,
    invalid_pair_policy: str,
    audit_output: Path,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    output = output.expanduser().resolve()
    audit_output = audit_output.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"PADS root does not exist: {root}")
    for path in (output, audit_output):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    labels, conditions, condition_counts = _load_patient_labels(root)
    rows: list[dict[str, str]] = []
    exclusions: list[dict[str, str]] = []
    for observation_path in sorted((root / "movement").glob("observation_*.json")):
        observation = json.loads(observation_path.read_text(encoding="utf-8"))
        subject_id = str(observation["subject_id"])
        if subject_id not in labels:
            continue
        matching = [
            (index, session)
            for index, session in enumerate(observation.get("session", []))
            if str(session.get("record_name", "")) == activity
        ]
        if not matching:
            exclusions.append(
                {"subject_id": subject_id, "activity": activity, "reason": "missing_session"}
            )
            continue
        for session_index, session in matching:
            try:
                rows.append(
                    _build_pair(
                        root,
                        subject_id,
                        activity,
                        session_index,
                        session,
                        observation,
                        labels[subject_id],
                        conditions[subject_id],
                    )
                )
            except Exception as error:
                exclusions.append(
                    {
                        "subject_id": subject_id,
                        "activity": activity,
                        "session_index": str(session_index),
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )
    report = {
        "status": "pass" if not exclusions else "fail",
        "data_root": str(root),
        "manifest": str(output),
        "activity": activity,
        "invalid_pair_policy": invalid_pair_policy,
        "pairs_written": len(rows),
        "pairs_excluded": len(exclusions),
        "label_counts": {
            label: sum(row["label"] == label for row in rows) for label in ("PD", "DD")
        },
        "source_condition_counts": condition_counts,
        "channel_order_per_wrist": [
            "AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"
        ],
        "wrist_order": ["left", "right"],
        "pairing_rule": "same observation JSON and same session object",
        "time_alignment": "independent clocks; offset recorded; late fusion required",
        "exclusions": exclusions,
    }
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if exclusions and invalid_pair_policy == "error":
        raise ValueError(
            f"Found {len(exclusions)} invalid pairs; see audit report {audit_output}"
        )
    if not rows:
        raise ValueError(f"No valid PADS pairs found for activity={activity!r}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return report


def main() -> None:
    args = parse_args()
    audit = args.audit_output or args.output.with_suffix(".audit.json")
    report = generate_manifest(
        args.root, args.output, args.activity, args.invalid_pair_policy, audit
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
