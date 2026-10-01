#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import torch

from src.datasets.builders import build_datasets, load_configured_records
from src.datasets.manifest import ManifestTimeSeriesDataset
from src.datasets.prepare_pads import condition_to_label
from src.datasets.splits import assert_disjoint_splits
from src.utils.config import load_config, require_pads_classification_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only PADS subject-v2 audit")
    parser.add_argument("--config", default="configs/pads_multi_activity_v2.yaml")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE"
    )
    parser.add_argument("--max-records", type=int, default=0)
    return parser.parse_args()


def audit_pads_classification(
    config: dict[str, Any], max_records: int = 0
) -> dict[str, Any]:
    require_pads_classification_config(config)
    data = config["data"]
    classes = [str(value) for value in data["labels"]]
    records = load_configured_records(data, classes)
    issues: list[str] = []
    selected = records if max_records <= 0 else records[:max_records]
    raw_dataset = ManifestTimeSeriesDataset(
        selected,
        raw_input_channels=int(data.get("raw_input_channels", 6)),
        sensor_mode=str(data.get("sensor_mode", "acc_gyro")),
        wrist_mode=str(data.get("wrist_mode", "bilateral")),
        sequence_length=None,
        crop="full",
        pad_value=float(data.get("pad_value", 0.0)),
        delimiter=str(data.get("delimiter", ",")),
        drop_time_column=bool(data.get("drop_time_column", False)),
    )
    for record in records:
        if classes == ["PD", "DD"]:
            expected = condition_to_label(record.source_condition)
            actual = classes[record.label]
            if expected != actual:
                issues.append(f"{record.pair_id}: label mapping mismatch")
    for index, record in enumerate(selected):
        try:
            signal = raw_dataset.full_signal(index)
            if signal.ndim != 3 or signal.shape[0] != 2:
                issues.append(f"{record.pair_id}: invalid shape {tuple(signal.shape)}")
            if not torch.isfinite(signal).all():
                issues.append(f"{record.pair_id}: non-finite signal")
        except Exception as error:
            issues.append(f"{record.pair_id}:{type(error).__name__}:{error}")
    bundle = build_datasets(config)
    split = bundle.split_summary
    assert_disjoint_splits(
        split["train_subjects"], split["validation_subjects"], split["test_subjects"]
    )
    expected_shape = (len(data["activities"]), 2, len(raw_dataset.channel_names), 1)
    if tuple(bundle.mean.shape) != expected_shape:
        issues.append(
            f"normalization shape {tuple(bundle.mean.shape)} != {expected_shape}"
        )
    if not torch.isfinite(bundle.mean).all() or not torch.isfinite(bundle.std).all():
        issues.append("normalization contains non-finite values")
    return {
        "status": "pass" if not issues else "fail",
        "read_only": True,
        "config": config["_meta"]["config_path"],
        "activity_manifests": dict(data["activity_manifests"]),
        "activities": list(data["activities"]),
        "wrist_mode": data.get("wrist_mode"),
        "sensor_mode": data.get("sensor_mode"),
        "pairs": len(records),
        "subjects": len({record.subject_id for record in records}),
        "class_counts": {
            classes[index]: count
            for index, count in sorted(Counter(record.label for record in records).items())
        },
        "pairs_checked": len(selected),
        "check_complete": len(selected) == len(records),
        "split": split,
        "normalization_shape": list(bundle.mean.shape),
        "normalization_fitted_on": "train_subjects_only",
        "issues": issues,
    }


def main() -> int:
    args = parse_args()
    report = audit_pads_classification(
        load_config(args.config, args.overrides), args.max_records
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
