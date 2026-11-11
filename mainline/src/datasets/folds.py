from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from src.utils.provenance import normalization_metadata

from .builders import (
    _activity_statistics,
    _make_dataset,
    _select_records,
    load_configured_records,
)
from .manifest import ManifestRecord
from .nested_cv import subject_metadata_from_records
from .subject_activity import SubjectActivityDataset


@dataclass
class SubjectFoldBundle:
    train: Dataset[dict[str, object]]
    validation: Dataset[dict[str, object]] | None
    test: Dataset[dict[str, object]] | None
    class_names: list[str]
    mean: torch.Tensor
    std: torch.Tensor
    split_summary: dict[str, Any]
    normalization: dict[str, Any]


def _subject_class_counts(
    records: Sequence[ManifestRecord],
    class_names: Sequence[str],
) -> dict[str, int]:
    labels: dict[str, int] = {}
    for record in records:
        previous = labels.setdefault(record.subject_id, record.label)
        if previous != record.label:
            raise ValueError(f"Inconsistent labels for subject={record.subject_id!r}")
    counts = Counter(labels.values())
    return {
        str(class_name): counts.get(index, 0)
        for index, class_name in enumerate(class_names)
    }


def build_subject_fold_datasets(
    config: Mapping[str, Any],
    *,
    train_subject_ids: Sequence[str],
    validation_subject_ids: Sequence[str] = (),
    test_subject_ids: Sequence[str] = (),
    fold_id: str,
) -> SubjectFoldBundle:
    data = config["data"]
    if str(data.get("type", "")).lower() != "pads":
        raise ValueError("Nested fold construction supports only data.type=pads")
    if str(data.get("unit", "")).lower() != "subject":
        raise ValueError("Nested folds require data.unit=subject")
    if str(data.get("normalization_scope", "")).lower() != (
        "activity_wrist_channel"
    ):
        raise ValueError(
            "V3 requires data.normalization_scope=activity_wrist_channel"
        )
    class_names = [str(value) for value in data["labels"]]
    if class_names != ["PD", "DD"]:
        raise ValueError("V3 label order must be exactly ['PD', 'DD']")
    activities = [str(value) for value in data["activities"]]
    records = load_configured_records(data, class_names)
    subject_strata, _ = subject_metadata_from_records(records, activities)
    available = set(subject_strata)
    train_subjects = set(str(value) for value in train_subject_ids)
    validation_subjects = set(str(value) for value in validation_subject_ids)
    test_subjects = set(str(value) for value in test_subject_ids)
    if not train_subjects:
        raise ValueError("Fold train subjects cannot be empty")
    intersections = {
        "train_validation": train_subjects & validation_subjects,
        "train_test": train_subjects & test_subjects,
        "validation_test": validation_subjects & test_subjects,
    }
    overlap = {name: sorted(value) for name, value in intersections.items() if value}
    if overlap:
        raise ValueError(f"Fold subject overlap detected: {overlap}")
    selected = train_subjects | validation_subjects | test_subjects
    unknown = selected - available
    unassigned = available - selected
    if unknown or unassigned:
        raise ValueError(
            "Fold subjects do not exactly cover configured manifests: "
            f"unknown={sorted(unknown)}, unassigned={sorted(unassigned)}"
        )

    train_records = _select_records(records, train_subjects)
    validation_records = _select_records(records, validation_subjects)
    test_records = _select_records(records, test_subjects)
    raw_train = _make_dataset(train_records, data, None, None, "train")
    if bool(data.get("standardize", True)):
        mean, std = _activity_statistics(
            raw_train,
            activities,
            epsilon=float(data.get("normalization_epsilon", 1e-6)),
        )
    else:
        channel_count = 6 if data.get("sensor_mode", "acc_gyro") == "acc_gyro" else 3
        shape = (len(activities), 2, channel_count, 1)
        mean = torch.zeros(shape, dtype=torch.float32)
        std = torch.ones(shape, dtype=torch.float32)

    train = SubjectActivityDataset(
        train_records, activities, data, mean, std, "train"
    )
    validation = (
        SubjectActivityDataset(
            validation_records, activities, data, mean, std, "validation"
        )
        if validation_records
        else None
    )
    test = (
        SubjectActivityDataset(test_records, activities, data, mean, std, "test")
        if test_records
        else None
    )
    summary = {
        "fold_id": str(fold_id),
        "train_subjects": sorted(train_subjects),
        "validation_subjects": sorted(validation_subjects),
        "test_subjects": sorted(test_subjects),
        "train_subject_count": len(train_subjects),
        "validation_subject_count": len(validation_subjects),
        "test_subject_count": len(test_subjects),
        "train_activity_records": len(train_records),
        "validation_activity_records": len(validation_records),
        "test_activity_records": len(test_records),
        "class_counts": {
            "train": _subject_class_counts(train_records, class_names),
            "validation": _subject_class_counts(validation_records, class_names),
            "test": _subject_class_counts(test_records, class_names),
        },
        "normalization_fitted_on": "train_subjects_only",
    }
    norm_metadata = normalization_metadata(
        mean,
        std,
        subject_ids=sorted(train_subjects),
        fold_id=str(fold_id),
    )
    return SubjectFoldBundle(
        train=train,
        validation=validation,
        test=test,
        class_names=class_names,
        mean=mean,
        std=std,
        split_summary=summary,
        normalization=norm_metadata,
    )
