from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .manifest import (
    ManifestRecord,
    ManifestTimeSeriesDataset,
    WRIST_ORDER,
    load_manifest,
    selected_channel_names,
)
from .splits import (
    assert_disjoint_splits,
    stratified_subject_folds,
    subject_labels,
    train_validation_subjects,
)
from .subject_activity import SubjectActivityDataset, collate_subject_activities


class RawSignalDataset(Protocol):
    def __len__(self) -> int: ...
    def statistics_signal(self, index: int) -> torch.Tensor: ...


@dataclass
class DatasetBundle:
    train: Dataset[dict[str, object]]
    validation: Dataset[dict[str, object]]
    test: Dataset[dict[str, object]]
    class_names: list[str]
    mean: torch.Tensor
    std: torch.Tensor
    split_summary: dict[str, Any]


def _statistics(
    dataset: RawSignalDataset, epsilon: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """按腕侧和通道拟合 [2,C,1] mean/std；调用方只能传训练受试者。"""
    if len(dataset) == 0:
        raise ValueError("Cannot compute normalization from an empty train split")
    channel_sum: torch.Tensor | None = None
    channel_square_sum: torch.Tensor | None = None
    count = 0
    for index in range(len(dataset)):
        signal = dataset.statistics_signal(index).to(dtype=torch.float64)
        if signal.ndim != 3 or signal.shape[0] != 2:
            raise ValueError(
                f"Expected statistics signal [2,C,T], got {tuple(signal.shape)}"
            )
        sample_sum = signal.sum(dim=-1, keepdim=True)
        square_sum = signal.square().sum(dim=-1, keepdim=True)
        channel_sum = sample_sum if channel_sum is None else channel_sum + sample_sum
        channel_square_sum = (
            square_sum
            if channel_square_sum is None
            else channel_square_sum + square_sum
        )
        count += signal.shape[-1]
    assert channel_sum is not None and channel_square_sum is not None
    mean = channel_sum / count
    variance = (channel_square_sum / count) - mean.square()
    std = variance.clamp_min(0).sqrt().clamp_min(float(epsilon))
    return mean.to(torch.float32), std.to(torch.float32)


def _activity_statistics(
    dataset: ManifestTimeSeriesDataset,
    activities: Sequence[str],
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit train-only [activity,2,channel,1] normalization statistics."""
    activity_indices = {activity: index for index, activity in enumerate(activities)}
    sums: list[torch.Tensor | None] = [None] * len(activities)
    square_sums: list[torch.Tensor | None] = [None] * len(activities)
    counts = [0] * len(activities)
    for record_index, record in enumerate(dataset.records):
        if record.activity not in activity_indices:
            continue
        activity_index = activity_indices[record.activity]
        signal = dataset.statistics_signal(record_index).to(dtype=torch.float64)
        sample_sum = signal.sum(dim=-1, keepdim=True)
        square_sum = signal.square().sum(dim=-1, keepdim=True)
        sums[activity_index] = (
            sample_sum
            if sums[activity_index] is None
            else sums[activity_index] + sample_sum
        )
        square_sums[activity_index] = (
            square_sum
            if square_sums[activity_index] is None
            else square_sums[activity_index] + square_sum
        )
        counts[activity_index] += int(signal.shape[-1])
    means: list[torch.Tensor] = []
    stds: list[torch.Tensor] = []
    for activity_index, activity in enumerate(activities):
        if counts[activity_index] == 0:
            raise ValueError(f"No training signal available for activity={activity!r}")
        assert sums[activity_index] is not None
        assert square_sums[activity_index] is not None
        mean = sums[activity_index] / counts[activity_index]
        variance = square_sums[activity_index] / counts[activity_index] - mean.square()
        means.append(mean)
        stds.append(variance.clamp_min(0).sqrt().clamp_min(float(epsilon)))
    return torch.stack(means).to(torch.float32), torch.stack(stds).to(torch.float32)


def _select_records(
    records: Sequence[ManifestRecord], subjects: set[str]
) -> list[ManifestRecord]:
    return [record for record in records if record.subject_id in subjects]


def _split_records(
    records: Sequence[ManifestRecord],
    num_folds: int,
    test_fold: int,
    validation_fraction: float,
    seed: int,
    use_manifest_folds: bool,
    split_file: str | None = None,
) -> tuple[list[ManifestRecord], list[ManifestRecord], list[ManifestRecord], dict[str, Any]]:
    # 必须在切窗/裁剪前按 subject 划分；一个人的所有活动和双腕只属于一个 split。
    if split_file is None and not 0 <= test_fold < num_folds:
        raise ValueError(f"test_fold must be in [0, {num_folds})")
    labels_by_subject = subject_labels(
        [record.subject_id for record in records],
        [record.label for record in records],
    )
    if split_file is not None:
        path = Path(split_file).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Frozen split file not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        required = {"train_subjects", "validation_subjects", "test_subjects"}
        missing = required - set(payload)
        if missing:
            raise ValueError(f"Frozen split is missing keys: {sorted(missing)}")
        train_subjects = {str(value) for value in payload["train_subjects"]}
        validation_subjects = {
            str(value) for value in payload["validation_subjects"]
        }
        test_subjects = {str(value) for value in payload["test_subjects"]}
        assert_disjoint_splits(train_subjects, validation_subjects, test_subjects)
        available = set(labels_by_subject)
        selected = train_subjects | validation_subjects | test_subjects
        if selected != available:
            raise ValueError(
                "Frozen split and manifest subjects differ: "
                f"unknown={sorted(selected-available)}, "
                f"unassigned={sorted(available-selected)}"
            )
        source = "frozen_file"
        split_extra = {
            "split_file": str(path),
            "split_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    else:
        manifest_has_folds = use_manifest_folds and all(
            record.fold is not None for record in records
        )
        if manifest_has_folds:
            assignments: dict[str, int] = {}
            for record in records:
                assert record.fold is not None
                if not 0 <= record.fold < num_folds:
                    raise ValueError(
                        f"Manifest fold {record.fold} for {record.pair_id} is invalid"
                    )
                previous = assignments.setdefault(record.subject_id, record.fold)
                if previous != record.fold:
                    raise ValueError(
                        f"Subject {record.subject_id!r} appears in multiple folds"
                    )
        else:
            assignments = stratified_subject_folds(
                labels_by_subject, num_folds=num_folds, seed=seed
            )
        test_subjects = {
            subject for subject, fold_value in assignments.items()
            if fold_value == test_fold
        }
        candidates = set(labels_by_subject) - test_subjects
        train_subjects, validation_subjects = train_validation_subjects(
            candidates,
            labels_by_subject,
            validation_fraction=validation_fraction,
            seed=seed + 7919,
        )
        assert_disjoint_splits(train_subjects, validation_subjects, test_subjects)
        source = "manifest" if manifest_has_folds else "generated"
        split_extra = {"test_fold": test_fold}
    if not test_subjects:
        raise ValueError("Test split contains no subjects")
    train = _select_records(records, train_subjects)
    validation = _select_records(records, validation_subjects)
    test = _select_records(records, test_subjects)
    summary = {
        "train_subjects": sorted(train_subjects),
        "validation_subjects": sorted(validation_subjects),
        "test_subjects": sorted(test_subjects),
        "train_samples": len(train),
        "validation_samples": len(validation),
        "test_samples": len(test),
        "fold_source": source,
        **split_extra,
    }
    return train, validation, test, summary


def _make_dataset(
    records: list[ManifestRecord],
    data_config: Mapping[str, Any],
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
    split_name: str,
) -> ManifestTimeSeriesDataset:
    sequence_length = data_config.get("sequence_length")
    return ManifestTimeSeriesDataset(
        records=records,
        raw_input_channels=int(data_config.get("raw_input_channels", 6)),
        sensor_mode=str(data_config.get("sensor_mode", "acc_gyro")),
        wrist_mode=str(data_config.get("wrist_mode", "bilateral")),
        sequence_length=None if sequence_length is None else int(sequence_length),
        crop=str(
            data_config.get(
                "train_crop" if split_name == "train" else "eval_crop",
                data_config.get("crop", "center"),
            )
        ),
        num_crops=int(
            data_config.get("eval_crop_count", 1) if split_name != "train" else 1
        ),
        pad_value=float(data_config.get("pad_value", 0.0)),
        delimiter=str(data_config.get("delimiter", ",")),
        drop_time_column=bool(data_config.get("drop_time_column", False)),
        mean=mean,
        std=std,
    )


def load_configured_records(
    data: Mapping[str, Any], class_names: list[str]
) -> list[ManifestRecord]:
    """Load either the legacy manifest or fixed-order per-activity manifests."""
    activity_manifests = data.get("activity_manifests")
    if activity_manifests is None:
        return load_manifest(
            manifest_path=data["manifest"],
            data_root=data["root"],
            labels=class_names,
            activity=data.get("activity"),
            fold_column=str(data.get("fold", {}).get("manifest_fold_column", "fold")),
        )
    if not isinstance(activity_manifests, Mapping):
        raise TypeError("data.activity_manifests must be an activity-to-path mapping")
    activities = [str(value) for value in data.get("activities", [])]
    records: list[ManifestRecord] = []
    for activity in activities:
        if activity not in activity_manifests:
            raise ValueError(f"Missing manifest for configured activity {activity!r}")
        records.extend(
            load_manifest(
                manifest_path=activity_manifests[activity],
                data_root=data["root"],
                labels=class_names,
                activity=activity,
                fold_column=str(
                    data.get("fold", {}).get("manifest_fold_column", "fold")
                ),
            )
        )
    pair_ids = [record.pair_id for record in records]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Configured activity manifests contain duplicate pair_id values")
    return records


def build_datasets(config: Mapping[str, Any]) -> DatasetBundle:
    data = config["data"]
    if str(data["type"]).lower() != "pads":
        raise ValueError(f"Unsupported data.type: {data['type']!r}")
    class_names = [str(value) for value in data["labels"]]
    records = load_configured_records(data, class_names)
    label_counts = Counter(record.label for record in records)
    minimum = int(data.get("minimum_samples_per_class", 1))
    too_small = {
        class_names[index]: label_counts.get(index, 0)
        for index in range(len(class_names))
        if label_counts.get(index, 0) < minimum
    }
    if too_small:
        raise ValueError(
            f"Dataset class counts are below minimum_samples_per_class={minimum}: "
            f"{too_small}"
        )
    fold = data["fold"]
    train_records, validation_records, test_records, summary = _split_records(
        records,
        num_folds=int(fold["num_folds"]),
        test_fold=int(fold["test_fold"]),
        validation_fraction=float(fold.get("validation_fraction", 0.0)),
        seed=int(config["experiment"]["seed"]),
        use_manifest_folds=bool(fold.get("manifest_fold_column")),
        split_file=fold.get("split_file"),
    )
    raw_train = _make_dataset(train_records, data, None, None, "train")
    channel_names = selected_channel_names(str(data.get("sensor_mode", "acc_gyro")))
    unit = str(data.get("unit", "activity")).lower()
    normalization_scope = str(
        data.get("normalization_scope", "wrist_channel")
    ).lower()
    if normalization_scope not in {"wrist_channel", "activity_wrist_channel"}:
        raise ValueError(
            "data.normalization_scope must be wrist_channel or activity_wrist_channel"
        )
    if normalization_scope == "activity_wrist_channel" and unit != "subject":
        raise ValueError("activity_wrist_channel normalization requires subject mode")
    activities = [str(value) for value in data.get("activities", [])]
    if bool(data.get("standardize", True)):
        if normalization_scope == "activity_wrist_channel":
            mean, std = _activity_statistics(
                raw_train,
                activities,
                epsilon=float(data.get("normalization_epsilon", 1e-6)),
            )
        else:
            mean, std = _statistics(
                raw_train, epsilon=float(data.get("normalization_epsilon", 1e-6))
            )
    else:
        shape = (
            (len(activities), 2, len(channel_names), 1)
            if normalization_scope == "activity_wrist_channel"
            else (2, len(channel_names), 1)
        )
        mean = torch.zeros(shape, dtype=torch.float32)
        std = torch.ones(shape, dtype=torch.float32)
    if unit == "subject":
        train = SubjectActivityDataset(
            train_records, activities, data, mean, std, "train"
        )
        validation = SubjectActivityDataset(
            validation_records, activities, data, mean, std, "validation"
        )
        test = SubjectActivityDataset(test_records, activities, data, mean, std, "test")
        summary.update(
            {
                "train_activity_records": len(train_records),
                "validation_activity_records": len(validation_records),
                "test_activity_records": len(test_records),
                "train_samples": len(train),
                "validation_samples": len(validation),
                "test_samples": len(test),
            }
        )
    else:
        train = _make_dataset(train_records, data, mean, std, "train")
        validation = _make_dataset(validation_records, data, mean, std, "validation")
        test = _make_dataset(test_records, data, mean, std, "test")
    if unit == "subject":
        overall_labels = subject_labels(
            [record.subject_id for record in records],
            [record.label for record in records],
        )
        class_counts = Counter(overall_labels.values())
        split_counts = {
            name: Counter(
                subject_labels(
                    [record.subject_id for record in selected],
                    [record.label for record in selected],
                ).values()
            )
            for name, selected in (
                ("train", train_records),
                ("validation", validation_records),
                ("test", test_records),
            )
        }
    else:
        class_counts = label_counts
        split_counts = {
            name: Counter(record.label for record in selected)
            for name, selected in (
                ("train", train_records),
                ("validation", validation_records),
                ("test", test_records),
            )
        }
    summary.update(
        {
            "class_names": class_names,
            "class_sample_counts": {
                class_names[i]: class_counts.get(i, 0)
                for i in range(len(class_names))
            },
            "split_class_sample_counts": {
                name: {
                    class_names[i]: split_counts[name].get(i, 0)
                    for i in range(len(class_names))
                }
                for name in ("train", "validation", "test")
            },
            "selection": {
                "unit": unit,
                "activity": data.get("activity"),
                "activities": data.get("activities"),
                "wrist_mode": data.get("wrist_mode", "bilateral"),
                "sensor_mode": data.get("sensor_mode", "acc_gyro"),
            },
            "input_metadata": {
                "wrist_order": list(WRIST_ORDER),
                "channel_names": channel_names,
                "tensor_layout": (
                    "[batch, activity, wrist, channel, time]"
                    if unit == "subject"
                    else "[batch, wrist, channel, time]"
                ),
            },
            "normalization": {
                "enabled": bool(data.get("standardize", True)),
                "fitted_on": "train_subjects_only",
                "shape": list(mean.shape),
                "scope": normalization_scope,
                "per_wrist_and_channel": True,
                "per_activity": normalization_scope == "activity_wrist_channel",
            },
        }
    )
    return DatasetBundle(
        train=train,
        validation=validation,
        test=test,
        class_names=class_names,
        mean=mean,
        std=std,
        split_summary=summary,
    )


def make_loader(
    dataset: Dataset[dict[str, object]],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    sampler: Sampler[int] | None = None,
) -> DataLoader[dict[str, object]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator,
        collate_fn=(
            collate_subject_activities
            if isinstance(dataset, SubjectActivityDataset)
            else None
        ),
    )
