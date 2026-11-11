from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from .manifest import ManifestRecord, ManifestTimeSeriesDataset


def sample_activity_keep_mask(count: int, dropout_probability: float) -> torch.Tensor:
    """Sample train-only activity dropout while retaining at least one activity."""

    if count < 1:
        raise ValueError("Activity dropout requires at least one available activity")
    if not 0.0 <= dropout_probability < 1.0:
        raise ValueError("train_activity_dropout must be in [0, 1)")
    scores = torch.rand(count)
    keep = scores >= dropout_probability
    if not bool(keep.any()):
        keep[scores.argmax()] = True
    return keep


def sample_wrist_rotation_matrices(
    wrist_count: int, maximum_degrees: float, *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Sample bounded 3-D rotations for train-only sensor orientation augmentation."""

    if wrist_count < 1:
        raise ValueError("wrist_count must be positive")
    if not 0.0 <= maximum_degrees <= 180.0:
        raise ValueError("train_rotation_degrees must be in [0, 180]")
    if maximum_degrees == 0:
        return torch.eye(3, dtype=dtype).expand(wrist_count, -1, -1).clone()
    axes = torch.randn(wrist_count, 3, dtype=dtype)
    axes = axes / axes.norm(dim=1, keepdim=True).clamp_min(1.0e-8)
    angles = (torch.rand(wrist_count, dtype=dtype) * 2.0 - 1.0) * (
        maximum_degrees * math.pi / 180.0
    )
    x, y, z = axes.unbind(dim=1)
    zeros = torch.zeros_like(x)
    skew = torch.stack(
        [zeros, -z, y, z, zeros, -x, -y, x, zeros], dim=1
    ).reshape(wrist_count, 3, 3)
    identity = torch.eye(3, dtype=dtype).expand(wrist_count, -1, -1)
    sin = torch.sin(angles).reshape(-1, 1, 1)
    cos = torch.cos(angles).reshape(-1, 1, 1)
    return identity + sin * skew + (1.0 - cos) * torch.bmm(skew, skew)


def rotate_acc_gyro(signal: torch.Tensor, matrices: torch.Tensor) -> torch.Tensor:
    """Apply one wrist-specific rotation consistently to Acc and Gyro vectors."""

    if signal.ndim != 3 or signal.shape[1] != 6:
        raise ValueError("Rotation augmentation expects [wrist,6,time]")
    if matrices.shape != (signal.shape[0], 3, 3):
        raise ValueError("Rotation matrices must have shape [wrist,3,3]")
    rotated = signal.clone()
    rotated[:, :3] = torch.einsum("wij,wjt->wit", matrices, signal[:, :3])
    rotated[:, 3:] = torch.einsum("wij,wjt->wit", matrices, signal[:, 3:])
    return rotated


class SubjectActivityDataset(Dataset[dict[str, object]]):
    """Group fixed-order activity records into one subject-level sample."""

    def __init__(
        self,
        records: Sequence[ManifestRecord],
        activities: Sequence[str],
        data_config: Mapping[str, Any],
        mean: torch.Tensor,
        std: torch.Tensor,
        split_name: str,
    ) -> None:
        self.records = list(records)
        self.activities = tuple(str(value) for value in activities)
        if not self.activities or len(self.activities) != len(set(self.activities)):
            raise ValueError("activities must contain unique non-empty values")
        selected = set(self.activities)
        grouped: dict[str, dict[str, int]] = defaultdict(dict)
        labels: dict[str, int] = {}
        for record_index, record in enumerate(self.records):
            if record.activity not in selected:
                continue
            previous = labels.setdefault(record.subject_id, record.label)
            if previous != record.label:
                raise ValueError(
                    f"Subject {record.subject_id!r} has inconsistent activity labels"
                )
            if record.activity in grouped[record.subject_id]:
                raise ValueError(
                    "Duplicate subject/activity pair: "
                    f"subject={record.subject_id!r}, activity={record.activity!r}"
                )
            grouped[record.subject_id][record.activity] = record_index
        if not grouped:
            raise ValueError("No subject records remain after activity selection")
        self.subject_ids = sorted(grouped)
        self.grouped_indices = grouped
        self.labels = labels
        crop_key = "train_crop" if split_name == "train" else "eval_crop"
        crop = str(data_config.get(crop_key, data_config.get("crop", "center")))
        num_crops = int(
            data_config.get("eval_crop_count", 1) if split_name != "train" else 1
        )
        sequence_length = data_config.get("sequence_length")
        self.base = ManifestTimeSeriesDataset(
            records=self.records,
            raw_input_channels=int(data_config.get("raw_input_channels", 6)),
            sensor_mode=str(data_config.get("sensor_mode", "acc_gyro")),
            wrist_mode=str(data_config.get("wrist_mode", "bilateral")),
            sequence_length=(
                None if sequence_length is None else int(sequence_length)
            ),
            crop=crop,
            pad_value=float(data_config.get("pad_value", 0.0)),
            delimiter=str(data_config.get("delimiter", ",")),
            drop_time_column=bool(data_config.get("drop_time_column", False)),
            # Subject-level normalization is applied per activity below so that
            # v2 can use [activity,wrist,channel,1] train-only statistics.
            mean=None,
            std=None,
            num_crops=num_crops,
        )
        self.wrist_mode = self.base.wrist_mode
        self.sensor_mode = self.base.sensor_mode
        self.activity_dropout = (
            float(data_config.get("train_activity_dropout", 0.0))
            if split_name == "train"
            else 0.0
        )
        if not 0.0 <= self.activity_dropout < 1.0:
            raise ValueError("train_activity_dropout must be in [0, 1)")
        self.rotation_degrees = (
            float(data_config.get("train_rotation_degrees", 0.0))
            if split_name == "train"
            else 0.0
        )
        if not 0.0 <= self.rotation_degrees <= 180.0:
            raise ValueError("train_rotation_degrees must be in [0, 180]")
        self.mean = mean
        self.std = std
        if self.mean.shape != self.std.shape:
            raise ValueError("normalization mean/std shapes must match")
        if self.mean.ndim == 4 and self.mean.shape[0] != len(self.activities):
            raise ValueError("per-activity normalization must have shape [A,2,C,1]")
        if self.mean.ndim not in {3, 4}:
            raise ValueError("normalization must be [2,C,1] or [A,2,C,1]")

    def _normalize(
        self, signal: torch.Tensor, wrist_mask: torch.Tensor, activity_index: int
    ) -> torch.Tensor:
        mean = self.mean if self.mean.ndim == 3 else self.mean[activity_index]
        std = self.std if self.std.ndim == 3 else self.std[activity_index]
        if signal.ndim == 4:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
            mask_shape = (1, 2, 1, 1)
        else:
            mask_shape = (2, 1, 1)
        normalized = (signal - mean) / std
        return normalized * wrist_mask.reshape(mask_shape)

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        subject_id = self.subject_ids[index]
        indices = self.grouped_indices[subject_id]
        loaded = {
            activity: self.base[indices[activity]]
            for activity in self.activities
            if activity in indices
        }
        if not loaded:
            raise RuntimeError(f"Subject {subject_id!r} has no available activity")
        loaded_names = list(loaded)
        keep_mask = sample_activity_keep_mask(len(loaded_names), self.activity_dropout)
        kept = {
            name for name, keep in zip(loaded_names, keep_mask.tolist()) if keep
        }
        rotation_matrices = (
            sample_wrist_rotation_matrices(2, self.rotation_degrees)
            if self.rotation_degrees > 0
            else None
        )
        template = next(iter(loaded.values()))["x"]
        if not isinstance(template, torch.Tensor):
            raise TypeError("Underlying dataset returned a non-tensor signal")
        signals: list[torch.Tensor] = []
        wrist_masks: list[torch.Tensor] = []
        activity_mask: list[bool] = []
        activity_lengths: list[int] = []
        pair_ids: dict[str, str | None] = {}
        paths: dict[str, dict[str, str] | None] = {}
        for activity_index, activity in enumerate(self.activities):
            item = loaded.get(activity) if activity in kept else None
            if item is None:
                signals.append(torch.zeros_like(template))
                wrist_masks.append(torch.zeros(2, dtype=torch.float32))
                activity_mask.append(False)
                activity_lengths.append(0)
                pair_ids[activity] = None
                paths[activity] = None
                continue
            signal = item["x"]
            wrist_mask = item["wrist_mask"]
            assert isinstance(signal, torch.Tensor)
            assert isinstance(wrist_mask, torch.Tensor)
            if rotation_matrices is not None:
                signal = rotate_acc_gyro(
                    signal, rotation_matrices.to(dtype=signal.dtype, device=signal.device)
                )
            signals.append(self._normalize(signal, wrist_mask, activity_index))
            wrist_masks.append(wrist_mask)
            activity_mask.append(True)
            activity_lengths.append(int(signal.shape[-1]))
            pair_ids[activity] = str(item["pair_id"])
            paths[activity] = {
                "left": str(item["left_path"]),
                "right": str(item["right_path"]),
            }
        # In full-length mode activities can have different T. Padding is only a
        # batch container; SubjectMFAM slices each activity back to its real length.
        if template.ndim == 4:
            signal_stack = torch.stack(signals, dim=1)
        else:
            maximum_length = max(activity_lengths)
            padded = [
                functional.pad(signal, (0, maximum_length - signal.shape[-1]))
                for signal in signals
            ]
            signal_stack = torch.stack(padded, dim=0)
        return {
            "x": signal_stack,
            "wrist_mask": torch.stack(wrist_masks, dim=0),
            "activity_mask": torch.tensor(activity_mask, dtype=torch.bool),
            "activity_lengths": torch.tensor(activity_lengths, dtype=torch.long),
            "y": torch.tensor(self.labels[subject_id], dtype=torch.long),
            "subject_id": subject_id,
            "sample_id": subject_id,
            "pair_id": subject_id,
            "activity": json.dumps(self.activities, ensure_ascii=False),
            "activities": json.dumps(self.activities, ensure_ascii=False),
            "activity_pair_ids": json.dumps(pair_ids, ensure_ascii=False),
            "left_path": json.dumps(paths, ensure_ascii=False),
            "right_path": json.dumps(paths, ensure_ascii=False),
            "wrist_mode": self.wrist_mode,
            "sensor_mode": self.sensor_mode,
        }


def collate_subject_activities(
    batch: list[dict[str, object]],
) -> dict[str, object]:
    """Right-pad subject tensors to the longest activity in the mini-batch."""
    if not batch:
        raise ValueError("Cannot collate an empty subject batch")
    tensors = [item["x"] for item in batch]
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("Subject batch contains a non-tensor signal")
    maximum_length = max(int(value.shape[-1]) for value in tensors)
    padded_batch: list[dict[str, object]] = []
    for item, value in zip(batch, tensors):
        assert isinstance(value, torch.Tensor)
        copied = dict(item)
        copied["x"] = functional.pad(value, (0, maximum_length - value.shape[-1]))
        padded_batch.append(copied)
    return default_collate(padded_batch)
