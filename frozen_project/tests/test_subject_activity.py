from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from src.datasets.builders import _activity_statistics
from src.datasets.manifest import ManifestRecord, ManifestTimeSeriesDataset
from src.datasets.subject_activity import (
    SubjectActivityDataset,
    collate_subject_activities,
)
from src.engine.checkpoint import input_metadata_from_config
from src.engine.runner import _forward_with_views
from src.models import build_model
from src.utils.config import load_config


def _record(
    tmp_path: Path,
    subject: str,
    activity: str,
    label: int,
    length: int = 64,
    value: float | None = None,
) -> ManifestRecord:
    base_value = float(label if value is None else value)
    for side, offset in (("left", 0.0), ("right", 1.0)):
        np.save(
            tmp_path / f"{subject}_{activity}_{side}.npy",
            np.full((6, length), base_value + offset, dtype=np.float32),
        )
    return ManifestRecord(
        left_path=tmp_path / f"{subject}_{activity}_left.npy",
        right_path=tmp_path / f"{subject}_{activity}_right.npy",
        subject_id=subject,
        label=label,
        activity=activity,
        pair_id=f"{subject}_{activity}",
        fold=None,
    )


def _data_config() -> dict:
    return {
        "raw_input_channels": 6,
        "sensor_mode": "acc_gyro",
        "wrist_mode": "bilateral",
        "sequence_length": 32,
        "train_crop": "center",
        "eval_crop": "multi",
        "eval_crop_count": 3,
        "pad_value": 0.0,
        "delimiter": ",",
        "drop_time_column": False,
    }


def _model_config(activities: list[str]) -> dict:
    config = deepcopy(load_config("configs/pads_multi_activity_v2.yaml"))
    config["data"]["activities"] = activities
    config["model"]["name"] = "subject_mfam"
    config["model"]["encoder"].update(
        {"branch_channels": 8, "hidden_dim": 16, "channel_reduction": 4}
    )
    config["model"]["mil"].update(
        {"window_seconds": 0.2, "attention_dim": 8}
    )
    config["model"]["activity_fusion"] = {
        "attention_dim": 8,
        "dropout": 0.0,
        "classifier_dropout": 0.0,
    }
    return config


def test_subject_dataset_fixed_activity_order_and_missing_mask(tmp_path: Path) -> None:
    activities = ["CrossArms", "Relaxed", "TouchNose"]
    records = [
        _record(tmp_path, "p1", "CrossArms", 0),
        _record(tmp_path, "p1", "TouchNose", 0),
    ]
    dataset = SubjectActivityDataset(
        records,
        activities,
        _data_config(),
        mean=torch.zeros(2, 6, 1),
        std=torch.ones(2, 6, 1),
        split_name="train",
    )
    item = dataset[0]
    assert item["x"].shape == (3, 2, 6, 32)
    assert item["wrist_mask"].shape == (3, 2)
    assert item["activity_mask"].tolist() == [True, False, True]
    assert item["activity_lengths"].tolist() == [32, 0, 32]
    assert torch.count_nonzero(item["x"][1]) == 0


def test_subject_dataset_multicrop_layout(tmp_path: Path) -> None:
    activities = ["CrossArms", "Relaxed"]
    records = [_record(tmp_path, "p1", activity, 0) for activity in activities]
    dataset = SubjectActivityDataset(
        records,
        activities,
        _data_config(),
        mean=torch.zeros(2, 6, 1),
        std=torch.ones(2, 6, 1),
        split_name="validation",
    )
    assert dataset[0]["x"].shape == (3, 2, 2, 6, 32)


def test_duplicate_subject_activity_is_rejected(tmp_path: Path) -> None:
    record = _record(tmp_path, "p1", "CrossArms", 0)
    with pytest.raises(ValueError, match="Duplicate subject/activity"):
        SubjectActivityDataset(
            [record, record],
            ["CrossArms"],
            _data_config(),
            torch.zeros(2, 6, 1),
            torch.ones(2, 6, 1),
            "train",
        )


def test_subject_mfam_mask_attention_and_backward() -> None:
    activities = ["CrossArms", "Relaxed", "TouchNose"]
    model = build_model(_model_config(activities))
    inputs = torch.randn(2, 3, 2, 6, 128, requires_grad=True)
    wrist_mask = torch.ones(2, 3, 2)
    activity_mask = torch.tensor([[True, False, True], [True, True, True]])
    outputs = model(inputs, wrist_mask, activity_mask)
    assert outputs["logits"].shape == (2, 2)
    assert outputs["activity_embeddings"].shape == (2, 3, 66)
    torch.testing.assert_close(
        outputs["activity_attention"].sum(dim=1), torch.ones(2)
    )
    assert outputs["activity_attention"][0, 1] == 0
    outputs["logits"].sum().backward()
    assert inputs.grad is not None
    assert torch.count_nonzero(inputs.grad[0, 1]) == 0


def test_variable_length_activities_ignore_padded_tail(tmp_path: Path) -> None:
    activities = ["CrossArms", "Relaxed"]
    records = [
        _record(tmp_path, "p1", "CrossArms", 0, length=96),
        _record(tmp_path, "p1", "Relaxed", 0, length=192),
    ]
    config = _data_config()
    config.update(
        {"sequence_length": None, "train_crop": "full", "eval_crop": "full"}
    )
    mean = torch.zeros(2, 2, 6, 1)
    std = torch.ones(2, 2, 6, 1)
    dataset = SubjectActivityDataset(records, activities, config, mean, std, "train")
    item = dataset[0]
    assert item["x"].shape == (2, 2, 6, 192)
    assert item["activity_lengths"].tolist() == [96, 192]
    batch = collate_subject_activities([item])
    model = build_model(_model_config(activities)).eval()
    original = batch["x"].clone()
    changed = original.clone()
    changed[:, 0, :, :, 96:] = 1000.0
    with torch.no_grad():
        first = model(
            original,
            batch["wrist_mask"],
            batch["activity_mask"],
            batch["activity_lengths"],
        )["logits"]
        second = model(
            changed,
            batch["wrist_mask"],
            batch["activity_mask"],
            batch["activity_lengths"],
        )["logits"]
    torch.testing.assert_close(first, second)


def test_subject_dataset_applies_per_activity_normalization(tmp_path: Path) -> None:
    activities = ["CrossArms", "Relaxed"]
    records = [
        _record(tmp_path, "p1", "CrossArms", 1, length=64, value=1.0),
        _record(tmp_path, "p1", "Relaxed", 1, length=64, value=5.0),
    ]
    # _record writes left=label and right=label+1 for every channel.
    mean = torch.tensor([[[[1.0]], [[2.0]]], [[[5.0]], [[6.0]]]]).expand(
        2, 2, 6, 1
    )
    std = torch.ones_like(mean)
    dataset = SubjectActivityDataset(
        records, activities, _data_config(), mean, std, "train"
    )
    assert torch.count_nonzero(dataset[0]["x"]) == 0


def test_activity_statistics_are_separate(tmp_path: Path) -> None:
    activities = ["CrossArms", "Relaxed"]
    records = [
        _record(tmp_path, "p1", "CrossArms", 0, value=1.0),
        _record(tmp_path, "p1", "Relaxed", 0, value=5.0),
    ]
    raw = ManifestTimeSeriesDataset(
        records,
        raw_input_channels=6,
        sensor_mode="acc_gyro",
        wrist_mode="bilateral",
        sequence_length=None,
        crop="full",
        pad_value=0.0,
        delimiter=",",
        drop_time_column=False,
    )
    mean, std = _activity_statistics(raw, activities, epsilon=1e-6)
    assert mean.shape == (2, 2, 6, 1)
    torch.testing.assert_close(mean[0, 0], torch.ones(6, 1))
    torch.testing.assert_close(mean[1, 0], torch.full((6, 1), 5.0))
    torch.testing.assert_close(std, torch.full_like(std, 1e-6))


def test_subject_multicrop_probabilities() -> None:
    activities = ["CrossArms", "Relaxed"]
    model = build_model(_model_config(activities)).eval()
    inputs = torch.randn(2, 3, 2, 2, 6, 128)
    wrist_mask = torch.ones(2, 2, 2)
    activity_mask = torch.ones(2, 2, dtype=torch.bool)
    outputs = _forward_with_views(model, inputs, wrist_mask, activity_mask)
    assert outputs["probabilities"].shape == (2, 2)
    assert outputs["activity_attention"].shape == (2, 2)


def test_subject_checkpoint_metadata_matches_v2() -> None:
    config = load_config("configs/pads_multi_activity_v2.yaml")
    assert input_metadata_from_config(config) == {
        "tensor_layout": "[batch, activity, wrist, channel, time]",
        "wrist_order": ["left", "right"],
        "wrist_mode": "bilateral",
        "sensor_mode": "acc_gyro",
        "channel_names": ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"],
        "unit": "subject",
        "activities": config["data"]["activities"],
        "activity_count": 11,
        "activity_fusion": "attention",
        "time_length": "variable_full_signal",
        "normalization_scope": "activity_wrist_channel",
    }
