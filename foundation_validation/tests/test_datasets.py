import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.datasets.builders import build_datasets
from src.datasets.manifest import ManifestTimeSeriesDataset, load_manifest
from src.datasets.prepare_pads import (
    EXPECTED_METADATA_CHANNELS,
    condition_to_label,
    generate_manifest,
)
from src.datasets.splits import assert_disjoint_splits, subject_labels


FIELDS = [
    "left_path", "right_path", "subject_id", "label", "activity", "pair_id",
    "fold", "source_condition",
]


def _write_pair(tmp_path: Path, subject: str, label: str, left: float, right: float) -> dict[str, str]:
    for side, value in (("left", left), ("right", right)):
        np.save(tmp_path / f"{subject}_{side}.npy", np.full((40, 6), value, np.float32))
    return {
        "left_path": f"{subject}_left.npy",
        "right_path": f"{subject}_right.npy",
        "subject_id": subject,
        "label": label,
        "activity": "CrossArms",
        "pair_id": f"{subject}_pair",
        "fold": "",
        "source_condition": "Parkinson's" if label == "PD" else "Essential Tremor",
    }


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _config(tmp_path: Path, manifest: Path, split: Path, wrist_mode: str = "bilateral") -> dict:
    return {
        "experiment": {"seed": 7},
        "data": {
            "type": "pads", "root": str(tmp_path), "manifest": str(manifest),
            "activity": "CrossArms", "labels": ["PD", "DD"],
            "raw_input_channels": 6, "sensor_mode": "acc_gyro",
            "wrist_mode": wrist_mode, "sequence_length": 32, "crop": "center",
            "pad_value": 0.0, "delimiter": ",", "drop_time_column": False,
            "minimum_samples_per_class": 1, "standardize": True,
            "normalization_epsilon": 1e-6,
            "fold": {"num_folds": 3, "test_fold": 0, "validation_fraction": 0.0,
                     "manifest_fold_column": None, "split_file": str(split)},
        },
    }


def test_bilateral_manifest_shape_mask_split_and_train_only_normalization(tmp_path: Path) -> None:
    rows = [
        _write_pair(tmp_path, "train_pd", "PD", 1.0, 10.0),
        _write_pair(tmp_path, "train_dd", "DD", 3.0, 14.0),
        _write_pair(tmp_path, "val_pd", "PD", 100.0, 200.0),
        _write_pair(tmp_path, "test_dd", "DD", -100.0, -200.0),
    ]
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, rows)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train_subjects": ["train_pd", "train_dd"],
                                 "validation_subjects": ["val_pd"],
                                 "test_subjects": ["test_dd"]}), encoding="utf-8")
    bundle = build_datasets(_config(tmp_path, manifest, split))
    assert_disjoint_splits(bundle.split_summary["train_subjects"],
                           bundle.split_summary["validation_subjects"],
                           bundle.split_summary["test_subjects"])
    assert bundle.train[0]["x"].shape == (2, 6, 32)
    assert bundle.train[0]["wrist_mask"].tolist() == [1.0, 1.0]
    torch.testing.assert_close(bundle.mean[0], torch.full((6, 1), 2.0))
    torch.testing.assert_close(bundle.mean[1], torch.full((6, 1), 12.0))
    torch.testing.assert_close(bundle.std[0], torch.ones((6, 1)))
    torch.testing.assert_close(bundle.std[1], torch.full((6, 1), 2.0))


@pytest.mark.parametrize("mode,expected", [("left", [1.0, 0.0]),
                                              ("right", [0.0, 1.0]),
                                              ("bilateral", [1.0, 1.0])])
def test_wrist_modes_keep_pair_shape_and_mask(tmp_path: Path, mode: str, expected: list[float]) -> None:
    row = _write_pair(tmp_path, "p1", "PD", 1.0, 2.0)
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [row])
    records = load_manifest(manifest, tmp_path, ["PD", "DD"], "CrossArms")
    dataset = ManifestTimeSeriesDataset(records, 6, "acc_gyro", mode, 32, "center",
                                        0.0, ",", False,
                                        mean=torch.zeros(2, 6, 1), std=torch.ones(2, 6, 1))
    item = dataset[0]
    assert item["x"].shape == (2, 6, 32)
    assert item["wrist_mask"].tolist() == expected
    if mode == "left":
        assert torch.count_nonzero(item["x"][1]) == 0
    if mode == "right":
        assert torch.count_nonzero(item["x"][0]) == 0


def test_sensor_modes_select_fixed_channel_order(tmp_path: Path) -> None:
    signal = np.tile(np.arange(6, dtype=np.float32), (40, 1))
    for side in ("left", "right"):
        np.save(tmp_path / f"p1_{side}.npy", signal)
    row = {"left_path": "p1_left.npy", "right_path": "p1_right.npy",
           "subject_id": "p1", "label": "PD", "activity": "CrossArms",
           "pair_id": "p1_pair", "fold": "", "source_condition": "Parkinson's"}
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [row])
    records = load_manifest(manifest, tmp_path, ["PD", "DD"], "CrossArms")
    gyro = ManifestTimeSeriesDataset(records, 6, "gyro", "bilateral", 32,
                                     "center", 0.0, ",", False)[0]["x"]
    assert gyro.shape == (2, 3, 32)
    np.testing.assert_array_equal(gyro[0, :, 0].numpy(), [3, 4, 5])


def test_mismatched_pair_length_is_rejected(tmp_path: Path) -> None:
    np.save(tmp_path / "left.npy", np.zeros((30, 6), np.float32))
    np.save(tmp_path / "right.npy", np.zeros((31, 6), np.float32))
    row = {"left_path": "left.npy", "right_path": "right.npy", "subject_id": "p1",
           "label": "PD", "activity": "CrossArms", "pair_id": "p1_pair",
           "fold": "", "source_condition": "Parkinson's"}
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [row])
    records = load_manifest(manifest, tmp_path, ["PD", "DD"], "CrossArms")
    dataset = ManifestTimeSeriesDataset(records, 6, "acc_gyro", "bilateral", 32,
                                        "center", 0.0, ",", False)
    with pytest.raises(ValueError, match="mismatch"):
        _ = dataset[0]


def test_prepare_pads_pairs_same_session_and_records_missing_wrist(tmp_path: Path) -> None:
    root = tmp_path / "pads"
    (root / "patients").mkdir(parents=True)
    (root / "movement" / "timeseries").mkdir(parents=True)
    for subject, complete in (("001", True), ("002", False)):
        (root / "patients" / f"patient_{subject}.json").write_text(
            json.dumps({"id": subject, "condition": "Parkinson's"}), encoding="utf-8"
        )
        records = []
        for location in (("LeftWrist", "RightWrist") if complete else ("LeftWrist",)):
            filename = f"timeseries/{subject}_CrossArms_{location}.txt"
            time = np.arange(16, dtype=np.float64)[:, None] / 100.0
            signal = np.zeros((16, 6), dtype=np.float64)
            np.savetxt(root / "movement" / filename,
                       np.concatenate((time, signal), axis=1), delimiter=",")
            records.append({"device_location": location, "file_name": filename,
                            "channels": EXPECTED_METADATA_CHANNELS})
        observation = {"subject_id": subject, "sampling_rate": 100,
                       "session": [{"record_name": "CrossArms", "records": records}]}
        (root / "movement" / f"observation_{subject}.json").write_text(
            json.dumps(observation), encoding="utf-8"
        )
    output = tmp_path / "paired.csv"
    audit = tmp_path / "audit.json"
    report = generate_manifest(root, output, "CrossArms", "exclude", audit)
    assert report["pairs_written"] == 1
    assert report["pairs_excluded"] == 1
    assert report["exclusions"][0]["subject_id"] == "002"
    records = load_manifest(output, root, ["PD", "DD"], "CrossArms")
    assert records[0].subject_id == "001"
    assert records[0].left_path.name.endswith("LeftWrist.txt")
    assert records[0].right_path.name.endswith("RightWrist.txt")


def test_inconsistent_subject_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        subject_labels(["p1", "p1"], [0, 1])


def test_pads_pd_vs_dd_condition_mapping() -> None:
    assert condition_to_label("Parkinson's") == "PD"
    assert condition_to_label("Essential Tremor") == "DD"
    assert condition_to_label("Healthy") is None
