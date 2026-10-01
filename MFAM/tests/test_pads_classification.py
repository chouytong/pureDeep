from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from src.datasets.builders import _split_records
from src.datasets.manifest import ManifestRecord, crop_or_pad, multi_crop_or_pad
from src.utils.config import load_config, require_pads_classification_config
from evaluate import _resolve_output_dir


def _record(subject_id: str, label: int) -> ManifestRecord:
    return ManifestRecord(
        left_path=Path(f"{subject_id}_left.txt"),
        right_path=Path(f"{subject_id}_right.txt"),
        subject_id=subject_id,
        label=label,
        activity="CrossArms",
        pair_id=subject_id,
        fold=None,
    )


def test_default_pads_classification_scope() -> None:
    config = load_config("configs/pads_multi_activity_v2.yaml")
    require_pads_classification_config(config)
    assert config["task"]["name"] == "pd_vs_dd_subject_multi_activity_full_length_l1"
    assert config["data"]["labels"] == ["PD", "DD"]
    assert config["data"]["wrist_mode"] == "bilateral"
    assert config["data"]["sensor_mode"] == "acc_gyro"
    assert config["data"]["unit"] == "subject"
    assert config["data"]["sequence_length"] is None


def test_classification_entry_rejects_nonclassification_config() -> None:
    with pytest.raises(ValueError, match="only supports PADS classification"):
        require_pads_classification_config(
            {"task": {"type": "regression"}, "data": {"type": "pads"}}
        )


def test_evaluation_refuses_nonempty_output(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    (output / "metrics.json").write_text("{}", encoding="utf-8")
    args = Namespace(output_dir=str(output), checkpoint="best.pt", split="test")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _resolve_output_dir(args)


def test_frozen_subject_split_is_loaded_and_hashed(tmp_path: Path) -> None:
    records = [_record(f"p{index}", index % 2) for index in range(6)]
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"train_subjects": ["p0", "p1", "p2", "p3"],
                                      "validation_subjects": ["p4"],
                                      "test_subjects": ["p5"]}), encoding="utf-8")
    train, validation, test, summary = _split_records(
        records, 5, 0, 0.0, 42, False, str(split_path)
    )
    assert [record.subject_id for record in train] == ["p0", "p1", "p2", "p3"]
    assert [record.subject_id for record in validation] == ["p4"]
    assert [record.subject_id for record in test] == ["p5"]
    assert summary["fold_source"] == "frozen_file"
    assert len(summary["split_file_sha256"]) == 64


def test_paired_crop_shapes_and_common_endpoints() -> None:
    signal = np.arange(2 * 3 * 20, dtype=np.float32).reshape(2, 3, 20)
    random_crop = crop_or_pad(signal, 8, "random", 0.0)
    views = multi_crop_or_pad(signal, 8, 3, 0.0)
    assert random_crop.shape == (2, 3, 8)
    assert views.shape == (3, 2, 3, 8)
    np.testing.assert_array_equal(views[0], signal[..., :8])
    np.testing.assert_array_equal(views[-1], signal[..., -8:])

