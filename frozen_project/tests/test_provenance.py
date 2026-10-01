from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from src.engine.checkpoint import validate_checkpoint_compatibility
from src.utils.provenance import (
    DEFAULT_LABEL_MAPPING,
    runtime_identity,
    semantic_config_sha256,
    sha256_file,
    validate_strict_provenance,
)


def _toy_config(tmp_path: Path) -> dict:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='toy'\n", encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text('{"fold": 0}\n', encoding="utf-8")
    manifest_a = tmp_path / "A.csv"
    manifest_b = tmp_path / "B.csv"
    manifest_a.write_text("a\n", encoding="utf-8")
    manifest_b.write_text("b\n", encoding="utf-8")
    preprocessing = tmp_path / "processing.json"
    preprocessing.write_text('{"trim": 12}\n', encoding="utf-8")
    return {
        "_meta": {"project_root": str(tmp_path), "config_path": str(tmp_path / "c.yaml")},
        "task": {"type": "pads_classification"},
        "experiment": {"name": "toy_v3"},
        "nested_cv": {"split_file": str(split)},
        "data": {
            "type": "pads",
            "unit": "subject",
            "labels": ["PD", "DD"],
            "activities": ["A", "B"],
            "activity_manifests": {
                "A": str(manifest_a),
                "B": str(manifest_b),
            },
            "wrist_order": ["left", "right"],
            "wrist_mode": "bilateral",
            "sensor_mode": "acc_gyro",
            "normalization_scope": "activity_wrist_channel",
        },
        "provenance": {
            "strict_checkpoint_validation": True,
            "experiment_version": "toy_v3",
            "pads_dataset_version": "1.0.0",
            "pads_doi": "10.13026/m0w9-zx22",
            "label_mapping": DEFAULT_LABEL_MAPPING,
            "preprocessing_artifacts": [str(preprocessing)],
        },
    }


def _strict_checkpoint(config: dict) -> tuple[dict, torch.Tensor, torch.Tensor]:
    mean = torch.zeros(2, 2, 6, 1)
    std = torch.ones(2, 2, 6, 1)
    return (
        {
            "class_names": ["PD", "DD"],
            "normalization": {"mean": mean, "std": std},
            "provenance": runtime_identity(config, mean, std),
        },
        mean,
        std,
    )


def test_strict_provenance_accepts_exact_runtime(tmp_path: Path) -> None:
    config = _toy_config(tmp_path)
    checkpoint, mean, std = _strict_checkpoint(config)
    validate_strict_provenance(checkpoint, config, mean, std)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("config", "config_sha256"),
        ("split", "split_sha256"),
        ("manifest", "manifest_sha256"),
        ("source", "source_tree_sha256"),
        ("activity", "activity_order"),
    ],
)
def test_strict_provenance_rejects_identity_mismatch(
    tmp_path: Path, mutation: str, match: str
) -> None:
    config = _toy_config(tmp_path)
    checkpoint, mean, std = _strict_checkpoint(config)
    changed = deepcopy(config)
    if mutation == "config":
        changed["experiment"]["note"] = "changed"
    elif mutation == "split":
        Path(changed["nested_cv"]["split_file"]).write_text(
            '{"fold": 1}\n', encoding="utf-8"
        )
    elif mutation == "manifest":
        Path(changed["data"]["activity_manifests"]["A"]).write_text(
            "changed\n", encoding="utf-8"
        )
    elif mutation == "source":
        (tmp_path / "pyproject.toml").write_text(
            "[project]\nname='changed'\n", encoding="utf-8"
        )
    elif mutation == "activity":
        changed["data"]["activities"] = ["B", "A"]
    with pytest.raises(ValueError, match=match):
        validate_strict_provenance(checkpoint, changed, mean, std)


def test_strict_provenance_rejects_normalization_mismatch(tmp_path: Path) -> None:
    config = _toy_config(tmp_path)
    checkpoint, mean, std = _strict_checkpoint(config)
    with pytest.raises(ValueError, match="normalization_sha256"):
        validate_strict_provenance(checkpoint, config, mean + 1.0, std)


def test_strict_provenance_rejects_class_order(tmp_path: Path) -> None:
    config = _toy_config(tmp_path)
    changed = deepcopy(config)
    changed["data"]["labels"] = ["DD", "PD"]
    with pytest.raises(ValueError, match="class order"):
        runtime_identity(changed)


def test_legacy_checkpoint_requires_explicit_sidecar(tmp_path: Path) -> None:
    config = _toy_config(tmp_path)
    config["provenance"]["strict_checkpoint_validation"] = False
    config["experiment"]["name"] = "legacy"
    checkpoint_path = tmp_path / "legacy.pt"
    checkpoint_path.write_bytes(b"legacy-checkpoint")
    mean = torch.zeros(2, 2, 6, 1)
    std = torch.ones(2, 2, 6, 1)
    runtime = runtime_identity(config, mean, std)
    clean_config = deepcopy(config)
    clean_config.pop("_meta")
    checkpoint = {
        "class_names": ["PD", "DD"],
        "config": clean_config,
        "normalization": {"mean": mean, "std": std},
        "input_metadata": {
            "tensor_layout": "[batch, activity, wrist, channel, time]",
            "wrist_order": ["left", "right"],
            "wrist_mode": "bilateral",
            "sensor_mode": "acc_gyro",
            "channel_names": ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"],
            "unit": "subject",
            "activities": ["A", "B"],
            "activity_count": 2,
            "activity_fusion": "attention",
            "time_length": "variable_full_signal",
            "normalization_scope": "activity_wrist_channel",
        },
    }
    sidecar = {
        "experiment_version": "v2_frozen_historical_baseline",
        "task": {
            key: runtime[key]
            for key in (
                "class_names",
                "label_mapping",
                "activity_order",
                "wrist_order",
                "sensor_mode",
                "channel_order",
            )
        },
        "frozen_split": {"sha256": runtime["split_sha256"]},
        "manifest_sha256": runtime["manifest_sha256"],
        "preprocessing_artifact_sha256": runtime["preprocessing_sha256"],
        "checkpoint_sha256": {"42": {"best": sha256_file(checkpoint_path)}},
    }
    sidecar_path = tmp_path / "legacy-provenance.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match="Legacy V2 checkpoint rejected"):
        validate_checkpoint_compatibility(
            checkpoint, config, checkpoint_path=checkpoint_path
        )
    validate_checkpoint_compatibility(
        checkpoint,
        config,
        checkpoint_path=checkpoint_path,
        runtime_mean=mean,
        runtime_std=std,
        allow_legacy_checkpoint=True,
        legacy_provenance=sidecar_path,
    )
    wrong_order = deepcopy(checkpoint)
    wrong_order["class_names"] = ["DD", "PD"]
    with pytest.raises(ValueError, match="class_names"):
        validate_checkpoint_compatibility(
            wrong_order,
            config,
            checkpoint_path=checkpoint_path,
            allow_legacy_checkpoint=True,
            legacy_provenance=sidecar_path,
        )
    assert semantic_config_sha256(clean_config) == semantic_config_sha256(config)


def test_legacy_checkpoint_rejects_unrecorded_bytes(tmp_path: Path) -> None:
    config = _toy_config(tmp_path)
    checkpoint, _, _ = _strict_checkpoint(config)
    checkpoint.pop("provenance")
    checkpoint["config"] = {k: v for k, v in config.items() if k != "_meta"}
    checkpoint["input_metadata"] = {
        "tensor_layout": "[batch, activity, wrist, channel, time]",
        "wrist_order": ["left", "right"],
        "wrist_mode": "bilateral",
        "sensor_mode": "acc_gyro",
        "channel_names": ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"],
        "unit": "subject",
        "activities": ["A", "B"],
        "activity_count": 2,
        "activity_fusion": "attention",
        "time_length": "variable_full_signal",
        "normalization_scope": "activity_wrist_channel",
    }
    path = tmp_path / "legacy.pt"
    path.write_bytes(b"actual")
    sidecar = tmp_path / "sidecar.json"
    sidecar.write_text(
        json.dumps(
            {
                "experiment_version": "v2_frozen_historical_baseline",
                "checkpoint_sha256": {"42": {"best": "0" * 64}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not recorded"):
        validate_checkpoint_compatibility(
            checkpoint,
            config,
            checkpoint_path=path,
            allow_legacy_checkpoint=True,
            legacy_provenance=sidecar,
        )
