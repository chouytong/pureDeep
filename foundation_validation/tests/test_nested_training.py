from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path

import pytest

from src.engine.nested_training import (
    _pool_inner_oof,
    _runtime_config,
    _threshold_metrics,
)
from src.utils.config import load_config


def test_runtime_fold_config_preserves_baseline_hyperparameters() -> None:
    base = load_config("configs/pads_multi_activity_v3_nested_cv.yaml")
    before = {
        "model": deepcopy(base["model"]),
        "loss": deepcopy(base["loss"]),
        "training": deepcopy(base["training"]),
        "data_processing": {
            key: deepcopy(base["data"][key])
            for key in (
                "activities",
                "wrist_mode",
                "sensor_mode",
                "sequence_length",
                "normalization_scope",
            )
        },
    }
    runtime = _runtime_config(
        base,
        phase="inner_training",
        fold_id="outer_0/inner_0",
        train_subjects=["001", "002"],
        validation_subjects=["003"],
        test_subjects=["004"],
    )
    assert runtime["model"] == before["model"]
    assert runtime["loss"] == before["loss"]
    assert runtime["training"] == before["training"]
    assert {
        key: runtime["data"][key] for key in before["data_processing"]
    } == before["data_processing"]
    assert "nested_runtime" not in base
    assert runtime["nested_runtime"]["train_subject_count"] == 2


def _write_oof(path: Path, subjects: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "subject_id",
                "target",
                "probability_pd",
                "probability_dd",
            ),
        )
        writer.writeheader()
        for index, subject in enumerate(subjects):
            writer.writerow(
                {
                    "subject_id": subject,
                    "target": index % 2,
                    "probability_pd": 0.7,
                    "probability_dd": 0.3,
                }
            )


def test_pool_inner_oof_covers_outer_train_and_excludes_test(
    tmp_path: Path,
) -> None:
    outer = {
        "outer_fold": 0,
        "train_subjects": ["a", "b", "c"],
        "test_subjects": ["z"],
        "inner_folds": [
            {"inner_fold": 0},
            {"inner_fold": 1},
            {"inner_fold": 2},
        ],
    }
    for inner_index, subject in enumerate(("a", "b", "c")):
        _write_oof(
            tmp_path
            / f"inner_{inner_index}"
            / "predictions"
            / "validation.csv",
            [subject],
        )
    rows, path = _pool_inner_oof(outer, tmp_path)
    assert {row["subject_id"] for row in rows} == {"a", "b", "c"}
    assert path.is_file()


def test_pool_inner_oof_rejects_outer_test_leakage(tmp_path: Path) -> None:
    outer = {
        "outer_fold": 0,
        "train_subjects": ["a", "b", "c"],
        "test_subjects": ["z"],
        "inner_folds": [
            {"inner_fold": 0},
            {"inner_fold": 1},
            {"inner_fold": 2},
        ],
    }
    for inner_index, subject in enumerate(("a", "b", "z")):
        _write_oof(
            tmp_path
            / f"inner_{inner_index}"
            / "predictions"
            / "validation.csv",
            [subject],
        )
    with pytest.raises(ValueError, match="outer-train|Outer-test"):
        _pool_inner_oof(outer, tmp_path)


def test_threshold_metrics_uses_supplied_inner_threshold() -> None:
    rows = [
        {"target": 0, "probabilities": [0.8, 0.2]},
        {"target": 0, "probabilities": [0.6, 0.4]},
        {"target": 1, "probabilities": [0.4, 0.6]},
        {"target": 1, "probabilities": [0.1, 0.9]},
    ]
    metrics = _threshold_metrics(rows, threshold=0.5, zero_division=0.0)
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["pd_recall"] == 1.0
    assert metrics["dd_recall"] == 1.0
