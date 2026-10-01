from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from src.datasets.builders import _activity_statistics
from src.datasets.manifest import ManifestRecord
from src.datasets.nested_cv import (
    STRATA,
    audit_nested_splits,
    generate_nested_splits,
)
from src.metrics.threshold import (
    final_epoch_from_inner_best,
    select_binary_threshold,
)


def _cohort() -> dict[str, str]:
    counts = {
        "PD": 276,
        "Other": 60,
        "ET": 28,
        "Atypical": 15,
        "MS": 11,
    }
    return {
        f"{stratum}-{index:03d}": stratum
        for stratum, count in counts.items()
        for index in range(count)
    }


def test_nested_cv_is_deterministic_complete_and_stratified() -> None:
    strata = _cohort()
    activities = {subject: {"A", "B"} for subject in strata}
    first = generate_nested_splits(strata, outer_folds=5, inner_folds=3, seed=42)
    repeated = generate_nested_splits(strata, outer_folds=5, inner_folds=3, seed=42)
    changed = generate_nested_splits(strata, outer_folds=5, inner_folds=3, seed=43)
    assert first == repeated
    assert first["outer"] != changed["outer"]
    audit = audit_nested_splits(first, strata, activities, ["A", "B"])
    assert audit["status"] == "pass"
    assert audit["outer_test_once"] is True
    assert audit["subject_counts"] == {
        "total": 390,
        "PD": 276,
        "DD": 114,
        "Other": 60,
        "ET": 28,
        "Atypical": 15,
        "MS": 11,
    }
    for outer in audit["folds"]:
        assert all(outer["test"][stratum] > 0 for stratum in STRATA)
        for inner in outer["inner"]:
            assert all(inner["validation"][stratum] > 0 for stratum in STRATA)


def test_nested_cv_audit_detects_outer_test_leakage() -> None:
    strata = _cohort()
    activities = {subject: {"A"} for subject in strata}
    payload = generate_nested_splits(strata, outer_folds=5, inner_folds=3, seed=42)
    broken = deepcopy(payload)
    leaked = broken["outer"][0]["test_subjects"][0]
    broken["outer"][0]["inner_folds"][0]["train_subjects"].append(leaked)
    audit = audit_nested_splits(broken, strata, activities, ["A"])
    assert audit["status"] == "fail"
    assert any("outer-test leakage" in error for error in audit["errors"])


class _RawTrainOnly:
    def __init__(self) -> None:
        self.records = [
            ManifestRecord(
                left_path=Path("left"),
                right_path=Path("right"),
                subject_id=f"train-{index}",
                label=index,
                activity="A",
                pair_id=f"pair-{index}",
                fold=None,
            )
            for index in range(2)
        ]
        self._signals = [
            torch.ones(2, 1, 4),
            torch.full((2, 1, 4), 3.0),
        ]

    def __len__(self) -> int:
        return len(self._signals)

    def statistics_signal(self, index: int) -> torch.Tensor:
        return self._signals[index]


def test_fold_normalization_uses_only_explicit_train_dataset() -> None:
    mean, std = _activity_statistics(_RawTrainOnly(), ["A"], epsilon=1e-6)
    assert mean.shape == (1, 2, 1, 1)
    assert torch.allclose(mean, torch.full_like(mean, 2.0))
    assert torch.allclose(std, torch.ones_like(std))
    validation_and_test_values = torch.tensor([100.0, 1000.0])
    assert not torch.isclose(mean.mean(), validation_and_test_values.mean())


def test_threshold_and_final_epoch_protocol() -> None:
    selected = select_binary_threshold(
        [0, 0, 1, 1],
        [0.1, 0.4, 0.6, 0.9],
        metric="balanced_accuracy",
    )
    assert selected["source"] == "inner_oof_only"
    assert selected["metric"] == "balanced_accuracy"
    assert selected["balanced_accuracy"] == 1.0
    epoch = final_epoch_from_inner_best([7, 11, 9])
    assert epoch["selected_epoch_count"] == 9
