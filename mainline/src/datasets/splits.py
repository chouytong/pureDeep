from __future__ import annotations

from collections import defaultdict
from typing import Hashable, Iterable, Mapping, Sequence

import numpy as np


def subject_labels(
    subject_ids: Sequence[str], labels: Sequence[int]
) -> dict[str, int]:
    """为每个受试者确定唯一类别；同一 ID 出现冲突标签时立即拒绝。"""
    if len(subject_ids) != len(labels):
        raise ValueError("subject_ids and labels must have the same length")
    result: dict[str, int] = {}
    for subject_id, label in zip(subject_ids, labels):
        label = int(label)
        if subject_id in result and result[subject_id] != label:
            raise ValueError(
                f"Subject {subject_id!r} has inconsistent labels: "
                f"{result[subject_id]} and {label}"
            )
        result[subject_id] = label
    return result


def stratified_subject_folds(
    labels_by_subject: Mapping[str, int],
    num_folds: int,
    seed: int,
) -> dict[str, int]:
    """按类别分层分配完整受试者，而不是分配窗口或记录。"""
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2")
    grouped: dict[int, list[str]] = defaultdict(list)
    for subject_id, label in labels_by_subject.items():
        grouped[int(label)].append(str(subject_id))

    rng = np.random.default_rng(seed)
    assignment: dict[str, int] = {}
    for label in sorted(grouped):
        subjects = sorted(grouped[label])
        rng.shuffle(subjects)
        for index, subject_id in enumerate(subjects):
            assignment[subject_id] = index % num_folds
    return assignment


def train_validation_subjects(
    candidate_subjects: Iterable[str],
    labels_by_subject: Mapping[str, int],
    validation_fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    """在外层训练候选中按受试者生成验证集，防止同一人的片段跨集合。"""
    candidates = sorted(set(candidate_subjects))
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if validation_fraction == 0 or len(candidates) < 2:
        return set(candidates), set()

    grouped: dict[int, list[str]] = defaultdict(list)
    for subject_id in candidates:
        grouped[int(labels_by_subject[subject_id])].append(subject_id)

    rng = np.random.default_rng(seed)
    validation: set[str] = set()
    for label in sorted(grouped):
        subjects = sorted(grouped[label])
        rng.shuffle(subjects)
        if len(subjects) <= 1:
            count = 0
        else:
            count = max(1, int(round(len(subjects) * validation_fraction)))
            count = min(count, len(subjects) - 1)
        validation.update(subjects[:count])
    training = set(candidates) - validation
    if not training:
        raise ValueError("Validation split consumed all training subjects")
    return training, validation


def assert_disjoint_splits(
    train_subjects: Iterable[Hashable],
    validation_subjects: Iterable[Hashable],
    test_subjects: Iterable[Hashable],
) -> None:
    # 这是训练前的硬门槛。任何交集都意味着受试者身份或相邻片段可能泄漏。
    train = set(train_subjects)
    validation = set(validation_subjects)
    test = set(test_subjects)
    intersections = {
        "train/validation": train & validation,
        "train/test": train & test,
        "validation/test": validation & test,
    }
    leaks = {name: values for name, values in intersections.items() if values}
    if leaks:
        raise ValueError(f"Subject leakage detected: {leaks}")
