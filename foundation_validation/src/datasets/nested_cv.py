from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from .manifest import ManifestRecord


STRATUM_BY_CONDITION = {
    "Parkinson's": "PD",
    "Other Movement Disorders": "Other",
    "Essential Tremor": "ET",
    "Atypical Parkinsonism": "Atypical",
    "Multiple Sclerosis": "MS",
}
STRATA = ("PD", "Other", "ET", "Atypical", "MS")


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def subject_metadata_from_records(
    records: Sequence[ManifestRecord],
    required_activities: Sequence[str],
) -> tuple[dict[str, str], dict[str, set[str]]]:
    required = set(str(value) for value in required_activities)
    strata: dict[str, str] = {}
    labels: dict[str, int] = {}
    activities: dict[str, set[str]] = defaultdict(set)
    for record in records:
        condition = record.source_condition
        if condition not in STRATUM_BY_CONDITION:
            raise ValueError(
                f"Unsupported source_condition={condition!r} for {record.subject_id!r}"
            )
        stratum = STRATUM_BY_CONDITION[condition]
        previous_stratum = strata.setdefault(record.subject_id, stratum)
        previous_label = labels.setdefault(record.subject_id, record.label)
        if previous_stratum != stratum or previous_label != record.label:
            raise ValueError(
                f"Inconsistent diagnosis metadata for subject={record.subject_id!r}"
            )
        if (stratum == "PD") != (record.label == 0):
            raise ValueError(
                f"Label/diagnosis mismatch for subject={record.subject_id!r}: "
                f"label={record.label}, stratum={stratum}"
            )
        activities[record.subject_id].add(record.activity)
    if not strata:
        raise ValueError("No subjects available for nested cross-validation")
    for subject_id in sorted(strata):
        missing = required - activities[subject_id]
        extra = activities[subject_id] - required
        if missing or extra:
            raise ValueError(
                f"Subject {subject_id!r} activity mismatch: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
    return strata, dict(activities)


def _stratified_assignment(
    subject_strata: Mapping[str, str],
    num_folds: int,
    seed: int,
) -> dict[str, int]:
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2")
    rng = np.random.default_rng(seed)
    assignment: dict[str, int] = {}
    for stratum in STRATA:
        subjects = sorted(
            subject for subject, value in subject_strata.items() if value == stratum
        )
        if len(subjects) < num_folds:
            raise ValueError(
                f"Stratum {stratum!r} has {len(subjects)} subjects, "
                f"fewer than num_folds={num_folds}"
            )
        order = np.asarray(subjects, dtype=object)
        rng.shuffle(order)
        for index, subject in enumerate(order.tolist()):
            assignment[str(subject)] = index % num_folds
    if set(assignment) != set(subject_strata):
        raise AssertionError("Stratified assignment did not cover every subject")
    return assignment


def generate_nested_splits(
    subject_strata: Mapping[str, str],
    *,
    outer_folds: int = 5,
    inner_folds: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    subject_strata = {
        str(subject): str(stratum) for subject, stratum in subject_strata.items()
    }
    unknown = sorted(set(subject_strata.values()) - set(STRATA))
    if unknown:
        raise ValueError(f"Unknown diagnosis strata: {unknown}")
    outer_assignment = _stratified_assignment(subject_strata, outer_folds, seed)
    outer_payload: list[dict[str, Any]] = []
    all_subjects = set(subject_strata)
    for outer_fold in range(outer_folds):
        test_subjects = {
            subject
            for subject, fold in outer_assignment.items()
            if fold == outer_fold
        }
        train_subjects = all_subjects - test_subjects
        inner_strata = {
            subject: subject_strata[subject] for subject in sorted(train_subjects)
        }
        inner_assignment = _stratified_assignment(
            inner_strata,
            inner_folds,
            seed + 1000 * (outer_fold + 1),
        )
        inner_payload = []
        for inner_fold in range(inner_folds):
            validation_subjects = sorted(
                subject
                for subject, fold in inner_assignment.items()
                if fold == inner_fold
            )
            inner_train_subjects = sorted(
                train_subjects - set(validation_subjects)
            )
            inner_payload.append(
                {
                    "inner_fold": inner_fold,
                    "train_subjects": inner_train_subjects,
                    "validation_subjects": validation_subjects,
                }
            )
        outer_payload.append(
            {
                "outer_fold": outer_fold,
                "train_subjects": sorted(train_subjects),
                "test_subjects": sorted(test_subjects),
                "inner_folds": inner_payload,
            }
        )
    return {
        "schema_version": 1,
        "protocol": "nested_subject_stratified_5_diagnosis_strata",
        "seed": int(seed),
        "outer_folds": int(outer_folds),
        "inner_folds": int(inner_folds),
        "strata": list(STRATA),
        "subject_strata_sha256": payload_sha256(subject_strata),
        "outer": outer_payload,
    }


def _counts(
    subjects: Sequence[str],
    subject_strata: Mapping[str, str],
) -> dict[str, int]:
    values = Counter(subject_strata[subject] for subject in subjects)
    return {
        "total": len(subjects),
        "PD": values["PD"],
        "DD": sum(values[name] for name in STRATA if name != "PD"),
        "Other": values["Other"],
        "ET": values["ET"],
        "Atypical": values["Atypical"],
        "MS": values["MS"],
    }


def audit_nested_splits(
    payload: Mapping[str, Any],
    subject_strata: Mapping[str, str],
    subject_activities: Mapping[str, set[str]],
    required_activities: Sequence[str],
) -> dict[str, Any]:
    all_subjects = set(subject_strata)
    required = set(required_activities)
    errors: list[str] = []
    outer_test_occurrences: Counter[str] = Counter()
    fold_stats: list[dict[str, Any]] = []
    outer_entries = list(payload.get("outer", []))
    if len(outer_entries) != int(payload.get("outer_folds", -1)):
        errors.append("outer fold count does not match metadata")
    for outer in outer_entries:
        outer_index = int(outer["outer_fold"])
        outer_train = set(str(value) for value in outer["train_subjects"])
        outer_test = set(str(value) for value in outer["test_subjects"])
        overlap = outer_train & outer_test
        if overlap:
            errors.append(f"outer {outer_index}: train/test overlap={sorted(overlap)}")
        if outer_train | outer_test != all_subjects:
            errors.append(f"outer {outer_index}: subject coverage mismatch")
        outer_test_occurrences.update(outer_test)
        for split_name, subjects in (("train", outer_train), ("test", outer_test)):
            for stratum in STRATA:
                if not any(subject_strata[s] == stratum for s in subjects):
                    errors.append(
                        f"outer {outer_index} {split_name}: missing stratum {stratum}"
                    )
        inner_stats: list[dict[str, Any]] = []
        validation_occurrences: Counter[str] = Counter()
        inner_entries = list(outer.get("inner_folds", []))
        if len(inner_entries) != int(payload.get("inner_folds", -1)):
            errors.append(f"outer {outer_index}: inner fold count mismatch")
        for inner in inner_entries:
            inner_index = int(inner["inner_fold"])
            inner_train = set(str(value) for value in inner["train_subjects"])
            inner_validation = set(
                str(value) for value in inner["validation_subjects"]
            )
            if inner_train & inner_validation:
                errors.append(
                    f"outer {outer_index} inner {inner_index}: train/validation overlap"
                )
            if (inner_train | inner_validation) != outer_train:
                errors.append(
                    f"outer {outer_index} inner {inner_index}: coverage mismatch"
                )
            if (inner_train | inner_validation) & outer_test:
                errors.append(
                    f"outer {outer_index} inner {inner_index}: outer-test leakage"
                )
            validation_occurrences.update(inner_validation)
            for split_name, subjects in (
                ("train", inner_train),
                ("validation", inner_validation),
            ):
                for stratum in STRATA:
                    if not any(subject_strata[s] == stratum for s in subjects):
                        errors.append(
                            f"outer {outer_index} inner {inner_index} "
                            f"{split_name}: missing stratum {stratum}"
                        )
            inner_stats.append(
                {
                    "inner_fold": inner_index,
                    "train": _counts(sorted(inner_train), subject_strata),
                    "validation": _counts(
                        sorted(inner_validation), subject_strata
                    ),
                }
            )
        if set(validation_occurrences) != outer_train or any(
            count != 1 for count in validation_occurrences.values()
        ):
            errors.append(
                f"outer {outer_index}: each outer-train subject must be "
                "inner-validation exactly once"
            )
        fold_stats.append(
            {
                "outer_fold": outer_index,
                "train": _counts(sorted(outer_train), subject_strata),
                "test": _counts(sorted(outer_test), subject_strata),
                "inner": inner_stats,
            }
        )
    if set(outer_test_occurrences) != all_subjects or any(
        count != 1 for count in outer_test_occurrences.values()
    ):
        errors.append("each subject must be outer-test exactly once")
    for subject in sorted(all_subjects):
        if subject not in subject_activities:
            errors.append(f"subject {subject}: no activity inventory")
            continue
        missing = required - set(subject_activities[subject])
        extra = set(subject_activities[subject]) - required
        if missing or extra:
            errors.append(
                f"subject {subject}: missing={sorted(missing)}, extra={sorted(extra)}"
            )
    unknown = {
        subject
        for outer in outer_entries
        for key in ("train_subjects", "test_subjects")
        for subject in outer[key]
        if subject not in all_subjects
    }
    if unknown:
        errors.append(f"unknown subjects in split: {sorted(unknown)}")
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "subject_counts": _counts(sorted(all_subjects), subject_strata),
        "required_activities": list(required_activities),
        "outer_test_once": not errors
        and all(outer_test_occurrences[s] == 1 for s in all_subjects),
        "folds": fold_stats,
    }


def render_split_summary(audit: Mapping[str, Any]) -> str:
    lines = [
        "# PADS V3 Nested Cross-Validation Split Summary",
        "",
        f"Status: **{audit['status']}**",
        "",
        "| Outer | Split | Total | PD | DD | Other | ET | Atypical | MS |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in audit["folds"]:
        for split_name in ("train", "test"):
            counts = fold[split_name]
            lines.append(
                f"| {fold['outer_fold']} | {split_name} | {counts['total']} | "
                f"{counts['PD']} | {counts['DD']} | {counts['Other']} | "
                f"{counts['ET']} | {counts['Atypical']} | {counts['MS']} |"
            )
    lines.extend(["", "## Inner folds", ""])
    for fold in audit["folds"]:
        lines.append(f"### Outer fold {fold['outer_fold']}")
        lines.append("")
        lines.append(
            "| Inner | Split | Total | PD | DD | Other | ET | Atypical | MS |"
        )
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
        for inner in fold["inner"]:
            for split_name in ("train", "validation"):
                counts = inner[split_name]
                lines.append(
                    f"| {inner['inner_fold']} | {split_name} | {counts['total']} | "
                    f"{counts['PD']} | {counts['DD']} | {counts['Other']} | "
                    f"{counts['ET']} | {counts['Atypical']} | {counts['MS']} |"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
