#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.datasets.builders import load_configured_records
from src.datasets.nested_cv import (
    audit_nested_splits,
    canonical_json_bytes,
    generate_nested_splits,
    render_split_summary,
    subject_metadata_from_records,
)
from src.utils.config import load_config, require_pads_classification_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and audit frozen PADS V3 nested-CV subject splits"
    )
    parser.add_argument(
        "--config",
        default="configs/pads_multi_activity_v3_nested_cv.yaml",
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _refuse_nonempty(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty split directory: {path}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    require_pads_classification_config(config)
    nested = config["nested_cv"]
    project_root = Path(config["_meta"]["project_root"])
    configured_split = Path(str(nested["split_file"])).expanduser()
    if not configured_split.is_absolute():
        configured_split = project_root / configured_split
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else configured_split.resolve().parent
    )
    _refuse_nonempty(output_dir)

    class_names = [str(value) for value in config["data"]["labels"]]
    activities = [str(value) for value in config["data"]["activities"]]
    records = load_configured_records(config["data"], class_names)
    subject_strata, subject_activities = subject_metadata_from_records(
        records, activities
    )
    payload = generate_nested_splits(
        subject_strata,
        outer_folds=int(nested["outer_folds"]),
        inner_folds=int(nested["inner_folds"]),
        seed=int(nested["seed"]),
    )
    repeated = generate_nested_splits(
        subject_strata,
        outer_folds=int(nested["outer_folds"]),
        inner_folds=int(nested["inner_folds"]),
        seed=int(nested["seed"]),
    )
    alternative = generate_nested_splits(
        subject_strata,
        outer_folds=int(nested["outer_folds"]),
        inner_folds=int(nested["inner_folds"]),
        seed=int(nested["seed"]) + 1,
    )
    audit = audit_nested_splits(
        payload, subject_strata, subject_activities, activities
    )
    audit["determinism"] = {
        "same_seed_identical": payload == repeated,
        "different_seed_changes_assignment": payload["outer"] != alternative["outer"],
    }
    if not all(audit["determinism"].values()):
        audit["status"] = "fail"
        audit["errors"].append("split determinism audit failed")
    if audit["status"] != "pass":
        raise RuntimeError(json.dumps(audit["errors"], ensure_ascii=False))

    split_bytes = canonical_json_bytes(payload)
    split_sha256 = hashlib.sha256(split_bytes).hexdigest()
    audit["split_sha256"] = split_sha256
    audit["record_count"] = len(records)
    audit["subject_strata"] = {
        subject: subject_strata[subject] for subject in sorted(subject_strata)
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "nested_cv_splits.json").write_bytes(split_bytes)
    (output_dir / "nested_cv_splits.json.sha256").write_text(
        f"{split_sha256}  nested_cv_splits.json\n", encoding="utf-8"
    )
    (output_dir / "split_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "split_summary.md").write_text(
        render_split_summary(audit), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": audit["status"],
                "output_dir": str(output_dir),
                "split_sha256": split_sha256,
                "record_count": len(records),
                "subject_counts": audit["subject_counts"],
                "determinism": audit["determinism"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
