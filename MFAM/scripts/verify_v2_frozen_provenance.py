#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from src.utils.provenance import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify frozen V2 artifacts without rewriting them"
    )
    parser.add_argument(
        "--provenance",
        default="outputs/pads_classification/v2_frozen_provenance/provenance.json",
    )
    return parser.parse_args()


def _check_file(
    root: Path,
    relative_path: str,
    expected: str,
    failures: list[dict[str, str]],
) -> None:
    path = (root / relative_path).resolve()
    try:
        actual = sha256_file(path)
    except FileNotFoundError:
        failures.append(
            {"path": relative_path, "expected": expected, "actual": "missing"}
        )
        return
    if actual != expected:
        failures.append(
            {"path": relative_path, "expected": expected, "actual": actual}
        )


def main() -> None:
    args = parse_args()
    provenance_path = Path(args.provenance).expanduser().resolve()
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    root = Path(provenance["project_root"]).resolve()
    failures: list[dict[str, str]] = []
    checked: set[str] = set()

    for key in ("canonical_config", "frozen_split"):
        item = provenance[key]
        _check_file(root, item["path"], item["sha256"], failures)
        checked.add(item["path"])

    config_path = root / provenance["canonical_config"]["path"]
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifests = raw_config["data"]["activity_manifests"]
    for activity, expected in provenance["manifest_sha256"].items():
        relative_path = str(manifests[activity])
        _check_file(root, relative_path, expected, failures)
        checked.add(relative_path)

    for group in (
        "preprocessing_artifact_sha256",
        "formal_artifact_sha256",
    ):
        for relative_path, expected in provenance[group].items():
            _check_file(root, relative_path, expected, failures)
            checked.add(relative_path)

    source_item = provenance["source_manifest"]
    source_path = str(source_item["path"])
    _check_file(root, source_path, source_item["sha256"], failures)
    checked.add(source_path)
    source_bytes = (root / source_path).read_bytes()
    source_tree_hash = hashlib.sha256(source_bytes).hexdigest()
    if source_tree_hash != source_item["source_tree_sha256"]:
        failures.append(
            {
                "path": source_path,
                "expected": source_item["source_tree_sha256"],
                "actual": source_tree_hash,
            }
        )

    checkpoint_hashes = {
        digest
        for value in provenance["checkpoint_sha256"].values()
        for digest in value.values()
    }
    artifact_checkpoint_hashes = {
        digest
        for path, digest in provenance["formal_artifact_sha256"].items()
        if path.endswith(".pt")
    }
    if checkpoint_hashes != artifact_checkpoint_hashes:
        failures.append(
            {
                "path": "checkpoint_sha256",
                "expected": str(sorted(checkpoint_hashes)),
                "actual": str(sorted(artifact_checkpoint_hashes)),
            }
        )
    normalization_hashes = set(
        provenance["normalization_artifact_sha256"].values()
    )
    artifact_normalization_hashes = {
        digest
        for path, digest in provenance["formal_artifact_sha256"].items()
        if path.endswith("/normalization.json")
    }
    if normalization_hashes != artifact_normalization_hashes:
        failures.append(
            {
                "path": "normalization_artifact_sha256",
                "expected": str(sorted(normalization_hashes)),
                "actual": str(sorted(artifact_normalization_hashes)),
            }
        )

    report: dict[str, Any] = {
        "status": "pass" if not failures else "fail",
        "provenance": str(provenance_path),
        "checked_file_count": len(checked),
        "checkpoint_count": len(checkpoint_hashes),
        "failures": failures,
        "historical_source_tree_sha256": source_item["source_tree_sha256"],
        "note": (
            "Historical source hashes are verified through the frozen manifest; "
            "current V3 source is intentionally allowed to differ."
        ),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
