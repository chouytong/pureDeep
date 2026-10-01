#!/usr/bin/env python3
"""Read-only integrity and completeness checks for the deep_final bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    sums = ROOT / "SHA256SUMS"
    if not sums.is_file():
        raise FileNotFoundError(sums)
    checked = 0
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        target = ROOT / relative
        if not target.is_file():
            raise FileNotFoundError(target)
        actual = sha256(target)
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch: {relative}")
        checked += 1

    summary = json.loads(
        (ROOT / "artifacts/final_nested_cv/nested_cv_summary.json").read_text()
    )
    if summary["subject_count"] != 390 or summary["outer_fold_count"] != 5:
        raise RuntimeError("Unexpected final nested-CV dimensions")
    if len(summary["outer_folds"]) != 5:
        raise RuntimeError("Expected five outer-fold summaries")

    prediction_path = (
        ROOT / "artifacts/final_nested_cv/outer_test_predictions_all_folds.csv"
    )
    with prediction_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 390:
        raise RuntimeError(f"Expected 390 pooled predictions, found {len(rows)}")

    for fold in range(5):
        model_dir = ROOT / f"models/outer_{fold}"
        required = [
            "final.pt",
            "final.pt.provenance.json",
            "config.yaml",
            "normalization.json",
            "split.json",
            "provenance.json",
            "metrics_thresholded.json",
            "metrics_default_argmax.json",
            "predictions.csv",
            "selection.json",
            "epochs.jsonl",
        ]
        missing = [name for name in required if not (model_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"outer_{fold} missing: {missing}")
        norm = json.loads((model_dir / "normalization.json").read_text())
        if norm.get("fitted_on") != "explicit_training_subjects_only":
            raise RuntimeError(f"outer_{fold} normalization provenance invalid")

    manifest = json.loads((ROOT / "BUNDLE_MANIFEST.json").read_text())
    if manifest["selected_model"] != "pads_pure_deep_normfree_moments_v8":
        raise RuntimeError("Bundle selected_model mismatch")
    for item in manifest["outer_models"]:
        checkpoint = ROOT / item["directory"] / "final.pt"
        if sha256(checkpoint) != item["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint}")
    final_metrics = manifest["final_evaluation"]
    pooled = summary["pooled_outer_test_metrics"]
    if abs(final_metrics["balanced_accuracy"] - pooled["balanced_accuracy"]) > 1e-12:
        raise RuntimeError("Manifest/final-summary balanced accuracy mismatch")

    frozen_project = ROOT / "frozen_project"
    sys.path.insert(0, str(frozen_project))
    from src.utils.provenance import source_tree_manifest

    _, source_hash = source_tree_manifest(frozen_project)
    expected_source_hash = "3f08832ee12c7b28c24f9b18c6248e892cbcc944429351d7fe2aad62d15336b6"
    if source_hash != expected_source_hash:
        raise RuntimeError("Frozen source-tree provenance mismatch")
    print(
        json.dumps(
            {
                "status": "PASS",
                "files_hashed": checked,
                "outer_models": 5,
                "pooled_predictions": len(rows),
                "subject_count": summary["subject_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
