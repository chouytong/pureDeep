#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from typing import Any

from src.engine.nested_training import _load_frozen_split, _run_inner_fold
from src.utils.artifacts import write_json
from src.utils.config import load_config, require_pads_classification_config, save_config
from src.utils.device import select_device
from src.utils.environment import collect_environment, write_environment
from src.utils.provenance import semantic_config_sha256, source_tree_manifest


METRICS = (
    "accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
    "macro_f1", "macro_auroc", "pd_recall", "dd_recall",
    "negative_log_likelihood", "brier_score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inner-CV-only pure-deep development; outer test is inaccessible"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    require_pads_classification_config(config)
    payload, audit, split_path, split_sha256 = _load_frozen_split(config)
    if audit.get("status") != "pass":
        raise ValueError("Frozen split audit did not pass")
    device = select_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("Formal development training requires CUDA")
    output = Path(args.output_dir).expanduser().resolve()
    _, source_hash = source_tree_manifest(config["_meta"]["project_root"])
    protocol_path = output / "development_protocol.json"
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        development = config.get("development", {})
        save_config(config, output / "config.yaml")
        write_environment(
            collect_environment(config["_meta"]["project_root"]),
            output / "environment.json",
        )
        write_json(output / "experiment_plan.json", {
            "hypothesis": development.get("hypothesis", (
                "Attention+mean+std pooling over learned temporal feature maps preserves "
                "global DD information better than fixed-band Hard Top-K MIL."
            )),
            "pure_deep": True,
            "handcrafted_features": False,
            "primary_metric": "balanced_accuracy",
            "comparison": development.get("comparison", ["M0", "R1"]),
            "seed": int(config["experiment"]["seed"]),
            "folds": "fixed 5 outer contexts x 3 inner validation folds",
            "stop_rule": development.get(
                "stop_rule",
                "Reject if BA gain <0.01 or BA/AUROC/DD-recall do not improve coherently",
            ),
        })
        write_json(protocol_path, {
            "status": "running",
            "scope": "development_inner_cv_only",
            "started_at_utc": utc_now(),
            "config_sha256": semantic_config_sha256(config),
            "split_sha256": split_sha256,
            "source_tree_sha256": source_hash,
            "outer_test_loader_created": False,
            "outer_test_signal_accessed": False,
            "outer_test_predictions_accessed": False,
        })
    summaries: list[dict[str, Any]] = []
    for outer in payload["outer"]:
        outer_index = int(outer["outer_fold"])
        for inner in outer["inner_folds"]:
            inner_index = int(inner["inner_fold"])
            summaries.append(_run_inner_fold(
                config, outer, inner,
                output / f"outer_{outer_index}" / f"inner_{inner_index}",
                device, resume=args.resume, smoke=False,
            ))
    metric_summary: dict[str, Any] = {}
    for name in METRICS:
        values = [float(row["validation_metrics"][name]) for row in summaries]
        metric_summary[name] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values),
            "values": values,
        }
    result = {
        "status": "complete",
        "scope": "development_inner_cv_only",
        "fold_count": len(summaries),
        "metric_summary": metric_summary,
        "fold_summaries": summaries,
        "outer_test_loader_created": False,
        "outer_test_signal_accessed": False,
        "outer_test_predictions_accessed": False,
        "completed_at_utc": utc_now(),
    }
    write_json(output / "development_summary.json", result)
    started_at = None
    if protocol_path.exists():
        with protocol_path.open(encoding="utf-8") as handle:
            started_at = json.load(handle).get("started_at_utc")
    protocol = {
        "status": "complete",
        "scope": "development_inner_cv_only",
        "started_at_utc": started_at,
        "completed_at_utc": result["completed_at_utc"],
        "config_sha256": semantic_config_sha256(config),
        "split_sha256": split_sha256,
        "source_tree_sha256": source_hash,
        "outer_test_loader_created": False,
        "outer_test_signal_accessed": False,
        "outer_test_predictions_accessed": False,
        "fold_count": len(summaries),
    }
    write_json(protocol_path, protocol)
    print(f"complete folds={len(summaries)} BA={metric_summary['balanced_accuracy']['mean']:.4f} "
          f"AUROC={metric_summary['macro_auroc']['mean']:.4f}")


if __name__ == "__main__":
    main()
