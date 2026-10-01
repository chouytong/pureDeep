#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from src.datasets.builders import make_loader
from src.datasets.folds import build_subject_fold_datasets
from src.engine.nested_training import run_nested_training
from src.models import build_model
from src.utils.config import load_config, require_pads_classification_config
from src.utils.provenance import runtime_identity, sha256_file
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run audited PADS V3 nested cross-validation"
    )
    parser.add_argument(
        "--config",
        default="configs/pads_multi_activity_v3_nested_cv.yaml",
    )
    parser.add_argument("--outer-fold", type=int, default=None)
    parser.add_argument("--inner-fold", type=int, default=None)
    parser.add_argument(
        "--mode", choices=["inner", "outer-final"], default="inner"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--smoke-train", action="store_true")
    action.add_argument("--run-full", action="store_true")
    parser.add_argument("--forward-smoke", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _resolve_project_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else project_root / path).resolve()


def _load_and_verify_split(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Path, str]:
    project_root = Path(config["_meta"]["project_root"])
    split_path = _resolve_project_path(
        str(config["nested_cv"]["split_file"]), project_root
    )
    expected_path = split_path.with_suffix(split_path.suffix + ".sha256")
    if not expected_path.is_file():
        raise FileNotFoundError(f"Missing split SHA-256 sidecar: {expected_path}")
    expected = expected_path.read_text(encoding="utf-8").split()[0]
    actual = sha256_file(split_path)
    if actual != expected:
        raise ValueError(
            f"Frozen nested split hash mismatch: expected={expected}, actual={actual}"
        )
    audit_path = split_path.parent / "split_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "pass" or audit.get("split_sha256") != actual:
        raise ValueError("Nested split audit is absent, failed, or hash-mismatched")
    return (
        json.loads(split_path.read_text(encoding="utf-8")),
        audit,
        split_path,
        actual,
    )


def _select_fold(
    payload: dict[str, Any],
    outer_fold: int,
    inner_fold: int | None,
    mode: str,
) -> tuple[list[str], list[str], list[str], str]:
    if not 0 <= outer_fold < int(payload["outer_folds"]):
        raise ValueError(f"outer_fold out of range: {outer_fold}")
    outer = payload["outer"][outer_fold]
    if int(outer["outer_fold"]) != outer_fold:
        raise ValueError("Outer fold ordering does not match fold identifiers")
    if mode == "inner":
        if inner_fold is None:
            raise ValueError("--inner-fold is required for --mode inner")
        if not 0 <= inner_fold < int(payload["inner_folds"]):
            raise ValueError(f"inner_fold out of range: {inner_fold}")
        inner = outer["inner_folds"][inner_fold]
        if int(inner["inner_fold"]) != inner_fold:
            raise ValueError("Inner fold ordering does not match fold identifiers")
        return (
            list(inner["train_subjects"]),
            list(inner["validation_subjects"]),
            list(outer["test_subjects"]),
            f"outer_{outer_fold}/inner_{inner_fold}",
        )
    if inner_fold is not None:
        raise ValueError("--inner-fold must be omitted for --mode outer-final")
    return (
        list(outer["train_subjects"]),
        [],
        list(outer["test_subjects"]),
        f"outer_{outer_fold}/outer_final",
    )


def _forward_smoke(
    config: dict[str, Any],
    bundle: Any,
) -> dict[str, Any]:
    loader = make_loader(
        bundle.train,
        batch_size=min(2, int(config["training"]["batch_size"])),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        seed=int(config["nested_cv"]["seed"]),
    )
    batch = next(iter(loader))
    model = build_model(config).cpu().eval()
    with torch.no_grad():
        outputs = model(
            batch["x"].cpu(),
            batch["wrist_mask"].cpu(),
            batch["activity_mask"].cpu(),
            batch["activity_lengths"].cpu(),
        )
    return {
        "batch_x_shape": list(batch["x"].shape),
        "batch_wrist_mask_shape": list(batch["wrist_mask"].shape),
        "batch_activity_mask_shape": list(batch["activity_mask"].shape),
        "logits_shape": list(outputs["logits"].shape),
        "probabilities_shape": list(outputs["probabilities"].shape),
        "finite_logits": bool(torch.isfinite(outputs["logits"]).all().item()),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    require_pads_classification_config(config)
    if args.smoke_train or args.run_full:
        if args.run_full and (
            args.outer_fold is not None or args.inner_fold is not None
        ):
            raise ValueError("Full nested CV always runs all folds; omit fold arguments")
        if args.smoke_train and (
            args.outer_fold is None or args.inner_fold is None
        ):
            raise ValueError("Smoke training requires --outer-fold and --inner-fold")
        result = run_nested_training(
            config,
            output_dir=args.output_dir,
            device_name=args.device,
            resume=args.resume,
            smoke=args.smoke_train,
            smoke_outer_fold=0 if args.outer_fold is None else args.outer_fold,
            smoke_inner_fold=0 if args.inner_fold is None else args.inner_fold,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    if args.outer_fold is None:
        raise ValueError("Dry-run requires --outer-fold")
    if args.resume:
        raise ValueError("--resume is only valid for smoke/full training")
    seed_everything(
        int(config["nested_cv"]["seed"]),
        bool(config["experiment"].get("deterministic", True)),
    )
    payload, split_audit, split_path, split_sha256 = _load_and_verify_split(config)
    train_subjects, validation_subjects, test_subjects, fold_id = _select_fold(
        payload, args.outer_fold, args.inner_fold, args.mode
    )
    bundle = build_subject_fold_datasets(
        config,
        train_subject_ids=train_subjects,
        validation_subject_ids=validation_subjects,
        test_subject_ids=test_subjects,
        fold_id=fold_id,
    )

    default_output = (
        Path(config["experiment"]["output_root"])
        / "dry_runs"
        / fold_id
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_output.resolve()
    )
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise FileExistsError(f"Refusing to overwrite dry-run output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    normalization = {
        **bundle.normalization,
        "mean": bundle.mean.tolist(),
        "std": bundle.std.tolist(),
    }
    outer_audit = split_audit["folds"][args.outer_fold]
    if args.mode == "inner":
        assert args.inner_fold is not None
        inner_audit = outer_audit["inner"][args.inner_fold]
        diagnosis_counts = {
            "train": inner_audit["train"],
            "validation": inner_audit["validation"],
            "outer_test_held_out": outer_audit["test"],
        }
    else:
        diagnosis_counts = {
            "train": outer_audit["train"],
            "validation": None,
            "outer_test_held_out": outer_audit["test"],
        }
    train_batch_size = int(config["training"]["batch_size"])
    evaluation_batch_size = int(config["evaluation"]["batch_size"])
    dataloader_sizes = {
        "train_samples": len(bundle.train),
        "train_batches": (len(bundle.train) + train_batch_size - 1)
        // train_batch_size,
        "validation_samples": (
            0 if bundle.validation is None else len(bundle.validation)
        ),
        "validation_batches": (
            0
            if bundle.validation is None
            else (len(bundle.validation) + evaluation_batch_size - 1)
            // evaluation_batch_size
        ),
        "outer_test_loader_created": False,
    }
    report: dict[str, Any] = {
        "status": "pass",
        "dry_run": True,
        "training_started": False,
        "optimizer_created": False,
        "outer_test_evaluated": False,
        "outer_test_signal_accessed": False,
        "mode": args.mode,
        "fold_id": fold_id,
        "split_file": str(split_path),
        "split_sha256": split_sha256,
        "diagnosis_counts": diagnosis_counts,
        "dataloader_sizes": dataloader_sizes,
        "model_selection_protocol": config["model_selection"],
        "planned_artifact_paths": {
            "checkpoints": str(output_dir / "checkpoints"),
            "predictions": str(output_dir / "predictions"),
            "provenance": str(output_dir / "provenance"),
        },
        "split_summary": bundle.split_summary,
        "normalization": bundle.normalization,
        "runtime_provenance": runtime_identity(
            config, mean=bundle.mean, std=bundle.std
        ),
    }
    if args.forward_smoke:
        report["forward_smoke"] = _forward_smoke(config, bundle)
    (output_dir / "normalization.json").write_text(
        json.dumps(normalization, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "dry_run.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
