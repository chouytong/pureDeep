from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import torch

from src.datasets.builders import build_datasets, make_loader
from src.datasets.folds import build_subject_fold_datasets
from src.engine.checkpoint import (
    load_checkpoint,
    validate_checkpoint_compatibility,
)
from src.engine.runner import evaluate_epoch
from src.losses import build_loss
from src.models import build_model
from src.utils.artifacts import write_json
from src.utils.config import load_config, require_pads_classification_config
from src.utils.device import select_device
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PADS classifier")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--outer-fold", type=int, default=None)
    parser.add_argument("--inner-fold", type=int, default=None)
    parser.add_argument("--allow-legacy-checkpoint", action="store_true")
    parser.add_argument("--legacy-provenance", default=None)
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE"
    )
    return parser.parse_args()

def _build_evaluation_bundle(config: dict, args: argparse.Namespace):
    nested = config.get("nested_cv")
    if not nested:
        if args.outer_fold is not None or args.inner_fold is not None:
            raise ValueError("Fold arguments require a nested-CV config")
        return build_datasets(config)
    if args.outer_fold is None:
        raise ValueError("--outer-fold is required for V3 nested-CV evaluation")
    project_root = Path(config["_meta"]["project_root"])
    split_path = Path(str(nested["split_file"])).expanduser()
    if not split_path.is_absolute():
        split_path = project_root / split_path
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    if not 0 <= args.outer_fold < int(payload["outer_folds"]):
        raise ValueError(f"outer fold out of range: {args.outer_fold}")
    outer = payload["outer"][args.outer_fold]
    if args.split == "validation":
        if args.inner_fold is None:
            raise ValueError(
                "--inner-fold is required for nested validation evaluation"
            )
        if not 0 <= args.inner_fold < int(payload["inner_folds"]):
            raise ValueError(f"inner fold out of range: {args.inner_fold}")
        inner = outer["inner_folds"][args.inner_fold]
        train_subjects = inner["train_subjects"]
        validation_subjects = inner["validation_subjects"]
        fold_id = f"outer_{args.outer_fold}/inner_{args.inner_fold}"
    else:
        if args.inner_fold is not None:
            raise ValueError("--inner-fold must be omitted for outer-test evaluation")
        train_subjects = outer["train_subjects"]
        validation_subjects = []
        fold_id = f"outer_{args.outer_fold}/outer_final"
    return build_subject_fold_datasets(
        config,
        train_subject_ids=train_subjects,
        validation_subject_ids=validation_subjects,
        test_subject_ids=outer["test_subjects"],
        fold_id=fold_id,
    )


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    """提前检查输出冲突，但只在评估成功后创建目录。"""
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
        if output_dir.exists() and (
            not output_dir.is_dir() or any(output_dir.iterdir())
        ):
            raise FileExistsError(
                f"Refusing to overwrite evaluation output: {output_dir}"
            )
        return output_dir
    root = Path(args.checkpoint).expanduser().resolve().parent.parent / "predictions"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = root / f"evaluate_{args.split}-{stamp}"
    suffix = 1
    while output_dir.exists():
        output_dir = root / f"evaluate_{args.split}-{stamp}-{suffix:02d}"
        suffix += 1
    return output_dir


def main() -> None:
    args = parse_args()
    output_dir = _resolve_output_dir(args)
    config = load_config(args.config, args.overrides)
    require_pads_classification_config(config)
    seed_everything(
        int(config["experiment"]["seed"]),
        bool(config["experiment"].get("deterministic", True)),
    )
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    validate_checkpoint_compatibility(
        checkpoint,
        config,
        checkpoint_path=args.checkpoint,
        allow_legacy_checkpoint=args.allow_legacy_checkpoint,
        legacy_provenance=args.legacy_provenance,
    )
    bundle = _build_evaluation_bundle(config, args)
    validate_checkpoint_compatibility(
        checkpoint,
        config,
        checkpoint_path=args.checkpoint,
        runtime_mean=bundle.mean,
        runtime_std=bundle.std,
        allow_legacy_checkpoint=args.allow_legacy_checkpoint,
        legacy_provenance=args.legacy_provenance,
    )
    # 评估必须复用 checkpoint 中由训练受试者拟合的统计量，不能重新用测试集拟合。
    checkpoint_mean = checkpoint["normalization"]["mean"].to(torch.float32)
    checkpoint_std = checkpoint["normalization"]["std"].to(torch.float32)
    if checkpoint_mean.shape != bundle.mean.shape:
        raise ValueError(
            "Checkpoint normalization channels do not match the current dataset"
        )
    for dataset in (bundle.train, bundle.validation, bundle.test):
        if dataset is not None:
            dataset.mean = checkpoint_mean
            dataset.std = checkpoint_std
    model = build_model(config).to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    criterion = build_loss(config).to(device)
    dataset = bundle.validation if args.split == "validation" else bundle.test
    if dataset is None:
        raise ValueError(f"Requested split {args.split!r} is empty")
    loader = make_loader(
        dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
        pin_memory=bool(config["data"].get("pin_memory", False))
        and device.type == "cuda",
        seed=int(config["experiment"]["seed"]),
    )
    metrics, predictions = evaluate_epoch(
        model,
        loader,
        criterion,
        device,
        mixed_precision=bool(config["training"].get("mixed_precision", False))
        and device.type == "cuda",
        zero_division=float(config["evaluation"].get("zero_division", 0.0)),
    )
    metrics["class_names"] = bundle.class_names
    for row in predictions:
        row["target_label"] = bundle.class_names[int(row["target"])]
        row["prediction_label"] = bundle.class_names[int(row["prediction"])]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metrics.json", metrics)
    if bool(config["evaluation"].get("save_predictions", True)):
        with (output_dir / "predictions.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=predictions[0].keys())
            writer.writeheader()
            writer.writerows(predictions)
    print(f"{args.split} metrics: {metrics}")
    print(f"Saved evaluation to: {output_dir}")


if __name__ == "__main__":
    main()
