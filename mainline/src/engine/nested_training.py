from __future__ import annotations

import csv
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.datasets.builders import make_loader
from src.datasets.folds import SubjectFoldBundle, build_subject_fold_datasets
from src.engine.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_compatibility,
)
from src.engine.optim import build_optimizer, build_scheduler
from src.engine.runner import (
    compute_activity_disease_prototypes,
    evaluate_epoch,
    train_epoch,
)
from src.engine.self_supervised import run_masked_reconstruction_pretraining
from src.losses import build_loss
from src.metrics.classification import classification_metrics
from src.metrics.threshold import (
    final_epoch_from_inner_best,
    select_binary_threshold,
)
from src.models import build_model
from src.utils.artifacts import append_jsonl, write_json
from src.utils.config import save_config
from src.utils.device import select_device
from src.utils.environment import collect_environment, write_environment
from src.utils.model_stats import count_parameters
from src.utils.provenance import (
    runtime_identity,
    semantic_config_sha256,
    sha256_file,
    sha256_json,
    source_tree_manifest,
)
from src.utils.seed import seed_everything


METRIC_NAMES = (
    "balanced_accuracy",
    "macro_f1",
    "accuracy",
    "macro_auroc",
    "negative_log_likelihood",
    "brier_score",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _metric_improved(
    value: float, best: float, mode: str, minimum_delta: float
) -> bool:
    if mode == "max":
        return value > best + minimum_delta
    if mode == "min":
        return value < best - minimum_delta
    raise ValueError(f"early_stopping.mode must be max or min, got {mode!r}")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _load_frozen_split(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Path, str]:
    root = Path(str(config["_meta"]["project_root"]))
    split_path = Path(str(config["nested_cv"]["split_file"])).expanduser()
    if not split_path.is_absolute():
        split_path = root / split_path
    split_path = split_path.resolve()
    digest_path = split_path.with_suffix(split_path.suffix + ".sha256")
    expected = digest_path.read_text(encoding="utf-8").split()[0]
    actual = sha256_file(split_path)
    if actual != expected:
        raise ValueError(
            f"Frozen split SHA-256 mismatch: expected={expected}, actual={actual}"
        )
    payload = _read_json(split_path)
    audit = _read_json(split_path.parent / "split_audit.json")
    if audit.get("status") != "pass" or audit.get("split_sha256") != actual:
        raise ValueError("Frozen split audit is failed or does not match split SHA-256")
    if int(payload["outer_folds"]) != 5 or int(payload["inner_folds"]) != 3:
        raise ValueError("V3 baseline requires exactly 5 outer x 3 inner folds")
    return payload, audit, split_path, actual


def _runtime_config(
    base: Mapping[str, Any],
    *,
    phase: str,
    fold_id: str,
    train_subjects: Sequence[str],
    validation_subjects: Sequence[str],
    test_subjects: Sequence[str],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = deepcopy(dict(base))
    runtime = {
        "phase": phase,
        "fold_id": fold_id,
        "train_subject_ids_sha256": sha256_json(sorted(train_subjects)),
        "validation_subject_ids_sha256": sha256_json(sorted(validation_subjects)),
        "test_subject_ids_sha256": sha256_json(sorted(test_subjects)),
        "train_subject_count": len(train_subjects),
        "validation_subject_count": len(validation_subjects),
        "test_subject_count": len(test_subjects),
    }
    if extra:
        runtime.update(dict(extra))
    config["nested_runtime"] = runtime
    return config


def _write_normalization(path: Path, bundle: SubjectFoldBundle) -> None:
    write_json(
        path,
        {
            **bundle.normalization,
            "mean": bundle.mean.tolist(),
            "std": bundle.std.tolist(),
        },
    )


def _loader(
    dataset: Any,
    config: Mapping[str, Any],
    *,
    train: bool,
    seed_offset: int,
):
    if dataset is None:
        return None
    data = config["data"]
    batch_size = int(
        config["training"]["batch_size"]
        if train
        else config["evaluation"]["batch_size"]
    )
    return make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=int(data.get("num_workers", 0)),
        pin_memory=bool(data.get("pin_memory", False))
        and torch.cuda.is_available(),
        seed=int(config["experiment"]["seed"]) + seed_offset,
    )


def _named_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    recalls = list(result.get("per_class_recall", []))
    if len(recalls) == 2:
        result["pd_recall"] = float(recalls[0])
        result["dd_recall"] = float(recalls[1])
    return result


def _resolve_train_only_class_weights(
    config: Mapping[str, Any], bundle: SubjectFoldBundle
) -> dict[str, Any]:
    """Resolve balanced CE weights from this fold's training labels only."""

    resolved = deepcopy(dict(config))
    specification = resolved.get("loss", {}).get("class_weights")
    if specification != "train_balanced":
        return resolved
    counts = bundle.split_summary["class_counts"]["train"]
    pd_count = int(counts["PD"])
    dd_count = int(counts["DD"])
    if pd_count <= 0 or dd_count <= 0:
        raise ValueError("Train-balanced loss requires both classes in the training fold")
    total = pd_count + dd_count
    weights = [total / (2.0 * pd_count), total / (2.0 * dd_count)]
    resolved["loss"] = {
        **resolved["loss"],
        "class_weights": weights,
        "class_weight_source": "current_fold_train_subject_labels_only",
        "class_weight_train_counts": {"PD": pd_count, "DD": dd_count},
    }
    return resolved


def _write_predictions(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    class_names: Sequence[str],
    *,
    threshold: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        probabilities = [float(value) for value in row["probabilities"]]
        copied = dict(row)
        copied["probabilities"] = json.dumps(probabilities)
        copied["probability_pd"] = probabilities[0]
        copied["probability_dd"] = probabilities[1]
        copied["target_label"] = class_names[int(row["target"])]
        copied["prediction_label"] = class_names[int(row["prediction"])]
        if threshold is not None:
            thresholded = int(probabilities[1] >= threshold)
            copied["prediction_thresholded"] = thresholded
            copied["prediction_thresholded_label"] = class_names[thresholded]
            copied["threshold"] = float(threshold)
        output_rows.append(copied)
    if not output_rows:
        raise ValueError("Cannot write empty predictions")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)


def _read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = []
    for row in rows:
        result.append(
            {
                "subject_id": row["subject_id"],
                "target": int(row["target"]),
                "probability_pd": float(row["probability_pd"]),
                "probability_dd": float(row["probability_dd"]),
            }
        )
    return result


def _threshold_metrics(
    rows: Sequence[Mapping[str, Any]],
    threshold: float,
    zero_division: float,
) -> dict[str, Any]:
    targets = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    probabilities = np.asarray(
        [
            [
                float(row.get("probability_pd", row["probabilities"][0])),
                float(row.get("probability_dd", row["probabilities"][1])),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    predictions = (probabilities[:, 1] >= float(threshold)).astype(np.int64)
    metrics = classification_metrics(
        targets,
        predictions,
        num_classes=2,
        zero_division=zero_division,
        probabilities=probabilities,
    )
    return _named_metrics(metrics)


def _prepare_stage(
    stage_dir: Path,
    config: Mapping[str, Any],
    bundle: SubjectFoldBundle,
    device: torch.device,
) -> None:
    for child in ("checkpoints", "logs", "predictions"):
        (stage_dir / child).mkdir(parents=True, exist_ok=True)
    save_config(config, stage_dir / "config.yaml")
    write_json(stage_dir / "split.json", bundle.split_summary)
    _write_normalization(stage_dir / "normalization.json", bundle)
    write_json(
        stage_dir / "provenance.json",
        runtime_identity(config, mean=bundle.mean, std=bundle.std),
    )
    write_json(
        stage_dir / "model.json",
        {**count_parameters(build_model(config)), "device": str(device)},
    )


def _resume_training_state(
    stage_dir: Path,
    config: Mapping[str, Any],
    bundle: SubjectFoldBundle,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    device: torch.device,
    resume: bool,
) -> tuple[int, float, int, int]:
    early = config["training"]["early_stopping"]
    mode = str(early.get("mode", "max"))
    best_metric = float("-inf") if mode == "max" else float("inf")
    if not resume:
        return 0, best_metric, 0, 0
    last_path = stage_dir / "checkpoints" / "last.pt"
    log_path = stage_dir / "logs" / "epochs.jsonl"
    if not last_path.is_file():
        return 0, best_metric, 0, 0
    checkpoint = load_checkpoint(
        last_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        map_location=device,
    )
    validate_checkpoint_compatibility(
        checkpoint,
        config,
        checkpoint_path=last_path,
        runtime_mean=bundle.mean,
        runtime_std=bundle.std,
    )
    records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    best_epoch = 0
    patience = 0
    for record in records:
        if record.get("improved"):
            best_epoch = int(record["epoch"]) + 1
            patience = 0
        else:
            patience += 1
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint["best_metric"]),
        best_epoch,
        patience,
    )


def _run_inner_fold(
    base_config: Mapping[str, Any],
    outer: Mapping[str, Any],
    inner: Mapping[str, Any],
    stage_dir: Path,
    device: torch.device,
    *,
    resume: bool,
    smoke: bool,
) -> dict[str, Any]:
    status_path = stage_dir / "stage_status.json"
    if resume and status_path.is_file():
        status = _read_json(status_path)
        if status.get("status") == "complete":
            return status["summary"]

    outer_index = int(outer["outer_fold"])
    inner_index = int(inner["inner_fold"])
    fold_id = f"outer_{outer_index}/inner_{inner_index}"
    config = _runtime_config(
        base_config,
        phase="inner_training",
        fold_id=fold_id,
        train_subjects=inner["train_subjects"],
        validation_subjects=inner["validation_subjects"],
        test_subjects=outer["test_subjects"],
        extra={"smoke_execution_limiter": bool(smoke)},
    )
    bundle = build_subject_fold_datasets(
        config,
        train_subject_ids=inner["train_subjects"],
        validation_subject_ids=inner["validation_subjects"],
        test_subject_ids=outer["test_subjects"],
        fold_id=fold_id,
    )
    config = _resolve_train_only_class_weights(config, bundle)
    if bundle.validation is None:
        raise ValueError("Inner training requires a validation dataset")
    _prepare_stage(stage_dir, config, bundle, device)
    write_json(
        status_path,
        {
            "status": "running",
            "phase": "inner",
            "fold_id": fold_id,
            "outer_test_loader_created": False,
            "started_at_utc": _utc_now(),
        },
    )

    seed_everything(
        int(config["experiment"]["seed"]),
        bool(config["experiment"].get("deterministic", True)),
    )
    model = build_model(config).to(device)
    criterion = build_loss(config).to(device)
    use_amp = bool(config["training"].get("mixed_precision", False)) and (
        device.type == "cuda"
    )
    train_loader = _loader(bundle.train, config, train=True, seed_offset=0)
    prototype_loader = _loader(bundle.train, config, train=False, seed_offset=23)
    validation_loader = _loader(
        bundle.validation, config, train=False, seed_offset=1
    )
    assert (
        train_loader is not None
        and prototype_loader is not None
        and validation_loader is not None
    )

    pretraining_summary = run_masked_reconstruction_pretraining(
        model,
        train_loader,
        config,
        device,
        stage_dir,
        smoke=smoke,
        resume=resume,
    )
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best_metric, best_epoch, patience_count = _resume_training_state(
        stage_dir,
        config,
        bundle,
        model,
        optimizer,
        scheduler,
        scaler,
        device,
        resume,
    )
    early = config["training"]["early_stopping"]
    metric_name = str(early["metric"])
    mode = str(early.get("mode", "max"))
    max_epochs = 1 if smoke else int(config["training"]["epochs"])
    max_batches = 1 if smoke else int(config["training"].get("overfit_batches", 0))
    for epoch in range(start_epoch, max_epochs):
        prototype_summary = None
        if bool(getattr(criterion, "requires_activity_prototypes", False)):
            prototypes, prototype_summary = compute_activity_disease_prototypes(
                model, prototype_loader, device
            )
            criterion.set_activity_disease_prototypes(prototypes)
        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler=scaler,
            mixed_precision=use_amp,
            gradient_clip_norm=config["training"].get("gradient_clip_norm"),
            max_batches=max_batches,
            zero_division=float(config["evaluation"].get("zero_division", 0.0)),
        )
        validation_metrics, _ = evaluate_epoch(
            model,
            validation_loader,
            criterion,
            device,
            mixed_precision=use_amp,
            max_batches=max_batches,
            zero_division=float(config["evaluation"].get("zero_division", 0.0)),
        )
        validation_metrics = _named_metrics(validation_metrics)
        if scheduler is not None:
            scheduler.step()
        current = float(validation_metrics[metric_name])
        improved = _metric_improved(
            current,
            best_metric,
            mode,
            float(early.get("minimum_delta", 0.0)),
        )
        if improved:
            best_metric = current
            best_epoch = epoch + 1
            patience_count = 0
            save_checkpoint(
                stage_dir / "checkpoints" / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_metric,
                config,
                bundle.class_names,
                bundle.mean,
                bundle.std,
            )
        else:
            patience_count += 1
        save_checkpoint(
            stage_dir / "checkpoints" / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_metric,
            config,
            bundle.class_names,
            bundle.mean,
            bundle.std,
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "selection_metric": metric_name,
            "selection_value": current,
            "improved": improved,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "patience_count": patience_count,
            "gpu_peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
            "timestamp": time.time(),
        }
        if prototype_summary is not None:
            record["train_fold_prototypes"] = prototype_summary
        append_jsonl(stage_dir / "logs" / "epochs.jsonl", record)
        write_json(stage_dir / "metrics.json", record)
        print(
            f"[inner {outer_index}.{inner_index}] epoch={epoch + 1}/{max_epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_bal_acc={validation_metrics['balanced_accuracy']:.4f} "
            f"val_macro_f1={validation_metrics['macro_f1']:.4f}",
            flush=True,
        )
        if (
            not smoke
            and bool(early.get("enabled", True))
            and patience_count >= int(early.get("patience", 12))
        ):
            break

    best_path = stage_dir / "checkpoints" / "best.pt"
    checkpoint = load_checkpoint(best_path, model=model, map_location=device)
    validate_checkpoint_compatibility(
        checkpoint,
        config,
        checkpoint_path=best_path,
        runtime_mean=bundle.mean,
        runtime_std=bundle.std,
    )
    validation_metrics, predictions = evaluate_epoch(
        model,
        validation_loader,
        criterion,
        device,
        mixed_precision=use_amp,
        max_batches=max_batches,
        zero_division=float(config["evaluation"].get("zero_division", 0.0)),
    )
    validation_metrics = _named_metrics(validation_metrics)
    write_json(stage_dir / "validation_metrics.json", validation_metrics)
    _write_predictions(
        stage_dir / "predictions" / "validation.csv",
        predictions,
        bundle.class_names,
    )
    summary = {
        "fold_id": fold_id,
        "best_epoch": int(checkpoint["epoch"]) + 1,
        "best_metric": float(checkpoint["best_metric"]),
        "selection_metric": metric_name,
        "validation_metrics": validation_metrics,
        "train_subject_ids_sha256": bundle.normalization["subject_ids_sha256"],
        "normalization_sha256": bundle.normalization["normalization_sha256"],
        "checkpoint_sha256": sha256_file(best_path),
        "pretraining": pretraining_summary,
        "smoke": bool(smoke),
    }
    write_json(
        status_path,
        {
            "status": "complete",
            "phase": "inner",
            "fold_id": fold_id,
            "outer_test_loader_created": False,
            "completed_at_utc": _utc_now(),
            "summary": summary,
        },
    )
    return summary


def _pool_inner_oof(
    outer: Mapping[str, Any],
    outer_dir: Path,
) -> tuple[list[dict[str, Any]], Path]:
    rows: list[dict[str, Any]] = []
    for inner in outer["inner_folds"]:
        path = (
            outer_dir
            / f"inner_{int(inner['inner_fold'])}"
            / "predictions"
            / "validation.csv"
        )
        rows.extend(_read_prediction_rows(path))
    subject_ids = [row["subject_id"] for row in rows]
    expected = set(str(value) for value in outer["train_subjects"])
    outer_test = set(str(value) for value in outer["test_subjects"])
    if len(subject_ids) != len(set(subject_ids)):
        raise ValueError("Inner OOF predictions contain duplicate subjects")
    if set(subject_ids) != expected:
        raise ValueError(
            "Inner OOF predictions do not exactly cover outer-train subjects"
        )
    if set(subject_ids) & outer_test:
        raise ValueError("Outer-test subject appeared in inner OOF predictions")
    output = outer_dir / "selection" / "inner_oof_predictions.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
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
        writer.writerows(rows)
    return rows, output


def _inner_metric_summary(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in (*METRIC_NAMES, "pd_recall", "dd_recall"):
        values = [
            float(summary["validation_metrics"][name])
            for summary in summaries
            if summary["validation_metrics"].get(name) is not None
        ]
        result[name] = {
            "values": values,
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=0)) if values else None,
        }
    return result


def _run_outer_final(
    base_config: Mapping[str, Any],
    outer: Mapping[str, Any],
    outer_dir: Path,
    device: torch.device,
    *,
    threshold: Mapping[str, Any],
    epoch_selection: Mapping[str, Any],
    oof_path: Path,
    resume: bool,
) -> dict[str, Any]:
    final_dir = outer_dir / "outer_final"
    status_path = final_dir / "stage_status.json"
    if resume and status_path.is_file():
        status = _read_json(status_path)
        if status.get("status") == "complete":
            return status["summary"]

    outer_index = int(outer["outer_fold"])
    fold_id = f"outer_{outer_index}/outer_final"
    fixed_epochs = int(epoch_selection["selected_epoch_count"])
    config = _runtime_config(
        base_config,
        phase="outer_final_training",
        fold_id=fold_id,
        train_subjects=outer["train_subjects"],
        validation_subjects=[],
        test_subjects=outer["test_subjects"],
        extra={
            "fixed_epoch_count": fixed_epochs,
            "early_stopping_on_outer_test": False,
            "threshold": dict(threshold),
            "inner_oof_predictions_sha256": sha256_file(oof_path),
        },
    )
    bundle = build_subject_fold_datasets(
        config,
        train_subject_ids=outer["train_subjects"],
        validation_subject_ids=[],
        test_subject_ids=outer["test_subjects"],
        fold_id=fold_id,
    )
    config = _resolve_train_only_class_weights(config, bundle)
    if bundle.test is None:
        raise ValueError("Outer-final training requires a held-out test dataset")
    _prepare_stage(final_dir, config, bundle, device)
    write_json(
        status_path,
        {
            "status": "running",
            "phase": "outer_final",
            "fold_id": fold_id,
            "fixed_epoch_count": fixed_epochs,
            "outer_test_loader_created": False,
            "outer_test_evaluated": False,
            "started_at_utc": _utc_now(),
        },
    )

    seed_everything(
        int(config["experiment"]["seed"]),
        bool(config["experiment"].get("deterministic", True)),
    )
    model = build_model(config).to(device)
    criterion = build_loss(config).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    use_amp = bool(config["training"].get("mixed_precision", False)) and (
        device.type == "cuda"
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    train_loader = _loader(bundle.train, config, train=True, seed_offset=0)
    assert train_loader is not None
    start_epoch = 0
    last_path = final_dir / "checkpoints" / "last.pt"
    if resume and last_path.is_file():
        checkpoint = load_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        validate_checkpoint_compatibility(
            checkpoint,
            config,
            checkpoint_path=last_path,
            runtime_mean=bundle.mean,
            runtime_std=bundle.std,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
    for epoch in range(start_epoch, fixed_epochs):
        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler=scaler,
            mixed_precision=use_amp,
            gradient_clip_norm=config["training"].get("gradient_clip_norm"),
            max_batches=int(config["training"].get("overfit_batches", 0)),
            zero_division=float(config["evaluation"].get("zero_division", 0.0)),
        )
        if scheduler is not None:
            scheduler.step()
        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            float(threshold["balanced_accuracy"]),
            config,
            bundle.class_names,
            bundle.mean,
            bundle.std,
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "fixed_epoch_count": fixed_epochs,
            "validation_loader_created": False,
            "outer_test_loader_created": False,
            "timestamp": time.time(),
        }
        append_jsonl(final_dir / "logs" / "epochs.jsonl", record)
        write_json(final_dir / "metrics.json", record)
        print(
            f"[outer-final {outer_index}] epoch={epoch + 1}/{fixed_epochs} "
            f"train_loss={train_metrics['loss']:.4f}",
            flush=True,
        )

    final_path = final_dir / "checkpoints" / "final.pt"
    save_checkpoint(
        final_path,
        model,
        optimizer,
        scheduler,
        scaler,
        fixed_epochs - 1,
        float(threshold["balanced_accuracy"]),
        config,
        bundle.class_names,
        bundle.mean,
        bundle.std,
    )
    checkpoint = load_checkpoint(final_path, model=model, map_location=device)
    validate_checkpoint_compatibility(
        checkpoint,
        config,
        checkpoint_path=final_path,
        runtime_mean=bundle.mean,
        runtime_std=bundle.std,
    )

    # The held-out loader is deliberately created only after final training is fixed.
    test_loader = _loader(bundle.test, config, train=False, seed_offset=2)
    assert test_loader is not None
    write_json(
        status_path,
        {
            "status": "running",
            "phase": "outer_final",
            "fold_id": fold_id,
            "fixed_epoch_count": fixed_epochs,
            "outer_test_loader_created": True,
            "outer_test_evaluated": False,
            "updated_at_utc": _utc_now(),
        },
    )
    default_metrics, predictions = evaluate_epoch(
        model,
        test_loader,
        criterion,
        device,
        mixed_precision=use_amp,
        zero_division=float(config["evaluation"].get("zero_division", 0.0)),
    )
    default_metrics = _named_metrics(default_metrics)
    thresholded = _threshold_metrics(
        predictions,
        float(threshold["threshold"]),
        float(config["evaluation"].get("zero_division", 0.0)),
    )
    result_dir = final_dir / "outer_test"
    write_json(result_dir / "metrics_default_argmax.json", default_metrics)
    write_json(result_dir / "metrics_thresholded.json", thresholded)
    _write_predictions(
        result_dir / "predictions.csv",
        predictions,
        bundle.class_names,
        threshold=float(threshold["threshold"]),
    )
    summary = {
        "fold_id": fold_id,
        "fixed_epoch_count": fixed_epochs,
        "threshold": float(threshold["threshold"]),
        "outer_test_subject_count": len(bundle.test),
        "metrics_thresholded": thresholded,
        "metrics_default_argmax": default_metrics,
        "normalization_sha256": bundle.normalization["normalization_sha256"],
        "checkpoint_sha256": sha256_file(final_path),
        "outer_test_evaluation_count": 1,
    }
    write_json(
        status_path,
        {
            "status": "complete",
            "phase": "outer_final",
            "fold_id": fold_id,
            "fixed_epoch_count": fixed_epochs,
            "outer_test_loader_created": True,
            "outer_test_evaluated": True,
            "outer_test_evaluation_count": 1,
            "completed_at_utc": _utc_now(),
            "summary": summary,
        },
    )
    return summary


def _aggregate_outer_results(
    run_dir: Path,
    outer_summaries: Sequence[Mapping[str, Any]],
    zero_division: float,
    model_name: str,
) -> dict[str, Any]:
    fold_metrics: dict[str, Any] = {}
    for name in (*METRIC_NAMES, "pd_recall", "dd_recall"):
        values = [
            float(summary["metrics_thresholded"][name])
            for summary in outer_summaries
            if summary["metrics_thresholded"].get(name) is not None
        ]
        fold_metrics[name] = {
            "values": values,
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=0)) if values else None,
        }
    all_rows: list[dict[str, Any]] = []
    for outer_index in range(5):
        all_rows.extend(
            _read_prediction_rows(
                run_dir
                / f"outer_{outer_index}"
                / "outer_final"
                / "outer_test"
                / "predictions.csv"
            )
        )
    subjects = [row["subject_id"] for row in all_rows]
    if len(subjects) != 390 or len(set(subjects)) != 390:
        raise ValueError("Outer-test predictions must contain 390 unique subjects")
    predictions_path = run_dir / "outer_test_predictions_all_folds.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("subject_id", "target", "probability_pd", "probability_dd"),
        )
        writer.writeheader()
        writer.writerows(all_rows)
    # Fold-specific thresholds differ, so pooled predictions are reconstructed per fold.
    pooled_targets: list[int] = []
    pooled_predictions: list[int] = []
    pooled_probabilities: list[list[float]] = []
    for outer_index, summary in enumerate(outer_summaries):
        rows = _read_prediction_rows(
            run_dir
            / f"outer_{outer_index}"
            / "outer_final"
            / "outer_test"
            / "predictions.csv"
        )
        threshold = float(summary["threshold"])
        for row in rows:
            pooled_targets.append(int(row["target"]))
            pooled_predictions.append(int(float(row["probability_dd"]) >= threshold))
            pooled_probabilities.append(
                [float(row["probability_pd"]), float(row["probability_dd"])]
            )
    pooled = classification_metrics(
        np.asarray(pooled_targets, dtype=np.int64),
        np.asarray(pooled_predictions, dtype=np.int64),
        num_classes=2,
        zero_division=zero_division,
        probabilities=np.asarray(pooled_probabilities, dtype=np.float64),
    )
    return {
        "protocol": "5x3_nested_subject_cv",
        "baseline": model_name,
        "outer_fold_count": 5,
        "subject_count": 390,
        "primary_metric": "balanced_accuracy",
        "fold_metric_summary": fold_metrics,
        "pooled_outer_test_metrics": _named_metrics(pooled),
        "outer_folds": list(outer_summaries),
        "outer_test_predictions_sha256": sha256_file(predictions_path),
    }


def _freeze_baseline(
    run_dir: Path,
    config: Mapping[str, Any],
    split_sha256: str,
    summary: Mapping[str, Any],
    model_name: str,
) -> dict[str, Any]:
    manifest_path = run_dir / "artifact_files.sha256"
    frozen_path = run_dir / "frozen_baseline_manifest.json"
    frozen_hash_path = run_dir / "frozen_baseline_manifest.json.sha256"
    excluded = {manifest_path.resolve(), frozen_path.resolve(), frozen_hash_path.resolve()}
    files = [
        path
        for path in run_dir.rglob("*")
        if path.is_file() and path.resolve() not in excluded
    ]
    entries = [
        {
            "relative_path": path.relative_to(run_dir).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in sorted(files, key=lambda value: value.relative_to(run_dir).as_posix())
    ]
    text = "".join(
        f"{entry['sha256']}  {entry['relative_path']}\n" for entry in entries
    )
    manifest_path.write_text(text, encoding="utf-8")
    artifact_tree_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    _, source_hash = source_tree_manifest(config["_meta"]["project_root"])
    frozen = {
        "schema_version": 1,
        "status": "frozen_complete",
        "baseline": model_name,
        "protocol": "5_outer_x_3_inner_nested_subject_cv",
        "created_at_utc": _utc_now(),
        "run_directory": str(run_dir),
        "config_sha256": semantic_config_sha256(config),
        "split_sha256": split_sha256,
        "source_tree_sha256": source_hash,
        "artifact_file_count": len(entries),
        "artifact_tree_sha256": artifact_tree_sha256,
        "primary_metric": "balanced_accuracy",
        "summary": dict(summary),
    }
    write_json(frozen_path, frozen)
    frozen_hash_path.write_text(
        f"{sha256_file(frozen_path)}  {frozen_path.name}\n",
        encoding="utf-8",
    )
    return frozen


def _initialize_run(
    config: Mapping[str, Any],
    output_dir: str | Path | None,
    *,
    resume: bool,
    split_path: Path,
    split_sha256: str,
    device: torch.device,
    model_name: str,
) -> Path:
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = (
            Path(str(config["experiment"]["output_root"]))
            / f"formal_{model_name}-{stamp}"
        ).resolve()
    else:
        run_dir = Path(output_dir).expanduser().resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and not resume:
        raise FileExistsError(f"Refusing to overwrite non-empty run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = run_dir / "protocol.json"
    _, current_source_hash = source_tree_manifest(
        config["_meta"]["project_root"]
    )
    if resume:
        protocol = _read_json(protocol_path)
        if protocol["base_config_sha256"] != semantic_config_sha256(config):
            raise ValueError("Resume config does not match existing formal run")
        if protocol["split_sha256"] != split_sha256:
            raise ValueError("Resume split does not match existing formal run")
        if protocol["source_tree_sha256"] != current_source_hash:
            raise ValueError("Resume source tree does not match existing formal run")
        return run_dir
    save_config(config, run_dir / "config.yaml")
    write_environment(
        collect_environment(config["_meta"]["project_root"]),
        run_dir / "environment.json",
    )
    (run_dir / "frozen_split.json").write_bytes(split_path.read_bytes())
    write_json(
        protocol_path,
        {
            "schema_version": 1,
            "baseline": model_name,
            "protocol": "5_outer_x_3_inner_nested_subject_cv",
            "base_config_sha256": semantic_config_sha256(config),
            "split_sha256": split_sha256,
            "source_tree_sha256": current_source_hash,
            "primary_metric": config["model_selection"]["primary_metric"],
            "threshold": config["model_selection"]["threshold"],
            "final_epoch": config["model_selection"]["final_epoch"],
            "model_frozen_before_outer_test": True,
            "outer_test_policy": "evaluate_each_subject_once_after_inner_selection",
            "device": str(device),
            "started_at_utc": _utc_now(),
        },
    )
    return run_dir


def run_nested_training(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path | None,
    device_name: str,
    resume: bool,
    smoke: bool,
    smoke_outer_fold: int = 0,
    smoke_inner_fold: int = 0,
) -> dict[str, Any]:
    payload, audit, split_path, split_sha256 = _load_frozen_split(config)
    if audit.get("status") != "pass":
        raise ValueError("Nested split audit must pass before training")
    device = select_device(device_name)
    model_name = str(config["experiment"]["name"])
    if device.type != "cuda" and not smoke:
        raise RuntimeError("Formal V3 nested CV requires a CUDA device")
    run_dir = _initialize_run(
        config,
        output_dir,
        resume=resume,
        split_path=split_path,
        split_sha256=split_sha256,
        device=device,
        model_name=model_name,
    )
    if smoke:
        outer = payload["outer"][smoke_outer_fold]
        inner = outer["inner_folds"][smoke_inner_fold]
        summary = _run_inner_fold(
            config,
            outer,
            inner,
            run_dir / f"outer_{smoke_outer_fold}" / f"inner_{smoke_inner_fold}",
            device,
            resume=resume,
            smoke=True,
        )
        result = {
            "status": "smoke_complete",
            "run_directory": str(run_dir),
            "outer_test_loader_created": False,
            "outer_test_evaluated": False,
            "summary": summary,
        }
        write_json(run_dir / "smoke_summary.json", result)
        return result

    outer_summaries: list[dict[str, Any]] = []
    for outer in payload["outer"]:
        outer_index = int(outer["outer_fold"])
        outer_dir = run_dir / f"outer_{outer_index}"
        inner_summaries = [
            _run_inner_fold(
                config,
                outer,
                inner,
                outer_dir / f"inner_{int(inner['inner_fold'])}",
                device,
                resume=resume,
                smoke=False,
            )
            for inner in outer["inner_folds"]
        ]
        oof_rows, oof_path = _pool_inner_oof(outer, outer_dir)
        threshold = select_binary_threshold(
            [int(row["target"]) for row in oof_rows],
            [float(row["probability_dd"]) for row in oof_rows],
            metric=str(config["model_selection"]["threshold"]["metric"]),
            zero_division=float(config["evaluation"].get("zero_division", 0.0)),
        )
        epoch_selection = final_epoch_from_inner_best(
            [int(summary["best_epoch"]) for summary in inner_summaries],
            strategy=str(config["model_selection"]["final_epoch"]["strategy"]),
        )
        selection = {
            "outer_fold": outer_index,
            "candidate_count": 1,
            "selected_candidate": model_name,
            "primary_metric": config["model_selection"]["primary_metric"],
            "inner_metric_summary": _inner_metric_summary(inner_summaries),
            "threshold": threshold,
            "final_epoch": epoch_selection,
            "inner_oof_subject_count": len(oof_rows),
            "inner_oof_predictions_sha256": sha256_file(oof_path),
            "outer_test_used_for_selection": False,
        }
        write_json(outer_dir / "selection" / "selection.json", selection)
        outer_summary = _run_outer_final(
            config,
            outer,
            outer_dir,
            device,
            threshold=threshold,
            epoch_selection=epoch_selection,
            oof_path=oof_path,
            resume=resume,
        )
        outer_summary["selection"] = selection
        outer_summaries.append(outer_summary)

    aggregate = _aggregate_outer_results(
        run_dir,
        outer_summaries,
        float(config["evaluation"].get("zero_division", 0.0)),
        model_name,
    )
    write_json(run_dir / "nested_cv_summary.json", aggregate)
    frozen = _freeze_baseline(
        run_dir, config, split_sha256, aggregate, model_name
    )
    write_json(
        run_dir / "run_status.json",
        {
            "status": "complete",
            "baseline": model_name,
            "completed_at_utc": _utc_now(),
            "frozen_manifest_sha256": sha256_file(
                run_dir / "frozen_baseline_manifest.json"
            ),
        },
    )
    return {
        "status": "complete",
        "run_directory": str(run_dir),
        "summary": aggregate,
        "frozen": frozen,
    }
