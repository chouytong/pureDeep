from __future__ import annotations

import contextlib
import csv
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.folds import build_subject_fold_datasets
from src.datasets.subject_activity import collate_subject_activities
from src.engine.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_compatibility,
)
from src.engine.nested_training import _resume_training_state
from src.engine.optim import build_optimizer, build_scheduler
from src.losses import build_loss
from src.metrics.classification import classification_metrics
from src.targeted_ablation.data import (
    HandcraftedCache,
    StatisticalFeatureDataset,
    fit_fold_statistical_transform,
)
from src.targeted_ablation.models import build_targeted_model
from src.utils.config import save_config
from src.utils.model_stats import count_parameters
from src.utils.provenance import sha256_file, sha256_json
from src.utils.seed import seed_everything


METRICS = (
    "balanced_accuracy",
    "macro_f1",
    "accuracy",
    "macro_auroc",
    "negative_log_likelihood",
    "brier_score",
    "pd_recall",
    "dd_recall",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: str | Path, value: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n")


def named_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    recalls = list(result.get("per_class_recall", []))
    if len(recalls) == 2:
        result["pd_recall"] = float(recalls[0])
        result["dd_recall"] = float(recalls[1])
    return result


def modifications_for_variant(variant: str) -> tuple[str, ...]:
    parts = tuple(value for value in str(variant).split("+") if value)
    if not parts or not set(parts) <= {"R1", "R2", "A1", "A2", "N1"}:
        raise ValueError(f"Invalid targeted variant: {variant}")
    if len(parts) != len(set(parts)) or len(parts) > 2:
        raise ValueError(f"Invalid targeted combination: {variant}")
    if len(set(parts) & {"R1", "R2"}) > 1:
        raise ValueError("R1 and R2 cannot be combined")
    return parts


def runtime_config(
    base: Mapping[str, Any],
    variant: str,
    outer_index: int,
    inner_index: int,
    train_subjects: Sequence[str],
    validation_subjects: Sequence[str],
    test_subjects: Sequence[str],
    plan_sha256: str,
) -> dict[str, Any]:
    config = deepcopy(dict(base))
    config["experiment"]["name"] = f"v3_targeted_ablation_{variant.replace('+', '_')}"
    config["targeted_ablation"] = {
        "variant": variant,
        "modifications": list(modifications_for_variant(variant)),
        "plan_sha256": plan_sha256,
        "outer_test_loader_created": False,
        "outer_test_signal_accessed": False,
        "outer_test_predictions_accessed": False,
    }
    config["nested_runtime"] = {
        "phase": "targeted_development_inner_training",
        "fold_id": f"outer_{outer_index}/inner_{inner_index}",
        "train_subject_ids_sha256": sha256_json(sorted(train_subjects)),
        "validation_subject_ids_sha256": sha256_json(sorted(validation_subjects)),
        "outer_test_subject_ids_sha256": sha256_json(sorted(test_subjects)),
        "train_subject_count": len(train_subjects),
        "validation_subject_count": len(validation_subjects),
        "outer_test_subject_count": len(test_subjects),
        "outer_test_ids_known_from_frozen_split_only": True,
    }
    return config


def subject_loader(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory) and torch.cuda.is_available(),
        persistent_workers=int(num_workers) > 0,
        generator=generator,
        collate_fn=collate_subject_activities,
    )


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True)


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    result = {
        "x": batch["x"].to(device, non_blocking=True),
        "y": batch["y"].to(device, non_blocking=True),
        "wrist_mask": batch["wrist_mask"].to(device, non_blocking=True),
        "activity_mask": batch["activity_mask"].to(device, non_blocking=True),
        "activity_lengths": batch["activity_lengths"].to(device, non_blocking=True),
    }
    if "statistical_features" in batch:
        result["statistical_features"] = batch["statistical_features"].to(
            device, non_blocking=True
        )
    else:
        result["statistical_features"] = None
    return result


def _forward(model: torch.nn.Module, tensors: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    return model(
        tensors["x"],
        tensors["wrist_mask"],
        tensors["activity_mask"],
        tensors["activity_lengths"],
        tensors["statistical_features"],
    )


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Any,
    *,
    mixed_precision: bool,
    gradient_clip_norm: float,
    max_batches: int = 0,
) -> dict[str, Any]:
    model.train()
    started = time.perf_counter()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    loss_sum = 0.0
    samples = 0
    batches = 0
    maximum_gradient_norm = 0.0
    nonfinite_gradient_batches = 0
    nonfinite_loss_batches = 0
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        tensors = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, mixed_precision):
            outputs = _forward(model, tensors)
            losses = criterion(outputs, tensors["y"])
        loss = losses["loss"]
        if not bool(torch.isfinite(loss)):
            nonfinite_loss_batches += 1
            raise FloatingPointError("Non-finite training loss")
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        gradient_tensors = [
            parameter.grad.detach().norm(2)
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        gradient_norm = torch.stack(gradient_tensors).norm(2)
        gradient_value = float(gradient_norm.detach().cpu())
        if not np.isfinite(gradient_value):
            nonfinite_gradient_batches += 1
            if scaler is None or not scaler.is_enabled():
                raise FloatingPointError("Non-finite FP32 gradient norm")
        else:
            maximum_gradient_norm = max(maximum_gradient_norm, gradient_value)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float(gradient_clip_norm)
            )
        if scaler is not None and scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        count = int(tensors["y"].shape[0])
        loss_sum += float(loss.detach().cpu()) * count
        samples += count
        batches += 1
        targets.append(tensors["y"].detach().cpu().numpy())
        predictions.append(outputs["logits"].detach().argmax(dim=-1).cpu().numpy())
    if samples == 0:
        raise RuntimeError("No training batches were processed")
    metrics = classification_metrics(
        np.concatenate(targets),
        np.concatenate(predictions),
        num_classes=2,
        zero_division=0.0,
    )
    metrics.update(
        {
            "loss": loss_sum / samples,
            "classification_loss": loss_sum / samples,
            "duration_seconds": time.perf_counter() - started,
            "batches": batches,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "maximum_gradient_norm_before_clipping": maximum_gradient_norm,
            "nonfinite_gradient_batches": nonfinite_gradient_batches,
            "nonfinite_loss_batches": nonfinite_loss_batches,
        }
    )
    return named_metrics(metrics)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    subtype_by_subject: Mapping[str, str],
    *,
    mixed_precision: bool,
    model_name: str,
    outer_index: int,
    inner_index: int,
    branch_mode: str = "both",
    max_batches: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    model.set_r1_branch_mode(branch_mode)
    started = time.perf_counter()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    loss_sum = 0.0
    samples = 0
    diagnostic_values: dict[str, list[float]] = {
        "mil_attention_entropy": [],
        "mil_instance_utilization": [],
        "r1_frequency_bag_l2": [],
        "r1_full_band_bag_l2": [],
        "r1_branch_cosine": [],
    }
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        tensors = _to_device(batch, device)
        with _autocast(device, mixed_precision):
            outputs = _forward(model, tensors)
            losses = criterion(outputs, tensors["y"])
        probability = outputs["probabilities"].detach().float().cpu()
        prediction = probability.argmax(dim=-1)
        target = tensors["y"].detach().cpu()
        count = int(target.shape[0])
        loss_sum += float(losses["loss"].detach().cpu()) * count
        samples += count
        targets.append(target.numpy())
        predictions.append(prediction.numpy())
        probabilities.append(probability.numpy())
        activity_mask = tensors["activity_mask"].detach().cpu().to(torch.float32)
        per_sample_diagnostics: dict[str, torch.Tensor] = {}
        for key in diagnostic_values:
            value = outputs[key].detach().float().cpu()
            averaged = (value * activity_mask).sum(dim=1) / activity_mask.sum(dim=1)
            per_sample_diagnostics[key] = averaged
            diagnostic_values[key].extend(float(v) for v in averaged.tolist())
        for index in range(count):
            subject = str(batch["subject_id"][index])
            rows.append(
                {
                    "model": model_name,
                    "outer_context": outer_index,
                    "inner_fold": inner_index,
                    "subject_id": subject,
                    "target": int(target[index]),
                    "diagnosis": "PD" if int(target[index]) == 0 else "DD",
                    "dd_subtype": subtype_by_subject[subject],
                    "probability_pd": float(probability[index, 0]),
                    "probability_dd": float(probability[index, 1]),
                    "prediction": int(prediction[index]),
                    "prediction_label": "PD" if int(prediction[index]) == 0 else "DD",
                    "threshold": 0.5,
                    "branch_mode": branch_mode,
                    "analysis_scope": "development_inner_cv_not_outer_test",
                    **{
                        key: float(value[index])
                        for key, value in per_sample_diagnostics.items()
                    },
                }
            )
    model.set_r1_branch_mode("both")
    if samples == 0:
        raise RuntimeError("No evaluation samples were processed")
    metrics = named_metrics(
        classification_metrics(
            np.concatenate(targets),
            np.concatenate(predictions),
            num_classes=2,
            zero_division=0.0,
            probabilities=np.concatenate(probabilities),
        )
    )
    metrics["loss"] = loss_sum / samples
    metrics["duration_seconds"] = time.perf_counter() - started
    diagnostics = {
        key: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "count": len(values),
        }
        for key, values in diagnostic_values.items()
    }
    return metrics, rows, diagnostics


def write_predictions(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write empty predictions")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_fold_summaries(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in METRICS:
        values = [float(row["validation_metrics"][metric]) for row in summaries]
        result[metric] = {
            "values": values,
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "median": float(np.median(values)),
        }
    return result


def train_fold(
    base_config: Mapping[str, Any],
    variant: str,
    outer: Mapping[str, Any],
    inner: Mapping[str, Any],
    stage_dir: Path,
    device: torch.device,
    subtype_by_subject: Mapping[str, str],
    plan_sha256: str,
    handcrafted_cache: HandcraftedCache | None,
    *,
    resume: bool,
    smoke: bool,
) -> dict[str, Any]:
    status_path = stage_dir / "stage_status.json"
    if resume and status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "complete":
            return status["summary"]
    outer_index = int(outer["outer_fold"])
    inner_index = int(inner["inner_fold"])
    modifications = modifications_for_variant(variant)
    config = runtime_config(
        base_config,
        variant,
        outer_index,
        inner_index,
        inner["train_subjects"],
        inner["validation_subjects"],
        outer["test_subjects"],
        plan_sha256,
    )
    bundle = build_subject_fold_datasets(
        config,
        train_subject_ids=inner["train_subjects"],
        validation_subject_ids=inner["validation_subjects"],
        test_subject_ids=outer["test_subjects"],
        fold_id=f"outer_{outer_index}/inner_{inner_index}",
    )
    if bundle.validation is None:
        raise ValueError("Targeted ablation requires inner validation data")
    preprocessing_metadata = None
    train_dataset: Any = bundle.train
    validation_dataset: Any = bundle.validation
    if "R2" in modifications:
        if handcrafted_cache is None:
            raise ValueError("R2 requires the frozen handcrafted feature cache")
        transform = fit_fold_statistical_transform(
            handcrafted_cache,
            inner["train_subjects"],
            inner["validation_subjects"],
            components=32,
            random_state=42,
        )
        train_dataset = StatisticalFeatureDataset(bundle.train, transform.transformed)
        validation_dataset = StatisticalFeatureDataset(
            bundle.validation, transform.transformed
        )
        preprocessing_dir = stage_dir / "statistical_preprocessing"
        preprocessing_dir.mkdir(parents=True, exist_ok=True)
        scaler_path = preprocessing_dir / "scaler.joblib"
        pca_path = preprocessing_dir / "pca.joblib"
        joblib.dump(transform.scaler, scaler_path)
        joblib.dump(transform.pca, pca_path)
        preprocessing_metadata = {
            **transform.metadata,
            "scaler_sha256": sha256_file(scaler_path),
            "pca_sha256": sha256_file(pca_path),
        }
        write_json(preprocessing_dir / "metadata.json", preprocessing_metadata)

    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "checkpoints").mkdir(exist_ok=True)
    (stage_dir / "logs").mkdir(exist_ok=True)
    (stage_dir / "predictions").mkdir(exist_ok=True)
    save_config(config, stage_dir / "config.yaml")
    write_json(stage_dir / "split.json", bundle.split_summary)
    write_json(
        stage_dir / "normalization.json",
        {**bundle.normalization, "mean": bundle.mean.tolist(), "std": bundle.std.tolist()},
    )
    seed_everything(42, True)
    model = build_targeted_model(config, modifications).to(device)
    parameter_summary = count_parameters(model)
    write_json(
        stage_dir / "model.json",
        {
            **parameter_summary,
            "variant": variant,
            "modifications": list(modifications),
            "groupnorm_replacements": model.groupnorm_replacements,
            "subject_embedding_dim": 514,
            "statistical_dim": model.statistical_dim,
            "classifier_input_dim": 514 + model.statistical_dim,
            "flops": "not_computed_reliably_for_variable_length_grouped_activities",
        },
    )
    write_json(
        status_path,
        {
            "status": "running",
            "started_at_utc": utc_now(),
            "variant": variant,
            "outer_context": outer_index,
            "inner_fold": inner_index,
            "outer_test_loader_created": False,
            "outer_test_signal_accessed": False,
        },
    )
    criterion = build_loss(config).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    use_amp = bool(config["training"].get("mixed_precision", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    train_loader = subject_loader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 0)),
        pin_memory=bool(config["data"].get("pin_memory", False)),
        seed=42,
    )
    train_eval_loader = subject_loader(
        train_dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
        pin_memory=bool(config["data"].get("pin_memory", False)),
        seed=42,
    )
    validation_loader = subject_loader(
        validation_dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
        pin_memory=bool(config["data"].get("pin_memory", False)),
        seed=43,
    )
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
    maximum_epochs = 1 if smoke else int(config["training"]["epochs"])
    maximum_batches = 1 if smoke else 0
    maximum_gradient_norm = 0.0
    total_nonfinite_gradient_batches = 0
    runtime_started = time.perf_counter()
    for epoch in range(start_epoch, maximum_epochs):
        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            mixed_precision=use_amp,
            gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
            max_batches=maximum_batches,
        )
        validation_metrics, _, _ = evaluate(
            model,
            validation_loader,
            criterion,
            device,
            subtype_by_subject,
            mixed_precision=use_amp,
            model_name=variant,
            outer_index=outer_index,
            inner_index=inner_index,
            max_batches=maximum_batches,
        )
        if scheduler is not None:
            scheduler.step()
        current = float(validation_metrics["balanced_accuracy"])
        improved = current > best_metric
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
        maximum_gradient_norm = max(
            maximum_gradient_norm,
            float(train_metrics["maximum_gradient_norm_before_clipping"]),
        )
        total_nonfinite_gradient_batches += int(
            train_metrics["nonfinite_gradient_batches"]
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "selection_metric": "balanced_accuracy",
            "selection_value": current,
            "improved": improved,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "patience_count": patience_count,
            "timestamp_utc": utc_now(),
        }
        append_jsonl(stage_dir / "logs" / "epochs.jsonl", record)
        write_json(stage_dir / "metrics.json", record)
        print(
            f"[{variant} {outer_index}.{inner_index}] epoch={epoch + 1}/{maximum_epochs} "
            f"train_ba={train_metrics['balanced_accuracy']:.4f} "
            f"val_ba={current:.4f} val_auc={validation_metrics['macro_auroc']:.4f}",
            flush=True,
        )
        if not smoke and patience_count >= int(config["training"]["early_stopping"]["patience"]):
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
    train_metrics, train_rows, train_diagnostics = evaluate(
        model,
        train_eval_loader,
        criterion,
        device,
        subtype_by_subject,
        mixed_precision=use_amp,
        model_name=variant,
        outer_index=outer_index,
        inner_index=inner_index,
    )
    validation_metrics, validation_rows, validation_diagnostics = evaluate(
        model,
        validation_loader,
        criterion,
        device,
        subtype_by_subject,
        mixed_precision=use_amp,
        model_name=variant,
        outer_index=outer_index,
        inner_index=inner_index,
    )
    write_predictions(stage_dir / "predictions" / "train.csv", train_rows)
    write_predictions(stage_dir / "predictions" / "validation.csv", validation_rows)
    branch_ablation = None
    if "R1" in modifications:
        branch_ablation = {}
        for mode in ("frequency_only", "fullband_only"):
            mode_metrics, mode_rows, _ = evaluate(
                model,
                validation_loader,
                criterion,
                device,
                subtype_by_subject,
                mixed_precision=use_amp,
                model_name=variant,
                outer_index=outer_index,
                inner_index=inner_index,
                branch_mode=mode,
            )
            branch_ablation[mode] = mode_metrics
            write_predictions(
                stage_dir / "predictions" / f"validation_{mode}.csv", mode_rows
            )
    summary = {
        "status": "complete",
        "variant": variant,
        "modifications": list(modifications),
        "outer_context": outer_index,
        "inner_fold": inner_index,
        "fold_id": f"outer_{outer_index}/inner_{inner_index}",
        "best_epoch": int(checkpoint["epoch"]) + 1,
        "best_metric": float(checkpoint["best_metric"]),
        "train_metrics_at_best": train_metrics,
        "validation_metrics": validation_metrics,
        "train_validation_balanced_accuracy_gap": float(
            train_metrics["balanced_accuracy"] - validation_metrics["balanced_accuracy"]
        ),
        "validation_diagnostics": validation_diagnostics,
        "train_diagnostics": train_diagnostics,
        "branch_ablation": branch_ablation,
        "parameter_count": parameter_summary,
        "runtime_seconds": time.perf_counter() - runtime_started,
        "maximum_gradient_norm_before_clipping": maximum_gradient_norm,
        "nonfinite_loss_or_gradient": total_nonfinite_gradient_batches > 0,
        "nonfinite_gradient_batches": total_nonfinite_gradient_batches,
        "checkpoint_sha256": sha256_file(best_path),
        "checkpoint_provenance_sha256": sha256_file(
            best_path.with_suffix(best_path.suffix + ".provenance.json")
        ),
        "normalization_sha256": bundle.normalization["normalization_sha256"],
        "statistical_preprocessing": preprocessing_metadata,
        "plan_sha256": plan_sha256,
        "outer_test_loader_created": False,
        "outer_test_signal_accessed": False,
        "outer_test_predictions_accessed": False,
        "smoke": smoke,
    }
    write_json(stage_dir / "fold_summary.json", summary)
    write_json(
        status_path,
        {
            "status": "complete",
            "completed_at_utc": utc_now(),
            "outer_test_loader_created": False,
            "outer_test_signal_accessed": False,
            "summary": summary,
        },
    )
    return summary


def run_variant(
    base_config: Mapping[str, Any],
    split_payload: Mapping[str, Any],
    variant: str,
    output_root: Path,
    device: torch.device,
    subtype_by_subject: Mapping[str, str],
    plan_sha256: str,
    handcrafted_cache: HandcraftedCache | None,
    *,
    resume: bool = True,
    smoke: bool = False,
    smoke_fold: tuple[int, int] = (0, 0),
) -> dict[str, Any]:
    variant_dir = output_root / "models" / variant.replace("+", "_")
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for outer in split_payload["outer"]:
        outer_index = int(outer["outer_fold"])
        for inner in outer["inner_folds"]:
            inner_index = int(inner["inner_fold"])
            if smoke and (outer_index, inner_index) != smoke_fold:
                continue
            stage_dir = variant_dir / f"outer_{outer_index}" / f"inner_{inner_index}"
            try:
                summaries.append(
                    train_fold(
                        base_config,
                        variant,
                        outer,
                        inner,
                        stage_dir,
                        device,
                        subtype_by_subject,
                        plan_sha256,
                        handcrafted_cache,
                        resume=resume,
                        smoke=smoke,
                    )
                )
            except Exception as error:
                failure = {
                    "variant": variant,
                    "outer_context": outer_index,
                    "inner_fold": inner_index,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "timestamp_utc": utc_now(),
                }
                failures.append(failure)
                write_json(stage_dir / "failure.json", failure)
                print(f"[FAILED] {failure}", flush=True)
                if smoke:
                    raise
    prediction_rows: list[dict[str, Any]] = []
    if summaries:
        for summary in summaries:
            stage_dir = (
                variant_dir
                / f"outer_{summary['outer_context']}"
                / f"inner_{summary['inner_fold']}"
            )
            with (stage_dir / "predictions" / "validation.csv").open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                prediction_rows.extend(csv.DictReader(stream))
        write_predictions(variant_dir / "development_predictions_all.csv", prediction_rows)
    result = {
        "status": "complete" if not failures and len(summaries) == (1 if smoke else 15) else "incomplete",
        "variant": variant,
        "modifications": list(modifications_for_variant(variant)),
        "fold_count": len(summaries),
        "failed_fold_count": len(failures),
        "failures": failures,
        "fold_metric_summary": aggregate_fold_summaries(summaries) if summaries else {},
        "fold_summaries": summaries,
        "prediction_rows": len(prediction_rows),
        "unique_validation_subjects": len({row["subject_id"] for row in prediction_rows}),
        "outer_test_accessed": False,
        "scope": "development_inner_cv_not_outer_test",
        "plan_sha256": plan_sha256,
    }
    write_json(variant_dir / "development_summary.json", result)
    return result
