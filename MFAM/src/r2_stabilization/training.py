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
from src.r2_stabilization.models import BRANCH_MODES, build_stabilized_r2
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
ACTIVATIONS = (
    "deep_embedding_raw",
    "statistical_embedding_raw",
    "deep_embedding_fusion",
    "statistical_embedding_fusion",
    "fused_embedding",
    "classifier_input",
    "logits",
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


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def named_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    recalls = list(result.get("per_class_recall", []))
    if len(recalls) == 2:
        result["pd_recall"] = float(recalls[0])
        result["dd_recall"] = float(recalls[1])
    return result


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
    config["experiment"]["name"] = f"v3_r2_stabilization_{variant}"
    config["training"]["mixed_precision"] = False
    config["r2_stabilization"] = {
        "variant": variant,
        "precision": "FP32",
        "plan_sha256": plan_sha256,
        "pca_components": 32,
        "outer_test_loader_created": False,
        "outer_test_signal_accessed": False,
        "outer_test_predictions_accessed": False,
        "outer_test_features_transformed": False,
    }
    config["nested_runtime"] = {
        "phase": "r2_stabilization_development_inner_training",
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


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "x": batch["x"].to(device, non_blocking=True),
        "y": batch["y"].to(device, non_blocking=True),
        "wrist_mask": batch["wrist_mask"].to(device, non_blocking=True),
        "activity_mask": batch["activity_mask"].to(device, non_blocking=True),
        "activity_lengths": batch["activity_lengths"].to(device, non_blocking=True),
        "statistical_features": batch["statistical_features"].to(
            device, non_blocking=True
        ),
    }


def _forward(model: torch.nn.Module, tensors: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    return model(
        tensors["x"],
        tensors["wrist_mask"],
        tensors["activity_mask"],
        tensors["activity_lengths"],
        tensors["statistical_features"],
    )


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True)


def _parameter_group(name: str) -> str:
    if name.startswith("core.base"):
        return "deep_branch"
    if name.startswith("core"):
        return "deep_branch"
    if name.startswith("r2_classifier") or name.startswith("gated_classifier"):
        return "fusion_classifier"
    if name.startswith("deep_norm") or name.startswith("statistical_norm"):
        return "fusion_normalization"
    if name.startswith("deep_projection"):
        return "deep_projection"
    if name.startswith("statistical_projection"):
        return "statistical_projection"
    if name.startswith("gate_layer"):
        return "fusion_gate"
    return "other"


def _group_tensor_norms(
    model: torch.nn.Module, *, gradient: bool
) -> tuple[dict[str, float], dict[str, bool], list[str]]:
    squared: dict[str, torch.Tensor] = {}
    finite: dict[str, bool] = {}
    nonfinite_names: list[str] = []
    for name, parameter in model.named_parameters():
        tensor = parameter.grad if gradient else parameter.detach()
        if tensor is None:
            continue
        group = _parameter_group(name)
        is_finite = bool(torch.isfinite(tensor).all())
        finite[group] = finite.get(group, True) and is_finite
        if not is_finite:
            nonfinite_names.append(name)
            continue
        value = tensor.detach().float().norm(2).square()
        squared[group] = squared.get(group, value.new_zeros(())) + value
    norms = {group: float(value.sqrt().cpu()) for group, value in squared.items()}
    for group in finite:
        norms.setdefault(group, float("nan") if not finite[group] else 0.0)
    return norms, finite, nonfinite_names


def _activation_batch(outputs: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ACTIVATIONS:
        tensor = outputs[name].detach().float()
        flattened = tensor.reshape(tensor.shape[0], -1)
        result[name] = {
            "dimension": int(flattened.shape[1]),
            "l2": [float(value) for value in flattened.norm(2, dim=1).cpu().tolist()],
            "max_abs": [
                float(value) for value in flattened.abs().amax(dim=1).cpu().tolist()
            ],
        }
    gate = outputs["gate"].detach().float().reshape(outputs["gate"].shape[0], -1)
    result["gate_values"] = [float(value) for value in gate.mean(dim=1).cpu().tolist()]
    return result


def _empty_numerics() -> dict[str, Any]:
    return {
        "loss": [],
        "global_gradient_preclip": [],
        "global_gradient_postclip": [],
        "scaler_scale_before": [],
        "scaler_scale_after": [],
        "skipped_optimizer_steps": 0,
        "nonfinite_gradient_batches": 0,
        "nonfinite_loss_batches": 0,
        "gradient_groups": {},
        "parameter_groups": {},
        "activations": {},
        "overflow_events": [],
    }


def _extend_numerics(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for name in (
        "loss",
        "global_gradient_preclip",
        "global_gradient_postclip",
        "scaler_scale_before",
        "scaler_scale_after",
    ):
        target[name].extend(source.get(name, []))
    for name in (
        "skipped_optimizer_steps",
        "nonfinite_gradient_batches",
        "nonfinite_loss_batches",
    ):
        target[name] += int(source.get(name, 0))
    target["overflow_events"].extend(source.get("overflow_events", []))
    for container in ("gradient_groups", "parameter_groups"):
        for group, values in source.get(container, {}).items():
            target[container].setdefault(group, []).extend(values)
    for name, values in source.get("activations", {}).items():
        record = target["activations"].setdefault(
            name, {"dimension": values["dimension"], "l2": [], "max_abs": []}
        )
        record["l2"].extend(values["l2"])
        record["max_abs"].extend(values["max_abs"])


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "count": int(array.size),
            "finite_count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p5": float("nan"),
            "p95": float("nan"),
            "minimum": float("nan"),
            "maximum": float("nan"),
        }
    return {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "median": float(np.median(finite)),
        "p5": float(np.percentile(finite, 5)),
        "p95": float(np.percentile(finite, 95)),
        "minimum": float(finite.min()),
        "maximum": float(finite.max()),
    }


def summarize_numerics(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "loss": _distribution(raw["loss"]),
        "global_gradient_preclip": _distribution(raw["global_gradient_preclip"]),
        "global_gradient_postclip": _distribution(raw["global_gradient_postclip"]),
        "scaler_scale_before": _distribution(raw["scaler_scale_before"]),
        "scaler_scale_after": _distribution(raw["scaler_scale_after"]),
        "skipped_optimizer_steps": int(raw["skipped_optimizer_steps"]),
        "nonfinite_gradient_batches": int(raw["nonfinite_gradient_batches"]),
        "nonfinite_loss_batches": int(raw["nonfinite_loss_batches"]),
        "gradient_groups": {
            group: _distribution(values)
            for group, values in sorted(raw["gradient_groups"].items())
        },
        "parameter_groups": {
            group: _distribution(values)
            for group, values in sorted(raw["parameter_groups"].items())
        },
        "activations": {
            name: {
                "dimension": int(values["dimension"]),
                "l2": _distribution(values["l2"]),
                "maximum_absolute": _distribution(values["max_abs"])["maximum"],
            }
            for name, values in sorted(raw["activations"].items())
        },
        "overflow_events": list(raw["overflow_events"]),
    }


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    *,
    mixed_precision: bool,
    gradient_clip_norm: float,
    epoch: int,
    max_batches: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.train()
    started = time.perf_counter()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    raw = _empty_numerics()
    loss_sum = 0.0
    samples = 0
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        tensors = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        scale_before = float(scaler.get_scale()) if scaler.is_enabled() else 1.0
        with _autocast(device, mixed_precision):
            outputs = _forward(model, tensors)
            losses = criterion(outputs, tensors["y"])
        loss = losses["loss"]
        raw["loss"].append(float(loss.detach().float().cpu()))
        if not bool(torch.isfinite(loss)):
            raw["nonfinite_loss_batches"] += 1
            raise FloatingPointError("Non-finite training loss")
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        scaled_norms, scaled_finite, scaled_nonfinite = _group_tensor_norms(
            model, gradient=True
        )
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        unscaled_norms, unscaled_finite, unscaled_nonfinite = _group_tensor_norms(
            model, gradient=True
        )
        all_nonfinite = sorted(set(scaled_nonfinite) | set(unscaled_nonfinite))
        finite_gradient = not all_nonfinite and all(unscaled_finite.values())
        if finite_gradient:
            squares = [value * value for value in unscaled_norms.values()]
            global_pre = float(np.sqrt(np.sum(squares)))
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float(gradient_clip_norm)
            )
            post_norms, _, _ = _group_tensor_norms(model, gradient=True)
            global_post = float(
                np.sqrt(np.sum([value * value for value in post_norms.values()]))
            )
        else:
            global_pre = float("nan")
            global_post = float("nan")
            raw["nonfinite_gradient_batches"] += 1
            if not scaler.is_enabled():
                raise FloatingPointError("Non-finite FP32 gradient")
        raw["global_gradient_preclip"].append(global_pre)
        raw["global_gradient_postclip"].append(global_post)
        for group, value in unscaled_norms.items():
            raw["gradient_groups"].setdefault(group, []).append(value)
        parameter_norms, _, _ = _group_tensor_norms(model, gradient=False)
        for group, value in parameter_norms.items():
            raw["parameter_groups"].setdefault(group, []).append(value)
        batch_activations = _activation_batch(outputs)
        for name in ACTIVATIONS:
            record = raw["activations"].setdefault(
                name,
                {
                    "dimension": batch_activations[name]["dimension"],
                    "l2": [],
                    "max_abs": [],
                },
            )
            record["l2"].extend(batch_activations[name]["l2"])
            record["max_abs"].extend(batch_activations[name]["max_abs"])

        if all_nonfinite and len(raw["overflow_events"]) < 32:
            raw["overflow_events"].append(
                {
                    "epoch": int(epoch),
                    "batch": int(batch_index),
                    "loss": float(loss.detach().float().cpu()),
                    "scale_before": scale_before,
                    "nonfinite_parameter_count": len(all_nonfinite),
                    "first_nonfinite_parameters": all_nonfinite[:12],
                    "nonfinite_groups": sorted(
                        {
                            _parameter_group(name) for name in all_nonfinite
                        }
                    ),
                    "scaled_gradient_group_finite": scaled_finite,
                }
            )

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            skipped = (not finite_gradient) or scale_after < scale_before
        else:
            optimizer.step()
            scale_after = 1.0
            skipped = False
        raw["scaler_scale_before"].append(scale_before)
        raw["scaler_scale_after"].append(scale_after)
        raw["skipped_optimizer_steps"] += int(skipped)

        count = int(tensors["y"].shape[0])
        loss_sum += float(loss.detach().cpu()) * count
        samples += count
        targets.append(tensors["y"].detach().cpu().numpy())
        predictions.append(outputs["logits"].detach().argmax(dim=-1).cpu().numpy())
    if samples == 0:
        raise RuntimeError("No training batches were processed")
    metrics = named_metrics(
        classification_metrics(
            np.concatenate(targets),
            np.concatenate(predictions),
            num_classes=2,
            zero_division=0.0,
        )
    )
    metrics.update(
        {
            "loss": loss_sum / samples,
            "duration_seconds": time.perf_counter() - started,
            "batches": len(targets),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "maximum_gradient_norm_before_clipping": _distribution(
                raw["global_gradient_preclip"]
            )["maximum"],
            "nonfinite_gradient_batches": raw["nonfinite_gradient_batches"],
            "nonfinite_loss_batches": raw["nonfinite_loss_batches"],
            "skipped_optimizer_steps": raw["skipped_optimizer_steps"],
        }
    )
    return metrics, raw


def _activation_summary(outputs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    raw = _empty_numerics()
    for batch in outputs:
        for name in ACTIVATIONS:
            record = raw["activations"].setdefault(
                name,
                {"dimension": batch[name]["dimension"], "l2": [], "max_abs": []},
            )
            record["l2"].extend(batch[name]["l2"])
            record["max_abs"].extend(batch[name]["max_abs"])
    return summarize_numerics(raw)["activations"]


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    subtype_by_subject: Mapping[str, str],
    *,
    model_name: str,
    outer_index: int,
    inner_index: int,
    branch_mode: str = "both",
    max_batches: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if branch_mode not in BRANCH_MODES:
        raise ValueError(branch_mode)
    model.eval()
    model.set_branch_mode(branch_mode)
    started = time.perf_counter()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    activation_batches: list[dict[str, Any]] = []
    loss_sum = 0.0
    samples = 0
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        tensors = _to_device(batch, device)
        outputs = _forward(model, tensors)
        losses = criterion(outputs, tensors["y"])
        probability = outputs["probabilities"].detach().float().cpu()
        prediction = probability.argmax(dim=-1)
        target = tensors["y"].detach().cpu()
        activation = _activation_batch(outputs)
        activation_batches.append(activation)
        count = int(target.shape[0])
        loss_sum += float(losses["loss"].detach().cpu()) * count
        samples += count
        targets.append(target.numpy())
        predictions.append(prediction.numpy())
        probabilities.append(probability.numpy())
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
                    "gate_mean": float(activation["gate_values"][index]),
                    **{
                        f"{name}_l2": float(activation[name]["l2"][index])
                        for name in ACTIVATIONS
                    },
                    "analysis_scope": "development_inner_cv_not_outer_test",
                }
            )
    model.set_branch_mode("both")
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
        "activation_statistics": _activation_summary(activation_batches),
        "gate": _distribution([float(row["gate_mean"]) for row in rows]),
    }
    return metrics, rows, diagnostics


def _prepare_fold(
    base_config: Mapping[str, Any],
    variant: str,
    outer: Mapping[str, Any],
    inner: Mapping[str, Any],
    stage_dir: Path,
    cache: HandcraftedCache,
    plan_sha256: str,
) -> tuple[dict[str, Any], Any, Any, Any, dict[str, Any]]:
    outer_index = int(outer["outer_fold"])
    inner_index = int(inner["inner_fold"])
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
        raise ValueError("Inner validation dataset is required")
    transform = fit_fold_statistical_transform(
        cache,
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
    metadata = {
        **transform.metadata,
        "scaler_sha256": sha256_file(scaler_path),
        "pca_sha256": sha256_file(pca_path),
        "outer_test_features_transformed": False,
    }
    write_json(preprocessing_dir / "metadata.json", metadata)
    return config, bundle, train_dataset, validation_dataset, metadata


def train_fold(
    base_config: Mapping[str, Any],
    variant: str,
    outer: Mapping[str, Any],
    inner: Mapping[str, Any],
    stage_dir: Path,
    device: torch.device,
    subtype_by_subject: Mapping[str, str],
    plan_sha256: str,
    cache: HandcraftedCache,
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
    config, bundle, train_dataset, validation_dataset, preprocessing = _prepare_fold(
        base_config, variant, outer, inner, stage_dir, cache, plan_sha256
    )
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
    model = build_stabilized_r2(config, variant).to(device)
    parameter_summary = count_parameters(model)
    write_json(
        stage_dir / "model.json",
        {
            **parameter_summary,
            "variant": variant,
            "precision": "FP32",
            "deep_embedding_dim": 514,
            "statistical_embedding_dim": 32,
            "classifier_input_dim": 64 if variant == "S3" else 546,
            "gated_projection_dim": 64 if variant == "S3" else None,
            "outer_test_loader_created": False,
        },
    )
    write_json(
        status_path,
        {
            "status": "running",
            "started_at_utc": utc_now(),
            "variant": variant,
            "outer_test_loader_created": False,
            "outer_test_signal_accessed": False,
        },
    )
    criterion = build_loss(config).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
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
    all_numerics = _empty_numerics()
    runtime_started = time.perf_counter()
    for epoch in range(start_epoch, maximum_epochs):
        train_metrics, epoch_numerics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            mixed_precision=False,
            gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
            epoch=epoch,
            max_batches=maximum_batches,
        )
        _extend_numerics(all_numerics, epoch_numerics)
        validation_metrics, _, _ = evaluate(
            model,
            validation_loader,
            criterion,
            device,
            subtype_by_subject,
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
        if not smoke and patience_count >= int(
            config["training"]["early_stopping"]["patience"]
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
    train_metrics, train_rows, train_diagnostics = evaluate(
        model,
        train_eval_loader,
        criterion,
        device,
        subtype_by_subject,
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
        model_name=variant,
        outer_index=outer_index,
        inner_index=inner_index,
    )
    write_csv(stage_dir / "predictions" / "train.csv", train_rows)
    write_csv(stage_dir / "predictions" / "validation.csv", validation_rows)
    branch_ablation: dict[str, Any] | None = None
    if variant in {"S2", "S3"}:
        branch_ablation = {}
        for mode in ("deep_disabled", "statistical_disabled"):
            mode_metrics, mode_rows, mode_diagnostics = evaluate(
                model,
                validation_loader,
                criterion,
                device,
                subtype_by_subject,
                model_name=variant,
                outer_index=outer_index,
                inner_index=inner_index,
                branch_mode=mode,
            )
            branch_ablation[mode] = {
                "metrics": mode_metrics,
                "diagnostics": mode_diagnostics,
            }
            write_csv(
                stage_dir / "predictions" / f"validation_{mode}.csv", mode_rows
            )
    numerical_summary = summarize_numerics(all_numerics)
    write_json(stage_dir / "training_numerics.json", numerical_summary)
    summary = {
        "status": "complete",
        "variant": variant,
        "precision": "FP32",
        "outer_context": outer_index,
        "inner_fold": inner_index,
        "fold_id": f"outer_{outer_index}/inner_{inner_index}",
        "best_epoch": int(checkpoint["epoch"]) + 1,
        "best_metric": float(checkpoint["best_metric"]),
        "train_metrics_at_best": train_metrics,
        "validation_metrics": validation_metrics,
        "train_validation_balanced_accuracy_gap": float(
            train_metrics["balanced_accuracy"]
            - validation_metrics["balanced_accuracy"]
        ),
        "train_diagnostics": train_diagnostics,
        "validation_diagnostics": validation_diagnostics,
        "branch_ablation": branch_ablation,
        "parameter_count": parameter_summary,
        "runtime_seconds": time.perf_counter() - runtime_started,
        "numerics": numerical_summary,
        "checkpoint_sha256": sha256_file(best_path),
        "checkpoint_provenance_sha256": sha256_file(
            best_path.with_suffix(best_path.suffix + ".provenance.json")
        ),
        "normalization_sha256": bundle.normalization["normalization_sha256"],
        "statistical_preprocessing": preprocessing,
        "plan_sha256": plan_sha256,
        "outer_test_loader_created": False,
        "outer_test_signal_accessed": False,
        "outer_test_predictions_accessed": False,
        "outer_test_features_transformed": False,
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


def run_variant(
    base_config: Mapping[str, Any],
    split_payload: Mapping[str, Any],
    variant: str,
    output_root: Path,
    device: torch.device,
    subtype_by_subject: Mapping[str, str],
    plan_sha256: str,
    cache: HandcraftedCache,
    *,
    resume: bool = True,
    smoke: bool = False,
) -> dict[str, Any]:
    variant_dir = output_root / "models" / variant
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for outer in split_payload["outer"]:
        outer_index = int(outer["outer_fold"])
        for inner in outer["inner_folds"]:
            inner_index = int(inner["inner_fold"])
            if smoke and (outer_index, inner_index) != (0, 0):
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
                        cache,
                        resume=resume,
                        smoke=smoke,
                    )
                )
            except Exception as error:
                failures.append(
                    {
                        "variant": variant,
                        "outer_context": outer_index,
                        "inner_fold": inner_index,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                write_json(stage_dir / "failure.json", failures[-1])
                if smoke:
                    raise
    all_rows: list[dict[str, Any]] = []
    for summary in summaries:
        all_rows.extend(
            list(
                csv.DictReader(
                    (
                        variant_dir
                        / f"outer_{summary['outer_context']}"
                        / f"inner_{summary['inner_fold']}"
                        / "predictions/validation.csv"
                    ).open(encoding="utf-8")
                )
            )
        )
    if all_rows:
        write_csv(variant_dir / "development_predictions_all.csv", all_rows)
    result = {
        "status": "complete" if not failures and len(summaries) == (1 if smoke else 15) else "failed",
        "variant": variant,
        "precision": "FP32",
        "fold_count": len(summaries),
        "failed_fold_count": len(failures),
        "failures": failures,
        "fold_metric_summary": aggregate_fold_summaries(summaries) if summaries else {},
        "fold_summaries": summaries,
        "prediction_rows": len(all_rows),
        "unique_validation_subjects": len({row["subject_id"] for row in all_rows}),
        "plan_sha256": plan_sha256,
        "scope": "development_inner_cv_not_outer_test",
        "outer_test_accessed": False,
    }
    write_json(variant_dir / "development_summary.json", result)
    return result


def diagnose_s0_amp(
    base_config: Mapping[str, Any],
    split_payload: Mapping[str, Any],
    output_root: Path,
    device: torch.device,
    plan_sha256: str,
    cache: HandcraftedCache,
) -> dict[str, Any]:
    fold_summaries: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for outer in split_payload["outer"]:
        outer_index = int(outer["outer_fold"])
        for inner in outer["inner_folds"]:
            inner_index = int(inner["inner_fold"])
            stage_dir = output_root / "s0_amp_diagnostic" / f"outer_{outer_index}" / f"inner_{inner_index}"
            config, _, train_dataset, _, preprocessing = _prepare_fold(
                base_config, "S1", outer, inner, stage_dir, cache, plan_sha256
            )
            config["training"]["mixed_precision"] = True
            seed_everything(42, True)
            model = build_stabilized_r2(config, "S1").to(device)
            criterion = build_loss(config).to(device)
            optimizer = build_optimizer(model, config)
            scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
            loader = subject_loader(
                train_dataset,
                batch_size=int(config["training"]["batch_size"]),
                shuffle=True,
                num_workers=int(config["data"].get("num_workers", 0)),
                pin_memory=bool(config["data"].get("pin_memory", False)),
                seed=42,
            )
            metrics, raw = train_epoch(
                model,
                loader,
                criterion,
                optimizer,
                device,
                scaler,
                mixed_precision=True,
                gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
                epoch=0,
            )
            summary = summarize_numerics(raw)
            fold = {
                "outer_context": outer_index,
                "inner_fold": inner_index,
                "batches": metrics["batches"],
                "nonfinite_gradient_batches": summary["nonfinite_gradient_batches"],
                "nonfinite_loss_batches": summary["nonfinite_loss_batches"],
                "skipped_optimizer_steps": summary["skipped_optimizer_steps"],
                "scaler_initial": summary["scaler_scale_before"]["maximum"],
                "scaler_final": summary["scaler_scale_after"]["minimum"],
                "preprocessing_fit_subject_ids_sha256": preprocessing[
                    "fit_subject_ids_sha256"
                ],
                "outer_test_loader_created": False,
                "outer_test_signal_accessed": False,
            }
            fold_summaries.append(fold)
            for event in summary["overflow_events"]:
                event_rows.append(
                    {
                        "outer_context": outer_index,
                        "inner_fold": inner_index,
                        **event,
                    }
                )
            for group, values in summary["gradient_groups"].items():
                group_rows.append(
                    {
                        "outer_context": outer_index,
                        "inner_fold": inner_index,
                        "group": group,
                        **values,
                    }
                )
            write_json(stage_dir / "diagnostic_summary.json", {"fold": fold, "numerics": summary})
    write_csv(output_root / "s0_amp_diagnostic_folds.csv", fold_summaries)
    if event_rows:
        write_csv(output_root / "s0_amp_overflow_events.csv", event_rows)
    write_csv(output_root / "s0_amp_gradient_groups.csv", group_rows)
    result = {
        "status": "complete",
        "fold_count": len(fold_summaries),
        "total_nonfinite_gradient_batches": sum(
            row["nonfinite_gradient_batches"] for row in fold_summaries
        ),
        "total_skipped_optimizer_steps": sum(
            row["skipped_optimizer_steps"] for row in fold_summaries
        ),
        "overflow_event_count": len(event_rows),
        "outer_test_accessed": False,
    }
    write_json(output_root / "s0_amp_diagnostic_summary.json", result)
    return result


@torch.no_grad()
def diagnose_s0_checkpoint_activations(
    base_config: Mapping[str, Any],
    split_payload: Mapping[str, Any],
    output_root: Path,
    device: torch.device,
    plan_sha256: str,
    cache: HandcraftedCache,
) -> dict[str, Any]:
    """Read-only inference over historical S0 checkpoints for scale diagnosis."""

    historical_root = Path(
        "/home/zyt/MFAM/outputs/pads_classification/v3_targeted_ablation/"
        "targeted_ablation_20260828/models/R2"
    )
    rows: list[dict[str, Any]] = []
    for outer in split_payload["outer"]:
        outer_index = int(outer["outer_fold"])
        for inner in outer["inner_folds"]:
            inner_index = int(inner["inner_fold"])
            stage_dir = (
                output_root
                / "s0_checkpoint_diagnostic"
                / f"outer_{outer_index}"
                / f"inner_{inner_index}"
            )
            config, _, _, validation_dataset, preprocessing = _prepare_fold(
                base_config, "S1", outer, inner, stage_dir, cache, plan_sha256
            )
            model = build_targeted_model(config, ("R2",)).to(device)
            checkpoint_path = (
                historical_root
                / f"outer_{outer_index}"
                / f"inner_{inner_index}"
                / "checkpoints/best.pt"
            )
            load_checkpoint(checkpoint_path, model=model, map_location=device)
            loader = subject_loader(
                validation_dataset,
                batch_size=int(config["evaluation"]["batch_size"]),
                shuffle=False,
                num_workers=int(config["data"].get("num_workers", 0)),
                pin_memory=bool(config["data"].get("pin_memory", False)),
                seed=43,
            )
            raw: dict[str, dict[str, Any]] = {
                name: {"dimension": 0, "l2": [], "max_abs": []}
                for name in (
                    "deep_embedding_raw",
                    "statistical_embedding_raw",
                    "fused_embedding",
                    "classifier_input",
                    "logits",
                )
            }
            model.eval()
            for batch in loader:
                tensors = _to_device(batch, device)
                outputs = model(
                    tensors["x"],
                    tensors["wrist_mask"],
                    tensors["activity_mask"],
                    tensors["activity_lengths"],
                    tensors["statistical_features"],
                )
                values = {
                    "deep_embedding_raw": outputs["bag_embedding"],
                    "statistical_embedding_raw": tensors["statistical_features"],
                    "fused_embedding": outputs["classifier_features"],
                    "classifier_input": outputs["classifier_features"],
                    "logits": outputs["logits"],
                }
                for name, tensor in values.items():
                    flat = tensor.detach().float().reshape(tensor.shape[0], -1)
                    raw[name]["dimension"] = int(flat.shape[1])
                    raw[name]["l2"].extend(
                        float(value) for value in flat.norm(2, dim=1).cpu().tolist()
                    )
                    raw[name]["max_abs"].extend(
                        float(value)
                        for value in flat.abs().amax(dim=1).cpu().tolist()
                    )
            for name, values in raw.items():
                distribution = _distribution(values["l2"])
                rows.append(
                    {
                        "model": "S0",
                        "partition": "validation",
                        "outer_context": outer_index,
                        "inner_fold": inner_index,
                        "activation": name,
                        "dimension": values["dimension"],
                        **{f"l2_{key}": value for key, value in distribution.items()},
                        "maximum_absolute": _distribution(values["max_abs"])["maximum"],
                        "checkpoint_sha256": sha256_file(checkpoint_path),
                        "pca_explained_variance_ratio_sum": preprocessing[
                            "pca_explained_variance_ratio_sum"
                        ],
                        "outer_test_accessed": False,
                    }
                )
    write_csv(output_root / "s0_checkpoint_activation_statistics.csv", rows)
    result = {
        "status": "complete",
        "fold_count": 15,
        "rows": len(rows),
        "outer_test_accessed": False,
    }
    write_json(output_root / "s0_checkpoint_diagnostic_summary.json", result)
    return result
