from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from src.datasets.manifest import WRIST_ORDER, selected_channel_names
from src.utils.provenance import (
    checkpoint_sidecar_payload,
    runtime_identity,
    validate_legacy_v2_sidecar,
    validate_strict_provenance,
)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def input_metadata_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    data = config.get("data", {})
    sensor_mode = str(data.get("sensor_mode", "acc_gyro"))
    metadata = {
        "tensor_layout": "[batch, wrist, channel, time]",
        "wrist_order": list(data.get("wrist_order", WRIST_ORDER)),
        "wrist_mode": str(data.get("wrist_mode", "bilateral")),
        "sensor_mode": sensor_mode,
        "channel_names": selected_channel_names(sensor_mode),
    }
    if str(data.get("unit", "activity")).lower() == "subject":
        activities = [str(value) for value in data["activities"]]
        metadata.update(
            {
                "unit": "subject",
                "tensor_layout": "[batch, activity, wrist, channel, time]",
                "activities": activities,
                "activity_count": len(activities),
                "activity_fusion": "attention",
            }
        )
    if data.get("sequence_length") is None:
        metadata["time_length"] = "variable_full_signal"
    if str(data.get("normalization_scope", "wrist_channel")) != "wrist_channel":
        metadata["normalization_scope"] = str(data["normalization_scope"])
    return metadata


def validate_checkpoint_input(
    checkpoint: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    expected = input_metadata_from_config(config)
    actual = checkpoint.get("input_metadata")
    if actual != expected:
        raise ValueError(
            "Checkpoint input metadata does not match current configuration: "
            f"checkpoint={actual}, expected={expected}"
        )


def validate_checkpoint_compatibility(
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    checkpoint_path: str | Path,
    runtime_mean: torch.Tensor | None = None,
    runtime_std: torch.Tensor | None = None,
    allow_legacy_checkpoint: bool = False,
    legacy_provenance: str | Path | None = None,
) -> None:
    """Validate V3 strictly; permit V2 only through an explicit sidecar opt-in."""
    validate_checkpoint_input(checkpoint, config)
    if checkpoint.get("provenance") is not None:
        if allow_legacy_checkpoint or legacy_provenance is not None:
            raise ValueError(
                "Legacy checkpoint options cannot be used with a V3 checkpoint."
            )
        validate_strict_provenance(
            checkpoint,
            config,
            runtime_mean=runtime_mean,
            runtime_std=runtime_std,
        )
        return

    if not allow_legacy_checkpoint:
        raise ValueError(
            "Legacy V2 checkpoint rejected. Re-run with --allow-legacy-checkpoint "
            "and --legacy-provenance pointing to the frozen V2 provenance sidecar."
        )
    if legacy_provenance is None:
        raise ValueError(
            "--legacy-provenance is required when --allow-legacy-checkpoint is set."
        )
    validate_legacy_v2_sidecar(
        checkpoint,
        checkpoint_path=checkpoint_path,
        config=config,
        sidecar_path=legacy_provenance,
        runtime_mean=runtime_mean,
        runtime_std=runtime_std,
    )


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    best_metric: float,
    config: Mapping[str, Any],
    class_names: list[str],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    clean_config = dict(config)
    clean_config.pop("_meta", None)
    strict_provenance = bool(
        config.get("provenance", {}).get("strict_checkpoint_validation", False)
    )
    payload = {
        "format_version": 3 if strict_provenance else 2,
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "model_state": unwrap_model(model).state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "scaler_state": None if scaler is None else scaler.state_dict(),
        "config": clean_config,
        "class_names": list(class_names),
        "normalization": {
            "mean": mean.detach().cpu(),
            "std": std.detach().cpu(),
            "fitted_on": "train_subjects_only",
        },
        # 加载 checkpoint 时必须核对这些字段，防止腕侧/通道顺序错位。
        "input_metadata": input_metadata_from_config(config),
    }
    if strict_provenance:
        payload["provenance"] = runtime_identity(config, mean=mean, std=std)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    if strict_provenance:
        sidecar = output.with_suffix(output.suffix + ".provenance.json")
        sidecar_temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
        sidecar_temporary.write_text(
            json.dumps(
                checkpoint_sidecar_payload(output, payload["provenance"]),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        sidecar_temporary.replace(sidecar)


def load_checkpoint(
    path: str | Path,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path, map_location=map_location, weights_only=False
    )
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(f"Invalid classification checkpoint: {checkpoint_path}")
    if model is not None:
        unwrap_model(model).load_state_dict(checkpoint["model_state"], strict=strict)
    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None and checkpoint.get("scaler_state") is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])
    return checkpoint
