from __future__ import annotations

import contextlib
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

from src.metrics import classification_metrics

from .checkpoint import unwrap_model


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def _move_batch(
    batch: dict[str, Any], device: torch.device
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    # Activity input is [B,2,C,T]; subject input is [B,A,2,C,T].
    inputs = batch["x"].to(device, non_blocking=True)
    targets = batch["y"].to(device, non_blocking=True)
    wrist_mask = batch["wrist_mask"].to(device, non_blocking=True)
    activity_mask = batch.get("activity_mask")
    if activity_mask is not None:
        activity_mask = activity_mask.to(device, non_blocking=True)
    activity_lengths = batch.get("activity_lengths")
    if activity_lengths is not None:
        activity_lengths = activity_lengths.to(device, non_blocking=True)
    return inputs, targets, wrist_mask, activity_mask, activity_lengths


def _model_forward(
    model: nn.Module,
    inputs: torch.Tensor,
    wrist_mask: torch.Tensor,
    activity_mask: torch.Tensor | None,
    activity_lengths: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | None]:
    if activity_mask is None:
        return model(inputs, wrist_mask)
    return model(inputs, wrist_mask, activity_mask, activity_lengths)


def _forward_with_views(
    model: nn.Module,
    inputs: torch.Tensor,
    wrist_mask: torch.Tensor,
    activity_mask: torch.Tensor | None = None,
    activity_lengths: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | None]:
    """对确定性裁剪逐视图推理，再平均类别概率，输出 [batch, classes]。"""
    if (activity_mask is None and inputs.ndim == 4) or (
        activity_mask is not None and inputs.ndim == 5
    ):
        return _model_forward(
            model, inputs, wrist_mask, activity_mask, activity_lengths
        )
    if activity_mask is not None:
        if inputs.ndim != 6:
            raise ValueError(
                "Expected [B,A,2,C,T] or [B,V,A,2,C,T], "
                f"got {tuple(inputs.shape)}"
            )
        # Process views sequentially to avoid multiplying A-activity GPU memory by V.
        view_outputs = [
            _model_forward(
                model,
                inputs[:, view],
                wrist_mask,
                activity_mask,
                activity_lengths,
            )
            for view in range(inputs.shape[1])
        ]
        probability_tensors = [output["probabilities"] for output in view_outputs]
        if any(value is None for value in probability_tensors):
            raise ValueError("Model outputs do not contain probabilities")
        probabilities = torch.stack(
            [value for value in probability_tensors if value is not None], dim=1
        ).mean(dim=1)
        attention_tensors = [output.get("activity_attention") for output in view_outputs]
        attention = None
        if all(value is not None for value in attention_tensors):
            attention = torch.stack(
                [value for value in attention_tensors if value is not None], dim=1
            ).mean(dim=1)
        return {
            "logits": probabilities.clamp_min(1e-8).log(),
            "probabilities": probabilities,
            "bag_embedding": None,
            "activity_attention": attention,
        }
    if inputs.ndim != 5:
        raise ValueError(
            "Expected [B,2,C,T] or [B,V,2,C,T], "
            f"got {tuple(inputs.shape)}"
        )
    batch_size, views, wrists, channels, length = inputs.shape
    flattened = inputs.reshape(batch_size * views, wrists, channels, length)
    # 每个裁剪视图继承同一条 pair 的腕侧有效性。
    flattened_mask = (
        wrist_mask[:, None, :]
        .expand(batch_size, views, wrists)
        .reshape(batch_size * views, wrists)
    )
    view_outputs = model(flattened, flattened_mask)
    view_probabilities = view_outputs["probabilities"]
    if view_probabilities is None:
        raise ValueError("Model outputs do not contain probabilities")
    probabilities = view_probabilities.reshape(batch_size, views, -1).mean(dim=1)
    logits = probabilities.clamp_min(1e-8).log()
    return {
        "logits": logits,
        "probabilities": probabilities,
        "bag_embedding": None,
    }


def _aggregate_losses(
    sums: dict[str, float], losses: dict[str, torch.Tensor], batch_size: int
) -> None:
    for key in ("loss", "classification_loss"):
        sums[key] = sums.get(key, 0.0) + float(losses[key].detach()) * batch_size


def _distributed_training_aggregate(
    targets: np.ndarray,
    predictions: np.ndarray,
    loss_sums: dict[str, float],
    sample_count: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], int]:
    if not (dist.is_available() and dist.is_initialized()):
        return targets, predictions, loss_sums, sample_count
    gathered: list[tuple[np.ndarray, np.ndarray] | None] = [
        None for _ in range(dist.get_world_size())
    ]
    dist.all_gather_object(gathered, (targets, predictions))
    targets = np.concatenate([item[0] for item in gathered if item is not None])
    predictions = np.concatenate(
        [item[1] for item in gathered if item is not None]
    )
    keys = ("loss", "classification_loss")
    totals = torch.tensor(
        [float(sample_count)] + [loss_sums.get(key, 0.0) for key in keys],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    sample_count = int(totals[0].item())
    loss_sums = {
        key: float(totals[index + 1].item()) for index, key in enumerate(keys)
    }
    return targets, predictions, loss_sums, sample_count


def train_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Any = None,
    mixed_precision: bool = False,
    gradient_clip_norm: float | None = None,
    max_batches: int = 0,
    zero_division: float = 0.0,
) -> dict[str, Any]:
    model.train()
    started = time.perf_counter()
    loss_sums: dict[str, float] = {}
    targets_all: list[np.ndarray] = []
    predictions_all: list[np.ndarray] = []
    sample_count = 0
    processed_batches = 0
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        inputs, targets, wrist_mask, activity_mask, activity_lengths = _move_batch(
            batch, device
        )
        if inputs.ndim not in {4, 5}:
            raise ValueError(
                "Training must yield one crop per record; set data.train_crop to "
                "'random', 'center', or 'start'"
            )
        optimizer.zero_grad(set_to_none=True)
        # AMP 只改变数值执行方式；数据划分和标准化在进入训练循环前已冻结。
        with _autocast(device, mixed_precision):
            outputs = _model_forward(
                model, inputs, wrist_mask, activity_mask, activity_lengths
            )
            losses = criterion(outputs, targets)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip_norm
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["loss"].backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip_norm
                )
            optimizer.step()

        batch_size = int(targets.shape[0])
        _aggregate_losses(loss_sums, losses, batch_size)
        sample_count += batch_size
        processed_batches += 1
        targets_all.append(targets.detach().cpu().numpy())
        predictions_all.append(
            outputs["logits"].detach().argmax(dim=-1).cpu().numpy()
        )

    if sample_count == 0:
        raise RuntimeError("No training batches were processed")
    targets_array, predictions_array, loss_sums, sample_count = (
        _distributed_training_aggregate(
            np.concatenate(targets_all),
            np.concatenate(predictions_all),
            loss_sums,
            sample_count,
            device,
        )
    )
    metrics = classification_metrics(
        targets_array,
        predictions_array,
        num_classes=unwrap_model(model).num_classes,
        zero_division=zero_division,
    )
    metrics.update({key: value / sample_count for key, value in loss_sums.items()})
    metrics["duration_seconds"] = time.perf_counter() - started
    metrics["batches"] = processed_batches
    metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
    return metrics


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    criterion: nn.Module,
    device: torch.device,
    mixed_precision: bool = False,
    max_batches: int = 0,
    zero_division: float = 0.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    started = time.perf_counter()
    loss_sums: dict[str, float] = {}
    targets_all: list[np.ndarray] = []
    predictions_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    sample_count = 0
    processed_batches = 0
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        inputs, targets, wrist_mask, activity_mask, activity_lengths = _move_batch(
            batch, device
        )
        with _autocast(device, mixed_precision):
            outputs = _forward_with_views(
                model, inputs, wrist_mask, activity_mask, activity_lengths
            )
            losses = criterion(outputs, targets)
        probabilities = outputs["probabilities"].detach().cpu()
        predictions = probabilities.argmax(dim=-1)
        batch_size = int(targets.shape[0])
        _aggregate_losses(loss_sums, losses, batch_size)
        sample_count += batch_size
        processed_batches += 1
        targets_cpu = targets.detach().cpu()
        targets_all.append(targets_cpu.numpy())
        predictions_all.append(predictions.numpy())
        probabilities_all.append(probabilities.numpy())
        for index in range(batch_size):
            # 保留 subject_id，便于复核逐样本预测与受试者划分是否一致。
            row = {
                "sample_id": str(batch["sample_id"][index]),
                "pair_id": str(batch["pair_id"][index]),
                "subject_id": str(batch["subject_id"][index]),
                "activity": str(batch["activity"][index]),
                "wrist_mode": str(batch["wrist_mode"][index]),
                "sensor_mode": str(batch["sensor_mode"][index]),
                "left_path": str(batch["left_path"][index]),
                "right_path": str(batch["right_path"][index]),
                "target": int(targets_cpu[index]),
                "prediction": int(predictions[index]),
                "probabilities": probabilities[index].tolist(),
            }
            attention = outputs.get("activity_attention")
            if attention is not None:
                row["activity_mask"] = batch["activity_mask"][index].tolist()
                row["activity_attention"] = attention[index].detach().cpu().tolist()
                row["activities"] = str(batch["activities"][index])
                row["activity_pair_ids"] = str(batch["activity_pair_ids"][index])
            rows.append(row)
    if sample_count == 0:
        raise RuntimeError("No evaluation batches were processed")
    metrics = classification_metrics(
        np.concatenate(targets_all),
        np.concatenate(predictions_all),
        num_classes=unwrap_model(model).num_classes,
        zero_division=zero_division,
        probabilities=np.concatenate(probabilities_all),
    )
    metrics.update({key: value / sample_count for key, value in loss_sums.items()})
    metrics["duration_seconds"] = time.perf_counter() - started
    metrics["batches"] = processed_batches
    return metrics, rows
