from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn


def build_optimizer(
    model: nn.Module, config: Mapping[str, Any]
) -> torch.optim.Optimizer:
    training = config["training"]
    name = str(training.get("optimizer", "adam")).lower()
    common = {
        "lr": float(training["learning_rate"]),
        "weight_decay": float(training.get("weight_decay", 0.0)),
    }
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            betas=tuple(float(v) for v in training.get("adam_betas", [0.9, 0.999])),
            **common,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            betas=tuple(float(v) for v in training.get("adam_betas", [0.9, 0.999])),
            **common,
        )
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            momentum=float(training.get("momentum", 0.9)),
            **common,
        )
    raise ValueError(f"Unsupported optimizer: {name!r}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
) -> torch.optim.lr_scheduler.LRScheduler | None:
    scheduler_config = config["training"].get("scheduler", {"type": "none"})
    name = str(scheduler_config.get("type", "none")).lower()
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config["training"]["epochs"]),
            eta_min=float(scheduler_config.get("minimum_lr", 0.0)),
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_config.get("step_size", 30)),
            gamma=float(scheduler_config.get("gamma", 0.1)),
        )
    raise ValueError(f"Unsupported scheduler: {name!r}")
