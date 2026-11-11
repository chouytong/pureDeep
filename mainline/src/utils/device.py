from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    is_main: bool


def select_device(requested: str = "auto", local_rank: int = 0) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{local_rank}")
        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device(f"cuda:{local_rank}")
    if requested == "mps":
        if not (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS was requested but is not available")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device(requested)


def init_distributed(enabled: bool | str = "auto", backend: str = "auto") -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    should_enable = world_size > 1 if enabled == "auto" else bool(enabled)
    if not should_enable:
        return DistributedContext(False, 0, 0, 1, True)

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if backend == "auto":
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return DistributedContext(True, rank, local_rank, world_size, rank == 0)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

