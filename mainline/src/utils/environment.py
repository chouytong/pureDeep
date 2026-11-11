from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def collect_environment(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    cuda_version = torch.version.cuda
    cudnn_version = (
        torch.backends.cudnn.version()
        if torch.backends.cudnn.is_available()
        else None
    )
    gpu_names = []
    if torch.cuda.is_available():
        gpu_names = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pyyaml": yaml.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": cuda_version,
        "cudnn_version": cudnn_version,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_names": gpu_names,
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "git_commit": _git_commit(root),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
    }


def write_environment(info: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(info, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

