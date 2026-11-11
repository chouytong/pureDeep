from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def create_run_directory(output_root: str | Path, experiment_name: str) -> Path:
    root = Path(output_root).expanduser()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = root / f"{experiment_name}-{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"{experiment_name}-{timestamp}-{suffix:02d}"
        suffix += 1
    for subdirectory in ("checkpoints", "logs", "predictions"):
        (candidate / subdirectory).mkdir(parents=True, exist_ok=False)
    return candidate.resolve()


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

