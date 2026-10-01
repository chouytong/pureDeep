#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" - <<'PY'
from pathlib import Path

files = [
    path for path in Path(".").rglob("*.py")
    if not path.name.startswith("._") and "outputs" not in path.parts
]
for path in files:
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")
print(f"Static compilation passed for {len(files)} Python files.")
PY

"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
from src.utils.config import load_config

# Validate both the frozen V2 baseline and the isolated V3 protocol.
paths = [
    Path("configs/pads_multi_activity_v2.yaml"),
    Path("configs/pads_multi_activity_v3_nested_cv.yaml"),
]
for path in paths:
    load_config(path)
    print(f"Validated {path}")
PY
