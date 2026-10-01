from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from src.datasets.manifest import selected_channel_names


PROVENANCE_SCHEMA_VERSION = 1
EXPECTED_CLASS_NAMES = ("PD", "DD")
DEFAULT_LABEL_MAPPING = {
    "Parkinson's": "PD",
    "Other Movement Disorders": "DD",
    "Essential Tremor": "DD",
    "Atypical Parkinsonism": "DD",
    "Multiple Sclerosis": "DD",
    "Healthy": "excluded",
}
DEFAULT_SOURCE_FILES = (
    "README.md",
    "data/README.md",
    "environment.yml",
    "requirements.txt",
    "pyproject.toml",
    "train.py",
    "evaluate.py",
    "inference.py",
)
DEFAULT_SOURCE_DIRECTORIES = ("configs", "src", "scripts", "tests")
SOURCE_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".toml"}


def sha256_file(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Cannot hash missing file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _clean_config(config: Mapping[str, Any]) -> dict[str, Any]:
    clean = copy.deepcopy(dict(config))
    clean.pop("_meta", None)
    return clean


def semantic_config_sha256(config: Mapping[str, Any]) -> str:
    """Hash the resolved effective config, including command-line overrides."""
    return sha256_json(_clean_config(config))


def normalization_sha256(mean: torch.Tensor, std: torch.Tensor) -> str:
    """Hash float32 normalization tensors with explicit names and shapes."""
    digest = hashlib.sha256(b"mfam-normalization-v1\0")
    for name, value in (("mean", mean), ("std", std)):
        tensor = value.detach().cpu().to(dtype=torch.float32).contiguous()
        digest.update(name.encode("ascii") + b"\0")
        digest.update(canonical_json_bytes(list(tensor.shape)) + b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def normalization_metadata(
    mean: torch.Tensor,
    std: torch.Tensor,
    subject_ids: Iterable[str],
    fold_id: str,
) -> dict[str, Any]:
    subjects = sorted({str(value) for value in subject_ids})
    if not subjects:
        raise ValueError("Normalization fitting subjects cannot be empty")
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "fold_id": str(fold_id),
        "fitted_on": "explicit_training_subjects_only",
        "subject_ids": subjects,
        "subject_count": len(subjects),
        "subject_ids_sha256": sha256_json(subjects),
        "mean_shape": list(mean.shape),
        "std_shape": list(std.shape),
        "normalization_sha256": normalization_sha256(mean, std),
    }


def source_tree_manifest(project_root: str | Path) -> tuple[list[dict[str, str]], str]:
    """Return sorted source hashes and a deterministic hash of their text manifest."""
    root = Path(project_root).expanduser().resolve()
    paths: set[Path] = set()
    for relative in DEFAULT_SOURCE_FILES:
        candidate = root / relative
        if candidate.is_file():
            paths.add(candidate)
    for relative in DEFAULT_SOURCE_DIRECTORIES:
        directory = root / relative
        if not directory.is_dir():
            continue
        for candidate in directory.rglob("*"):
            if not candidate.is_file():
                continue
            if candidate.suffix not in SOURCE_SUFFIXES:
                continue
            if candidate.suffix == ".pyc" or "__pycache__" in candidate.parts:
                continue
            paths.add(candidate)
    entries = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix())
    ]
    text = "".join(
        f"{entry['sha256']}  {entry['relative_path']}\n" for entry in entries
    )
    return entries, hashlib.sha256(text.encode("utf-8")).hexdigest()


def _project_root(config: Mapping[str, Any]) -> Path:
    meta = config.get("_meta", {})
    value = meta.get("project_root")
    if value:
        return Path(str(value)).expanduser().resolve()
    return Path.cwd().resolve()


def _resolve(path: str | Path, root: Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else root / value).resolve()


def _split_path(config: Mapping[str, Any], root: Path) -> Path:
    nested = config.get("nested_cv", {})
    value = nested.get("split_file") or config.get("data", {}).get("fold", {}).get(
        "split_file"
    )
    if not value:
        raise ValueError("Strict provenance requires a split file")
    return _resolve(str(value), root)


def _preprocessing_paths(config: Mapping[str, Any], root: Path) -> list[Path]:
    configured = config.get("provenance", {}).get("preprocessing_artifacts")
    if configured:
        return [_resolve(str(value), root) for value in configured]
    data_root = Path(str(config["data"]["root"])).expanduser().resolve()
    defaults = (
        data_root / "processing_config.json",
        data_root / "audit.json",
        data_root / "quality_summary.json",
    )
    return [path for path in defaults if path.is_file()]


def runtime_identity(
    config: Mapping[str, Any],
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> dict[str, Any]:
    root = _project_root(config)
    data = config["data"]
    classes = [str(value) for value in data["labels"]]
    if tuple(classes) != EXPECTED_CLASS_NAMES:
        raise ValueError(
            f"PADS PD-vs-DD class order must be {list(EXPECTED_CLASS_NAMES)}, got {classes}"
        )
    activities = [str(value) for value in data["activities"]]
    manifests = data.get("activity_manifests")
    if not isinstance(manifests, Mapping) or set(manifests) != set(activities):
        raise ValueError("Strict provenance requires one manifest per configured activity")
    _, tree_hash = source_tree_manifest(root)
    preprocessing = _preprocessing_paths(config, root)
    provenance_config = config.get("provenance", {})
    result: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "experiment_version": str(
            provenance_config.get("experiment_version", config["experiment"]["name"])
        ),
        "pads_dataset_version": str(
            provenance_config.get("pads_dataset_version", "1.0.0")
        ),
        "pads_doi": str(provenance_config.get("pads_doi", "10.13026/m0w9-zx22")),
        "class_names": classes,
        "label_mapping": dict(
            provenance_config.get("label_mapping", DEFAULT_LABEL_MAPPING)
        ),
        "activity_order": activities,
        "wrist_order": [str(value) for value in data.get("wrist_order", ["left", "right"])],
        "sensor_mode": str(data.get("sensor_mode", "acc_gyro")),
        "channel_order": selected_channel_names(str(data.get("sensor_mode", "acc_gyro"))),
        "config_sha256": semantic_config_sha256(config),
        "split_sha256": sha256_file(_split_path(config, root)),
        "manifest_sha256": {
            activity: sha256_file(_resolve(str(manifests[activity]), root))
            for activity in activities
        },
        "preprocessing_sha256": {
            path.relative_to(root).as_posix(): sha256_file(path)
            for path in sorted(preprocessing)
        },
        "source_tree_sha256": tree_hash,
    }
    if mean is not None or std is not None:
        if mean is None or std is None:
            raise ValueError("Both mean and std are required for normalization provenance")
        result["normalization_sha256"] = normalization_sha256(mean, std)
    return result


def _mismatch(name: str, actual: Any, expected: Any) -> ValueError:
    return ValueError(
        f"Checkpoint provenance mismatch for {name}: "
        f"checkpoint={actual!r}, runtime={expected!r}"
    )


def validate_strict_provenance(
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    runtime_mean: torch.Tensor | None = None,
    runtime_std: torch.Tensor | None = None,
) -> None:
    actual = checkpoint.get("provenance")
    if not isinstance(actual, Mapping):
        raise ValueError("Checkpoint does not contain strict provenance metadata")
    stored_normalization = checkpoint.get("normalization")
    if not isinstance(stored_normalization, Mapping):
        raise ValueError("Checkpoint does not contain normalization tensors")
    checkpoint_norm_hash = normalization_sha256(
        stored_normalization["mean"], stored_normalization["std"]
    )
    if actual.get("normalization_sha256") != checkpoint_norm_hash:
        raise ValueError("Checkpoint normalization hash does not match stored tensors")
    expected = runtime_identity(config, runtime_mean, runtime_std)
    keys = (
        "schema_version",
        "experiment_version",
        "pads_dataset_version",
        "pads_doi",
        "class_names",
        "label_mapping",
        "activity_order",
        "wrist_order",
        "sensor_mode",
        "channel_order",
        "config_sha256",
        "split_sha256",
        "manifest_sha256",
        "preprocessing_sha256",
        "source_tree_sha256",
    )
    for key in keys:
        if actual.get(key) != expected.get(key):
            raise _mismatch(key, actual.get(key), expected.get(key))
    if runtime_mean is not None and actual.get("normalization_sha256") != expected.get(
        "normalization_sha256"
    ):
        raise _mismatch(
            "normalization_sha256",
            actual.get("normalization_sha256"),
            expected.get("normalization_sha256"),
        )
    if list(checkpoint.get("class_names", [])) != list(EXPECTED_CLASS_NAMES):
        raise ValueError("Checkpoint class_names must be exactly ['PD', 'DD']")


def _flatten_checkpoint_hashes(provenance: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    for value in provenance.get("checkpoint_sha256", {}).values():
        if isinstance(value, Mapping):
            result.update(str(item) for item in value.values())
        else:
            result.add(str(value))
    return result


def validate_legacy_v2_sidecar(
    checkpoint: Mapping[str, Any],
    checkpoint_path: str | Path,
    config: Mapping[str, Any],
    sidecar_path: str | Path,
    runtime_mean: torch.Tensor | None = None,
    runtime_std: torch.Tensor | None = None,
) -> None:
    sidecar_file = Path(sidecar_path).expanduser().resolve()
    sidecar = json.loads(sidecar_file.read_text(encoding="utf-8"))
    if sidecar.get("experiment_version") != "v2_frozen_historical_baseline":
        raise ValueError("Legacy sidecar is not the frozen V2 provenance artifact")
    digest = sha256_file(checkpoint_path)
    if digest not in _flatten_checkpoint_hashes(sidecar):
        raise ValueError("Legacy checkpoint SHA-256 is not recorded in the V2 sidecar")
    expected_task = sidecar["task"]
    runtime = runtime_identity(config, runtime_mean, runtime_std)
    for sidecar_key, runtime_key in (
        ("class_names", "class_names"),
        ("label_mapping", "label_mapping"),
        ("activity_order", "activity_order"),
        ("wrist_order", "wrist_order"),
        ("sensor_mode", "sensor_mode"),
        ("channel_order", "channel_order"),
    ):
        if expected_task[sidecar_key] != runtime[runtime_key]:
            raise _mismatch(runtime_key, expected_task[sidecar_key], runtime[runtime_key])
    if sidecar["frozen_split"]["sha256"] != runtime["split_sha256"]:
        raise _mismatch(
            "split_sha256", sidecar["frozen_split"]["sha256"], runtime["split_sha256"]
        )
    if sidecar["manifest_sha256"] != runtime["manifest_sha256"]:
        raise _mismatch(
            "manifest_sha256", sidecar["manifest_sha256"], runtime["manifest_sha256"]
        )
    expected_preprocessing = sidecar["preprocessing_artifact_sha256"]
    if expected_preprocessing != runtime["preprocessing_sha256"]:
        raise _mismatch(
            "preprocessing_sha256", expected_preprocessing, runtime["preprocessing_sha256"]
        )
    if list(checkpoint.get("class_names", [])) != list(EXPECTED_CLASS_NAMES):
        raise ValueError("Legacy checkpoint class_names must be exactly ['PD', 'DD']")
    stored_config = checkpoint.get("config")
    if not isinstance(stored_config, Mapping):
        raise ValueError("Legacy checkpoint does not contain its effective config")
    if semantic_config_sha256(stored_config) != semantic_config_sha256(config):
        raise ValueError("Legacy checkpoint effective config does not match runtime config")
    if runtime_mean is not None:
        if runtime_std is None:
            raise ValueError("runtime_std is required with runtime_mean")
        stored = checkpoint["normalization"]
        if normalization_sha256(stored["mean"], stored["std"]) != normalization_sha256(
            runtime_mean, runtime_std
        ):
            raise ValueError("Legacy checkpoint normalization does not match runtime split")


def checkpoint_sidecar_payload(
    checkpoint_path: str | Path, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    path = Path(checkpoint_path).expanduser().resolve()
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "checkpoint_path": str(path),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_provenance": dict(provenance),
    }
