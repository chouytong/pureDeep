from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping

import yaml

from src.datasets.manifest import SENSOR_INDICES, WRIST_ORDER


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            raise KeyError(f"Required environment variable {name!r} is not set")

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def deep_merge(base: MutableMapping[str, Any], update: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in update.items():
        if key in base and isinstance(base[key], MutableMapping) and isinstance(value, Mapping):
            deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _set_nested(config: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current = config
    for part in parts[:-1]:
        child = current.get(part)
        if child is None:
            child = {}
            current[part] = child
        if not isinstance(child, MutableMapping):
            raise TypeError(f"Cannot set {dotted_key!r}: {part!r} is not a mapping")
        current = child
    current[parts[-1]] = value


def parse_overrides(items: Iterable[str] | None) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got {item!r}")
        key, raw_value = item.split("=", 1)
        _set_nested(parsed, key, yaml.safe_load(raw_value))
    return parsed


def _find_project_root(config_path: Path) -> Path:
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return config_path.parent


def _absolute(value: Any, project_root: Path) -> Any:
    if not isinstance(value, str) or not value:
        return value
    path = Path(value).expanduser()
    return str(path.resolve() if path.is_absolute() else (project_root / path).resolve())


def _resolve_paths(config: MutableMapping[str, Any], project_root: Path) -> None:
    experiment = config.get("experiment", {})
    if "output_root" in experiment:
        experiment["output_root"] = _absolute(experiment["output_root"], project_root)
    data = config.get("data", {})
    for key in ("root", "manifest"):
        if key in data:
            data[key] = _absolute(data[key], project_root)
    activity_manifests = data.get("activity_manifests")
    if isinstance(activity_manifests, MutableMapping):
        for activity, value in activity_manifests.items():
            activity_manifests[activity] = _absolute(value, project_root)
    fold = data.get("fold", {})
    if fold.get("split_file"):
        fold["split_file"] = _absolute(fold["split_file"], project_root)
    checkpoint = config.get("training", {}).get("checkpoint", {})
    if checkpoint.get("resume"):
        checkpoint["resume"] = _absolute(checkpoint["resume"], project_root)


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise TypeError(f"Top-level YAML value must be a mapping: {config_path}")
    base_reference = loaded.pop("_base_", None)
    if base_reference is not None:
        base_path = (config_path.parent / str(base_reference)).resolve()
        with base_path.open("r", encoding="utf-8") as stream:
            base_loaded = yaml.safe_load(stream)
        if not isinstance(base_loaded, dict):
            raise TypeError(f"Base YAML value must be a mapping: {base_path}")
        loaded = dict(deep_merge(base_loaded, loaded))
    config = _expand_environment(loaded)
    deep_merge(config, parse_overrides(overrides))
    project_root = _find_project_root(config_path)
    _resolve_paths(config, project_root)
    validate_config(config)
    config["_meta"] = {"config_path": str(config_path), "project_root": str(project_root)}
    return config


def _validate_classification(config: Mapping[str, Any]) -> None:
    model = config["model"]
    data = config["data"]
    labels = [str(value) for value in data.get("labels", [])]
    if len(labels) < 2 or len(labels) != len(set(labels)):
        raise ValueError("data.labels must contain at least two unique class names")
    sensor_mode = str(data.get("sensor_mode", "acc_gyro")).lower()
    if sensor_mode not in SENSOR_INDICES:
        raise ValueError(
            f"data.sensor_mode must be one of {sorted(SENSOR_INDICES)}"
        )
    expected_channels = len(SENSOR_INDICES[sensor_mode])
    if int(model["input_channels"]) != expected_channels:
        raise ValueError(
            "model.input_channels must match data.sensor_mode: "
            f"expected {expected_channels}, got {model['input_channels']}"
        )
    if int(data.get("raw_input_channels", 6)) != 6:
        raise ValueError("PADS raw_input_channels must be 6 (Acc XYZ + Gyro XYZ)")
    if int(model["num_classes"]) < 2:
        raise ValueError("model.num_classes must be at least 2")
    if len(labels) != int(model["num_classes"]):
        raise ValueError("len(data.labels) must equal model.num_classes")
    sequence_length = data.get("sequence_length")
    if sequence_length is not None and int(sequence_length) < 2:
        raise ValueError("data.sequence_length must be null or at least 2")
    if sequence_length is None:
        if str(data.get("train_crop", data.get("crop", "full"))) not in {
            "full", "none"
        }:
            raise ValueError("variable-length mode requires train_crop=full")
        if str(data.get("eval_crop", data.get("crop", "full"))) not in {
            "full", "none"
        }:
            raise ValueError("variable-length mode requires eval_crop=full")
    sample_rate = float(model["sample_rate"])
    if sample_rate <= 0:
        raise ValueError("model.sample_rate must be positive")
    if "sample_rate" in data and not abs(float(data["sample_rate"]) - sample_rate) < 1e-8:
        raise ValueError("data.sample_rate must equal model.sample_rate")
    model_name = str(model.get("name", "mfam")).lower()
    if model_name not in {
        "mfam",
        "subject_mfam",
        "multi_activity_mfam",
        "pure_deep_subject",
    }:
        raise ValueError(f"Unsupported classification model.name: {model_name!r}")
    if model_name in {"mfam", "subject_mfam", "multi_activity_mfam"}:
        bands = model["frequency"]["bands"]
        nyquist = sample_rate / 2.0
        for low, high in bands:
            if not (0 <= float(low) < float(high) <= nyquist):
                raise ValueError(
                    f"Invalid frequency band [{low}, {high}] for Nyquist {nyquist}"
                )
        retention = float(model["mil"]["retention_ratio"])
        overlap = float(model["mil"]["overlap"])
        if not 0 < retention <= 1 or not 0 <= overlap < 1:
            raise ValueError("Invalid MIL retention_ratio or overlap")
    wrist_mode = str(data.get("wrist_mode", "bilateral")).lower()
    if wrist_mode not in {"left", "right", "bilateral"}:
        raise ValueError(
            "data.wrist_mode must be left, right, or bilateral"
        )
    wrist_order = [str(value).lower() for value in data.get("wrist_order", WRIST_ORDER)]
    if wrist_order != list(WRIST_ORDER):
        raise ValueError("data.wrist_order must be [left, right]")
    if str(data.get("invalid_pair_policy", "error")) not in {"error", "exclude"}:
        raise ValueError("data.invalid_pair_policy must be error or exclude")
    unit = str(data.get("unit", "activity")).lower()
    if unit not in {"activity", "subject"}:
        raise ValueError("data.unit must be activity or subject")
    if unit == "subject":
        activities = [str(value) for value in data.get("activities", [])]
        if not activities or len(activities) != len(set(activities)):
            raise ValueError("subject mode requires unique data.activities")
        manifests = data.get("activity_manifests")
        if not isinstance(manifests, Mapping):
            raise ValueError("subject mode requires data.activity_manifests")
        if set(manifests) != set(activities):
            raise ValueError(
                "data.activity_manifests keys must exactly match data.activities"
            )
        if model_name not in {"subject_mfam", "multi_activity_mfam", "pure_deep_subject"}:
            raise ValueError("subject mode requires a subject-level model")
    elif model_name in {"subject_mfam", "multi_activity_mfam"}:
        raise ValueError("subject_mfam requires data.unit=subject")
    normalization_scope = str(
        data.get("normalization_scope", "wrist_channel")
    ).lower()
    if normalization_scope not in {"wrist_channel", "activity_wrist_channel"}:
        raise ValueError(
            "data.normalization_scope must be wrist_channel or activity_wrist_channel"
        )
    if normalization_scope == "activity_wrist_channel" and unit != "subject":
        raise ValueError("activity_wrist_channel normalization requires subject mode")
    fraction = float(data.get("fold", {}).get("validation_fraction", 0.0))
    if not 0 <= fraction < 1:
        raise ValueError("data.fold.validation_fraction must be in [0, 1)")


def require_pads_classification_config(config: Mapping[str, Any]) -> None:
    """拒绝把非 PADS 分类配置误传给训练、评估或推理入口。"""
    task_type = str(config.get("task", {}).get("type", "classification")).lower()
    if task_type not in {"classification", "pads_classification"}:
        raise ValueError(
            "This entry point only supports PADS classification; "
            f"received task.type={task_type!r}"
        )
    if str(config.get("data", {}).get("type", "")).lower() != "pads":
        raise ValueError(
            "This entry point only supports data.type='pads'."
        )


def validate_config(config: Mapping[str, Any]) -> None:
    for section in (
        "task",
        "experiment",
        "data",
        "model",
        "loss",
        "training",
        "evaluation",
    ):
        if section not in config:
            raise KeyError(f"Missing required config section: {section}")
    require_pads_classification_config(config)
    _validate_classification(config)


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    clean = copy.deepcopy(dict(config))
    clean.pop("_meta", None)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(clean, stream, sort_keys=False, allow_unicode=True)
