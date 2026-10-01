from copy import deepcopy
from pathlib import Path

import pytest
import torch

from src.engine.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_compatibility,
    validate_checkpoint_input,
)
from src.models import build_model
from src.utils.config import load_config


def _config() -> dict:
    config = deepcopy(load_config("configs/pads_multi_activity_v2.yaml"))
    config["data"]["activities"] = ["CrossArms", "Relaxed"]
    config["model"]["encoder"].update(
        {"branch_channels": 8, "hidden_dim": 16, "channel_reduction": 4}
    )
    config["model"]["mil"].update({"window_seconds": 0.5, "attention_dim": 8})
    return config


def test_checkpoint_round_trip_and_input_metadata(tmp_path: Path) -> None:
    config = _config()
    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "checkpoint.pt"
    mean, std = torch.zeros(2, 2, 6, 1), torch.ones(2, 2, 6, 1)
    save_checkpoint(path, model, optimizer, None, None, 2, 0.75, config,
                    ["PD", "DD"], mean, std)
    restored = build_model(config)
    checkpoint = load_checkpoint(path, model=restored)
    validate_checkpoint_input(checkpoint, config)
    assert checkpoint["format_version"] == 2
    assert checkpoint["normalization"]["mean"].shape == (2, 2, 6, 1)
    for expected, actual in zip(model.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)


def test_checkpoint_rejects_wrist_mode_mismatch(tmp_path: Path) -> None:
    config = _config()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, build_model(config), None, None, None, 0, 0.0, config,
                    ["PD", "DD"], torch.zeros(2, 2, 6, 1), torch.ones(2, 2, 6, 1))
    checkpoint = load_checkpoint(path)
    changed = deepcopy(config)
    changed["data"]["wrist_mode"] = "left"
    with pytest.raises(ValueError, match="metadata"):
        validate_checkpoint_input(checkpoint, changed)
def test_v3_checkpoint_writes_and_validates_strict_provenance(
    tmp_path: Path,
) -> None:
    config = deepcopy(load_config("configs/pads_multi_activity_v3_nested_cv.yaml"))
    config["model"]["encoder"].update(
        {"branch_channels": 8, "hidden_dim": 16, "channel_reduction": 4}
    )
    config["model"]["mil"].update({"window_seconds": 0.5, "attention_dim": 8})
    model = build_model(config)
    path = tmp_path / "v3.pt"
    mean = torch.zeros(11, 2, 6, 1)
    std = torch.ones(11, 2, 6, 1)
    save_checkpoint(
        path,
        model,
        None,
        None,
        None,
        1,
        0.5,
        config,
        ["PD", "DD"],
        mean,
        std,
    )
    checkpoint = load_checkpoint(path)
    assert checkpoint["format_version"] == 3
    assert path.with_suffix(".pt.provenance.json").is_file()
    validate_checkpoint_compatibility(
        checkpoint,
        config,
        checkpoint_path=path,
        runtime_mean=mean,
        runtime_std=std,
    )
