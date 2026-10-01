from copy import deepcopy

import pytest
import torch

from src.models import MFAM
from src.utils.config import load_config


def _config() -> dict:
    config = deepcopy(load_config("configs/pads_multi_activity_v2.yaml"))
    config["model"]["num_classes"] = 3
    config["model"]["sample_rate"] = 64.0
    config["model"]["encoder"].update(
        {"branch_channels": 8, "hidden_dim": 16, "channel_reduction": 4}
    )
    config["model"]["mil"].update({"window_seconds": 0.5, "attention_dim": 8})
    return config


def _model() -> MFAM:
    config = _config()["model"]
    return MFAM(
        input_channels=int(config["input_channels"]),
        num_classes=int(config["num_classes"]),
        sample_rate=float(config["sample_rate"]),
        frequency_config=config["frequency"],
        encoder_config=config["encoder"],
        mil_config=config["mil"],
        classifier_config={"dropout": 0.0},
    )


@pytest.mark.parametrize("mode,mask", [
    ("left", [[1.0, 0.0]] * 3),
    ("right", [[0.0, 1.0]] * 3),
    ("bilateral", [[1.0, 1.0]] * 3),
])
def test_mfam_three_wrist_modes_forward_backward(mode: str, mask: list[list[float]]) -> None:
    model = _model()
    inputs = torch.randn(3, 2, 6, 256, requires_grad=True)
    outputs = model(inputs, torch.tensor(mask))
    assert outputs["logits"].shape == (3, 3)
    assert outputs["wrist_embeddings"].shape == (3, 2, 16)
    assert outputs["fusion_features"].shape == (3, 66)
    outputs["logits"].sum().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    invalid_index = 1 if mode == "left" else 0 if mode == "right" else None
    if invalid_index is not None:
        assert torch.count_nonzero(inputs.grad[:, invalid_index]) == 0


def test_mfam_uses_one_shared_wrist_encoder() -> None:
    model = _model()
    assert hasattr(model, "wrist_encoder")
    assert not hasattr(model, "left_encoder")
    assert not hasattr(model, "right_encoder")
    parameter_ids = [id(parameter) for parameter in model.wrist_encoder.parameters()]
    assert len(parameter_ids) == len(set(parameter_ids))


def test_mfam_rejects_nonfinite_input() -> None:
    model = _model()
    inputs = torch.zeros(1, 2, 6, 256)
    inputs[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        model(inputs, torch.ones(1, 2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_mfam_cuda_forward_backward() -> None:
    model = _model().cuda()
    inputs = torch.randn(2, 2, 6, 256, device="cuda", requires_grad=True)
    outputs = model(inputs, torch.ones(2, 2, device="cuda"))
    outputs["logits"].sum().backward()
    assert inputs.grad is not None
