from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from src.engine.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_compatibility,
)
from src.models.subject_mfam import build_subject_mfam
from src.r2_stabilization.models import StabilizedR2
from src.r2_stabilization.training import runtime_config
from src.targeted_ablation.data import HandcraftedCache, fit_fold_statistical_transform
from src.targeted_ablation.models import TargetedSubjectMFAM
from src.utils.config import load_config
from src.utils.model_stats import count_parameters


CONFIG_PATH = Path("/home/zyt/MFAM/configs/pads_multi_activity_v3_nested_cv.yaml")


def config():
    return load_config(CONFIG_PATH)


def batch(length: int = 128):
    x = torch.randn(2, 11, 2, 6, length, dtype=torch.float32)
    wrist = torch.ones(2, 11, 2)
    activity = torch.ones(2, 11, dtype=torch.bool)
    lengths = torch.full((2, 11), length, dtype=torch.long)
    statistics = torch.randn(2, 32, dtype=torch.float32)
    return x, wrist, activity, lengths, statistics


def test_s1_matches_original_r2_forward_exactly():
    reference = TargetedSubjectMFAM(config(), ("R2",)).eval()
    stabilized = StabilizedR2(config(), "S1").eval()
    stabilized.core.base.load_state_dict(reference.base.state_dict())
    stabilized.r2_classifier.load_state_dict(reference.classifier.state_dict())
    values = batch()
    with torch.no_grad():
        expected = reference(*values)
        actual = stabilized(*values)
    torch.testing.assert_close(actual["logits"], expected["logits"], rtol=0, atol=0)
    torch.testing.assert_close(
        actual["probabilities"], expected["probabilities"], rtol=0, atol=0
    )
    assert actual["classifier_input"].shape == (2, 546)


def test_s1_fp32_forward_backward_has_no_amp_dependency():
    model = StabilizedR2(config(), "S1")
    output = model(*batch())
    assert output["logits"].dtype == torch.float32
    output["logits"].sum().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)


def test_s2_layernorm_fusion_shape_and_scale():
    model = StabilizedR2(config(), "S2").eval()
    with torch.no_grad():
        output = model(*batch())
    assert output["deep_embedding_fusion"].shape == (2, 514)
    assert output["statistical_embedding_fusion"].shape == (2, 32)
    assert output["classifier_input"].shape == (2, 546)
    torch.testing.assert_close(
        output["deep_embedding_fusion"].mean(dim=-1),
        torch.zeros(2),
        atol=1e-5,
        rtol=0,
    )
    torch.testing.assert_close(
        output["statistical_embedding_fusion"].mean(dim=-1),
        torch.zeros(2),
        atol=1e-5,
        rtol=0,
    )


def test_s3_gated_fusion_is_fixed_64d_and_gate_in_range():
    model = StabilizedR2(config(), "S3")
    output = model(*batch())
    assert output["deep_embedding_fusion"].shape == (2, 64)
    assert output["statistical_embedding_fusion"].shape == (2, 64)
    assert output["gate"].shape == (2, 64)
    assert output["fused_embedding"].shape == (2, 64)
    assert bool(((output["gate"] >= 0) & (output["gate"] <= 1)).all())
    output["logits"].sum().backward()
    assert model.gate_layer.weight.grad is not None


def test_branch_disabling_removes_exact_branch_contribution():
    values = batch()
    for variant in ("S1", "S2", "S3"):
        model = StabilizedR2(config(), variant).eval()
        model.set_branch_mode("deep_disabled")
        with torch.no_grad():
            deep_disabled = model(*values)
        assert torch.count_nonzero(deep_disabled["deep_embedding_fusion"]) == 0
        model.set_branch_mode("statistical_disabled")
        with torch.no_grad():
            statistical_disabled = model(*values)
        assert (
            torch.count_nonzero(statistical_disabled["statistical_embedding_fusion"])
            == 0
        )


def test_parameter_counts_are_deterministic_low_capacity():
    first = {
        variant: count_parameters(StabilizedR2(config(), variant))["total_parameters"]
        for variant in ("S1", "S2", "S3")
    }
    second = {
        variant: count_parameters(StabilizedR2(config(), variant))["total_parameters"]
        for variant in ("S1", "S2", "S3")
    }
    assert first == second
    assert first["S1"] == 123000
    assert first["S2"] > first["S1"]
    assert first["S3"] > first["S2"]


def test_scaler_and_pca_remain_train_only():
    rng = np.random.default_rng(9)
    subject_ids = tuple(f"{index:03d}" for index in range(40))
    matrix = rng.normal(size=(40, 4928))
    matrix[35:] += 1000.0
    cache = HandcraftedCache(
        matrix=matrix,
        subject_ids=subject_ids,
        labels=np.asarray([index % 2 for index in range(40)]),
        subject_to_index={subject: index for index, subject in enumerate(subject_ids)},
    )
    result = fit_fold_statistical_transform(
        cache, subject_ids[:35], subject_ids[35:], components=32, random_state=42
    )
    np.testing.assert_allclose(result.scaler.mean_, matrix[:35].mean(axis=0))
    assert result.metadata["validation_labels_used_for_fit"] is False
    assert result.metadata["outer_test_features_transformed"] is False
    assert result.metadata["feature_count_output"] == 32


def test_runtime_config_disables_amp_and_outer_access():
    runtime = runtime_config(
        deepcopy(config()),
        "S3",
        0,
        0,
        ["001"],
        ["002"],
        ["003"],
        "plan",
    )
    assert runtime["training"]["mixed_precision"] is False
    policy = runtime["r2_stabilization"]
    assert policy["precision"] == "FP32"
    assert policy["outer_test_loader_created"] is False
    assert policy["outer_test_signal_accessed"] is False
    assert policy["outer_test_predictions_accessed"] is False
    assert policy["outer_test_features_transformed"] is False


def test_checkpoint_provenance_roundtrip(tmp_path: Path):
    runtime = runtime_config(
        config(), "S2", 0, 0, ["001"], ["002"], ["003"], "plan"
    )
    model = StabilizedR2(runtime, "S2")
    mean = torch.zeros(11, 2, 6, 1)
    std = torch.ones(11, 2, 6, 1)
    path = tmp_path / "best.pt"
    save_checkpoint(
        path, model, None, None, None, 0, 0.5, runtime, ["PD", "DD"], mean, std
    )
    loaded = StabilizedR2(runtime, "S2")
    checkpoint = load_checkpoint(path, model=loaded)
    validate_checkpoint_compatibility(
        checkpoint,
        runtime,
        checkpoint_path=path,
        runtime_mean=mean,
        runtime_std=std,
    )
    assert path.with_suffix(".pt.provenance.json").is_file()


def test_original_m0_behavior_remains_unchanged():
    torch.manual_seed(4)
    original = build_subject_mfam(config()).eval()
    state = {name: value.detach().clone() for name, value in original.state_dict().items()}
    _ = StabilizedR2(config(), "S3")
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
