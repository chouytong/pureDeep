from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.engine.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_compatibility,
)
from src.models.mil import AttentionMIL
from src.models.subject_mfam import build_subject_mfam
from src.targeted_ablation.data import (
    HandcraftedCache,
    fit_fold_statistical_transform,
)
from src.targeted_ablation.models import (
    MaskedMeanActivityAggregator,
    TargetedSubjectMFAM,
)
from src.targeted_ablation.training import runtime_config
from src.utils.config import load_config
from src.utils.model_stats import count_parameters


CONFIG_PATH = Path("/home/zyt/MFAM/configs/pads_multi_activity_v3_nested_cv.yaml")


def config():
    return load_config(CONFIG_PATH)


def batch(length: int = 128):
    x = torch.randn(2, 11, 2, 6, length)
    wrist = torch.ones(2, 11, 2)
    activity = torch.ones(2, 11, dtype=torch.bool)
    lengths = torch.full((2, 11), length, dtype=torch.long)
    return x, wrist, activity, lengths


def test_targeted_m0_wrapper_matches_current_subject_mfam_exactly():
    torch.manual_seed(7)
    frozen = build_subject_mfam(config()).eval()
    targeted = TargetedSubjectMFAM(config(), []).eval()
    targeted.base.load_state_dict(frozen.state_dict())
    x, wrist, activity, lengths = batch()
    with torch.no_grad():
        expected = frozen(x, wrist, activity, lengths)
        actual = targeted(x, wrist, activity, lengths)
    torch.testing.assert_close(actual["logits"], expected["logits"], rtol=0, atol=0)
    torch.testing.assert_close(
        actual["probabilities"], expected["probabilities"], rtol=0, atol=0
    )


def test_r1_full_band_shape_parameter_count_and_backward():
    model = TargetedSubjectMFAM(config(), ["R1"])
    x, wrist, activity, lengths = batch()
    output = model(x, wrist, activity, lengths)
    assert output["logits"].shape == (2, 2)
    assert output["classifier_features"].shape == (2, 514)
    assert output["r1_frequency_bag_l2"].shape == (2, 11)
    assert output["r1_full_band_bag_l2"].shape == (2, 11)
    assert torch.isfinite(output["logits"]).all()
    output["logits"].sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    m0_count = count_parameters(TargetedSubjectMFAM(config(), []))["total_parameters"]
    r1_count = count_parameters(model)["total_parameters"]
    assert r1_count > m0_count


def test_r2_statistical_residual_shape_and_backward():
    model = TargetedSubjectMFAM(config(), ["R2"])
    x, wrist, activity, lengths = batch()
    statistics = torch.randn(2, 32)
    output = model(x, wrist, activity, lengths, statistics)
    assert output["classifier_features"].shape == (2, 546)
    assert output["logits"].shape == (2, 2)
    output["logits"].sum().backward()
    assert model.classifier[-1].in_features == 546


def test_a1_disables_every_hard_topk_path():
    model = TargetedSubjectMFAM(config(), ["A1"])
    mil_modules = [module for module in model.modules() if isinstance(module, AttentionMIL)]
    assert mil_modules and all(module.hard_gating is False for module in mil_modules)
    x, wrist, activity, lengths = batch(256)
    with torch.no_grad():
        output = model.eval()(x, wrist, activity, lengths)
    torch.testing.assert_close(
        output["mil_instance_utilization"],
        torch.ones_like(output["mil_instance_utilization"]),
    )


def test_a2_masked_mean_is_exact_and_parameter_free():
    module = MaskedMeanActivityAggregator()
    features = torch.tensor([[[1.0, 2.0], [3.0, 6.0], [100.0, 100.0]]])
    mask = torch.tensor([[True, True, False]])
    output = module(features, mask)
    torch.testing.assert_close(output["subject_embedding"], torch.tensor([[2.0, 4.0]]))
    assert sum(parameter.numel() for parameter in module.parameters()) == 0
    m0 = count_parameters(TargetedSubjectMFAM(config(), []))["total_parameters"]
    a2 = count_parameters(TargetedSubjectMFAM(config(), ["A2"]))["total_parameters"]
    assert a2 < m0


def test_n1_replaces_temporal_batchnorm_with_fixed_groupnorm():
    model = TargetedSubjectMFAM(config(), ["N1"])
    temporal = model.base.activity_encoder.wrist_encoder
    assert not any(isinstance(module, nn.BatchNorm1d) for module in temporal.modules())
    groupnorms = [module for module in temporal.modules() if isinstance(module, nn.GroupNorm)]
    assert groupnorms
    assert all(module.num_groups == 8 for module in groupnorms)
    assert model.groupnorm_replacements == len(groupnorms)
    x, wrist, activity, lengths = batch()
    assert model(x, wrist, activity, lengths)["logits"].shape == (2, 2)


def test_parameter_counts_are_deterministic_and_ordered():
    variants = {
        "M0": [],
        "R1": ["R1"],
        "R2": ["R2"],
        "A1": ["A1"],
        "A2": ["A2"],
        "N1": ["N1"],
    }
    first = {
        name: count_parameters(TargetedSubjectMFAM(config(), mods))["total_parameters"]
        for name, mods in variants.items()
    }
    second = {
        name: count_parameters(TargetedSubjectMFAM(config(), mods))["total_parameters"]
        for name, mods in variants.items()
    }
    assert first == second
    assert first["A1"] == first["M0"]
    assert first["N1"] == first["M0"]
    assert first["R1"] > first["R2"] > first["M0"] > first["A2"]


def test_r2_preprocessing_is_fit_on_train_subjects_only():
    rng = np.random.default_rng(3)
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
    assert set(result.transformed) == set(subject_ids)
    assert all(value.shape == (32,) for value in result.transformed.values())


def test_checkpoint_provenance_roundtrip(tmp_path: Path):
    base = config()
    runtime = runtime_config(
        base,
        "A1",
        0,
        0,
        ["001", "002"],
        ["003"],
        ["004"],
        "61dc8ef8923c7d8cadfee7615f5bd0247486a3ec619d14295bff2d29c45de448",
    )
    model = TargetedSubjectMFAM(runtime, ["A1"])
    mean = torch.zeros(11, 2, 6, 1)
    std = torch.ones(11, 2, 6, 1)
    path = tmp_path / "best.pt"
    save_checkpoint(path, model, None, None, None, 0, 0.5, runtime, ["PD", "DD"], mean, std)
    loaded = TargetedSubjectMFAM(runtime, ["A1"])
    checkpoint = load_checkpoint(path, model=loaded)
    validate_checkpoint_compatibility(
        checkpoint, runtime, checkpoint_path=path, runtime_mean=mean, runtime_std=std
    )
    assert path.with_suffix(".pt.provenance.json").is_file()


def test_runtime_config_explicitly_forbids_outer_test_access():
    runtime = runtime_config(
        deepcopy(config()),
        "N1",
        0,
        0,
        ["001"],
        ["002"],
        ["003"],
        "plan",
    )
    policy = runtime["targeted_ablation"]
    assert policy["outer_test_loader_created"] is False
    assert policy["outer_test_signal_accessed"] is False
    assert policy["outer_test_predictions_accessed"] is False
