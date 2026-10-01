from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import src.r2_stabilization.training as engine
from src.engine.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_compatibility,
)
from src.s4_final.models import S4SubjectMFAM
from src.s4_final.training import _runtime_config
from src.targeted_ablation.data import HandcraftedCache, fit_fold_statistical_transform
from src.targeted_ablation.models import FullBandResidualWristEncoder, TargetedSubjectMFAM
from src.utils.config import load_config
from src.utils.provenance import sha256_file


ROOT = Path("/home/zyt/MFAM")
CONFIG_PATH = ROOT / "configs/pads_multi_activity_v3_nested_cv.yaml"


def config():
    return load_config(CONFIG_PATH)


def batch(length: int = 128):
    return (
        torch.randn(2, 11, 2, 6, length, dtype=torch.float32),
        torch.ones(2, 11, 2),
        torch.ones(2, 11, dtype=torch.bool),
        torch.full((2, 11), length, dtype=torch.long),
        torch.randn(2, 32, dtype=torch.float32),
    )


def test_s4_forward_shape_and_fp32_backward():
    model = S4SubjectMFAM(config())
    output = model(*batch())
    assert output["logits"].shape == (2, 2)
    assert output["frequency_embedding_raw"].shape == (2, 514)
    assert output["fullband_embedding_raw"].shape == (2, 514)
    assert output["statistical_embedding_raw"].shape == (2, 32)
    assert output["classifier_input"].shape == (2, 1060)
    assert output["logits"].dtype == torch.float32
    output["logits"].sum().backward()
    gradients = [value.grad for value in model.parameters() if value.grad is not None]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)


def test_s4_fullband_definition_is_exact_r1_module_definition():
    s4 = S4SubjectMFAM(config())
    r1 = TargetedSubjectMFAM(config(), ("R1",))
    s4_wrist = s4.core.base.activity_encoder.wrist_encoder
    r1_wrist = r1.base.activity_encoder.wrist_encoder
    assert isinstance(s4_wrist, FullBandResidualWristEncoder)
    assert isinstance(r1_wrist, FullBandResidualWristEncoder)
    for component in ("full_band_encoder", "full_band_mil"):
        s4_shapes = [(name, tuple(value.shape)) for name, value in getattr(s4_wrist, component).state_dict().items()]
        r1_shapes = [(name, tuple(value.shape)) for name, value in getattr(r1_wrist, component).state_dict().items()]
        assert s4_shapes == r1_shapes
    assert isinstance(s4_wrist.branch_projection, torch.nn.Identity)


def test_s4_branch_disable_is_inference_masking_without_retraining():
    model = S4SubjectMFAM(config()).eval()
    values = batch()
    with torch.no_grad():
        model.set_branch_mode("frequency_disabled")
        frequency = model(*values)
        model.set_branch_mode("fullband_disabled")
        fullband = model(*values)
        model.set_branch_mode("statistical_disabled")
        statistical = model(*values)
    assert torch.count_nonzero(frequency["frequency_embedding_fusion"]) == 0
    assert torch.count_nonzero(fullband["fullband_embedding_fusion"]) == 0
    assert torch.count_nonzero(statistical["statistical_embedding_fusion"]) == 0
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_s4_pca32_remains_inner_train_only():
    rng = np.random.default_rng(11)
    subjects = tuple(f"{index:03d}" for index in range(40))
    matrix = rng.normal(size=(40, 4928))
    matrix[35:] += 1000.0
    cache = HandcraftedCache(
        matrix=matrix, subject_ids=subjects,
        labels=np.asarray([index % 2 for index in range(40)]),
        subject_to_index={subject: index for index, subject in enumerate(subjects)},
    )
    result = fit_fold_statistical_transform(
        cache, subjects[:35], subjects[35:], components=32, random_state=42
    )
    np.testing.assert_allclose(result.scaler.mean_, matrix[:35].mean(axis=0))
    assert result.metadata["validation_labels_used_for_fit"] is False
    assert result.metadata["outer_test_features_transformed"] is False
    assert result.metadata["feature_count_input"] == 4928
    assert result.metadata["feature_count_output"] == 32


def test_s4_runtime_and_checkpoint_provenance_forbid_outer_access(tmp_path: Path):
    runtime = _runtime_config(
        engine.runtime_config, config(), "S4", 0, 0,
        ["001"], ["002"], ["003"], "plan",
    )
    assert runtime["training"]["mixed_precision"] is False
    policy = runtime["s4_final_internal_ablation"]
    assert policy["precision"] == "FP32"
    assert policy["outer_test_loader_created"] is False
    assert policy["outer_test_signal_accessed"] is False
    assert policy["outer_test_predictions_accessed"] is False
    assert policy["outer_test_features_transformed"] is False
    model = S4SubjectMFAM(runtime)
    mean, std = torch.zeros(11, 2, 6, 1), torch.ones(11, 2, 6, 1)
    path = tmp_path / "best.pt"
    save_checkpoint(path, model, None, None, None, 0, 0.5, runtime, ["PD", "DD"], mean, std)
    loaded = S4SubjectMFAM(runtime)
    checkpoint = load_checkpoint(path, model=loaded)
    validate_checkpoint_compatibility(
        checkpoint, runtime, checkpoint_path=path, runtime_mean=mean, runtime_std=std
    )
    assert path.with_suffix(".pt.provenance.json").is_file()


def test_frozen_m0_and_s1_integrity():
    m0 = ROOT / "outputs/pads_classification/v3_analysis_and_baselines/analysis_and_baselines_20260828/baselines/M0_subject_mfam_frozen_inner_reference/development_predictions_all.csv"
    s1 = ROOT / "outputs/pads_classification/v3_r2_stabilization/stabilization_20260831/models/S1/development_predictions_all.csv"
    assert sha256_file(m0) == "e71d98a0487a02c92ef0f5302117af072b74d88686cab895768abb842e020fdb"
    assert sha256_file(s1) == "7cc98528c7b10781cab4c65624f21053f051373da409bdd8ee649ae4371e5509"
