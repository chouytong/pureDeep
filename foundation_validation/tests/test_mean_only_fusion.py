from __future__ import annotations

import copy

import torch

from src.models.pure_deep import PureDeepSubjectModel


def _config(mode: str) -> dict:
    return {
        "pure_deep": {
            "feature_dim": 8,
            "dropout": 0.0,
            "pooling": "attention_mean_std",
            "temporal": {
                "normalization": "group_norm",
                "normalization_groups": 4,
            },
        },
        "bilateral_fusion": {"mode": mode},
        "activity_fusion": {
            "mode": "attention",
            "attention_dim": 4,
            "dropout": 0.0,
            "classifier_dropout": 0.0,
        },
    }


def test_mean_only_activity_is_exact_masked_wrist_mean():
    torch.manual_seed(11)
    model = PureDeepSubjectModel(["a"], 6, 2, _config("mean_only")).eval()
    inputs = torch.randn(2, 1, 2, 6, 96)
    activity_mask = torch.ones(2, 1, dtype=torch.bool)
    lengths = torch.full((2, 1), 96, dtype=torch.long)
    wrist_mask = torch.tensor([[[1, 1]], [[1, 0]]], dtype=torch.float32)
    with torch.no_grad():
        activity = model._encode_activities(
            inputs, wrist_mask, activity_mask, lengths
        )
        left = model.wrist_encoder(inputs[:, 0, 0])["bag_embedding"]
        right = model.wrist_encoder(inputs[:1, 0, 1])["bag_embedding"]
    assert activity.shape == (2, 1, 8)
    assert torch.allclose(activity[0, 0], (left[0] + right[0]) / 2, atol=1e-6)
    assert torch.allclose(activity[1, 0], left[1], atol=1e-6)


def test_mean_only_changes_only_downstream_feature_dimension_and_is_smaller():
    full = PureDeepSubjectModel(["a", "b"], 6, 2, _config("full"))
    mean = PureDeepSubjectModel(["a", "b"], 6, 2, _config("mean_only"))
    assert full.wrist_encoder.feature_dim == mean.wrist_encoder.feature_dim == 8
    assert full.activity_feature_dim == 34
    assert mean.activity_feature_dim == 8
    assert sum(p.numel() for p in mean.parameters()) < sum(
        p.numel() for p in full.parameters()
    )
    full_wrist = copy.deepcopy(full.wrist_encoder.state_dict())
    mean.wrist_encoder.load_state_dict(full_wrist, strict=True)


def test_invalid_bilateral_fusion_mode_is_rejected():
    config = _config("unknown")
    try:
        PureDeepSubjectModel(["a"], 6, 2, config)
    except ValueError as error:
        assert "bilateral_fusion.mode" in str(error)
    else:
        raise AssertionError("invalid bilateral fusion mode was accepted")
