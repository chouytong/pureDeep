import math

import torch

from src.models.mil import AttentionMIL


def test_mil_shape_attention_sum_and_topk_count() -> None:
    module = AttentionMIL(
        feature_dim=8,
        window_size=16,
        stride=8,
        retention_ratio=0.3,
    )
    features = torch.randn(3, 8, 128, requires_grad=True)
    output = module(features)
    instance_count = 15
    assert output["instances"].shape == (3, instance_count, 8)
    assert output["bag_embedding"].shape == (3, 8)
    assert torch.allclose(
        output["attention"].sum(dim=-1), torch.ones(3), atol=1e-6
    )
    assert torch.all(
        output["topk_mask"].sum(dim=-1) == math.ceil(0.3 * instance_count)
    )
    output["bag_embedding"].sum().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_mil_short_sequence_becomes_one_instance() -> None:
    module = AttentionMIL(feature_dim=4, window_size=64, stride=32)
    output = module(torch.randn(2, 4, 10))
    assert output["instances"].shape == (2, 1, 4)
    assert output["topk_mask"].all()
