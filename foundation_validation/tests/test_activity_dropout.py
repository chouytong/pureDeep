import torch

from src.datasets.subject_activity import (
    rotate_acc_gyro,
    sample_activity_keep_mask,
    sample_wrist_rotation_matrices,
)


def test_activity_dropout_zero_keeps_everything():
    assert sample_activity_keep_mask(11, 0.0).tolist() == [True] * 11


def test_activity_dropout_always_keeps_at_least_one():
    torch.manual_seed(42)
    for _ in range(100):
        mask = sample_activity_keep_mask(11, 0.999)
        assert mask.dtype == torch.bool
        assert bool(mask.any())


def test_activity_dropout_rejects_invalid_probability():
    try:
        sample_activity_keep_mask(11, 1.0)
    except ValueError as error:
        assert "[0, 1)" in str(error)
    else:
        raise AssertionError("Expected invalid dropout probability to fail")


def test_rotation_is_orthogonal_and_preserves_vector_norms():
    torch.manual_seed(3)
    matrices = sample_wrist_rotation_matrices(2, 15.0)
    identity = torch.eye(3).expand(2, -1, -1)
    assert torch.allclose(
        torch.bmm(matrices.transpose(1, 2), matrices), identity, atol=1e-5
    )
    signal = torch.randn(2, 6, 50)
    rotated = rotate_acc_gyro(signal, matrices)
    assert torch.allclose(
        signal[:, :3].norm(dim=1), rotated[:, :3].norm(dim=1), atol=1e-5
    )
    assert torch.allclose(
        signal[:, 3:].norm(dim=1), rotated[:, 3:].norm(dim=1), atol=1e-5
    )
