import torch

from src.losses import ClassificationLoss


def test_classification_loss_forward_backward() -> None:
    criterion = ClassificationLoss(label_smoothing=0.05)
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    losses = criterion({"logits": logits}, torch.tensor([0, 1]))
    assert set(losses) == {"loss", "classification_loss"}
    assert torch.equal(losses["loss"], losses["classification_loss"])
    losses["loss"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_classification_loss_rejects_missing_logits() -> None:
    criterion = ClassificationLoss()
    try:
        criterion({}, torch.tensor([0]))
    except ValueError as error:
        assert "logits" in str(error)
    else:
        raise AssertionError("Missing logits must raise ValueError")
