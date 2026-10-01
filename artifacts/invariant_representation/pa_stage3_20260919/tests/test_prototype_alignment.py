from __future__ import annotations

import torch

from src.losses.mfam_loss import ClassificationLoss


def test_activity_conditioned_prototype_loss_is_train_only_and_differentiable():
    criterion = ClassificationLoss(
        prototype_alignment_weight=0.05,
        prototype_margin=0.2,
    )
    prototypes = torch.randn(3, 2, 8)
    criterion.set_activity_disease_prototypes(prototypes)
    embeddings = torch.randn(4, 3, 8, requires_grad=True)
    outputs = {
        "logits": torch.randn(4, 2, requires_grad=True),
        "activity_embeddings": embeddings,
        "activity_mask": torch.ones(4, 3, dtype=torch.bool),
    }
    targets = torch.tensor([0, 1, 0, 1])
    criterion.train()
    losses = criterion(outputs, targets)
    assert set(losses) == {
        "loss", "classification_loss", "prototype_alignment_loss",
        "prototype_compactness_loss", "prototype_separation_loss",
    }
    losses["loss"].backward()
    assert embeddings.grad is not None
    criterion.eval()
    evaluation_losses = criterion(outputs, targets)
    assert set(evaluation_losses) == {"loss", "classification_loss"}


def test_zero_lambda_matches_classification_only():
    criterion = ClassificationLoss(prototype_alignment_weight=0.0)
    outputs = {"logits": torch.randn(5, 2)}
    losses = criterion(outputs, torch.tensor([0, 1, 0, 1, 0]))
    assert torch.equal(losses["loss"], losses["classification_loss"])
