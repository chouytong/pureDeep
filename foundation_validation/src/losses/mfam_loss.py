from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


class ClassificationLoss(nn.Module):
    """PADS 多类别交叉熵。

    ``outputs['logits']`` 的形状必须为 ``[batch, classes]``，``targets`` 为
    ``[batch]`` 的整数类别索引。类别权重只能来自配置，不能使用验证集或测试集
    标签进行估计，以免把测试分布泄漏到训练目标中。
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        class_weights: list[float] | None = None,
        prototype_alignment_weight: float = 0.0,
        prototype_margin: float = 0.2,
        prototype_separation_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        self.register_buffer(
            "class_weights",
            None
            if class_weights is None
            else torch.tensor(class_weights, dtype=torch.float32),
        )
        self.prototype_alignment_weight = float(prototype_alignment_weight)
        self.prototype_margin = float(prototype_margin)
        self.prototype_separation_weight = float(prototype_separation_weight)
        if self.prototype_alignment_weight < 0.0:
            raise ValueError("prototype_alignment_weight must be non-negative")
        if self.prototype_margin < 0.0:
            raise ValueError("prototype_margin must be non-negative")
        self.register_buffer("activity_disease_prototypes", None, persistent=False)

    @property
    def requires_activity_prototypes(self) -> bool:
        return self.prototype_alignment_weight > 0.0

    def set_activity_disease_prototypes(self, prototypes: torch.Tensor) -> None:
        if prototypes.ndim != 3 or prototypes.shape[1] != 2:
            raise ValueError("Expected activity x disease x feature prototypes")
        self.activity_disease_prototypes = prototypes.detach()

    def _prototype_loss(
        self,
        outputs: Mapping[str, torch.Tensor | None],
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings = outputs.get("activity_embeddings")
        if embeddings is None:
            raise ValueError("Prototype alignment requires activity_embeddings")
        prototypes = self.activity_disease_prototypes
        if prototypes is None:
            raise RuntimeError("Train-fold activity/disease prototypes were not set")
        if embeddings.ndim != 3 or prototypes.shape != (
            embeddings.shape[1], 2, embeddings.shape[2]
        ):
            raise ValueError("Activity embeddings and prototypes are incompatible")
        mask = outputs.get("activity_mask")
        if mask is None:
            mask = torch.ones(
                embeddings.shape[:2], dtype=torch.bool, device=embeddings.device
            )
        else:
            mask = mask.to(device=embeddings.device, dtype=torch.bool)
        normalized = F.normalize(embeddings, dim=-1)
        normalized_prototypes = F.normalize(
            prototypes.to(device=embeddings.device, dtype=embeddings.dtype), dim=-1
        )
        activity_index = torch.arange(embeddings.shape[1], device=embeddings.device)
        positive = normalized_prototypes[activity_index[None, :], targets[:, None]]
        negative = normalized_prototypes[
            activity_index[None, :], (1 - targets)[:, None]
        ]
        positive_distance = 1.0 - torch.sum(normalized * positive, dim=-1)
        negative_distance = 1.0 - torch.sum(normalized * negative, dim=-1)
        compactness = positive_distance[mask].mean()
        separation = F.relu(
            self.prototype_margin + positive_distance - negative_distance
        )[mask].mean()
        alignment = compactness + self.prototype_separation_weight * separation
        return alignment, compactness, separation

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor | None],
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        logits = outputs.get("logits")
        if logits is None:
            raise ValueError("Model outputs do not contain logits")
        loss = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )
        result = {"loss": loss, "classification_loss": loss}
        if self.training and self.requires_activity_prototypes:
            alignment, compactness, separation = self._prototype_loss(outputs, targets)
            result.update({
                "loss": loss + self.prototype_alignment_weight * alignment,
                "prototype_alignment_loss": alignment,
                "prototype_compactness_loss": compactness,
                "prototype_separation_loss": separation,
            })
        return result


def build_loss(config: Mapping[str, Any]) -> ClassificationLoss:
    loss_config = config["loss"]
    return ClassificationLoss(
        label_smoothing=float(loss_config.get("label_smoothing", 0.0)),
        class_weights=loss_config.get("class_weights"),
        prototype_alignment_weight=float(
            loss_config.get("prototype_alignment_weight", 0.0)
        ),
        prototype_margin=float(loss_config.get("prototype_margin", 0.2)),
        prototype_separation_weight=float(
            loss_config.get("prototype_separation_weight", 1.0)
        ),
    )
