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
    ) -> None:
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        self.register_buffer(
            "class_weights",
            None
            if class_weights is None
            else torch.tensor(class_weights, dtype=torch.float32),
        )

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
        return {"loss": loss, "classification_loss": loss}


def build_loss(config: Mapping[str, Any]) -> ClassificationLoss:
    loss_config = config["loss"]
    return ClassificationLoss(
        label_smoothing=float(loss_config.get("label_smoothing", 0.0)),
        class_weights=loss_config.get("class_weights"),
    )
