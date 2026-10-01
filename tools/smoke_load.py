#!/usr/bin/env python3
"""Load all frozen outer models and run one CPU forward pass; never trains."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from src.engine.checkpoint import (  # noqa: E402
    input_metadata_from_config,
    load_checkpoint,
)
from src.models import build_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def without_meta(config: dict) -> dict:
    clean = copy.deepcopy(config)
    clean.pop("_meta", None)
    return clean


@torch.no_grad()
def main() -> None:
    results = []
    for fold in range(5):
        model_dir = ROOT / f"models/outer_{fold}"
        config = load_config(model_dir / "config.yaml")
        checkpoint = load_checkpoint(model_dir / "final.pt", map_location="cpu")
        if checkpoint["format_version"] != 3:
            raise RuntimeError(f"outer_{fold}: expected checkpoint format v3")
        if without_meta(config) != checkpoint["config"]:
            raise RuntimeError(f"outer_{fold}: checkpoint/config mismatch")
        if checkpoint["input_metadata"] != input_metadata_from_config(config):
            raise RuntimeError(f"outer_{fold}: input metadata mismatch")
        if checkpoint["normalization"].get("fitted_on") != "train_subjects_only":
            raise RuntimeError(f"outer_{fold}: invalid normalization provenance")

        model = build_model(config).cpu().eval()
        load_checkpoint(model_dir / "final.pt", model=model, map_location="cpu")
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != 71026:
            raise RuntimeError(
                f"outer_{fold}: expected 71026 parameters, got {parameter_count}"
            )

        activities = len(config["data"]["activities"])
        signal = torch.zeros(1, activities, 2, 6, 128)
        wrist_mask = torch.ones(1, activities, 2)
        activity_mask = torch.ones(1, activities, dtype=torch.bool)
        lengths = torch.full((1, activities), 128, dtype=torch.long)
        output = model(signal, wrist_mask, activity_mask, lengths)
        probabilities = output["probabilities"]
        if probabilities.shape != (1, 2) or not torch.isfinite(probabilities).all():
            raise RuntimeError(f"outer_{fold}: invalid smoke output")
        if not torch.allclose(probabilities.sum(dim=-1), torch.ones(1), atol=1e-6):
            raise RuntimeError(f"outer_{fold}: probabilities do not sum to one")
        results.append(
            {
                "outer_fold": fold,
                "epoch": checkpoint["epoch"] + 1,
                "parameters": parameter_count,
                "output_shape": list(probabilities.shape),
                "status": "PASS",
            }
        )
    print(json.dumps({"status": "PASS", "models": results}, indent=2))


if __name__ == "__main__":
    main()
