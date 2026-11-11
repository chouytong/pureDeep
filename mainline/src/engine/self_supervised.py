from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import nn
import torch.nn.functional as functional

from src.models.pure_deep import PureDeepSubjectModel, PureDeepWristEncoder
from src.utils.artifacts import append_jsonl, write_json
from src.utils.provenance import sha256_file


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sample_contiguous_span_mask(
    batch_size: int,
    length: int,
    mask_ratio: float,
    span_length: int,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Sample an exact-size union of non-overlapping contiguous blocks."""

    if batch_size < 1 or length < 2:
        raise ValueError("Mask sampling requires a positive batch and length >= 2")
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must be in (0, 1)")
    if span_length < 1:
        raise ValueError("span_length must be positive")
    target = max(1, min(length - 1, int(round(length * mask_ratio))))
    block = min(int(span_length), length)
    block_count = int(math.ceil(length / block))
    result = torch.zeros((batch_size, length), dtype=torch.bool, device=device)
    for sample_index in range(batch_size):
        remaining = target
        for block_index in torch.randperm(block_count, device=device).tolist():
            start = block_index * block
            stop = min(length, start + block, start + remaining)
            result[sample_index, start:stop] = True
            remaining -= stop - start
            if remaining == 0:
                break
        if remaining != 0:
            raise RuntimeError("Failed to construct the requested temporal mask")
    return result


class MaskedSequenceReconstructor(nn.Module):
    """Temporary decoder around the retained V8 wrist encoder."""

    def __init__(self, encoder: PureDeepWristEncoder, input_channels: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.input_channels = int(input_channels)
        reconstruction_channels = int(encoder.feature_dim)
        if encoder.moment_encoder is not None:
            reconstruction_channels += int(encoder.moment_encoder.feature_dim)
        hidden_channels = max(32, reconstruction_channels // 2)
        self.mask_token = nn.Parameter(torch.zeros(1, self.input_channels, 1))
        self.decoder = nn.Sequential(
            nn.Conv1d(reconstruction_channels, hidden_channels, 7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden_channels, self.input_channels, 1),
        )

    def forward(
        self, inputs: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if inputs.ndim != 3 or inputs.shape[1] != self.input_channels:
            raise ValueError("Expected inputs shaped [batch, channels, time]")
        if mask.shape != (inputs.shape[0], inputs.shape[-1]):
            raise ValueError("Mask must have shape [batch, time]")
        expanded_mask = mask.unsqueeze(1)
        masked_inputs = torch.where(expanded_mask, self.mask_token, inputs)
        encoded = self.encoder(masked_inputs)
        # The retained encoder stem has stride two. Nearest repetition restores
        # the input grid without CUDA's nondeterministic linear-upsample backward.
        restored = encoded["encoded_features"].repeat_interleave(2, dim=-1)
        maps = [restored[..., : inputs.shape[-1]]]
        if "moment_features" in encoded:
            maps.append(encoded["moment_features"])
        reconstruction = self.decoder(torch.cat(maps, dim=1))
        return {
            "reconstruction": reconstruction,
            "masked_inputs": masked_inputs,
            "mask": mask,
        }


def iter_valid_wrist_groups(
    batch: Mapping[str, Any], device: torch.device
) -> Iterable[torch.Tensor]:
    """Yield unpadded valid wrists grouped by their true activity length."""

    inputs = batch["x"].to(device, non_blocking=True)
    activity_mask = batch["activity_mask"].to(device, non_blocking=True).bool()
    wrist_mask = batch["wrist_mask"].to(device, non_blocking=True).bool()
    lengths = batch["activity_lengths"].to(device, non_blocking=True).long()
    batch_size, activity_count, wrist_count, channels, _ = inputs.shape
    flat_inputs = inputs.reshape(batch_size * activity_count, wrist_count, channels, -1)
    flat_wrist = wrist_mask.reshape(-1, wrist_count)
    flat_lengths = lengths.reshape(-1)
    valid_activities = activity_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    for length in torch.unique(flat_lengths.index_select(0, valid_activities)).tolist():
        activity_indices = valid_activities[
            flat_lengths.index_select(0, valid_activities) == int(length)
        ]
        selected = flat_inputs.index_select(0, activity_indices)[..., : int(length)]
        selected_wrist = flat_wrist.index_select(0, activity_indices).reshape(-1)
        wrists = selected.reshape(-1, channels, int(length))
        wrist_indices = selected_wrist.nonzero(as_tuple=False).squeeze(1)
        if wrist_indices.numel():
            yield wrists.index_select(0, wrist_indices)


def run_masked_reconstruction_pretraining(
    model: nn.Module,
    train_loader: Any,
    config: Mapping[str, Any],
    device: torch.device,
    stage_dir: Path,
    *,
    smoke: bool = False,
    resume: bool = False,
) -> dict[str, Any] | None:
    specification = dict(config.get("self_supervised", {}))
    if not bool(specification.get("enabled", False)):
        return None
    if not isinstance(model, PureDeepSubjectModel):
        raise TypeError("Masked reconstruction requires PureDeepSubjectModel")
    if not isinstance(model.wrist_encoder, PureDeepWristEncoder):
        raise TypeError("Masked reconstruction requires the time-domain wrist encoder")

    pretrain_dir = stage_dir / "pretraining"
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    status_path = pretrain_dir / "status.json"
    encoder_path = pretrain_dir / "wrist_encoder.pt"
    if resume and status_path.is_file() and encoder_path.is_file():
        status = torch.load(encoder_path, map_location=device, weights_only=False)
        model.wrist_encoder.load_state_dict(status["wrist_encoder_state_dict"])
        summary = dict(status["summary"])
        summary["loaded_from_completed_pretraining"] = True
        return summary

    epochs = 1 if smoke else int(specification.get("epochs", 10))
    learning_rate = float(specification.get("learning_rate", 1.0e-3))
    weight_decay = float(specification.get("weight_decay", 1.0e-4))
    mask_ratio = float(specification.get("mask_ratio", 0.3))
    span_length = int(specification.get("span_length", 25))
    gradient_clip = float(specification.get("gradient_clip_norm", 1.0))
    max_batches = 1 if smoke else int(specification.get("max_batches", 0))
    runtime = dict(config.get("nested_runtime", {}))
    audit = {
        "objective": "masked_sequence_reconstruction",
        "labels_used": False,
        "validation_subjects_used": False,
        "outer_test_loader_created": False,
        "outer_test_signal_accessed": False,
        "train_subject_ids_sha256": runtime.get("train_subject_ids_sha256"),
        "train_subject_count": runtime.get("train_subject_count"),
        "mask_ratio": mask_ratio,
        "span_length": span_length,
        "epochs": epochs,
    }
    write_json(status_path, {"status": "running", "started_at_utc": _utc_now(), **audit})

    pretrainer = MaskedSequenceReconstructor(
        model.wrist_encoder, int(model.input_channels)
    ).to(device)
    optimizer = torch.optim.AdamW(
        pretrainer.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    epoch_losses: list[float] = []
    for epoch in range(epochs):
        pretrainer.train()
        squared_error = 0.0
        masked_values = 0
        batch_count = 0
        for batch_index, batch in enumerate(train_loader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            optimizer.zero_grad(set_to_none=True)
            loss_sum: torch.Tensor | None = None
            value_count = 0
            for wrists in iter_valid_wrist_groups(batch, device):
                mask = sample_contiguous_span_mask(
                    wrists.shape[0], wrists.shape[-1], mask_ratio, span_length,
                    device=device,
                )
                output = pretrainer(wrists, mask)
                selected = mask.unsqueeze(1).expand_as(wrists)
                group_error = functional.mse_loss(
                    output["reconstruction"][selected], wrists[selected], reduction="sum"
                )
                loss_sum = group_error if loss_sum is None else loss_sum + group_error
                value_count += int(selected.sum().item())
            if loss_sum is None or value_count == 0:
                raise RuntimeError("Pretraining batch contained no valid masked values")
            loss = loss_sum / value_count
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite masked reconstruction loss")
            loss.backward()
            nn.utils.clip_grad_norm_(pretrainer.parameters(), gradient_clip)
            optimizer.step()
            squared_error += float(loss_sum.detach().item())
            masked_values += value_count
            batch_count += 1
        epoch_loss = squared_error / masked_values
        epoch_losses.append(epoch_loss)
        append_jsonl(pretrain_dir / "epochs.jsonl", {
            "epoch": epoch + 1,
            "masked_mse": epoch_loss,
            "batches": batch_count,
            "masked_values": masked_values,
            "timestamp_utc": _utc_now(),
        })
        print(
            f"[pretrain] epoch={epoch + 1}/{epochs} masked_mse={epoch_loss:.6f}",
            flush=True,
        )

    summary = {
        **audit,
        "initial_masked_mse": epoch_losses[0],
        "final_masked_mse": epoch_losses[-1],
        "epoch_losses": epoch_losses,
        "decoder_discarded_before_supervised_training": True,
        "supervised_optimizer_reinitialized": True,
    }
    torch.save({
        "wrist_encoder_state_dict": model.wrist_encoder.state_dict(),
        "summary": summary,
    }, encoder_path)
    summary["wrist_encoder_checkpoint_sha256"] = sha256_file(encoder_path)
    write_json(status_path, {
        "status": "complete", "completed_at_utc": _utc_now(), **summary
    })
    return summary
