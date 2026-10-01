from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.analysis.baselines import PREDICTION_FIELDS
from src.analysis.common import (
    binary_metrics,
    canonical_sha256,
    prediction_row,
    read_json,
    sha256_file,
    summarize_fold_metrics,
    write_csv,
    write_json,
)
from src.analysis.features import ACTIVITIES, WRISTS
from src.utils.seed import seed_everything


class ProcessedSubjectDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        processed_root: str | Path,
        subject_ids: Sequence[str],
        labels: Mapping[str, int],
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        self.root = Path(processed_root)
        self.subject_ids = [str(value) for value in subject_ids]
        self.labels = labels
        self.mean = mean.to(torch.float32)
        self.std = std.to(torch.float32)
        if self.mean.shape != (len(ACTIVITIES), 2, 6, 1):
            raise ValueError(f"Expected normalization [11,2,6,1], got {tuple(self.mean.shape)}")

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        subject = self.subject_ids[index]
        tensors: list[torch.Tensor] = []
        lengths: list[int] = []
        for activity_index, activity in enumerate(ACTIVITIES):
            wrists = []
            for wrist in WRISTS:
                signal = np.load(
                    self.root / "signals" / activity / subject / f"{wrist}.npy",
                    allow_pickle=False,
                )
                wrists.append(torch.from_numpy(signal).to(torch.float32))
            stacked = torch.stack(wrists)
            stacked = (stacked - self.mean[activity_index]) / self.std[activity_index]
            tensors.append(stacked)
            lengths.append(int(stacked.shape[-1]))
        maximum = max(lengths)
        padded = [torch.nn.functional.pad(value, (0, maximum - value.shape[-1])) for value in tensors]
        return {
            "x": torch.stack(padded),
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "activity_mask": torch.ones(len(ACTIVITIES), dtype=torch.bool),
            "y": torch.tensor(int(self.labels[subject]), dtype=torch.long),
            "subject_id": subject,
        }


class SimpleSubjectCNN(nn.Module):
    """Shared Conv1D + masked global mean; no MFAM-specific mechanisms."""

    def __init__(self, input_channels: int = 6, hidden_channels: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, hidden_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, 2),
        )

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor, activity_mask: torch.Tensor
    ) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,A,W,C,T], got {tuple(x.shape)}")
        batch, activities, wrists, channels, timepoints = x.shape
        flat = x.reshape(batch * activities * wrists, channels, timepoints)
        encoded = self.encoder(flat)
        output_lengths = torch.div(lengths + 3, 4, rounding_mode="floor")
        output_lengths = output_lengths.unsqueeze(-1).expand(-1, -1, wrists).reshape(-1)
        positions = torch.arange(encoded.shape[-1], device=encoded.device).unsqueeze(0)
        time_mask = positions < output_lengths.to(encoded.device).unsqueeze(1)
        pooled = (encoded * time_mask.unsqueeze(1)).sum(dim=-1) / output_lengths.clamp_min(1).to(encoded.device).unsqueeze(1)
        pooled = pooled.reshape(batch, activities, wrists, -1).reshape(batch, activities, -1)
        mask = activity_mask.to(encoded.device).unsqueeze(-1)
        subject = (pooled * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.classifier(subject)


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, Any], list[str], np.ndarray, np.ndarray]:
    model.eval()
    subjects: list[str] = []
    targets: list[int] = []
    probabilities: list[float] = []
    for batch in loader:
        logits = model(batch["x"].to(device), batch["lengths"].to(device), batch["activity_mask"].to(device))
        pdd = torch.softmax(logits, dim=-1)[:, 1]
        subjects.extend([str(value) for value in batch["subject_id"]])
        targets.extend(batch["y"].numpy().tolist())
        probabilities.extend(pdd.cpu().numpy().tolist())
    y = np.asarray(targets, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    pred = (p >= 0.5).astype(np.int64)
    return binary_metrics(y, pred, probability_dd=p), subjects, y, p


def run_simple_cnn(
    *,
    processed_root: str | Path,
    labels_by_subject: Mapping[str, int],
    split: Mapping[str, Any],
    conditions: Mapping[str, str],
    frozen_dir: Path,
    output_dir: Path,
    device: str = "cuda",
) -> dict[str, Any]:
    candidates = (
        ("N1a_simple_cnn_lr1e4_drop0p2", 1e-4, 0.2),
        ("N1b_simple_cnn_lr2e4_drop0p4", 2e-4, 0.4),
    )
    torch_device = torch.device(device)
    summaries: dict[str, Any] = {}
    for model_name, learning_rate, dropout in candidates:
        fold_records: list[dict[str, Any]] = []
        all_predictions: list[dict[str, Any]] = []
        for outer_def in split["outer"]:
            outer = int(outer_def["outer_fold"])
            for inner_def in outer_def["inner_folds"]:
                inner = int(inner_def["inner_fold"])
                stage = output_dir / model_name / f"outer_{outer}" / f"inner_{inner}"
                stage.mkdir(parents=True, exist_ok=False)
                train_subjects = [str(value) for value in inner_def["train_subjects"]]
                validation_subjects = [str(value) for value in inner_def["validation_subjects"]]
                normalization_path = frozen_dir / f"outer_{outer}" / f"inner_{inner}" / "normalization.json"
                normalization = read_json(normalization_path)
                if set(normalization["subject_ids"]) != set(train_subjects):
                    raise ValueError("Frozen train-only normalization does not match inner train subjects")
                mean = torch.tensor(normalization["mean"], dtype=torch.float32)
                std = torch.tensor(normalization["std"], dtype=torch.float32)
                train_dataset = ProcessedSubjectDataset(processed_root, train_subjects, labels_by_subject, mean, std)
                validation_dataset = ProcessedSubjectDataset(processed_root, validation_subjects, labels_by_subject, mean, std)
                generator = torch.Generator().manual_seed(42)
                train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, generator=generator, num_workers=0)
                validation_loader = DataLoader(validation_dataset, batch_size=8, shuffle=False, num_workers=0)
                seed_everything(42)
                model = SimpleSubjectCNN(dropout=dropout).to(torch_device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
                best_metric = -np.inf; best_epoch = 0; patience = 0; best_state = None; history = []
                started = time.perf_counter()
                for epoch in range(1, 31):
                    model.train(); losses = []
                    for batch in train_loader:
                        optimizer.zero_grad(set_to_none=True)
                        logits = model(batch["x"].to(torch_device), batch["lengths"].to(torch_device), batch["activity_mask"].to(torch_device))
                        loss = torch.nn.functional.cross_entropy(logits, batch["y"].to(torch_device))
                        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
                        losses.append(float(loss.detach().cpu()))
                    validation_metrics, _, _, _ = _evaluate(model, validation_loader, torch_device)
                    value = float(validation_metrics["balanced_accuracy"])
                    improved = value > best_metric
                    if improved:
                        best_metric = value; best_epoch = epoch; patience = 0
                        best_state = copy.deepcopy(model.state_dict())
                    else:
                        patience += 1
                    history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": validation_metrics, "improved": improved, "patience": patience})
                    if patience >= 8:
                        break
                if best_state is None:
                    raise RuntimeError("Simple CNN did not produce a checkpoint")
                model.load_state_dict(best_state)
                metrics, validation_order, targets, probability = _evaluate(model, validation_loader, torch_device)
                runtime = time.perf_counter() - started
                predictions = [
                    prediction_row(
                        subject_id=subject, condition=conditions[subject], target=int(target),
                        prediction=int(pdd >= 0.5), outer_context=outer, inner_fold=inner,
                        probability_dd=float(pdd), decision_score=None, threshold=0.5,
                    )
                    for subject, target, pdd in zip(validation_order, targets, probability)
                ]
                checkpoint_path = stage / "best.pt"
                torch.save({"model_state": best_state, "config": {"learning_rate": learning_rate, "dropout": dropout}, "normalization_sha256": normalization["normalization_sha256"]}, checkpoint_path)
                metadata = {
                    "status": "complete", "scope": "development_inner_cv", "outer_test_accessed": False,
                    "model_name": model_name, "representation": "shared simple Conv1D, masked temporal mean, wrist concatenate, activity masked mean",
                    "forbidden_mfam_components_present": False, "outer_context": outer, "inner_fold": inner,
                    "train_subject_count": len(train_subjects), "validation_subject_count": len(validation_subjects),
                    "train_subject_ids_sha256": canonical_sha256(train_subjects), "validation_subject_ids_sha256": canonical_sha256(validation_subjects),
                    "normalization_sha256": normalization["normalization_sha256"], "seed": 42,
                    "parameter_count": parameter_count(model), "best_epoch": best_epoch, "epochs_run": len(history),
                    "runtime_seconds": runtime, "checkpoint_sha256": sha256_file(checkpoint_path), "metrics": metrics,
                }
                write_json(stage / "config.json", {"learning_rate": learning_rate, "dropout": dropout, "weight_decay": 1e-4, "batch_size": 8, "max_epochs": 30, "early_stopping_patience": 8, "candidate_grid_fixed_before_run": True})
                write_json(stage / "history.json", history); write_json(stage / "metrics.json", metrics); write_json(stage / "metadata.json", metadata)
                write_csv(stage / "predictions.csv", predictions, PREDICTION_FIELDS)
                fold_records.append(metadata); all_predictions.extend(predictions)
        y = np.asarray([row["binary_label"] for row in all_predictions]); p = np.asarray([row["probability_dd"] for row in all_predictions]); pred = (p >= 0.5).astype(int)
        summary = {
            "model_name": model_name, "scope": "development_inner_cv_estimate_not_outer_test", "fold_count": 15,
            "fold_metric_summary": summarize_fold_metrics([row["metrics"] for row in fold_records]),
            "pooled_repeated_validation_metrics": binary_metrics(y, pred, probability_dd=p),
            "parameter_count": fold_records[0]["parameter_count"], "outer_test_accessed": False,
        }
        write_csv(output_dir / model_name / "development_predictions_all.csv", all_predictions, PREDICTION_FIELDS)
        write_json(output_dir / model_name / "development_summary.json", summary)
        summaries[model_name] = summary
    return summaries
