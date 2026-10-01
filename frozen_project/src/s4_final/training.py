from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import src.r2_stabilization.training as engine
from src.engine.checkpoint import load_checkpoint
from src.losses import build_loss
from src.s4_final.models import BRANCH_MODES, build_s4
from src.targeted_ablation.data import HandcraftedCache
from src.utils.model_stats import count_parameters


ACTIVATIONS = (
    "frequency_embedding_raw", "fullband_embedding_raw", "statistical_embedding_raw",
    "frequency_embedding_fusion", "fullband_embedding_fusion",
    "statistical_embedding_fusion", "frequency_fullband_cosine",
    "classifier_input", "logits",
)


def _runtime_config(original, base, variant, outer, inner, train, validation, test, plan):
    config = original(base, variant, outer, inner, train, validation, test, plan)
    config["experiment"]["name"] = "v3_s4_final_internal_ablation"
    config.pop("r2_stabilization", None)
    config["s4_final_internal_ablation"] = {
        "variant": "S4", "precision": "FP32", "plan_sha256": plan,
        "frequency_dim": 514, "fullband_dim": 514,
        "statistical_dim": 32, "classifier_input_dim": 1060,
        "outer_test_loader_created": False,
        "outer_test_signal_accessed": False,
        "outer_test_predictions_accessed": False,
        "outer_test_features_transformed": False,
    }
    config["nested_runtime"]["phase"] = "s4_final_internal_development_inner_training"
    return config


@torch.no_grad()
def _vectors(model, loader, device, mode: str) -> dict[str, np.ndarray]:
    model.eval()
    model.set_branch_mode(mode)
    logits, probabilities, cosines = [], [], []
    for batch in loader:
        tensors = engine._to_device(batch, device)
        outputs = engine._forward(model, tensors)
        logits.append(outputs["logits"].detach().float().cpu().numpy())
        probabilities.append(outputs["probabilities"].detach().float().cpu().numpy())
        cosines.append(outputs["frequency_fullband_cosine"].detach().float().cpu().numpy())
    model.set_branch_mode("both")
    return {
        "logits": np.concatenate(logits),
        "probabilities": np.concatenate(probabilities),
        "frequency_fullband_cosine": np.concatenate(cosines).reshape(-1),
    }


def _posthoc_diagnostics(
    base_config: Mapping[str, Any], outer: Mapping[str, Any], inner: Mapping[str, Any],
    stage_dir: Path, device: torch.device, subtype_by_subject: Mapping[str, str],
    plan_sha256: str, cache: HandcraftedCache,
) -> dict[str, Any]:
    config, _, _, validation_dataset, _ = engine._prepare_fold(
        base_config, "S4", outer, inner, stage_dir, cache, plan_sha256
    )
    loader = engine.subject_loader(
        validation_dataset,
        batch_size=int(config["evaluation"]["batch_size"]), shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
        pin_memory=bool(config["data"].get("pin_memory", False)), seed=43,
    )
    model = build_s4(config).to(device)
    load_checkpoint(stage_dir / "checkpoints/best.pt", model=model, map_location=device)
    criterion = build_loss(config).to(device)
    outer_index, inner_index = int(outer["outer_fold"]), int(inner["inner_fold"])
    baseline_metrics, _, baseline_diag = engine.evaluate(
        model, loader, criterion, device, subtype_by_subject,
        model_name="S4", outer_index=outer_index, inner_index=inner_index,
    )
    baseline_vectors = _vectors(model, loader, device, "both")
    result: dict[str, Any] = {
        "both": {"metrics": baseline_metrics, "diagnostics": baseline_diag},
        "frequency_fullband_cosine": {
            "mean": float(baseline_vectors["frequency_fullband_cosine"].mean()),
            "std": float(baseline_vectors["frequency_fullband_cosine"].std(ddof=1)),
            "median": float(np.median(baseline_vectors["frequency_fullband_cosine"])),
        },
        "statistical_pairwise_cosine": "not_defined_without_forbidden_projection_514_vs_32",
    }
    for mode in BRANCH_MODES[1:]:
        metrics, rows, diagnostics = engine.evaluate(
            model, loader, criterion, device, subtype_by_subject,
            model_name="S4", outer_index=outer_index, inner_index=inner_index,
            branch_mode=mode,
        )
        engine.write_csv(stage_dir / "predictions" / f"validation_{mode}.csv", rows)
        vectors = _vectors(model, loader, device, mode)
        result[mode] = {
            "metrics": metrics,
            "diagnostics": diagnostics,
            "delta_ba_from_both": float(metrics["balanced_accuracy"] - baseline_metrics["balanced_accuracy"]),
            "delta_auroc_from_both": float(metrics["macro_auroc"] - baseline_metrics["macro_auroc"]),
            "mean_absolute_probability_dd_change": float(
                np.abs(vectors["probabilities"][:, 1] - baseline_vectors["probabilities"][:, 1]).mean()
            ),
            "mean_logit_l2_change": float(
                np.linalg.norm(vectors["logits"] - baseline_vectors["logits"], axis=1).mean()
            ),
        }
    return result


def train_s4_fold(
    base_config: Mapping[str, Any], outer: Mapping[str, Any], inner: Mapping[str, Any],
    stage_dir: Path, device: torch.device, subtype_by_subject: Mapping[str, str],
    plan_sha256: str, cache: HandcraftedCache, *, resume: bool, smoke: bool,
) -> dict[str, Any]:
    originals = {
        "build": engine.build_stabilized_r2, "runtime": engine.runtime_config,
        "activations": engine.ACTIVATIONS, "modes": engine.BRANCH_MODES,
        "group": engine._parameter_group,
    }
    def group(name: str) -> str:
        if "full_band_encoder" in name or "full_band_mil" in name:
            return "fullband_branch"
        if "frequency_path" in name:
            return "frequency_branch"
        if name.startswith("classifier"):
            return "fusion_classifier"
        return originals["group"](name)
    engine.build_stabilized_r2 = lambda config, variant: build_s4(config)
    engine.runtime_config = lambda *args: _runtime_config(originals["runtime"], *args)
    engine.ACTIVATIONS = ACTIVATIONS
    engine.BRANCH_MODES = BRANCH_MODES
    engine._parameter_group = group
    try:
        summary = engine.train_fold(
            base_config, "S4", outer, inner, stage_dir, device,
            subtype_by_subject, plan_sha256, cache, resume=resume, smoke=smoke,
        )
        diagnostics = _posthoc_diagnostics(
            base_config, outer, inner, stage_dir, device, subtype_by_subject,
            plan_sha256, cache,
        )
        summary["branch_ablation"] = diagnostics
        model_info = {
            **summary["parameter_count"], "variant": "S4", "precision": "FP32",
            "frequency_embedding_dim": 514, "fullband_embedding_dim": 514,
            "statistical_embedding_dim": 32, "classifier_input_dim": 1060,
            "r1_branch_projection_used": False,
            "outer_test_loader_created": False,
        }
        engine.write_json(stage_dir / "model.json", model_info)
        engine.write_json(stage_dir / "fold_summary.json", summary)
        status = json.loads((stage_dir / "stage_status.json").read_text())
        status["summary"] = summary
        engine.write_json(stage_dir / "stage_status.json", status)
        return summary
    finally:
        engine.build_stabilized_r2 = originals["build"]
        engine.runtime_config = originals["runtime"]
        engine.ACTIVATIONS = originals["activations"]
        engine.BRANCH_MODES = originals["modes"]
        engine._parameter_group = originals["group"]


def run_s4(
    base_config: Mapping[str, Any], split: Mapping[str, Any], output: Path,
    device: torch.device, subtype_by_subject: Mapping[str, str], plan_sha256: str,
    cache: HandcraftedCache, *, resume: bool = True, smoke: bool = False,
) -> dict[str, Any]:
    summaries, failures = [], []
    for outer in split["outer"]:
        for inner in outer["inner_folds"]:
            o, i = int(outer["outer_fold"]), int(inner["inner_fold"])
            if smoke and (o, i) != (0, 0):
                continue
            stage = output / "models/S4" / f"outer_{o}" / f"inner_{i}"
            try:
                summaries.append(train_s4_fold(
                    base_config, outer, inner, stage, device, subtype_by_subject,
                    plan_sha256, cache, resume=resume, smoke=smoke,
                ))
            except Exception as error:
                failure = {"outer_context": o, "inner_fold": i, "error_type": type(error).__name__, "error": str(error)}
                failures.append(failure); engine.write_json(stage / "failure.json", failure)
                if smoke: raise
    all_rows = []
    for summary in summaries:
        path = output / "models/S4" / f"outer_{summary['outer_context']}" / f"inner_{summary['inner_fold']}" / "predictions/validation.csv"
        all_rows.extend(list(csv.DictReader(path.open())))
    if all_rows: engine.write_csv(output / "models/S4/development_predictions_all.csv", all_rows)
    result = {
        "status": "complete" if not failures and len(summaries) == (1 if smoke else 15) else "failed",
        "variant": "S4", "precision": "FP32", "fold_count": len(summaries),
        "failed_fold_count": len(failures), "failures": failures,
        "fold_metric_summary": engine.aggregate_fold_summaries(summaries) if summaries else {},
        "fold_summaries": summaries, "prediction_rows": len(all_rows),
        "unique_validation_subjects": len({row["subject_id"] for row in all_rows}),
        "plan_sha256": plan_sha256, "scope": "development_inner_cv_not_outer_test",
        "outer_test_accessed": False,
    }
    engine.write_json(output / "models/S4/development_summary.json", result)
    return result
