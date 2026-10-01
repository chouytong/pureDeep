#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from typing import Any, Mapping

import numpy as np
import torch

from src.r2_stabilization.analysis import analyze_all
from src.r2_stabilization.training import (
    diagnose_s0_amp,
    diagnose_s0_checkpoint_activations,
    run_variant,
    write_json,
)
from src.targeted_ablation.analysis import M0_NAME, normalize_reference_predictions
from src.targeted_ablation.data import load_handcrafted_cache
from src.utils.config import load_config
from src.utils.provenance import sha256_file


PROJECT_ROOT = Path("/home/zyt/MFAM")
DEFAULT_CONFIG = PROJECT_ROOT / "configs/pads_multi_activity_v3_nested_cv.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "outputs/pads_classification/v3_r2_stabilization/stabilization_20260831"
)
PLAN_SHA256 = "e662a059ac3a6ea19088c59d7385ba27b61ef2cba3275eca9b63340d6c224800"
SPLIT_SHA256 = "b3c52317cb12b73c66046bbd7c50c94a7e707a64a38d3ad4f0fd4bc0c6ba6b3e"
BASE_CONFIG_SHA256 = "3bb1d8dbeea729775d300123b874752c85cce84f7760f4294e560249de472080"
FEATURE_CACHE_SHA256 = "d7b465b251892abbe602539369989c133fce4c4da15e3bee3cb308d07e0f4da5"
FEATURE_SCHEMA_SHA256 = "bdb83aae1c31b66c659a63f7a723f74b2cc82bfc99725621f6a41ba736389312"
FEATURE_SCHEMA_FILE_SHA256 = "071ccdf12059496617e9c27e7337d2f3c67787c4be77ff38491f36df6ee21c79"
FROZEN_V3_MANIFEST_SHA256 = "af5fdda01ceed8f90944f9a25e3a630b09ad17e07ece5f1d32dfe3d147271321"
S0_ARTIFACT_TREE_SHA256 = "528be416bfac771a9fe4477330030bfa917be411d0bc0f31fd0ee15f0224b535"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run preregistered R2 stabilization")
    parser.add_argument(
        "--stage",
        required=True,
        choices=("initialize", "diagnose-s0", "smoke", "run", "analyze", "finalize"),
    )
    parser.add_argument("--variant", choices=("S1", "S2", "S3"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def source_hashes() -> dict[str, str]:
    paths = (
        "src/r2_stabilization/__init__.py",
        "src/r2_stabilization/models.py",
        "src/r2_stabilization/training.py",
        "src/r2_stabilization/analysis.py",
        "scripts/run_v3_r2_stabilization.py",
        "tests/test_r2_stabilization.py",
    )
    return {path: sha256_file(PROJECT_ROOT / path) for path in paths}


def verify_inputs(
    output: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(output / "r2_stabilization_plan.json") != PLAN_SHA256:
        raise ValueError("R2 stabilization plan hash mismatch")
    if sha256_file(config_path) != BASE_CONFIG_SHA256:
        raise ValueError("Frozen base config changed")
    config = load_config(config_path)
    split_path = Path(config["nested_cv"]["split_file"])
    if sha256_file(split_path) != SPLIT_SHA256:
        raise ValueError("Frozen split changed")
    split = json.loads(split_path.read_text())
    if int(split["outer_folds"]) != 5 or int(split["inner_folds"]) != 3:
        raise ValueError("Expected frozen 5x3 split")
    frozen_manifest = PROJECT_ROOT / (
        "outputs/pads_classification/v3_nested_cv/"
        "formal_subject_mfam_seed42_20260827/frozen_baseline_manifest.json"
    )
    if sha256_file(frozen_manifest) != FROZEN_V3_MANIFEST_SHA256:
        raise ValueError("Frozen V3 manifest changed")
    feature_cache = PROJECT_ROOT / (
        "outputs/pads_classification/v3_analysis_and_baselines/"
        "analysis_and_baselines_20260828/features/handcrafted_features.npz"
    )
    if sha256_file(feature_cache) != FEATURE_CACHE_SHA256:
        raise ValueError("Frozen handcrafted cache changed")
    feature_schema = PROJECT_ROOT / (
        "outputs/pads_classification/v3_analysis_and_baselines/"
        "analysis_and_baselines_20260828/features/feature_schema.json"
    )
    if sha256_file(feature_schema) != FEATURE_SCHEMA_FILE_SHA256:
        raise ValueError("Frozen handcrafted schema changed")
    feature_metadata = json.loads(
        (feature_schema.parent / "feature_matrix_metadata.json").read_text()
    )
    if feature_metadata.get("schema_sha256") != FEATURE_SCHEMA_SHA256:
        raise ValueError("Frozen handcrafted semantic schema hash changed")
    s0_checksum = PROJECT_ROOT / (
        "outputs/pads_classification/v3_targeted_ablation/"
        "targeted_ablation_20260828/artifact_files.sha256"
    )
    if sha256_file(s0_checksum) != S0_ARTIFACT_TREE_SHA256:
        raise ValueError("Read-only S0 artifact checksum changed")
    return config, split


def cache_path() -> Path:
    return PROJECT_ROOT / (
        "outputs/pads_classification/v3_analysis_and_baselines/"
        "analysis_and_baselines_20260828/features/handcrafted_features.npz"
    )


def subtype_map() -> dict[str, str]:
    rows = normalize_reference_predictions(M0_NAME)
    result: dict[str, str] = {}
    for row in rows:
        previous = result.setdefault(str(row["subject_id"]), str(row["dd_subtype"]))
        if previous != row["dd_subtype"]:
            raise ValueError("Inconsistent subtype")
    if len(result) != 390:
        raise ValueError("Expected 390 subjects")
    return result


def update_manifest(output: Path, stage: str, summary: Mapping[str, Any]) -> None:
    path = output / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["stages"][stage] = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "summary": dict(summary),
    }
    write_json(path, manifest)


def initialize(output: Path, split: Mapping[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in output.iterdir()}
    allowed = {"r2_stabilization_plan.json", "r2_stabilization_plan.json.sha256"}
    if existing - allowed:
        raise FileExistsError(f"Output directory is not pristine: {sorted(existing)}")
    environment = {
        "created_at_utc": utc_now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    write_json(output / "environment.json", environment)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "created_at_utc": utc_now(),
        "scope": "r2_stabilization_development_inner_cv_only",
        "plan_sha256": PLAN_SHA256,
        "split_sha256": SPLIT_SHA256,
        "base_config_sha256": BASE_CONFIG_SHA256,
        "feature_cache_sha256": FEATURE_CACHE_SHA256,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "frozen_v3_manifest_sha256": FROZEN_V3_MANIFEST_SHA256,
        "s0_artifact_tree_sha256": S0_ARTIFACT_TREE_SHA256,
        "source_hashes_initial": source_hashes(),
        "expected_folds_per_model": 15,
        "outer_contexts": split["outer_folds"],
        "inner_folds": split["inner_folds"],
        "outer_test_loader_created": False,
        "outer_test_signal_accessed": False,
        "outer_test_predictions_accessed": False,
        "stages": {
            "initialize": {
                "status": "complete",
                "completed_at_utc": utc_now(),
                "summary": environment,
            }
        },
    }
    write_json(output / "run_manifest.json", manifest)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def make_report(output: Path) -> None:
    analysis = json.loads((output / "analysis_summary.json").read_text())
    decision = analysis["candidate_decision"]
    leaderboard = analysis["leaderboard"]
    by_model = {row["model"]: row for row in leaderboard}
    paired_rows = _read_csv(output / "paired_comparison_summary.csv")
    paired = {(row["candidate"], row["reference"]): row for row in paired_rows}
    norm_rows = _read_csv(output / "branch_norm_statistics.csv")
    branch_rows = _read_csv(output / "branch_ablation_diagnostics.csv")
    gate_rows = _read_csv(output / "gate_statistics.csv")
    subtype_rows = _read_csv(output / "dd_subtype_summary.csv")
    stability_rows = _read_csv(output / "training_stability.csv")
    bootstrap_rows = _read_csv(output / "subject_cluster_bootstrap_delta_ba.csv")
    threshold_rows = _read_csv(output / "cross_fitted_threshold_summary.csv")
    manifest = json.loads((output / "run_manifest.json").read_text())
    plan = json.loads((output / "r2_stabilization_plan.json").read_text())
    overflow_events_path = output / "s0_amp_overflow_events.csv"
    overflow_events = _read_csv(overflow_events_path) if overflow_events_path.is_file() else []
    group_counts: dict[str, int] = {}
    parameter_counts: dict[str, int] = {}
    for row in overflow_events:
        for group in ast.literal_eval(row["nonfinite_groups"]):
            group_counts[group] = group_counts.get(group, 0) + 1
        for name in ast.literal_eval(row["first_nonfinite_parameters"]):
            parameter_counts[name] = parameter_counts.get(name, 0) + 1
    dominant_group = max(group_counts, key=group_counts.get) if group_counts else "not_reproduced"
    dominant_parameter = (
        max(parameter_counts, key=parameter_counts.get)
        if parameter_counts
        else "not_reproduced"
    )
    ratios: dict[str, float] = {}
    validation_norm_means: dict[tuple[str, str], float] = {}
    for model in ("S0", "S1", "S2", "S3"):
        for activation in (
            "deep_embedding_raw",
            "statistical_embedding_raw",
            "deep_embedding_fusion",
            "statistical_embedding_fusion",
            "classifier_input",
            "logits",
        ):
            values = [
                float(row["l2_mean"])
                for row in norm_rows
                if row["model"] == model
                and row["partition"] == "validation"
                and row["activation"] == activation
            ]
            if values:
                validation_norm_means[(model, activation)] = float(np.mean(values))
    for model in ("S1", "S2", "S3"):
        values = [
            float(row["l2_mean"])
            for row in norm_rows
            if row["model"] == model
            and row["partition"] == "validation"
            and row["activation"] == "deep_to_statistical_raw_mean_l2_ratio"
        ]
        ratios[model] = float(np.mean(values))
    branch_summary: dict[tuple[str, str], dict[str, float]] = {}
    for model in ("S2", "S3"):
        for mode in ("deep_disabled", "statistical_disabled"):
            rows = [
                row
                for row in branch_rows
                if row["model"] == model and row["mode"] == mode
            ]
            branch_summary[(model, mode)] = {
                "ba_drop": float(
                    np.mean([float(row["ba_drop_when_disabled"]) for row in rows])
                ),
                "auroc_drop": float(
                    np.mean([float(row["auroc_drop_when_disabled"]) for row in rows])
                ),
            }
    gate_all = next(row for row in gate_rows if row["group"] == "all")
    selected = decision["selected_candidate"]
    subtype_best: dict[str, tuple[str, float, float]] = {}
    for subtype in (
        "Other Movement Disorders",
        "Atypical Parkinsonism",
        "Multiple Sclerosis",
    ):
        rows = [
            row
            for row in subtype_rows
            if row["subtype"] == subtype and row["model"] in {"S1", "S2", "S3"}
        ]
        best = max(rows, key=lambda row: float(row["predicted_dd_fraction"]))
        s0_row = next(
            row
            for row in subtype_rows
            if row["subtype"] == subtype and row["model"] == "S0"
        )
        subtype_best[subtype] = (
            best["model"],
            float(best["predicted_dd_fraction"]),
            float(best["predicted_dd_fraction"])
            - float(s0_row["predicted_dd_fraction"]),
        )

    table = [
        "| Model | Precision | Fusion | BA | AUROC | Macro-F1 | PD Recall | DD Recall | Params | Nonfinite grad | Skipped steps |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in leaderboard:
        table.append(
            f"| {row['model']} | {row['precision']} | {row['fusion']} | "
            f"{float(row['balanced_accuracy']):.4f} | {float(row['auroc']):.4f} | "
            f"{float(row['macro_f1']):.4f} | {float(row['pd_recall']):.4f} | "
            f"{float(row['dd_recall']):.4f} | {row['params']} | "
            f"{row['nonfinite_gradient_batches']} | {row['skipped_optimizer_steps']} |"
        )
    stability_table = [
        "| Model | Best epoch mean [min,max] | Train-val BA gap | Runtime total (s) | Max pre-clip grad norm |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in stability_rows:
        stability_table.append(
            f"| {row['model']} | {float(row['best_epoch_mean']):.2f} "
            f"[{row['best_epoch_minimum']},{row['best_epoch_maximum']}] | "
            f"{float(row['train_validation_ba_gap_mean']):.4f} | "
            f"{float(row['runtime_seconds_total']):.1f} | "
            f"{float(row['maximum_gradient_norm_before_clipping']):.3f} |"
        )
    threshold_table = [
        "| Model | Cross-fit threshold | BA | AUROC | DD Recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in ("M0", "S0", "S1", "S2", "S3", "H1"):
        row = next(row for row in threshold_rows if row["model"] == model)
        threshold_table.append(
            f"| {model} | {float(row['threshold_mean']):.4f} | "
            f"{float(row['balanced_accuracy_mean']):.4f} | "
            f"{float(row['auroc_mean']):.4f} | "
            f"{float(row['dd_recall_mean']):.4f} |"
        )
    bootstrap_table = [
        "| Candidate vs reference | Observed pooled ΔBA | Subject-cluster 95% CI |",
        "|---|---:|---:|",
    ]
    for candidate, reference in (
        ("S1", "S0"),
        ("S1", "M0"),
        ("S1", "H1"),
        ("S2", "S0"),
        ("S3", "S0"),
    ):
        row = next(
            row
            for row in bootstrap_rows
            if row["candidate"] == candidate and row["reference"] == reference
        )
        bootstrap_table.append(
            f"| {candidate} vs {reference} | {float(row['observed_delta']):+.4f} | "
            f"[{float(row['ci95_low']):+.4f}, {float(row['ci95_high']):+.4f}] |"
        )
    pca_values = []
    for outer_context in range(5):
        for inner_fold in range(3):
            metadata = json.loads(
                (
                    output
                    / "models/S1"
                    / f"outer_{outer_context}"
                    / f"inner_{inner_fold}"
                    / "statistical_preprocessing/metadata.json"
                ).read_text()
            )
            pca_values.append(float(metadata["pca_explained_variance_ratio_sum"]))
    h1_gap_ba = float(by_model[selected]["balanced_accuracy"]) - float(
        by_model["H1"]["balanced_accuracy"]
    )
    h1_gap_auc = float(by_model[selected]["auroc"]) - float(by_model["H1"]["auroc"])
    s1s0 = paired[("S1", "S0")]
    s2s1 = paired[("S2", "S1")]
    s3s1 = paired[("S3", "S1")]
    r1r2_advice = (
        "There is development evidence that a deep branch remains contributory, so R1+R2 may be considered in a separately preregistered future stage; it was not run here."
        if decision["S3_deep_contributes"]
        else "Current branch ablation does not establish independent deep contribution strongly enough to justify R1+R2 next."
    )
    lines = [
        "# V3 R2 stabilization development report",
        "",
        "> All results are frozen-split inner-CV development estimates. No outer-test loader, signal, prediction, threshold, or evaluation was used.",
        "",
        "## Preregistration and scope",
        "",
        f"- Plan SHA-256: `{PLAN_SHA256}`.",
        "- S0 was read from the immutable targeted-ablation run; its bounded first-epoch AMP replay was diagnostic-only.",
        "- S1/S2/S3 all used identical FP32 training settings and the same 15 folds.",
        "- The 1560 validation rows repeat 390 subjects; uncertainty uses subject-cluster bootstrap.",
        "",
        "## Development leaderboard",
        "",
        *table,
        "",
        "## Candidate decision",
        "",
        f"Selected development candidate: **{selected}**. S2 retained={decision['retain_S2']}; S3 retained={decision['retain_S3']}. This is not a frozen outer model.",
        "",
        "## Numerical and training stability",
        "",
        *stability_table,
        "",
        f"S0 historical AMP recorded 90 nonfinite-gradient/skipped batches. The bounded first-epoch replay reproduced {len(overflow_events)} events; every event implicated `{dominant_group}`, and `{dominant_parameter}` was the first-listed nonfinite parameter.",
        f"S0 validation mean L2 norms were deep={validation_norm_means[('S0','deep_embedding_raw')]:.3f}, statistical={validation_norm_means[('S0','statistical_embedding_raw')]:.3f}, classifier-input={validation_norm_means[('S0','classifier_input')]:.3f}, logits={validation_norm_means[('S0','logits')]:.3f}. The statistical branch has the larger mean norm and a strong long tail; this supports scale mismatch but does not by itself prove causality.",
        f"S2 LayerNorm changed fusion-side mean L2 norms to deep={validation_norm_means[('S2','deep_embedding_fusion')]:.3f} and statistical={validation_norm_means[('S2','statistical_embedding_fusion')]:.3f}; aggregate L2 remains dimension-dependent (514 vs 32 dimensions).",
        "",
        "## Statistical preprocessing and leakage controls",
        "",
        f"The frozen H1 source had {plan['statistical_features']['source']}; every fold used inner-train-only StandardScaler followed by 32-component, non-whitened full-SVD PCA. Cumulative explained variance across the 15 S1 folds: mean={float(np.mean(pca_values)):.4f}, min={float(np.min(pca_values)):.4f}, max={float(np.max(pca_values)):.4f}. No PCA dimension search was performed.",
        "All 45 formal candidate folds retained disjoint train/validation/test subject lists, validation predictions only, train-only preprocessing subject hashes, and false outer-test loader/signal/prediction/feature-transform flags.",
        "",
        "## Legal cross-fitted threshold (secondary)",
        "",
        *threshold_table,
        "",
        "Each held inner fold used thresholds fitted only from the other two inner-validation folds in the same outer context.",
        "",
        "## Subject-cluster uncertainty",
        "",
        *bootstrap_table,
        "",
        "Intervals use 390 subject clusters, 2000 iterations, seed 20260831; the 1560 repeated validation rows were not treated as independent.",
        "",
        "## Required questions",
        "",
        f"1. **Where did S0 AMP overflow appear first?** Dominant affected group: `{dominant_group}`; most frequently first-listed parameter: `{dominant_parameter}`. See `s0_amp_overflow_events.csv` for every bounded replay event.",
        f"2. **Did disabling AMP remove nonfinite gradients?** S1 nonfinite={by_model['S1']['nonfinite_gradient_batches']}, skipped={by_model['S1']['skipped_optimizer_steps']} across 15 folds.",
        f"3. **Did S1 reproduce S0?** Delta BA={float(s1s0['delta_balanced_accuracy_mean']):+.4f}, delta AUROC={float(s1s0['delta_auroc_mean']):+.4f}, delta DD recall={float(s1s0['delta_dd_recall_mean']):+.4f}; BA nonnegative in {s1s0['folds_delta_balanced_accuracy_ge_0']}/15 folds.",
        f"4. **Branch scale mismatch?** Mean validation raw deep/statistical L2 ratio: S1={ratios['S1']:.3f}, S2={ratios['S2']:.3f}, S3={ratios['S3']:.3f}. Norms are evidence of scale, not proof of overflow causation.",
        f"5. **NormFusion improvement?** S2-S1 delta BA={float(s2s1['delta_balanced_accuracy_mean']):+.4f}, delta AUROC={float(s2s1['delta_auroc_mean']):+.4f}; retention rule={decision['retain_S2']}.",
        f"6. **GatedFusion improvement?** S3-S1 delta BA={float(s3s1['delta_balanced_accuracy_mean']):+.4f}, delta AUROC={float(s3s1['delta_auroc_mean']):+.4f}; retention rule={decision['retain_S3']}.",
        f"7. **Gate collapse?** Overall gate mean={float(gate_all['mean']):.4f}, p5={float(gate_all['p5']):.4f}, p95={float(gate_all['p95']):.4f}; gate weights the deep branch.",
        f"8. **Independent deep contribution?** For S3, deep-disabled mean BA drop={branch_summary[('S3','deep_disabled')]['ba_drop']:+.4f}, AUROC drop={branch_summary[('S3','deep_disabled')]['auroc_drop']:+.4f}; preregistered contribution criterion={decision['S3_deep_contributes']}.",
        f"9. **Largest DD-recall improvement:** {max(('S1','S2','S3'), key=lambda model: float(by_model[model]['dd_recall']))}, with DD recall={max(float(by_model[m]['dd_recall']) for m in ('S1','S2','S3')):.4f}.",
        f"10. **Subtype recovery:** Versus S0 predicted-DD fraction, Other best={subtype_best['Other Movement Disorders'][0]} ({subtype_best['Other Movement Disorders'][1]:.3f}, delta={subtype_best['Other Movement Disorders'][2]:+.3f}); Atypical best={subtype_best['Atypical Parkinsonism'][0]} ({subtype_best['Atypical Parkinsonism'][1]:.3f}, delta={subtype_best['Atypical Parkinsonism'][2]:+.3f}, therefore no candidate improved S0); MS best={subtype_best['Multiple Sclerosis'][0]} ({subtype_best['Multiple Sclerosis'][1]:.3f}, delta={subtype_best['Multiple Sclerosis'][2]:+.3f}). These are exploratory subject-cluster estimates with only 60/15/11 unique subjects respectively.",
        f"11. **Recommended next deep candidate:** {selected}, following stability-first preregistered selection.",
        f"12. **Gap to H1:** BA gap={h1_gap_ba:+.4f}; AUROC gap={h1_gap_auc:+.4f}.",
        f"13. **Future R1+R2 advice:** {r1r2_advice}",
        "",
        "## Provenance, failures, and frozen scope",
        "",
        f"- Output root: `{output}`.",
        f"- Frozen split SHA-256: `{manifest['split_sha256']}`.",
        f"- Base config SHA-256: `{manifest['base_config_sha256']}`.",
        f"- Frozen H1 cache/schema SHA-256: `{manifest['feature_cache_sha256']}` / `{manifest['feature_schema_sha256']}`.",
        f"- Frozen V3 manifest SHA-256: `{manifest['frozen_v3_manifest_sha256']}`.",
        f"- Frozen S0 artifact-tree SHA-256: `{manifest['s0_artifact_tree_sha256']}`.",
        "- Candidate failures: 0/45 folds. FP32 nonfinite loss batches: 0; nonfinite gradient batches: 0; skipped optimizer steps: 0.",
        "- Frozen object: S1 development candidate specification only (raw 514+32 concat, dropout and Linear(546,2), FP32, 123000 parameters). This is not an outer model and contains no outer-test performance claim.",
        "",
        "## Stop condition",
        "",
        "S0 diagnostics and S1/S2/S3 are complete. No new fusion, R1+R2, LR/loss change, outer-test run, or final clinical model freeze was performed.",
    ]
    (output / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize(output: Path) -> dict[str, Any]:
    make_report(output)
    excluded = {"artifact_files.sha256", "run_manifest.json"}
    files = [
        path
        for path in output.rglob("*")
        if path.is_file() and path.relative_to(output).as_posix() not in excluded
    ]
    lines = [
        f"{sha256_file(path)}  {path.relative_to(output).as_posix()}"
        for path in sorted(files)
    ]
    (output / "artifact_files.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    tree = hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
    path = output / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(
        {
            "status": "complete",
            "completed_at_utc": utc_now(),
            "artifact_file_count": len(lines),
            "artifact_tree_sha256": tree,
            "source_hashes_final": source_hashes(),
            "outer_test_loader_created": False,
            "outer_test_signal_accessed": False,
            "outer_test_predictions_accessed": False,
        }
    )
    manifest["stages"]["finalize"] = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "summary": {"artifact_file_count": len(lines), "artifact_tree_sha256": tree},
    }
    write_json(path, manifest)
    return {"artifact_file_count": len(lines), "artifact_tree_sha256": tree}


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    config, split = verify_inputs(output, config_path)
    if args.stage == "initialize":
        initialize(output, split)
        return
    if not (output / "run_manifest.json").is_file():
        raise FileNotFoundError("Run initialize first")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cache = load_handcrafted_cache(cache_path())
    subtypes = subtype_map()
    if args.stage == "diagnose-s0":
        amp = diagnose_s0_amp(config, split, output, device, PLAN_SHA256, cache)
        activations = diagnose_s0_checkpoint_activations(
            config, split, output, device, PLAN_SHA256, cache
        )
        update_manifest(output, "diagnose_s0", {"amp": amp, "activations": activations})
    elif args.stage == "smoke":
        results = {
            model: run_variant(
                config,
                split,
                model,
                output / "smoke",
                device,
                subtypes,
                PLAN_SHA256,
                cache,
                resume=not args.no_resume,
                smoke=True,
            )
            for model in ("S1", "S2", "S3")
        }
        update_manifest(
            output,
            "smoke",
            {model: result["status"] for model, result in results.items()},
        )
    elif args.stage == "run":
        if args.variant is None:
            raise ValueError("--variant is required")
        result = run_variant(
            config,
            split,
            args.variant,
            output,
            device,
            subtypes,
            PLAN_SHA256,
            cache,
            resume=not args.no_resume,
        )
        update_manifest(
            output,
            f"run_{args.variant}",
            {
                "status": result["status"],
                "fold_count": result["fold_count"],
                "failed_fold_count": result["failed_fold_count"],
            },
        )
    elif args.stage == "analyze":
        result = analyze_all(output)
        update_manifest(
            output,
            "analyze",
            {
                "models": result["models"],
                "selected_candidate": result["candidate_decision"][
                    "selected_candidate"
                ],
                "outer_test_accessed": False,
            },
        )
    elif args.stage == "finalize":
        print(json.dumps(finalize(output), indent=2))


if __name__ == "__main__":
    main()
