#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from typing import Any, Mapping

import torch

from src.targeted_ablation.analysis import (
    M0_NAME,
    analyze_all,
    normalize_reference_predictions,
    select_combinations,
)
from src.targeted_ablation.data import load_handcrafted_cache
from src.targeted_ablation.training import run_variant, write_json
from src.utils.config import load_config
from src.utils.provenance import sha256_file


PROJECT_ROOT = Path("/home/zyt/MFAM")
DEFAULT_CONFIG = PROJECT_ROOT / "configs/pads_multi_activity_v3_nested_cv.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/pads_classification/v3_targeted_ablation/targeted_ablation_20260828"
PLAN_SHA256 = "61dc8ef8923c7d8cadfee7615f5bd0247486a3ec619d14295bff2d29c45de448"
SPLIT_SHA256 = "b3c52317cb12b73c66046bbd7c50c94a7e707a64a38d3ad4f0fd4bc0c6ba6b3e"
BASE_CONFIG_SHA256 = "3bb1d8dbeea729775d300123b874752c85cce84f7760f4294e560249de472080"
FEATURE_CACHE_SHA256 = "d7b465b251892abbe602539369989c133fce4c4da15e3bee3cb308d07e0f4da5"
FROZEN_V3_MANIFEST_SHA256 = "af5fdda01ceed8f90944f9a25e3a630b09ad17e07ece5f1d32dfe3d147271321"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run preregistered V3 targeted ablations")
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "initialize",
            "smoke",
            "run-single",
            "select-combinations",
            "run-combinations",
            "analyze",
            "finalize",
        ),
    )
    parser.add_argument("--variant", choices=("R1", "R2", "A1", "A2", "N1"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def verify_inputs(output: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = output / "targeted_ablation_plan.json"
    if sha256_file(plan) != PLAN_SHA256:
        raise ValueError("Preregistered targeted ablation plan hash mismatch")
    if sha256_file(config_path) != BASE_CONFIG_SHA256:
        raise ValueError("Frozen V3 base config hash mismatch")
    config = load_config(config_path)
    split_path = Path(config["nested_cv"]["split_file"])
    if sha256_file(split_path) != SPLIT_SHA256:
        raise ValueError("Frozen nested split hash mismatch")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if int(split["outer_folds"]) != 5 or int(split["inner_folds"]) != 3:
        raise ValueError("Expected frozen 5x3 nested split")
    frozen_manifest = (
        PROJECT_ROOT
        / "outputs/pads_classification/v3_nested_cv/formal_subject_mfam_seed42_20260827/frozen_baseline_manifest.json"
    )
    if sha256_file(frozen_manifest) != FROZEN_V3_MANIFEST_SHA256:
        raise ValueError("Frozen V3 manifest changed")
    feature_cache = (
        PROJECT_ROOT
        / "outputs/pads_classification/v3_analysis_and_baselines/analysis_and_baselines_20260828/features/handcrafted_features.npz"
    )
    if sha256_file(feature_cache) != FEATURE_CACHE_SHA256:
        raise ValueError("Frozen handcrafted feature cache changed")
    return config, split


def source_hashes() -> dict[str, str]:
    paths = (
        "src/targeted_ablation/__init__.py",
        "src/targeted_ablation/models.py",
        "src/targeted_ablation/data.py",
        "src/targeted_ablation/training.py",
        "src/targeted_ablation/analysis.py",
        "scripts/run_v3_targeted_ablation.py",
        "tests/test_targeted_ablation.py",
    )
    return {path: sha256_file(PROJECT_ROOT / path) for path in paths}


def update_manifest(output: Path, stage: str, summary: Mapping[str, Any]) -> None:
    path = output / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["stages"][stage] = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "summary": dict(summary),
    }
    write_json(path, manifest)


def initialize(output: Path, config: dict[str, Any], split: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    existing = [path.name for path in output.iterdir()]
    allowed = {"targeted_ablation_plan.json", "targeted_ablation_plan.json.sha256"}
    if set(existing) - allowed:
        raise FileExistsError(f"Initialization output is not pristine: {existing}")
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
        "scope": "targeted_development_inner_cv_only",
        "plan_sha256": PLAN_SHA256,
        "split_sha256": SPLIT_SHA256,
        "base_config_sha256": BASE_CONFIG_SHA256,
        "feature_cache_sha256": FEATURE_CACHE_SHA256,
        "frozen_v3_manifest_sha256": FROZEN_V3_MANIFEST_SHA256,
        "source_hashes_initial": source_hashes(),
        "outer_test_loader_created": False,
        "outer_test_signal_accessed": False,
        "outer_test_predictions_accessed_for_selection": False,
        "expected_development_folds_per_model": 15,
        "split_outer_folds": split["outer_folds"],
        "split_inner_folds": split["inner_folds"],
        "stages": {
            "initialize": {
                "status": "complete",
                "completed_at_utc": utc_now(),
                "summary": environment,
            }
        },
    }
    write_json(output / "run_manifest.json", manifest)


def subtype_map() -> dict[str, str]:
    rows = normalize_reference_predictions(M0_NAME)
    result: dict[str, str] = {}
    for row in rows:
        previous = result.setdefault(str(row["subject_id"]), str(row["dd_subtype"]))
        if previous != row["dd_subtype"]:
            raise ValueError("Inconsistent subtype in frozen development predictions")
    if len(result) != 390:
        raise ValueError("Frozen development reference must contain 390 subjects")
    return result


def cache_path() -> Path:
    return (
        PROJECT_ROOT
        / "outputs/pads_classification/v3_analysis_and_baselines/analysis_and_baselines_20260828/features/handcrafted_features.npz"
    )


def select_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def make_report(output: Path) -> None:
    summary = json.loads((output / "analysis_summary.json").read_text(encoding="utf-8"))
    leaderboard = summary["leaderboard"]
    paired = {row["model"]: row for row in summary["paired_comparisons"]}
    selection = summary["combination_selection"]
    stability = {
        row["model"]: row
        for row in csv.DictReader((output / "training_stability.csv").open())
    }
    subtype_rows = list(
        csv.DictReader((output / "dd_subtype_subject_cluster_summary.csv").open())
    )
    by_model = {row["model"]: row for row in leaderboard}
    trained = ["R1", "R2", "A1", "A2", "N1", *selection["selected_combinations"]]
    best_trained = max(trained, key=lambda name: float(by_model[name]["balanced_accuracy"]))
    best_auc = max(trained, key=lambda name: float(by_model[name]["delta_auroc_vs_M0"]))
    best_dd = max(trained, key=lambda name: float(by_model[name]["dd_recall"]))
    m0_ba = float(by_model[M0_NAME]["balanced_accuracy"])
    h1_ba = float(by_model["H1_handcrafted_logistic"]["balanced_accuracy"])

    table = [
        "| Model | Modification | BA | Delta BA vs M0 | AUROC | Delta AUROC | Macro-F1 | PD Recall | DD Recall | Params |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in leaderboard:
        table.append(
            f"| {row['model']} | {row['modification']} | {float(row['balanced_accuracy']):.4f} | "
            f"{float(row['delta_ba_vs_M0']):+.4f} | {float(row['auroc']):.4f} | "
            f"{float(row['delta_auroc_vs_M0']):+.4f} | {float(row['macro_f1']):.4f} | "
            f"{float(row['pd_recall']):.4f} | {float(row['dd_recall']):.4f} | {row['params']} |"
        )

    subtype_best = {}
    for subtype in ("Atypical Parkinsonism", "Other Movement Disorders", "Multiple Sclerosis"):
        candidates = [
            row for row in subtype_rows if row["subtype"] == subtype and row["model"] in trained
        ]
        subtype_best[subtype] = max(
            candidates, key=lambda row: float(row["predicted_dd_fraction"])
        )

    lines = [
        "# V3 SubjectMFAM targeted ablation development report",
        "",
        "> All numbers are development inner-CV estimates. No new outer-test training, loading, prediction, threshold selection, or evaluation was performed.",
        "",
        "## Evidence boundary",
        "",
        f"- Preregistered plan SHA-256: `{PLAN_SHA256}`.",
        "- The 1560 validation rows contain repeated observations of 390 subjects; uncertainty uses subject-cluster bootstrap, never independent-row inference.",
        "- M0, H1 and MiniRocket are read-only references from the prior audited run.",
        "- Primary comparison uses default threshold 0.5; cross-fitted thresholds use only the other two inner validation folds in the same outer context.",
        "",
        "## Development leaderboard",
        "",
        *table,
        "",
        "## Preregistered combination decision",
        "",
        f"Selected combinations: {selection['selected_combinations'] or 'none'}. Eligibility thresholds were not relaxed.",
        "",
        "## Answers to the 12 required questions",
        "",
        f"1. **Is representation the largest bottleneck?** Best representation delta: R1={float(by_model['R1']['delta_ba_vs_M0']):+.4f} BA, R2={float(by_model['R2']['delta_ba_vs_M0']):+.4f} BA. Interpretation must follow these observed effects rather than the prior hypothesis.",
        f"2. **Does full-band recover information?** R1 BA={float(by_model['R1']['balanced_accuracy']):.4f}, AUROC={float(by_model['R1']['auroc']):.4f}, DD recall={float(by_model['R1']['dd_recall']):.4f}; paired BA improvement in {paired['R1']['folds_delta_balanced_accuracy_ge_0']}/15 folds.",
        f"3. **Does handcrafted residual recover information?** R2 BA={float(by_model['R2']['balanced_accuracy']):.4f}, AUROC={float(by_model['R2']['auroc']):.4f}; H1 BA={h1_ba:.4f}. R2 is not called synergistic unless it exceeds H1 with supporting ranking evidence.",
        f"4. **Is Hard Top-K harmful?** A1 delta BA={float(by_model['A1']['delta_ba_vs_M0']):+.4f}, delta AUROC={float(by_model['A1']['delta_auroc_vs_M0']):+.4f}, DD recall={float(by_model['A1']['dd_recall']):.4f}.",
        f"5. **Is activity attention necessary?** A2 delta BA={float(by_model['A2']['delta_ba_vs_M0']):+.4f}, delta AUROC={float(by_model['A2']['delta_auroc_vs_M0']):+.4f}; it removes activity-attention parameters.",
        f"6. **Does BatchNorm affect stability?** N1 delta BA={float(by_model['N1']['delta_ba_vs_M0']):+.4f}; best epochs {stability['N1']['best_epoch_minimum']}-{stability['N1']['best_epoch_maximum']}, validation BA SD={float(json.loads((output/'models/N1/development_summary.json').read_text())['fold_metric_summary']['balanced_accuracy']['std']):.4f}, train-validation gap mean={float(stability['N1']['train_validation_ba_gap_mean']):.4f}.",
        f"7. **Largest AUROC gain:** {best_auc}, delta AUROC={float(by_model[best_auc]['delta_auroc_vs_M0']):+.4f}.",
        "8. **Threshold-only behavior:** compare `cross_fitted_threshold_summary.csv` with default metrics; a BA change without AUROC gain is reported as threshold behavior, not representation recovery.",
        f"9. **Largest DD-recall improvement:** {best_dd}, DD recall={float(by_model[best_dd]['dd_recall']):.4f} versus M0={float(by_model[M0_NAME]['dd_recall']):.4f}.",
        f"10. **Subtype recovery:** Atypical best={subtype_best['Atypical Parkinsonism']['model']} ({float(subtype_best['Atypical Parkinsonism']['predicted_dd_fraction']):.3f}); Other best={subtype_best['Other Movement Disorders']['model']} ({float(subtype_best['Other Movement Disorders']['predicted_dd_fraction']):.3f}); MS best={subtype_best['Multiple Sclerosis']['model']} ({float(subtype_best['Multiple Sclerosis']['predicted_dd_fraction']):.3f}). All are exploratory cluster-aware estimates.",
        f"11. **Can a simple MFAM variant approach H1?** Best trained variant is {best_trained}, BA={float(by_model[best_trained]['balanced_accuracy']):.4f}; remaining gap to H1={float(by_model[best_trained]['balanced_accuracy'])-h1_ba:+.4f}.",
        f"12. **Minimal next-version candidate:** {best_trained} is the highest-BA preregistered trained candidate. Final recommendation also considers parameter count, AUROC, fold consistency and cluster interval; this report does not freeze or outer-test it.",
        "",
        "## Stability, paired and uncertainty artifacts",
        "",
        "- `paired_fold_differences.csv`: all 15 paired fold differences.",
        "- `subject_cluster_bootstrap_delta_ba.csv`: 2000 subject-cluster bootstrap replicates summarized as percentile intervals.",
        "- `training_stability.csv`: epochs, train-validation gaps, runtime, gradients, attention entropy and instance utilization.",
        "- `dd_subtype_subject_cluster_summary.csv`: subtype N, four-context prediction summaries and subject-cluster intervals.",
        "- `r1_branch_diagnostics.csv`: branch activation, cosine and frequency-only/full-band-only inference diagnostics.",
        "",
        "## Stop condition",
        "",
        "The preregistered first round and allowed combinations are complete. No third round, loss tuning, new candidate, or outer-test run was started.",
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
    (output / "artifact_files.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tree = hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "complete",
            "completed_at_utc": utc_now(),
            "artifact_file_count": len(lines),
            "artifact_tree_sha256": tree,
            "source_hashes_final": source_hashes(),
            "outer_test_loader_created": False,
            "outer_test_signal_accessed": False,
            "outer_test_predictions_accessed_for_selection": False,
        }
    )
    manifest["stages"]["finalize"] = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "summary": {"artifact_file_count": len(lines), "artifact_tree_sha256": tree},
    }
    write_json(manifest_path, manifest)
    return {"artifact_file_count": len(lines), "artifact_tree_sha256": tree}


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    config, split = verify_inputs(output, config_path)
    if args.stage == "initialize":
        initialize(output, config, split)
        return
    if not (output / "run_manifest.json").is_file():
        raise FileNotFoundError("Run initialize first")
    device = select_device(args.device)
    subtypes = subtype_map()
    cache = load_handcrafted_cache(cache_path())
    if args.stage == "smoke":
        results = {}
        for variant in ("R1", "R2", "A1", "A2", "N1"):
            results[variant] = run_variant(
                config,
                split,
                variant,
                output / "smoke",
                device,
                subtypes,
                PLAN_SHA256,
                cache if variant == "R2" else None,
                resume=not args.no_resume,
                smoke=True,
            )
        update_manifest(output, "smoke", {name: value["status"] for name, value in results.items()})
    elif args.stage == "run-single":
        if args.variant is None:
            raise ValueError("--variant is required for run-single")
        result = run_variant(
            config,
            split,
            args.variant,
            output,
            device,
            subtypes,
            PLAN_SHA256,
            cache if args.variant == "R2" else None,
            resume=not args.no_resume,
        )
        update_manifest(output, f"single_{args.variant}", {"status": result["status"], "fold_count": result["fold_count"]})
    elif args.stage == "select-combinations":
        result = select_combinations(output)
        update_manifest(output, "select_combinations", result)
    elif args.stage == "run-combinations":
        selection = json.loads((output / "combination_selection.json").read_text())
        results = {}
        for variant in selection["selected_combinations"]:
            results[variant] = run_variant(
                config,
                split,
                variant,
                output,
                device,
                subtypes,
                PLAN_SHA256,
                cache if "R2" in variant.split("+") else None,
                resume=not args.no_resume,
            )
        update_manifest(output, "run_combinations", {name: value["status"] for name, value in results.items()})
    elif args.stage == "analyze":
        result = analyze_all(output)
        update_manifest(output, "analyze", {"models": result["models"], "outer_test_accessed": False})
    elif args.stage == "finalize":
        print(json.dumps(finalize(output), indent=2))


if __name__ == "__main__":
    main()
