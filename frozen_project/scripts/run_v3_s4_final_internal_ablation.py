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

import numpy as np
import torch

from src.r2_stabilization.training import write_json
from src.s4_final.analysis import analyze_s4
from src.s4_final.training import run_s4
from src.targeted_ablation.analysis import M0_NAME, normalize_reference_predictions
from src.targeted_ablation.data import load_handcrafted_cache
from src.utils.config import load_config
from src.utils.provenance import sha256_file


PROJECT_ROOT = Path("/home/zyt/MFAM")
DEFAULT_CONFIG = PROJECT_ROOT / "configs/pads_multi_activity_v3_nested_cv.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "outputs/pads_classification/v3_s4_final_internal_ablation/"
    "s4_final_internal_ablation_20260901"
)
PLAN_SHA256 = "3ae383049af664dc970d9893927d2b8543afa04c1d8d83ff98ea2a641bd1d157"
SPLIT_SHA256 = "b3c52317cb12b73c66046bbd7c50c94a7e707a64a38d3ad4f0fd4bc0c6ba6b3e"
BASE_CONFIG_SHA256 = "3bb1d8dbeea729775d300123b874752c85cce84f7760f4294e560249de472080"
FEATURE_CACHE_SHA256 = "d7b465b251892abbe602539369989c133fce4c4da15e3bee3cb308d07e0f4da5"
FEATURE_SCHEMA_FILE_SHA256 = "071ccdf12059496617e9c27e7337d2f3c67787c4be77ff38491f36df6ee21c79"
FEATURE_SCHEMA_SHA256 = "bdb83aae1c31b66c659a63f7a723f74b2cc82bfc99725621f6a41ba736389312"
FROZEN_V3_MANIFEST_SHA256 = "af5fdda01ceed8f90944f9a25e3a630b09ad17e07ece5f1d32dfe3d147271321"
ANALYSIS_ARTIFACT_TREE_SHA256 = "045f747df2de455cf758d6a0960ef123423732a0169b0603195078f95d9be107"
TARGETED_ARTIFACT_TREE_SHA256 = "528be416bfac771a9fe4477330030bfa917be411d0bc0f31fd0ee15f0224b535"
STABILIZATION_ARTIFACT_TREE_SHA256 = "d46afeb3b62045d427c2b455cdddb71606910a56424af5eeca5aa9d86fa8aa5c"
M0_PREDICTIONS_SHA256 = "e71d98a0487a02c92ef0f5302117af072b74d88686cab895768abb842e020fdb"
S1_PREDICTIONS_SHA256 = "7cc98528c7b10781cab4c65624f21053f051373da409bdd8ee649ae4371e5509"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run preregistered S4 final internal ablation")
    parser.add_argument(
        "--stage", required=True,
        choices=("initialize", "smoke", "run", "analyze", "verify", "finalize"),
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def source_hashes() -> dict[str, str]:
    paths = (
        "src/s4_final/__init__.py", "src/s4_final/models.py",
        "src/s4_final/training.py", "src/s4_final/analysis.py",
        "scripts/run_v3_s4_final_internal_ablation.py", "tests/test_s4_final.py",
    )
    return {path: sha256_file(PROJECT_ROOT / path) for path in paths}


def frozen_paths(config_path: Path, output: Path) -> dict[str, tuple[Path, str]]:
    return {
        "plan": (output / "s4_preregistered_plan.json", PLAN_SHA256),
        "base_config": (config_path, BASE_CONFIG_SHA256),
        "frozen_v3_manifest": (PROJECT_ROOT / "outputs/pads_classification/v3_nested_cv/formal_subject_mfam_seed42_20260827/frozen_baseline_manifest.json", FROZEN_V3_MANIFEST_SHA256),
        "handcrafted_cache": (PROJECT_ROOT / "outputs/pads_classification/v3_analysis_and_baselines/analysis_and_baselines_20260828/features/handcrafted_features.npz", FEATURE_CACHE_SHA256),
        "handcrafted_schema_file": (PROJECT_ROOT / "outputs/pads_classification/v3_analysis_and_baselines/analysis_and_baselines_20260828/features/feature_schema.json", FEATURE_SCHEMA_FILE_SHA256),
        "analysis_artifact_tree": (PROJECT_ROOT / "outputs/pads_classification/v3_analysis_and_baselines/analysis_and_baselines_20260828/artifact_files.sha256", ANALYSIS_ARTIFACT_TREE_SHA256),
        "targeted_artifact_tree": (PROJECT_ROOT / "outputs/pads_classification/v3_targeted_ablation/targeted_ablation_20260828/artifact_files.sha256", TARGETED_ARTIFACT_TREE_SHA256),
        "stabilization_artifact_tree": (PROJECT_ROOT / "outputs/pads_classification/v3_r2_stabilization/stabilization_20260831/artifact_files.sha256", STABILIZATION_ARTIFACT_TREE_SHA256),
        "m0_predictions": (PROJECT_ROOT / "outputs/pads_classification/v3_analysis_and_baselines/analysis_and_baselines_20260828/baselines/M0_subject_mfam_frozen_inner_reference/development_predictions_all.csv", M0_PREDICTIONS_SHA256),
        "s1_predictions": (PROJECT_ROOT / "outputs/pads_classification/v3_r2_stabilization/stabilization_20260831/models/S1/development_predictions_all.csv", S1_PREDICTIONS_SHA256),
    }


def verify_inputs(output: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    checks: dict[str, Any] = {}
    for name, (path, expected) in frozen_paths(config_path, output).items():
        observed = sha256_file(path)
        checks[name] = {"path": str(path), "expected_sha256": expected, "observed_sha256": observed, "pass": observed == expected}
        if observed != expected:
            raise ValueError(f"Frozen input hash mismatch: {name}")
    config = load_config(config_path)
    split_path = Path(config["nested_cv"]["split_file"])
    observed_split = sha256_file(split_path)
    checks["split"] = {"path": str(split_path), "expected_sha256": SPLIT_SHA256, "observed_sha256": observed_split, "pass": observed_split == SPLIT_SHA256}
    if observed_split != SPLIT_SHA256:
        raise ValueError("Frozen split changed")
    split = json.loads(split_path.read_text())
    if int(split["outer_folds"]) != 5 or int(split["inner_folds"]) != 3:
        raise ValueError("Expected frozen 5x3 split")
    schema = json.loads((frozen_paths(config_path, output)["handcrafted_schema_file"][0].parent / "feature_matrix_metadata.json").read_text())
    checks["feature_semantic_schema"] = {"expected_sha256": FEATURE_SCHEMA_SHA256, "observed_sha256": schema.get("schema_sha256"), "pass": schema.get("schema_sha256") == FEATURE_SCHEMA_SHA256}
    if not checks["feature_semantic_schema"]["pass"]:
        raise ValueError("Frozen feature semantic schema changed")
    checks["outer_test_contract"] = {
        "outer_test_loader_created": False, "outer_test_signal_accessed": False,
        "outer_test_predictions_accessed": False, "outer_test_features_transformed": False,
        "pass": True,
    }
    return config, split, checks


def cache_path() -> Path:
    return frozen_paths(DEFAULT_CONFIG, DEFAULT_OUTPUT)["handcrafted_cache"][0]


def subtype_map() -> dict[str, str]:
    result: dict[str, str] = {}
    for row in normalize_reference_predictions(M0_NAME):
        subject, subtype = str(row["subject_id"]), str(row["dd_subtype"])
        previous = result.setdefault(subject, subtype)
        if previous != subtype:
            raise ValueError("Inconsistent DD subtype")
    if len(result) != 390:
        raise ValueError("Expected 390 subjects")
    return result


def update_manifest(output: Path, stage: str, summary: Mapping[str, Any]) -> None:
    path = output / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["stages"][stage] = {"status": "complete", "completed_at_utc": utc_now(), "summary": dict(summary)}
    write_json(path, manifest)


def initialize(output: Path, split: Mapping[str, Any], checks: Mapping[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in output.iterdir()}
    allowed = {"s4_preregistered_plan.json", "s4_preregistered_plan.json.sha256"}
    if existing - allowed:
        raise FileExistsError(f"Output directory is not pristine: {sorted(existing)}")
    environment = {
        "created_at_utc": utc_now(), "python": platform.python_version(),
        "platform": platform.platform(), "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    write_json(output / "environment.json", environment)
    write_json(output / "frozen_verification_initial.json", {"status": "pass", "checks": checks})
    manifest = {
        "schema_version": 1, "status": "running", "created_at_utc": utc_now(),
        "scope": "s4_final_internal_development_inner_cv_only",
        "plan_sha256": PLAN_SHA256, "split_sha256": SPLIT_SHA256,
        "base_config_sha256": BASE_CONFIG_SHA256,
        "feature_cache_sha256": FEATURE_CACHE_SHA256,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "analysis_artifact_tree_sha256": ANALYSIS_ARTIFACT_TREE_SHA256,
        "targeted_artifact_tree_sha256": TARGETED_ARTIFACT_TREE_SHA256,
        "stabilization_artifact_tree_sha256": STABILIZATION_ARTIFACT_TREE_SHA256,
        "source_hashes_initial": source_hashes(), "expected_fold_count": 15,
        "outer_contexts": split["outer_folds"], "inner_folds": split["inner_folds"],
        "precision": "FP32", "architecture_search_stop_after_this_run": True,
        "outer_test_loader_created": False, "outer_test_signal_accessed": False,
        "outer_test_predictions_accessed": False, "outer_test_features_transformed": False,
        "stages": {"initialize": {"status": "complete", "completed_at_utc": utc_now(), "summary": environment}},
    }
    write_json(output / "run_manifest.json", manifest)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _fmt(value: Any) -> str:
    return f"{float(value):.4f}"


def make_reports(output: Path) -> None:
    analysis = json.loads((output / "analysis_summary.json").read_text())
    decision = analysis["candidate_decision"]
    leaderboard = {row["model"]: row for row in analysis["leaderboard"]}
    paired = {row["reference"]: row for row in analysis["paired_comparisons"]}
    criteria = decision["criteria"]
    branch = analysis["branch_contribution"]
    stability = analysis["training_stability"]
    subtype = _read_csv(output / "dd_subtype_summary.csv")
    manifest = json.loads((output / "run_manifest.json").read_text())
    selected = decision["selected_development_candidate"]
    s4 = leaderboard["S4"]
    s1 = leaderboard["S1"]
    criteria_lines = [
        f"- {'PASS' if value['pass'] else 'FAIL'} — {name}: observed `{value['observed']}`, threshold `{value['threshold']}`."
        for name, value in criteria.items()
    ]
    branch_lines = [
        f"| {row['diagnostic']} | {row['mode']} | {_fmt(row['balanced_accuracy_mean'])} | {_fmt(row['auroc_mean'])} | {float(row['delta_ba_from_D0_mean']):+.4f} | {float(row['delta_auroc_from_D0_mean']):+.4f} | {_fmt(row['mean_absolute_probability_dd_change'])} | {_fmt(row['mean_logit_l2_change'])} |"
        for row in branch
    ]
    final = [
        "# S4 final internal ablation report", "",
        "> Frozen 5×3 inner-CV development evidence only. No outer-test loader, signal, feature transform, prediction, threshold, or metric was used.", "",
        "## S4 development result", "",
        "| BA | AUROC | Macro-F1 | Accuracy | PD recall | DD recall | NLL | DD Brier |", "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {_fmt(s4['balanced_accuracy'])} | {_fmt(s4['auroc'])} | {_fmt(s4['macro_f1'])} | {_fmt(s4['accuracy'])} | {_fmt(s4['pd_recall'])} | {_fmt(s4['dd_recall'])} | {_fmt(s4['negative_log_likelihood'])} | {_fmt(s4['binary_dd_brier'])} |", "",
        "## S4 vs S1 paired decision evidence", "",
        f"Mean/median ΔBA={float(paired['S1']['delta_balanced_accuracy_mean']):+.4f}/{float(paired['S1']['delta_balanced_accuracy_median']):+.4f}; nonnegative BA folds={paired['S1']['folds_delta_balanced_accuracy_ge_0']}/15. Mean/median ΔAUROC={float(paired['S1']['delta_auroc_mean']):+.4f}/{float(paired['S1']['delta_auroc_median']):+.4f}; mean ΔDD recall={float(paired['S1']['delta_dd_recall_mean']):+.4f}.", "",
        "## Preregistered success criteria", "", *criteria_lines, "",
        f"Mechanical decision: **{decision['decision']}**. Selected deep development specification: **{selected}**.", "",
        "## Inference-only branch contribution", "",
        "| Diagnostic | Mode | BA | AUROC | ΔBA vs D0 | ΔAUROC vs D0 | mean |ΔP(DD)| | mean logit L2 change |", "|---|---|---:|---:|---:|---:|---:|---:|",
        *branch_lines, "",
        "A disabled-branch degradation shows contribution to this trained predictor; it does not establish causal synergy.", "",
        "## Numerical stability and complexity", "",
        f"S4: {stability['S4']['parameter_count_total']} parameters; total runtime={stability['S4']['runtime_seconds_total']:.1f}s; best epoch mean/median={stability['S4']['best_epoch_mean']:.2f}/{stability['S4']['best_epoch_median']:.1f}; validation BA SD={float(s4['balanced_accuracy_sd']):.4f}; train-validation BA gap mean={stability['S4']['train_validation_ba_gap_mean']:.4f}; nonfinite loss/gradient/skipped={stability['S4']['nonfinite_loss_batches']}/{stability['S4']['nonfinite_gradient_batches']}/{stability['S4']['skipped_optimizer_steps']}.",
        f"S1: {stability['S1']['parameter_count_total']} parameters; total runtime={stability['S1']['runtime_seconds_total']:.1f}s; validation BA SD={float(s1['balanced_accuracy_sd']):.4f}; train-validation BA gap mean={stability['S1']['train_validation_ba_gap_mean']:.4f}.", "",
        "## Artifacts and provenance", "",
        f"- Plan SHA-256: `{PLAN_SHA256}`.", f"- Frozen split SHA-256: `{SPLIT_SHA256}`.",
        f"- Read-only stabilization artifact-tree SHA-256: `{STABILIZATION_ARTIFACT_TREE_SHA256}`.",
        "- New server source files: 6; existing project source files modified: 0.",
        "- No temporary server implementation or duplicated feature extractor was created; S4 reuses the R1/S1 modules and frozen training engine.",
        "- Branch representation details, all 15 paired fold rows, cluster bootstrap, cross-fitted thresholds and DD subtype estimates are stored as CSV/JSON beside this report.", "",
        "## Stop condition", "",
        "The internal architecture search on these 390 subjects is closed after this preregistered decision. No S5, retuning, new fusion, outer-test run, cleanup, or additional architecture experiment was started.",
    ]
    (output / "FINAL_REPORT.md").write_text("\n".join(final) + "\n", encoding="utf-8")

    old = _read_csv(PROJECT_ROOT / "outputs/pads_classification/v3_analysis_and_baselines/analysis_and_baselines_20260828/development_leaderboard.csv")
    old_map = {row["model"]: row for row in old}
    targeted = _read_csv(PROJECT_ROOT / "outputs/pads_classification/v3_targeted_ablation/targeted_ablation_20260828/final_development_leaderboard.csv")
    targeted_map = {row["model"]: row for row in targeted}
    stabilized = _read_csv(PROJECT_ROOT / "outputs/pads_classification/v3_r2_stabilization/stabilization_20260831/development_leaderboard.csv")
    stabilized_map = {row["model"]: row for row in stabilized}
    v2 = json.loads((PROJECT_ROOT / "outputs/pads_classification/formal_v2_full_length_ensemble_20260821/test_final/metrics.json").read_text())["metrics"]
    unified = [
        ("V2", "SubjectMFAM 3-seed ensemble", "historical frozen test", v2["balanced_accuracy"], v2["auroc"], v2["confusion_matrix"][1][1] / sum(v2["confusion_matrix"][1]), "Historical split/threshold result; not comparable as inner-CV evidence"),
        ("V3", "M0 SubjectMFAM", "development inner-CV", old_map[M0_NAME]["balanced_accuracy"], old_map[M0_NAME]["auroc"], old_map[M0_NAME]["dd_recall"], "Deep representation misses substantial DD information"),
        ("V3", "H1 handcrafted logistic", "development inner-CV", old_map["H1_handcrafted_logistic"]["balanced_accuracy"], old_map["H1_handcrafted_logistic"]["auroc"], old_map["H1_handcrafted_logistic"]["dd_recall"], "Strongest classical ranking reference by AUROC"),
        ("V3", "H1b balanced logistic", "development inner-CV", old_map["H1b_handcrafted_logistic_balanced"]["balanced_accuracy"], old_map["H1b_handcrafted_logistic_balanced"]["auroc"], old_map["H1b_handcrafted_logistic_balanced"]["dd_recall"], "Class balancing adds only small BA over H1"),
        ("V3", "MiniRocket", "development inner-CV", old_map["MR1_minirocket_multivariate_ridge"]["balanced_accuracy"], old_map["MR1_minirocket_multivariate_ridge"]["auroc"], old_map["MR1_minirocket_multivariate_ridge"]["dd_recall"], "Alternative temporal representation beats M0"),
        ("V3", "Simple CNN N1a", "development inner-CV", old_map["N1a_simple_cnn_lr1e4_drop0p2"]["balanced_accuracy"], old_map["N1a_simple_cnn_lr1e4_drop0p2"]["auroc"], old_map["N1a_simple_cnn_lr1e4_drop0p2"]["dd_recall"], "Collapsed to all-PD"),
        ("V3", "R1 full-band", "development inner-CV", targeted_map["R1"]["balanced_accuracy"], targeted_map["R1"]["auroc"], targeted_map["R1"]["dd_recall"], "Small, consistent full-band gain"),
        ("V3", "R2/S0 AMP", "development inner-CV", stabilized_map["S0"]["balanced_accuracy"], stabilized_map["S0"]["auroc"], stabilized_map["S0"]["dd_recall"], "Statistical residual helps but AMP is numerically invalid"),
        ("V3", "S1 R2-FP32", "development inner-CV", stabilized_map["S1"]["balanced_accuracy"], stabilized_map["S1"]["auroc"], stabilized_map["S1"]["dd_recall"], "Stable previous deep candidate"),
        ("V3", "S2 NormFusion", "development inner-CV", stabilized_map["S2"]["balanced_accuracy"], stabilized_map["S2"]["auroc"], stabilized_map["S2"]["dd_recall"], "Dual LayerNorm fusion rejected"),
        ("V3", "S3 GatedFusion", "development inner-CV", stabilized_map["S3"]["balanced_accuracy"], stabilized_map["S3"]["auroc"], stabilized_map["S3"]["dd_recall"], "No clear improvement over raw concat"),
        ("V3", "S4 S1+R1", "development inner-CV", s4["balanced_accuracy"], s4["auroc"], s4["dd_recall"], decision["decision"]),
    ]
    table = ["| Stage | Model | Evidence type | BA | AUROC | DD Recall | Key conclusion |", "|---|---|---|---:|---:|---:|---|"]
    table.extend(f"| {a} | {b} | {c} | {float(d):.4f} | {float(e):.4f} | {float(f):.4f} | {g} |" for a,b,c,d,e,f,g in unified)
    subtype_table = ["| Model | ET | Other | Atypical | MS |", "|---|---:|---:|---:|---:|"]
    subtype_names = {"Essential Tremor": "ET", "Other Movement Disorders": "Other", "Atypical Parkinsonism": "Atypical", "Multiple Sclerosis": "MS"}
    for model in ("M0", "H1", "S1", "S4"):
        values = {subtype_names[row["subtype"]]: float(row["predicted_dd_fraction"]) for row in subtype if row["model"] == model and row["subtype"] in subtype_names}
        subtype_table.append(f"| {model} | {values['ET']:.3f} | {values['Other']:.3f} | {values['Atypical']:.3f} | {values['MS']:.3f} |")
    stage = [
        "# MFAM project stage summary — 2026-09-01", "",
        "## Research task and evidence boundary", "",
        "The project performs subject-level PD-versus-DD classification on the 390-subject PADS subset using 11 activities, bilateral wrists and accelerometer+gyroscope signals; labels are 0=PD and 1=DD. It is not healthy-control classification, UPDRS/severity regression, or clinical deployment validation.", "",
        "V2 used one frozen train/validation/test split, three seeds and a probability ensemble. V3 replaced model-development evidence with a frozen 5×3 nested-CV design across five diagnosis strata, train-only normalization/PCA, subject-disjoint folds, explicit provenance and anti-leakage flags. Historical V2 test evidence and V3 inner-CV development estimates remain distinct.", "",
        "## Unified evidence table", "", *table, "",
        "## Hypotheses weakened or rejected", "",
        "- The old split is not established as the main performance problem: V3's stricter protocol changed estimation, while matched baselines still show a large representation gap.",
        "- Threshold is not the main bottleneck: the frozen diagnostic threshold improved BA by only +0.0113, far below the H1 and MiniRocket representation gains.",
        "- Class weighting is not the main solution: H1b improves BA only about +0.0029 over H1.",
        "- Timestamp gaps and robust-outlier warnings are not supported as primary error drivers: only four gap records existed and the audited associations were small/inconclusive or opposite the proposed direction.",
        "- GroupNorm, masked-mean activity fusion and removal of Hard Top-K did not improve matched development BA (N1, A2 and A1 respectively).",
        "- Dual LayerNorm fusion was harmful (S2); gated fusion did not clearly outperform raw concat (S3).", "",
        "## Evidence that remains supported", "",
        "- Handcrafted statistics remain strong (H1/H1b) and the statistical residual substantially improves SubjectMFAM ranking.",
        "- M0 misses DD-relevant information; both MiniRocket and H1 materially outperform it.",
        "- R1 provides a small but fold-consistent full-band improvement; S4 determines whether that signal adds enough beyond S1 under the locked rule.",
        "- FP32 eliminated the R2 AMP nonfinite-gradient/skipped-step instability.",
        "- Acc+Gyro (BA 0.692) exceeds Acc-only (0.670) and Gyro-only (0.668); bilateral wrists (0.692) exceed either wrist alone; full multi-activity H1 exceeds every single activity.",
        "- DD subtype heterogeneity is pronounced, with ET generally easier and Atypical/MS estimates much less certain.", "",
        "## Exploratory DD subtype recovery", "", *subtype_table, "",
        "Values are repeated-inner-validation predicted-DD fractions with subject-cluster uncertainty in the accompanying CSV. Atypical (N=15) and MS (N=11) are especially small; all subtype conclusions are exploratory and did not select the model.", "",
        "## Current code and provenance state", "",
        "Maintain the primary nested-CV/provenance path, the audited handcrafted-feature baseline path, targeted ablations, R2 stabilization, and this S4 runner. Experimental entry points have accumulated and should be archived or documented in a separate future maintenance task; none were removed here because they are part of frozen experiment provenance. There is no duplicate feature extractor in S4, but its runner is intentionally experiment-specific to prevent accidental scope expansion.",
        "This round added six source files and modified zero existing source files. No broad cleanup or refactor was performed.", "",
        "## Frozen model choices", "",
        "- Current historical deep baseline: V2 SubjectMFAM three-seed ensemble (historical frozen split/test evidence).",
        "- Current strongest classical reference: H1 handcrafted logistic by AUROC; H1b has the slightly higher BA but is a balanced-sensitivity variant.",
        f"- Current selected deep development candidate: **{selected}**. " + ("S4 replaces S1 as selected development specification." if selected == "S4" else "S1 remains selected; S4 is rejected."), "",
        "The selected object is a development specification, not a final, outer-tested, clinically validated model.", "",
        "## Research-design limitation", "",
        "The same 390 subjects have informed nested-CV development, baseline analysis, error analysis and multiple preregistered ablations. Any further result on these people cannot honestly be called a completely untouched independent final validation. Strong generalization evidence now requires an external dataset or genuinely new holdout cohort; re-randomizing these 390 subjects cannot manufacture a blind test.", "",
        "## Next directions (do not execute automatically)", "",
        "1. Validate the frozen selected specification on an external dataset or genuinely new cohort.",
        "2. Prepare interpretation, uncertainty and reporting for the selected deep candidate without new architecture search.",
        "3. In a separately authorized maintenance task, archive/document experimental entry points while preserving frozen provenance.", "",
        "## Stop", "",
        "Internal architecture development on the current 390 subjects stops here. No additional model, hyperparameter, fusion or outer-test experiment was started.",
    ]
    (output / "PROJECT_STAGE_SUMMARY_20260901.md").write_text("\n".join(stage) + "\n", encoding="utf-8")


def finalize(output: Path) -> dict[str, Any]:
    make_reports(output)
    excluded = {"artifact_files.sha256", "run_manifest.json"}
    files = [path for path in output.rglob("*") if path.is_file() and path.relative_to(output).as_posix() not in excluded]
    lines = [f"{sha256_file(path)}  {path.relative_to(output).as_posix()}" for path in sorted(files)]
    checksum_text = "\n".join(lines) + "\n"
    (output / "artifact_files.sha256").write_text(checksum_text, encoding="utf-8")
    tree = hashlib.sha256(checksum_text.encode()).hexdigest()
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update({
        "status": "complete", "completed_at_utc": utc_now(),
        "artifact_file_count": len(lines), "artifact_tree_sha256": tree,
        "source_hashes_final": source_hashes(), "architecture_search_status": "closed",
        "outer_test_loader_created": False, "outer_test_signal_accessed": False,
        "outer_test_predictions_accessed": False, "outer_test_features_transformed": False,
    })
    manifest["stages"]["finalize"] = {"status": "complete", "completed_at_utc": utc_now(), "summary": {"artifact_file_count": len(lines), "artifact_tree_sha256": tree}}
    write_json(manifest_path, manifest)
    return {"artifact_file_count": len(lines), "artifact_tree_sha256": tree}


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    config, split, checks = verify_inputs(output, config_path)
    if args.stage == "initialize":
        initialize(output, split, checks)
        return
    if not (output / "run_manifest.json").is_file():
        raise FileNotFoundError("Run initialize first")
    if args.stage == "verify":
        result = {"status": "pass", "verified_at_utc": utc_now(), "checks": checks, "source_hashes": source_hashes(), "outer_test_accessed": False}
        write_json(output / "frozen_verification_final.json", result)
        update_manifest(output, "verify", {"status": "pass", "check_count": len(checks)})
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cache = load_handcrafted_cache(cache_path())
    subtypes = subtype_map()
    if args.stage == "smoke":
        result = run_s4(config, split, output / "smoke", device, subtypes, PLAN_SHA256, cache, resume=not args.no_resume, smoke=True)
        update_manifest(output, "smoke", {"status": result["status"], "fold_count": result["fold_count"], "outer_test_accessed": False})
    elif args.stage == "run":
        result = run_s4(config, split, output, device, subtypes, PLAN_SHA256, cache, resume=not args.no_resume)
        update_manifest(output, "run_S4", {"status": result["status"], "fold_count": result["fold_count"], "failed_fold_count": result["failed_fold_count"], "outer_test_accessed": False})
    elif args.stage == "analyze":
        result = analyze_s4(output)
        update_manifest(output, "analyze", {"selected_development_candidate": result["candidate_decision"]["selected_development_candidate"], "s4_passes": result["candidate_decision"]["s4_passes_all_core_criteria"], "outer_test_accessed": False})
    elif args.stage == "finalize":
        print(json.dumps(finalize(output), indent=2))


if __name__ == "__main__":
    main()
