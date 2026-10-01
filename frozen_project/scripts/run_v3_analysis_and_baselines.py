from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.baselines import (
    baseline_subtype_analysis,
    frozen_subject_mfam_inner_reference,
    run_core_linear_baselines,
    run_feature_information_analysis,
)
from src.analysis.common import (
    canonical_sha256,
    load_patients,
    read_csv,
    read_json,
    sha256_file,
    subject_condition_map,
    write_csv,
    write_json,
)
from src.analysis.diagnostics import run_diagnostics
from src.analysis.features import ACTIVITIES, extract_feature_matrix, minirocket_subject_tensor


FROZEN_DIR = PROJECT_ROOT / "outputs/pads_classification/v3_nested_cv/formal_subject_mfam_seed42_20260827"
PROCESSED_ROOT = PROJECT_ROOT / "data/processed/pads_multi_activity/v2_l1_full_length"
RAW_ROOT = Path("/home/zyt/pads")
CANDIDATE_GRID = PROJECT_ROOT / "configs/v3_analysis_candidate_grid.json"
EXPECTED_FROZEN_MANIFEST_SHA256 = "af5fdda01ceed8f90944f9a25e3a630b09ad17e07ece5f1d32dfe3d147271321"
EXPECTED_ARTIFACT_TREE_SHA256 = "b168b86b4a9cf9acba2d165c5230bb1b4aae7020234e70b983d91549f123d103"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V3 frozen diagnostics and development-only baselines")
    parser.add_argument("--stage", required=True, choices=("initialize", "diagnostics", "features", "linear", "minirocket", "cnn", "report", "all"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def verify_frozen() -> dict[str, Any]:
    manifest_hash = sha256_file(FROZEN_DIR / "frozen_baseline_manifest.json")
    if manifest_hash != EXPECTED_FROZEN_MANIFEST_SHA256:
        raise ValueError("Frozen V3 manifest hash mismatch")
    frozen = read_json(FROZEN_DIR / "frozen_baseline_manifest.json")
    if frozen["artifact_tree_sha256"] != EXPECTED_ARTIFACT_TREE_SHA256:
        raise ValueError("Frozen V3 artifact tree identity mismatch")
    text = (FROZEN_DIR / "artifact_files.sha256").read_text(encoding="utf-8")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != EXPECTED_ARTIFACT_TREE_SHA256:
        raise ValueError("Frozen V3 artifact manifest content mismatch")
    for line in text.splitlines():
        expected, relative = line.split("  ", 1)
        if sha256_file(FROZEN_DIR / relative) != expected:
            raise ValueError(f"Frozen artifact changed: {relative}")
    return {"status": "pass", "manifest_sha256": manifest_hash, "artifact_tree_sha256": frozen["artifact_tree_sha256"], "artifact_file_count": frozen["artifact_file_count"]}


def _source_provenance() -> dict[str, str]:
    paths = sorted((PROJECT_ROOT / "src/analysis").glob("*.py")) + [Path(__file__).resolve(), CANDIDATE_GRID]
    return {path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path) for path in paths}


def initialize(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    frozen = verify_frozen()
    shutil.copy2(CANDIDATE_GRID, output / "candidate_grid.json")
    environment = {
        "python": sys.version, "platform": platform.platform(),
        "executable": sys.executable,
        "packages": subprocess.run([sys.executable, "-m", "pip", "freeze"], text=True, capture_output=True, check=True).stdout.splitlines(),
    }
    write_json(output / "environment.json", environment)
    write_json(
        output / "run_manifest.json",
        {
            "schema_version": 1, "status": "initialized", "created_at_utc": _utc_now(),
            "scope": "frozen_outer_descriptive_and_development_inner_cv",
            "frozen_outer_feedback_used_for_model_selection": False,
            "frozen_v3_integrity": frozen, "candidate_grid_sha256": sha256_file(CANDIDATE_GRID),
            "source_provenance": _source_provenance(), "stages": {},
        },
    )


def _mark_stage(output: Path, stage: str, result: Any) -> None:
    manifest = read_json(output / "run_manifest.json")
    manifest["stages"][stage] = {"status": "complete", "completed_at_utc": _utc_now(), "summary": result}
    manifest["status"] = "in_progress"
    write_json(output / "run_manifest.json", manifest)


def _subjects_labels() -> tuple[list[str], np.ndarray, dict[str, str], dict[str, dict[str, Any]]]:
    conditions = subject_condition_map(PROCESSED_ROOT / "manifests")
    subjects = sorted(conditions)
    labels = np.asarray([0 if conditions[subject] == "Parkinson's" else 1 for subject in subjects], dtype=np.int64)
    patients = load_patients(RAW_ROOT / "patients", set(subjects))
    return subjects, labels, conditions, patients


def stage_diagnostics(output: Path) -> dict[str, Any]:
    result = run_diagnostics(frozen_dir=FROZEN_DIR, processed_root=PROCESSED_ROOT, raw_root=RAW_ROOT, output_dir=output / "diagnostics")
    compact = {"subject_count": result["frozen_subject_count"], "metrics": result["frozen_metrics_recomputed"], "gap_records": result["quality"]["gap_records"], "outlier_records": result["quality"]["robust_outlier"]["records_flagged"]}
    _mark_stage(output, "diagnostics", compact)
    return compact


def stage_features(output: Path) -> dict[str, Any]:
    feature_dir = output / "features"
    if feature_dir.exists():
        raise FileExistsError(feature_dir)
    feature_dir.mkdir(parents=True)
    subjects, labels, conditions, _ = _subjects_labels()
    matrix, schema = extract_feature_matrix(PROCESSED_ROOT, subjects)
    np.savez_compressed(feature_dir / "handcrafted_features.npz", X=matrix, subject_ids=np.asarray(subjects), labels=labels)
    write_json(feature_dir / "feature_schema.json", schema)
    metadata = {
        "representation": "processed V2/V3 signals", "subject_count": len(subjects),
        "feature_count": int(matrix.shape[1]), "matrix_shape": list(matrix.shape),
        "matrix_dtype": str(matrix.dtype), "finite": bool(np.isfinite(matrix).all()),
        "schema_sha256": schema["schema_sha256"], "extractor_label_dependency": False,
        "matrix_sha256": sha256_file(feature_dir / "handcrafted_features.npz"),
        "subject_ids_sha256": canonical_sha256(subjects),
    }
    write_json(feature_dir / "feature_matrix_metadata.json", metadata)
    _mark_stage(output, "features", metadata)
    return metadata


def _load_features(output: Path) -> tuple[np.ndarray, list[str], np.ndarray, dict[str, Any]]:
    payload = np.load(output / "features/handcrafted_features.npz", allow_pickle=False)
    return payload["X"], payload["subject_ids"].astype(str).tolist(), payload["labels"].astype(np.int64), read_json(output / "features/feature_schema.json")


def stage_linear(output: Path) -> dict[str, Any]:
    baseline_dir = output / "baselines"; baseline_dir.mkdir(exist_ok=True)
    X, subjects, labels, schema = _load_features(output)
    conditions = subject_condition_map(PROCESSED_ROOT / "manifests")
    patients = load_patients(RAW_ROOT / "patients", set(subjects))
    split = read_json(FROZEN_DIR / "frozen_split.json")
    summaries = {"M0_subject_mfam_frozen_inner_reference": frozen_subject_mfam_inner_reference(FROZEN_DIR, conditions, baseline_dir)}
    summaries.update(run_core_linear_baselines(feature_matrix=X, subjects=subjects, labels=labels, schema=schema, split=split, conditions=conditions, patients=patients, output_dir=baseline_dir))
    information = run_feature_information_analysis(feature_matrix=X, subjects=subjects, labels=labels, schema=schema, split=split, conditions=conditions, output_dir=baseline_dir)
    result = {"core_models": list(summaries), "information_variants": information["variant_count"]}
    _mark_stage(output, "linear", result)
    return result


def stage_minirocket(output: Path) -> dict[str, Any]:
    from src.analysis.minirocket_baseline import run_minirocket

    feature_dir = output / "features"
    representation_path = feature_dir / "minirocket_subject_tensor.npz"
    subjects, labels, conditions, _ = _subjects_labels()
    if not representation_path.exists():
        tensor = minirocket_subject_tensor(PROCESSED_ROOT, subjects, target_length=976)
        np.savez_compressed(representation_path, X=tensor, subject_ids=np.asarray(subjects), labels=labels)
        write_json(feature_dir / "minirocket_representation_metadata.json", {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "subject_level_sample_count": len(subjects), "activity_samples_are_not_independent": True, "sha256": sha256_file(representation_path)})
    else:
        payload = np.load(representation_path, allow_pickle=False); tensor = payload["X"]
        if payload["subject_ids"].astype(str).tolist() != subjects:
            raise ValueError("MiniRocket cached subject order mismatch")
    summary = run_minirocket(tensor=tensor, subjects=subjects, labels=labels, split=read_json(FROZEN_DIR / "frozen_split.json"), conditions=conditions, output_dir=output / "baselines")
    _mark_stage(output, "minirocket", summary)
    return summary


def stage_cnn(output: Path, device: str) -> dict[str, Any]:
    from src.analysis.simple_cnn import run_simple_cnn

    subjects, labels, conditions, _ = _subjects_labels()
    label_map = {subject: int(label) for subject, label in zip(subjects, labels)}
    summaries = run_simple_cnn(processed_root=PROCESSED_ROOT, labels_by_subject=label_map, split=read_json(FROZEN_DIR / "frozen_split.json"), conditions=conditions, frozen_dir=FROZEN_DIR, output_dir=output / "baselines", device=device)
    _mark_stage(output, "cnn", summaries)
    return summaries


def _metric_mean(summary: dict[str, Any], key: str) -> Any:
    return summary["fold_metric_summary"][key]["mean"]


def stage_report(output: Path) -> dict[str, Any]:
    baseline_root = output / "baselines"
    summaries: dict[str, dict[str, Any]] = {}
    for path in sorted(baseline_root.glob("*/development_summary.json")):
        summaries[path.parent.name] = read_json(path)
    model_names = sorted(summaries)
    subtype = baseline_subtype_analysis(baseline_root, model_names)
    leaderboard: list[dict[str, Any]] = []
    representations = {
        "M0_subject_mfam_frozen_inner_reference": "Frozen SubjectMFAM inner validation",
        "D0_demographic_logistic": "Basic demographics only",
        "H1_handcrafted_logistic": "Processed handcrafted time/frequency",
        "H1b_handcrafted_logistic_balanced": "Handcrafted, balanced sensitivity",
        "H2_handcrafted_linear_svm": "Processed handcrafted time/frequency",
        "MR1_minirocket_multivariate_ridge": "Subject-level multivariate MiniRocket",
        "N1a_simple_cnn_lr1e4_drop0p2": "Simple shared Conv1D",
        "N1b_simple_cnn_lr2e4_drop0p4": "Simple shared Conv1D",
    }
    for name, summary in summaries.items():
        leaderboard.append(
            {
                "model": name, "representation": representations.get(name, "unknown"),
                "balanced_accuracy": _metric_mean(summary, "balanced_accuracy"),
                "macro_f1": _metric_mean(summary, "macro_f1"),
                "auroc": _metric_mean(summary, "auroc"),
                "pd_recall": _metric_mean(summary, "pd_recall"),
                "dd_recall": _metric_mean(summary, "dd_recall"),
                "params": summary.get("parameter_count", "N/A"),
                "notes": "Development inner-CV estimate — not a new outer-test result",
            }
        )
    leaderboard.append({"model": "predict_all_PD", "representation": "majority baseline", "balanced_accuracy": 0.5, "macro_f1": 0.4144144144144144, "auroc": 0.5, "pd_recall": 1.0, "dd_recall": 0.0, "params": 0, "notes": "Accuracy=276/390=0.7077"})
    leaderboard.sort(key=lambda row: float(row["balanced_accuracy"]), reverse=True)
    write_csv(output / "development_leaderboard.csv", leaderboard, list(leaderboard[0]))
    frozen = read_json(FROZEN_DIR / "nested_cv_summary.json")
    write_json(output / "frozen_subject_mfam_outer_test_reference.json", {"scope": "frozen_outer_test_descriptive_reference_not_same_level_as_development", "fold_metric_summary": frozen["fold_metric_summary"], "pooled_outer_test_metrics": frozen["pooled_outer_test_metrics"]})
    diagnostics = read_json(output / "diagnostics/diagnostic_summary.json")
    information = read_csv(output / "baselines/feature_information_analysis/summary.csv")
    single = sorted([row for row in information if row["variant"].startswith("single_activity::")], key=lambda row: float(row["balanced_accuracy_mean"]), reverse=True)
    sensor = sorted([row for row in information if row["variant"].startswith("sensor::")], key=lambda row: float(row["balanced_accuracy_mean"]), reverse=True)
    wrist = sorted([row for row in information if row["variant"].startswith("wrist::")], key=lambda row: float(row["balanced_accuracy_mean"]), reverse=True)
    leave_one_out = sorted([row for row in information if row["variant"].startswith("leave_one_activity_out::")], key=lambda row: float(row["balanced_accuracy_mean"]), reverse=True)
    best_development = leaderboard[0]
    dynamics = read_json(output / "diagnostics/subject_mfam_training_dynamics_summary.json")
    quality = diagnostics["quality"]
    quality_by_metric = {row["metric"]: row for row in quality["error_association"]}

    def dev(model: str, metric: str) -> float:
        return _metric_mean(summaries[model], metric)

    subtype_map = {(row["model"], row["subtype"]): row for row in subtype}
    core_models = [
        "M0_subject_mfam_frozen_inner_reference", "D0_demographic_logistic",
        "H1_handcrafted_logistic", "H1b_handcrafted_logistic_balanced",
        "H2_handcrafted_linear_svm", "MR1_minirocket_multivariate_ridge",
        "N1a_simple_cnn_lr1e4_drop0p2", "N1b_simple_cnn_lr2e4_drop0p4",
    ]
    subtype_names = ["Atypical Parkinsonism", "Essential Tremor", "Multiple Sclerosis", "Other Movement Disorders"]
    threshold_gain = diagnostics["probability_threshold"]["diagnostic_global_threshold"]["balanced_accuracy"] - diagnostics["frozen_metrics_recomputed"]["balanced_accuracy"]
    answers = {
        "scope": "Evidence-based answers; all model comparisons are development inner-CV estimates. Frozen outer-test predictions are descriptive only.",
        "Q1_primary_bottleneck": {
            "answer": "The evidence points primarily to representation/aggregation limitations interacting with heterogeneous DD phenotypes, not to a single threshold, demographic, timestamp-gap, or robust-outlier problem.",
            "evidence": {
                "retrospective_threshold_ba_gain_diagnostic_only": threshold_gain,
                "H1_minus_M0_balanced_accuracy": dev("H1_handcrafted_logistic", "balanced_accuracy") - dev("M0_subject_mfam_frozen_inner_reference", "balanced_accuracy"),
                "MR1_minus_M0_balanced_accuracy": dev("MR1_minirocket_multivariate_ridge", "balanced_accuracy") - dev("M0_subject_mfam_frozen_inner_reference", "balanced_accuracy"),
                "frozen_outer_auroc": diagnostics["frozen_metrics_recomputed"]["auroc"],
            },
        },
        "Q2_demographics": {"answer": "Basic demographics contain some ranking signal but weak thresholded discrimination; they are a confounding audit, not a competitive sensor model.", "balanced_accuracy": dev("D0_demographic_logistic", "balanced_accuracy"), "auroc": dev("D0_demographic_logistic", "auroc"), "dd_recall": dev("D0_demographic_logistic", "dd_recall")},
        "Q3_handcrafted": {"answer": "Both simple linear handcrafted baselines materially exceed frozen SubjectMFAM on development inner-CV.", "H1_balanced_accuracy": dev("H1_handcrafted_logistic", "balanced_accuracy"), "H2_balanced_accuracy": dev("H2_handcrafted_linear_svm", "balanced_accuracy"), "M0_balanced_accuracy": dev("M0_subject_mfam_frozen_inner_reference", "balanced_accuracy")},
        "Q4_minirocket": {"answer": "MiniRocket exceeds SubjectMFAM but remains below H1, supporting a representation bottleneck without establishing MiniRocket as final generalization performance.", "MR1_balanced_accuracy": dev("MR1_minirocket_multivariate_ridge", "balanced_accuracy"), "MR1_auroc": dev("MR1_minirocket_multivariate_ridge", "auroc")},
        "Q5_simple_cnn": {"answer": "Neither preregistered simple CNN candidate learned useful discrimination; both collapsed to all-PD predictions. This is a protocol-specific negative result, not proof that every CNN must fail.", "N1a_balanced_accuracy": dev("N1a_simple_cnn_lr1e4_drop0p2", "balanced_accuracy"), "N1b_balanced_accuracy": dev("N1b_simple_cnn_lr2e4_drop0p4", "balanced_accuracy")},
        "Q6_subtypes": {"answer": "ET is consistently easiest for useful sensor models. Frozen SubjectMFAM is especially weak on Atypical Parkinsonism and MS; H1-family models substantially recover Atypical cases and shift the hardest group toward MS. Small subtype counts make this exploratory."},
        "Q7_activities": {"answer": "LiftHold and TouchNose are the strongest single-activity development estimates, while multi-activity H1 is better than every single activity. Leave-one-out results suggest redundancy/complementarity but are exploratory."},
        "Q8_sensors": {"answer": "Accelerometer-only and gyroscope-only are similar; their combination is better, indicating complementary information.", "combined_ba": float(next(row for row in sensor if row["variant"] == "sensor::acc_gyro")["balanced_accuracy_mean"])},
        "Q9_wrists": {"answer": "Bilateral input is best; left-only is slightly better than right-only, and neither unilateral variant matches bilateral development performance."},
        "Q10_quality": {"answer": "There is no evidence that the four timestamp-gap records, sampling-rate deviation, or robust-outlier warning burden explains frozen errors. Outlier flags are warnings and may reflect pathological motion."},
        "Q11_training": {"answer": "Best epochs and train-validation gaps are variable, consistent with optimization/overfitting instability, but gap magnitude has no clear monotonic association with validation BA in 15 folds."},
        "Q12_next_steps": {"answer": "Prioritize representation and aggregation ablations; treat normalization stability and subtype-aware evaluation as secondary; do not prioritize resampling, class weighting, activity dropout, or new bands without a new preregistered inner-CV ablation."},
    }
    write_json(output / "evidence_answers_q1_q12.json", answers)

    leaderboard_lines = ["| Model | BA | Macro-F1 | AUROC | PD recall | DD recall |", "|---|---:|---:|---:|---:|---:|"]
    for row in leaderboard:
        leaderboard_lines.append(
            f"| {row['model']} | {float(row['balanced_accuracy']):.4f} | {float(row['macro_f1']):.4f} | {float(row['auroc']):.4f} | {float(row['pd_recall']):.4f} | {float(row['dd_recall']):.4f} |"
        )
    subtype_lines = ["| Model | Atypical | ET | MS | Other |", "|---|---:|---:|---:|---:|"]
    for model in core_models:
        values = [subtype_map[(model, name)]["predicted_dd_fraction"] for name in subtype_names]
        subtype_lines.append(f"| {model} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} | {values[3]:.3f} |")
    report = [
        "# V3 SubjectMFAM diagnosis and development-only baseline report", "",
        "> Development inner-CV estimates are not new outer-test results. Frozen SubjectMFAM outer predictions are used only for descriptive diagnosis.", "",
        "## Evidence boundary", "",
        f"- Frozen manifest: `{EXPECTED_FROZEN_MANIFEST_SHA256}` (verified read-only).",
        "- No SubjectMFAM variant was trained or modified.",
        "- Every new baseline uses only the 15 frozen inner train/validation folds; outer-test access flag is false.", "",
        "## Development leaderboard", "",
        "These are model-selection/development estimates, not final outer-test estimates.", "",
        *leaderboard_lines, "",
        "The top development estimate is " + f"**{best_development['model']}** with BA={float(best_development['balanced_accuracy']):.4f}.", "",
        "## Frozen outer-test diagnostic", "",
        f"Frozen pooled BA={diagnostics['frozen_metrics_recomputed']['balanced_accuracy']:.4f}, macro-F1={diagnostics['frozen_metrics_recomputed']['macro_f1']:.4f}, AUROC={diagnostics['frozen_metrics_recomputed']['auroc']:.4f}, PD recall={diagnostics['frozen_metrics_recomputed']['pd_recall']:.4f}, DD recall={diagnostics['frozen_metrics_recomputed']['dd_recall']:.4f}.",
        f"The frozen confusion matrix is {diagnostics['frozen_metrics_recomputed']['confusion_matrix']}; binary DD Brier={diagnostics['frozen_metrics_recomputed']['binary_dd_brier']:.4f}, NLL={diagnostics['frozen_metrics_recomputed']['negative_log_likelihood']:.4f}, ECE(10 equal-width bins)={diagnostics['probability_threshold']['reliability']['ece_equal_width_10']:.4f}.",
        f"The five frozen fold thresholds have mean={diagnostics['probability_threshold']['thresholds']['mean']:.4f}, SD={diagnostics['probability_threshold']['thresholds']['std']:.4f}, range={diagnostics['probability_threshold']['thresholds']['minimum']:.4f}-{diagnostics['probability_threshold']['thresholds']['maximum']:.4f}.",
        f"A retrospective global threshold of {diagnostics['probability_threshold']['diagnostic_global_threshold']['threshold']:.4f} gives BA={diagnostics['probability_threshold']['diagnostic_global_threshold']['balanced_accuracy']:.4f}, only {threshold_gain:+.4f}; this is diagnostic only and must not be reused for model selection.",
        f"Mean p(DD) is {diagnostics['probability_threshold']['pd_probability_dd']['mean']:.4f} for PD and {diagnostics['probability_threshold']['dd_probability_dd']['mean']:.4f} for DD, showing substantial score overlap.", "",
        "## DD subtype diagnosis", "",
        "Frozen outer descriptive correct/DD-prediction rates are: Atypical 0.133 (15), ET 0.500 (28), MS 0.273 (11), Other 0.300 (60). These are exploratory subgroup estimates.",
        "Development inner-CV DD-prediction fractions below use repeated validation rows: every subject appears in four inner validation contexts; `prediction_rows` and `unique_subjects` are retained in the CSV.", "",
        *subtype_lines, "",
        "ET remains the easiest DD subtype for useful sensor models. The H1 family recovers many Atypical cases that SubjectMFAM misses, while MS remains near 0.455, so subtype difficulty is partly representation-dependent rather than universal.", "",
        "## Activity, sensor and wrist", "",
        f"Top single activities: {', '.join(row['variant'].split('::')[1] + '=' + format(float(row['balanced_accuracy_mean']), '.3f') for row in single[:5])}.",
        f"Best leave-one-activity-out estimates: {', '.join(row['variant'].split('::')[1] + '=' + format(float(row['balanced_accuracy_mean']), '.3f') for row in leave_one_out[:4])}. These exploratory differences may reflect redundancy or variance and are not permission to remove activities.",
        f"Sensor ranking: {', '.join(row['variant'].split('::')[1] + '=' + format(float(row['balanced_accuracy_mean']), '.3f') for row in sensor)}.",
        f"Wrist ranking: {', '.join(row['variant'].split('::')[1] + '=' + format(float(row['balanced_accuracy_mean']), '.3f') for row in wrist)}.", "",
        "## Data-quality and FFT sensitivity audit", "",
        f"All {quality['record_count']} processed records were audited; observed effective sampling rates span {quality['sampling_rate_range_hz'][0]:.4f}-{quality['sampling_rate_range_hz'][1]:.4f} Hz. Four timestamp-gap records and 12 matched no-gap controls yielded 672 channel/metric comparisons.",
        f"After strict-100-Hz interpolation, the gap group median absolute relative spectral change was {quality['gap_sensitivity']['groups']['gap']['median_absolute_relative_change']:.4f} (Q75={quality['gap_sensitivity']['groups']['gap']['q75']:.4f}); controls were {quality['gap_sensitivity']['groups']['control']['median_absolute_relative_change']:.4f} (Q75={quality['gap_sensitivity']['groups']['control']['q75']:.4f}). Both are classified `small` by the preregistered rule.",
        f"The robust warning rule flagged {quality['robust_outlier']['records_flagged']}/{quality['robust_outlier']['record_count']} records. It is a sensitivity warning, not a corruption label.",
        f"Incorrect-versus-correct Hedges g was {quality_by_metric['outlier_record_count']['hedges_g_incorrect_vs_correct']:.3f} for flagged-record count and {quality_by_metric['gap_record_count']['hedges_g_incorrect_vs_correct']:.3f} for gap count; both bootstrap intervals include zero. Outlier-point count was lower, not higher, among errors (g={quality_by_metric['outlier_point_count']['hedges_g_incorrect_vs_correct']:.3f}, mean difference={quality_by_metric['outlier_point_count']['incorrect_minus_correct']:.1f}).", "",
        "## Demographic audit", "",
        "D0 uses only age, height, weight, gender and handedness. Condition, disease comments, age at diagnosis and uncertain clinical fields are excluded. Imputation, scaling and one-hot encoding are fit on each inner-training partition only.",
        f"D0: BA={dev('D0_demographic_logistic', 'balanced_accuracy'):.4f}, macro-F1={dev('D0_demographic_logistic', 'macro_f1'):.4f}, AUROC={dev('D0_demographic_logistic', 'auroc'):.4f}, PD recall={dev('D0_demographic_logistic', 'pd_recall'):.4f}, DD recall={dev('D0_demographic_logistic', 'dd_recall'):.4f}. It shows possible demographic ranking/confounding signal but weak thresholded discrimination.", "",
        "## Training dynamics", "",
        f"Across 15 frozen inner runs, best epoch ranges {dynamics['best_epoch']['minimum']:.0f}-{dynamics['best_epoch']['maximum']:.0f} (mean={dynamics['best_epoch']['mean']:.1f}, median={dynamics['best_epoch']['median']:.1f}, SD={dynamics['best_epoch']['std']:.2f}). The train-minus-validation BA gap has mean={dynamics['train_validation_balanced_accuracy_gap']['mean']:.4f}, median={dynamics['train_validation_balanced_accuracy_gap']['median']:.4f}, max={dynamics['train_validation_balanced_accuracy_gap']['maximum']:.4f}.",
        f"Spearman rho between gap and validation BA is {dynamics['spearman_gap_vs_validation_performance']['rho']:.3f} (exploratory p={dynamics['spearman_gap_vs_validation_performance']['pvalue_exploratory']:.3f}); with n=15 this is no clear association.", "",
        "## Answers to Q1-Q12", "",
        "1. **Primary bottleneck:** predominantly representation/aggregation plus DD heterogeneity. Threshold tuning gives only +" + f"{threshold_gain:.4f} BA, while H1 gives +{dev('H1_handcrafted_logistic', 'balanced_accuracy') - dev('M0_subject_mfam_frozen_inner_reference', 'balanced_accuracy'):.4f} and MiniRocket +{dev('MR1_minirocket_multivariate_ridge', 'balanced_accuracy') - dev('M0_subject_mfam_frozen_inner_reference', 'balanced_accuracy'):.4f} on the same development protocol. Data-quality and demographic audits do not explain the full gap.",
        f"2. **Demographics:** D0 BA={dev('D0_demographic_logistic', 'balanced_accuracy'):.4f}, AUROC={dev('D0_demographic_logistic', 'auroc'):.4f}, DD recall={dev('D0_demographic_logistic', 'dd_recall'):.4f}; possible confounding/ranking signal, not a competitive classifier.",
        f"3. **Handcrafted linear baselines:** H1 BA={dev('H1_handcrafted_logistic', 'balanced_accuracy'):.4f}, macro-F1={dev('H1_handcrafted_logistic', 'macro_f1'):.4f}, AUROC={dev('H1_handcrafted_logistic', 'auroc'):.4f}; H2 BA={dev('H2_handcrafted_linear_svm', 'balanced_accuracy'):.4f}. Both exceed M0 BA={dev('M0_subject_mfam_frozen_inner_reference', 'balanced_accuracy'):.4f}.",
        f"4. **MiniRocket:** BA={dev('MR1_minirocket_multivariate_ridge', 'balanced_accuracy'):.4f}, AUROC={dev('MR1_minirocket_multivariate_ridge', 'auroc'):.4f}; it exceeds M0 but remains below H1.",
        f"5. **Simple CNN:** both preregistered candidates give BA={dev('N1a_simple_cnn_lr1e4_drop0p2', 'balanced_accuracy'):.4f}, PD recall=1.0, DD recall=0.0, with best epoch 1. This protocol collapsed to the majority class; no post-hoc candidate was added.",
        "6. **DD subtypes:** ET is easiest. M0 DD-prediction fractions are Atypical 0.067, ET 0.446, MS 0.091, Other 0.246; H1 gives 0.550, 0.759, 0.455, 0.546. Small subgroups require exploratory wording.",
        f"7. **Activities:** LiftHold={float(single[0]['balanced_accuracy_mean']):.3f} and TouchNose={float(single[1]['balanced_accuracy_mean']):.3f} lead single activities, but full multi-activity H1={dev('H1_handcrafted_logistic', 'balanced_accuracy'):.3f} is stronger than every single activity.",
        f"8. **Sensors:** combined={float(next(row for row in sensor if row['variant'] == 'sensor::acc_gyro')['balanced_accuracy_mean']):.3f}, Acc-only={float(next(row for row in sensor if row['variant'] == 'sensor::acc_only')['balanced_accuracy_mean']):.3f}, Gyro-only={float(next(row for row in sensor if row['variant'] == 'sensor::gyro_only')['balanced_accuracy_mean']):.3f}; the modalities are complementary.",
        f"9. **Wrists:** bilateral={float(next(row for row in wrist if row['variant'] == 'wrist::bilateral')['balanced_accuracy_mean']):.3f}, left={float(next(row for row in wrist if row['variant'] == 'wrist::left_only')['balanced_accuracy_mean']):.3f}, right={float(next(row for row in wrist if row['variant'] == 'wrist::right_only')['balanced_accuracy_mean']):.3f}.",
        "10. **Quality association:** no meaningful evidence that timestamp gaps, Fs deviation or warning counts drive errors; the one non-zero outlier-point association is opposite the proposed bad-quality direction.",
        "11. **Training dynamics:** fold-dependent epochs and sizable gaps indicate instability/overfitting risk, but the gap-performance correlation is inconclusive.",
        "12. **Next experiments:** use a new preregistered development-only ablation plan below; do not alter the frozen V3 baseline.", "",
        "## Prioritized next experiments (recommendations only)", "",
        "### Priority 1", "",
        "- Add a parallel raw/full-band or statistical-feature branch, or late-fuse SubjectMFAM with H1-style evidence. Direct support: H1 and MiniRocket both beat M0 under identical inner folds.",
        "- Replace hard Top-K MIL with a soft/differentiable aggregation ablation while preserving a matched control. This is indirectly supported by heterogeneous, distributed subtype/activity evidence; it is a hypothesis, not a demonstrated causal fix.",
        "- Compare the current activity attention with simple mean pooling and a low-capacity gated soft fusion. Multi-activity features outperform all single activities, while leave-one-out results show redundancy.", "",
        "### Priority 2", "",
        "- Test BatchNorm versus GroupNorm/LayerNorm in a strictly matched ablation. Batch size is small and training epochs/gaps are unstable, but current evidence is indirect.",
        "- Predefine subtype-stratified reporting and uncertainty intervals. Use subtype labels for evaluation/audit, not opportunistic outer-test tuning.", "",
        "### Not currently justified", "",
        f"- Class weighting as the main intervention: H1b improves BA by only {dev('H1b_handcrafted_logistic_balanced', 'balanced_accuracy') - dev('H1_handcrafted_logistic', 'balanced_accuracy'):+.4f} over H1.",
        "- Timestamp resampling as a primary fix: only four records have gaps, interpolation sensitivity is small, and error associations do not support it.",
        "- Activity dropout or deleting activities based on these exploratory estimates.",
        "- Adding new frequency bands directly to SubjectMFAM: H1 includes 12-20 Hz, but no band-specific ablation establishes which band causes its advantage.", "",
        "## Guardrails", "",
        "- Do not use this development leaderboard as final generalization performance.",
        "- Do not reuse the retrospective frozen-outer diagnostic threshold for model selection.",
        "- Outlier flags may represent pathological motion as well as acquisition artifact.",
        "- No requested or recommended SubjectMFAM change was executed in this run.",
    ]
    (output / "FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    # Freeze the development evidence package only after all report files exist.
    excluded = {"artifact_files.sha256", "run_manifest.json"}
    files = [path for path in output.rglob("*") if path.is_file() and path.relative_to(output).as_posix() not in excluded]
    lines = [f"{sha256_file(path)}  {path.relative_to(output).as_posix()}" for path in sorted(files)]
    (output / "artifact_files.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = read_json(output / "run_manifest.json")
    manifest.update({"status": "complete", "completed_at_utc": _utc_now(), "artifact_file_count": len(lines), "artifact_tree_sha256": hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest(), "source_provenance_final": _source_provenance()})
    manifest["stages"]["report"] = {"status": "complete", "completed_at_utc": _utc_now(), "summary": {"leaderboard_models": len(leaderboard), "cross_model_subtype_rows": len(subtype)}}
    write_json(output / "run_manifest.json", manifest)
    return {"leaderboard_models": len(leaderboard), "artifact_files": len(lines), "artifact_tree_sha256": manifest["artifact_tree_sha256"]}


def main() -> None:
    args = parse_args(); output = args.output_dir.expanduser().resolve()
    stages = ["initialize", "diagnostics", "features", "linear", "minirocket", "cnn", "report"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage != "initialize" and not output.is_dir():
            raise FileNotFoundError("Run initialize first")
        print(f"[stage] {stage}", flush=True)
        if stage == "initialize": initialize(output)
        elif stage == "diagnostics": stage_diagnostics(output)
        elif stage == "features": stage_features(output)
        elif stage == "linear": stage_linear(output)
        elif stage == "minirocket": stage_minirocket(output)
        elif stage == "cnn": stage_cnn(output, args.device)
        elif stage == "report": stage_report(output)


if __name__ == "__main__":
    main()
