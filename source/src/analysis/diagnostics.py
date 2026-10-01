from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats

from src.analysis.common import (
    binary_metrics,
    calibration_table,
    frozen_outer_assignments,
    load_patients,
    read_csv,
    read_json,
    subject_condition_map,
    write_csv,
    write_json,
)
from src.analysis.features import CHANNELS, spectral_features


FROZEN_EXPECTED_MANIFEST = "af5fdda01ceed8f90944f9a25e3a630b09ad17e07ece5f1d32dfe3d147271321"
TREMOR_RELATED_ACTIVITIES = {"Relaxed", "RelaxedTask", "StretchHold", "HoldWeight"}


def _describe(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {key: None for key in ("n", "mean", "median", "std", "q25", "q75", "iqr", "minimum", "maximum")}
    q25, q75 = np.percentile(array, [25, 75])
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _histogram(values: Sequence[float], bins: int = 10) -> list[dict[str, Any]]:
    counts, edges = np.histogram(np.asarray(values, dtype=np.float64), bins=np.linspace(0, 1, bins + 1))
    return [
        {"lower": float(edges[i]), "upper": float(edges[i + 1]), "count": int(counts[i])}
        for i in range(bins)
    ]


def _subtype_rows(predictions: Sequence[Mapping[str, Any]], group_key: str = "condition") -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row[group_key])].append(row)
    rows: list[dict[str, Any]] = []
    for group in sorted(grouped):
        items = grouped[group]
        pdd = [float(item["probability_dd"]) for item in items]
        confidence = [max(value, 1.0 - value) for value in pdd]
        targets = np.asarray([int(item["target"]) for item in items])
        predicted = np.asarray([int(item["prediction"]) for item in items])
        correct = predicted == targets
        description = _describe(pdd)
        rows.append(
            {
                "group": group,
                "n": len(items),
                "unique_subjects": len({str(item["subject_id"]) for item in items}),
                "mean_probability_dd": description["mean"],
                "median_probability_dd": description["median"],
                "std_probability_dd": description["std"],
                "q25_probability_dd": description["q25"],
                "q75_probability_dd": description["q75"],
                "iqr_probability_dd": description["iqr"],
                "correct_rate": float(correct.mean()),
                "predicted_as_dd_proportion": float(predicted.mean()),
                "confidence_mean": float(np.mean(confidence)),
                "confidence_q25": float(np.percentile(confidence, 25)),
                "confidence_median": float(np.median(confidence)),
                "confidence_q75": float(np.percentile(confidence, 75)),
                "confidence_histogram": json.dumps(_histogram(confidence), separators=(",", ":")),
                "interpretation_scope": "descriptive_exploratory",
            }
        )
    return rows


def _diagnostic_threshold(targets: np.ndarray, probability_dd: np.ndarray) -> dict[str, Any]:
    candidates = np.unique(np.concatenate([[0.0], probability_dd, [1.0]]))
    best: tuple[float, float] | None = None
    for threshold in candidates:
        score = binary_metrics(targets, (probability_dd >= threshold).astype(int), probability_dd=probability_dd)[
            "balanced_accuracy"
        ]
        candidate = (float(score), float(threshold))
        if best is None or candidate[0] > best[0] or (candidate[0] == best[0] and abs(candidate[1] - 0.5) < abs(best[1] - 0.5)):
            best = candidate
    assert best is not None
    return {
        "scope": "diagnostic_only_not_for_model_selection",
        "threshold": best[1],
        "balanced_accuracy": best[0],
        "candidate_count": int(candidates.size),
    }


def _frozen_predictions(frozen_dir: Path, conditions: Mapping[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split = read_json(frozen_dir / "frozen_split.json")
    assignments = frozen_outer_assignments(split)
    summary = read_json(frozen_dir / "nested_cv_summary.json")
    thresholds = {index: float(item["threshold"]) for index, item in enumerate(summary["outer_folds"])}
    rows: list[dict[str, Any]] = []
    for raw in read_csv(frozen_dir / "outer_test_predictions_all_folds.csv"):
        subject = raw["subject_id"]
        target = int(raw["target"])
        probability_dd = float(raw["probability_dd"])
        fold = assignments[subject]
        threshold = thresholds[fold]
        prediction = int(probability_dd >= threshold)
        condition = conditions[subject]
        rows.append(
            {
                "subject_id": subject,
                "target": target,
                "diagnosis": "DD" if target else "PD",
                "condition": condition,
                "outer_fold": fold,
                "probability_pd": float(raw["probability_pd"]),
                "probability_dd": probability_dd,
                "selected_threshold": threshold,
                "prediction": prediction,
                "predicted_label": "DD" if prediction else "PD",
                "correct": bool(prediction == target),
                "error_category": ("DD" if target else "PD") + "_to_" + ("DD" if prediction else "PD"),
                "margin_to_threshold": float(probability_dd - threshold),
                "confidence": float(max(probability_dd, 1.0 - probability_dd)),
                "analysis_scope": "frozen_outer_test_descriptive_only",
            }
        )
    if len(rows) != 390 or len({row["subject_id"] for row in rows}) != 390:
        raise ValueError("Frozen predictions must cover 390 unique subjects")
    return rows, summary


def _load_quality_records(path: Path, conditions: Mapping[str, str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            time = np.loadtxt(record["source_path"], delimiter=",", usecols=0, dtype=np.float64)
            dt = np.diff(time)
            median_dt = float(np.median(dt))
            gap_mask = dt > 10.0 * median_dt
            max_dt = float(np.max(dt))
            condition = conditions[str(record["subject_id"])]
            processed_counts = [int(value) for value in record["processed_channels"]["robust_outlier_count"]]
            records.append(
                {
                    "subject_id": str(record["subject_id"]),
                    "diagnosis": "PD" if condition == "Parkinson's" else "DD",
                    "subtype": condition,
                    "activity": str(record["activity"]),
                    "wrist": str(record["wrist"]),
                    "nominal_length": int(record["raw_length"]),
                    "processed_length": int(record["processed_length"]),
                    "median_dt_seconds": median_dt,
                    "effective_sampling_rate_hz": float(1.0 / median_dt),
                    "fs_deviation_from_100_hz": float(1.0 / median_dt - 100.0),
                    "absolute_fs_deviation_hz": float(abs(1.0 / median_dt - 100.0)),
                    "max_dt_seconds": max_dt,
                    "max_dt_over_median_dt": float(max_dt / median_dt),
                    "gap_count_over_10x_median": int(gap_mask.sum()),
                    "gap_duration_excess_seconds": float(np.sum(dt[gap_mask] - median_dt)),
                    "outlier_point_count": int(sum(processed_counts)),
                    "outlier_channel_count": int(sum(value > 0 for value in processed_counts)),
                    "outlier_record_flag": int(any(value > 0 for value in processed_counts)),
                    "processed_outlier_counts": processed_counts,
                    "processed_path": str(record["processed_path"]),
                    "source_path": str(record["source_path"]),
                    "trimmed_start_samples": int(record["trimmed_start_samples"]),
                }
            )
    if len(records) != 8580:
        raise ValueError(f"Expected 8580 quality records, found {len(records)}")
    return records


def _aggregate_rate(rows: Sequence[Mapping[str, Any]], keys: Sequence[str], value_key: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(int(row[value_key]))
    output: list[dict[str, Any]] = []
    for group, values in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        entry = {key: value for key, value in zip(keys, group)}
        entry.update({"n": len(values), "flag_count": int(sum(values)), "flag_rate": float(np.mean(values))})
        output.append(entry)
    return output


def _outlier_analysis(records: Sequence[Mapping[str, Any]], output_dir: Path) -> dict[str, Any]:
    long_rows: list[dict[str, Any]] = []
    for record in records:
        for channel_index, channel in enumerate(CHANNELS):
            count = int(record["processed_outlier_counts"][channel_index])
            long_rows.append(
                {
                    "subject_id": record["subject_id"],
                    "diagnosis": record["diagnosis"],
                    "subtype": record["subtype"],
                    "activity": record["activity"],
                    "activity_group": "tremor_related" if record["activity"] in TREMOR_RELATED_ACTIVITIES else "other",
                    "wrist": record["wrist"],
                    "sensor": "Acc" if channel.startswith("Acc") else "Gyro",
                    "channel": channel,
                    "outlier_count": count,
                    "outlier_flag": int(count > 0),
                }
            )
    write_csv(output_dir / "robust_outlier_channel_records.csv", long_rows, list(long_rows[0]))
    record_rates = _aggregate_rate(records, ("diagnosis", "subtype", "activity", "wrist"), "outlier_record_flag")
    channel_rates = _aggregate_rate(
        long_rows, ("diagnosis", "subtype", "activity", "activity_group", "wrist", "sensor", "channel"), "outlier_flag"
    )
    sensor_rates = _aggregate_rate(long_rows, ("diagnosis", "subtype", "activity_group", "sensor"), "outlier_flag")
    subject_flags: dict[tuple[str, str, str], int] = defaultdict(int)
    for record in records:
        key = (str(record["subject_id"]), str(record["diagnosis"]), str(record["subtype"]))
        subject_flags[key] = max(subject_flags[key], int(record["outlier_record_flag"]))
    subject_rows = [
        {"subject_id": key[0], "diagnosis": key[1], "subtype": key[2], "outlier_subject_flag": value}
        for key, value in sorted(subject_flags.items())
    ]
    subject_rates = _aggregate_rate(subject_rows, ("diagnosis", "subtype"), "outlier_subject_flag")
    write_csv(output_dir / "robust_outlier_record_rates.csv", record_rates, list(record_rates[0]))
    write_csv(output_dir / "robust_outlier_channel_rates.csv", channel_rates, list(channel_rates[0]))
    write_csv(output_dir / "robust_outlier_sensor_rates.csv", sensor_rates, list(sensor_rates[0]))
    write_csv(output_dir / "robust_outlier_subject_rates.csv", subject_rates, list(subject_rates[0]))
    return {
        "warning_definition": "abs(x-median)/(1.4826*MAD) > 20 per processed channel; zero-MAD channels are not flagged",
        "record_count": len(records),
        "records_flagged": int(sum(int(row["outlier_record_flag"]) for row in records)),
        "channel_rows": len(long_rows),
        "tremor_related_activity_definition": sorted(TREMOR_RELATED_ACTIVITIES),
        "interpretation": "A flag is not automatically data corruption; it may indicate acquisition artifact or pathological motion.",
    }


def _gap_sensitivity(records: Sequence[Mapping[str, Any]], output_dir: Path) -> dict[str, Any]:
    gaps = [row for row in records if int(row["gap_count_over_10x_median"]) > 0]
    controls: list[Mapping[str, Any]] = []
    matches: list[dict[str, Any]] = []
    for gap in gaps:
        candidates = [
            row for row in records
            if row["activity"] == gap["activity"] and row["wrist"] == gap["wrist"]
            and row["subtype"] == gap["subtype"] and int(row["gap_count_over_10x_median"]) == 0
            and row["subject_id"] != gap["subject_id"]
        ]
        candidates.sort(key=lambda row: (float(row["absolute_fs_deviation_hz"]), str(row["subject_id"])))
        for control in candidates[:3]:
            controls.append(control)
            matches.append(
                {
                    "gap_subject_id": gap["subject_id"], "activity": gap["activity"], "wrist": gap["wrist"],
                    "subtype": gap["subtype"], "control_subject_id": control["subject_id"],
                    "matching": "same activity, wrist, and source condition; lowest Fs deviation; no >10x gap",
                }
            )
    write_csv(output_dir / "gap_matched_controls.csv", matches, list(matches[0]))
    selected: list[tuple[str, Mapping[str, Any]]] = [("gap", row) for row in gaps] + [("control", row) for row in controls]
    comparison: list[dict[str, Any]] = []
    spectral_names = (
        "dominant_frequency_hz_0p5_20", "dominant_power_0p5_20", "spectral_entropy_0p5_20",
        "total_power_0p5_20", "band_power_0p5_3", "band_power_3_7", "band_power_7_12",
    )
    for record_type, record in selected:
        processed = np.load(record["processed_path"], allow_pickle=False).astype(np.float64)
        raw_time = np.loadtxt(record["source_path"], delimiter=",", usecols=0, dtype=np.float64)
        time = raw_time[int(record["trimmed_start_samples"]):]
        grid = np.arange(time[0], time[-1] + 1e-12, 0.01)
        resampled = np.vstack([np.interp(grid, time, channel) for channel in processed])
        for channel_index, channel in enumerate(CHANNELS):
            existing = spectral_features(processed[channel_index], 100.0)
            strict = spectral_features(resampled[channel_index], 100.0)
            for metric in spectral_names:
                a = float(existing[metric]); b = float(strict[metric])
                relative = float((b - a) / max(abs(a), 1e-12))
                comparison.append(
                    {
                        "record_type": record_type, "subject_id": record["subject_id"],
                        "diagnosis": record["diagnosis"], "subtype": record["subtype"],
                        "activity": record["activity"], "wrist": record["wrist"], "channel": channel,
                        "metric": metric, "existing_value": a, "resampled_value": b,
                        "relative_change": relative, "absolute_relative_change": abs(relative),
                        "interpolation": "linear on processed samples using original timestamps; strict 100 Hz grid",
                    }
                )
    write_csv(output_dir / "fft_timestamp_sensitivity.csv", comparison, list(comparison[0]))
    classification: dict[str, Any] = {}
    for record_type in ("gap", "control"):
        values = [row["absolute_relative_change"] for row in comparison if row["record_type"] == record_type]
        median = float(np.median(values)); q75 = float(np.percentile(values, 75))
        category = "negligible" if median < 0.01 else "small" if median < 0.05 else "potentially_material"
        classification[record_type] = {"median_absolute_relative_change": median, "q75": q75, "classification": category}
    return {"gap_record_count": len(gaps), "matched_control_count": len(controls), "comparison_rows": len(comparison), "classification_rule": "median absolute relative change <1%=negligible, 1-5%=small, >=5%=potentially material", "groups": classification}


def _hedges_g(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    pooled = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2))
    if pooled == 0:
        return 0.0
    d = (float(np.mean(a)) - float(np.mean(b))) / pooled
    correction = 1.0 - 3.0 / (4.0 * (len(a) + len(b)) - 9.0)
    return float(d * correction)


def _bootstrap_difference(a: np.ndarray, b: np.ndarray, seed: int = 42, repeats: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = np.asarray([
        rng.choice(a, size=len(a), replace=True).mean() - rng.choice(b, size=len(b), replace=True).mean()
        for _ in range(repeats)
    ])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _quality_error_association(
    predictions: list[dict[str, Any]], records: Sequence[Mapping[str, Any]], patients: Mapping[str, Mapping[str, Any]], output_dir: Path
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["subject_id"])].append(row)
    error_rows: list[dict[str, Any]] = []
    for prediction in predictions:
        subject = prediction["subject_id"]
        quality = grouped[subject]
        patient = patients[subject]
        error_rows.append(
            {
                **prediction,
                "outlier_record_count": int(sum(int(row["outlier_record_flag"]) for row in quality)),
                "outlier_channel_count": int(sum(int(row["outlier_channel_count"]) for row in quality)),
                "outlier_point_count": int(sum(int(row["outlier_point_count"]) for row in quality)),
                "gap_record_count": int(sum(int(row["gap_count_over_10x_median"]) > 0 for row in quality)),
                "gap_count": int(sum(int(row["gap_count_over_10x_median"]) for row in quality)),
                "max_absolute_fs_deviation_hz": float(max(float(row["absolute_fs_deviation_hz"]) for row in quality)),
                "mean_absolute_fs_deviation_hz": float(np.mean([float(row["absolute_fs_deviation_hz"]) for row in quality])),
                "age": patient.get("age"), "height": patient.get("height"), "weight": patient.get("weight"),
                "gender": patient.get("gender"), "handedness": patient.get("handedness"),
            }
        )
    write_csv(output_dir / "frozen_error_table.csv", error_rows, list(error_rows[0]))
    association: list[dict[str, Any]] = []
    for metric in ("outlier_record_count", "outlier_channel_count", "outlier_point_count", "gap_record_count", "max_absolute_fs_deviation_hz", "mean_absolute_fs_deviation_hz"):
        correct = np.asarray([float(row[metric]) for row in error_rows if row["correct"]], dtype=np.float64)
        incorrect = np.asarray([float(row[metric]) for row in error_rows if not row["correct"]], dtype=np.float64)
        ci_low, ci_high = _bootstrap_difference(incorrect, correct)
        association.append(
            {
                "metric": metric, "correct_n": len(correct), "incorrect_n": len(incorrect),
                "correct_mean": float(correct.mean()), "incorrect_mean": float(incorrect.mean()),
                "incorrect_minus_correct": float(incorrect.mean() - correct.mean()),
                "bootstrap_95ci_low": ci_low, "bootstrap_95ci_high": ci_high,
                "hedges_g_incorrect_vs_correct": _hedges_g(incorrect, correct),
                "scope": "frozen_outer_descriptive_not_for_tuning",
            }
        )
    write_csv(output_dir / "quality_error_associations.csv", association, list(association[0]))
    return association


def _demographic_audit(patients: Mapping[str, Mapping[str, Any]], output_dir: Path) -> dict[str, Any]:
    fields = sorted({key for patient in patients.values() for key in patient})
    allowed = {"age", "height", "weight", "gender", "handedness"}
    forbidden = {"condition", "disease_comment", "age_at_diagnosis"}
    rows: list[dict[str, Any]] = []
    for field in fields:
        values = [patient.get(field) for patient in patients.values()]
        nonmissing = [value for value in values if value is not None and value != ""]
        if field in allowed:
            decision = "allowed_basic_demographic"
        elif field in forbidden or field in {"resource_type", "id", "study_id"}:
            decision = "excluded_identifier_or_diagnosis_leakage"
        else:
            decision = "conservatively_excluded_uncertain_or_nonbasic"
        rows.append(
            {
                "field": field, "decision": decision, "n_nonmissing": len(nonmissing),
                "n_missing": len(values) - len(nonmissing), "unique_nonmissing": len({str(value) for value in nonmissing}),
                "examples": json.dumps([str(value) for value in nonmissing[:5]], ensure_ascii=False),
            }
        )
    write_csv(output_dir / "demographic_field_audit.csv", rows, list(rows[0]))
    result = {
        "allowed_numeric": ["age", "height", "weight"],
        "allowed_categorical": ["gender", "handedness"],
        "excluded_leakage": sorted(forbidden),
        "excluded_uncertain": sorted(set(fields) - allowed - forbidden - {"resource_type", "id", "study_id"}),
        "principle": "Only basic pre-diagnostic demographics are allowed; uncertain fields are excluded.",
    }
    write_json(output_dir / "demographic_audit.json", result)
    return result


def _training_dynamics(frozen_dir: Path, output_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for outer in range(5):
        for inner in range(3):
            stage = frozen_dir / f"outer_{outer}" / f"inner_{inner}"
            status = read_json(stage / "stage_status.json")
            best_epoch = int(status["summary"]["best_epoch"])
            logs = [json.loads(line) for line in (stage / "logs" / "epochs.jsonl").read_text().splitlines()]
            best = next(item for item in logs if int(item["epoch"]) == best_epoch - 1)
            final = logs[-1]
            rows.append(
                {
                    "outer_context": outer, "inner_fold": inner, "best_epoch": best_epoch,
                    "epochs_run": len(logs), "best_train_loss": best["train"]["loss"],
                    "best_validation_loss": best["validation"]["loss"],
                    "best_train_balanced_accuracy": best["train"]["balanced_accuracy"],
                    "best_validation_balanced_accuracy": best["validation"]["balanced_accuracy"],
                    "best_train_macro_f1": best["train"]["macro_f1"],
                    "best_validation_macro_f1": best["validation"]["macro_f1"],
                    "balanced_accuracy_gap": best["train"]["balanced_accuracy"] - best["validation"]["balanced_accuracy"],
                    "macro_f1_gap": best["train"]["macro_f1"] - best["validation"]["macro_f1"],
                    "final_train_loss": final["train"]["loss"], "final_validation_loss": final["validation"]["loss"],
                    "final_patience_count": final["patience_count"],
                }
            )
    write_csv(output_dir / "subject_mfam_training_dynamics.csv", rows, list(rows[0]))
    gaps = np.asarray([row["balanced_accuracy_gap"] for row in rows])
    performance = np.asarray([row["best_validation_balanced_accuracy"] for row in rows])
    epochs = np.asarray([row["best_epoch"] for row in rows])
    rho, pvalue = stats.spearmanr(gaps, performance)
    medians = [int(np.median([row["best_epoch"] for row in rows if row["outer_context"] == outer])) for outer in range(5)]
    summary = {
        "fold_count": 15, "best_epoch": _describe(epochs), "train_validation_balanced_accuracy_gap": _describe(gaps),
        "best_validation_balanced_accuracy": _describe(performance),
        "spearman_gap_vs_validation_performance": {"rho": float(rho), "pvalue_exploratory": float(pvalue)},
        "outer_final_epoch_medians": medians,
        "scope": "frozen_training_log_diagnostic_no_model_change",
    }
    write_json(output_dir / "subject_mfam_training_dynamics_summary.json", summary)
    return summary


def run_diagnostics(
    *, frozen_dir: str | Path, processed_root: str | Path, raw_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    frozen = Path(frozen_dir).resolve(); processed = Path(processed_root).resolve(); raw = Path(raw_root).resolve(); output = Path(output_dir).resolve()
    if frozen == output or frozen in output.parents:
        raise ValueError("Analysis output must not be inside frozen baseline")
    output.mkdir(parents=True, exist_ok=False)
    conditions = subject_condition_map(processed / "manifests")
    patients = load_patients(raw / "patients", set(conditions))
    predictions, frozen_summary = _frozen_predictions(frozen, conditions)
    write_csv(output / "frozen_outer_predictions_enriched.csv", predictions, list(predictions[0]))
    subtype = _subtype_rows(predictions)
    write_csv(output / "frozen_subtype_summary.csv", subtype, list(subtype[0]))
    fold_subtype = _subtype_rows([{**row, "fold_condition": f"fold{row['outer_fold']}|{row['condition']}"} for row in predictions], "fold_condition")
    write_csv(output / "frozen_fold_subtype_summary.csv", fold_subtype, list(fold_subtype[0]))
    thresholds = [float(item["threshold"]) for item in frozen_summary["outer_folds"]]
    fold_probability: list[dict[str, Any]] = []
    for fold in range(5):
        for diagnosis in ("PD", "DD"):
            values = [row["probability_dd"] for row in predictions if row["outer_fold"] == fold and row["diagnosis"] == diagnosis]
            fold_probability.append({"outer_fold": fold, "diagnosis": diagnosis, **_describe(values)})
    write_csv(output / "frozen_fold_probability_distributions.csv", fold_probability, list(fold_probability[0]))
    y = np.asarray([row["target"] for row in predictions]); p = np.asarray([row["probability_dd"] for row in predictions]); pred = np.asarray([row["prediction"] for row in predictions])
    calibration_rows, reliability = calibration_table(y, p)
    write_csv(output / "frozen_calibration_curve.csv", calibration_rows, list(calibration_rows[0]))
    probability_summary = {
        "thresholds": {**_describe(thresholds), "values": thresholds},
        "frozen_selected_threshold_metrics": binary_metrics(y, pred, probability_dd=p),
        "reliability": reliability,
        "diagnostic_global_threshold": _diagnostic_threshold(y, p),
        "pd_probability_dd": _describe(p[y == 0]), "dd_probability_dd": _describe(p[y == 1]),
        "conclusion_guardrail": "Retrospective threshold result is diagnostic only and cannot be used for model selection.",
    }
    write_json(output / "frozen_probability_threshold_diagnostics.json", probability_summary)
    training = _training_dynamics(frozen, output)
    quality_records = _load_quality_records(processed / "quality_records.jsonl", conditions)
    flat_quality = [{key: value for key, value in row.items() if key != "processed_outlier_counts"} for row in quality_records]
    write_csv(output / "record_quality_subject_activity_wrist.csv", flat_quality, list(flat_quality[0]))
    gaps = [row for row in flat_quality if row["gap_count_over_10x_median"] > 0]
    write_csv(output / "timestamp_gap_records.csv", gaps, list(gaps[0]))
    gap_sensitivity = _gap_sensitivity(quality_records, output)
    outliers = _outlier_analysis(quality_records, output)
    associations = _quality_error_association(predictions, quality_records, patients, output)
    demographic = _demographic_audit(patients, output)
    summary = {
        "scope": "frozen_outer_descriptive_diagnostics_only",
        "frozen_subject_count": 390,
        "frozen_metrics_recomputed": binary_metrics(y, pred, probability_dd=p),
        "subtype_summary": subtype,
        "probability_threshold": probability_summary,
        "training_dynamics": training,
        "quality": {
            "record_count": len(quality_records), "sampling_rate_range_hz": [min(row["effective_sampling_rate_hz"] for row in quality_records), max(row["effective_sampling_rate_hz"] for row in quality_records)],
            "gap_records": len(gaps), "gap_sensitivity": gap_sensitivity, "robust_outlier": outliers,
            "error_association": associations,
        },
        "demographic_audit": demographic,
    }
    write_json(output / "diagnostic_summary.json", summary)
    return summary
