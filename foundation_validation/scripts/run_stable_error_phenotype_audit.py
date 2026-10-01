#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


ACTIVITIES = ("CrossArms", "DrinkGlas", "HoldWeight", "LiftHold")
WRISTS = ("left", "right")
SENSORS = {"acc": slice(0, 3), "gyro": slice(3, 6)}
GROUPS = ("stable_correct", "unstable", "stable_error")
FORBIDDEN = ("outer_final", "outer_test", "outer-test", "final_nested_cv")
FREQUENCY_FEATURES = {
    "bandpower_0p5_3",
    "bandpower_3_7",
    "bandpower_7_12",
    "bandpower_12_25",
    "dominant_frequency_0p5_12",
    "spectral_entropy_0p5_25",
    "tremor_peak_ratio_3_7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stable-error phenotype and data audit")
    parser.add_argument("--deep-root", type=Path, default=Path("/home/zyt/deep_final"))
    parser.add_argument("--data-root", type=Path, default=Path("/home/zyt/MFAM/data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def assert_development_only(paths: Iterable[Path]) -> None:
    for path in paths:
        lowered = str(path.resolve()).lower()
        if any(token in lowered for token in FORBIDDEN):
            raise RuntimeError(f"Forbidden outer-information path: {path}")


def bh_fdr(values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(values), dtype=float)
    result = np.full_like(values, np.nan)
    valid = np.isfinite(values)
    if not valid.any():
        return result
    p = values[valid]
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    result[valid] = restored
    return result


def cliffs_delta(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if not len(left) or not len(right):
        return float("nan")
    u = mannwhitneyu(left, right, alternative="two-sided").statistic
    return float(2.0 * u / (len(left) * len(right)) - 1.0)


def compare_continuous(
    frame: pd.DataFrame,
    value_columns: list[str],
    strata: list[str],
    family: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = [((), frame)] if not strata else frame.groupby(strata, dropna=False)
    for key, part in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        context = dict(zip(strata, key))
        for value in value_columns:
            for comparison in ("stable_correct", "unstable"):
                a = pd.to_numeric(
                    part.loc[part.stability_group == "stable_error", value], errors="coerce"
                ).dropna().to_numpy(float)
                b = pd.to_numeric(
                    part.loc[part.stability_group == comparison, value], errors="coerce"
                ).dropna().to_numpy(float)
                if len(a) < 3 or len(b) < 3:
                    continue
                test = mannwhitneyu(a, b, alternative="two-sided")
                rows.append({
                    **context,
                    "family": family,
                    "feature": value,
                    "group_a": "stable_error",
                    "group_b": comparison,
                    "n_a": len(a),
                    "n_b": len(b),
                    "mean_a": float(np.mean(a)),
                    "mean_b": float(np.mean(b)),
                    "median_a": float(np.median(a)),
                    "median_b": float(np.median(b)),
                    "mean_difference": float(np.mean(a) - np.mean(b)),
                    "cliffs_delta": cliffs_delta(a, b),
                    "p_value": float(test.pvalue),
                })
    result = pd.DataFrame(rows)
    if len(result):
        result["q_value"] = bh_fdr(result.p_value)
    return result


def cramers_v(table: np.ndarray) -> float:
    if table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan")
    chi2 = chi2_contingency(table, correction=False).statistic
    n = table.sum()
    return float(math.sqrt((chi2 / n) / max(1, min(table.shape[0] - 1, table.shape[1] - 1))))


def categorical_audit(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    omnibus: list[dict[str, Any]] = []
    levels: list[dict[str, Any]] = []
    for label_name, part in frame.groupby("label_name"):
        for column in columns:
            clean = part[["stability_group", column]].copy()
            clean[column] = clean[column].fillna("Missing").astype(str)
            table = pd.crosstab(clean.stability_group, clean[column]).reindex(GROUPS, fill_value=0)
            table = table.loc[:, table.sum(axis=0) > 0]
            if table.shape[0] >= 2 and table.shape[1] >= 2:
                test = chi2_contingency(table, correction=False)
                omnibus.append({
                    "label_name": label_name,
                    "feature": column,
                    "n": int(table.to_numpy().sum()),
                    "levels": int(table.shape[1]),
                    "cramers_v": cramers_v(table.to_numpy()),
                    "p_value": float(test.pvalue),
                })
            for level in sorted(clean[column].unique()):
                for comparison in ("stable_correct", "unstable"):
                    a = clean.stability_group == "stable_error"
                    b = clean.stability_group == comparison
                    if a.sum() == 0 or b.sum() == 0:
                        continue
                    table2 = np.array([
                        [(a & (clean[column] == level)).sum(), (a & (clean[column] != level)).sum()],
                        [(b & (clean[column] == level)).sum(), (b & (clean[column] != level)).sum()],
                    ])
                    odds, p = fisher_exact(table2)
                    levels.append({
                        "label_name": label_name,
                        "feature": column,
                        "level": level,
                        "group_a": "stable_error",
                        "group_b": comparison,
                        "n_a": int(a.sum()),
                        "n_b": int(b.sum()),
                        "count_a": int(table2[0, 0]),
                        "count_b": int(table2[1, 0]),
                        "fraction_a": float(table2[0, 0] / max(1, a.sum())),
                        "fraction_b": float(table2[1, 0] / max(1, b.sum())),
                        "odds_ratio": float(odds),
                        "p_value": float(p),
                    })
    omni = pd.DataFrame(omnibus)
    level = pd.DataFrame(levels)
    if len(omni):
        omni["q_value"] = bh_fdr(omni.p_value)
    if len(level):
        level["q_value"] = bh_fdr(level.p_value)
    return omni, level


def load_subjects(deep_root: Path, raw_root: Path) -> pd.DataFrame:
    stability_path = deep_root / "artifacts/representation_diagnostics/v8_gn_stage2/subject_error_stability.csv"
    assert_development_only([stability_path])
    stability = pd.read_csv(stability_path, dtype={"subject_id": str})
    stability["subject_id"] = stability.subject_id.str.zfill(3)
    # The patient JSON is the canonical phenotype source for condition/subtype.
    stability = stability.drop(columns=["condition"], errors="ignore")
    rows = []
    keyword_map = {
        "comment_tremor": r"tremor",
        "comment_hypokinesia": r"hypokines|bradykines|akinetic",
        "comment_rigidity": r"rigid",
        "comment_dystonia": r"dystoni",
        "comment_gait_balance": r"gait|balance|postural|fall",
        "comment_drug_induced": r"drug.induced|medication.induced",
        "comment_functional": r"functional|psychogenic",
        "comment_vascular": r"vascular|infarct|stroke",
    }
    for path in sorted((raw_root / "patients").glob("patient_*.json")):
        item = json.loads(path.read_text())
        sid = str(item["id"]).zfill(3)
        row = {"subject_id": sid, **item}
        comment = str(item.get("disease_comment") or "").lower()
        for name, pattern in keyword_map.items():
            row[name] = bool(re.search(pattern, comment, flags=re.I))
        age = pd.to_numeric(item.get("age"), errors="coerce")
        diagnosis = pd.to_numeric(item.get("age_at_diagnosis"), errors="coerce")
        height = pd.to_numeric(item.get("height"), errors="coerce")
        weight = pd.to_numeric(item.get("weight"), errors="coerce")
        row["disease_duration"] = age - diagnosis if np.isfinite(age) and np.isfinite(diagnosis) else np.nan
        row["bmi"] = weight / ((height / 100.0) ** 2) if np.isfinite(height) and height > 0 and np.isfinite(weight) else np.nan
        observation = raw_root / "movement" / f"observation_{sid}.json"
        row["device_id"] = json.loads(observation.read_text()).get("device_id") if observation.exists() else None
        rows.append(row)
    subjects = stability.merge(pd.DataFrame(rows), on="subject_id", how="left", validate="one_to_one")
    subjects["label_name"] = subjects.label.map({0: "PD", 1: "DD"})
    return subjects


def spectral_features(signal: np.ndarray, fs: float = 100.0) -> dict[str, float]:
    nperseg = min(256, signal.shape[-1])
    frequencies, psd = welch(signal, fs=fs, axis=-1, nperseg=nperseg, detrend="constant")
    power = psd.sum(axis=0)
    total_mask = (frequencies >= 0.5) & (frequencies <= 25.0)
    total = float(np.trapezoid(power[total_mask], frequencies[total_mask])) + 1.0e-12
    result: dict[str, float] = {}
    for low, high, name in (
        (0.5, 3.0, "bandpower_0p5_3"),
        (3.0, 7.0, "bandpower_3_7"),
        (7.0, 12.0, "bandpower_7_12"),
        (12.0, 25.0, "bandpower_12_25"),
    ):
        mask = (frequencies >= low) & (frequencies < high)
        result[name] = float(np.trapezoid(power[mask], frequencies[mask]) / total)
    dominant_mask = (frequencies >= 0.5) & (frequencies <= 12.0)
    result["dominant_frequency_0p5_12"] = float(frequencies[dominant_mask][np.argmax(power[dominant_mask])])
    normalized = power[total_mask] / max(power[total_mask].sum(), 1.0e-12)
    result["spectral_entropy_0p5_25"] = float(
        -(normalized * np.log(normalized + 1.0e-12)).sum() / np.log(max(2, len(normalized)))
    )
    tremor = (frequencies >= 3.0) & (frequencies <= 7.0)
    result["tremor_peak_ratio_3_7"] = float(power[tremor].max() / (power[total_mask].mean() + 1.0e-12))
    return result


def basic_signal_features(signal: np.ndarray, fs: float = 100.0) -> dict[str, float]:
    centered = signal - signal.mean(axis=1, keepdims=True)
    vector = np.linalg.norm(centered, axis=0)
    delta = np.diff(signal, axis=1)
    delta_norm = np.linalg.norm(delta, axis=0)
    scale = np.median(np.abs(centered - np.median(centered, axis=1, keepdims=True)), axis=1)
    robust_z = np.abs(centered) / (1.4826 * scale[:, None] + 1.0e-8)
    result = {
        "motion_rms": float(np.sqrt(np.mean(centered ** 2))),
        "vector_magnitude_mean": float(vector.mean()),
        "vector_magnitude_p95": float(np.quantile(vector, 0.95)),
        "jerk_rms": float(np.sqrt(np.mean(delta ** 2)) * fs),
        "low_motion_fraction": float(np.mean(delta_norm < 1.0e-3)),
        "robust_outlier_fraction": float(np.mean(robust_z > 20.0)),
        "range_mean": float(np.mean(np.ptp(signal, axis=1))),
        "extreme_repeat_fraction": float(np.mean((signal == signal.min(axis=1, keepdims=True)) | (signal == signal.max(axis=1, keepdims=True)))),
    }
    result.update(spectral_features(centered, fs=fs))
    return result


def load_signal_features(processed_root: Path, subjects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subject_info = subjects.set_index("subject_id")[["label_name", "stability_group"]].to_dict("index")
    for activity in ACTIVITIES:
        for sid, info in subject_info.items():
            wrist_features: dict[tuple[str, str], dict[str, float]] = {}
            wrist_vectors: dict[tuple[str, str], np.ndarray] = {}
            for wrist in WRISTS:
                path = processed_root / "signals" / activity / sid / f"{wrist}.npy"
                signal = np.load(path, allow_pickle=False).astype(np.float64)
                for sensor, channel_slice in SENSORS.items():
                    selected = signal[channel_slice]
                    features = basic_signal_features(selected)
                    wrist_features[(wrist, sensor)] = features
                    wrist_vectors[(wrist, sensor)] = np.linalg.norm(
                        selected - selected.mean(axis=1, keepdims=True), axis=0
                    )
                    rows.append({
                        "subject_id": sid,
                        **info,
                        "activity": activity,
                        "sensor": sensor,
                        "scope": wrist,
                        **features,
                    })
            for sensor in SENSORS:
                left = wrist_features[("left", sensor)]
                right = wrist_features[("right", sensor)]
                rows.append({
                    "subject_id": sid,
                    **info,
                    "activity": activity,
                    "sensor": sensor,
                    "scope": "bilateral_mean",
                    **{key: 0.5 * (left[key] + right[key]) for key in left},
                })
                rows.append({
                    "subject_id": sid,
                    **info,
                    "activity": activity,
                    "sensor": sensor,
                    "scope": "bilateral_asymmetry",
                    **{key: abs(left[key] - right[key]) for key in left},
                })
                lv = wrist_vectors[("left", sensor)]
                rv = wrist_vectors[("right", sensor)]
                corr = float(np.corrcoef(lv, rv)[0, 1]) if np.std(lv) > 0 and np.std(rv) > 0 else 0.0
                rows.append({
                    "subject_id": sid,
                    **info,
                    "activity": activity,
                    "sensor": sensor,
                    "scope": "bilateral_relation",
                    "zero_lag_magnitude_correlation": corr,
                })
    return pd.DataFrame(rows)


def load_quality(processed_root: Path, subjects: pd.DataFrame) -> pd.DataFrame:
    subject_info = subjects.set_index("subject_id")[["label_name", "stability_group"]].to_dict("index")
    rows = []
    with (processed_root / "quality_records.jsonl").open() as stream:
        for line in stream:
            item = json.loads(line)
            sid = str(item["subject_id"]).zfill(3)
            if sid not in subject_info:
                continue
            raw = item["raw_channels"]
            processed = item["processed_channels"]
            raw_length = max(1, int(item["raw_length"]))
            processed_length = max(1, int(item["processed_length"]))
            rows.append({
                "subject_id": sid,
                **subject_info[sid],
                "activity": item["activity"],
                "wrist": item["wrist"],
                "activity_scope": "high_risk" if item["activity"] in ACTIVITIES else "other",
                "sampling_rate_error": float(item["relative_sampling_rate_error"]),
                "max_dt_ratio": float(item["max_dt_deviation_seconds"] / max(item["median_dt_seconds"], 1e-12)),
                "raw_outlier_fraction": float(sum(raw["robust_outlier_count"]) / (6 * raw_length)),
                "processed_outlier_fraction": float(sum(processed["robust_outlier_count"]) / (6 * processed_length)),
                "raw_zero_difference_fraction": float(np.mean(raw["zero_difference_fraction"])),
                "processed_zero_difference_fraction": float(np.mean(processed["zero_difference_fraction"])),
                "raw_maximum_constant_run": float(raw["maximum_constant_run"]),
                "processed_maximum_constant_run": float(processed["maximum_constant_run"]),
                "processed_duration_seconds": float(item["processed_duration_seconds"]),
            })
    records = pd.DataFrame(rows)
    values = [c for c in records.columns if c not in {
        "subject_id", "label_name", "stability_group", "activity", "wrist", "activity_scope"
    }]
    aggregated = records.groupby(
        ["subject_id", "label_name", "stability_group", "activity_scope"], as_index=False
    )[values].mean()
    all_aggregated = records.groupby(
        ["subject_id", "label_name", "stability_group"], as_index=False
    )[values].mean()
    all_aggregated["activity_scope"] = "all"
    return pd.concat([aggregated, all_aggregated], ignore_index=True)


def representation_audit(deep_root: Path, subjects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    embedding_root = deep_root / "artifacts/representation_diagnostics/v8_gn_stage2/embeddings"
    files = sorted(embedding_root.glob("seed*_outer*_inner*.npz"))
    assert_development_only(files)
    if len(files) != 45:
        raise RuntimeError(f"Expected 45 development embedding files, found {len(files)}")
    stability = subjects.set_index("subject_id")[["label_name", "stability_group"]].to_dict("index")
    rows = []
    for path in files:
        match = re.fullmatch(r"seed(\d+)_outer(\d+)_inner(\d+)\.npz", path.name)
        if not match:
            raise RuntimeError(path.name)
        seed, outer, inner = map(int, match.groups())
        z = np.load(path, allow_pickle=False)
        train_x = z["train_subject_embedding"].astype(float)
        train_y = z["train_label"].astype(int)
        val_x = z["validation_subject_embedding"].astype(float)
        val_y = z["validation_label"].astype(int)
        scaler = StandardScaler().fit(train_x)
        train_scaled = scaler.transform(train_x)
        val_scaled = scaler.transform(val_x)
        centroids = {label: train_scaled[train_y == label].mean(axis=0) for label in (0, 1)}
        distances = pairwise_distances(val_scaled, train_scaled, metric="euclidean")
        order = np.argsort(distances, axis=1)
        for index, sid_raw in enumerate(z["validation_subject_id"]):
            sid = str(sid_raw).zfill(3)
            d_pd = float(np.linalg.norm(val_scaled[index] - centroids[0]))
            d_dd = float(np.linalg.norm(val_scaled[index] - centroids[1]))
            row = {
                "seed": seed,
                "outer": outer,
                "inner": inner,
                "subject_id": sid,
                "label": int(val_y[index]),
                **stability[sid],
                "distance_pd_centroid": d_pd,
                "distance_dd_centroid": d_dd,
                "pd_like_centroid_margin": d_dd - d_pd,
            }
            for k in (5, 10):
                neighbor_labels = train_y[order[index, :k]]
                row[f"knn{k}_pd_fraction"] = float(np.mean(neighbor_labels == 0))
                row[f"knn{k}_same_label_fraction"] = float(np.mean(neighbor_labels == val_y[index]))
            rows.append(row)
    appearances = pd.DataFrame(rows)
    numeric = [c for c in appearances.columns if c.startswith("distance_") or c.startswith("pd_like_") or c.startswith("knn")]
    split = appearances.groupby(
        ["outer", "inner", "subject_id", "label", "label_name", "stability_group"], as_index=False
    )[numeric].mean()
    subject = split.groupby(
        ["subject_id", "label", "label_name", "stability_group"], as_index=False
    )[numeric].mean()
    return appearances, split, subject


def activity_pd_like_audit(deep_root: Path, subjects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = deep_root / "artifacts/representation_diagnostics/v8_gn_error_oriented_20260921/results/subject_appearances_45_seed_folds.csv"
    assert_development_only([path])
    frame = pd.read_csv(path, dtype={"subject_id": str})
    frame["subject_id"] = frame.subject_id.str.zfill(3)
    activity_columns = [f"mean_margin_{activity}" for activity in (
        "CrossArms", "DrinkGlas", "Entrainment", "HoldWeight", "LiftHold", "PointFinger",
        "Relaxed", "RelaxedTask", "StretchHold", "TouchIndex", "TouchNose"
    )]
    split = frame.groupby(
        ["outer", "inner", "subject_id", "label", "stability_group"], as_index=False
    )[activity_columns].mean()
    split["label_name"] = split.label.map({0: "PD", 1: "DD"})
    dd = split.loc[split.label == 1].copy()
    for column in activity_columns:
        dd[column.replace("mean_margin_", "pd_like_")] = -dd[column]
    pd_like_columns = [c for c in dd if c.startswith("pd_like_")]
    dd["pd_like_activity_mean"] = dd[pd_like_columns].mean(axis=1)
    dd["pd_like_activity_sd"] = dd[pd_like_columns].std(axis=1, ddof=0)
    dd["pd_like_activity_fraction"] = (dd[pd_like_columns] > 0).mean(axis=1)
    subject = dd.groupby(
        ["subject_id", "label", "label_name", "stability_group"], as_index=False
    )[[*pd_like_columns, "pd_like_activity_mean", "pd_like_activity_sd", "pd_like_activity_fraction"]].mean()
    long = subject.melt(
        id_vars=["subject_id", "label_name", "stability_group"],
        value_vars=pd_like_columns,
        var_name="activity",
        value_name="pd_like_score",
    )
    long["activity"] = long.activity.str.replace("pd_like_", "", regex=False)
    return split, subject, long


def raw_signal_geometry(signals: pd.DataFrame) -> pd.DataFrame:
    identity = ["subject_id", "label_name", "stability_group"]
    measurement = [
        column for column in signals.columns
        if column not in {*identity, "activity", "sensor", "scope"}
    ]
    rows: list[dict[str, Any]] = []
    feature_sets = {
        "time": [column for column in measurement if column not in FREQUENCY_FEATURES],
        "frequency": [column for column in measurement if column in FREQUENCY_FEATURES],
        "all": measurement,
    }
    subject_info = signals[identity].drop_duplicates().set_index("subject_id")
    for feature_family, selected in feature_sets.items():
        long = signals.melt(
            id_vars=[*identity, "activity", "sensor", "scope"],
            value_vars=selected,
            var_name="feature",
            value_name="value",
        )
        long["coordinate"] = (
            long.activity.astype(str) + "|" + long.sensor.astype(str) + "|"
            + long.scope.astype(str) + "|" + long.feature.astype(str)
        )
        matrix = long.pivot_table(index="subject_id", columns="coordinate", values="value", aggfunc="mean")
        matrix = matrix.reindex(subject_info.index)
        matrix = matrix.loc[:, matrix.notna().sum() >= int(0.95 * len(matrix))]
        matrix = matrix.fillna(matrix.median())
        scaled = StandardScaler().fit_transform(matrix.to_numpy(float))
        ids = matrix.index.to_numpy(str)
        labels = subject_info.loc[ids, "label_name"].to_numpy(str)
        groups = subject_info.loc[ids, "stability_group"].to_numpy(str)
        references = groups == "stable_correct"
        centroids = {
            label: scaled[references & (labels == label)].mean(axis=0)
            for label in ("PD", "DD")
        }
        reference_x = scaled[references]
        reference_labels = labels[references]
        reference_ids = ids[references]
        distances = pairwise_distances(scaled, reference_x)
        for index, sid in enumerate(ids):
            d_pd = float(np.linalg.norm(scaled[index] - centroids["PD"]))
            d_dd = float(np.linalg.norm(scaled[index] - centroids["DD"]))
            order = np.argsort(distances[index])
            order = order[reference_ids[order] != sid]
            row = {
                "subject_id": sid,
                "label_name": labels[index],
                "stability_group": groups[index],
                "feature_family": feature_family,
                "feature_count": int(matrix.shape[1]),
                "distance_pd_stable_correct_centroid": d_pd,
                "distance_dd_stable_correct_centroid": d_dd,
                "pd_like_raw_centroid_margin": d_dd - d_pd,
            }
            for k in (5, 10):
                neighbor = reference_labels[order[:k]]
                row[f"knn{k}_pd_fraction_among_stable_correct"] = float(np.mean(neighbor == "PD"))
            rows.append(row)
    return pd.DataFrame(rows)


def add_split_consistency(
    comparisons: pd.DataFrame,
    feature_frame: pd.DataFrame,
    validation_split: pd.DataFrame,
    keys: list[str],
) -> pd.DataFrame:
    if not len(comparisons):
        return comparisons
    membership = validation_split[["outer", "inner", "subject_id"]].drop_duplicates()
    merged = membership.merge(feature_frame, on="subject_id", how="inner")
    counts = []
    for row in comparisons.itertuples(index=False):
        mask = np.ones(len(merged), dtype=bool)
        for key in keys:
            mask &= merged[key].astype(str).to_numpy() == str(getattr(row, key))
        subset = merged.loc[mask]
        deltas = []
        for _, part in subset.groupby(["outer", "inner"]):
            a = part.loc[part.stability_group == "stable_error", row.feature]
            b = part.loc[part.stability_group == row.group_b, row.feature]
            if len(a) and len(b):
                deltas.append(float(a.mean() - b.mean()))
        expected = np.sign(row.mean_difference)
        consistency = float(np.mean(np.sign(deltas) == expected)) if deltas and expected else float("nan")
        counts.append((len(deltas), consistency))
    result = comparisons.copy()
    result["split_count"] = [item[0] for item in counts]
    result["direction_consistency"] = [item[1] for item in counts]
    return result


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.index.name or not isinstance(frame.index, pd.RangeIndex):
        frame = frame.reset_index()
    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for values in frame.itertuples(index=False, name=None):
        escaped = [str(value).replace("|", "\\|").replace("\n", " ") for value in values]
        rows.append("| " + " | ".join(escaped) + " |")
    return "\n".join([header, separator, *rows])


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 12) -> str:
    if not len(frame):
        return "无。"
    shown = frame.loc[:, [c for c in columns if c in frame.columns]].head(limit).copy()
    for column in shown.select_dtypes(include=[np.number]).columns:
        shown[column] = shown[column].map(lambda x: f"{x:.4g}" if np.isfinite(x) else "NA")
    return dataframe_to_markdown(shown)


def write_report(
    output: Path,
    subjects: pd.DataFrame,
    meta_cont: pd.DataFrame,
    meta_omni: pd.DataFrame,
    meta_levels: pd.DataFrame,
    quality_cmp: pd.DataFrame,
    signal_cmp: pd.DataFrame,
    raw_geometry_cmp: pd.DataFrame,
    repr_cmp: pd.DataFrame,
    activity_cmp: pd.DataFrame,
) -> None:
    cohort = subjects.groupby(["label_name", "stability_group"]).size().unstack(fill_value=0)
    subtype = subjects.loc[subjects.label_name == "DD"].groupby(["condition", "stability_group"]).size().unstack(fill_value=0)
    significant_metadata = pd.concat([
        meta_cont.loc[meta_cont.q_value < 0.05] if len(meta_cont) else pd.DataFrame(),
    ], ignore_index=True)
    significant_quality = quality_cmp.loc[quality_cmp.q_value < 0.05].sort_values("q_value") if len(quality_cmp) else quality_cmp
    significant_signal = signal_cmp.loc[signal_cmp.q_value < 0.05].sort_values("q_value") if len(signal_cmp) else signal_cmp
    robust_frequency = signal_cmp.loc[
        (signal_cmp.label_name == "DD")
        & (signal_cmp.group_b == "stable_correct")
        & signal_cmp.feature.isin(FREQUENCY_FEATURES)
        & (signal_cmp.q_value < 0.05)
        & (signal_cmp.direction_consistency >= 0.8)
        & (signal_cmp.cliffs_delta.abs() >= 0.33)
    ].sort_values("q_value") if len(signal_cmp) else signal_cmp
    lines = [
        "# Stable-error phenotype & data audit",
        "",
        "## Protocol and evidence boundary",
        "",
        "This audit is development-only. It uses the frozen stable-error labels, 45 inner-development embedding files, seed aggregation within each `(outer, inner)` split, subject-level raw/processed signals, and patient metadata. It does not read outer-final/test predictions, metrics, or artifacts and does not train or update a model.",
        "",
        "## Cohort",
        "",
        dataframe_to_markdown(cohort),
        "",
        "DD subtype counts:",
        "",
        dataframe_to_markdown(subtype),
        "",
        "## Metadata",
        "",
        f"Continuous tests significant after BH-FDR: {len(significant_metadata)}/{len(meta_cont)}.",
        "",
        markdown_table(significant_metadata.sort_values("q_value") if len(significant_metadata) else significant_metadata,
                       ["label_name", "feature", "group_b", "mean_a", "mean_b", "cliffs_delta", "q_value"]),
        "",
        f"Categorical omnibus tests significant after BH-FDR: {int((meta_omni.q_value < .05).sum()) if len(meta_omni) else 0}/{len(meta_omni)}.",
        "",
        markdown_table(meta_omni.sort_values("q_value") if len(meta_omni) else meta_omni,
                       ["label_name", "feature", "cramers_v", "q_value"]),
        "",
        "Significant one-vs-rest categorical levels:",
        "",
        markdown_table(meta_levels.loc[meta_levels.q_value < .05].sort_values("q_value") if len(meta_levels) else meta_levels,
                       ["label_name", "feature", "level", "group_b", "fraction_a", "fraction_b", "odds_ratio", "q_value"]),
        "",
        "## Data quality",
        "",
        f"Significant quality comparisons after BH-FDR: {len(significant_quality)}/{len(quality_cmp)}.",
        "",
        markdown_table(significant_quality,
                       ["label_name", "activity_scope", "feature", "group_b", "mean_a", "mean_b", "cliffs_delta", "q_value"]),
        "",
        "## Raw signal time/frequency audit",
        "",
        f"Significant signal comparisons after BH-FDR: {len(significant_signal)}/{len(signal_cmp)}.",
        "",
        markdown_table(significant_signal,
                       ["label_name", "activity", "sensor", "scope", "feature", "group_b", "mean_difference", "cliffs_delta", "q_value", "direction_consistency"], limit=20),
        "",
        "Frequency-branch evidence rule required DD stable-error vs stable-correct, BH q<0.05, |Cliff's delta|>=0.33, and same-direction difference in at least 12/15 development splits.",
        "",
        f"Frequency features meeting that exploratory rule: {len(robust_frequency)}.",
        "",
        markdown_table(robust_frequency,
                       ["activity", "sensor", "scope", "feature", "mean_a", "mean_b", "cliffs_delta", "q_value", "direction_consistency"]),
        "",
        "## Raw-signal phenotype geometry",
        "",
        "Raw time/frequency feature vectors were standardized descriptively and compared with PD/DD stable-correct reference centroids and nearest stable-correct neighbors. This is not a fitted classifier.",
        "",
        markdown_table(raw_geometry_cmp.sort_values("q_value") if len(raw_geometry_cmp) else raw_geometry_cmp,
                       ["label_name", "feature_family", "feature", "group_b", "mean_a", "mean_b", "cliffs_delta", "q_value"]),
        "",
        "## Frozen representation geometry",
        "",
        markdown_table(repr_cmp.sort_values("q_value") if len(repr_cmp) else repr_cmp,
                       ["label_name", "feature", "group_b", "mean_a", "mean_b", "cliffs_delta", "q_value"]),
        "",
        "## DD activity-wise PD-like evidence",
        "",
        markdown_table(activity_cmp.sort_values("q_value") if len(activity_cmp) else activity_cmp,
                       ["activity", "feature", "group_b", "mean_a", "mean_b", "cliffs_delta", "q_value"]),
        "",
        "## Interpretation boundary",
        "",
        "Associations with a frozen stable-error label are descriptive, not causal. The same subjects contribute to several development validation appearances, so inferential tests use one aggregated row per subject; split-direction counts are robustness checks, not additional independent samples. Free-text clinical keyword flags are lexical summaries only. Frequency findings are hypothesis-generating and cannot justify a frequency network unless they meet the stated stability rule and remain clinically/data-quality interpretable.",
        "",
    ]
    (output / "STABLE_ERROR_PHENOTYPE_DATA_AUDIT_REPORT.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    deep_root = args.deep_root.resolve()
    data_root = args.data_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    processed_root = data_root / "processed/pads_multi_activity/v2_l1_full_length"
    raw_root = data_root / "raw/pads"
    assert_development_only([deep_root, data_root, output, processed_root, raw_root])

    subjects = load_subjects(deep_root, raw_root)
    subjects.to_csv(output / "subject_metadata_and_stability.csv", index=False)
    metadata_continuous = ["age", "age_at_diagnosis", "disease_duration", "height", "weight", "bmi"]
    meta_cont = compare_continuous(subjects, metadata_continuous, ["label_name"], "metadata_continuous")
    meta_cont.to_csv(output / "metadata_continuous_comparisons_bh.csv", index=False)
    categorical = [
        "condition", "gender", "handedness", "appearance_in_kinship",
        "appearance_in_first_grade_kinship", "effect_of_alcohol_on_tremor", "device_id",
        "comment_tremor", "comment_hypokinesia", "comment_rigidity", "comment_dystonia",
        "comment_gait_balance", "comment_drug_induced", "comment_functional", "comment_vascular",
    ]
    meta_omni, meta_levels = categorical_audit(subjects, categorical)
    meta_omni.to_csv(output / "metadata_categorical_omnibus_bh.csv", index=False)
    meta_levels.to_csv(output / "metadata_categorical_levels_bh.csv", index=False)

    quality = load_quality(processed_root, subjects)
    quality.to_csv(output / "subject_quality_features.csv", index=False)
    quality_values = [c for c in quality.columns if c not in {
        "subject_id", "label_name", "stability_group", "activity_scope"
    }]
    quality_cmp = compare_continuous(
        quality, quality_values, ["label_name", "activity_scope"], "data_quality"
    )
    quality_cmp.to_csv(output / "quality_comparisons_bh.csv", index=False)

    signals = load_signal_features(processed_root, subjects)
    signals.to_csv(output / "high_risk_activity_signal_features.csv", index=False)
    signal_values = [c for c in signals.columns if c not in {
        "subject_id", "label_name", "stability_group", "activity", "sensor", "scope"
    }]
    signal_cmp = compare_continuous(
        signals, signal_values, ["label_name", "activity", "sensor", "scope"], "signal"
    )

    appearances, repr_split, repr_subject = representation_audit(deep_root, subjects)
    appearances.to_csv(output / "representation_geometry_45_seed_folds.csv", index=False)
    repr_split.to_csv(output / "representation_geometry_15_splits.csv", index=False)
    repr_subject.to_csv(output / "representation_geometry_subject.csv", index=False)
    repr_values = [c for c in repr_subject.columns if c not in {
        "subject_id", "label", "label_name", "stability_group"
    }]
    repr_cmp = compare_continuous(
        repr_subject, repr_values, ["label_name"], "representation_geometry"
    )
    repr_cmp.to_csv(output / "representation_geometry_comparisons_bh.csv", index=False)

    activity_split, activity_subject, activity_long = activity_pd_like_audit(deep_root, subjects)
    activity_split.to_csv(output / "activity_pd_like_15_splits.csv", index=False)
    activity_subject.to_csv(output / "activity_pd_like_subject.csv", index=False)
    activity_long.to_csv(output / "activity_pd_like_long.csv", index=False)
    activity_cmp = compare_continuous(activity_long, ["pd_like_score"], ["activity"], "dd_activity_pd_like")
    summary_values = ["pd_like_activity_mean", "pd_like_activity_sd", "pd_like_activity_fraction"]
    activity_summary_cmp = compare_continuous(
        activity_subject, summary_values, [], "dd_activity_pd_like_summary"
    )
    activity_cmp = pd.concat([activity_cmp, activity_summary_cmp], ignore_index=True, sort=False)
    activity_cmp["q_value"] = bh_fdr(activity_cmp.p_value)
    activity_cmp.to_csv(output / "activity_pd_like_comparisons_bh.csv", index=False)

    feature_subject = signals.drop_duplicates(["subject_id", "activity", "sensor", "scope"])
    signal_cmp = add_split_consistency(
        signal_cmp, feature_subject, repr_split, ["label_name", "activity", "sensor", "scope"]
    )
    signal_cmp.to_csv(output / "signal_comparisons_bh.csv", index=False)

    raw_geometry = raw_signal_geometry(signals)
    raw_geometry.to_csv(output / "raw_signal_geometry_subject.csv", index=False)
    raw_geometry_values = [
        "distance_pd_stable_correct_centroid", "distance_dd_stable_correct_centroid",
        "pd_like_raw_centroid_margin", "knn5_pd_fraction_among_stable_correct",
        "knn10_pd_fraction_among_stable_correct",
    ]
    raw_geometry_cmp = compare_continuous(
        raw_geometry, raw_geometry_values, ["label_name", "feature_family"], "raw_signal_geometry"
    )
    raw_geometry_cmp.to_csv(output / "raw_signal_geometry_comparisons_bh.csv", index=False)

    write_report(
        output, subjects, meta_cont, meta_omni, meta_levels, quality_cmp,
        signal_cmp, raw_geometry_cmp, repr_cmp, activity_cmp,
    )
    manifest = {
        "status": "complete",
        "development_only": True,
        "model_training": False,
        "outer_information_used": False,
        "subjects": int(len(subjects)),
        "embedding_files": 45,
        "high_risk_activities": list(ACTIVITIES),
        "output_files": sorted(path.name for path in output.iterdir() if path.is_file()),
    }
    (output / "audit_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
