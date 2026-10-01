from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import stats

from src.analysis import ANALYSIS_VERSION
from src.analysis.common import canonical_sha256


ACTIVITIES = (
    "CrossArms", "DrinkGlas", "Entrainment", "HoldWeight", "LiftHold",
    "PointFinger", "Relaxed", "RelaxedTask", "StretchHold", "TouchIndex",
    "TouchNose",
)
WRISTS = ("left", "right")
CHANNELS = ("AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ")
DERIVED = ("AccMag", "GyroMag")
SIGNALS = CHANNELS + DERIVED
TIME_FEATURES = (
    "mean", "std", "variance", "rms", "median", "mad", "iqr", "minimum",
    "maximum", "peak_to_peak", "skewness", "kurtosis_excess",
    "signal_energy", "zero_crossing_rate_centered", "derivative_rms",
    "derivative_std",
)
SPECTRAL_FEATURES = (
    "dominant_frequency_hz_0p5_20", "dominant_power_0p5_20",
    "spectral_entropy_0p5_20", "total_power_0p5_20", "band_power_0p5_3",
    "band_power_3_7", "band_power_7_12", "band_power_12_20",
    "band_fraction_0p5_3", "band_fraction_3_7", "band_fraction_7_12",
    "band_fraction_12_20",
)


def _finite(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def time_features(signal: np.ndarray, sample_rate: float = 100.0) -> dict[str, float]:
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if values.size < 3 or not np.isfinite(values).all():
        raise ValueError("Signal must contain at least three finite values")
    centered = values - np.mean(values)
    derivative = np.diff(values) * float(sample_rate)
    crossing = np.mean((centered[:-1] * centered[1:]) < 0)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "variance": float(np.var(values)),
        "rms": float(np.sqrt(np.mean(values * values))),
        "median": float(np.median(values)),
        "mad": float(np.median(np.abs(values - np.median(values)))),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "peak_to_peak": float(np.ptp(values)),
        "skewness": _finite(stats.skew(values, bias=False)),
        "kurtosis_excess": _finite(stats.kurtosis(values, fisher=True, bias=False)),
        "signal_energy": float(np.sum(values * values)),
        "zero_crossing_rate_centered": float(crossing),
        # Acc derivative is g/s; gyro derivative is rad/s^2.
        "derivative_rms": float(np.sqrt(np.mean(derivative * derivative))),
        "derivative_std": float(np.std(derivative)),
    }


def _band_power(frequencies: np.ndarray, density: np.ndarray, low: float, high: float) -> float:
    mask = (frequencies >= low) & (frequencies < high)
    if int(mask.sum()) < 2:
        return 0.0
    return float(np.trapezoid(density[mask], frequencies[mask]))


def spectral_features(signal: np.ndarray, sample_rate: float = 100.0) -> dict[str, float]:
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if values.size < 8 or not np.isfinite(values).all():
        raise ValueError("Signal must contain at least eight finite values")
    # Fixed mean subtraction only for spectra; time-domain features are unchanged.
    centered = values - values.mean()
    spectrum = np.fft.rfft(centered)
    frequencies = np.fft.rfftfreq(values.size, d=1.0 / float(sample_rate))
    density = (np.abs(spectrum) ** 2) / (float(sample_rate) * values.size)
    if values.size > 1:
        density[1:-1] *= 2.0
    analysis = (frequencies >= 0.5) & (frequencies <= 20.0)
    selected_density = density[analysis]
    selected_frequency = frequencies[analysis]
    dominant_index = int(np.argmax(selected_density))
    total = _band_power(frequencies, density, 0.5, 20.0000001)
    bands = {
        "0p5_3": _band_power(frequencies, density, 0.5, 3.0),
        "3_7": _band_power(frequencies, density, 3.0, 7.0),
        "7_12": _band_power(frequencies, density, 7.0, 12.0),
        "12_20": _band_power(frequencies, density, 12.0, 20.0000001),
    }
    mass = selected_density / max(float(selected_density.sum()), np.finfo(float).eps)
    entropy = -float(np.sum(mass * np.log(np.clip(mass, 1e-15, None))))
    entropy /= max(float(np.log(max(mass.size, 2))), np.finfo(float).eps)
    result = {
        "dominant_frequency_hz_0p5_20": float(selected_frequency[dominant_index]),
        "dominant_power_0p5_20": float(selected_density[dominant_index]),
        "spectral_entropy_0p5_20": entropy,
        "total_power_0p5_20": total,
    }
    for name, value in bands.items():
        result[f"band_power_{name}"] = value
        result[f"band_fraction_{name}"] = float(value / max(total, np.finfo(float).eps))
    return result


def signal_feature_vector(signal: np.ndarray, sample_rate: float = 100.0) -> list[float]:
    time = time_features(signal, sample_rate)
    spectral = spectral_features(signal, sample_rate)
    return [time[name] for name in TIME_FEATURES] + [spectral[name] for name in SPECTRAL_FEATURES]


def feature_schema(activities: Sequence[str] = ACTIVITIES) -> dict[str, Any]:
    columns = [
        f"{activity}|{wrist}|{signal}|{feature}"
        for activity in activities
        for wrist in WRISTS
        for signal in SIGNALS
        for feature in TIME_FEATURES + SPECTRAL_FEATURES
    ]
    payload = {
        "schema_version": 1,
        "extractor_version": ANALYSIS_VERSION,
        "representation": "processed_v2_l1_full_length_float32",
        "sample_rate_assumption_hz": 100.0,
        "spectral_detrend": "subtract_signal_mean",
        "time_domain_detrend": "none",
        "vector_magnitude": "sqrt(x^2+y^2+z^2) on processed Acc/Gyro triples",
        "activities": list(activities),
        "wrists": list(WRISTS),
        "signals": list(SIGNALS),
        "features": list(TIME_FEATURES + SPECTRAL_FEATURES),
        "columns": columns,
        "label_dependency": False,
    }
    payload["schema_sha256"] = canonical_sha256(payload)
    return payload


def extract_subject_features(
    processed_root: str | Path,
    subject_id: str,
    activities: Sequence[str] = ACTIVITIES,
) -> np.ndarray:
    root = Path(processed_root)
    values: list[float] = []
    for activity in activities:
        for wrist in WRISTS:
            path = root / "signals" / activity / subject_id / f"{wrist}.npy"
            signal = np.load(path, allow_pickle=False).astype(np.float64, copy=False)
            if signal.ndim != 2 or signal.shape[0] != 6:
                raise ValueError(f"Expected [6,T] at {path}, got {signal.shape}")
            expanded = list(signal) + [
                np.sqrt(np.sum(signal[0:3] ** 2, axis=0)),
                np.sqrt(np.sum(signal[3:6] ** 2, axis=0)),
            ]
            for channel in expanded:
                values.extend(signal_feature_vector(channel))
    output = np.asarray(values, dtype=np.float64)
    if not np.isfinite(output).all():
        raise ValueError(f"Non-finite feature for subject {subject_id}")
    return output


def extract_feature_matrix(
    processed_root: str | Path,
    subject_ids: Sequence[str],
    activities: Sequence[str] = ACTIVITIES,
) -> tuple[np.ndarray, dict[str, Any]]:
    schema = feature_schema(activities)
    matrix = np.vstack([
        extract_subject_features(processed_root, str(subject), activities)
        for subject in subject_ids
    ])
    if matrix.shape != (len(subject_ids), len(schema["columns"])):
        raise AssertionError("Feature matrix and schema are inconsistent")
    return matrix, schema


def select_feature_indices(
    schema: dict[str, Any],
    *,
    include_activities: Iterable[str] | None = None,
    exclude_activities: Iterable[str] | None = None,
    wrists: Iterable[str] | None = None,
    sensors: Iterable[str] | None = None,
) -> np.ndarray:
    include = None if include_activities is None else set(include_activities)
    exclude = set() if exclude_activities is None else set(exclude_activities)
    wrist_set = None if wrists is None else set(wrists)
    sensor_set = None if sensors is None else set(sensors)
    selected: list[int] = []
    for index, column in enumerate(schema["columns"]):
        activity, wrist, signal, _ = column.split("|", 3)
        sensor = "Acc" if signal.startswith("Acc") else "Gyro"
        if include is not None and activity not in include:
            continue
        if activity in exclude:
            continue
        if wrist_set is not None and wrist not in wrist_set:
            continue
        if sensor_set is not None and sensor not in sensor_set:
            continue
        selected.append(index)
    if not selected:
        raise ValueError("Feature subset is empty")
    return np.asarray(selected, dtype=np.int64)


def minirocket_subject_tensor(
    processed_root: str | Path,
    subject_ids: Sequence[str],
    activities: Sequence[str] = ACTIVITIES,
    target_length: int = 976,
) -> np.ndarray:
    """Return [subject, activity*wrist*channel, time], retaining activity identity."""
    root = Path(processed_root)
    output = np.empty(
        (len(subject_ids), len(activities) * 2 * 6, target_length), dtype=np.float32
    )
    destination = np.linspace(0.0, 1.0, target_length)
    for subject_index, subject in enumerate(subject_ids):
        channel_index = 0
        for activity in activities:
            for wrist in WRISTS:
                signal = np.load(
                    root / "signals" / activity / str(subject) / f"{wrist}.npy",
                    allow_pickle=False,
                )
                source = np.linspace(0.0, 1.0, signal.shape[-1])
                for channel in signal:
                    output[subject_index, channel_index] = np.interp(
                        destination, source, channel
                    )
                    channel_index += 1
    if not np.isfinite(output).all():
        raise ValueError("MiniRocket representation contains non-finite values")
    return output
