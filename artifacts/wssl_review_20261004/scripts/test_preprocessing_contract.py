"""Read-only development audit of retained WSSL preprocessing and fixed cache.

Writes only aggregate audit results and clearly labelled synthetic contract fixtures.
Does not train, instantiate HarNet, read outer predictions, or change input assets.
Execution rechecks all 8,580 wrists and synthetic normalization/mask contracts.
The companion root audit separately recomputes the 15 real train-only normalizations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly


ACTIVITIES = [
    "CrossArms", "DrinkGlas", "Entrainment", "HoldWeight", "LiftHold",
    "PointFinger", "Relaxed", "RelaxedTask", "StretchHold", "TouchIndex",
    "TouchNose",
]
EXPECTED_SPLIT_SHA = "b3c52317cb12b73c66046bbd7c50c94a7e707a64a38d3ad4f0fd4bc0c6ba6b3e"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def synthetic_contract(output: Path):
    from src.datasets import folds
    from src.datasets.manifest import ManifestRecord
    from src.datasets.preprocessing import remove_acceleration_trend
    from src.datasets.subject_activity import SubjectActivityDataset, collate_subject_activities

    fixtures = output / "synthetic_contract_fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    records = []
    config = {"data": dict(
        type="pads", unit="subject", normalization_scope="activity_wrist_channel",
        labels=["PD", "DD"], activities=ACTIVITIES, raw_input_channels=6,
        sensor_mode="acc_gyro", wrist_mode="bilateral", sequence_length=None,
        train_crop="full", eval_crop="full", eval_crop_count=1,
        pad_value=0.0, delimiter=",", drop_time_column=False, standardize=True,
        normalization_epsilon=1e-6, train_activity_dropout=0.0,
        train_rotation_degrees=0.0,
    )}
    for si, subject in enumerate(["fixture_train_PD", "fixture_train_DD", "fixture_valid_DD"]):
        for ai, activity in enumerate(ACTIVITIES):
            length = 17 if ai % 2 == 0 else 29
            paths = []
            for wi in range(2):
                signal = np.asarray(
                    (1000 if si == 2 else si * 10) + ai + wi * 3
                    + np.arange(6)[:, None] * 0.2
                    + np.arange(length)[None, :] * 0.1,
                    dtype=np.float32,
                )
                path = fixtures / f"{subject}_{activity}_{wi}.npy"
                if path.exists():
                    assert np.array_equal(np.load(path), signal)
                else:
                    np.save(path, signal, allow_pickle=False)
                paths.append(path)
            records.append(ManifestRecord(
                paths[0], paths[1], subject, int(si != 0), activity,
                f"{subject}_{activity}", None,
                source_condition="Parkinson's" if si == 0 else "Other Movement Disorders",
            ))
    original_loader = folds.load_configured_records
    folds.load_configured_records = lambda *args, **kwargs: records
    try:
        bundle = folds.build_subject_fold_datasets(
            config, train_subject_ids=["fixture_train_PD", "fixture_train_DD"],
            validation_subject_ids=["fixture_valid_DD"], test_subject_ids=(),
            fold_id="synthetic_no_outer_contract",
        )
    finally:
        folds.load_configured_records = original_loader
    assert bundle.test is None
    for ai, activity in enumerate(ACTIVITIES):
        signals = []
        for record in records:
            if record.activity == activity and record.subject_id.startswith("fixture_train"):
                signals.append(torch.from_numpy(np.stack([
                    np.load(record.left_path), np.load(record.right_path)
                ])).double())
        joined = torch.cat(signals, dim=-1)
        expected_mean = joined.mean(dim=-1, keepdim=True).float()
        expected_std = joined.std(dim=-1, keepdim=True, correction=0).clamp_min(1e-6).float()
        assert torch.allclose(bundle.mean[ai], expected_mean, atol=1e-6, rtol=0)
        assert torch.allclose(bundle.std[ai], expected_std, atol=1e-6, rtol=0)
    valid = bundle.validation[0]
    assert valid["subject_id"] == "fixture_valid_DD"
    assert valid["activity_mask"].all() and valid["wrist_mask"].bool().all()
    assert float(valid["x"][0, 0, 0, 0]) > 100, "Validation entered normalization fitting"
    expected_lengths = torch.tensor([17 if ai % 2 == 0 else 29 for ai in range(11)])
    assert torch.equal(valid["activity_lengths"], expected_lengths)
    batch = collate_subject_activities([bundle.train[0], valid])
    for ai, length in enumerate(expected_lengths.tolist()):
        assert (batch["x"][:, ai, :, :, length:] == 0).all()
    unilateral = SubjectActivityDataset(
        records, ACTIVITIES, dict(config["data"], wrist_mode="left"),
        bundle.mean, bundle.std, "validation",
    )[0]
    assert (unilateral["x"][:, 1] == 0).all()
    assert torch.equal(unilateral["wrist_mask"], torch.tensor([1., 0.]).expand(11, 2))
    observed = np.random.default_rng(42).normal(size=(6, 64))
    processed, report = remove_acceleration_trend(
        observed, regularization=50, rho=40, max_iterations=5000,
        absolute_tolerance=1e-4, relative_tolerance=1e-4,
    )
    assert report.converged and np.array_equal(processed[3:], observed[3:])
    assert np.isfinite(processed).all()
    return dict(status="PASS", train_only_normalization=True,
                id_order_activity_order_wrist_mask_and_padding=True,
                gyro_unchanged_by_acc_detrending=True,
                fixtures_are_synthetic_not_clinical=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("/home/zyt/deep_final"))
    parser.add_argument("--output", type=Path, default=Path("/home/zyt/deep_final/artifacts/wssl_review_20261004/analysis"))
    args = parser.parse_args()
    foundation = args.project / "foundation_validation"
    sys.path.insert(0, str(foundation))
    from src.datasets.builders import load_configured_records
    from src.datasets.manifest import SENSOR_CHANNELS, WRIST_ORDER

    args.output.mkdir(parents=True, exist_ok=True)
    baseline = args.project / "artifacts/phase3b_external_ssl_20260928"
    split_path = foundation / "splits/pads_classification/v3_nested_cv/seed42/nested_cv_splits.json"
    assert sha(split_path) == EXPECTED_SPLIT_SHA
    split = json.loads(split_path.read_text())
    # The enclosing historical JSON key is called outer, but only inner-development
    # train/validation assignments are used. No outer test IDs/data/outcomes are used.
    units = [(context["outer_fold"], inner["inner_fold"], inner)
             for context in split["outer"] for inner in context["inner_folds"]]
    assert len(units) == 15
    development_ids = {str(s) for _, _, inner in units
                       for role in ("train_subjects", "validation_subjects")
                       for s in inner[role]}
    extraction_audit = json.loads((baseline / "analysis/extraction_audit.json").read_text())
    cache_path = baseline / "analysis/ssl_embeddings_all.npz"
    assert sha(cache_path) == extraction_audit["output_sha256"]
    cache = np.load(cache_path, allow_pickle=False)
    ids = cache["subject_ids"].astype(str).tolist()
    assert len(ids) == len(set(ids)) == 390 and set(ids) == development_ids
    assert cache["activities"].astype(str).tolist() == ACTIVITIES
    assert cache["pretrained"].shape == (390, 11, 2, 1024)
    assert np.isfinite(cache["pretrained"]).all()
    assert tuple(WRIST_ORDER) == ("left", "right")
    assert tuple(SENSOR_CHANNELS) == ("AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ")
    first = baseline / "runs/b_str_pretrained/seed42/outer_0/inner_0/checkpoints/best.pt"
    first_checkpoint = torch.load(first, map_location="cpu", weights_only=False)
    config = first_checkpoint["config"]
    data = config["data"]
    assert data["activities"] == ACTIVITIES and data["labels"] == ["PD", "DD"]
    assert data["sequence_length"] is None and data["train_crop"] == data["eval_crop"] == "full"
    assert data["sensor_mode"] == "acc_gyro" and data["wrist_mode"] == "bilateral"
    assert data["normalization_scope"] == "activity_wrist_channel"
    assert not data.get("train_activity_dropout", 0) and not data.get("train_rotation_degrees", 0)
    all_records = load_configured_records(data, ["PD", "DD"])
    records = [record for record in all_records if record.subject_id in development_ids]
    assert len(records) == 4290
    assert len({(r.subject_id, r.activity) for r in records}) == 4290
    labels = {record.subject_id: record.label for record in records}
    assert Counter(labels.values()) == {0: 276, 1: 114}
    id_index = {subject: index for index, subject in enumerate(ids)}
    activity_index = {activity: index for index, activity in enumerate(ACTIVITIES)}
    raw_root = Path("/home/zyt/MFAM/data/raw/pads/movement/timeseries")
    # Check units and nominal sampling for every development observation, rather
    # than infer physical units from one example or from waveform amplitudes.
    metadata_subjects = set()
    metadata_record_count = 0
    expected_channels = ["Time", "Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z",
                         "Gyroscope_X", "Gyroscope_Y", "Gyroscope_Z"]
    expected_units = ["s", "g", "g", "g", "rad/s", "rad/s", "rad/s"]
    for path in sorted(raw_root.parent.glob("observation_*.json")):
        observation = json.loads(path.read_text())
        subject = str(observation["subject_id"])
        if subject not in development_ids:
            continue
        assert subject not in metadata_subjects
        metadata_subjects.add(subject)
        assert float(observation["sampling_rate"]) == 100.0
        selected_sessions = [session for session in observation["session"]
                             if session["record_name"] in ACTIVITIES]
        assert Counter(session["record_name"] for session in selected_sessions) == Counter(ACTIVITIES)
        for session in selected_sessions:
            records_by_wrist = {record["device_location"]: record for record in session["records"]}
            assert set(records_by_wrist) == {"LeftWrist", "RightWrist"}
            assert len(session["records"]) == 2
            for record in records_by_wrist.values():
                assert record["channels"] == expected_channels
                assert record["units"] == expected_units
                expected_name = f"{subject}_{session['record_name']}_{record['device_location']}.txt"
                assert Path(record["file_name"]).name == expected_name
                metadata_record_count += 1
    assert metadata_subjects == development_ids and metadata_record_count == 8580
    digest = hashlib.sha256()
    counts = np.zeros((390, 11, 2), dtype=np.int16)
    length_counts, gap_records = Counter(), 0
    clipped, total_values, max_time_gap_ratio = 0, 0, 0.
    max_wrist_clock_offset, minimum_rate, maximum_rate = 0., float("inf"), 0.
    for ri, record in enumerate(records):
        wrist_times = []
        for wi, wrist in enumerate(["LeftWrist", "RightWrist"]):
            raw_path = raw_root / f"{record.subject_id}_{record.record_name}_{wrist}.txt"
            raw64 = np.loadtxt(raw_path, delimiter=",", dtype=np.float64)
            raw = raw64.astype(np.float32)
            assert raw.shape[1] == 7 and np.isfinite(raw64).all()
            dt = np.diff(raw64[:, 0])
            assert (dt > 0).all()
            rate = 1 / float(np.median(dt))
            minimum_rate, maximum_rate = min(minimum_rate, rate), max(maximum_rate, rate)
            ratio = float(dt.max() / np.median(dt))
            max_time_gap_ratio = max(max_time_gap_ratio, ratio)
            gap_records += int(ratio > 10)
            wrist_times.append(raw64[:, 0])
            signal_path = record.left_path if wi == 0 else record.right_path
            processed = np.load(signal_path, allow_pickle=False)
            assert processed.dtype == np.float32
            assert processed.shape == (6, len(raw) - 48) and np.isfinite(processed).all()
            assert np.array_equal(processed[3:], raw[48:, 4:7].T), "Gyro differs from fixed post-trim raw"
            expected = record.left_length if wi == 0 else record.right_length
            acc = raw[48:, 1:4]
            assert len(acc) == expected and len(acc) in (976, 2000)
            length_counts[len(acc)] += 1
            clipped += int(((acc < -3) | (acc > 3)).sum())
            total_values += acc.size
            acc = np.clip(acc, -3, 3)
            if len(acc) < 1000:
                segments = [np.pad(acc, ((12, 12), (0, 0)), mode="edge")]
            else:
                segments = [acc[:1000], acc[-1000:]]
            key = (id_index[record.subject_id], activity_index[record.activity], wi)
            for segment in segments:
                transformed = resample_poly(segment, 3, 10, axis=0).T.astype(np.float32)
                assert transformed.shape == (3, 300) and np.isfinite(transformed).all()
                digest.update(transformed.tobytes())
                digest.update(np.asarray(key, dtype=np.int16).tobytes())
                counts[key] += 1
        assert wrist_times[0].shape == wrist_times[1].shape
        max_wrist_clock_offset = max(max_wrist_clock_offset,
                                    float(np.abs(wrist_times[0] - wrist_times[1]).max()))
        if (ri + 1) % 390 == 0:
            print(f"raw/processed contract {ri + 1}/4290", flush=True)
    assert np.array_equal(counts, cache["window_counts"])
    assert digest.hexdigest() == extraction_audit["resampled_window_sha256"]
    assert clipped == extraction_audit["values_clipped_before_resampling"]
    assert total_values == extraction_audit["raw_acc_values"]
    synthetic = synthetic_contract(args.output)
    result = dict(
        status="PASS", outer_predictions_or_outcomes_accessed=False,
        models_trained=False, formal_inputs_modified=False,
        subject_count=390, activity_records=4290, wrist_records=8580,
        raw_gyro_exact_after_48_sample_trim=True,
        all_cached_window_inputs_sha_exact=True,
        cache_subject_ids_and_activity_order_exact=True,
        cache_independently_maps_subjects_no_fold_fit=True,
        cached_feature_values_verified_by_sha_not_new_HarNet_inference=True,
        cache_sha256=sha(cache_path), resampled_window_sha256=digest.hexdigest(),
        unit_metadata_observations=390, unit_metadata_wrist_records=metadata_record_count,
        physical_units=["s", "g", "g", "g", "rad/s", "rad/s", "rad/s"],
        metadata_nominal_sampling_rate_hz=100.0,
        post_trim_length_counts=length_counts, window_count=int(counts.sum()),
        clipped_values=clipped, raw_acc_values=total_values,
        clipping_fraction=clipped / total_values,
        effective_sampling_rate_hz=[minimum_rate, maximum_rate],
        timestamp_gap_gt_10_median_records=gap_records,
        maximum_gap_over_median=max_time_gap_ratio,
        maximum_wrist_clock_offset_seconds=max_wrist_clock_offset,
        actual_resampling_is_uniform_index_polyphase_not_timestamp_interpolation=True,
        real_normalization_recomputation="companion root audit; not duplicated here",
        synthetic_contract=synthetic,
        limitations=[
            "No clinical proof that detrending, clipping, padding or resampling preserves every motor phenotype.",
            "Timestamp gaps and wrist clock offsets are described; no signal correction or exclusion was applied.",
            "Physical units are validated from metadata; this does not prove per-device calibration accuracy.",
            "No HarNet encoder inference was repeated; frozen feature bytes and resampled inputs were checked against formal extraction provenance.",
        ],
    )
    (args.output / "preprocessing_contract.json").write_text(json.dumps(jsonable(result), indent=2) + "\n")
    print(json.dumps(jsonable(result), indent=2), flush=True)


if __name__ == "__main__":
    main()
