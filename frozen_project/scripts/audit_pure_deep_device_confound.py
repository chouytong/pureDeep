from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
import statistics

import numpy as np
from sklearn.metrics import roc_auc_score


def _load_devices(pattern: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in glob.glob(pattern):
        with open(name, encoding="utf-8") as handle:
            payload = json.load(handle)
        subject = str(payload["subject_id"])
        device = str(payload["device_id"])
        previous = result.setdefault(subject, device)
        if previous != device:
            raise ValueError(f"Subject {subject} has inconsistent devices")
    return result


def _read_fold(path: Path) -> dict[str, tuple[int, float]]:
    result = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            subject = str(row["subject_id"])
            result[subject] = (int(row["target"]), float(row["probability_dd"]))
    return result


def _metrics(targets: np.ndarray, scores: np.ndarray) -> dict[str, float | int | None]:
    predictions = (scores >= 0.5).astype(np.int64)
    tp = int(np.sum((targets == 1) & (predictions == 1)))
    tn = int(np.sum((targets == 0) & (predictions == 0)))
    fp = int(np.sum((targets == 0) & (predictions == 1)))
    fn = int(np.sum((targets == 1) & (predictions == 0)))
    pd_recall = tn / (tn + fp) if tn + fp else None
    dd_recall = tp / (tp + fn) if tp + fn else None
    precision_pd = tn / (tn + fn) if tn + fn else 0.0
    precision_dd = tp / (tp + fp) if tp + fp else 0.0
    f1_pd = 2 * precision_pd * pd_recall / (precision_pd + pd_recall) if pd_recall is not None and precision_pd + pd_recall else 0.0
    f1_dd = 2 * precision_dd * dd_recall / (precision_dd + dd_recall) if dd_recall is not None and precision_dd + dd_recall else 0.0
    return {
        "n": int(targets.size),
        "pd_n": int(np.sum(targets == 0)),
        "dd_n": int(np.sum(targets == 1)),
        "accuracy": float(np.mean(predictions == targets)),
        "balanced_accuracy": (
            float((pd_recall + dd_recall) / 2)
            if pd_recall is not None and dd_recall is not None else None
        ),
        "macro_f1": float((f1_pd + f1_dd) / 2),
        "auroc": (
            float(roc_auc_score(targets, scores))
            if np.unique(targets).size == 2 else None
        ),
        "pd_recall": pd_recall,
        "dd_recall": dd_recall,
        "mean_probability_dd": float(np.mean(scores)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def _nullable_mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return statistics.mean(finite) if finite else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit released PADS device_id metadata using inner-validation predictions only; "
            "the field is not assumed to prove physical watch generation."
        )
    )
    parser.add_argument("--seed-run", action="append", required=True, type=Path)
    parser.add_argument(
        "--observation-glob",
        default="data/raw/pads/movement/observation_*.json",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    devices = _load_devices(args.observation_glob)

    rows: list[dict] = []
    shifts: list[dict] = []
    device_only_rows: list[dict] = []
    for outer in range(5):
        seed_oof: list[dict[str, tuple[int, float]]] = []
        for run in args.seed_run:
            combined: dict[str, tuple[int, float]] = {}
            for inner in range(3):
                path = run / f"outer_{outer}" / f"inner_{inner}" / "predictions" / "validation.csv"
                fold = _read_fold(path)
                overlap = set(combined).intersection(fold)
                if overlap:
                    raise ValueError(f"Duplicate inner-OOF subjects: {sorted(overlap)[:3]}")
                combined.update(fold)
            seed_oof.append(combined)
        subjects = sorted(seed_oof[0])
        if any(sorted(item) != subjects for item in seed_oof):
            raise ValueError("Seed runs do not contain identical OOF subjects")
        targets = np.asarray([seed_oof[0][subject][0] for subject in subjects], dtype=np.int64)
        scores = np.asarray(
            [statistics.mean(item[subject][1] for item in seed_oof) for subject in subjects],
            dtype=np.float64,
        )
        subject_devices = np.asarray([devices[subject] for subject in subjects], dtype=object)

        for group in ["all", "Apple Watch Series 3", "Apple Watch Series 4"]:
            mask = np.ones(len(subjects), dtype=bool) if group == "all" else subject_devices == group
            row = {"outer_context": outer, "group": group, **_metrics(targets[mask], scores[mask])}
            rows.append(row)

        target_shift = {"outer_context": outer}
        for target, label in [(0, "pd"), (1, "dd")]:
            means = {}
            for device, short in [("Apple Watch Series 3", "series3"), ("Apple Watch Series 4", "series4")]:
                mask = (targets == target) & (subject_devices == device)
                means[short] = float(np.mean(scores[mask])) if np.any(mask) else None
                target_shift[f"{label}_{short}_n"] = int(np.sum(mask))
                target_shift[f"{label}_{short}_mean_probability_dd"] = means[short]
            target_shift[f"{label}_series4_minus_series3"] = (
                means["series4"] - means["series3"]
                if means["series4"] is not None and means["series3"] is not None else None
            )
        shifts.append(target_shift)

        device_scores = (subject_devices == "Apple Watch Series 4").astype(np.float64)
        device_only_rows.append({"outer_context": outer, **_metrics(targets, device_scores)})

    summary = {
        "protocol": {
            "scope": "three-seed ensemble of fixed inner-validation OOF predictions",
            "outer_test_accessed": False,
            "threshold": 0.5,
            "warning": "Outer contexts overlap in subjects; context dispersion is descriptive.",
            "hardware_generation_confirmed": False,
            "metadata_interpretation": (
                "PADS observation JSON contains Series 3/4 device_id values, but the peer-reviewed "
                "paper and PhysioNet methods state that all participants used Series 4. Treat "
                "device_id only as an inconsistent metadata/acquisition-batch proxy until the "
                "dataset authors clarify it."
            ),
        },
        "device_counts_full_metadata": {
            name: sum(device == name for device in devices.values()) for name in sorted(set(devices.values()))
        },
        "context_mean_metrics": {
            group: {
                metric: _nullable_mean([
                    row[metric] for row in rows if row["group"] == group
                ])
                for metric in ("n", "pd_n", "dd_n", "accuracy", "balanced_accuracy", "macro_f1", "auroc", "pd_recall", "dd_recall", "mean_probability_dd")
            }
            for group in ["all", "Apple Watch Series 3", "Apple Watch Series 4"]
        },
        "mean_within_label_device_score_shift": {
            label: _nullable_mean([row[f"{label}_series4_minus_series3"] for row in shifts])
            for label in ("pd", "dd")
        },
        "device_id_only_negative_control_context_mean": {
            metric: _nullable_mean([row[metric] for row in device_only_rows])
            for metric in ("accuracy", "balanced_accuracy", "macro_f1", "auroc", "pd_recall", "dd_recall")
        },
        "context_rows": rows,
        "within_label_shifts": shifts,
        "device_only_rows": device_only_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "device_confound_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with (args.output_dir / "device_group_context_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: summary[key] for key in (
        "protocol", "context_mean_metrics", "mean_within_label_device_score_shift",
        "device_id_only_negative_control_context_mean",
    )}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
