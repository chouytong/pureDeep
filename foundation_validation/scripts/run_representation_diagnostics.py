#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.datasets.builders import _select_records, load_configured_records, make_loader
from src.datasets.subject_activity import SubjectActivityDataset
from src.models import build_model
from src.utils.config import load_config


ACTIVITIES = (
    "CrossArms", "DrinkGlas", "Entrainment", "HoldWeight", "LiftHold",
    "PointFinger", "Relaxed", "RelaxedTask", "StretchHold", "TouchIndex",
    "TouchNose",
)
METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "auroc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen V8-GN representation diagnostics on inner development only"
    )
    parser.add_argument(
        "--run", action="append", required=True,
        help="SEED:/absolute/path/to/frozen/development/run",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-folds", type=int, default=0)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_dict(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(y, probability)),
    }


def safe_silhouette(x: np.ndarray, y: np.ndarray, metric: str) -> float:
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 2:
        return float("nan")
    return float(silhouette_score(x, y, metric=metric))


def load_fold_datasets(stage: Path, config: dict[str, Any]):
    split = read_json(stage / "split.json")
    normalization = read_json(stage / "normalization.json")
    mean = torch.tensor(normalization["mean"], dtype=torch.float32)
    std = torch.tensor(normalization["std"], dtype=torch.float32)
    data = config["data"]
    records = load_configured_records(data, ["PD", "DD"])
    train_records = _select_records(records, set(split["train_subjects"]))
    validation_records = _select_records(records, set(split["validation_subjects"]))
    train = SubjectActivityDataset(
        train_records, data["activities"], data, mean, std, "validation"
    )
    validation = SubjectActivityDataset(
        validation_records, data["activities"], data, mean, std, "validation"
    )
    condition = {}
    for record in records:
        condition.setdefault(record.subject_id, record.source_condition)
    return train, validation, condition


@torch.inference_mode()
def extract_dataset(model, dataset, config, device: torch.device) -> dict[str, Any]:
    loader = make_loader(
        dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
        pin_memory=False,
        seed=int(config["experiment"]["seed"]) + 991,
    )
    subject_ids: list[str] = []
    labels: list[np.ndarray] = []
    subject_embeddings: list[np.ndarray] = []
    raw_activity_embeddings: list[np.ndarray] = []
    final_activity_embeddings: list[np.ndarray] = []
    attentions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    model.eval()
    for batch in loader:
        x = batch["x"].to(device)
        wrist = batch["wrist_mask"].to(device)
        mask = batch["activity_mask"].to(device)
        lengths = batch["activity_lengths"].to(device)
        raw = model._encode_activities(x, wrist, mask, lengths)
        aggregated = model.activity_aggregator(raw, mask)
        logits = model.classifier(aggregated["subject_embedding"])
        subject_ids.extend(str(value) for value in batch["subject_id"])
        labels.append(batch["y"].numpy())
        subject_embeddings.append(aggregated["subject_embedding"].cpu().numpy())
        raw_activity_embeddings.append(raw.cpu().numpy())
        final_activity_embeddings.append(
            aggregated["activity_embeddings"].cpu().numpy()
        )
        attentions.append(aggregated["activity_attention"].cpu().numpy())
        probabilities.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return {
        "subject_id": np.asarray(subject_ids),
        "label": np.concatenate(labels),
        "subject_embedding": np.concatenate(subject_embeddings),
        "raw_activity_embedding": np.concatenate(raw_activity_embeddings),
        "final_activity_embedding": np.concatenate(final_activity_embeddings),
        "attention": np.concatenate(attentions),
        "probability_dd": np.concatenate(probabilities),
    }


def disease_probe(train: dict[str, Any], val: dict[str, Any]):
    scaler = StandardScaler().fit(train["subject_embedding"])
    x_train = scaler.transform(train["subject_embedding"])
    x_val = scaler.transform(val["subject_embedding"])
    classifier = LogisticRegression(
        max_iter=2000, class_weight="balanced", solver="lbfgs", C=1.0
    ).fit(x_train, train["label"])
    probability = classifier.predict_proba(x_val)[:, 1]
    metrics = metric_dict(val["label"], probability)
    centroids = np.stack([
        x_train[train["label"] == label].mean(axis=0) for label in (0, 1)
    ])
    distances = np.linalg.norm(x_val[:, None, :] - centroids[None, :, :], axis=-1)
    centroid_prediction = distances.argmin(axis=1)
    centroid_ba = balanced_accuracy_score(val["label"], centroid_prediction)
    within = np.mean([
        np.mean(np.sum((x_val[val["label"] == label] - x_val[val["label"] == label].mean(0)) ** 2, axis=1))
        for label in (0, 1)
    ])
    fisher = float(np.sum((x_val[val["label"] == 0].mean(0) - x_val[val["label"] == 1].mean(0)) ** 2) / max(within, 1e-12))
    true_distance = distances[np.arange(len(x_val)), val["label"]]
    other_distance = distances[np.arange(len(x_val)), 1 - val["label"]]
    return {
        **{f"disease_probe_{key}": value for key, value in metrics.items()},
        "disease_centroid_ba": float(centroid_ba),
        "disease_silhouette_euclidean": safe_silhouette(x_val, val["label"], "euclidean"),
        "disease_silhouette_cosine": safe_silhouette(x_val, val["label"], "cosine"),
        "disease_fisher_ratio": fisher,
    }, x_train, x_val, other_distance - true_distance


def domain_auc(x_train: np.ndarray, y_train: np.ndarray,
               x_val: np.ndarray, y_val: np.ndarray, seed: int) -> tuple[float, float]:
    values = []
    for disease in (0, 1):
        left = x_train[y_train == disease]
        right = x_val[y_val == disease]
        x = np.concatenate([left, right])
        y = np.concatenate([np.zeros(len(left), dtype=int), np.ones(len(right), dtype=int)])
        folds = min(5, int(np.bincount(y).min()))
        cv = StratifiedKFold(folds, shuffle=True, random_state=seed + disease)
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"),
        )
        score = cross_val_predict(estimator, x, y, cv=cv, method="predict_proba")[:, 1]
        values.append(float(roc_auc_score(y, score)))
    return float(np.mean(values)), float(max(values))


def representation_shift(train: dict[str, Any], val: dict[str, Any],
                         x_train: np.ndarray, x_val: np.ndarray, seed: int):
    domain_mean, domain_max = domain_auc(
        train["subject_embedding"], train["label"],
        val["subject_embedding"], val["label"], seed,
    )
    mean_shift = float(np.linalg.norm(x_train.mean(0) - x_val.mean(0)) /
                       math.sqrt(np.mean(np.sum((x_train - x_train.mean(0)) ** 2, axis=1))))
    cov_train = np.cov(x_train, rowvar=False)
    cov_val = np.cov(x_val, rowvar=False)
    coral = float(np.linalg.norm(cov_train - cov_val, ord="fro") /
                  max(0.5 * (np.linalg.norm(cov_train, ord="fro") + np.linalg.norm(cov_val, ord="fro")), 1e-12))
    drifts = []
    for label in (0, 1):
        drifts.append(float(np.linalg.norm(
            x_train[train["label"] == label].mean(0) -
            x_val[val["label"] == label].mean(0)
        )))
    return {
        "domain_probe_auc_disease_controlled_mean": domain_mean,
        "domain_probe_auc_disease_controlled_max": domain_max,
        "normalized_mean_shift": mean_shift,
        "coral_covariance_shift": coral,
        "pd_centroid_drift": drifts[0],
        "dd_centroid_drift": drifts[1],
    }


def activity_and_identity(train: dict[str, Any], val: dict[str, Any]):
    a = train["raw_activity_embedding"].shape[1]
    d = train["raw_activity_embedding"].shape[2]
    flat_train = train["raw_activity_embedding"].reshape(-1, d)
    flat_val = val["raw_activity_embedding"].reshape(-1, d)
    activity_train = np.tile(np.arange(a), len(train["label"]))
    activity_val = np.tile(np.arange(a), len(val["label"]))
    scaler = StandardScaler().fit(flat_train)
    z_train = scaler.transform(flat_train).reshape(len(train["label"]), a, d)
    z_val = scaler.transform(flat_val).reshape(len(val["label"]), a, d)
    classifier = LogisticRegression(
        max_iter=1500, class_weight="balanced", solver="lbfgs", C=1.0
    ).fit(z_train.reshape(-1, d), activity_train)
    activity_prediction = classifier.predict(z_val.reshape(-1, d))
    activity_accuracy = float(accuracy_score(activity_val, activity_prediction))
    train_activity_centroids = z_train.mean(axis=0)
    residual = z_val - train_activity_centroids[None]
    normed = residual / np.clip(np.linalg.norm(residual, axis=-1, keepdims=True), 1e-12, None)
    ranks = []
    self_similarities = []
    competitor_similarities = []
    top1 = 0
    top5 = 0
    n = len(val["label"])
    for activity in range(a):
        other = [index for index in range(a) if index != activity]
        candidates = normed[:, other].mean(axis=1)
        candidates /= np.clip(np.linalg.norm(candidates, axis=1, keepdims=True), 1e-12, None)
        similarities = normed[:, activity] @ candidates.T
        order = np.argsort(-similarities, axis=1)
        for subject in range(n):
            rank = int(np.where(order[subject] == subject)[0][0]) + 1
            ranks.append(rank)
            top1 += rank == 1
            top5 += rank <= 5
            self_similarities.append(float(similarities[subject, subject]))
            same_disease = np.where((val["label"] == val["label"][subject]) & (np.arange(n) != subject))[0]
            competitor_similarities.append(float(similarities[subject, same_disease].mean()))
    subject_means = residual.mean(axis=1)
    global_mean = residual.mean(axis=(0, 1))
    total_ss = float(np.sum((residual - global_mean) ** 2))
    disease_means = np.stack([subject_means[val["label"] == label].mean(0) for label in (0, 1)])
    disease_ss = float(a * sum(
        np.sum(val["label"] == label) * np.sum((disease_means[label] - global_mean) ** 2)
        for label in (0, 1)
    ))
    subject_ss = float(a * sum(
        np.sum((subject_means[index] - disease_means[val["label"][index]]) ** 2)
        for index in range(n)
    ))
    activity_means = residual.mean(axis=0)
    activity_ss = float(n * np.sum((activity_means - global_mean) ** 2))
    dispersion = np.mean(np.linalg.norm(residual - subject_means[:, None, :], axis=-1), axis=1)
    return {
        "activity_probe_accuracy": activity_accuracy,
        "subject_retrieval_top1": float(top1 / len(ranks)),
        "subject_retrieval_top5": float(top5 / len(ranks)),
        "subject_retrieval_mrr": float(np.mean(1.0 / np.asarray(ranks))),
        "subject_retrieval_chance": float(1.0 / n),
        "same_subject_similarity": float(np.mean(self_similarities)),
        "same_disease_other_subject_similarity": float(np.mean(competitor_similarities)),
        "subject_similarity_gap": float(np.mean(np.asarray(self_similarities) - np.asarray(competitor_similarities))),
        "disease_variance_ratio": disease_ss / max(total_ss, 1e-12),
        "subject_within_disease_variance_ratio": subject_ss / max(total_ss, 1e-12),
        "activity_shift_variance_ratio": activity_ss / max(total_ss, 1e-12),
    }, residual, dispersion


def per_activity_disease(train: dict[str, Any], val: dict[str, Any], seed: int,
                         outer: int, inner: int):
    rows = []
    for index, activity in enumerate(ACTIVITIES):
        pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"),
        )
        pipeline.fit(train["raw_activity_embedding"][:, index], train["label"])
        probability = pipeline.predict_proba(val["raw_activity_embedding"][:, index])[:, 1]
        metrics = metric_dict(val["label"], probability)
        rows.append({"seed": seed, "outer": outer, "inner": inner,
                     "activity": activity, **metrics,
                     "mean_attention": float(val["attention"][:, index].mean())})
    return rows


def error_rows(seed: int, outer: int, inner: int, train: dict[str, Any],
               val: dict[str, Any], x_train: np.ndarray, x_val: np.ndarray,
               centroid_margin: np.ndarray, residual: np.ndarray,
               dispersion: np.ndarray, condition: dict[str, str]):
    train_norm = x_train / np.clip(np.linalg.norm(x_train, axis=1, keepdims=True), 1e-12, None)
    val_norm = x_val / np.clip(np.linalg.norm(x_val, axis=1, keepdims=True), 1e-12, None)
    nearest = np.argmax(val_norm @ train_norm.T, axis=1)
    attention = val["attention"]
    entropy = -np.sum(attention * np.log(np.clip(attention, 1e-12, None)), axis=1) / math.log(attention.shape[1])
    prediction = (val["probability_dd"] >= 0.5).astype(int)
    rows = []
    for index, subject_id in enumerate(val["subject_id"]):
        subject_activity = residual[index]
        normed = subject_activity / np.clip(np.linalg.norm(subject_activity, axis=1, keepdims=True), 1e-12, None)
        consistency = float(np.mean(normed @ normed.T))
        rows.append({
            "seed": seed, "outer": outer, "inner": inner,
            "subject_id": str(subject_id), "label": int(val["label"][index]),
            "condition": str(condition.get(str(subject_id), "unknown")),
            "error": int(prediction[index] != val["label"][index]),
            "probability_dd": float(val["probability_dd"][index]),
            "confidence_margin": float(abs(val["probability_dd"][index] - 0.5) * 2),
            "centroid_margin": float(centroid_margin[index]),
            "embedding_norm": float(np.linalg.norm(val["subject_embedding"][index])),
            "attention_entropy": float(entropy[index]),
            "attention_max": float(attention[index].max()),
            "activity_dispersion": float(dispersion[index]),
            "activity_consistency": consistency,
            "nearest_train_label_agreement": int(train["label"][nearest[index]] == val["label"][index]),
        })
    return rows


def representative_pca(train: dict[str, Any], val: dict[str, Any], output: Path):
    scaler = StandardScaler().fit(train["subject_embedding"])
    x = np.concatenate([
        scaler.transform(train["subject_embedding"]),
        scaler.transform(val["subject_embedding"]),
    ])
    projection = PCA(n_components=2, random_state=42).fit_transform(x)
    split = np.array(["train"] * len(train["label"]) + ["validation"] * len(val["label"]))
    label = np.concatenate([train["label"], val["label"]])
    pd.DataFrame({
        "split": split,
        "label": label,
        "pc1": projection[:, 0],
        "pc2": projection[:, 1],
    }).to_csv(output, index=False)


def summarize_errors(frame: pd.DataFrame):
    features = [
        "confidence_margin", "centroid_margin", "embedding_norm", "attention_entropy",
        "attention_max", "activity_dispersion", "activity_consistency",
        "nearest_train_label_agreement",
    ]
    grouped = frame.groupby("subject_id").agg(
        label=("label", "first"), condition=("condition", "first"),
        appearances=("error", "size"), error_rate=("error", "mean"),
        **{feature: (feature, "mean") for feature in features},
    ).reset_index()
    grouped["stability_group"] = np.where(
        grouped.error_rate >= 0.75, "stable_error",
        np.where(grouped.error_rate <= 0.25, "stable_correct", "unstable"),
    )
    comparison = []
    left = grouped[grouped.stability_group == "stable_error"]
    right = grouped[grouped.stability_group == "stable_correct"]
    for feature in features:
        a = left[feature].to_numpy(dtype=float)
        b = right[feature].to_numpy(dtype=float)
        pooled = math.sqrt(max(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) /
                               max(len(a) + len(b) - 2, 1), 1e-12))
        effect = float((np.mean(a) - np.mean(b)) / pooled)
        statistic, p = mannwhitneyu(a, b, alternative="two-sided")
        comparison.append({
            "feature": feature,
            "stable_error_mean": float(np.mean(a)),
            "stable_correct_mean": float(np.mean(b)),
            "cohen_d_error_minus_correct": effect,
            "mannwhitney_u": float(statistic), "p_value": float(p),
        })
    return grouped, pd.DataFrame(comparison)


def main() -> None:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Embedding extraction requires CUDA for this audit")
    runs = []
    for item in args.run:
        seed_text, path_text = item.split(":", 1)
        runs.append((int(seed_text), Path(path_text).expanduser().resolve()))
    fold_rows = []
    activity_rows = []
    errors = []
    representative_written = False
    processed = 0
    extraction_dir = output / "embeddings"
    extraction_dir.mkdir(exist_ok=True)
    for seed, run in runs:
        for stage in sorted(run.glob("outer_*/inner_*")):
            if args.max_folds and processed >= args.max_folds:
                break
            outer = int(stage.parent.name.split("_")[-1])
            inner = int(stage.name.split("_")[-1])
            config = load_config(stage / "config.yaml")
            model = build_model(config).to(device)
            checkpoint = torch.load(
                stage / "checkpoints" / "best.pt", map_location=device,
                weights_only=False,
            )
            model.load_state_dict(checkpoint["model_state"], strict=True)
            train_dataset, val_dataset, condition = load_fold_datasets(stage, config)
            train = extract_dataset(model, train_dataset, config, device)
            val = extract_dataset(model, val_dataset, config, device)
            np.savez_compressed(
                extraction_dir / f"seed{seed}_outer{outer}_inner{inner}.npz",
                train_subject_id=train["subject_id"], train_label=train["label"],
                train_subject_embedding=train["subject_embedding"],
                train_raw_activity_embedding=train["raw_activity_embedding"],
                train_attention=train["attention"],
                validation_subject_id=val["subject_id"], validation_label=val["label"],
                validation_subject_embedding=val["subject_embedding"],
                validation_raw_activity_embedding=val["raw_activity_embedding"],
                validation_attention=val["attention"],
                validation_probability_dd=val["probability_dd"],
            )
            disease, x_train, x_val, centroid_margin = disease_probe(train, val)
            shift = representation_shift(train, val, x_train, x_val, seed + outer * 10 + inner)
            identity, residual, dispersion = activity_and_identity(train, val)
            head = metric_dict(val["label"], val["probability_dd"])
            fold_rows.append({
                "seed": seed, "outer": outer, "inner": inner,
                **{f"head_{key}": value for key, value in head.items()},
                **disease, **shift, **identity,
            })
            activity_rows.extend(per_activity_disease(train, val, seed, outer, inner))
            errors.extend(error_rows(
                seed, outer, inner, train, val, x_train, x_val,
                centroid_margin, residual, dispersion, condition,
            ))
            if not representative_written and seed == runs[0][0]:
                representative_pca(train, val, output / "representative_subject_pca.csv")
                representative_written = True
            processed += 1
            print(f"processed seed={seed} outer={outer} inner={inner}", flush=True)
        if args.max_folds and processed >= args.max_folds:
            break
    folds = pd.DataFrame(fold_rows)
    activities = pd.DataFrame(activity_rows)
    error_frame = pd.DataFrame(errors)
    subjects, error_comparison = summarize_errors(error_frame)
    folds.to_csv(output / "fold_diagnostics.csv", index=False)
    activities.to_csv(output / "activity_diagnostics.csv", index=False)
    error_frame.to_csv(output / "validation_appearances.csv", index=False)
    subjects.to_csv(output / "subject_error_stability.csv", index=False)
    error_comparison.to_csv(output / "stable_error_feature_comparison.csv", index=False)
    summary: dict[str, Any] = {
        "protocol": {
            "scope": "fixed inner-development train/validation only",
            "outer_test_accessed": False,
            "architecture_modified": False,
            "backbone_retrained": False,
            "fold_count": int(len(folds)),
            "seed_count": int(folds.seed.nunique()),
        },
        "fold_metrics": {},
        "activity_summary": {},
        "stable_errors": {
            "subject_count": int(len(subjects)),
            "stable_error_count": int((subjects.stability_group == "stable_error").sum()),
            "stable_correct_count": int((subjects.stability_group == "stable_correct").sum()),
            "unstable_count": int((subjects.stability_group == "unstable").sum()),
            "mean_appearances": float(subjects.appearances.mean()),
        },
        "shift_performance_correlations": {},
    }
    for column in folds.columns:
        if column in {"seed", "outer", "inner"}:
            continue
        summary["fold_metrics"][column] = {
            "mean": float(folds[column].mean()),
            "seed_level_means": {
                str(int(seed)): float(group[column].mean())
                for seed, group in folds.groupby("seed")
            },
            "fold_std": float(folds[column].std(ddof=1)),
            "minimum": float(folds[column].min()),
            "maximum": float(folds[column].max()),
        }
    for activity, group in activities.groupby("activity"):
        summary["activity_summary"][activity] = {
            key: float(group[key].mean())
            for key in (*METRICS, "mean_attention")
        }
    for shift_name in (
        "domain_probe_auc_disease_controlled_mean", "normalized_mean_shift",
        "coral_covariance_shift", "pd_centroid_drift", "dd_centroid_drift",
    ):
        rho, p = spearmanr(folds[shift_name], folds["head_balanced_accuracy"])
        summary["shift_performance_correlations"][shift_name] = {
            "spearman_rho_with_head_ba": float(rho), "p_value": float(p)
        }
    condition = subjects.groupby(["condition", "label"]).agg(
        subjects=("subject_id", "size"), mean_error_rate=("error_rate", "mean"),
        stable_error_rate=("stability_group", lambda value: float((value == "stable_error").mean())),
    ).reset_index()
    condition.to_csv(output / "condition_error_summary.csv", index=False)
    (output / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
