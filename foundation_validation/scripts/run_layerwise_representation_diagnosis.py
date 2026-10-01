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
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.datasets.builders import _select_records, load_configured_records, make_loader
from src.datasets.subject_activity import SubjectActivityDataset
from src.models import build_model
from src.utils.config import load_config


LAYER_ORDER = (
    "wrist_encoder",
    "bilateral_activity",
    "activity_context",
    "subject_aggregation",
    "classifier_logits",
)
TOKEN_LAYERS = {"wrist_encoder", "bilateral_activity", "activity_context"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen V8-GN layer-wise representation diagnosis")
    parser.add_argument("--run", action="append", required=True, help="SEED:/absolute/run/path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-folds", type=int, default=0)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_fold_datasets(stage: Path, config: dict[str, Any]):
    split = read_json(stage / "split.json")
    normalization = read_json(stage / "normalization.json")
    mean = torch.tensor(normalization["mean"], dtype=torch.float32)
    std = torch.tensor(normalization["std"], dtype=torch.float32)
    data = config["data"]
    records = load_configured_records(data, ["PD", "DD"])
    train_records = _select_records(records, set(split["train_subjects"]))
    val_records = _select_records(records, set(split["validation_subjects"]))
    train = SubjectActivityDataset(train_records, data["activities"], data, mean, std, "validation")
    val = SubjectActivityDataset(val_records, data["activities"], data, mean, std, "validation")
    condition = {record.subject_id: record.source_condition for record in records}
    return train, val, condition


@torch.inference_mode()
def encode_layers(model, inputs, wrist_mask, activity_mask, activity_lengths):
    batch, activities = inputs.shape[:2]
    flat_inputs = inputs.reshape(batch * activities, 2, inputs.shape[3], inputs.shape[4])
    flat_wrist_mask = wrist_mask.reshape(-1, 2)
    flat_lengths = activity_lengths.reshape(-1)
    valid_activities = activity_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    wrist_result = inputs.new_zeros((batch * activities, 2, model.wrist_encoder.feature_dim))
    activity_result = inputs.new_zeros((batch * activities, model.wrist_fusion.fusion_dim))
    for length in torch.unique(flat_lengths.index_select(0, valid_activities)).tolist():
        selected_activity = valid_activities[
            flat_lengths.index_select(0, valid_activities) == int(length)
        ]
        selected = flat_inputs.index_select(0, selected_activity)[..., : int(length)]
        selected_wrist_mask = flat_wrist_mask.index_select(0, selected_activity)
        wrist_flat = selected.reshape(-1, selected.shape[2], selected.shape[3])
        valid_wrist = selected_wrist_mask.reshape(-1).bool()
        wrist_indices = valid_wrist.nonzero(as_tuple=False).squeeze(1)
        encoded = model.wrist_encoder(wrist_flat.index_select(0, wrist_indices))["bag_embedding"]
        wrists = selected.new_zeros((selected.shape[0] * 2, model.wrist_encoder.feature_dim))
        wrists = wrists.index_copy(0, wrist_indices, encoded).reshape(selected.shape[0], 2, -1)
        fused, masked_wrists = model.wrist_fusion.fusion_features(wrists, selected_wrist_mask)
        wrist_result = wrist_result.index_copy(0, selected_activity, masked_wrists)
        activity_result = activity_result.index_copy(0, selected_activity, fused)
    wrist_result = wrist_result.reshape(batch, activities, 2, -1)
    activity_result = activity_result.reshape(batch, activities, -1)
    aggregated = model.activity_aggregator(activity_result, activity_mask)
    subject = aggregated["subject_embedding"]
    logits = model.classifier(subject)
    return {
        "wrist_encoder": wrist_result,
        "bilateral_activity": activity_result,
        "activity_context": aggregated["activity_embeddings"],
        "subject_aggregation": subject,
        "classifier_logits": logits,
        "attention": aggregated["activity_attention"],
    }


def masked_wrist_activity_mean(wrist, wrist_mask):
    mask = wrist_mask[..., None].astype(np.float64)
    return (wrist * mask).sum(axis=2) / np.clip(mask.sum(axis=2), 1.0, None)


def masked_token_mean(tokens, activity_mask):
    mask = activity_mask[..., None].astype(np.float64)
    return (tokens * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1.0, None)


@torch.inference_mode()
def extract_dataset(model, dataset, config, device, condition):
    loader = make_loader(
        dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
        pin_memory=False,
        seed=int(config["experiment"]["seed"]) + 1701,
    )
    collected: dict[str, list[np.ndarray]] = {layer: [] for layer in LAYER_ORDER}
    collected.update({"attention": [], "activity_mask": [], "wrist_mask": [], "label": []})
    subject_ids: list[str] = []
    conditions: list[str] = []
    max_equivalence = {"activity": 0.0, "subject": 0.0, "logits": 0.0}
    checked = False
    model.eval()
    for batch in loader:
        inputs = batch["x"].to(device)
        wrist_mask = batch["wrist_mask"].to(device)
        activity_mask = batch["activity_mask"].to(device)
        lengths = batch["activity_lengths"].to(device)
        outputs = encode_layers(model, inputs, wrist_mask, activity_mask, lengths)
        if not checked:
            reference_activity = model._encode_activities(inputs, wrist_mask, activity_mask, lengths)
            reference = model(inputs, wrist_mask, activity_mask, lengths)
            max_equivalence["activity"] = float((outputs["bilateral_activity"] - reference_activity).abs().max())
            max_equivalence["subject"] = float((outputs["subject_aggregation"] - reference["bag_embedding"]).abs().max())
            max_equivalence["logits"] = float((outputs["classifier_logits"] - reference["logits"]).abs().max())
            if max(max_equivalence.values()) > 1.0e-5:
                raise RuntimeError(f"Layer extraction does not match frozen forward pass: {max_equivalence}")
            checked = True
        for key in LAYER_ORDER:
            collected[key].append(outputs[key].cpu().numpy())
        collected["attention"].append(outputs["attention"].cpu().numpy())
        collected["activity_mask"].append(batch["activity_mask"].numpy())
        collected["wrist_mask"].append(batch["wrist_mask"].numpy())
        collected["label"].append(batch["y"].numpy())
        ids = [str(value) for value in batch["subject_id"]]
        subject_ids.extend(ids)
        conditions.extend(str(condition.get(value, "unknown")) for value in ids)
    result = {key: np.concatenate(value) for key, value in collected.items()}
    result["subject_id"] = np.asarray(subject_ids)
    result["condition"] = np.asarray(conditions)
    result["wrist_activity_tokens"] = masked_wrist_activity_mean(
        result["wrist_encoder"], result["wrist_mask"]
    )
    result["layer_subject"] = {
        "wrist_encoder": masked_token_mean(result["wrist_activity_tokens"], result["activity_mask"]),
        "bilateral_activity": masked_token_mean(result["bilateral_activity"], result["activity_mask"]),
        "activity_context": masked_token_mean(result["activity_context"], result["activity_mask"]),
        "subject_aggregation": result["subject_aggregation"],
        "classifier_logits": result["classifier_logits"],
    }
    result["layer_tokens"] = {
        "wrist_encoder": result["wrist_activity_tokens"],
        "bilateral_activity": result["bilateral_activity"],
        "activity_context": result["activity_context"],
    }
    return result, max_equivalence


def safe_silhouette(x, y, metric="euclidean"):
    counts = np.unique(y, return_counts=True)[1]
    if len(counts) < 2 or counts.min() < 2:
        return float("nan")
    return float(silhouette_score(x, y, metric=metric))


def disease_probe(train_x, train_y, val_x, val_y):
    scaler = StandardScaler().fit(train_x)
    z_train = scaler.transform(train_x)
    z_val = scaler.transform(val_x)
    classifier = LogisticRegression(
        max_iter=2500, class_weight="balanced", solver="lbfgs", C=1.0
    ).fit(z_train, train_y)
    probability = classifier.predict_proba(z_val)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    centroids = np.stack([z_train[train_y == label].mean(axis=0) for label in (0, 1)])
    distances = np.linalg.norm(z_val[:, None, :] - centroids[None, :, :], axis=-1)
    centroid_prediction = distances.argmin(axis=1)
    val_centroids = np.stack([z_val[val_y == label].mean(axis=0) for label in (0, 1)])
    within = float(np.mean([
        np.mean(np.sum((z_val[val_y == label] - val_centroids[label]) ** 2, axis=1))
        for label in (0, 1)
    ]))
    between = float(np.sum((val_centroids[0] - val_centroids[1]) ** 2))
    return {
        "disease_probe_ba": float(balanced_accuracy_score(val_y, prediction)),
        "disease_probe_auroc": float(roc_auc_score(val_y, probability)),
        "disease_centroid_ba": float(balanced_accuracy_score(val_y, centroid_prediction)),
        "disease_silhouette_euclidean": safe_silhouette(z_val, val_y, "euclidean"),
        "disease_silhouette_cosine": safe_silhouette(z_val, val_y, "cosine"),
        "within_class_distance": within,
        "between_class_distance": between,
        "between_within_distance_ratio": between / max(within, 1.0e-12),
    }, z_train, z_val


def variance_decomposition(z, labels):
    global_mean = z.mean(axis=0)
    between = 0.0
    within = 0.0
    for label in (0, 1):
        group = z[labels == label]
        centroid = group.mean(axis=0)
        between += len(group) * float(np.sum((centroid - global_mean) ** 2))
        within += float(np.sum((group - centroid) ** 2))
    total = between + within
    return {
        "disease_between_variance_ratio": between / max(total, 1.0e-12),
        "subject_within_disease_variance_ratio": within / max(total, 1.0e-12),
        "between_within_variance_ratio": between / max(within, 1.0e-12),
    }


def disease_controlled_domain_auc(train_x, train_y, val_x, val_y, seed):
    values = []
    for disease in (0, 1):
        left = train_x[train_y == disease]
        right = val_x[val_y == disease]
        x = np.concatenate([left, right])
        y = np.concatenate([np.zeros(len(left), dtype=int), np.ones(len(right), dtype=int)])
        folds = min(5, int(np.bincount(y).min()))
        cv = StratifiedKFold(folds, shuffle=True, random_state=seed + disease)
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1500, class_weight="balanced", solver="liblinear"),
        )
        score = cross_val_predict(estimator, x, y, cv=cv, method="predict_proba")[:, 1]
        values.append(float(roc_auc_score(y, score)))
    return float(np.mean(values))


def representation_shift(train_x, train_y, val_x, val_y, z_train, z_val, seed):
    mean_shift = float(
        np.linalg.norm(z_train.mean(0) - z_val.mean(0))
        / max(math.sqrt(np.mean(np.sum((z_train - z_train.mean(0)) ** 2, axis=1))), 1.0e-12)
    )
    cov_train = np.atleast_2d(np.cov(z_train, rowvar=False))
    cov_val = np.atleast_2d(np.cov(z_val, rowvar=False))
    coral = float(
        np.linalg.norm(cov_train - cov_val, ord="fro")
        / max(0.5 * (np.linalg.norm(cov_train, ord="fro") + np.linalg.norm(cov_val, ord="fro")), 1.0e-12)
    )
    return {
        "domain_probe_auc": disease_controlled_domain_auc(train_x, train_y, val_x, val_y, seed),
        "normalized_mean_shift": mean_shift,
        "coral_covariance_shift": coral,
        "pd_centroid_drift": float(np.linalg.norm(z_train[train_y == 0].mean(0) - z_val[val_y == 0].mean(0))),
        "dd_centroid_drift": float(np.linalg.norm(z_train[train_y == 1].mean(0) - z_val[val_y == 1].mean(0))),
    }


def token_identity_activity(train_tokens, val_tokens, val_labels):
    subjects_train, activities, dim = train_tokens.shape
    subjects_val = len(val_tokens)
    activity_train = np.tile(np.arange(activities), subjects_train)
    activity_val = np.tile(np.arange(activities), subjects_val)
    scaler = StandardScaler().fit(train_tokens.reshape(-1, dim))
    train_z = scaler.transform(train_tokens.reshape(-1, dim)).reshape(subjects_train, activities, dim)
    val_z = scaler.transform(val_tokens.reshape(-1, dim)).reshape(subjects_val, activities, dim)
    classifier = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", C=1.0)
    classifier.fit(train_z.reshape(-1, dim), activity_train)
    activity_accuracy = float(accuracy_score(activity_val, classifier.predict(val_z.reshape(-1, dim))))
    train_activity_centroid = train_z.mean(axis=0)
    residual = val_z - train_activity_centroid[None]
    normed = residual / np.clip(np.linalg.norm(residual, axis=-1, keepdims=True), 1.0e-12, None)
    ranks, self_similarity, competitor_similarity = [], [], []
    for activity in range(activities):
        other = [index for index in range(activities) if index != activity]
        candidates = normed[:, other].mean(axis=1)
        candidates /= np.clip(np.linalg.norm(candidates, axis=1, keepdims=True), 1.0e-12, None)
        similarities = normed[:, activity] @ candidates.T
        order = np.argsort(-similarities, axis=1)
        for subject in range(subjects_val):
            rank = int(np.where(order[subject] == subject)[0][0]) + 1
            ranks.append(rank)
            self_similarity.append(float(similarities[subject, subject]))
            same_disease = np.where(
                (val_labels == val_labels[subject]) & (np.arange(subjects_val) != subject)
            )[0]
            competitor_similarity.append(float(similarities[subject, same_disease].mean()))
    ranks = np.asarray(ranks)
    return {
        "activity_probe_accuracy": activity_accuracy,
        "subject_retrieval_top1": float(np.mean(ranks == 1)),
        "subject_retrieval_top5": float(np.mean(ranks <= 5)),
        "subject_retrieval_mrr": float(np.mean(1.0 / ranks)),
        "same_subject_similarity": float(np.mean(self_similarity)),
        "same_disease_other_subject_similarity": float(np.mean(competitor_similarity)),
        "subject_similarity_gap": float(np.mean(np.asarray(self_similarity) - np.asarray(competitor_similarity))),
    }


def dd_subtype_diagnostics(train_z, train_labels, train_condition, val_z, val_labels, val_condition):
    train_mask = train_labels == 1
    val_mask = val_labels == 1
    train_names = train_condition[train_mask]
    val_names = val_condition[val_mask]
    common = sorted(set(train_names) & set(val_names))
    common = [name for name in common if np.sum(train_names == name) >= 2 and np.sum(val_names == name) >= 2]
    if len(common) < 2:
        return {"dd_subtype_silhouette": float("nan"), "dd_subtype_centroid_accuracy": float("nan"), "dd_subtype_count": len(common)}
    keep_val = np.isin(val_names, common)
    x_val = val_z[val_mask][keep_val]
    y_val = val_names[keep_val]
    centroids = np.stack([train_z[train_mask][train_names == name].mean(0) for name in common])
    prediction = np.asarray(common)[np.linalg.norm(x_val[:, None] - centroids[None], axis=-1).argmin(axis=1)]
    return {
        "dd_subtype_silhouette": safe_silhouette(x_val, y_val, "cosine"),
        "dd_subtype_centroid_accuracy": float(accuracy_score(y_val, prediction)),
        "dd_subtype_count": len(common),
    }


def head_metrics(logits, labels):
    shifted = logits - logits.max(axis=1, keepdims=True)
    probability = np.exp(shifted)[:, 1] / np.exp(shifted).sum(axis=1)
    prediction = (probability >= 0.5).astype(int)
    return {
        "head_balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "head_auroc": float(roc_auc_score(labels, probability)),
    }


def paired_wilcoxon(values):
    values = np.asarray(values, dtype=float)
    nonzero = values[np.abs(values) > 1.0e-12]
    if len(nonzero) == 0:
        return 1.0
    return float(wilcoxon(nonzero, alternative="two-sided").pvalue)


def main() -> None:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    embedding_dir = output / "embeddings"
    embedding_dir.mkdir(exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Layer-wise extraction requires CUDA")
    runs = []
    for item in args.run:
        seed, path = item.split(":", 1)
        runs.append((int(seed), Path(path).expanduser().resolve()))
    fold_rows, equivalence_rows = [], []
    processed = 0
    for seed, run in runs:
        for stage in sorted(run.glob("outer_*/inner_*")):
            if args.max_folds and processed >= args.max_folds:
                break
            outer = int(stage.parent.name.split("_")[-1])
            inner = int(stage.name.split("_")[-1])
            config = load_config(stage / "config.yaml")
            model = build_model(config).to(device)
            checkpoint = torch.load(stage / "checkpoints/best.pt", map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state"], strict=True)
            train_dataset, val_dataset, condition = load_fold_datasets(stage, config)
            train, train_equivalence = extract_dataset(model, train_dataset, config, device, condition)
            val, val_equivalence = extract_dataset(model, val_dataset, config, device, condition)
            equivalence_rows.append({
                "seed": seed, "outer": outer, "inner": inner,
                **{f"train_{key}": value for key, value in train_equivalence.items()},
                **{f"validation_{key}": value for key, value in val_equivalence.items()},
            })
            head = head_metrics(val["classifier_logits"], val["label"])
            archive: dict[str, Any] = {
                "validation_subject_id": val["subject_id"],
                "validation_label": val["label"],
                "validation_condition": val["condition"],
                "validation_wrist_tokens": val["layer_tokens"]["wrist_encoder"],
                "validation_activity_fused": val["layer_tokens"]["bilateral_activity"],
                "validation_activity_context": val["layer_tokens"]["activity_context"],
            }
            for layer in LAYER_ORDER:
                train_x = train["layer_subject"][layer]
                val_x = val["layer_subject"][layer]
                disease, z_train, z_val = disease_probe(train_x, train["label"], val_x, val["label"])
                variance = variance_decomposition(z_val, val["label"])
                shift = representation_shift(
                    train_x, train["label"], val_x, val["label"], z_train, z_val,
                    seed + outer * 100 + inner * 10 + LAYER_ORDER.index(layer),
                )
                token_metrics = {
                    "activity_probe_accuracy": float("nan"),
                    "subject_retrieval_top1": float("nan"),
                    "subject_retrieval_top5": float("nan"),
                    "subject_retrieval_mrr": float("nan"),
                    "same_subject_similarity": float("nan"),
                    "same_disease_other_subject_similarity": float("nan"),
                    "subject_similarity_gap": float("nan"),
                }
                if layer in TOKEN_LAYERS:
                    token_metrics = token_identity_activity(
                        train["layer_tokens"][layer], val["layer_tokens"][layer], val["label"]
                    )
                subtype = dd_subtype_diagnostics(
                    z_train, train["label"], train["condition"],
                    z_val, val["label"], val["condition"],
                )
                fold_rows.append({
                    "seed": seed, "outer": outer, "inner": inner, "layer": layer,
                    "representation_dim": int(train_x.shape[1]), **head,
                    **disease, **variance, **shift, **token_metrics, **subtype,
                })
                archive[f"train_subject_{layer}"] = train_x
                archive[f"validation_subject_{layer}"] = val_x
            np.savez_compressed(
                embedding_dir / f"seed{seed}_outer{outer}_inner{inner}.npz", **archive
            )
            processed += 1
            print(f"processed seed={seed} outer={outer} inner={inner}", flush=True)
        if args.max_folds and processed >= args.max_folds:
            break
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(output / "layer_fold_metrics_45_seed_folds.csv", index=False)
    pd.DataFrame(equivalence_rows).to_csv(output / "extraction_equivalence_checks.csv", index=False)
    numeric = [column for column in folds.columns if column not in {"seed", "outer", "inner", "layer"}]
    aggregated = folds.groupby(["outer", "inner", "layer"], as_index=False)[numeric].mean()
    aggregated.to_csv(output / "layer_metrics_15_independent_splits.csv", index=False)
    summary_rows = []
    for layer, group in aggregated.groupby("layer", sort=False):
        for metric in numeric:
            values = group[metric].dropna()
            summary_rows.append({
                "layer": layer, "metric": metric, "n_splits": int(len(values)),
                "mean": float(values.mean()) if len(values) else float("nan"),
                "std_across_splits": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                "minimum": float(values.min()) if len(values) else float("nan"),
                "maximum": float(values.max()) if len(values) else float("nan"),
            })
    pd.DataFrame(summary_rows).to_csv(output / "layer_metric_summary.csv", index=False)
    relationships = []
    relation_metrics = [
        "disease_probe_ba", "disease_probe_auroc", "disease_centroid_ba",
        "disease_silhouette_cosine", "between_within_distance_ratio",
        "disease_between_variance_ratio", "subject_within_disease_variance_ratio",
        "activity_probe_accuracy", "subject_retrieval_top1", "subject_similarity_gap",
        "domain_probe_auc", "normalized_mean_shift", "coral_covariance_shift",
    ]
    for layer, group in aggregated.groupby("layer", sort=False):
        for metric in relation_metrics:
            valid = group[[metric, "head_balanced_accuracy"]].dropna()
            variable = (
                len(valid) >= 3
                and valid[metric].nunique() > 1
                and valid["head_balanced_accuracy"].nunique() > 1
            )
            rho, p = (
                spearmanr(valid[metric], valid["head_balanced_accuracy"])
                if variable else (float("nan"), float("nan"))
            )
            relationships.append({
                "layer": layer, "metric": metric, "n_splits": int(len(valid)),
                "spearman_rho_with_head_ba": float(rho), "p_value": float(p),
            })
    pd.DataFrame(relationships).to_csv(output / "layer_metric_head_ba_relationships.csv", index=False)
    transitions = []
    transition_metrics = [
        "disease_probe_ba", "disease_probe_auroc", "disease_centroid_ba",
        "disease_silhouette_cosine", "between_within_distance_ratio",
        "disease_between_variance_ratio", "subject_within_disease_variance_ratio",
        "activity_probe_accuracy", "subject_retrieval_top1", "subject_retrieval_top5",
        "subject_similarity_gap",
        "domain_probe_auc", "normalized_mean_shift", "coral_covariance_shift",
    ]
    indexed = aggregated.set_index(["outer", "inner", "layer"])
    split_keys = sorted(set((int(row.outer), int(row.inner)) for row in aggregated.itertuples()))
    for previous, current in zip(LAYER_ORDER[:-1], LAYER_ORDER[1:]):
        for metric in transition_metrics:
            delta = np.asarray([
                indexed.loc[(outer, inner, current), metric] - indexed.loc[(outer, inner, previous), metric]
                for outer, inner in split_keys
            ], dtype=float)
            delta = delta[np.isfinite(delta)]
            transitions.append({
                "transition": f"{previous}->{current}", "metric": metric,
                "n_splits": int(len(delta)), "mean_delta": float(delta.mean()),
                "positive_splits": int(np.sum(delta > 0)), "negative_splits": int(np.sum(delta < 0)),
                "wilcoxon_p": paired_wilcoxon(delta),
            })
    pd.DataFrame(transitions).to_csv(output / "adjacent_layer_transitions.csv", index=False)
    protocol = {
        "scope": "fixed inner-development train/validation only",
        "outer_test_accessed": False,
        "architecture_modified": False,
        "model_retrained": False,
        "training_modules_added": False,
        "seeds": sorted(int(value) for value in folds.seed.unique()),
        "raw_seed_fold_layer_rows": int(len(folds)),
        "primary_independent_splits": int(aggregated[["outer", "inner"]].drop_duplicates().shape[0]),
        "seed_aggregation": "arithmetic mean within identical outer/inner split before inference",
        "layer_definitions": {
            "wrist_encoder": "64-D wrist encoder output; wrists averaged within activity and activities mean-pooled only for disease/shift comparison",
            "bilateral_activity": "258-D bilateral fusion output before activity identity embedding",
            "activity_context": "258-D activity representation after activity embedding plus LayerNorm",
            "subject_aggregation": "258-D learned attention-weighted subject embedding",
            "classifier_logits": "2-D frozen classifier logits",
        },
        "activity_subject_metrics_note": "Activity probe and cross-activity subject retrieval are undefined after subject aggregation and recorded as NaN.",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(protocol, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
