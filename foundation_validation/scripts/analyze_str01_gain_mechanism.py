#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import mannwhitneyu, rankdata, spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ACTIVITIES = (
    "CrossArms", "DrinkGlas", "Entrainment", "HoldWeight", "LiftHold",
    "PointFinger", "Relaxed", "RelaxedTask", "StretchHold", "TouchIndex",
    "TouchNose",
)
HIGH_RISK = {"CrossArms", "DrinkGlas", "HoldWeight"}
SIGNAL_FEATURES = (
    "motion_rms", "jerk_rms", "robust_outlier_fraction", "bandpower_0p5_3",
    "bandpower_3_7", "dominant_frequency_0p5_12",
    "spectral_entropy_0p5_25", "tremor_peak_ratio_3_7",
    "zero_lag_magnitude_correlation",
)


def arguments():
    parser = argparse.ArgumentParser(description="STR-01 gain mechanism diagnosis")
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--stable-audit-dir", type=Path, required=True)
    parser.add_argument("--h1-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def bh(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    result = np.empty(len(values), dtype=float)
    running = 1.0
    for position in range(len(values) - 1, -1, -1):
        index = order[position]
        running = min(running, values[index] * len(values) / (position + 1))
        result[index] = min(1.0, running)
    return result


def paired_p(delta):
    values = np.asarray(delta, dtype=float)
    values = values[np.isfinite(values) & (np.abs(values) > 1e-12)]
    return 1.0 if not len(values) else float(wilcoxon(values).pvalue)


def rank_biserial(delta):
    values = np.asarray(delta, dtype=float)
    values = values[np.isfinite(values) & (np.abs(values) > 1e-12)]
    if not len(values):
        return 0.0
    ranks = rankdata(np.abs(values), method="average")
    return float((ranks[values > 0].sum() - ranks[values < 0].sum()) / ranks.sum())


def cliffs_delta(left, right):
    left, right = np.asarray(left, float), np.asarray(right, float)
    if not len(left) or not len(right):
        return np.nan
    return float((np.greater(left[:, None], right[None]).sum()
                  - np.less(left[:, None], right[None]).sum())
                 / (len(left) * len(right)))


def safe_auc(y, score):
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan


def safe_silhouette(x, y, metric):
    counts = np.unique(y, return_counts=True)[1]
    if len(counts) < 2 or counts.min() < 2:
        return np.nan
    return float(silhouette_score(x, y, metric=metric))


def classification_metrics(y, margin):
    prediction = (margin >= 0).astype(int)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "auroc": safe_auc(y, margin),
        "pd_recall": float(np.mean(prediction[y == 0] == 0)),
        "dd_recall": float(np.mean(prediction[y == 1] == 1)),
    }


def disease_probe(train_x, train_y, validation_x, validation_y):
    scaler = StandardScaler().fit(train_x)
    z_train, z_validation = scaler.transform(train_x), scaler.transform(validation_x)
    model = LogisticRegression(
        max_iter=2500, class_weight="balanced", solver="lbfgs", C=1.0
    ).fit(z_train, train_y)
    probability = model.predict_proba(z_validation)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    centroids = np.stack([z_train[train_y == label].mean(0) for label in (0, 1)])
    centroid_prediction = np.linalg.norm(
        z_validation[:, None] - centroids[None], axis=-1
    ).argmin(1)
    return {
        "disease_probe_ba": float(balanced_accuracy_score(validation_y, prediction)),
        "disease_probe_auroc": safe_auc(validation_y, probability),
        "disease_centroid_ba": float(
            balanced_accuracy_score(validation_y, centroid_prediction)
        ),
        "disease_silhouette_euclidean": safe_silhouette(
            z_validation, validation_y, "euclidean"
        ),
        "disease_silhouette_cosine": safe_silhouette(
            z_validation, validation_y, "cosine"
        ),
    }, z_train, z_validation


def disease_controlled_domain_auc(train_x, train_y, validation_x, validation_y, seed):
    values = []
    for disease in (0, 1):
        left, right = train_x[train_y == disease], validation_x[validation_y == disease]
        x = np.concatenate([left, right])
        y = np.concatenate([
            np.zeros(len(left), dtype=int), np.ones(len(right), dtype=int)
        ])
        folds = min(5, int(np.bincount(y).min()))
        cv = StratifiedKFold(folds, shuffle=True, random_state=seed + disease)
        estimator = make_pipeline(
            StandardScaler(), LogisticRegression(
                max_iter=1500, class_weight="balanced", solver="liblinear"
            )
        )
        probability = cross_val_predict(
            estimator, x, y, cv=cv, method="predict_proba"
        )[:, 1]
        values.append(roc_auc_score(y, probability))
    return float(np.mean(values))


def shift_metrics(train_x, train_y, validation_x, validation_y, z_train, z_val, seed):
    denominator = max(
        math.sqrt(np.mean(np.sum((z_train - z_train.mean(0)) ** 2, axis=1))),
        1e-12,
    )
    cov_train = np.atleast_2d(np.cov(z_train, rowvar=False))
    cov_val = np.atleast_2d(np.cov(z_val, rowvar=False))
    coral_denominator = max(
        0.5 * (np.linalg.norm(cov_train, "fro") + np.linalg.norm(cov_val, "fro")),
        1e-12,
    )
    return {
        "domain_probe_auc": disease_controlled_domain_auc(
            train_x, train_y, validation_x, validation_y, seed
        ),
        "normalized_mean_shift": float(
            np.linalg.norm(z_train.mean(0) - z_val.mean(0)) / denominator
        ),
        "coral_covariance_shift": float(
            np.linalg.norm(cov_train - cov_val, "fro") / coral_denominator
        ),
        "pd_centroid_drift": float(np.linalg.norm(
            z_train[train_y == 0].mean(0) - z_val[validation_y == 0].mean(0)
        )),
        "dd_centroid_drift": float(np.linalg.norm(
            z_train[train_y == 1].mean(0) - z_val[validation_y == 1].mean(0)
        )),
    }


def token_identity(tokens_train, tokens_validation, validation_y):
    n_train, activities, dimension = tokens_train.shape
    n_validation = len(tokens_validation)
    labels_train = np.tile(np.arange(activities), n_train)
    labels_validation = np.tile(np.arange(activities), n_validation)
    scaler = StandardScaler().fit(tokens_train.reshape(-1, dimension))
    train_z = scaler.transform(tokens_train.reshape(-1, dimension)).reshape(
        n_train, activities, dimension
    )
    validation_z = scaler.transform(
        tokens_validation.reshape(-1, dimension)
    ).reshape(n_validation, activities, dimension)
    model = LogisticRegression(
        max_iter=2500, class_weight="balanced", solver="lbfgs", C=1.0
    ).fit(train_z.reshape(-1, dimension), labels_train)
    activity_accuracy = accuracy_score(
        labels_validation, model.predict(validation_z.reshape(-1, dimension))
    )
    activity_centroid = train_z.mean(0)
    residual = validation_z - activity_centroid[None]
    residual /= np.clip(np.linalg.norm(residual, axis=-1, keepdims=True), 1e-12, None)
    ranks, self_similarity, other_similarity = [], [], []
    for activity in range(activities):
        other = [index for index in range(activities) if index != activity]
        candidates = residual[:, other].mean(1)
        candidates /= np.clip(np.linalg.norm(candidates, axis=1, keepdims=True), 1e-12, None)
        similarities = residual[:, activity] @ candidates.T
        order = np.argsort(-similarities, axis=1)
        for subject in range(n_validation):
            ranks.append(int(np.where(order[subject] == subject)[0][0]) + 1)
            self_similarity.append(similarities[subject, subject])
            same = np.where(
                (validation_y == validation_y[subject])
                & (np.arange(n_validation) != subject)
            )[0]
            other_similarity.append(similarities[subject, same].mean())
    ranks = np.asarray(ranks)
    return {
        "activity_probe_accuracy": float(activity_accuracy),
        "subject_retrieval_top1": float(np.mean(ranks == 1)),
        "subject_retrieval_top5": float(np.mean(ranks <= 5)),
        "subject_retrieval_mrr": float(np.mean(1.0 / ranks)),
        "same_subject_similarity": float(np.mean(self_similarity)),
        "same_disease_other_subject_similarity": float(np.mean(other_similarity)),
        "subject_similarity_gap": float(
            np.mean(np.asarray(self_similarity) - np.asarray(other_similarity))
        ),
    }


def geometry_arrays(train_x, train_y, validation_x, validation_y):
    scaler = StandardScaler().fit(train_x)
    train_z, validation_z = scaler.transform(train_x), scaler.transform(validation_x)
    centroids = np.stack([train_z[train_y == label].mean(0) for label in (0, 1)])
    distances = np.linalg.norm(validation_z[:, None] - centroids[None], axis=-1)
    pairwise = np.linalg.norm(validation_z[:, None] - train_z[None], axis=-1)
    nearest = np.argsort(pairwise, axis=1)
    result = {
        "distance_pd_centroid": distances[:, 0],
        "distance_dd_centroid": distances[:, 1],
        "dd_like_centroid_margin": distances[:, 0] - distances[:, 1],
    }
    for k in (5, 10):
        labels = train_y[nearest[:, :k]]
        result[f"knn{k}_dd_fraction"] = labels.mean(1)
        result[f"knn{k}_same_label_fraction"] = (
            labels == validation_y[:, None]
        ).mean(1)
    return result


def representation_diagnostics(files, output):
    representation_rows, token_rows, head_rows, geometry_rows = [], [], [], []
    pattern = re.compile(r"seed(\d+)_outer(\d+)_inner(\d+)\.npz")
    for number, path in enumerate(files, 1):
        seed, outer, inner = map(int, pattern.fullmatch(path.name).groups())
        payload = np.load(path)
        train_y = payload["v8_train_label"]
        validation_y = payload["v8_validation_label"]
        representations = {
            "v8_subject": (
                payload["v8_train_subject_embedding"],
                payload["v8_validation_subject_embedding"],
            ),
            "str_subject": (
                payload["str_train_subject_embedding"],
                payload["str_validation_subject_embedding"],
            ),
            "str_structured": (
                payload["str_train_structured_embedding"],
                payload["str_validation_structured_embedding"],
            ),
            "str_combined": (
                payload["str_train_decision_input"],
                payload["str_validation_decision_input"],
            ),
        }
        for name, (train_x, validation_x) in representations.items():
            metrics, z_train, z_val = disease_probe(
                train_x, train_y, validation_x, validation_y
            )
            metrics.update(shift_metrics(
                train_x, train_y, validation_x, validation_y,
                z_train, z_val, seed,
            ))
            representation_rows.append({
                "seed": seed, "outer": outer, "inner": inner,
                "representation": name, "dimension": train_x.shape[1], **metrics,
            })
            geometry = geometry_arrays(
                train_x, train_y, validation_x, validation_y
            )
            ids = payload["v8_validation_subject_id"].astype(str)
            for index, subject_id in enumerate(ids):
                geometry_rows.append({
                    "seed": seed, "outer": outer, "inner": inner,
                    "subject_id": subject_id.zfill(3),
                    "label": int(validation_y[index]), "representation": name,
                    **{key: float(value[index]) for key, value in geometry.items()},
                })
        token_representations = {
            "v8_activity_context": (
                payload["v8_train_activity_embeddings"],
                payload["v8_validation_activity_embeddings"],
            ),
            "str_activity_context": (
                payload["str_train_activity_embeddings"],
                payload["str_validation_activity_embeddings"],
            ),
            "str_structured_tokens": (
                payload["str_train_structured_tokens"],
                payload["str_validation_structured_tokens"],
            ),
            "str_combined_activity": (
                np.concatenate([
                    payload["str_train_activity_embeddings"],
                    payload["str_train_structured_tokens"],
                ], axis=2),
                np.concatenate([
                    payload["str_validation_activity_embeddings"],
                    payload["str_validation_structured_tokens"],
                ], axis=2),
            ),
        }
        for name, (train_tokens, validation_tokens) in token_representations.items():
            token_rows.append({
                "seed": seed, "outer": outer, "inner": inner,
                "representation": name, "dimension": train_tokens.shape[2],
                **token_identity(train_tokens, validation_tokens, validation_y),
            })
        for name, logits in {
            "v8_head": payload["v8_validation_final_logits"],
            "str_base_head": payload["str_validation_base_logits"],
            "str_residual_head": payload["str_validation_residual_logits"],
            "str_final_head": payload["str_validation_final_logits"],
        }.items():
            head_rows.append({
                "seed": seed, "outer": outer, "inner": inner, "head": name,
                **classification_metrics(validation_y, logits[:, 1] - logits[:, 0]),
            })
        print(f"representation diagnostics {number}/{len(files)}", flush=True)

    raw = pd.DataFrame(representation_rows)
    raw.to_csv(output / "representation_metrics_45_seed_folds.csv", index=False)
    metrics = [column for column in raw if column not in {
        "seed", "outer", "inner", "representation", "dimension"
    }]
    splits = raw.groupby(
        ["outer", "inner", "representation"], as_index=False
    )[metrics].mean()
    splits.to_csv(output / "representation_metrics_15_splits.csv", index=False)
    splits.groupby("representation")[metrics].agg(
        ["mean", "std"]
    ).to_csv(output / "representation_metrics_summary.csv")

    token_raw = pd.DataFrame(token_rows)
    token_raw.to_csv(output / "token_metrics_45_seed_folds.csv", index=False)
    token_metrics = [column for column in token_raw if column not in {
        "seed", "outer", "inner", "representation", "dimension"
    }]
    token_splits = token_raw.groupby(
        ["outer", "inner", "representation"], as_index=False
    )[token_metrics].mean()
    token_splits.to_csv(output / "token_metrics_15_splits.csv", index=False)
    token_splits.groupby("representation")[token_metrics].agg(
        ["mean", "std"]
    ).to_csv(output / "token_metrics_summary.csv")

    heads = pd.DataFrame(head_rows)
    heads.to_csv(output / "head_metrics_45_seed_folds.csv", index=False)
    head_metrics = [column for column in heads if column not in {
        "seed", "outer", "inner", "head"
    }]
    head_splits = heads.groupby(
        ["outer", "inner", "head"], as_index=False
    )[head_metrics].mean()
    head_splits.to_csv(output / "head_metrics_15_splits.csv", index=False)

    geometry = pd.DataFrame(geometry_rows)
    geometry.to_csv(output / "geometry_45_seed_folds.csv", index=False)
    geometry.groupby(
        ["outer", "inner", "subject_id", "label", "representation"],
        as_index=False,
    ).mean(numeric_only=True).to_csv(output / "geometry_15_splits.csv", index=False)
    return splits, token_splits, head_splits


def paired_representation_inference(splits, token_splits, output):
    rows = []
    metrics = [
        "disease_probe_ba", "disease_probe_auroc", "disease_centroid_ba",
        "disease_silhouette_euclidean", "disease_silhouette_cosine",
        "domain_probe_auc", "normalized_mean_shift", "coral_covariance_shift",
        "pd_centroid_drift", "dd_centroid_drift",
    ]
    indexed = splits.set_index(["outer", "inner", "representation"])
    units = sorted({(int(row.outer), int(row.inner)) for row in splits.itertuples()})
    for comparison, model, baseline in (
        ("str_subject_vs_v8", "str_subject", "v8_subject"),
        ("str_combined_vs_v8", "str_combined", "v8_subject"),
    ):
        for metric in metrics:
            delta = np.asarray([
                indexed.loc[(outer, inner, model), metric]
                - indexed.loc[(outer, inner, baseline), metric]
                for outer, inner in units
            ])
            rows.append({
                "comparison": comparison, "metric": metric,
                "model_mean": float(np.mean([
                    indexed.loc[(outer, inner, model), metric]
                    for outer, inner in units
                ])),
                "baseline_mean": float(np.mean([
                    indexed.loc[(outer, inner, baseline), metric]
                    for outer, inner in units
                ])),
                "mean_delta": float(delta.mean()),
                "wins": int((delta > 0).sum()), "losses": int((delta < 0).sum()),
                "wilcoxon_p": paired_p(delta),
                "rank_biserial": rank_biserial(delta),
            })
    result = pd.DataFrame(rows)
    result["bh_q_across_20_tests"] = bh(result.wilcoxon_p)
    result.to_csv(output / "representation_paired_inference.csv", index=False)

    token_rows = []
    token_metrics = [
        "activity_probe_accuracy", "subject_retrieval_top1",
        "subject_retrieval_top5", "subject_retrieval_mrr",
        "subject_similarity_gap",
    ]
    indexed = token_splits.set_index(["outer", "inner", "representation"])
    for comparison, model, baseline in (
        ("str_activity_vs_v8", "str_activity_context", "v8_activity_context"),
        ("str_combined_activity_vs_v8", "str_combined_activity", "v8_activity_context"),
    ):
        for metric in token_metrics:
            delta = np.asarray([
                indexed.loc[(outer, inner, model), metric]
                - indexed.loc[(outer, inner, baseline), metric]
                for outer, inner in units
            ])
            token_rows.append({
                "comparison": comparison, "metric": metric,
                "mean_delta": delta.mean(), "wins": int((delta > 0).sum()),
                "losses": int((delta < 0).sum()),
                "wilcoxon_p": paired_p(delta),
                "rank_biserial": rank_biserial(delta),
            })
    token_result = pd.DataFrame(token_rows)
    token_result["bh_q_across_10_tests"] = bh(token_result.wilcoxon_p)
    token_result.to_csv(output / "token_paired_inference.csv", index=False)


def build_validation_records(files, output):
    pattern = re.compile(r"seed(\d+)_outer(\d+)_inner(\d+)\.npz")
    by_split = {}
    for path in files:
        seed, outer, inner = map(int, pattern.fullmatch(path.name).groups())
        by_split.setdefault((outer, inner), {})[seed] = np.load(path)
    records, activity_split_rows = [], []
    for (outer, inner), payloads in sorted(by_split.items()):
        reference = payloads[42]
        ids = reference["v8_validation_subject_id"].astype(str)
        y = reference["v8_validation_label"].astype(int)
        for seed, payload in payloads.items():
            if not np.array_equal(ids, payload["v8_validation_subject_id"].astype(str)):
                raise ValueError(f"Subject mismatch in {outer}/{inner}/{seed}")
        def mean_array(key):
            return np.mean([payload[key] for payload in payloads.values()], axis=0)
        v8_margin = mean_array("v8_validation_final_logits")
        v8_margin = v8_margin[:, 1] - v8_margin[:, 0]
        base_margin = mean_array("str_validation_base_logits")
        base_margin = base_margin[:, 1] - base_margin[:, 0]
        residual_margin = mean_array("str_validation_residual_logits")
        residual_margin = residual_margin[:, 1] - residual_margin[:, 0]
        final_margin = mean_array("str_validation_final_logits")
        final_margin = final_margin[:, 1] - final_margin[:, 0]
        def mean_probability(key):
            probabilities = []
            for payload in payloads.values():
                logits = payload[key]
                probabilities.append(expit(logits[:, 1] - logits[:, 0]))
            return np.mean(probabilities, axis=0)
        v8_probability = mean_probability("v8_validation_final_logits")
        base_probability = mean_probability("str_validation_base_logits")
        residual_probability = mean_probability("str_validation_residual_logits")
        final_probability = mean_probability("str_validation_final_logits")
        activity_arrays = {
            "v8": mean_array("v8_validation_base_activity_margin"),
            "str_base": mean_array("str_validation_base_activity_margin"),
            "residual": mean_array("str_validation_residual_activity_margin"),
            "str_final": mean_array("str_validation_final_activity_margin"),
        }
        for index, subject_id in enumerate(ids):
            target = int(y[index])
            direction = 1.0 if target == 1 else -1.0
            v8_correct = bool((v8_probability[index] >= 0.5) == target)
            str_correct = bool((final_probability[index] >= 0.5) == target)
            base_correct = bool((base_probability[index] >= 0.5) == target)
            residual_correct = bool((residual_probability[index] >= 0.5) == target)
            if v8_correct and str_correct:
                group = "both_correct"
            elif (not v8_correct) and str_correct:
                group = "v8_wrong_str_correct"
            elif v8_correct and (not str_correct):
                group = "v8_correct_str_wrong"
            else:
                group = "both_wrong"
            residual_true = direction * residual_margin[index]
            if (not base_correct) and str_correct:
                internal = "residual_correction"
            elif base_correct and (not str_correct):
                internal = "residual_harm"
            elif base_correct and str_correct:
                internal = "enhance_correct" if residual_true > 0 else "weaken_correct"
            else:
                internal = "toward_correction" if residual_true > 0 else "reinforce_error"
            record = {
                "outer": outer, "inner": inner, "subject_id": subject_id.zfill(3),
                "target": target, "label_name": "DD" if target else "PD",
                "v8_margin_dd": float(v8_margin[index]),
                "str_base_margin_dd": float(base_margin[index]),
                "str_residual_margin_dd": float(residual_margin[index]),
                "str_final_margin_dd": float(final_margin[index]),
                "v8_probability_dd": float(v8_probability[index]),
                "str_base_probability_dd": float(base_probability[index]),
                "str_residual_probability_dd": float(residual_probability[index]),
                "str_final_probability_dd": float(final_probability[index]),
                "v8_correct": v8_correct, "str_base_correct": base_correct,
                "str_residual_correct": residual_correct, "str_correct": str_correct,
                "correctness_group": group, "str_internal_pattern": internal,
                "v8_true_margin": direction * float(v8_margin[index]),
                "str_base_true_margin": direction * float(base_margin[index]),
                "str_residual_true_margin": residual_true,
                "str_final_true_margin": direction * float(final_margin[index]),
                "base_adaptation_true_margin": direction * float(
                    base_margin[index] - v8_margin[index]
                ),
                "total_gain_true_margin": direction * float(
                    final_margin[index] - v8_margin[index]
                ),
            }
            residual_true_by_activity = direction * activity_arrays["residual"][index]
            top_activity = int(np.argmax(residual_true_by_activity))
            for activity_index, activity in enumerate(ACTIVITIES):
                for path_name, values in activity_arrays.items():
                    record[f"{path_name}_activity_true_margin::{activity}"] = (
                        direction * float(values[index, activity_index])
                    )
                record[f"residual_positive::{activity}"] = bool(
                    residual_true_by_activity[activity_index] > 0
                )
                record[f"residual_top::{activity}"] = bool(activity_index == top_activity)
                counterfactual_probability = np.mean([
                    expit(
                        (payload["str_validation_final_logits"][index, 1]
                         - payload["str_validation_final_logits"][index, 0])
                        - payload["str_validation_residual_activity_margin"][
                            index, activity_index
                        ]
                    ) for payload in payloads.values()
                ])
                counterfactual_correct = bool(
                    (counterfactual_probability >= 0.5) == target
                )
                record[f"residual_pivotal::{activity}"] = bool(
                    str_correct and not counterfactual_correct
                )
            records.append(record)
        for activity_index, activity in enumerate(ACTIVITIES):
            row = {"outer": outer, "inner": inner, "activity": activity}
            for path_name, values in activity_arrays.items():
                row[f"{path_name}_activity_auroc"] = safe_auc(
                    y, values[:, activity_index]
                )
            activity_split_rows.append(row)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "validation_records_15_splits.csv", index=False)
    pd.DataFrame(activity_split_rows).to_csv(
        output / "activity_evidence_15_splits.csv", index=False
    )
    return frame


def seedfold_stability(files):
    rows = []
    for path in files:
        payload = np.load(path)
        ids = payload["v8_validation_subject_id"].astype(str)
        labels = payload["v8_validation_label"].astype(int)
        for model, key in (
            ("v8", "v8_validation_final_logits"),
            ("str", "str_validation_final_logits"),
        ):
            logits = payload[key]
            prediction = (logits[:, 1] - logits[:, 0] >= 0).astype(int)
            for subject_id, label, predicted in zip(ids, labels, prediction):
                rows.append({
                    "subject_id": subject_id.zfill(3), "model": model,
                    "correct": bool(predicted == label),
                })
    frame = pd.DataFrame(rows)
    result = []
    for (subject_id, model), group in frame.groupby(["subject_id", "model"]):
        error_rate = 1.0 - group.correct.mean()
        if error_rate <= 0.25:
            status = "stable_correct"
        elif error_rate >= 0.75:
            status = "stable_error"
        else:
            status = "unstable"
        result.append({
            "subject_id": subject_id, "model": model,
            "seedfold_appearances": len(group), "seedfold_stability": status,
            "seedfold_error_rate": error_rate,
        })
    wide = pd.DataFrame(result).pivot(
        index="subject_id", columns="model",
        values=["seedfold_appearances", "seedfold_stability", "seedfold_error_rate"],
    )
    wide.columns = [f"{model}_{metric}" for metric, model in wide.columns]
    return wide.reset_index()


def add_stability_and_h1(frame, h1_path, output, files):
    def stability(correct):
        error_rate = 1.0 - np.mean(correct)
        if error_rate <= 0.25:
            return "stable_correct"
        if error_rate >= 0.75:
            return "stable_error"
        return "unstable"
    subject_rows = []
    for subject_id, group in frame.groupby("subject_id"):
        counts = group.correctness_group.value_counts()
        top = counts.index[0]
        persistent = top if int(counts.iloc[0]) >= 3 else "mixed"
        subject_rows.append({
            "subject_id": subject_id, "label_name": group.label_name.iloc[0],
            "appearances": len(group),
            "v8_stability": stability(group.v8_correct.to_numpy(bool)),
            "str_stability": stability(group.str_correct.to_numpy(bool)),
            "v8_error_rate": 1.0 - group.v8_correct.mean(),
            "str_error_rate": 1.0 - group.str_correct.mean(),
            "persistent_category": persistent,
            **{f"count_{name}": int(counts.get(name, 0)) for name in (
                "both_correct", "v8_wrong_str_correct", "v8_correct_str_wrong",
                "both_wrong",
            )},
        })
    subjects = pd.DataFrame(subject_rows)
    subjects = subjects.merge(
        seedfold_stability(files), on="subject_id", how="left", validate="one_to_one"
    )
    subjects.to_csv(output / "subject_stability_transitions.csv", index=False)
    h1 = pd.read_csv(h1_path)
    h1["subject_id"] = h1.subject_id.astype(str).str.zfill(3)
    h1 = h1.drop(columns=[
        column for column in ("v8_probability_dd", "v8_correct", "correctness_group",
                              "stability_group") if column in h1
    ])
    merged = frame.merge(h1, on=["outer", "inner", "subject_id", "target", "label_name"],
                         how="left", validate="one_to_one")
    if merged.h1_probability_dd.isna().any():
        raise ValueError("H1 record merge produced missing rows")
    merged = merged.merge(
        subjects[["subject_id", "v8_stability", "str_stability", "persistent_category"]],
        on="subject_id", how="left", validate="many_to_one",
    )
    merged.to_csv(output / "validation_records_with_h1.csv", index=False)
    return merged, subjects


def group_and_path_analysis(frame, subjects, output):
    group_counts = frame.groupby(
        ["correctness_group", "label_name", "v8_stability", "str_stability"],
        as_index=False,
    ).agg(rows=("subject_id", "size"), subjects=("subject_id", "nunique"))
    group_counts.to_csv(output / "correctness_group_counts.csv", index=False)
    margins = frame.groupby(
        ["correctness_group", "label_name"], as_index=False
    ).agg(
        rows=("subject_id", "size"), subjects=("subject_id", "nunique"),
        v8_true_margin=("v8_true_margin", "mean"),
        str_base_true_margin=("str_base_true_margin", "mean"),
        residual_true_margin=("str_residual_true_margin", "mean"),
        final_true_margin=("str_final_true_margin", "mean"),
        base_adaptation=("base_adaptation_true_margin", "mean"),
        total_gain=("total_gain_true_margin", "mean"),
        h1_correct_rate=("h1_correct", "mean"),
    )
    margins.to_csv(output / "decision_margin_by_group.csv", index=False)
    frame.groupby(
        ["correctness_group", "str_internal_pattern", "label_name"], as_index=False
    ).size().rename(columns={"size": "rows"}).to_csv(
        output / "internal_decision_patterns.csv", index=False
    )
    subjects.groupby(
        ["v8_stability", "str_stability", "label_name"], as_index=False
    ).size().rename(columns={"size": "subjects"}).to_csv(
        output / "stable_error_transition_counts.csv", index=False
    )
    subjects.groupby(
        ["v8_seedfold_stability", "str_seedfold_stability", "label_name"],
        as_index=False,
    ).size().rename(columns={"size": "subjects"}).to_csv(
        output / "stable_error_transition_counts_12_seedfold_appearances.csv",
        index=False,
    )

    split_rows = []
    for (outer, inner), group in frame.groupby(["outer", "inner"]):
        v8 = classification_metrics(
            group.target.to_numpy(), group.v8_probability_dd.to_numpy() - 0.5
        )
        stru = classification_metrics(
            group.target.to_numpy(), group.str_final_probability_dd.to_numpy() - 0.5
        )
        split_rows.append({
            "outer": outer, "inner": inner,
            "v8_ba": v8["balanced_accuracy"], "str_ba": stru["balanced_accuracy"],
            "delta_ba": stru["balanced_accuracy"] - v8["balanced_accuracy"],
            "v8_auroc": v8["auroc"], "str_auroc": stru["auroc"],
            "delta_auroc": stru["auroc"] - v8["auroc"],
            "rescue_rate": float(np.mean(
                group.correctness_group == "v8_wrong_str_correct"
            )),
            "harm_rate": float(np.mean(
                group.correctness_group == "v8_correct_str_wrong"
            )),
            "net_rescue_rate": float(np.mean(
                group.correctness_group == "v8_wrong_str_correct"
            ) - np.mean(group.correctness_group == "v8_correct_str_wrong")),
            "residual_correction_rate": float(np.mean(
                group.str_internal_pattern == "residual_correction"
            )),
            "base_residual_prediction_disagreement": float(np.mean(
                group.str_base_correct != group.str_residual_correct
            )),
            "base_residual_margin_spearman": float(spearmanr(
                group.str_base_margin_dd, group.str_residual_margin_dd
            ).statistic),
        })
    splits = pd.DataFrame(split_rows)
    splits.to_csv(output / "split_gain_and_complementarity.csv", index=False)
    relationships = []
    for mechanism in (
        "rescue_rate", "harm_rate", "net_rescue_rate", "residual_correction_rate",
        "base_residual_prediction_disagreement", "base_residual_margin_spearman",
    ):
        for metric in ("delta_ba", "delta_auroc"):
            rho, p = spearmanr(splits[mechanism], splits[metric])
            relationships.append({
                "mechanism": mechanism, "classification_metric": metric,
                "spearman_rho": rho, "p_value": p,
            })
    relationships = pd.DataFrame(relationships)
    relationships["bh_q_across_12_tests"] = bh(relationships.p_value)
    relationships.to_csv(
        output / "complementarity_gain_relationships.csv", index=False
    )
    return splits


def activity_analysis(frame, output):
    profile_rows, split_rows = [], []
    for (group_name, label), group in frame.groupby(["correctness_group", "label_name"]):
        for activity in ACTIVITIES:
            profile_rows.append({
                "correctness_group": group_name, "label_name": label,
                "activity": activity, "rows": len(group),
                "v8_true_margin_mean": group[f"v8_activity_true_margin::{activity}"].mean(),
                "str_base_true_margin_mean": group[f"str_base_activity_true_margin::{activity}"].mean(),
                "residual_true_margin_mean": group[f"residual_activity_true_margin::{activity}"].mean(),
                "str_final_true_margin_mean": group[f"str_final_activity_true_margin::{activity}"].mean(),
                "residual_positive_fraction": group[f"residual_positive::{activity}"].mean(),
                "residual_top_fraction": group[f"residual_top::{activity}"].mean(),
                "residual_pivotal_fraction": group[f"residual_pivotal::{activity}"].mean(),
            })
    pd.DataFrame(profile_rows).to_csv(output / "activity_margin_profiles.csv", index=False)
    for (outer, inner), split in frame.groupby(["outer", "inner"]):
        rescue = split[split.correctness_group == "v8_wrong_str_correct"]
        harm = split[split.correctness_group == "v8_correct_str_wrong"]
        for activity in ACTIVITIES:
            split_rows.append({
                "outer": outer, "inner": inner, "activity": activity,
                "rescue_n": len(rescue), "harm_n": len(harm),
                "rescue_residual_true_margin": rescue[
                    f"residual_activity_true_margin::{activity}"
                ].mean(),
                "harm_residual_true_margin": harm[
                    f"residual_activity_true_margin::{activity}"
                ].mean(),
                "rescue_positive_fraction": rescue[
                    f"residual_positive::{activity}"
                ].mean(),
                "rescue_top_fraction": rescue[f"residual_top::{activity}"].mean(),
                "rescue_pivotal_fraction": rescue[
                    f"residual_pivotal::{activity}"
                ].mean(),
            })
    split_frame = pd.DataFrame(split_rows)
    split_frame.to_csv(output / "activity_rescue_participation_15_splits.csv", index=False)
    inference = []
    evidence = pd.read_csv(output / "activity_evidence_15_splits.csv")
    for activity in ACTIVITIES:
        rows = split_frame[split_frame.activity == activity]
        delta = (
            rows.rescue_residual_true_margin - rows.harm_residual_true_margin
        ).dropna().to_numpy()
        evidence_rows = evidence[evidence.activity == activity]
        auc_delta = (
            evidence_rows.str_final_activity_auroc
            - evidence_rows.v8_activity_auroc
        ).to_numpy()
        inference.extend([
            {
                "analysis": "rescue_vs_harm_residual_margin",
                "activity": activity, "mean_delta": float(delta.mean()),
                "wins": int((delta > 0).sum()), "losses": int((delta < 0).sum()),
                "wilcoxon_p": paired_p(delta), "rank_biserial": rank_biserial(delta),
            },
            {
                "analysis": "str_vs_v8_activity_auroc",
                "activity": activity, "mean_delta": float(auc_delta.mean()),
                "wins": int((auc_delta > 0).sum()),
                "losses": int((auc_delta < 0).sum()),
                "wilcoxon_p": paired_p(auc_delta),
                "rank_biserial": rank_biserial(auc_delta),
            },
        ])
    inference = pd.DataFrame(inference)
    inference["bh_q_across_22_tests"] = bh(inference.wilcoxon_p)
    inference.to_csv(output / "activity_paired_inference.csv", index=False)
    summary = split_frame.groupby("activity", as_index=False).agg(
        rescue_residual_true_margin=("rescue_residual_true_margin", "mean"),
        harm_residual_true_margin=("harm_residual_true_margin", "mean"),
        rescue_positive_fraction=("rescue_positive_fraction", "mean"),
        rescue_top_fraction=("rescue_top_fraction", "mean"),
        rescue_pivotal_fraction=("rescue_pivotal_fraction", "mean"),
    ).merge(
        evidence.groupby("activity", as_index=False).agg(
            v8_activity_auroc=("v8_activity_auroc", "mean"),
            residual_activity_auroc=("residual_activity_auroc", "mean"),
            str_final_activity_auroc=("str_final_activity_auroc", "mean"),
        ), on="activity",
    )
    summary.to_csv(output / "activity_summary.csv", index=False)


def subject_level_external_profiles(frame, subjects, stable_dir, output):
    metadata = pd.read_csv(stable_dir / "subject_metadata_and_stability.csv")
    metadata["subject_id"] = metadata.subject_id.astype(str).str.zfill(3)
    metadata = metadata.rename(columns={"stability_group": "frozen_v8_stability"})
    subject_full = subjects.merge(metadata.drop(columns=[
        column for column in ("label", "label_name")
        if column in metadata
    ]), on="subject_id", how="left")
    mismatch = subject_full[
        subject_full.v8_seedfold_stability != subject_full.frozen_v8_stability
    ]
    if len(mismatch):
        raise ValueError(
            f"V8 stability does not match frozen audit for {len(mismatch)} subjects"
        )
    subject_full.to_csv(output / "subject_profiles.csv", index=False)
    subject_full.groupby(
        ["persistent_category", "label_name", "condition"], as_index=False
    ).size().rename(columns={"size": "subjects"}).to_csv(
        output / "condition_by_persistent_group.csv", index=False
    )

    geometry = pd.read_csv(output / "geometry_15_splits.csv", dtype={"subject_id": str})
    geometry["subject_id"] = geometry.subject_id.str.zfill(3)
    geometry = geometry.merge(
        subjects[["subject_id", "persistent_category", "label_name"]],
        on="subject_id", how="left",
    )
    geometry.groupby(
        ["representation", "persistent_category", "label_name"], as_index=False
    ).mean(numeric_only=True).to_csv(output / "geometry_by_persistent_group.csv", index=False)

    activity = pd.read_csv(stable_dir / "activity_pd_like_long.csv")
    activity["subject_id"] = activity.subject_id.astype(str).str.zfill(3)
    activity = activity.merge(
        subjects[["subject_id", "persistent_category"]], on="subject_id", how="left"
    )
    activity.groupby(
        ["persistent_category", "label_name", "activity"], as_index=False
    ).agg(
        subjects=("subject_id", "nunique"),
        pd_like_score_mean=("pd_like_score", "mean"),
        pd_like_score_sd=("pd_like_score", "std"),
    ).to_csv(output / "raw_activity_phenotype_by_group.csv", index=False)

    signals = pd.read_csv(stable_dir / "high_risk_activity_signal_features.csv")
    signals["subject_id"] = signals.subject_id.astype(str).str.zfill(3)
    signals = signals.merge(
        subjects[["subject_id", "persistent_category"]], on="subject_id", how="left"
    )
    signals.groupby(
        ["persistent_category", "label_name", "activity", "sensor", "scope"],
        as_index=False,
    )[list(SIGNAL_FEATURES)].mean().to_csv(
        output / "raw_signal_profiles_by_group.csv", index=False
    )
    comparisons = []
    for keys, group in signals.groupby(["label_name", "activity", "sensor", "scope"]):
        rescue = group[group.persistent_category == "v8_wrong_str_correct"]
        overlap = group[group.persistent_category == "both_wrong"]
        for feature in SIGNAL_FEATURES:
            left, right = rescue[feature].dropna().to_numpy(), overlap[feature].dropna().to_numpy()
            if len(left) >= 3 and len(right) >= 3:
                comparisons.append({
                    "label_name": keys[0], "activity": keys[1], "sensor": keys[2],
                    "scope": keys[3], "feature": feature,
                    "rescue_n": len(left), "both_wrong_n": len(right),
                    "rescue_mean": left.mean(), "both_wrong_mean": right.mean(),
                    "cliffs_delta": cliffs_delta(left, right),
                    "mannwhitney_p": mannwhitneyu(
                        left, right, alternative="two-sided"
                    ).pvalue,
                })
    comparisons = pd.DataFrame(comparisons)
    if len(comparisons):
        comparisons["bh_q"] = bh(comparisons.mannwhitney_p)
    comparisons.to_csv(output / "raw_signal_rescue_vs_both_wrong_bh.csv", index=False)

    h1_columns = [column for column in frame if column.startswith(
        "h1_true_direction_contribution_"
    )]
    h1_profile = frame.groupby(
        ["correctness_group", "label_name"], as_index=False
    ).agg(
        rows=("subject_id", "size"), h1_correct_rate=("h1_correct", "mean"),
        h1_probability_dd=("h1_probability_dd", "mean"),
        **{column: (column, "mean") for column in h1_columns},
    )
    h1_profile.to_csv(output / "h1_profile_by_correctness_group.csv", index=False)
    h1_subject = frame.groupby(
        ["subject_id", "label_name", "persistent_category"], as_index=False
    )[h1_columns].mean()
    h1_comparisons = []
    for label, group in h1_subject.groupby("label_name"):
        rescue = group[group.persistent_category == "v8_wrong_str_correct"]
        overlap = group[group.persistent_category == "both_wrong"]
        for column in h1_columns:
            left, right = rescue[column].dropna().to_numpy(), overlap[column].dropna().to_numpy()
            if len(left) >= 3 and len(right) >= 3:
                h1_comparisons.append({
                    "label_name": label, "feature": column,
                    "rescue_mean": left.mean(), "both_wrong_mean": right.mean(),
                    "cliffs_delta": cliffs_delta(left, right),
                    "mannwhitney_p": mannwhitneyu(
                        left, right, alternative="two-sided"
                    ).pvalue,
                })
    h1_comparisons = pd.DataFrame(h1_comparisons)
    h1_comparisons["bh_q"] = bh(h1_comparisons.mannwhitney_p)
    h1_comparisons.to_csv(output / "h1_rescue_vs_both_wrong_bh.csv", index=False)

    geometry_subject = geometry.groupby(
        ["subject_id", "label_name", "persistent_category", "representation"],
        as_index=False,
    ).mean(numeric_only=True)
    geometry_comparisons = []
    metrics = [
        "dd_like_centroid_margin", "knn5_dd_fraction",
        "knn5_same_label_fraction", "knn10_same_label_fraction",
    ]
    for (label, representation), group in geometry_subject.groupby(
        ["label_name", "representation"]
    ):
        rescue = group[group.persistent_category == "v8_wrong_str_correct"]
        overlap = group[group.persistent_category == "both_wrong"]
        for metric in metrics:
            left, right = rescue[metric].dropna().to_numpy(), overlap[metric].dropna().to_numpy()
            if len(left) >= 3 and len(right) >= 3:
                geometry_comparisons.append({
                    "label_name": label, "representation": representation,
                    "metric": metric, "rescue_n": len(left), "both_wrong_n": len(right),
                    "rescue_mean": left.mean(), "both_wrong_mean": right.mean(),
                    "cliffs_delta": cliffs_delta(left, right),
                    "mannwhitney_p": mannwhitneyu(
                        left, right, alternative="two-sided"
                    ).pvalue,
                })
    geometry_comparisons = pd.DataFrame(geometry_comparisons)
    if len(geometry_comparisons):
        geometry_comparisons["bh_q"] = bh(geometry_comparisons.mannwhitney_p)
    geometry_comparisons.to_csv(
        output / "geometry_rescue_vs_both_wrong_bh.csv", index=False
    )


def main():
    args = arguments()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(args.embedding_dir.glob("seed*_outer*_inner*.npz"))
    if len(files) != 45:
        raise ValueError(f"Expected 45 paired archives, found {len(files)}")
    representation_splits, token_splits, _ = representation_diagnostics(files, output)
    paired_representation_inference(representation_splits, token_splits, output)
    records = build_validation_records(files, output)
    records, subjects = add_stability_and_h1(
        records, args.h1_records, output, files
    )
    group_and_path_analysis(records, subjects, output)
    activity_analysis(records, output)
    subject_level_external_profiles(records, subjects, args.stable_audit_dir, output)
    protocol = {
        "scope": "fixed inner-development only",
        "outer_information_used": False,
        "models_trained": False,
        "models_modified": False,
        "seeds": [42, 43, 44],
        "seed_handling": "seed margins/representations aggregated within matched split before subject decision-pattern analysis; representation metrics averaged within split before inference",
        "independent_units": "15 fixed inner splits for matched representation/activity inference; unique subjects for persistent phenotype comparisons",
        "activity_margin": "exact additive DD-minus-PD logit contribution with path bias excluded",
        "multiple_comparison": "BH-FDR within prespecified analysis families",
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
