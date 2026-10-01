#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, wilcoxon

from src.datasets.builders import make_loader
from src.models import build_model
from src.utils.config import load_config

from run_layerwise_representation_diagnosis import (
    disease_probe,
    encode_layers,
    head_metrics,
    load_fold_datasets,
    representation_shift,
    token_identity_activity,
    variance_decomposition,
)


COMPONENTS = ("left", "right", "mean", "absdiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen V8-GN bilateral fusion component diagnosis")
    parser.add_argument("--run", action="append", required=True, help="SEED:/absolute/run/path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-folds", type=int, default=0)
    return parser.parse_args()


@torch.inference_mode()
def extract_components(model, dataset, config, device):
    loader = make_loader(
        dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
        pin_memory=False,
        seed=int(config["experiment"]["seed"]) + 1901,
    )
    wrist_embeddings, wrist_masks, activity_masks = [], [], []
    labels, logits, subject_ids = [], [], []
    max_equivalence = {"activity": 0.0, "subject": 0.0, "logits": 0.0}
    checked = False
    model.eval()
    for batch in loader:
        inputs = batch["x"].to(device)
        wrist_mask = batch["wrist_mask"].to(device)
        activity_mask = batch["activity_mask"].to(device)
        lengths = batch["activity_lengths"].to(device)
        output = encode_layers(model, inputs, wrist_mask, activity_mask, lengths)
        if not checked:
            reference_activity = model._encode_activities(inputs, wrist_mask, activity_mask, lengths)
            reference = model(inputs, wrist_mask, activity_mask, lengths)
            max_equivalence = {
                "activity": float((output["bilateral_activity"] - reference_activity).abs().max()),
                "subject": float((output["subject_aggregation"] - reference["bag_embedding"]).abs().max()),
                "logits": float((output["classifier_logits"] - reference["logits"]).abs().max()),
            }
            if max(max_equivalence.values()) > 1.0e-5:
                raise RuntimeError(f"Component extraction differs from frozen forward: {max_equivalence}")
            checked = True
        wrist_embeddings.append(output["wrist_encoder"].cpu().numpy())
        wrist_masks.append(batch["wrist_mask"].numpy())
        activity_masks.append(batch["activity_mask"].numpy())
        labels.append(batch["y"].numpy())
        logits.append(output["classifier_logits"].cpu().numpy())
        subject_ids.extend(str(value) for value in batch["subject_id"])
    wrist = np.concatenate(wrist_embeddings)
    wrist_mask = np.concatenate(wrist_masks).astype(bool)
    activity_mask = np.concatenate(activity_masks).astype(bool)
    left = wrist[:, :, 0]
    right = wrist[:, :, 1]
    count = wrist_mask.sum(axis=2, keepdims=True)
    mean = wrist.sum(axis=2) / np.clip(count, 1, None)
    both = wrist_mask.all(axis=2)
    absdiff = np.abs(left - right) * both[..., None]
    return {
        "subject_id": np.asarray(subject_ids),
        "label": np.concatenate(labels),
        "logits": np.concatenate(logits),
        "activity_mask": activity_mask,
        "wrist_mask": wrist_mask,
        "components": {"left": left, "right": right, "mean": mean, "absdiff": absdiff},
        "component_masks": {
            "left": activity_mask & wrist_mask[:, :, 0],
            "right": activity_mask & wrist_mask[:, :, 1],
            "mean": activity_mask & wrist_mask.any(axis=2),
            "absdiff": activity_mask & both,
        },
    }, max_equivalence


def masked_activity_mean(tokens, mask):
    weights = mask[..., None].astype(np.float64)
    return (tokens * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1.0, None)


def paired_p_value(delta):
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    nonzero = delta[np.abs(delta) > 1.0e-12]
    return float(wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0


def main() -> None:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    embedding_dir = output / "embeddings"
    embedding_dir.mkdir(exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Bilateral component extraction requires CUDA")
    runs = []
    for item in args.run:
        seed, path = item.split(":", 1)
        runs.append((int(seed), Path(path).expanduser().resolve()))
    component_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    equivalence_rows: list[dict[str, Any]] = []
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
            train_dataset, val_dataset, _ = load_fold_datasets(stage, config)
            train, train_equivalence = extract_components(model, train_dataset, config, device)
            val, val_equivalence = extract_components(model, val_dataset, config, device)
            equivalence_rows.append({
                "seed": seed, "outer": outer, "inner": inner,
                **{f"train_{key}": value for key, value in train_equivalence.items()},
                **{f"validation_{key}": value for key, value in val_equivalence.items()},
            })
            head = head_metrics(val["logits"], val["label"])
            for split_name, extracted in (("train", train), ("validation", val)):
                valid_entries = extracted["activity_mask"][..., None]
                total = int(valid_entries.sum() * 2)
                missing_left = int((extracted["activity_mask"] & ~extracted["wrist_mask"][:, :, 0]).sum())
                missing_right = int((extracted["activity_mask"] & ~extracted["wrist_mask"][:, :, 1]).sum())
                incomplete = int((extracted["activity_mask"] & ~extracted["wrist_mask"].all(axis=2)).sum())
                mask_rows.append({
                    "seed": seed, "outer": outer, "inner": inner, "split": split_name,
                    "subject_count": len(extracted["label"]),
                    "valid_activity_count": int(extracted["activity_mask"].sum()),
                    "wrist_entries": total,
                    "missing_left": missing_left, "missing_right": missing_right,
                    "missing_wrist_entries": missing_left + missing_right,
                    "missing_wrist_rate": (missing_left + missing_right) / max(total, 1),
                    "incomplete_bilateral_activities": incomplete,
                    "incomplete_bilateral_activity_rate": incomplete / max(int(extracted["activity_mask"].sum()), 1),
                    "unique_wrist_mask_patterns": int(np.unique(extracted["wrist_mask"].reshape(len(extracted["label"]), -1), axis=0).shape[0]),
                })
            archive: dict[str, Any] = {
                "validation_subject_id": val["subject_id"],
                "validation_label": val["label"],
                "validation_wrist_mask": val["wrist_mask"],
            }
            for component in COMPONENTS:
                train_tokens = train["components"][component]
                val_tokens = val["components"][component]
                train_mask = train["component_masks"][component]
                val_mask = val["component_masks"][component]
                if not (train_mask.all() and val_mask.all()):
                    complete_train = train_mask.all(axis=1)
                    complete_val = val_mask.all(axis=1)
                    identity = token_identity_activity(
                        train_tokens[complete_train], val_tokens[complete_val], val["label"][complete_val]
                    )
                else:
                    identity = token_identity_activity(train_tokens, val_tokens, val["label"])
                train_x = masked_activity_mean(train_tokens, train_mask)
                val_x = masked_activity_mean(val_tokens, val_mask)
                disease, z_train, z_val = disease_probe(
                    train_x, train["label"], val_x, val["label"]
                )
                variance = variance_decomposition(z_val, val["label"])
                shift = representation_shift(
                    train_x, train["label"], val_x, val["label"], z_train, z_val,
                    seed + outer * 100 + inner * 10 + COMPONENTS.index(component),
                )
                component_rows.append({
                    "seed": seed, "outer": outer, "inner": inner,
                    "component": component, "representation_dim": int(train_x.shape[1]),
                    **head, **disease, **variance, **identity, **shift,
                })
                archive[f"train_subject_{component}"] = train_x
                archive[f"validation_subject_{component}"] = val_x
                archive[f"validation_activity_{component}"] = val_tokens
            np.savez_compressed(
                embedding_dir / f"seed{seed}_outer{outer}_inner{inner}.npz", **archive
            )
            processed += 1
            print(f"processed seed={seed} outer={outer} inner={inner}", flush=True)
        if args.max_folds and processed >= args.max_folds:
            break
    raw = pd.DataFrame(component_rows)
    masks = pd.DataFrame(mask_rows)
    equivalence = pd.DataFrame(equivalence_rows)
    raw.to_csv(output / "component_metrics_45_seed_folds.csv", index=False)
    masks.to_csv(output / "wrist_mask_audit_45_seed_folds.csv", index=False)
    equivalence.to_csv(output / "extraction_equivalence_checks.csv", index=False)
    numeric = [column for column in raw.columns if column not in {"seed", "outer", "inner", "component"}]
    aggregated = raw.groupby(["outer", "inner", "component"], as_index=False)[numeric].mean()
    aggregated.to_csv(output / "component_metrics_15_independent_splits.csv", index=False)
    mask_numeric = [column for column in masks.columns if column not in {"seed", "outer", "inner", "split"}]
    mask_aggregated = masks.groupby(["outer", "inner", "split"], as_index=False)[mask_numeric].mean()
    mask_aggregated.to_csv(output / "wrist_mask_audit_15_splits.csv", index=False)
    summary = []
    for component, group in aggregated.groupby("component", sort=False):
        for metric in numeric:
            values = group[metric].dropna()
            summary.append({
                "component": component, "metric": metric, "n_splits": len(values),
                "mean": float(values.mean()) if len(values) else float("nan"),
                "std_across_splits": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                "minimum": float(values.min()) if len(values) else float("nan"),
                "maximum": float(values.max()) if len(values) else float("nan"),
            })
    pd.DataFrame(summary).to_csv(output / "component_metric_summary.csv", index=False)
    relationship_metrics = [
        "disease_probe_ba", "disease_probe_auroc", "disease_centroid_ba",
        "disease_silhouette_cosine", "between_within_distance_ratio",
        "disease_between_variance_ratio", "subject_within_disease_variance_ratio",
        "subject_retrieval_top1", "subject_retrieval_top5", "subject_similarity_gap",
        "same_subject_similarity", "domain_probe_auc", "normalized_mean_shift",
        "coral_covariance_shift",
    ]
    relationships = []
    for component, group in aggregated.groupby("component", sort=False):
        for metric in relationship_metrics:
            valid = group[[metric, "head_balanced_accuracy"]].dropna()
            variable = len(valid) >= 3 and valid[metric].nunique() > 1 and valid.head_balanced_accuracy.nunique() > 1
            rho, p = spearmanr(valid[metric], valid.head_balanced_accuracy) if variable else (float("nan"), float("nan"))
            relationships.append({
                "component": component, "metric": metric, "n_splits": len(valid),
                "spearman_rho_with_head_ba": float(rho), "p_value": float(p),
            })
    relationship_frame = pd.DataFrame(relationships)
    finite = relationship_frame.p_value.notna()
    p_values = relationship_frame.loc[finite, "p_value"].to_numpy(dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    q_values = np.empty_like(adjusted)
    q_values[order] = np.clip(adjusted, 0.0, 1.0)
    relationship_frame["bh_fdr_q_across_reported_relationships"] = np.nan
    relationship_frame.loc[finite, "bh_fdr_q_across_reported_relationships"] = q_values
    relationship_frame.to_csv(output / "component_metric_head_ba_relationships.csv", index=False)
    comparison_metrics = relationship_metrics + ["activity_probe_accuracy"]
    indexed = aggregated.set_index(["outer", "inner", "component"])
    split_keys = sorted({(int(row.outer), int(row.inner)) for row in aggregated.itertuples()})
    pair_rows = []
    for left_index, left_name in enumerate(COMPONENTS):
        for right_name in COMPONENTS[left_index + 1:]:
            for metric in comparison_metrics:
                delta = np.asarray([
                    indexed.loc[(outer, inner, left_name), metric]
                    - indexed.loc[(outer, inner, right_name), metric]
                    for outer, inner in split_keys
                ], dtype=float)
                delta = delta[np.isfinite(delta)]
                pair_rows.append({
                    "comparison": f"{left_name}-{right_name}", "metric": metric,
                    "n_splits": len(delta), "mean_delta": float(delta.mean()),
                    "left_higher_splits": int(np.sum(delta > 0)),
                    "right_higher_splits": int(np.sum(delta < 0)),
                    "wilcoxon_p": paired_p_value(delta),
                })
    pd.DataFrame(pair_rows).to_csv(output / "component_pairwise_comparisons.csv", index=False)
    winner_rows = []
    for metric in comparison_metrics:
        for outer, inner in split_keys:
            values = {
                component: float(indexed.loc[(outer, inner, component), metric])
                for component in COMPONENTS
            }
            winner_rows.append({
                "outer": outer, "inner": inner, "metric": metric,
                "highest_component": max(values, key=values.get),
                "lowest_component": min(values, key=values.get),
                **values,
            })
    pd.DataFrame(winner_rows).to_csv(output / "component_split_winners.csv", index=False)
    protocol = {
        "scope": "fixed inner-development train/validation only",
        "outer_test_accessed": False,
        "architecture_modified": False,
        "model_retrained": False,
        "training_recipe_modified": False,
        "components": {name: "64-D" for name in COMPONENTS},
        "primary_independent_splits": 15,
        "seed_aggregation": "mean of seeds 42/43/44 within identical outer/inner split",
        "causal_boundary": "Passive probes quantify information and shift; they do not establish causal contribution to frozen-head performance.",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(protocol, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
