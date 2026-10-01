#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch

from src.datasets.builders import _select_records, load_configured_records, make_loader
from src.datasets.subject_activity import SubjectActivityDataset
from src.models import build_model
from src.utils.config import load_config


def arguments():
    parser = argparse.ArgumentParser(
        description="Read-only paired V8-GN/STR-01 mechanism extraction"
    )
    parser.add_argument("--v8-run", action="append", required=True,
                        help="SEED:/absolute/run/path")
    parser.add_argument("--str-run", action="append", required=True,
                        help="SEED:/absolute/run/path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_runs(items):
    result = {}
    for item in items:
        seed, path = item.split(":", 1)
        result[int(seed)] = Path(path).expanduser().resolve()
    return result


def datasets(stage, config):
    split = read_json(stage / "split.json")
    normalization = read_json(stage / "normalization.json")
    mean = torch.tensor(normalization["mean"], dtype=torch.float32)
    std = torch.tensor(normalization["std"], dtype=torch.float32)
    records = load_configured_records(config["data"], ["PD", "DD"])
    train = SubjectActivityDataset(
        _select_records(records, set(split["train_subjects"])),
        config["data"]["activities"], config["data"], mean, std, "validation",
    )
    validation = SubjectActivityDataset(
        _select_records(records, set(split["validation_subjects"])),
        config["data"]["activities"], config["data"], mean, std, "validation",
    )
    return train, validation


def classifier_linear(model):
    layers = [module for module in model.classifier.modules()
              if isinstance(module, torch.nn.Linear)]
    if len(layers) != 1:
        raise RuntimeError(f"Expected one classifier Linear, found {len(layers)}")
    return layers[0]


@torch.inference_mode()
def extract(model, dataset, config, device, structured):
    loader = make_loader(
        dataset, batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False, num_workers=int(config["data"].get("num_workers", 0)),
        pin_memory=False, seed=int(config["experiment"]["seed"]) + 2301,
    )
    keys = [
        "label", "activity_mask", "subject_embedding", "activity_embeddings",
        "activity_attention", "base_logits", "final_logits",
        "base_activity_margin",
    ]
    if structured:
        keys.extend([
            "structured_tokens", "structured_embedding", "residual_logits",
            "residual_activity_margin", "final_activity_margin",
        ])
    arrays = {key: [] for key in keys}
    subject_ids = []
    base_layer = classifier_linear(model)
    base_weight = base_layer.weight[1] - base_layer.weight[0]
    base_bias_margin = float((base_layer.bias[1] - base_layer.bias[0]).cpu())
    if structured:
        residual_layer = model.structured_token_residual.residual_head
        residual_weight = (
            residual_layer.weight[1] - residual_layer.weight[0]
        ).reshape(model.activity_count, -1)
        residual_bias_margin = float(
            (residual_layer.bias[1] - residual_layer.bias[0]).cpu()
        )
    else:
        residual_bias_margin = 0.0

    model.eval()
    for batch in loader:
        output = model(
            batch["x"].to(device), batch["wrist_mask"].to(device),
            batch["activity_mask"].to(device),
            batch["activity_lengths"].to(device),
        )
        activity_embeddings = output["activity_embeddings"]
        attention = output["activity_attention"]
        base_activity_margin = (
            activity_embeddings * base_weight[None, None, :]
        ).sum(-1) * attention
        arrays["label"].append(batch["y"].numpy())
        arrays["activity_mask"].append(batch["activity_mask"].numpy())
        arrays["subject_embedding"].append(output["bag_embedding"].cpu().numpy())
        arrays["activity_embeddings"].append(activity_embeddings.cpu().numpy())
        arrays["activity_attention"].append(attention.cpu().numpy())
        arrays["base_logits"].append(
            output.get("base_logits", output["logits"]).cpu().numpy()
        )
        arrays["final_logits"].append(output["logits"].cpu().numpy())
        arrays["base_activity_margin"].append(base_activity_margin.cpu().numpy())
        if structured:
            tokens = output["structured_activity_tokens"]
            residual_activity_margin = (
                tokens * residual_weight[None, :, :]
            ).sum(-1)
            arrays["structured_tokens"].append(tokens.cpu().numpy())
            arrays["structured_embedding"].append(
                output["structured_residual_embedding"].cpu().numpy()
            )
            arrays["residual_logits"].append(
                output["structured_residual_logits"].cpu().numpy()
            )
            arrays["residual_activity_margin"].append(
                residual_activity_margin.cpu().numpy()
            )
            arrays["final_activity_margin"].append(
                (base_activity_margin + residual_activity_margin).cpu().numpy()
            )
        subject_ids.extend(str(value) for value in batch["subject_id"])
    result = {key: np.concatenate(value) for key, value in arrays.items()}
    result["subject_id"] = np.asarray(subject_ids)
    result["base_bias_margin"] = np.asarray(base_bias_margin, dtype=np.float32)
    result["residual_bias_margin"] = np.asarray(
        residual_bias_margin, dtype=np.float32
    )
    if structured:
        result["decision_input"] = np.concatenate([
            result["subject_embedding"], result["structured_embedding"]
        ], axis=1)
    return result


def margin(logits):
    return logits[:, 1] - logits[:, 0]


def audit(payload, structured):
    base_error = np.max(np.abs(
        margin(payload["base_logits"])
        - (payload["base_activity_margin"].sum(1)
           + float(payload["base_bias_margin"]))
    ))
    final_error = np.max(np.abs(
        margin(payload["final_logits"])
        - (margin(payload["base_logits"])
           + (margin(payload["residual_logits"]) if structured else 0.0))
    ))
    residual_error = 0.0
    if structured:
        residual_error = np.max(np.abs(
            margin(payload["residual_logits"])
            - (payload["residual_activity_margin"].sum(1)
               + float(payload["residual_bias_margin"]))
        ))
    return float(base_error), float(residual_error), float(final_error)


def main():
    args = arguments()
    v8_runs, str_runs = parse_runs(args.v8_run), parse_runs(args.str_run)
    if set(v8_runs) != set(str_runs):
        raise ValueError("V8 and STR seed sets differ")
    output = args.output_dir.expanduser().resolve()
    embedding_dir = output / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Paired representation extraction requires CUDA")
    checks = []
    for seed in sorted(v8_runs):
        v8_stages = {
            (int(stage.parent.name.split("_")[-1]), int(stage.name.split("_")[-1])): stage
            for stage in v8_runs[seed].glob("outer_*/inner_*")
        }
        str_stages = {
            (int(stage.parent.name.split("_")[-1]), int(stage.name.split("_")[-1])): stage
            for stage in str_runs[seed].glob("outer_*/inner_*")
        }
        if set(v8_stages) != set(str_stages):
            raise ValueError(f"Stage mismatch for seed {seed}")
        for outer, inner in sorted(v8_stages):
            models = {}
            extracted = {}
            for name, stage, structured in (
                ("v8", v8_stages[(outer, inner)], False),
                ("str", str_stages[(outer, inner)], True),
            ):
                config = load_config(stage / "config.yaml")
                model = build_model(config).to(device)
                print(
                    f"loading name={name} seed={seed} outer={outer} inner={inner} "
                    f"stage={stage} structured_model="
                    f"{model.structured_token_residual is not None}",
                    flush=True,
                )
                checkpoint = torch.load(
                    stage / "checkpoints/best.pt", map_location=device,
                    weights_only=False,
                )
                model.load_state_dict(checkpoint["model_state"], strict=True)
                train_data, validation_data = datasets(stage, config)
                extracted[(name, "train")] = extract(
                    model, train_data, config, device, structured
                )
                extracted[(name, "validation")] = extract(
                    model, validation_data, config, device, structured
                )
                models[name] = model
            for partition in ("train", "validation"):
                v8 = extracted[("v8", partition)]
                stru = extracted[("str", partition)]
                if not np.array_equal(v8["subject_id"], stru["subject_id"]):
                    raise ValueError(
                        f"Subject ordering mismatch {seed}/{outer}/{inner}/{partition}"
                    )
                if not np.array_equal(v8["label"], stru["label"]):
                    raise ValueError(
                        f"Label mismatch {seed}/{outer}/{inner}/{partition}"
                    )
            errors = {}
            for name, structured in (("v8", False), ("str", True)):
                for partition in ("train", "validation"):
                    base, residual, final = audit(
                        extracted[(name, partition)], structured
                    )
                    errors[f"{name}_{partition}_base_error"] = base
                    errors[f"{name}_{partition}_residual_error"] = residual
                    errors[f"{name}_{partition}_final_error"] = final
            checks.append({
                "seed": seed, "outer": outer, "inner": inner,
                **errors,
                "str_subject_dimension": int(
                    extracted[("str", "train")]["subject_embedding"].shape[1]
                ),
                "str_structured_dimension": int(
                    extracted[("str", "train")]["structured_embedding"].shape[1]
                ),
            })
            archive = {}
            for (name, partition), payload in extracted.items():
                for key, value in payload.items():
                    archive[f"{name}_{partition}_{key}"] = value
            np.savez_compressed(
                embedding_dir / f"seed{seed}_outer{outer}_inner{inner}.npz",
                **archive,
            )
            print(f"extracted seed={seed} outer={outer} inner={inner}", flush=True)
            del models, extracted
            torch.cuda.empty_cache()
    with (output / "extraction_checks.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checks[0]))
        writer.writeheader()
        writer.writerows(checks)
    protocol = {
        "scope": "fixed inner-development train/validation only",
        "outer_information_used": False,
        "models_retrained": False,
        "architecture_modified": False,
        "seeds": sorted(v8_runs),
        "embedding_files": len(checks),
        "activity_margin_definition": (
            "exact additive DD-minus-PD logit contribution; activity attention "
            "times original classifier projection for base path, and fixed head "
            "weight block times ordered token for residual path"
        ),
        "biases": "reported separately and excluded from per-activity contributions",
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(protocol, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
