#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.datasets.builders import _select_records, load_configured_records, make_loader
from src.datasets.subject_activity import SubjectActivityDataset
from src.models import build_model
from src.utils.config import load_config


def arguments():
    parser = argparse.ArgumentParser(
        description="Extract STR-01 development-only decision representations"
    )
    parser.add_argument("--run", action="append", required=True,
                        help="SEED:/absolute/run/path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def datasets(stage, config):
    split = read_json(stage / "split.json")
    normalization = read_json(stage / "normalization.json")
    mean = torch.tensor(normalization["mean"], dtype=torch.float32)
    std = torch.tensor(normalization["std"], dtype=torch.float32)
    records = load_configured_records(config["data"], ["PD", "DD"])
    train = SubjectActivityDataset(
        _select_records(records, set(split["train_subjects"])),
        config["data"]["activities"], config["data"], mean, std, "validation"
    )
    validation = SubjectActivityDataset(
        _select_records(records, set(split["validation_subjects"])),
        config["data"]["activities"], config["data"], mean, std, "validation"
    )
    return train, validation


@torch.inference_mode()
def extract(model, dataset, config, device):
    loader = make_loader(
        dataset, batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False, num_workers=int(config["data"].get("num_workers", 0)),
        pin_memory=False, seed=int(config["experiment"]["seed"]) + 1901,
    )
    arrays = {key: [] for key in (
        "label", "activity_mask", "subject_embedding", "structured_tokens",
        "structured_embedding", "base_logits", "residual_logits", "final_logits",
    )}
    subject_ids = []
    model.eval()
    for batch in loader:
        output = model(
            batch["x"].to(device), batch["wrist_mask"].to(device),
            batch["activity_mask"].to(device), batch["activity_lengths"].to(device),
        )
        arrays["label"].append(batch["y"].numpy())
        arrays["activity_mask"].append(batch["activity_mask"].numpy())
        arrays["subject_embedding"].append(output["bag_embedding"].cpu().numpy())
        arrays["structured_tokens"].append(
            output["structured_activity_tokens"].cpu().numpy()
        )
        arrays["structured_embedding"].append(
            output["structured_residual_embedding"].cpu().numpy()
        )
        arrays["base_logits"].append(output["base_logits"].cpu().numpy())
        arrays["residual_logits"].append(
            output["structured_residual_logits"].cpu().numpy()
        )
        arrays["final_logits"].append(output["logits"].cpu().numpy())
        subject_ids.extend(str(value) for value in batch["subject_id"])
    result = {key: np.concatenate(value) for key, value in arrays.items()}
    result["subject_id"] = np.asarray(subject_ids)
    result["decision_input"] = np.concatenate([
        result["subject_embedding"], result["structured_embedding"]
    ], axis=1)
    return result


def main():
    args = arguments()
    output = args.output_dir.expanduser().resolve()
    embedding_dir = output / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("STR-01 extraction requires CUDA")
    checks = []
    for item in args.run:
        seed_text, run_text = item.split(":", 1)
        seed, run = int(seed_text), Path(run_text).expanduser().resolve()
        for stage in sorted(run.glob("outer_*/inner_*")):
            match = re.fullmatch(r"outer_(\d+)", stage.parent.name)
            outer = int(match.group(1))
            inner = int(stage.name.split("_")[-1])
            config = load_config(stage / "config.yaml")
            model = build_model(config).to(device)
            checkpoint = torch.load(
                stage / "checkpoints/best.pt", map_location=device, weights_only=False
            )
            model.load_state_dict(checkpoint["model_state"], strict=True)
            train_data, validation_data = datasets(stage, config)
            train = extract(model, train_data, config, device)
            validation = extract(model, validation_data, config, device)
            max_combination_error = max(
                float(np.max(np.abs(part["final_logits"] - (
                    part["base_logits"] + part["residual_logits"]
                )))) for part in (train, validation)
            )
            checks.append({
                "seed": seed, "outer": outer, "inner": inner,
                "max_logit_combination_error": max_combination_error,
                "structured_dimension": int(train["structured_embedding"].shape[1]),
                "decision_input_dimension": int(train["decision_input"].shape[1]),
            })
            archive = {}
            for partition, payload in (("train", train), ("validation", validation)):
                for key, value in payload.items():
                    archive[f"{partition}_{key}"] = value
            np.savez_compressed(
                embedding_dir / f"seed{seed}_outer{outer}_inner{inner}.npz", **archive
            )
            print(f"extracted seed={seed} outer={outer} inner={inner}", flush=True)
    frame = pd.DataFrame(checks)
    frame.to_csv(output / "extraction_checks.csv", index=False)
    protocol = {
        "scope": "fixed inner-development train/validation only",
        "outer_information_used": False,
        "seeds": sorted(frame.seed.unique().astype(int).tolist()),
        "embedding_files": int(len(frame)),
        "structured_representation": "ordered 11 x 16 projected bilateral activity tokens",
        "decision_input": "258-D original subject embedding concatenated with 176-D structured residual embedding",
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
