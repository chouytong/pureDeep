#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.datasets.builders import _split_records, load_configured_records
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create one reusable subject split for all wrist modes"
    )
    parser.add_argument("--config", default="configs/pads_multi_activity_v2.yaml")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE"
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen split: {output}")
    config = load_config(args.config, args.overrides)
    data = config["data"]
    records = load_configured_records(data, [str(value) for value in data["labels"]])
    fold = data["fold"]
    _, _, _, summary = _split_records(
        records,
        num_folds=int(fold["num_folds"]),
        test_fold=int(fold["test_fold"]),
        validation_fraction=float(fold.get("validation_fraction", 0.0)),
        seed=int(config["experiment"]["seed"]),
        use_manifest_folds=bool(fold.get("manifest_fold_column")),
        split_file=None,
    )
    payload = {
        "task": config["task"]["name"],
        "labels": data["labels"],
        "activity": data.get("activity"),
        "seed": int(config["experiment"]["seed"]),
        "train_subjects": summary["train_subjects"],
        "validation_subjects": summary["validation_subjects"],
        "test_subjects": summary["test_subjects"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
