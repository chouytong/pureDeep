from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.datasets.builders import load_configured_records
from src.datasets.manifest import ManifestTimeSeriesDataset, load_manifest
from src.datasets.subject_activity import SubjectActivityDataset
from src.engine.checkpoint import (
    load_checkpoint,
    validate_checkpoint_compatibility,
)
from src.models import build_model
from src.utils.config import load_config, require_pads_classification_config
from src.utils.device import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer one validated bilateral PADS manifest pair"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument(
        "--pair-id", help="pair_id from a paired manifest; preserved for compatibility"
    )
    selector.add_argument("--subject-id", help="subject_id for multi-activity inference")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    parser.add_argument("--allow-legacy-checkpoint", action="store_true")
    parser.add_argument("--legacy-provenance", default=None)
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE"
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve() if args.output else None
    if output_path is not None and output_path.exists():
        raise FileExistsError(f"Refusing to overwrite inference output: {output_path}")
    config = load_config(args.config, args.overrides)
    require_pads_classification_config(config)
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    validate_checkpoint_compatibility(
        checkpoint,
        config,
        checkpoint_path=args.checkpoint,
        allow_legacy_checkpoint=args.allow_legacy_checkpoint,
        legacy_provenance=args.legacy_provenance,
    )
    classes = [str(value) for value in config["data"]["labels"]]
    if checkpoint.get("class_names") != classes:
        raise ValueError("Checkpoint class_names do not match configured label order")

    data = config["data"]
    if str(data.get("unit", "activity")).lower() == "subject":
        records = load_configured_records(data, classes)
        subject_id = args.subject_id
        if args.pair_id is not None:
            matched = [record for record in records if record.pair_id == args.pair_id]
            if len(matched) != 1:
                raise ValueError(
                    f"Expected one manifest row for pair_id={args.pair_id!r}, "
                    f"found {len(matched)}"
                )
            subject_id = matched[0].subject_id
        assert subject_id is not None
        selected = [record for record in records if record.subject_id == subject_id]
        if not selected:
            raise ValueError(f"No configured activity found for subject_id={subject_id!r}")
        normalization = checkpoint["normalization"]
        dataset = SubjectActivityDataset(
            records=selected,
            activities=data["activities"],
            data_config={**data, "eval_crop": data.get("inference_crop", "center")},
            mean=normalization["mean"].to(torch.float32),
            std=normalization["std"].to(torch.float32),
            split_name="inference",
        )
        item = dataset[0]
        signal = item["x"]
        wrist_mask = item["wrist_mask"]
        activity_mask = item["activity_mask"]
        activity_lengths = item["activity_lengths"]
        assert isinstance(signal, torch.Tensor)
        assert isinstance(wrist_mask, torch.Tensor)
        assert isinstance(activity_mask, torch.Tensor)
        assert isinstance(activity_lengths, torch.Tensor)
        if signal.ndim != 4:
            raise ValueError(
                f"Subject inference expects [A,2,C,T], got {tuple(signal.shape)}"
            )
        device = select_device(args.device)
        model = build_model(config).to(device)
        load_checkpoint(args.checkpoint, model=model, map_location=device)
        model.eval()
        outputs = model(
            signal.unsqueeze(0).to(device),
            wrist_mask.unsqueeze(0).to(device),
            activity_mask.unsqueeze(0).to(device),
            activity_lengths.unsqueeze(0).to(device),
        )
        probabilities = outputs["probabilities"][0].detach().cpu().tolist()
        attention = outputs["activity_attention"][0].detach().cpu().tolist()
        prediction = int(np.argmax(probabilities))
        result = {
            "subject_id": subject_id,
            "selection_pair_id": args.pair_id,
            "wrist_mode": data.get("wrist_mode", "bilateral"),
            "sensor_mode": data.get("sensor_mode", "acc_gyro"),
            "activities": list(data["activities"]),
            "activity_mask": activity_mask.tolist(),
            "activity_lengths": dict(zip(data["activities"], activity_lengths.tolist())),
            "activity_attention": dict(zip(data["activities"], attention)),
            "activity_pair_ids": json.loads(str(item["activity_pair_ids"])),
            "prediction_index": prediction,
            "prediction_label": classes[prediction],
            "probabilities": dict(zip(classes, probabilities)),
        }
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.subject_id is not None:
        raise ValueError("--subject-id requires a subject-level multi-activity config")
    records = load_manifest(
        data["manifest"],
        data["root"],
        labels=classes,
        activity=data.get("activity"),
        fold_column=str(data.get("fold", {}).get("manifest_fold_column", "fold")),
    )
    selected = [record for record in records if record.pair_id == args.pair_id]
    if len(selected) != 1:
        raise ValueError(
            f"Expected exactly one manifest row for pair_id={args.pair_id!r}, "
            f"found {len(selected)}"
        )
    normalization = checkpoint["normalization"]
    dataset = ManifestTimeSeriesDataset(
        records=selected,
        raw_input_channels=int(data.get("raw_input_channels", 6)),
        sensor_mode=str(data.get("sensor_mode", "acc_gyro")),
        wrist_mode=str(data.get("wrist_mode", "bilateral")),
        sequence_length=(
            None
            if data.get("sequence_length") is None
            else int(data["sequence_length"])
        ),
        crop=str(data.get("inference_crop", "center")),
        pad_value=float(data.get("pad_value", 0.0)),
        delimiter=str(data.get("delimiter", ",")),
        drop_time_column=bool(data.get("drop_time_column", False)),
        mean=normalization["mean"].to(torch.float32),
        std=normalization["std"].to(torch.float32),
    )
    item = dataset[0]
    signal = item["x"]
    mask = item["wrist_mask"]
    if signal.ndim != 3:
        raise ValueError(f"Inference expects one crop [2,C,T], got {tuple(signal.shape)}")
    device = select_device(args.device)
    model = build_model(config).to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    model.eval()
    outputs = model(signal.unsqueeze(0).to(device), mask.unsqueeze(0).to(device))
    probabilities = outputs["probabilities"][0].detach().cpu().tolist()
    prediction = int(np.argmax(probabilities))
    record = selected[0]
    result = {
        "pair_id": record.pair_id,
        "subject_id": record.subject_id,
        "activity": record.activity,
        "left_path": str(record.left_path),
        "right_path": str(record.right_path),
        "wrist_mode": data.get("wrist_mode", "bilateral"),
        "sensor_mode": data.get("sensor_mode", "acc_gyro"),
        "wrist_mask": mask.tolist(),
        "prediction_index": prediction,
        "prediction_label": classes[prediction],
        "probabilities": dict(zip(classes, probabilities)),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
