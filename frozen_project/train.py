from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from src.datasets.builders import build_datasets, make_loader
from src.engine.checkpoint import (
    input_metadata_from_config,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_input,
)
from src.engine.optim import build_optimizer, build_scheduler
from src.engine.runner import evaluate_epoch, train_epoch
from src.losses import build_loss
from src.models import build_model
from src.utils.artifacts import append_jsonl, create_run_directory, write_json
from src.utils.config import (
    load_config,
    require_pads_classification_config,
    save_config,
)
from src.utils.device import cleanup_distributed, init_distributed, select_device
from src.utils.environment import collect_environment, write_environment
from src.utils.model_stats import count_parameters
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PADS classifier")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _metric_improved(
    value: float, best: float, mode: str, minimum_delta: float
) -> bool:
    if mode == "max":
        return value > best + minimum_delta
    if mode == "min":
        return value < best - minimum_delta
    raise ValueError(f"early_stopping.mode must be max or min, got {mode!r}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    require_pads_classification_config(config)
    if config.get("nested_cv"):
        raise ValueError(
            "Nested-CV configs must use scripts/run_nested_cv.py; the legacy "
            "train.py path cannot access V3 outer-test subjects safely."
        )
    distributed = init_distributed(
        config.get("distributed", {}).get("enabled", "auto"),
        config.get("distributed", {}).get("backend", "auto"),
    )
    seed = int(config["experiment"]["seed"]) + distributed.rank
    seed_everything(seed, bool(config["experiment"].get("deterministic", True)))
    device = select_device(args.device, distributed.local_rank)
    if distributed.enabled and device.type != "cuda":
        device = torch.device("cpu")

    bundle = build_datasets(config)
    if len(bundle.validation) == 0:
        raise ValueError(
            "Training requires a non-empty validation split for early stopping. "
            "Set data.fold.validation_fraction or provide a frozen split file."
        )
    model = build_model(config).to(device)
    if distributed.enabled:
        model = DistributedDataParallel(
            model,
            device_ids=[distributed.local_rank] if device.type == "cuda" else None,
        )
    criterion = build_loss(config).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    use_amp = bool(config["training"].get("mixed_precision", False)) and (
        device.type == "cuda"
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_sampler = (
        DistributedSampler(
            bundle.train,
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            shuffle=True,
            seed=int(config["experiment"]["seed"]),
        )
        if distributed.enabled
        else None
    )
    data_config = config["data"]
    train_loader = make_loader(
        bundle.train,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=bool(data_config.get("pin_memory", False)) and device.type == "cuda",
        seed=seed,
        sampler=train_sampler,
    )
    validation_loader = make_loader(
        bundle.validation,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=bool(data_config.get("pin_memory", False)) and device.type == "cuda",
        seed=seed + 1,
    )
    test_loader = make_loader(
        bundle.test,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=bool(data_config.get("pin_memory", False)) and device.type == "cuda",
        seed=seed + 2,
    )

    run_path: str | None = None
    if distributed.is_main:
        if args.output_dir:
            run_dir = Path(args.output_dir).expanduser().resolve()
            if run_dir.exists() and any(run_dir.iterdir()):
                raise FileExistsError(
                    f"Refusing to overwrite non-empty output: {run_dir}"
                )
            for child in ("checkpoints", "logs", "predictions"):
                (run_dir / child).mkdir(parents=True, exist_ok=True)
        else:
            run_dir = create_run_directory(
                config["experiment"]["output_root"],
                config["experiment"]["name"],
            )
        run_path = str(run_dir)
    if distributed.enabled:
        broadcast_value = [run_path]
        torch.distributed.broadcast_object_list(broadcast_value, src=0)
        run_path = broadcast_value[0]
    if run_path is None:
        raise RuntimeError("Failed to establish a shared run directory")
    run_dir = Path(run_path)

    start_epoch = 0
    best_metric: float
    early = config["training"].get("early_stopping", {})
    metric_name = str(early.get("metric", "macro_f1"))
    metric_mode = str(early.get("mode", "max"))
    best_metric = float("-inf") if metric_mode == "max" else float("inf")
    resume = config["training"].get("checkpoint", {}).get("resume")
    if resume:
        checkpoint = load_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        if checkpoint.get("class_names") != bundle.class_names:
            raise ValueError("Checkpoint class_names do not match current data labels")
        validate_checkpoint_input(checkpoint, config)
        checkpoint_mean = checkpoint["normalization"]["mean"].to(
            device="cpu", dtype=torch.float32
        )
        checkpoint_std = checkpoint["normalization"]["std"].to(
            device="cpu", dtype=torch.float32
        )
        if not (
            torch.allclose(checkpoint_mean, bundle.mean)
            and torch.allclose(checkpoint_std, bundle.std)
        ):
            raise ValueError(
                "Checkpoint normalization does not match the current train split"
            )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint.get("best_metric", best_metric))

    if distributed.is_main:
        save_config(config, run_dir / "config.yaml")
        write_environment(
            collect_environment(Path(__file__).resolve().parent),
            run_dir / "environment.json",
        )
        write_json(run_dir / "split.json", bundle.split_summary)
        write_json(
            run_dir / "normalization.json",
            {
                "fitted_on": "train_subjects_only",
                "mean": bundle.mean.tolist(),
                "std": bundle.std.tolist(),
                "input_metadata": input_metadata_from_config(config),
            },
        )
        write_json(
            run_dir / "model.json",
            {
                **count_parameters(model),
                "device": str(device),
            },
        )
        print(f"Run directory: {run_dir}")
        print(json.dumps(bundle.split_summary, ensure_ascii=False))

    patience_count = 0
    epochs = int(config["training"]["epochs"])
    overfit_batches = int(config["training"].get("overfit_batches", 0))
    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler=scaler,
            mixed_precision=use_amp,
            gradient_clip_norm=config["training"].get("gradient_clip_norm"),
            max_batches=overfit_batches,
            zero_division=float(config["evaluation"].get("zero_division", 0.0)),
        )
        validation_metrics, _ = evaluate_epoch(
            model,
            validation_loader,
            criterion,
            device,
            mixed_precision=use_amp,
            max_batches=overfit_batches,
            zero_division=float(config["evaluation"].get("zero_division", 0.0)),
        )
        if scheduler is not None:
            scheduler.step()

        if distributed.is_main:
            record = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
                "gpu_peak_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else 0
                ),
                "timestamp": time.time(),
            }
            append_jsonl(run_dir / "logs" / "epochs.jsonl", record)
            write_json(run_dir / "metrics.json", record)
            current = float(validation_metrics[metric_name])
            improved = _metric_improved(
                current,
                best_metric,
                metric_mode,
                float(early.get("minimum_delta", 0.0)),
            )
            if improved:
                best_metric = current
                patience_count = 0
                save_checkpoint(
                    run_dir / "checkpoints" / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_metric,
                    config,
                    bundle.class_names,
                    bundle.mean,
                    bundle.std,
                )
            else:
                patience_count += 1
            save_checkpoint(
                run_dir / "checkpoints" / "last.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_metric,
                config,
                bundle.class_names,
                bundle.mean,
                bundle.std,
            )
            save_every = int(
                config["training"].get("checkpoint", {}).get("save_every", 0)
            )
            if save_every and (epoch + 1) % save_every == 0:
                save_checkpoint(
                    run_dir / "checkpoints" / f"epoch_{epoch + 1:04d}.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_metric,
                    config,
                    bundle.class_names,
                    bundle.mean,
                    bundle.std,
                )
            print(
                f"epoch={epoch + 1}/{epochs} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={validation_metrics['loss']:.4f} "
                f"val_macro_f1={validation_metrics['macro_f1']:.4f}"
            )
        should_stop = bool(early.get("enabled", False)) and (
            patience_count >= int(early.get("patience", 10))
        )
        if distributed.enabled:
            flag = torch.tensor([int(should_stop)], device=device)
            torch.distributed.broadcast(flag, src=0)
            should_stop = bool(flag.item())
        if should_stop:
            break

    # 测试集在训练和 early stopping 期间从未参与模型选择。训练结束后只载入
    # best.pt 评估一次；overfit_batches>0 时仍只属于小规模流程测试证据。
    if bool(config["evaluation"].get("run_test_after_training", True)):
        if distributed.enabled:
            torch.distributed.barrier()
        best_path = run_dir / "checkpoints" / "best.pt"
        load_checkpoint(best_path, model=model, map_location=device)
        test_metrics, test_predictions = evaluate_epoch(
            model,
            test_loader,
            criterion,
            device,
            mixed_precision=use_amp,
            max_batches=overfit_batches,
            zero_division=float(config["evaluation"].get("zero_division", 0.0)),
        )
        test_metrics["class_names"] = bundle.class_names
        for row in test_predictions:
            row["target_label"] = bundle.class_names[int(row["target"])]
            row["prediction_label"] = bundle.class_names[int(row["prediction"])]
        if distributed.is_main:
            write_json(run_dir / "test_metrics.json", test_metrics)
            if bool(config["evaluation"].get("save_predictions", True)):
                prediction_path = run_dir / "predictions" / "test" / "predictions.csv"
                prediction_path.parent.mkdir(parents=True, exist_ok=False)
                with prediction_path.open("x", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(
                        stream, fieldnames=test_predictions[0].keys()
                    )
                    writer.writeheader()
                    writer.writerows(test_predictions)
            print(
                "test "
                f"accuracy={test_metrics['accuracy']:.4f} "
                f"balanced_accuracy={test_metrics['balanced_accuracy']:.4f} "
                f"macro_f1={test_metrics['macro_f1']:.4f}"
            )
    cleanup_distributed()


if __name__ == "__main__":
    main()
