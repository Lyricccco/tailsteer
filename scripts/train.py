#!/usr/bin/env python3
"""Train the configured TailSteer algorithm from complete cache shards."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from tailsteer.cache.dataset import split_dataset
from tailsteer.cache.train_selection import select_training_dataset
from tailsteer.models import create_model
from tailsteer.training.checkpoint import resolve_resume
from tailsteer.training.distributed import initialize_distributed, prepare_model
from tailsteer.training.factory import (
    build_algorithm,
    build_training_dataset,
    build_training_regularizer,
    requires_independent_teacher,
)
from tailsteer.training.trainer import Trainer
from tailsteer.utils import (
    cache_root,
    configure_logging,
    data_root,
    load_config,
    model_root,
    run_root,
    save_resolved_config,
    seed_everything,
    workspace_root,
)


def _git_commit(repository: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _scheduler(
    optimizer, *, warmup: int, total: int, schedule_steps: int | None = None
):
    # An explicit horizon lets duration ablations share the same LR trajectory.
    # Omitted: preserve the historical warmup + cosine-to-max_steps schedule.
    if schedule_steps is not None:
        if (
            isinstance(schedule_steps, bool)
            or not isinstance(schedule_steps, int)
            or schedule_steps <= max(0, warmup)
        ):
            raise ValueError(
                "lr_schedule_steps must be a positive integer > warmup_steps"
            )
        total = schedule_steps

    def scale(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--resume", default=None, help="checkpoint path, auto, or none")
    parser.add_argument(
        "--stop-at-step",
        type=int,
        default=None,
        help=(
            "Gracefully stop after this optimizer step. This transient boundary "
            "does not change training.max_steps or the scheduler horizon."
        ),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    distributed_config = config.get("distributed", {"mode": "single"})
    context = initialize_distributed(
        str(distributed_config.get("mode", "single")),
        timeout_seconds=distributed_config.get("timeout_seconds"),
    )
    try:
        seed_everything(int(config.get("seed", 0)), rank=context.rank)
        run_dir = Path(config["run"]["output_dir"]).expanduser().resolve()
        repository = Path(__file__).resolve().parents[1]
        resume_value = (
            args.resume if args.resume is not None else config["run"].get("resume")
        )
        setup_error = None
        if context.is_main:
            try:
                if run_dir.exists() and resume_value in {None, "", "none"}:
                    raise FileExistsError(
                        f"run directory already exists; use --resume auto: {run_dir}"
                    )
                run_dir.mkdir(parents=True, exist_ok=True)
                resolved_config = run_dir / "config.yaml"
                if resolved_config.is_file():
                    existing_config = yaml.safe_load(
                        resolved_config.read_text(encoding="utf-8")
                    )
                    if existing_config != config:
                        raise ValueError("resume config differs from the saved config")
                else:
                    save_resolved_config(config, run_dir)
                metadata_path = run_dir / "metadata.json"
                if not metadata_path.is_file():
                    upstream = yaml.safe_load(
                        (repository / "upstream.lock").read_text(encoding="utf-8")
                    )
                    metadata = {
                        "started_at": datetime.now(timezone.utc).isoformat(),
                        "tailsteer_commit": _git_commit(repository),
                        "upstream": upstream,
                        "model_name": config["model"]["name"],
                        "model_path": config["model"]["path"],
                        "seed": int(config.get("seed", 0)),
                        "initial_world_size": context.world_size,
                        "workspace_root": str(workspace_root()),
                        "data_root": str(data_root()),
                        "cache_root": str(cache_root()),
                        "model_root": str(model_root()),
                        "run_root": str(run_root()),
                    }
                    metadata_path.write_text(
                        json.dumps(metadata, indent=2, sort_keys=True, default=str),
                        encoding="utf-8",
                    )
            except Exception as exc:
                setup_error = f"{type(exc).__name__}: {exc}"
        setup_error = context.broadcast_object(setup_error)
        if setup_error is not None:
            raise RuntimeError(f"training run setup failed: {setup_error}")
        logger = configure_logging(rank=context.rank, output_dir=run_dir)
        logger.info("TailSteer repository: %s", workspace_root())
        logger.info("TailSteer data root: %s", data_root())
        logger.info("TailSteer cache root: %s", cache_root())
        logger.info("TailSteer model root: %s", model_root())
        logger.info("TailSteer run root: %s", run_root())

        # Public execution path: config -> model -> dataset -> algorithm ->
        # distributed model -> optimizer -> Trainer.
        data_config = config["data"]
        dataset_spec = build_training_dataset(config)
        dataset = dataset_spec.dataset
        train_dataset, validation_dataset = split_dataset(
            dataset,
            seed=int(data_config.get("split_seed", 42)),
            validation_fraction=float(data_config.get("validation_fraction", 0.01)),
        )
        train_dataset, selection_summary = select_training_dataset(
            train_dataset, data_config
        )
        if selection_summary is not None:
            logger.info("training data selection: %s", selection_summary)
            if context.rank == 0:
                (run_dir / "data-selection.json").write_text(
                    json.dumps(
                        {
                            **selection_summary,
                            "validation_items": len(validation_dataset),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        visual = config["training"].get("visual_validation", {})
        visual_enabled = isinstance(visual, dict) and bool(visual.get("enabled", False))
        model = create_model(
            config["model"],
            device=context.device,
            include_vae=visual_enabled,
            independent_actors=requires_independent_teacher(config),
            gradient_checkpointing=bool(
                config["training"].get("gradient_checkpointing", False)
            ),
            export_path=config["model"].get("init_export_path"),
        )
        model.configure_trainable(
            config["training"].get("trainable", {"policy": "full"})
        )
        trainable_manifest = [
            {"name": name, "numel": parameter.numel()}
            for name, parameter in model.get_trainable_module().named_parameters()
            if parameter.requires_grad
        ]
        total_numel = sum(
            parameter.numel() for parameter in model.get_trainable_module().parameters()
        )
        trainable_numel = sum(item["numel"] for item in trainable_manifest)
        if context.is_main:
            (run_dir / "trainable-parameters.json").write_text(
                json.dumps(trainable_manifest, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            logger.info(
                "trainable parameters: %d tensors, %d / %d elements (%.3f%%)",
                len(trainable_manifest),
                trainable_numel,
                total_numel,
                100.0 * trainable_numel / total_numel,
            )
        context.barrier()
        algorithm = build_algorithm(
            config,
            model,
            dataset_spec.contract,
            distributed_context=context,
        )
        describe_sampling = getattr(
            getattr(algorithm, "sampler", None), "describe", None
        )
        if context.is_main and callable(describe_sampling):
            distribution = describe_sampling()
            (run_dir / "sampling-distribution.json").write_text(
                json.dumps(distribution, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            logger.info(
                "target sampling: %s; region probabilities: %s",
                distribution["strategy"],
                distribution["region_probabilities"],
            )
        prepare_model(model, context, distributed_config.get("fsdp2"))
        regularizer = build_training_regularizer(
            config, model, distributed_context=context
        )
        parameters = model.trainable_parameters()
        if not parameters:
            raise RuntimeError("no trainable parameters after distributed preparation")
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"].get("weight_decay", 0.0)),
        )
        batch_size = int(config["training"].get("batch_size", 1))
        per_rank_samples = math.ceil(len(train_dataset) / context.world_size)
        batches = math.ceil(per_rank_samples / batch_size)
        updates_per_epoch = math.ceil(
            batches / int(config["training"].get("gradient_accumulation_steps", 1))
        )
        max_steps = config["training"].get("max_steps")
        total_steps = (
            int(max_steps)
            if max_steps is not None
            else updates_per_epoch * int(config["training"]["max_epochs"])
        )
        scheduler = _scheduler(
            optimizer,
            warmup=int(config["training"].get("warmup_steps", 0)),
            total=total_steps,
            schedule_steps=config["training"].get("lr_schedule_steps"),
        )
        trainer = Trainer(
            model=model,
            algorithm=algorithm,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            config=config,
            run_dir=run_dir,
            context=context,
            logger=logger,
            collate_fn=dataset_spec.collate_fn,
            regularizer=regularizer,
        )
        trainer.resume(resolve_resume(run_dir, resume_value))
        state = trainer.train(stop_at_step=args.stop_at_step)
        if context.is_main:
            print(json.dumps(state, indent=2, sort_keys=True))
        return 0
    finally:
        context.close()


if __name__ == "__main__":
    raise SystemExit(main())
