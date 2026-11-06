"""A recognizable PyTorch loop with accumulation, validation, and resume."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from tailsteer.cache.dataset import cache_collate, cache_record

from .checkpoint import (
    export_inference_model,
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from .distributed import maybe_no_sync
from .visual_validation import (
    aggregate_visual_records,
    image_edit_metrics,
    save_fixed_sample_manifest,
    save_visual_panel,
    select_fixed_local_edit_indices,
)


def _metric_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.8f}"


def _requires_gradient_sync(mode: str, *, optimizer_boundary: bool) -> bool:
    """Keep FSDP2 accumulated gradients sharded between microbatches.

    Disabling FSDP2 reduce-scatter retains a full unsharded gradient in
    ``reduce_dtype`` on every rank. That is prohibitively large for the models
    trained here. Synchronizing each microbatch still accumulates the reduced
    shards in ``parameter.grad``; the optimizer boundary remains independent.
    """

    return optimizer_boundary or mode == "fsdp2"


class Trainer:
    def __init__(
        self,
        *,
        model: Any,
        algorithm: Any,
        optimizer: Any,
        scheduler: Any,
        train_dataset: Any,
        validation_dataset: Any,
        config: dict[str, Any],
        run_dir: str | Path,
        context: Any,
        logger: Any,
        collate_fn: Any = cache_collate,
        regularizer: Any | None = None,
    ) -> None:
        import torch
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler

        self.model = model
        self.algorithm = algorithm
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.context = context
        self.logger = logger
        self.regularizer = regularizer
        self.validation_dataset = validation_dataset
        self.checkpoint_io = config["training"].get("checkpoint_io")
        self._algorithm_accepts_training = (
            "training" in inspect.signature(algorithm.compute_loss).parameters
        )
        self.root = model.get_trainable_module()
        self.parameters = model.trainable_parameters()
        training = config["training"]
        self.loader_generator = torch.Generator()
        self.train_sampler = (
            DistributedSampler(
                train_dataset,
                num_replicas=context.world_size,
                rank=context.rank,
                shuffle=True,
                seed=int(config.get("seed", 0)),
                drop_last=False,
            )
            if context.world_size > 1
            else None
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=int(training.get("batch_size", 1)),
            sampler=self.train_sampler,
            shuffle=self.train_sampler is None,
            num_workers=int(training.get("num_workers", 0)),
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn,
            drop_last=False,
            generator=self.loader_generator,
        )
        self.validation_loader = None
        if len(validation_dataset):
            from torch.utils.data import Subset

            by_task: dict[str, list[int]] = {}
            for index in range(len(validation_dataset)):
                task = str(cache_record(validation_dataset, index)["task"])
                by_task.setdefault(task, []).append(index)
            ordered_indices: list[int] = []
            block = max(1, context.world_size)
            task_names = sorted(by_task)
            while any(by_task.values()):
                for task in task_names:
                    ordered_indices.extend(by_task[task][:block])
                    del by_task[task][:block]
            loss_validation_dataset = Subset(validation_dataset, ordered_indices)
            validation_sampler = (
                DistributedSampler(
                    loss_validation_dataset,
                    num_replicas=context.world_size,
                    rank=context.rank,
                    shuffle=False,
                    drop_last=False,
                )
                if context.world_size > 1
                else None
            )
            self.validation_loader = DataLoader(
                loss_validation_dataset,
                batch_size=int(training.get("batch_size", 1)),
                sampler=validation_sampler,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn,
            )
        self.state = {"global_step": 0, "epoch": 0, "batch_in_epoch": 0}
        visual = training.get("visual_validation", {})
        self.visual_validation = dict(visual) if isinstance(visual, dict) else {}
        self.visual_indices: tuple[int, ...] = ()
        if bool(self.visual_validation.get("enabled", False)):
            if not callable(getattr(model, "generate_validation_sample", None)):
                raise TypeError("model does not support full visual validation")
            self.visual_indices = select_fixed_local_edit_indices(
                validation_dataset,
                count=int(self.visual_validation.get("samples", 10)),
                seed=int(self.visual_validation.get("seed", config.get("seed", 0))),
            )
            if context.is_main:
                save_fixed_sample_manifest(
                    validation_dataset,
                    self.visual_indices,
                    self.run_dir / "validation" / "fixed-samples.json",
                )
            context.barrier()

    def resume(self, checkpoint: Path | None) -> None:
        if checkpoint is None:
            return
        self.state = load_checkpoint(
            checkpoint,
            root=self.root,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            context=self.context,
            checkpoint_io=self.checkpoint_io,
        )
        self.logger.info(
            "resumed %s at step=%d epoch=%d batch=%d",
            checkpoint,
            self.state["global_step"],
            self.state["epoch"],
            self.state["batch_in_epoch"],
        )

    def _compute_loss(self, batch: Any, *, training: bool) -> Any:
        if self._algorithm_accepts_training:
            result = self.algorithm.compute_loss(batch, training=training)
        else:
            result = self.algorithm.compute_loss(batch)
        if not training or self.regularizer is None:
            return result
        from .algorithms.common import AlgorithmResult

        task_loss = result.loss
        anchor_loss, anchor_metrics = self.regularizer()
        metrics = {**result.metrics, "task_loss": task_loss.detach(), **anchor_metrics}
        return AlgorithmResult(
            loss=task_loss + anchor_loss,
            metrics=metrics,
            per_sample=result.per_sample,
        )

    def _validate(self) -> None:
        import json

        import torch

        if self.validation_loader is None:
            return
        was_training = self.root.training
        self.root.eval()
        limit = int(self.config["training"].get("validation_batches", 20))
        local_results: list[dict[str, Any]] = []
        with torch.no_grad():
            for batch_index, batch in enumerate(self.validation_loader):
                if limit > 0 and batch_index >= limit:
                    break
                result = self._compute_loss(batch, training=False)
                per_sample = result.per_sample
                if per_sample is None:
                    per_sample = result.loss.detach().expand(len(batch["metadata"]))
                for metadata, loss in zip(
                    batch["metadata"], per_sample.detach().float().cpu(), strict=True
                ):
                    local_results.append(
                        {
                            "sample_id": str(metadata["sample_id"]),
                            "task": str(metadata["task"]),
                            "loss": float(loss.item()),
                        }
                    )
        gathered = self.context.gather_objects(local_results)
        if self.context.is_main:
            # DistributedSampler pads to equalize FSDP forward counts. Deduplicate
            # those padded records so the report describes the true validation set.
            unique = {
                item["sample_id"]: item
                for rank_records in gathered
                for item in rank_records
            }
            values = list(unique.values())
            local = [item["loss"] for item in values if item["task"] == "local_edit"]
            noop = [item["loss"] for item in values if item["task"] == "noop"]
            all_losses = [item["loss"] for item in values]
            report = {
                "step": int(self.state["global_step"]),
                "sample_count": len(values),
                "loss": sum(all_losses) / len(all_losses) if all_losses else None,
                "local_edit_keep_region_loss": (
                    sum(local) / len(local) if local else None
                ),
                "local_edit_count": len(local),
                "noop_full_image_loss": sum(noop) / len(noop) if noop else None,
                "noop_count": len(noop),
            }
            destination = (
                self.run_dir
                / "validation"
                / f"step-{int(self.state['global_step']):08d}"
            )
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "loss-metrics.json").write_text(
                json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
            )
            self.logger.info(
                "validation step=%d loss=%s local_keep=%s noop_full=%s",
                self.state["global_step"],
                _metric_text(report["loss"]),
                _metric_text(report["local_edit_keep_region_loss"]),
                _metric_text(report["noop_full_image_loss"]),
            )
        if self.visual_indices:
            self._validate_visual()
        self.root.train(was_training)

    def _validate_visual(self) -> None:
        import json
        import math

        import torch

        step = int(self.state["global_step"])
        destination = self.run_dir / "validation" / f"step-{step:08d}"
        rounds = math.ceil(len(self.visual_indices) / self.context.world_size)
        local_records: list[dict[str, Any]] = []
        with torch.no_grad():
            for round_index in range(rounds):
                order = round_index * self.context.world_size + self.context.rank
                valid = order < len(self.visual_indices)
                dataset_index = self.visual_indices[order if valid else 0]
                sample = self.validation_dataset[dataset_index]
                source, generated = self.model.generate_validation_sample(
                    sample, self.visual_validation
                )
                if not valid:
                    continue
                sample_id = str(sample.metadata["sample_id"])
                stem = f"{order:02d}-{sample_id.replace('/', '_').replace(':', '_')}"
                rank_dir = destination / "images"
                rank_dir.mkdir(parents=True, exist_ok=True)
                source.save(rank_dir / f"{stem}-source.png")
                generated.save(rank_dir / f"{stem}-generated.png")
                save_visual_panel(
                    source,
                    generated,
                    sample.keep_mask,
                    rank_dir / f"{stem}-panel.png",
                )
                local_records.append(
                    {
                        "order": order,
                        "sample_id": sample_id,
                        "source_identity": str(sample.metadata["source_identity"]),
                        "prompt": str(sample.metadata["prompt"]),
                        "seed": int(sample.metadata["seed"]),
                        "source_image": f"images/{stem}-source.png",
                        "generated_image": f"images/{stem}-generated.png",
                        "panel": f"images/{stem}-panel.png",
                        "metrics": image_edit_metrics(
                            source, generated, sample.keep_mask
                        ),
                    }
                )
        gathered = self.context.gather_objects(local_records)
        if self.context.is_main:
            records = sorted(
                [item for rank_records in gathered for item in rank_records],
                key=lambda item: item["order"],
            )
            describe_inference = getattr(
                self.model, "validation_inference_config", None
            )
            inference = (
                describe_inference(self.visual_validation)
                if callable(describe_inference)
                else {
                    "num_steps": int(self.visual_validation.get("num_steps", 50)),
                    "timestep_shift": float(
                        self.visual_validation.get("timestep_shift", 3.0)
                    ),
                    "cfg_scale": float(self.visual_validation.get("cfg_scale", 4.0)),
                    "img_cfg_scale": float(
                        self.visual_validation.get("img_cfg_scale", 1.0)
                    ),
                }
            )
            report = {
                "step": step,
                "sample_count": len(records),
                "selection_pool": "held_out_validation_sources",
                "inference": inference,
                "aggregate": aggregate_visual_records(records),
                "samples": records,
                "quality_note": (
                    "The cache has no edited reference images. Image quality is "
                    "reported with source-relative technical/locality metrics and "
                    "saved panels; semantic edit correctness requires visual review."
                ),
            }
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "visual-report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            self.logger.info(
                "visual validation step=%d samples=%d panels=%s",
                step,
                len(records),
                destination / "images",
            )
        self.context.barrier()

    def _checkpoint(self) -> Path:
        checkpoint = save_checkpoint(
            self.run_dir,
            step=int(self.state["global_step"]),
            root=self.root,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            trainer_state=dict(self.state),
            context=self.context,
            checkpoint_io=self.checkpoint_io,
        )
        if bool(self.config["training"].get("export_inference", True)):
            export_inference_model(
                self.run_dir,
                name=f"step-{int(self.state['global_step']):08d}",
                model=self.model,
                root=self.root,
                step=int(self.state["global_step"]),
                context=self.context,
            )
        return checkpoint

    def train(self, *, stop_at_step: int | None = None) -> dict[str, Any]:
        """Run until the configured target or an optional process boundary.

        ``stop_at_step`` is deliberately a transient boundary for an outer
        train/inference chain.  It never changes the configured schedule or
        optimizer horizon, so a later ``resume=auto`` continues the same
        training run without resetting the scheduler.
        """
        import torch

        training = self.config["training"]
        accumulation = int(training.get("gradient_accumulation_steps", 1))
        if accumulation < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        max_epochs = training.get("max_epochs")
        max_steps = training.get("max_steps")
        if max_epochs is None and max_steps is None:
            raise ValueError("training requires max_epochs or max_steps")
        epoch_limit = int(max_epochs) if max_epochs is not None else 2**31 - 1
        step_limit = int(max_steps) if max_steps is not None else None
        if stop_at_step is not None:
            if isinstance(stop_at_step, bool) or not isinstance(stop_at_step, int):
                raise TypeError("stop_at_step must be an integer")
            if stop_at_step < 1:
                raise ValueError("stop_at_step must be positive")
            step_limit = (
                int(stop_at_step)
                if step_limit is None
                else min(step_limit, int(stop_at_step))
            )
        log_every = int(training.get("log_every", 10))
        validate_every = int(training.get("validate_every", 0))
        checkpoint_every = int(training.get("checkpoint_every", 0))
        max_grad_norm = training.get("max_grad_norm", 1.0)
        if step_limit is not None and int(self.state["global_step"]) >= step_limit:
            self.logger.info(
                "training target already reached at step=%d (max_steps=%d)",
                self.state["global_step"],
                step_limit,
            )
            return dict(self.state)
        distributed_mode = (
            self.context.mode if self.context.world_size > 1 else "single"
        )
        if accumulation > 1 and distributed_mode == "fsdp2":
            self.logger.info(
                "FSDP2 accumulation will reduce-scatter every microbatch to "
                "keep pending gradients sharded"
            )
        self.optimizer.zero_grad(set_to_none=True)
        stopped = False

        for epoch in range(int(self.state["epoch"]), epoch_limit):
            # Epoch-scoped seeds make iterator creation and worker RNG independent
            # of how many prior epochs a resumed process happened to construct.
            loader_seed = (
                int(self.config.get("seed", 0))
                + epoch * max(1, self.context.world_size)
                + self.context.rank
            )
            self.loader_generator.manual_seed(loader_seed)
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            pending = 0
            resume_batch = (
                int(self.state["batch_in_epoch"])
                if epoch == int(self.state["epoch"])
                else 0
            )
            for batch_index, batch in enumerate(self.train_loader):
                if batch_index < resume_batch:
                    continue
                is_last_batch = batch_index + 1 == len(self.train_loader)
                optimizer_boundary = pending + 1 == accumulation or is_last_batch
                with maybe_no_sync(
                    self.root,
                    distributed_mode,
                    synchronize=_requires_gradient_sync(
                        distributed_mode,
                        optimizer_boundary=optimizer_boundary,
                    ),
                ):
                    result = self._compute_loss(batch, training=True)
                    (result.loss / accumulation).backward()
                pending += 1
                if not optimizer_boundary:
                    continue
                if pending < accumulation:
                    scale = accumulation / pending
                    for parameter in self.parameters:
                        if parameter.grad is not None:
                            parameter.grad.mul_(scale)
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters, float(max_grad_norm)
                    )
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                pending = 0
                self.state["global_step"] += 1
                self.state["epoch"] = epoch
                self.state["batch_in_epoch"] = batch_index + 1
                if is_last_batch:
                    self.state["epoch"] = epoch + 1
                    self.state["batch_in_epoch"] = 0
                step = int(self.state["global_step"])
                if log_every and step % log_every == 0:
                    reduced = self.context.reduce_mean(result.loss.detach().float())
                    reduced_metrics = {
                        name: self.context.reduce_mean(value.detach().float().mean())
                        for name, value in sorted(result.metrics.items())
                        if hasattr(value, "detach") and name != "loss"
                    }
                    if self.context.is_main:
                        metric_text = " ".join(
                            f"{name}={float(value.item()):.6g}"
                            for name, value in reduced_metrics.items()
                        )
                        self.logger.info(
                            "train step=%d epoch=%d batch=%d loss=%.8f lr=%.3e %s",
                            step,
                            epoch,
                            batch_index,
                            reduced.item(),
                            self.optimizer.param_groups[0]["lr"],
                            metric_text,
                        )
                if validate_every and step % validate_every == 0:
                    self._validate()
                if checkpoint_every and step % checkpoint_every == 0:
                    checkpoint = self._checkpoint()
                    if self.context.is_main:
                        self.logger.info("saved checkpoint %s", checkpoint)
                if step_limit is not None and step >= step_limit:
                    stopped = True
                    break
            if stopped:
                break

        if int(self.state["global_step"]) < 1:
            raise RuntimeError("training completed without an optimizer step")
        existing = latest_checkpoint(self.run_dir)
        if existing is None or int(existing.name.removeprefix("step-")) != int(
            self.state["global_step"]
        ):
            self._checkpoint()
        return dict(self.state)
