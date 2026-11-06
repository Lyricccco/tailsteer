"""Collective DCP resume plus separate portable inference exports."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tailsteer.utils import capture_rng_state, restore_rng_state


@dataclass(frozen=True)
class CheckpointIOConfig:
    """Memory and writer policy for a sharded DCP checkpoint.

    ``cpu_offload=False`` keeps the state dict as references to the live local
    DTensor shards. ``FileSystemWriter`` then stages only a bounded amount of
    GPU data to CPU while writing instead of materializing every model and
    optimizer shard on host memory at once.
    """

    cpu_offload: bool = True
    writer_threads: int = 1
    per_thread_copy_ahead_bytes: int = 10_000_000

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> CheckpointIOConfig:
        raw = {} if value is None else dict(value)
        unknown = set(raw) - {
            "cpu_offload",
            "writer_threads",
            "per_thread_copy_ahead_bytes",
        }
        if unknown:
            raise ValueError(f"unknown checkpoint_io options: {sorted(unknown)}")
        cpu_offload = raw.get("cpu_offload", True)
        if not isinstance(cpu_offload, bool):
            raise TypeError("checkpoint_io.cpu_offload must be a boolean")

        def positive_integer(name: str, default: int) -> int:
            item = raw.get(name, default)
            if isinstance(item, bool) or not isinstance(item, int) or item < 1:
                raise ValueError(f"checkpoint_io.{name} must be a positive integer")
            return item

        return cls(
            cpu_offload=cpu_offload,
            writer_threads=positive_integer("writer_threads", 1),
            per_thread_copy_ahead_bytes=positive_integer(
                "per_thread_copy_ahead_bytes", 10_000_000
            ),
        )


def latest_checkpoint(root: str | Path) -> Path | None:
    checkpoints = Path(root).expanduser().resolve() / "checkpoints"
    complete = [
        path
        for path in checkpoints.glob("step-*")
        if path.is_dir() and (path / "COMPLETE").is_file()
    ]
    return max(
        complete, key=lambda path: int(path.name.removeprefix("step-")), default=None
    )


def resolve_resume(root: str | Path, resume: str | Path | None) -> Path | None:
    if resume in {None, "", "none"}:
        return None
    if str(resume) == "auto":
        return latest_checkpoint(root)
    path = Path(resume).expanduser().resolve()
    if not (path / "COMPLETE").is_file():
        raise RuntimeError(f"checkpoint is incomplete: {path}")
    return path


def _dcp_state(
    root: Any, optimizer: Any, *, cpu_offload: bool = True
) -> dict[str, Any]:
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

    model_state, optimizer_state = get_state_dict(
        root,
        optimizer,
        options=StateDictOptions(
            full_state_dict=False, cpu_offload=cpu_offload, strict=True
        ),
    )
    return {"model": model_state, "optimizer": optimizer_state}


def _initialize_missing_optimizer_state(optimizer: Any) -> None:
    """Create zero state only for parameters skipped by earlier lazy Adam steps."""

    import torch

    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    ]
    missing = [
        parameter for parameter in parameters if parameter not in optimizer.state
    ]
    if not missing:
        return
    missing_ids = {id(parameter) for parameter in missing}
    saved_grads = [parameter.grad for parameter in parameters]
    saved_lrs = [group.get("lr") for group in optimizer.param_groups]
    try:
        for parameter in parameters:
            parameter.grad = (
                torch.zeros_like(parameter) if id(parameter) in missing_ids else None
            )
        for group in optimizer.param_groups:
            if "lr" in group:
                group["lr"] = 0.0
        optimizer.step(closure=None)
        for parameter in missing:
            step = optimizer.state[parameter].get("step")
            if isinstance(step, torch.Tensor):
                step.zero_()
            elif step is not None:
                optimizer.state[parameter]["step"] = 0
    finally:
        for group, learning_rate in zip(optimizer.param_groups, saved_lrs, strict=True):
            if learning_rate is not None:
                group["lr"] = learning_rate
        for parameter, gradient in zip(parameters, saved_grads, strict=True):
            parameter.grad = gradient


def save_checkpoint(
    run_dir: str | Path,
    *,
    step: int,
    root: Any,
    optimizer: Any,
    scheduler: Any,
    trainer_state: dict[str, Any],
    context: Any,
    checkpoint_io: Mapping[str, Any] | None = None,
) -> Path:
    import torch
    import torch.distributed.checkpoint as dcp

    checkpoints = Path(run_dir).expanduser().resolve() / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    final = checkpoints / f"step-{step:08d}"
    if (final / "COMPLETE").is_file():
        return final
    building = checkpoints / f".step-{step:08d}.building"
    if context.is_main and building.exists():
        shutil.rmtree(building)
    context.barrier()
    building.mkdir(parents=True, exist_ok=True)
    io_config = CheckpointIOConfig.from_mapping(checkpoint_io)
    state = _dcp_state(root, optimizer, cpu_offload=io_config.cpu_offload)
    try:
        writer = dcp.FileSystemWriter(
            building / "dcp",
            single_file_per_rank=True,
            sync_files=True,
            thread_count=io_config.writer_threads,
            per_thread_copy_ahead=io_config.per_thread_copy_ahead_bytes,
            cache_staged_state_dict=False,
        )
        dcp.save(
            state,
            storage_writer=writer,
            process_group=context.checkpoint_group,
        )
    finally:
        # In CPU-offload mode this promptly releases the temporary state copies;
        # in bounded mode it only drops references to the live DTensor shards.
        del state
    process_state = {
        "schema_version": 1,
        "trainer": trainer_state,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng": capture_rng_state(),
    }
    torch.save(process_state, building / f"state-rank-{context.rank:05d}.pt")
    context.barrier()
    if context.is_main:
        torch.save(process_state, building / "state.pt")
        (building / "COMPLETE").write_text("complete\n", encoding="utf-8")
        if final.exists():
            shutil.rmtree(final)
        os.replace(building, final)
    context.barrier()
    return final


def load_checkpoint(
    checkpoint: str | Path,
    *,
    root: Any,
    optimizer: Any,
    scheduler: Any,
    context: Any,
    checkpoint_io: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_state_dict,
        set_state_dict,
    )

    source = Path(checkpoint).expanduser().resolve()
    io_config = CheckpointIOConfig.from_mapping(checkpoint_io)
    options = StateDictOptions(
        full_state_dict=False,
        cpu_offload=io_config.cpu_offload,
        strict=True,
    )
    _initialize_missing_optimizer_state(optimizer)
    model_state, optimizer_state = get_state_dict(root, optimizer, options=options)
    optimizer_defaults = {
        name: dict(values) for name, values in optimizer_state.get("state", {}).items()
    }
    checkpoint_id = str(source / "dcp")
    # Model weights remain strict. AdamW initializes every lazy state entry when
    # a fresh optimizer is prepared for loading, while a checkpoint legitimately
    # omits entries for parameters that received no gradient before the save.
    # Load that optimizer subtree partially so those entries retain their zero
    # initialization without weakening model-key validation.
    model_container = {"model": model_state}
    dcp.load(
        model_container,
        checkpoint_id=checkpoint_id,
        process_group=context.checkpoint_group,
    )
    model_state = model_container["model"]
    optimizer_container = {"optimizer": optimizer_state}
    dcp.load(
        optimizer_container,
        checkpoint_id=checkpoint_id,
        planner=dcp.DefaultLoadPlanner(allow_partial_load=True),
        process_group=context.checkpoint_group,
    )
    optimizer_state = optimizer_container["optimizer"]
    loaded_optimizer_state = optimizer_state.setdefault("state", {})
    for name, default_values in optimizer_defaults.items():
        loaded_values = loaded_optimizer_state.setdefault(name, {})
        for state_name, default_value in default_values.items():
            loaded_values.setdefault(state_name, default_value)
    set_state_dict(
        root,
        optimizer,
        model_state_dict=model_state,
        optim_state_dict=optimizer_state,
        options=options,
    )
    process_state = torch.load(
        source / f"state-rank-{context.rank:05d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    if scheduler is not None and process_state["scheduler"] is not None:
        scheduler.load_state_dict(process_state["scheduler"])
    restore_rng_state(process_state["rng"])
    return dict(process_state["trainer"])


def load_model_checkpoint(
    checkpoint: str | Path,
    *,
    root: Any,
    context: Any,
    checkpoint_io: Mapping[str, Any] | None = None,
) -> None:
    """Load only model weights from a DCP checkpoint.

    Portable inference export does not need the training optimizer, scheduler,
    RNG state, or trainer progress.  Keeping this path separate from
    :func:`load_checkpoint` prevents an exporter from materializing the large
    Adam state of a full-model run.
    """

    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        set_model_state_dict,
    )

    source = Path(checkpoint).expanduser().resolve()
    if not (source / "COMPLETE").is_file():
        raise RuntimeError(f"checkpoint is incomplete: {source}")
    io_config = CheckpointIOConfig.from_mapping(checkpoint_io)
    options = StateDictOptions(
        full_state_dict=False,
        cpu_offload=io_config.cpu_offload,
        strict=True,
    )
    model_state = get_model_state_dict(root, options=options)
    container = {"model": model_state}
    dcp.load(
        container,
        checkpoint_id=str(source / "dcp"),
        process_group=context.checkpoint_group,
    )
    incompatible = set_model_state_dict(
        root,
        container["model"],
        options=options,
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "model-only checkpoint load is incompatible: "
            f"missing={incompatible.missing_keys[:3]} "
            f"unexpected={incompatible.unexpected_keys[:3]}"
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _unwrap(module: Any) -> Any:
    while hasattr(module, "module"):
        module = module.module
    return module


def _gather_fsdp2_model_state_dict(
    root: Any, context: Any, *, tensor_dtype: Any | None = None
) -> dict[str, Any]:
    """Gather an FSDP2 state dict without the unsupported coalesced collective.

    PyTorch 2.5's ``get_model_state_dict(full_state_dict=True)`` batches DTensor
    redistributions through ``allgather_into_tensor_coalesced``.  That collective
    is not implemented by the NCCL backend shipped in the training environment.
    Gathering one DTensor at a time uses the supported ``all_gather_into_tensor``
    path and also bounds temporary GPU memory to one parameter.
    """

    import torch
    from torch.distributed.tensor import DTensor

    sharded = root.state_dict()
    gathered: dict[str, Any] = {}
    total = len(sharded)
    if context.is_main:
        print(f"inference export: gathering {total} FSDP2 tensors", flush=True)
    with torch.no_grad():
        for index, (name, value) in enumerate(sharded.items(), start=1):
            if isinstance(value, DTensor):
                full = value.full_tensor()
            elif isinstance(value, torch.Tensor):
                full = value
            else:
                raise TypeError(
                    f"unsupported FSDP2 state value for {name!r}: {type(value)!r}"
                )
            if context.is_main:
                gathered[name] = (
                    full.detach()
                    .to(device="cpu", dtype=tensor_dtype, copy=True)
                    .contiguous()
                )
            del full
            if context.is_main and (index % 100 == 0 or index == total):
                print(f"inference export: gathered {index}/{total} tensors", flush=True)
    return gathered


def export_inference_model(
    run_dir: str | Path,
    *,
    name: str,
    model: Any,
    root: Any,
    step: int,
    context: Any,
    tensor_dtype: Any | None = None,
) -> Path:
    # A backend may train with FP32 masters while retaining a smaller native
    # inference format.  Resolve this before gathering an FSDP state dict so a
    # 9B FP32 master is never materialized as a full FP32 CPU export.
    export_dtype = tensor_dtype
    dtype_hook = getattr(model, "inference_export_dtype", None)
    if export_dtype is None and callable(dtype_hook):
        export_dtype = dtype_hook()
    if context.mode == "fsdp2" and context.world_size > 1:
        state = _gather_fsdp2_model_state_dict(root, context, tensor_dtype=export_dtype)
    else:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )

        state = get_model_state_dict(
            root,
            options=StateDictOptions(
                full_state_dict=True, cpu_offload=True, strict=True
            ),
        )
    destination = Path(run_dir).expanduser().resolve() / "exports" / name
    if context.is_main:
        from safetensors.torch import save_file

        building = destination.with_name(f".{destination.name}.building")
        if building.exists():
            shutil.rmtree(building)
        building.mkdir(parents=True, exist_ok=True)
        policy = model.trainable_policy
        prepare = getattr(model, "prepare_inference_export", None)
        if callable(prepare):
            state, weights, manifest = prepare(state, step=step)
        elif policy == "lora":
            state = {key: value for key, value in state.items() if "lora_" in key}
            peft_config = next(
                iter(_unwrap(model.get_trainable_module()).peft_config.values())
            )
            lora_config = _jsonable(peft_config.to_dict())
            weights = "adapter_model.safetensors"
            manifest = {
                "schema_version": 1,
                "model": "flux2_klein_9b",
                "step": int(step),
                "trainable_policy": policy,
                "weights": weights,
                "lora_config": lora_config,
            }
        else:
            lora_config = None
            weights = "transformer.safetensors"
            manifest = {
                "schema_version": 1,
                "model": "flux2_klein_9b",
                "step": int(step),
                "trainable_policy": policy,
                "weights": weights,
                "lora_config": lora_config,
            }
        if not state:
            raise RuntimeError("portable inference export has no tensors")
        save_file(
            {
                key: value.detach()
                .to(device="cpu", dtype=export_dtype, copy=True)
                .contiguous()
                for key, value in state.items()
            },
            building / weights,
        )
        (building / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        (building / "COMPLETE").write_text("complete\n", encoding="utf-8")
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(building, destination)
    context.barrier()
    return destination
