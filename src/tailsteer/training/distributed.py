"""Thin wrappers around official DDP and composable FSDP2 primitives."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any


@dataclass
class DistributedContext:
    mode: str
    rank: int
    local_rank: int
    world_size: int
    device: Any
    owns_process_group: bool = False
    checkpoint_group: Any | None = None

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.world_size > 1:
            import torch.distributed as dist

            options = (
                {"device_ids": [self.local_rank]}
                if self.device.type == "cuda" and dist.get_backend() == "nccl"
                else {}
            )
            dist.barrier(**options)

    def reduce_mean(self, value: Any) -> Any:
        if self.world_size > 1:
            import torch.distributed as dist

            value = value.clone()
            dist.all_reduce(value)
            value /= self.world_size
        return value

    def reduce_max_int(self, value: int) -> int:
        """Return the largest small integer held by any training rank."""

        if self.world_size == 1:
            return int(value)
        import torch
        import torch.distributed as dist

        tensor = torch.tensor(int(value), dtype=torch.int64, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return int(tensor.item())

    def broadcast_object(self, value: Any, *, source: int = 0) -> Any:
        if self.world_size == 1:
            return value
        import torch.distributed as dist

        values = [value]
        dist.broadcast_object_list(
            values,
            src=source,
            group=self.checkpoint_group,
        )
        return values[0]

    def gather_objects(self, value: Any) -> list[Any]:
        if self.world_size == 1:
            return [value]
        import torch.distributed as dist

        values = [None] * self.world_size
        dist.all_gather_object(values, value, group=self.checkpoint_group)
        return values

    def close(self) -> None:
        if self.checkpoint_group is not None:
            import torch.distributed as dist

            dist.destroy_process_group(self.checkpoint_group)
            self.checkpoint_group = None
        if self.owns_process_group:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()


def initialize_distributed(
    mode: str, *, timeout_seconds: int | None = None
) -> DistributedContext:
    import torch
    import torch.distributed as dist

    group_options = {}
    if timeout_seconds is not None:
        from datetime import timedelta

        if int(timeout_seconds) <= 0:
            raise ValueError("distributed timeout_seconds must be positive")
        group_options["timeout"] = timedelta(seconds=int(timeout_seconds))

    if mode not in {"single", "ddp", "fsdp2"}:
        raise ValueError("distributed.mode must be single, ddp, or fsdp2")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if mode == "single" and world_size != 1:
        raise ValueError("single mode cannot run under a multi-process torchrun")
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        if mode == "fsdp2":
            raise RuntimeError("FSDP2 full training requires CUDA")
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    owns = False
    checkpoint_group = None
    if mode in {"ddp", "fsdp2"} and world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            "nccl" if device.type == "cuda" else "gloo", **group_options
        )
        owns = True
    if mode in {"ddp", "fsdp2"} and world_size > 1:
        # DCP planning and CPU-offloaded state should not travel through the
        # numerical NCCL communicator. A failed CPU collective there can poison
        # later training collectives.
        checkpoint_group = dist.new_group(backend="gloo", **group_options)
    return DistributedContext(
        mode,
        rank,
        local_rank,
        world_size,
        device,
        owns,
        checkpoint_group,
    )


def _dtype(name: str | None) -> Any | None:
    import torch

    if name is None:
        return None
    normalized = str(name).removeprefix("torch.")
    if normalized not in {"float32", "float16", "bfloat16"}:
        raise ValueError(f"invalid distributed dtype {name!r}")
    return getattr(torch, normalized)


def prepare_model(
    model: Any,
    context: DistributedContext,
    options: dict[str, Any] | None = None,
) -> Any:
    """Wrap/shard the authoritative transformer before optimizer creation."""

    options = options or {}
    prepare_activation_checkpointing = getattr(
        model, "prepare_activation_checkpointing", None
    )
    defer_checkpointing = getattr(model, "defer_activation_checkpointing", None)
    deferred = callable(defer_checkpointing) and bool(defer_checkpointing())
    if callable(prepare_activation_checkpointing) and not deferred:
        prepare_activation_checkpointing()
    root = model.get_trainable_module()
    if context.mode == "single" or context.world_size == 1:
        if callable(prepare_activation_checkpointing) and deferred:
            prepare_activation_checkpointing()
        finalize = getattr(model, "prepare_optimizer_parameters", None)
        if callable(finalize):
            finalize()
        return root
    if context.mode == "ddp":
        if callable(prepare_activation_checkpointing) and deferred:
            prepare_activation_checkpointing()
        finalize = getattr(model, "prepare_optimizer_parameters", None)
        if callable(finalize):
            finalize()
        from torch.nn.parallel import DistributedDataParallel

        wrapped = DistributedDataParallel(
            root,
            device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        model.replace_trainable_module(wrapped)
        return wrapped

    from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("cuda", (context.world_size,), mesh_dim_names=("fsdp",))
    policy = MixedPrecisionPolicy(
        param_dtype=_dtype(options.get("param_dtype", "bfloat16")),
        reduce_dtype=_dtype(options.get("reduce_dtype", "float32")),
        output_dtype=_dtype(options.get("output_dtype")),
        cast_forward_inputs=bool(options.get("cast_forward_inputs", True)),
    )
    reshard = options.get("reshard_after_forward", True)
    prepare_unit = getattr(model, "prepare_sharding_unit", None)
    prepare_root = getattr(model, "prepare_sharding_root", None)
    for unit in model.sharding_units():
        if callable(prepare_unit):
            prepare_unit(unit)
        fully_shard(unit, mesh=mesh, reshard_after_forward=reshard, mp_policy=policy)
    if callable(prepare_activation_checkpointing) and deferred:
        # Checkpointing outside each FSDP unit makes recomputation re-enter the
        # unit's pre-forward unshard hook instead of seeing sharded DTensors.
        prepare_activation_checkpointing()
    if callable(prepare_root):
        prepare_root()
    wrapped = fully_shard(
        root, mesh=mesh, reshard_after_forward=reshard, mp_policy=policy
    )
    model.replace_trainable_module(wrapped)
    after_sharding = getattr(model, "after_sharding_root", None)
    if callable(after_sharding):
        after_sharding()
    auxiliary_groups = getattr(model, "auxiliary_sharding_groups", None)
    replace_auxiliary = getattr(model, "replace_auxiliary_sharded_module", None)
    if callable(auxiliary_groups):
        if not callable(replace_auxiliary):
            raise TypeError(
                "model exposes auxiliary sharding groups without a replacement hook"
            )
        # Some large backends materialize the frozen teacher lazily here. The
        # authoritative student is already sharded, preventing two replicated
        # actors and their FSDP conversion buffers from overlapping.
        for name, auxiliary_root, units in auxiliary_groups():
            for unit in units:
                fully_shard(
                    unit,
                    mesh=mesh,
                    reshard_after_forward=reshard,
                    mp_policy=policy,
                )
            auxiliary_wrapped = fully_shard(
                auxiliary_root,
                mesh=mesh,
                reshard_after_forward=reshard,
                mp_policy=policy,
            )
            replace_auxiliary(name, auxiliary_wrapped)
    finalize = getattr(model, "prepare_optimizer_parameters", None)
    if callable(finalize):
        finalize()
    return wrapped


@contextmanager
def maybe_no_sync(root: Any, mode: str, *, synchronize: bool) -> Iterator[None]:
    if synchronize:
        with nullcontext():
            yield
    elif mode == "ddp":
        with root.no_sync():
            yield
    elif mode == "fsdp2":
        setter = getattr(root, "set_requires_gradient_sync", None)
        if not callable(setter):
            raise TypeError("FSDP2 root lacks gradient synchronization control")
        setter(False, recurse=True)
        try:
            yield
        finally:
            setter(True, recurse=True)
    else:
        with nullcontext():
            yield
