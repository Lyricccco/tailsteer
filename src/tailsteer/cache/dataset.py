"""Compose native trajectory caches as storage-neutral training samples."""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .format import (
    STEP_RECORDS_LAYOUT,
    TRAJECTORY_GROUPED_LAYOUT,
    read_cache_records,
    require_mask_variant,
    validate_cache,
)
from .sample import CacheSample, normalize_metadata
from .tail_anchor_view import TailAnchorViewDataset
from .training_inputs import select_task_records


class CacheDataset:
    """Read the native v2-on-disk cache format."""

    cache_format = "native"

    def __init__(
        self,
        path: str | Path,
        *,
        steps: Sequence[int] | None = None,
        records: Sequence[dict[str, Any]] | None = None,
        include_terminal_state: bool = False,
        mask_variant: str = "default",
        tasks: Sequence[str] | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        manifest = validate_cache(self.path)
        self.storage_layout = str(manifest.get("storage_layout", STEP_RECORDS_LAYOUT))
        all_records = tuple(
            read_cache_records(self.path)
            if records is None or include_terminal_state
            else records
        )
        selected = all_records if records is None else tuple(records)
        selected = select_task_records(selected, tasks)
        if steps is not None:
            allowed = {int(step) for step in steps}
            selected = tuple(
                item for item in selected if int(item["step_index"]) in allowed
            )
        if not selected:
            raise ValueError(f"cache dataset has no selected records: {self.path}")
        self.records = selected
        self.mask_key = require_mask_variant(selected, mask_variant)
        self.mask_variant = mask_variant
        self.terminal_records: dict[tuple[str, int], dict[str, Any]] = {}
        if include_terminal_state:
            if self.storage_layout != TRAJECTORY_GROUPED_LAYOUT:
                raise ValueError(
                    "terminal targets require a grouped trajectory cache with "
                    f"a final state: {self.path}"
                )
            for record in all_records:
                if record.get("state_kind") != "final":
                    continue
                key = (str(record["shard"]), int(record["offset"]))
                if key in self.terminal_records:
                    raise ValueError(f"multiple cached terminal states: {key}")
                self.terminal_records[key] = record
            for record in selected:
                key = (str(record["shard"]), int(record["offset"]))
                if key not in self.terminal_records:
                    raise ValueError(
                        "terminal target requires a cached final state for "
                        f"every selected trajectory: {self.path} {key}"
                    )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> CacheSample:
        from safetensors import safe_open

        record = self.records[index]
        with safe_open(
            self.path / record["shard"], framework="pt", device="cpu"
        ) as handle:
            offset = int(record["offset"])
            reference = (
                handle.get_tensor("reference_latent")[offset]
                if record.get("has_reference")
                else None
            )
            latent = (
                handle.get_slice("trajectory_latents")[
                    offset, int(record["point_offset"])
                ]
                if self.storage_layout == TRAJECTORY_GROUPED_LAYOUT
                else handle.get_tensor("anchor_latent")[offset]
            )
            terminal_record = self.terminal_records.get((str(record["shard"]), offset))
            terminal = (
                handle.get_slice("trajectory_latents")[
                    offset, int(terminal_record["point_offset"])
                ]
                if terminal_record is not None
                else None
            )
            metadata = normalize_metadata(record)
            if self.mask_variant != "default":
                metadata["mask_variant"] = self.mask_variant
            return CacheSample(
                latent=latent,
                source_latent=handle.get_tensor("source_latent")[offset],
                keep_mask=handle.get_tensor(self.mask_key)[offset],
                reference_latent=reference,
                terminal_latent=terminal,
                metadata=metadata,
            )


def open_cache_dataset(
    path: str | Path,
    *,
    steps: Sequence[int] | None = None,
    cache_format: str = "native",
    shard_root: str | Path | None = None,
    expected_records: int | None = None,
    expected_task: str | None = None,
    verify_shard_hashes: bool = False,
    include_terminal_state: bool = False,
    mask_variant: str = "default",
    tasks: Sequence[str] | None = None,
) -> Any:
    """Open one explicitly identified cache format without heuristic conversion."""

    if cache_format == "native":
        return CacheDataset(
            path,
            steps=steps,
            include_terminal_state=include_terminal_state,
            mask_variant=mask_variant,
            tasks=tasks,
        )
    if cache_format == "tail_anchor_view":
        if mask_variant != "default" or tasks is not None:
            raise ValueError(
                "mask/task selection requires a native cache, not tail_anchor_view"
            )
        if include_terminal_state:
            raise ValueError(
                "terminal targets require a grouped trajectory cache; "
                "tail_anchor_view does not provide terminal states"
            )
        if shard_root is None:
            raise ValueError("tail_anchor_view cache requires shard_root")
        return TailAnchorViewDataset(
            path,
            shard_root=shard_root,
            steps=steps,
            expected_records=expected_records,
            expected_task=expected_task,
            verify_shard_hashes=verify_shard_hashes,
        )
    raise ValueError(f"unsupported cache format: {cache_format!r}")


def compose_cache_datasets(
    cache_configs: Sequence[dict[str, Any]],
    *,
    steps: Sequence[int] | None,
    strict_overlap: bool = False,
    check_overlap: bool = True,
    include_terminal_state: bool = False,
) -> Any:
    from torch.utils.data import ConcatDataset

    datasets = []
    owners: dict[str, str] = {}
    for config in cache_configs:
        dataset = open_cache_dataset(
            config["path"],
            steps=steps,
            cache_format=str(config.get("format", "native")),
            shard_root=config.get("shard_root"),
            expected_records=config.get("expected_records"),
            expected_task=config.get("expected_task"),
            verify_shard_hashes=bool(config.get("verify_shard_hashes", False)),
            include_terminal_state=include_terminal_state,
            mask_variant=config.get("mask_variant", "default"),
            tasks=config.get("tasks"),
        )
        current_path = str(dataset.path)
        if check_overlap:
            overlaps = sorted(
                {
                    str(record["source_identity"])
                    for record in dataset.records
                    if str(record["source_identity"]) in owners
                }
            )
            if overlaps:
                message = (
                    f"cache {current_path} overlaps {len(overlaps)} source "
                    f"identities; first={overlaps[0]!r} from "
                    f"{owners[overlaps[0]]}"
                )
                if strict_overlap:
                    raise ValueError(message)
                warnings.warn(message, stacklevel=2)
            for record in dataset.records:
                owners.setdefault(str(record["source_identity"]), current_path)
        datasets.extend([dataset] * int(config.get("repeat", 1)))
    if not datasets:
        raise ValueError("data.caches must be non-empty")
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def split_dataset(
    dataset: Any, *, seed: int, validation_fraction: float
) -> tuple[Any, Any]:
    from torch.utils.data import Subset

    from .split import source_partition

    records = _flatten_records(dataset)
    train, validation = [], []
    for index, record in enumerate(records):
        partition = source_partition(
            str(record["source_identity"]),
            seed=seed,
            validation_fraction=validation_fraction,
        )
        (validation if partition == "validation" else train).append(index)
    if not train:
        raise ValueError("source split produced no training records")
    return Subset(dataset, train), Subset(dataset, validation)


def _flatten_records(dataset: Any) -> list[dict[str, Any]]:
    if hasattr(dataset, "records"):
        return list(dataset.records)
    records = []
    for child in dataset.datasets:
        records.extend(child.records)
    return records


def cache_record(dataset: Any, index: int) -> dict[str, Any]:
    """Resolve metadata through Subset/ConcatDataset without opening a shard."""

    from torch.utils.data import ConcatDataset, Subset

    if isinstance(dataset, Subset):
        return cache_record(dataset.dataset, int(dataset.indices[index]))
    if isinstance(dataset, ConcatDataset):
        if index < 0:
            index += len(dataset)
        if index < 0 or index >= len(dataset):
            raise IndexError(index)
        child_index = 0
        while index >= dataset.cumulative_sizes[child_index]:
            child_index += 1
        previous = 0 if child_index == 0 else dataset.cumulative_sizes[child_index - 1]
        return cache_record(dataset.datasets[child_index], index - previous)
    if hasattr(dataset, "records"):
        return dict(dataset.records[index])
    raise TypeError(f"dataset {type(dataset).__name__} does not expose cache records")


def cache_collate(items: Sequence[CacheSample]) -> dict[str, Any]:
    import torch

    if not items:
        raise ValueError("cannot collate an empty cache batch")
    metadata = [dict(item.metadata) for item in items]
    result = {
        "latent": torch.stack([item.latent for item in items]),
        "source_latent": torch.stack([item.source_latent for item in items]),
        "keep_mask": torch.stack([item.keep_mask for item in items]),
        "metadata": metadata,
    }
    if all(item.reference_latent is not None for item in items):
        result["reference_latent"] = torch.stack(
            [item.reference_latent for item in items]
        )
    if any(item.terminal_latent is not None for item in items):
        if not all(item.terminal_latent is not None for item in items):
            raise ValueError(
                "cache batch mixes samples with and without terminal states"
            )
        result["terminal_latent"] = torch.stack(
            [item.terminal_latent for item in items]
        )
    return result


def source_counts(dataset: Any) -> Counter[str]:
    return Counter(str(item["source_identity"]) for item in _flatten_records(dataset))
