"""Segment-level views over grouped trajectory caches.

One logical item reads only the state slices required by the selected segment.
This is important for packed-pixel caches where a complete trajectory can be
close to one hundred megabytes per source.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from .format import (
    TRAJECTORY_GROUPED_LAYOUT,
    mask_tensor_key,
    read_cache_records,
    require_mask_variant,
    validate_cache,
)
from .sample import normalize_metadata
from .training_inputs import select_task_records


@dataclass(frozen=True)
class CachedTrajectoryPoint:
    step_index: int
    sigma: float
    timestep: float
    state_kind: str


@dataclass(frozen=True)
class TrajectoryContract:
    model: str
    representation: str
    points: tuple[CachedTrajectoryPoint, ...]
    num_steps: int
    height: int
    width: int

    def point(self, step_index: int) -> CachedTrajectoryPoint:
        for point in self.points:
            if point.step_index == int(step_index):
                return point
        raise KeyError(f"trajectory contract has no step {step_index}")

    @property
    def forward_points(self) -> tuple[CachedTrajectoryPoint, ...]:
        return tuple(point for point in self.points if point.state_kind != "final")

    @property
    def terminal_point(self) -> CachedTrajectoryPoint:
        final = tuple(point for point in self.points if point.state_kind == "final")
        if len(final) != 1:
            raise ValueError("trajectory contract must have exactly one final point")
        return final[0]

    def compatible_with(self, other: TrajectoryContract) -> bool:
        return (
            self.model,
            self.representation,
            self.points,
            self.num_steps,
            self.height,
            self.width,
        ) == (
            other.model,
            other.representation,
            other.points,
            other.num_steps,
            other.height,
            other.width,
        )


@dataclass(frozen=True)
class TrajectorySegmentSample:
    left_latent: Any
    right_latent: Any
    source_latent: Any
    keep_mask: Any
    left_metadata: Mapping[str, Any]
    right_metadata: Mapping[str, Any]
    metadata: Mapping[str, Any]
    is_endpoint: bool
    reference_latent: Any | None = None


@dataclass(frozen=True)
class TrajectorySourceSample:
    """A lightweight locator for one complete cached trajectory."""

    path: Path
    shard: str
    offset: int
    points: Mapping[int, Mapping[str, Any]]
    metadata: Mapping[str, Any]
    mask_variant: str = "default"


@dataclass(frozen=True)
class CachedPathReference:
    """Cached states needed to evaluate a fixed path target.

    The tensors keep their leading batch dimension so that the reference can
    be passed directly alongside a :class:`PreparedTrajectoryBatch`.  The
    resolver currently supports one source trajectory per item, matching the
    batch-size-one contract of teacher-guided training.
    """

    current_state: Any
    following_state: Any
    terminal_state: Any
    current_point: Any
    following_point: Any


class CachedPathReferenceResolver:
    """Resolve model-schedule points from one grouped trajectory cache.

    Cache points are sparse, while the live inference schedule is dense.  Each
    requested point is therefore read exactly when it is needed and linearly
    interpolated in the cache's decreasing-sigma coordinate.  This keeps cache
    IO and interpolation rules in one model-neutral place for C and D.
    """

    def __init__(self, batch: Mapping[str, Any]) -> None:
        sources = batch["trajectory_sources"]
        if len(sources) != 1:
            raise ValueError("cached path references currently require batch_size=1")
        self.source = sources[0]
        if not self.source.points:
            raise ValueError("trajectory source has no cached points")
        terminal_steps = [
            int(step)
            for step, record in self.source.points.items()
            if str(record.get("state_kind", "forward_input")) == "final"
        ]
        if len(terminal_steps) != 1:
            raise ValueError(
                "cached path references require exactly one cached terminal point"
            )
        self.terminal_step = terminal_steps[0]

    def resolve(self, current_point: Any, following_point: Any) -> CachedPathReference:
        from safetensors import safe_open

        if float(following_point.step) <= float(current_point.step):
            raise ValueError("cached path following point must be after current point")
        with safe_open(
            self.source.path / self.source.shard, framework="pt", device="cpu"
        ) as handle:
            current = _materialize_source_state(handle, self.source, current_point)
            following = _materialize_source_state(handle, self.source, following_point)
            terminal_point = self.source.points[self.terminal_step]
            terminal = _materialize_source_state(
                handle,
                self.source,
                _metadata_point(terminal_point, self.terminal_step),
            )
        import torch

        return CachedPathReference(
            current_state=torch.stack([current]),
            following_state=torch.stack([following]),
            terminal_state=torch.stack([terminal]),
            current_point=current_point,
            following_point=following_point,
        )


def _contract(
    manifest: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> TrajectoryContract:
    by_step: dict[int, set[tuple[float, float, str]]] = {}
    for record in records:
        by_step.setdefault(int(record["step_index"]), set()).add(
            (
                float(record["sigma"]),
                float(record["timestep"]),
                str(record.get("state_kind", "forward_input")),
            )
        )
    ambiguous = {step: values for step, values in by_step.items() if len(values) != 1}
    if ambiguous:
        raise ValueError(f"cache has inconsistent point metadata: {ambiguous}")
    points = tuple(
        CachedTrajectoryPoint(step, *next(iter(by_step[step])))
        for step in sorted(by_step)
    )
    first = records[0]
    creation = manifest.get("creation_config", {})
    cache_config = creation.get("cache", {}) if isinstance(creation, Mapping) else {}
    num_steps = int(cache_config.get("num_steps", len(points) - 1))
    return TrajectoryContract(
        model=str(manifest.get("model", "")),
        representation=str(
            manifest.get("representation", first.get("state_representation", "native"))
        ),
        points=points,
        num_steps=num_steps,
        height=int(first["height"]),
        width=int(first["width"]),
    )


def read_trajectory_contract(path: str | Path) -> TrajectoryContract:
    root = Path(path).expanduser().resolve()
    manifest = validate_cache(root)
    if str(manifest.get("storage_layout")) != TRAJECTORY_GROUPED_LAYOUT:
        raise ValueError(
            f"teacher-guided training requires grouped trajectories: {root}"
        )
    records = read_cache_records(root)
    return _contract(manifest, records)


def _metadata_point(record: Mapping[str, Any], step: int) -> CachedTrajectoryPoint:
    return CachedTrajectoryPoint(
        int(step),
        float(record["sigma"]),
        float(record["timestep"]),
        str(record.get("state_kind", "forward_input")),
    )


class TrajectorySegmentDataset:
    """Expose adjacent cached states and an explicit supervision endpoint."""

    cache_format = "native_trajectory_segments"

    def __init__(
        self,
        path: str | Path,
        *,
        supervision_step: int,
        endpoint_repeat: int = 1,
        mask_variant: str = "default",
        tasks: Sequence[str] | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        manifest = validate_cache(self.path)
        if str(manifest.get("storage_layout")) != TRAJECTORY_GROUPED_LAYOUT:
            raise ValueError(
                "trajectory segment dataset requires trajectory_grouped_v2"
            )
        if endpoint_repeat < 1:
            raise ValueError("endpoint_repeat must be positive")
        raw_records = select_task_records(read_cache_records(self.path), tasks)
        self.mask_key = require_mask_variant(raw_records, mask_variant)
        self.contract = _contract(manifest, raw_records)
        endpoint = self.contract.point(int(supervision_step))
        if endpoint.state_kind == "final":
            raise ValueError("supervision_step must be a model forward, not terminal")

        grouped: dict[tuple[str, int], dict[int, dict[str, Any]]] = {}
        for record in raw_records:
            key = (str(record["shard"]), int(record["offset"]))
            grouped.setdefault(key, {})[int(record["step_index"])] = dict(record)
        self._groups = tuple((key, records) for key, records in sorted(grouped.items()))
        forward_steps = tuple(
            point.step_index
            for point in self.contract.forward_points
            if point.step_index <= int(supervision_step)
        )
        if not forward_steps or forward_steps[-1] != int(supervision_step):
            raise ValueError("supervision_step is not available in every trajectory")
        pairs = tuple(pairwise(forward_steps))
        entries: list[tuple[int, int, int, bool]] = []
        public_records: list[dict[str, Any]] = []
        for group_index, (_key, records) in enumerate(self._groups):
            if any(step not in records for step in forward_steps):
                raise ValueError("grouped trajectory lacks a required forward point")
            for left, right in pairs:
                entries.append((group_index, left, right, False))
                public_records.append(
                    self._public_record(records[right], left=left, right=right)
                )
            for repeat_index in range(endpoint_repeat):
                entries.append(
                    (group_index, int(supervision_step), int(supervision_step), True)
                )
                public_record = self._public_record(
                    records[int(supervision_step)],
                    left=int(supervision_step),
                    right=int(supervision_step),
                )
                if endpoint_repeat > 1:
                    public_record["sample_id"] += f":repeat-{repeat_index}"
                public_records.append(public_record)
        self._entries = tuple(entries)
        self.records = tuple(public_records)

    @staticmethod
    def _public_record(
        record: Mapping[str, Any], *, left: int, right: int
    ) -> dict[str, Any]:
        value = normalize_metadata(record)
        return {
            **value,
            "sample_id": f"{value['source_id']}:segment-{left}-{right}",
            "left_step": int(left),
            "right_step": int(right),
        }

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> TrajectorySegmentSample:
        from safetensors import safe_open

        group_index, left_step, right_step, is_endpoint = self._entries[index]
        (shard_name, offset), records = self._groups[group_index]
        left_record, right_record = records[left_step], records[right_step]
        with safe_open(self.path / shard_name, framework="pt", device="cpu") as handle:
            states = handle.get_slice("trajectory_latents")
            left = states[offset, int(left_record["point_offset"])]
            right = (
                left
                if is_endpoint
                else states[offset, int(right_record["point_offset"])]
            )
            reference = (
                handle.get_tensor("reference_latent")[offset]
                if right_record.get("has_reference")
                else None
            )
            return TrajectorySegmentSample(
                left_latent=left,
                right_latent=right,
                source_latent=handle.get_tensor("source_latent")[offset],
                keep_mask=handle.get_tensor(self.mask_key)[offset],
                reference_latent=reference,
                left_metadata=normalize_metadata(left_record),
                right_metadata=normalize_metadata(right_record),
                metadata=self.records[index],
                is_endpoint=is_endpoint,
            )


class TrajectorySourceDataset:
    """Expose one lightweight item per source trajectory.

    The algorithm samples a target inference step before reading its bracketing
    states, so cache point density cannot define the training distribution.
    """

    cache_format = "native_trajectory_sources"

    def __init__(
        self,
        path: str | Path,
        *,
        supervision_step: int,
        mask_variant: str = "default",
        tasks: Sequence[str] | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        manifest = validate_cache(self.path)
        if str(manifest.get("storage_layout")) != TRAJECTORY_GROUPED_LAYOUT:
            raise ValueError("trajectory source dataset requires trajectory_grouped_v2")
        raw_records = select_task_records(read_cache_records(self.path), tasks)
        require_mask_variant(raw_records, mask_variant)
        self.mask_variant = mask_variant
        self.contract = _contract(manifest, raw_records)
        endpoint = self.contract.point(int(supervision_step))
        if endpoint.state_kind == "final":
            raise ValueError("supervision_step must be a model forward, not terminal")
        forward_steps = tuple(
            point.step_index
            for point in self.contract.forward_points
            if point.step_index <= int(supervision_step)
        )
        if not forward_steps or forward_steps[-1] != int(supervision_step):
            raise ValueError("supervision_step is not available in the trajectory")
        # Keep the terminal state in each source locator.  C + cached_path
        # needs the endpoint to construct its correction, while the legacy
        # terminal_bridge target continues to read only forward states.
        required_steps = forward_steps + (self.contract.terminal_point.step_index,)

        grouped: dict[tuple[str, int], dict[int, dict[str, Any]]] = {}
        for record in raw_records:
            key = (str(record["shard"]), int(record["offset"]))
            grouped.setdefault(key, {})[int(record["step_index"])] = dict(record)
        groups = []
        public_records = []
        for (shard, offset), records in sorted(grouped.items()):
            if any(step not in records for step in required_steps):
                raise ValueError("grouped trajectory lacks a required forward point")
            selected = {step: records[step] for step in required_steps}
            metadata = normalize_metadata(records[int(supervision_step)])
            metadata = {
                **metadata,
                "sample_id": f"{metadata['source_id']}:trajectory-source",
            }
            if mask_variant != "default":
                metadata["mask_variant"] = mask_variant
            groups.append((shard, offset, selected, metadata))
            public_records.append(dict(metadata))
        self._groups = tuple(groups)
        self.records = tuple(public_records)

    def __len__(self) -> int:
        return len(self._groups)

    def __getitem__(self, index: int) -> TrajectorySourceSample:
        shard, offset, points, metadata = self._groups[index]
        return TrajectorySourceSample(
            path=self.path,
            shard=shard,
            offset=offset,
            points={
                step: normalize_metadata(record) for step, record in points.items()
            },
            metadata=metadata,
            mask_variant=self.mask_variant,
        )


def trajectory_source_collate(
    items: Sequence[TrajectorySourceSample],
) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty trajectory source batch")
    return {
        "trajectory_sources": list(items),
        "metadata": [dict(item.metadata) for item in items],
    }


def materialize_trajectory_state(
    batch: Mapping[str, Any],
    point: Any,
    *,
    state_key: str = "latent",
) -> dict[str, Any]:
    """Read and interpolate the cached trajectory state at ``point``."""

    import torch
    from safetensors import safe_open

    sources = batch["trajectory_sources"]
    if len(sources) != 1:
        raise ValueError("trajectory materialization currently requires batch_size=1")
    source = sources[0]
    with safe_open(source.path / source.shard, framework="pt", device="cpu") as handle:
        state, interpolation, left_step, right_step = _materialize_source_state(
            handle, source, point, return_details=True
        )
        source_latent = handle.get_tensor("source_latent")[source.offset]
        keep_mask = handle.get_tensor(mask_tensor_key(source.mask_variant))[
            source.offset
        ]

    target_step = float(point.step)
    metadata = {
        **source.metadata,
        "step_index": target_step,
        "sigma": float(point.sigma),
        "timestep": float(point.timestep),
        "state_kind": "interpolated_forward_input",
        "cache_left_step": left_step,
        "cache_right_step": right_step,
        "cache_interpolation": interpolation,
    }
    return {
        state_key: torch.stack([state]),
        "source_latent": torch.stack([source_latent]),
        "keep_mask": torch.stack([keep_mask]),
        "metadata": [metadata],
    }


def _materialize_source_state(
    handle: Any,
    source: TrajectorySourceSample,
    point: Any,
    *,
    return_details: bool = False,
) -> Any:
    """Materialize one source state from an already-open shard handle."""

    raw_step = getattr(point, "step", None)
    if raw_step is None:
        raw_step = point.step_index
    target_step = float(raw_step)
    ordered_steps = sorted(int(step) for step in source.points)
    if not ordered_steps[0] <= target_step <= ordered_steps[-1]:
        raise ValueError(
            f"target step {target_step} is outside cached support "
            f"[{ordered_steps[0]}, {ordered_steps[-1]}]"
        )
    right_index = next(
        index for index, step in enumerate(ordered_steps) if step >= target_step
    )
    right_step = ordered_steps[right_index]
    left_step = (
        right_step if right_step == target_step else ordered_steps[right_index - 1]
    )
    left_record = source.points[left_step]
    right_record = source.points[right_step]
    states = handle.get_slice("trajectory_latents")
    left = states[source.offset, int(left_record["point_offset"])]
    if left_step == right_step:
        state = left
        interpolation = 0.0
    else:
        right = states[source.offset, int(right_record["point_offset"])]
        left_sigma = float(left_record["sigma"])
        right_sigma = float(right_record["sigma"])
        denominator = left_sigma - right_sigma
        if denominator <= 0:
            raise ValueError("cached trajectory sigma must decrease with step")
        interpolation = (left_sigma - float(point.sigma)) / denominator
        if not -1e-5 <= interpolation <= 1.0 + 1e-5:
            raise ValueError(
                f"point sigma {point.sigma} is outside cache bracket "
                f"[{right_sigma}, {left_sigma}]"
            )
        interpolation = min(max(interpolation, 0.0), 1.0)
        state = (left.float() + interpolation * (right.float() - left.float())).to(
            dtype=left.dtype
        )
    if return_details:
        return state, interpolation, left_step, right_step
    return state


def trajectory_segment_collate(
    items: Sequence[TrajectorySegmentSample],
) -> dict[str, Any]:
    import torch

    if not items:
        raise ValueError("cannot collate an empty trajectory segment batch")
    result = {
        "left_latent": torch.stack([item.left_latent for item in items]),
        "right_latent": torch.stack([item.right_latent for item in items]),
        "source_latent": torch.stack([item.source_latent for item in items]),
        "keep_mask": torch.stack([item.keep_mask for item in items]),
        "left_metadata": [dict(item.left_metadata) for item in items],
        "right_metadata": [dict(item.right_metadata) for item in items],
        "metadata": [dict(item.metadata) for item in items],
        "is_endpoint": torch.tensor([item.is_endpoint for item in items]),
    }
    if all(item.reference_latent is not None for item in items):
        result["reference_latent"] = torch.stack(
            [item.reference_latent for item in items]
        )
    return result


def compose_trajectory_segment_datasets(
    cache_configs: Sequence[Mapping[str, Any]],
    *,
    supervision_step: int,
    endpoint_repeat: int = 1,
) -> tuple[Any, TrajectoryContract]:
    from torch.utils.data import ConcatDataset

    datasets = []
    contract = None
    for config in cache_configs:
        dataset = TrajectorySegmentDataset(
            config["path"],
            supervision_step=supervision_step,
            endpoint_repeat=int(config.get("endpoint_repeat", endpoint_repeat)),
            mask_variant=config.get("mask_variant", "default"),
            tasks=config.get("tasks"),
        )
        if contract is None:
            contract = dataset.contract
        elif not contract.compatible_with(dataset.contract):
            raise ValueError(
                "trajectory caches have incompatible model/schedule contracts"
            )
        datasets.extend([dataset] * int(config.get("repeat", 1)))
    if not datasets or contract is None:
        raise ValueError("data.caches must be non-empty")
    combined = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    return combined, contract


def compose_trajectory_source_datasets(
    cache_configs: Sequence[Mapping[str, Any]],
    *,
    supervision_step: int,
) -> tuple[Any, TrajectoryContract]:
    from torch.utils.data import ConcatDataset

    datasets = []
    contract = None
    for config in cache_configs:
        dataset = TrajectorySourceDataset(
            config["path"],
            supervision_step=supervision_step,
            mask_variant=config.get("mask_variant", "default"),
            tasks=config.get("tasks"),
        )
        if contract is None:
            contract = dataset.contract
        elif not contract.compatible_with(dataset.contract):
            raise ValueError(
                "trajectory caches have incompatible model/schedule contracts"
            )
        datasets.extend([dataset] * int(config.get("repeat", 1)))
    if not datasets or contract is None:
        raise ValueError("data.caches must be non-empty")
    combined = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    return combined, contract
