"""Human-readable sidecars around shape-homogeneous safetensors shards."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STEP_RECORDS_LAYOUT = "step_records_v1"
TRAJECTORY_GROUPED_LAYOUT = "trajectory_grouped_v2"
POINT_EXTENSION_MARKER = ".extend-points.json"
MASK_TENSORS = {
    "default": "keep_mask",
    "fine": "keep_mask_fine",
    "dilate5": "keep_mask_dilate5",
}


def mask_tensor_key(variant: str) -> str:
    if variant not in MASK_TENSORS:
        raise ValueError(f"unknown cache mask_variant: {variant!r}")
    return MASK_TENSORS[variant]


def require_mask_variant(records, variant: str) -> str:
    key = mask_tensor_key(variant)
    if variant != "default" and any(
        variant not in record.get("mask_variants", []) for record in records
    ):
        raise ValueError(f"cache does not contain requested mask_variant: {variant}")
    return key


@dataclass(frozen=True)
class CacheExample:
    anchor_latent: Any
    source_latent: Any
    keep_mask: Any
    metadata: Mapping[str, Any]
    reference_latent: Any | None = None
    mask_variants: Mapping[str, Any] | None = None


def _mask_variant_tensors(
    examples: Sequence[CacheExample], *, grouped: bool
) -> dict[str, Any]:
    import torch

    first = examples[0].mask_variants or {}
    if set(first) - {"fine", "dilate5"}:
        raise ValueError("unsupported extra cache mask variants")
    for item in examples:
        masks = item.mask_variants or {}
        if set(masks) != set(first):
            raise ValueError("cache shard mixes mask variants")
        for name, mask in masks.items():
            # Kontext is already on the VAE grid: fine shares feature channels.
            expected = (
                tuple(item.anchor_latent.shape)
                if (
                    name == "fine"
                    and item.metadata.get("state_representation")
                    != "flux1_unpacked_vae_latent"
                )
                else tuple(item.keep_mask.shape)
            )
            if tuple(mask.shape) != expected or mask.dtype != torch.uint8:
                raise ValueError(f"invalid {name} mask shape/dtype")
            if (
                grouped
                and mask is not first[name]
                and not torch.equal(mask, first[name])
            ):
                raise ValueError(f"{name} mask differs inside one trajectory")
    return {
        mask_tensor_key(name): (
            mask.unsqueeze(0)
            if grouped
            else torch.stack([item.mask_variants[name] for item in examples])
        )
        for name, mask in first.items()
    }


def _mask_variant_metadata(example: CacheExample) -> dict[str, Any]:
    return (
        {"mask_variants": sorted(example.mask_variants)}
        if example.mask_variants
        else {}
    )


class CacheWriter:
    def __init__(self, directory: str | Path, *, rank: int = 0) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        rank_floor = rank * 1_000_000
        existing = [
            int(path.stem.rsplit("-", 1)[1])
            for path in self.directory.glob("shard-*.safetensors")
            if int(path.stem.rsplit("-", 1)[1]) // 1_000_000 == rank
        ]
        self.next_index = max(existing, default=rank_floor - 1) + 1

    def write(
        self, examples: Sequence[CacheExample]
    ) -> tuple[Path, Path, list[dict[str, Any]]]:
        import torch

        if not examples:
            raise ValueError("cannot write an empty cache shard")
        first = examples[0]
        signature = (
            tuple(first.anchor_latent.shape),
            tuple(first.source_latent.shape),
            tuple(first.keep_mask.shape),
            first.reference_latent is not None,
        )
        if any(
            (
                tuple(item.anchor_latent.shape),
                tuple(item.source_latent.shape),
                tuple(item.keep_mask.shape),
                item.reference_latent is not None,
            )
            != signature
            for item in examples
        ):
            raise ValueError("cache shard mixes tensor shapes")
        tensors = {
            "anchor_latent": torch.stack([item.anchor_latent for item in examples]),
            "source_latent": torch.stack([item.source_latent for item in examples]),
            "keep_mask": torch.stack([item.keep_mask for item in examples]),
            **_mask_variant_tensors(examples, grouped=False),
        }
        if first.reference_latent is not None:
            tensors["reference_latent"] = torch.stack(
                [item.reference_latent for item in examples]
            )
        shard_name = self._next_shard_name()
        records = [
            {
                **dict(item.metadata),
                **_mask_variant_metadata(item),
                "shard": shard_name,
                "offset": offset,
                "has_reference": item.reference_latent is not None,
            }
            for offset, item in enumerate(examples)
        ]
        return self._commit(shard_name, tensors, records)

    def _next_shard_name(self) -> str:
        index = self.next_index
        self.next_index += 1
        return f"shard-{index:07d}.safetensors"

    def _commit(
        self,
        shard_name: str,
        tensors: Mapping[str, Any],
        records: Sequence[Mapping[str, Any]],
    ) -> tuple[Path, Path, list[dict[str, Any]]]:
        from safetensors.torch import save_file

        metadata_name = shard_name.replace("shard-", "metadata-").replace(
            ".safetensors", ".jsonl"
        )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{shard_name}.", suffix=".tmp", dir=self.directory
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        shard = self.directory / shard_name
        sidecar = self.directory / metadata_name
        resolved_records = [dict(record) for record in records]
        temporary_sidecar = sidecar.with_name(f".{sidecar.name}.tmp-{os.getpid()}")
        try:
            save_file(
                {name: value.contiguous() for name, value in tensors.items()},
                temporary,
            )
            os.replace(temporary, shard)
            content = "".join(
                json.dumps(item, sort_keys=True) + "\n" for item in resolved_records
            )
            temporary_sidecar.write_text(content, encoding="utf-8")
            os.replace(temporary_sidecar, sidecar)
        except BaseException:
            temporary.unlink(missing_ok=True)
            temporary_sidecar.unlink(missing_ok=True)
            raise
        return shard, sidecar, resolved_records


class TrajectoryCacheWriter(CacheWriter):
    """Store one source trajectory without repeating source tensors per point."""

    def write(
        self, examples: Sequence[CacheExample]
    ) -> tuple[Path, Path, list[dict[str, Any]]]:
        import torch

        if not examples:
            raise ValueError("cannot write an empty trajectory cache shard")
        first = examples[0]
        source_id = str(first.metadata["source_id"])
        signature = (
            tuple(first.anchor_latent.shape),
            tuple(first.source_latent.shape),
            tuple(first.keep_mask.shape),
            first.reference_latent is not None,
        )
        for item in examples:
            current = (
                tuple(item.anchor_latent.shape),
                tuple(item.source_latent.shape),
                tuple(item.keep_mask.shape),
                item.reference_latent is not None,
            )
            if current != signature:
                raise ValueError("trajectory cache shard mixes tensor shapes")
            if str(item.metadata["source_id"]) != source_id:
                raise ValueError("trajectory cache shard mixes source trajectories")
            if not torch.equal(item.source_latent, first.source_latent):
                raise ValueError("source_latent differs inside one trajectory")
            if not torch.equal(item.keep_mask, first.keep_mask):
                raise ValueError("keep_mask differs inside one trajectory")
            if first.reference_latent is not None and not torch.equal(
                item.reference_latent, first.reference_latent
            ):
                raise ValueError("reference_latent differs inside one trajectory")

        step_indices = tuple(int(item.metadata["step_index"]) for item in examples)
        if step_indices != tuple(sorted(step_indices)) or len(set(step_indices)) != len(
            step_indices
        ):
            raise ValueError("trajectory cache points must be unique and increasing")

        tensors = {
            "trajectory_latents": torch.stack(
                [item.anchor_latent for item in examples]
            ).unsqueeze(0),
            "source_latent": first.source_latent.unsqueeze(0),
            "keep_mask": first.keep_mask.unsqueeze(0),
            **_mask_variant_tensors(examples, grouped=True),
        }
        if first.reference_latent is not None:
            tensors["reference_latent"] = first.reference_latent.unsqueeze(0)
        shard_name = self._next_shard_name()
        records = [
            {
                **dict(item.metadata),
                **_mask_variant_metadata(item),
                "shard": shard_name,
                "offset": 0,
                "point_offset": point_offset,
                "has_reference": item.reference_latent is not None,
                "storage_layout": TRAJECTORY_GROUPED_LAYOUT,
            }
            for point_offset, item in enumerate(examples)
        ]
        return self._commit(shard_name, tensors, records)


def read_cache_records(path: str | Path) -> tuple[dict[str, Any], ...]:
    root = Path(path).expanduser().resolve()
    records: list[dict[str, Any]] = []
    for sidecar in sorted(root.glob("metadata-*.jsonl")):
        for line_number, line in enumerate(
            sidecar.read_text(encoding="utf-8").splitlines(), 1
        ):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"invalid cache record at {sidecar}:{line_number}")
                records.append(value)
    return tuple(records)


def validate_cache(path: str | Path, *, allow_updating: bool = False) -> dict[str, Any]:
    """Validate all contents; only the extension publisher may bypass markers."""
    from safetensors import safe_open

    root = Path(path).expanduser().resolve()
    if not allow_updating and (root / POINT_EXTENSION_MARKER).exists():
        raise RuntimeError(f"cache point extension is in progress: {root}")
    if not allow_updating and not (root / "COMPLETE").is_file():
        raise RuntimeError(f"cache is incomplete: {root}")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    storage_layout = str(manifest.get("storage_layout", STEP_RECORDS_LAYOUT))
    if storage_layout not in {STEP_RECORDS_LAYOUT, TRAJECTORY_GROUPED_LAYOUT}:
        raise ValueError(f"unsupported cache storage layout: {storage_layout!r}")
    records = read_cache_records(root)
    if len(records) != int(manifest["records"]):
        raise ValueError("cache manifest record count differs from sidecars")
    source_steps: dict[str, set[int]] = {}
    seen_samples: set[str] = set()
    for record in records:
        sample_id = str(record["sample_id"])
        if sample_id in seen_samples:
            raise ValueError(f"duplicate cache sample_id: {sample_id}")
        seen_samples.add(sample_id)
        source_steps.setdefault(str(record["source_id"]), set()).add(
            int(record["step_index"])
        )
    required = set(int(step) for step in manifest["state_points"])
    for source_id, steps in source_steps.items():
        if steps != required:
            raise ValueError(f"source {source_id!r} has incomplete state points")
    records_by_shard: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        records_by_shard.setdefault(str(record["shard"]), []).append(record)
    for shard_name, shard_records in sorted(records_by_shard.items()):
        shard = root / shard_name
        if not shard.is_file():
            raise FileNotFoundError(shard)
        with safe_open(shard, framework="pt", device="cpu") as handle:
            tensor_keys = set(handle.keys())
            required_tensors = (
                {"trajectory_latents", "source_latent", "keep_mask"}
                if storage_layout == TRAJECTORY_GROUPED_LAYOUT
                else {"anchor_latent", "source_latent", "keep_mask"}
            )
            if not required_tensors.issubset(tensor_keys):
                raise ValueError(f"cache shard lacks required tensors: {shard}")
            configured = (
                manifest.get("creation_config", {})
                .get("cache", {})
                .get("mask_variants")
            )
            declared = {
                tuple(sorted(record.get("mask_variants", [])))
                for record in shard_records
            }
            if len(declared) != 1:
                raise ValueError(
                    f"cache shard mixes mask variant declarations: {shard}"
                )
            variants = next(iter(declared))
            if configured is not None and set(variants) != set(configured):
                raise ValueError(
                    f"cache mask variants differ from creation config: {shard}"
                )
            source_shape = handle.get_slice("source_latent").get_shape()
            for variant in variants:
                key = mask_tensor_key(variant)
                if variant == "default" or key not in tensor_keys:
                    raise ValueError(
                        f"cache shard lacks declared mask variant: {variant}"
                    )
                tensor = handle.get_slice(key)
                expected = (
                    source_shape
                    if variant == "fine"
                    and manifest.get("representation") != "flux1_unpacked_vae_latent"
                    else handle.get_slice("keep_mask").get_shape()
                )
                if tensor.get_shape() != expected or tensor.get_dtype() != "U8":
                    raise ValueError(f"invalid {variant} mask shape/dtype: {shard}")
            if storage_layout == TRAJECTORY_GROUPED_LAYOUT:
                trajectory_shape = handle.get_slice("trajectory_latents").get_shape()
                source_shape = handle.get_slice("source_latent").get_shape()
                mask_shape = handle.get_slice("keep_mask").get_shape()
                if len(trajectory_shape) < 3:
                    raise ValueError(f"invalid trajectory tensor shape: {shard}")
                if (
                    source_shape[0] != trajectory_shape[0]
                    or mask_shape[0] != trajectory_shape[0]
                ):
                    raise ValueError(
                        f"trajectory tensor batch dimensions differ: {shard}"
                    )
                records_by_offset: dict[int, list[dict[str, Any]]] = {}
                for record in shard_records:
                    if record.get("storage_layout") != TRAJECTORY_GROUPED_LAYOUT:
                        raise ValueError(
                            f"cache record has the wrong storage layout: {shard}"
                        )
                    offset = int(record["offset"])
                    point_offset = int(record["point_offset"])
                    if not 0 <= offset < trajectory_shape[0]:
                        raise ValueError(f"cache shard offset is out of range: {shard}")
                    if not 0 <= point_offset < trajectory_shape[1]:
                        raise ValueError(
                            f"cache trajectory point offset is out of range: {shard}"
                        )
                    records_by_offset.setdefault(offset, []).append(record)
                for offset, trajectory_records in records_by_offset.items():
                    point_offsets = {
                        int(record["point_offset"]) for record in trajectory_records
                    }
                    if point_offsets != set(range(trajectory_shape[1])):
                        raise ValueError(
                            "cache trajectory has incomplete point offsets: "
                            f"{shard} offset={offset}"
                        )
                    source_ids = {
                        str(record["source_id"]) for record in trajectory_records
                    }
                    if len(source_ids) != 1:
                        raise ValueError(
                            f"cache trajectory offset mixes sources: {shard}"
                        )
                if set(records_by_offset) != set(range(trajectory_shape[0])):
                    raise ValueError(f"cache shard has unindexed trajectories: {shard}")
    if len(source_steps) != int(manifest["sources"]):
        raise ValueError("cache manifest source count differs")
    visualization_summary = manifest.get("visualizations")
    if visualization_summary is not None:
        from .visualization import validate_visualizations

        if not isinstance(visualization_summary, Mapping):
            raise TypeError("cache visualizations manifest must be a mapping")
        validate_visualizations(
            root,
            visualization_summary,
            expected_samples=int(manifest["sources"]),
        )
    return manifest
