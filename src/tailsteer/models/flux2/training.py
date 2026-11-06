"""FLUX.2 interpretation of storage-neutral cached training samples."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Flux2TrainingBatch:
    latent: Any
    source_latent: Any
    keep_mask: Any
    prompts: tuple[str, ...]
    step_index: int
    sigma: float
    timestep: float
    height: int
    width: int


def prepare_flux2_training_batch(
    batch: Mapping[str, Any], *, device: Any, dtype: Any
) -> Flux2TrainingBatch:
    latent = batch["latent"]
    source = batch["source_latent"]
    mask = batch["keep_mask"]
    metadata = tuple(batch["metadata"])
    if latent.ndim != 4 or source.shape != latent.shape:
        raise ValueError("FLUX.2 cache states must share NCHW layout")
    if len(metadata) != latent.shape[0]:
        raise ValueError("FLUX.2 metadata and tensor batch sizes differ")
    first = metadata[0]
    homogeneous = ("step_index", "sigma", "timestep", "height", "width")
    for field in homogeneous:
        if len({item[field] for item in metadata}) != 1:
            raise ValueError(f"FLUX.2 batch mixes {field}")
    return Flux2TrainingBatch(
        latent=latent.to(device=device, dtype=dtype),
        source_latent=source.to(device=device, dtype=dtype),
        keep_mask=mask.to(device=device),
        prompts=tuple(str(item["prompt"]) for item in metadata),
        step_index=int(first["step_index"]),
        sigma=float(first["sigma"]),
        timestep=float(first["timestep"]),
        height=int(first["height"]),
        width=int(first["width"]),
    )
