"""Storage-neutral cache samples consumed by model-specific training code."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CacheSample:
    """One cached state without assumptions about its tensor layout.

    ``latent`` may be an image-like NCHW tensor, packed image tokens, or another
    model-native representation.  The cache reader only preserves values; the
    selected model backend is responsible for interpreting them.
    """

    latent: Any
    source_latent: Any
    keep_mask: Any
    metadata: Mapping[str, Any]
    reference_latent: Any | None = None
    terminal_latent: Any | None = None


def normalize_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the small metadata vocabulary shared by training backends."""

    required = (
        "sample_id",
        "source_id",
        "source_identity",
        "prompt",
        "seed",
        "step_index",
        "sigma",
        "timestep",
        "height",
        "width",
        "task",
    )
    missing = [name for name in required if name not in value]
    if missing:
        raise KeyError(f"cache metadata lacks required fields: {missing}")
    return {
        **dict(value),
        "sample_id": str(value["sample_id"]),
        "source_id": str(value["source_id"]),
        "source_identity": str(value["source_identity"]),
        "prompt": str(value["prompt"]),
        "seed": int(value["seed"]),
        "step_index": int(value["step_index"]),
        "sigma": float(value["sigma"]),
        "timestep": float(value["timestep"]),
        "height": int(value["height"]),
        "width": int(value["width"]),
        "task": str(value["task"]),
    }
