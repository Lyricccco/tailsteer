"""Shared result type for TailSteer training algorithms."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AlgorithmResult:
    loss: Any
    metrics: Mapping[str, Any]
    per_sample: Any | None = None


def masked_per_sample(error: Any, mask: Any) -> Any:
    """Average feature channels, then average only selected locations."""

    import torch

    if error.ndim != mask.ndim or error.shape[0] != mask.shape[0]:
        raise ValueError(
            "masked loss error/mask ranks or batch sizes differ: "
            f"error={tuple(error.shape)} mask={tuple(mask.shape)}"
        )
    for error_size, mask_size in zip(error.shape[1:], mask.shape[1:], strict=True):
        if mask_size not in (1, error_size):
            raise ValueError(
                "masked loss mask is not broadcastable: "
                f"error={tuple(error.shape)} mask={tuple(mask.shape)}"
            )

    # Backend masks normally contain a singleton feature/channel axis.  Reduce
    # that axis before applying spatial/token weighting so channels are not
    # counted as extra locations.
    reduced = error
    for dimension in range(error.ndim - 1, 0, -1):
        if mask.shape[dimension] == 1 and error.shape[dimension] != 1:
            reduced = reduced.mean(dim=dimension, keepdim=True)
    expanded_mask = mask.to(device=error.device, dtype=error.dtype)
    if reduced.shape != expanded_mask.shape:
        reduced, expanded_mask = torch.broadcast_tensors(reduced, expanded_mask)
    dimensions = tuple(range(1, reduced.ndim))
    denominator = expanded_mask.sum(dim=dimensions).clamp_min(1.0)
    return (reduced * expanded_mask).sum(dim=dimensions) / denominator
