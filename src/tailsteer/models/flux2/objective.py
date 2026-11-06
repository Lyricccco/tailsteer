"""FLUX.2-specific preservation objective and decreasing-sigma convention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .training import Flux2TrainingBatch


@dataclass(frozen=True)
class Flux2ObjectiveResult:
    loss: Any
    metrics: dict[str, Any]
    per_sample: Any


def _spatial_mask(mask: Any, latent: Any) -> Any:
    if mask.ndim == latent.ndim - 1:
        mask = mask.unsqueeze(1)
    if mask.ndim != latent.ndim or mask.shape[0] != latent.shape[0]:
        raise ValueError("FLUX.2 mask and latent ranks/batch sizes differ")
    if mask.shape[1] not in {1, latent.shape[1]}:
        raise ValueError("FLUX.2 mask channels must be one or match the latent")
    if mask.shape[2:] != latent.shape[2:]:
        raise ValueError("FLUX.2 mask and latent spatial dimensions differ")
    return mask.float()


def _masked_per_sample(spatial_error: Any, mask: Any) -> Any:
    if mask.shape[1] != 1:
        mask = mask.mean(dim=1, keepdim=True)
    spatial_error = spatial_error.mean(dim=1, keepdim=True)
    dimensions = tuple(range(1, spatial_error.ndim))
    denominator = mask.sum(dim=dimensions).clamp_min(1.0)
    return (spatial_error * mask).sum(dim=dimensions) / denominator


def flux2_preservation_velocity(
    prediction: Any,
    batch: Flux2TrainingBatch,
    *,
    weight: float = 1.0,
) -> Flux2ObjectiveResult:
    """Preserve z0 while FLUX integrates along decreasing sigma."""

    if batch.sigma <= 0:
        raise ValueError("FLUX.2 preservation requires sigma > 0")
    predicted = prediction.float()
    current = batch.latent.float()
    source = batch.source_latent.float()
    if predicted.shape != current.shape or source.shape != current.shape:
        raise ValueError("FLUX.2 prediction/source/state shapes differ")
    mask = _spatial_mask(batch.keep_mask, current)
    target = (source - current) / (-batch.sigma)
    per_sample = _masked_per_sample((predicted - target).square(), mask)
    raw_loss = per_sample.mean()
    predicted_z0 = current + predicted * (-batch.sigma)
    z0 = _masked_per_sample((predicted_z0 - source).square(), mask).mean()
    return Flux2ObjectiveResult(
        loss=raw_loss * float(weight),
        per_sample=per_sample,
        metrics={
            "loss": raw_loss.detach() * float(weight),
            "masked_velocity_mse": raw_loss.detach(),
            "masked_z0_mse": z0.detach(),
            "valid_mask_fraction": mask.mean().detach(),
        },
    )
