"""SenseNova preservation objective for positive-time pixel flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SenseNovaObjectiveResult:
    loss: Any
    metrics: dict[str, Any]
    per_sample: Any


def sensenova_preservation_velocity(
    prediction: Any,
    current: Any,
    source: Any,
    keep_mask: Any,
    *,
    sigma: float,
    weight: float = 1.0,
) -> SenseNovaObjectiveResult:
    """Preserve z1 while SenseNova integrates from noise at t=0 to image at t=1."""

    if sigma <= 0:
        raise ValueError("SenseNova preservation requires sigma > 0")
    predicted = prediction.float()
    current = current.float()
    source = source.float()
    if current.shape != source.shape or current.shape != predicted.shape:
        raise ValueError("SenseNova prediction/source/state shapes differ")
    mask = keep_mask.float()
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.ndim != 3 or mask.shape[1] * mask.shape[2] != current.shape[1]:
        raise ValueError("SenseNova mask grid differs from packed token grid")
    mask = mask.reshape(mask.shape[0], current.shape[1], 1)
    target = (source - current) / float(sigma)
    per_token = (predicted - target).square().mean(dim=-1, keepdim=True)
    denominator = mask.sum(dim=(1, 2)).clamp_min(1.0)
    per_sample = (per_token * mask).sum(dim=(1, 2)) / denominator
    raw_loss = per_sample.mean()
    predicted_source = current + predicted * float(sigma)
    z1_token = (predicted_source - source).square().mean(dim=-1, keepdim=True)
    z1 = ((z1_token * mask).sum(dim=(1, 2)) / denominator).mean()
    return SenseNovaObjectiveResult(
        loss=raw_loss * float(weight),
        per_sample=per_sample,
        metrics={
            "loss": raw_loss.detach() * float(weight),
            "masked_velocity_mse": raw_loss.detach(),
            "masked_z1_mse": z1.detach(),
            "valid_mask_fraction": mask.mean().detach(),
        },
    )
