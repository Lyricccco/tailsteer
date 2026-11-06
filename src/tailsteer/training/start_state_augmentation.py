"""Small, model-neutral augmentations for cached training states."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StartStateAugmentation:
    """Apply an optional RMS-preserving jitter to a cached state.

    The perturbation is deliberately local: it changes the state presented to
    the objective without pretending to be a new teacher trajectory.  A zero
    scale (and the default disabled configuration) is bit-for-bit a no-op.
    """

    enabled: bool = False
    mode: str = "rms_preserving_jitter"
    noise_scale: float = 0.0
    orthogonalize: bool = True

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "StartStateAugmentation":
        values = {} if config is None else dict(config)
        enabled = bool(values.get("enabled", False))
        mode = str(values.get("mode", "rms_preserving_jitter"))
        if mode != "rms_preserving_jitter":
            raise ValueError(
                "start_state_augmentation.mode must be 'rms_preserving_jitter'"
            )
        raw_scale = values.get("noise_scale", 0.0)
        if isinstance(raw_scale, bool) or not isinstance(raw_scale, (int, float)):
            raise TypeError("start_state_augmentation.noise_scale must be real")
        noise_scale = float(raw_scale)
        if not math.isfinite(noise_scale) or noise_scale < 0:
            raise ValueError(
                "start_state_augmentation.noise_scale must be finite and non-negative"
            )
        return cls(
            enabled=enabled,
            mode=mode,
            noise_scale=noise_scale,
            orthogonalize=bool(values.get("orthogonalize", True)),
        )

    def apply(self, state: Any, sigma: float, *, training: bool) -> Any:
        """Return a jittered state, preserving each sample's global RMS."""

        import torch

        if not self.enabled or not training or self.noise_scale == 0.0:
            return state
        if not torch.is_floating_point(state):
            raise TypeError("start-state jitter requires a floating-point state")
        if state.ndim < 2:
            raise ValueError("start-state jitter requires a batched state")

        original = state.float()
        noise = torch.randn_like(original)
        flat_state = original.flatten(1)
        flat_noise = noise.flatten(1)
        if self.orthogonalize:
            denominator = flat_state.square().sum(dim=1, keepdim=True).clamp_min(1e-12)
            projection = (flat_noise * flat_state).sum(
                dim=1, keepdim=True
            ) / denominator
            flat_noise = flat_noise - projection * flat_state
            noise = flat_noise.view_as(original)

        perturbed = original + abs(float(sigma)) * self.noise_scale * noise
        original_rms = original.square().mean(
            dim=tuple(range(1, original.ndim)), keepdim=True
        )
        perturbed_rms = perturbed.square().mean(
            dim=tuple(range(1, perturbed.ndim)), keepdim=True
        )
        scale = (original_rms / perturbed_rms.clamp_min(1e-12)).sqrt()
        return (perturbed * scale).to(dtype=state.dtype)


def configured_start_state_augmentation(
    training: Mapping[str, Any],
) -> StartStateAugmentation:
    """Read the shared training-level augmentation without model special cases."""

    return StartStateAugmentation.from_config(training.get("start_state_augmentation"))
