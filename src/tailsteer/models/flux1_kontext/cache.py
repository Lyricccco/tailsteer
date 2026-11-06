"""Keep masks on Kontext's unpacked, 8x-downsampled VAE grid."""

import numpy as np

from tailsteer.models.masks import dilated_keep_mask


def keep_mask_variants(
    change_mask: np.ndarray,
    *,
    state_shape: tuple[int, int, int],
    variants: list[str],
) -> dict[str, np.ndarray]:
    if len(state_shape) != 3 or state_shape[0] != 16:
        raise ValueError("Kontext state must be an unpacked [16,H,W] VAE latent")
    _, height, width = state_shape
    if change_mask.shape != (height * 8, width * 8):
        raise ValueError("Kontext change mask must use the original 8x image grid")
    if set(variants) - {"fine", "dilate5"}:
        raise ValueError("unsupported Kontext mask variant")
    # Spatial subpositions are NOT packed into channels here. All 16 feature
    # channels share a mask, so retain [1,H,W] and broadcast only during loss.
    return {
        name: dilated_keep_mask(
            change_mask, height, width, radius=0 if name == "fine" else 5
        )
        for name in variants
    }
