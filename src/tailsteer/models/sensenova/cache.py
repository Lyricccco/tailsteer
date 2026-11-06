"""Keep masks for SenseNova's packed RGB pixels, without loading model weights."""

import numpy as np

from tailsteer.models.masks import dilated_keep_mask


def sensenova_keep_mask_variants(
    change_mask: np.ndarray,
    *,
    state_shape: tuple[int, int, int],
    grid: tuple[int, int],
    variants: list[str],
) -> dict[str, np.ndarray]:
    height, width = change_mask.shape
    grid_h, grid_w = grid
    if grid_h < 1 or grid_w < 1 or height % grid_h or width % grid_w:
        raise ValueError("SenseNova image dimensions must align to the token grid")
    patch = height // grid_h
    if (
        width // grid_w != patch
        or state_shape != (1, grid_h * grid_w, patch * patch * 3)
        or set(variants) - {"fine", "dilate5"}
    ):
        raise ValueError("invalid SenseNova packed state shape or mask variants")
    result = {}
    if "fine" in variants:
        # Official patchify(channel_first=False): token row/column, pixel dy/dx,
        # then RGB. Keep the historical singleton state batch axis on disk.
        rgb = np.repeat((~change_mask.astype(bool))[..., None], 3, axis=-1)
        result["fine"] = (
            rgb.reshape(grid_h, patch, grid_w, patch, 3)
            .transpose(0, 2, 1, 3, 4)
            .reshape(state_shape)
            .astype(np.uint8)
        )
    if "dilate5" in variants:
        result["dilate5"] = dilated_keep_mask(change_mask, grid_h, grid_w, radius=5)
    return result
