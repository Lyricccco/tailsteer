"""Image-space keep-mask operations shared by native state representations."""

import cv2
import numpy as np


def project_keep_mask(change: np.ndarray, height: int, width: int) -> np.ndarray:
    """Conservative projection: any changed pixel excludes its grid cell."""
    image_h, image_w = change.shape
    if height < 1 or width < 1 or image_h % height or image_w % width:
        raise ValueError("image dimensions are not divisible by the state grid")
    return ~change.astype(bool).reshape(
        height, image_h // height, width, image_w // width
    ).any(axis=(1, 3))


def dilated_keep_mask(
    change: np.ndarray, height: int, width: int, *, radius: int
) -> np.ndarray:
    """Dilate at ORIGINAL image resolution, then pool to a singleton-channel grid."""
    if radius < 0:
        raise ValueError("mask dilation radius must be nonnegative")
    expanded = cv2.dilate(
        np.uint8(change),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2),
    )
    return project_keep_mask(expanded, height, width)[None].astype(np.uint8)
