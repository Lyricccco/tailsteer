"""Native FLUX.2 reference-image geometry, before VAE encoding."""

from __future__ import annotations

import math
from typing import Any

FLUX2_SINGLE_REFERENCE_MAX_PIXELS = 2024**2
FLUX2_MULTI_REFERENCE_MAX_PIXELS = 1024**2
FLUX2_REFERENCE_MULTIPLE = 16
FLUX2_REFERENCE_MIN_SIDE = 64
FLUX2_REFERENCE_MAX_ASPECT_RATIO = 8.0


def flux2_reference_max_pixels(reference_count: int) -> int:
    """Return the native per-image area cap for one conditioning call."""

    if reference_count < 1:
        raise ValueError("FLUX.2 reference_count must be positive")
    return (
        FLUX2_SINGLE_REFERENCE_MAX_PIXELS
        if reference_count == 1
        else FLUX2_MULTI_REFERENCE_MAX_PIXELS
    )


def flux2_reference_geometry(size, *, reference_count: int = 1) -> dict:
    """Describe the native area cap and minimal crop without allocating an image."""
    width, height = size
    if width < FLUX2_REFERENCE_MIN_SIDE or height < FLUX2_REFERENCE_MIN_SIDE:
        raise ValueError(
            "FLUX.2 reference image has a side shorter than "
            f"{FLUX2_REFERENCE_MIN_SIDE}: {(width, height)}"
        )
    aspect_ratio = max(width / height, height / width)
    if aspect_ratio > FLUX2_REFERENCE_MAX_ASPECT_RATIO:
        raise ValueError(
            "FLUX.2 reference image aspect ratio exceeds "
            f"{FLUX2_REFERENCE_MAX_ASPECT_RATIO:g}: {(width, height)}"
        )

    max_pixels = flux2_reference_max_pixels(reference_count)
    if width * height > max_pixels:
        scale = math.sqrt(max_pixels / (width * height))
        width, height = int(width * scale), int(height * scale)

    cropped_width = width // FLUX2_REFERENCE_MULTIPLE * FLUX2_REFERENCE_MULTIPLE
    cropped_height = height // FLUX2_REFERENCE_MULTIPLE * FLUX2_REFERENCE_MULTIPLE
    left = (width - cropped_width) // 2
    top = (height - cropped_height) // 2
    return {
        "original_size": list(size),
        "resized_size": [width, height],
        "reference_size": [cropped_width, cropped_height],
        "generation_size": [cropped_width, cropped_height],
        "crop_box": [left, top, left + cropped_width, top + cropped_height],
    }


def prepare_flux2_reference_image(
    image: Any,
    *,
    reference_count: int = 1,
) -> Any:
    """Match the native RGB/area-cap/16px crop; never crop to a square."""
    from PIL import Image

    value = image.convert("RGB")
    geometry = flux2_reference_geometry(value.size, reference_count=reference_count)
    size = tuple(geometry["resized_size"])
    if value.size != size:
        value = value.resize(size, Image.Resampling.LANCZOS)
    return value.crop(geometry["crop_box"])
