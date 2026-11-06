from __future__ import annotations

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from tailsteer.models.flux2.preprocessing import (  # noqa: E402
    FLUX2_MULTI_REFERENCE_MAX_PIXELS,
    FLUX2_SINGLE_REFERENCE_MAX_PIXELS,
    flux2_reference_max_pixels,
    prepare_flux2_reference_image,
)


@pytest.mark.parametrize(
    ("size", "reference_count", "expected"),
    [
        ((1024, 683), 1, (1024, 672)),
        ((2048, 1365), 1, (2048, 1360)),
        ((2048, 1365), 2, (1248, 832)),
        ((4096, 4096), 1, (2016, 2016)),
        ((4096, 4096), 2, (1024, 1024)),
    ],
)
def test_flux2_reference_geometry_matches_native_repo(
    size, reference_count, expected
) -> None:
    image = Image.new("RGBA", size, (1, 2, 3, 4))

    prepared = prepare_flux2_reference_image(
        image,
        reference_count=reference_count,
    )

    assert prepared.mode == "RGB"
    assert prepared.size == expected
    assert prepared.width % 16 == 0
    assert prepared.height % 16 == 0


def test_flux2_reference_area_cap_is_per_image_not_shared() -> None:
    assert flux2_reference_max_pixels(1) == FLUX2_SINGLE_REFERENCE_MAX_PIXELS
    assert flux2_reference_max_pixels(2) == FLUX2_MULTI_REFERENCE_MAX_PIXELS
    assert flux2_reference_max_pixels(3) == FLUX2_MULTI_REFERENCE_MAX_PIXELS


@pytest.mark.parametrize("size", [(63, 512), (64, 513)])
def test_flux2_reference_geometry_rejects_native_invalid_inputs(size) -> None:
    with pytest.raises(ValueError):
        prepare_flux2_reference_image(Image.new("RGB", size))
