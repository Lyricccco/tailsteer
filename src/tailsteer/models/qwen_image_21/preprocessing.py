"""Inference canvas geometry; cache/training still use their square inputs."""

from contextlib import contextmanager

PREPROCESSING = {"policy": "qwen21_native_min_crop_v1", "image_multiple": 32}


@contextmanager
def reference_canvas_context(pipeline):
    """The pipeline already aligns/resizes the reference for both encoders.

    A second resize in Qwen-VL would enlarge small inputs to its minimum area,
    leaving image placeholders on a different grid from the VAE tokens.
    Normalization and patch extraction remain unchanged. Use under runtime lock.
    """
    processor = pipeline.processor.image_processor
    original = processor.do_resize
    processor.do_resize = False
    try:
        yield
    finally:
        processor.do_resize = original


def source_geometry(size, *, input_preprocessed=False):
    width, height = size
    if min(width, height) < 32:
        raise ValueError("Qwen inference requires both image sides to be at least 32")
    target_width, target_height = width // 32 * 32, height // 32 * 32
    if input_preprocessed and (target_width, target_height) != (width, height):
        raise ValueError("Qwen prepared input dimensions must be divisible by 32")
    left, top = (width - target_width) // 2, (height - target_height) // 2
    return {
        **PREPROCESSING,
        "original_size": [width, height],
        "reference_size": [target_width, target_height],
        "generation_size": [target_width, target_height],
        "crop_box": [left, top, left + target_width, top + target_height],
        "crop_fraction": 1 - target_width * target_height / (width * height),
    }


def prepare_source(source, *, input_preprocessed=False):
    geometry = source_geometry(source.size, input_preprocessed=input_preprocessed)
    image = source.convert("RGB")
    if image.size == tuple(geometry["generation_size"]):
        return image
    return image.crop(geometry["crop_box"])
