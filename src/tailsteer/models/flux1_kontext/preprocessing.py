"""Kontext inference geometry; cache/training retain their own square inputs."""

from __future__ import annotations

from tailsteer.models.flux2.preprocessing import (
    FLUX2_SINGLE_REFERENCE_MAX_PIXELS,
    flux2_reference_geometry,
    prepare_flux2_reference_image,
)

PREPROCESSING = {
    "policy": "kontext_flux2_geometry_v1",
    "image_multiple": 16,
    "max_pixels": FLUX2_SINGLE_REFERENCE_MAX_PIXELS,
}


def source_geometry(size, *, input_preprocessed=False):
    if input_preprocessed:
        if any(side <= 0 or side % 16 for side in size):
            raise ValueError(
                "Prepared Kontext dimensions must be positive multiples of 16"
            )
        geometry = {
            "original_size": list(size),
            "resized_size": list(size),
            "reference_size": list(size),
            "generation_size": list(size),
            "crop_box": [0, 0, *size],
        }
    else:
        geometry = flux2_reference_geometry(size)
    return {**PREPROCESSING, **geometry}


def prepare_source(image, *, input_preprocessed=False):
    if not input_preprocessed:
        return prepare_flux2_reference_image(image)
    source_geometry(image.size, input_preprocessed=True)
    return image.convert("RGB")


def encode_source(ae, image, *, device):
    """BFL's FP32 encode and scaled BF16 conditioning, without a spatial transform."""
    import numpy as np
    import torch

    pixels = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0
    pixels = pixels.permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad(), torch.autocast(torch.device(device).type, enabled=False):
        return ae.encode(pixels).to(torch.bfloat16)


def prepare_shared_inputs(runtime, source, *, prompt, seed):
    """Use native packing/text/noise with reference and target on the same canvas.

    BFL's prepare_kontext always resizes references to a resolution bucket, even
    with explicit target dimensions. Only this input boundary is replaced;
    the model, schedule and denoise loop remain native.
    """
    source = prepare_source(source, input_preprocessed=True)
    state = encode_source(runtime.ae, source, device=runtime.device)
    return prepare_latent_inputs(runtime, state, prompt=prompt, seed=seed)


def prepare_latent_inputs(runtime, state, *, prompt, seed):
    """Pack an already scaled BFL AE state; never decode or scale it again."""
    import torch
    from einops import rearrange
    from flux.sampling import get_noise, prepare

    if state.ndim == 3:
        state = state.unsqueeze(0)
    if (
        state.ndim != 4
        or state.shape[:2] != (1, 16)
        or any(side % 2 for side in state.shape[-2:])
    ):
        raise ValueError("Kontext reference latent must be [1, 16, even H, even W]")
    state = state.to(device=runtime.device, dtype=torch.bfloat16)
    height, width = (int(side) * 8 for side in state.shape[-2:])
    noise = get_noise(1, height, width, runtime.device, torch.bfloat16, int(seed))
    inputs = prepare(runtime.t5, runtime.clip, noise, prompt)
    inputs["img_cond_seq"] = rearrange(
        state, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2
    )
    # Same grid as target tokens, with BFL's reference-image identifier.
    inputs["img_cond_seq_ids"] = inputs["img_ids"].clone()
    inputs["img_cond_seq_ids"][..., 0] = 1
    return inputs, height, width
