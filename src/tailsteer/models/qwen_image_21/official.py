"""Run the official Diffusers pipeline; observe, never replace, its sampler.

Canonical states are normalized VAE latents [64,H/16,W/16], a lossless
spatial unflattening of the native [1,HW/256,64] sequence. They are neither
Qwen-VL embeddings nor the DiT's internal coarse attention tokens.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

QWEN21_SOURCE_REVISION = "fb7ae1d1f9611cd91524d03c53c5246b36ac8577"
DIFFUSERS_REVISION = "1d5d056ecff3a8c1d887a7dd1aa66ca4a0323458"


def unpack_state(tokens: Any, *, width: int, height: int) -> Any:
    import torch

    shape = (1, (height // 16) * (width // 16), 64)
    if width % 16 or height % 16 or tuple(tokens.shape) != shape:
        raise ValueError(f"unexpected Qwen-Image 2.1 token shape: {tokens.shape}")
    state = tokens[0].T.reshape(64, height // 16, width // 16).detach().cpu().clone()
    if not torch.isfinite(state).all():
        raise RuntimeError("Qwen-Image 2.1 produced a non-finite state")
    return state.contiguous()


def composite_rgb(image: Any) -> Any:
    """Explicit display/filter convention; never discard generated alpha."""
    from PIL import Image

    rgba = image.convert("RGBA")
    return Image.alpha_composite(
        Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba
    ).convert("RGB")


@dataclass
class QwenImage21Trace:
    image: Any = None
    rgba_image: Any = None
    source_state: Any = None
    forward_states: dict[int, Any] = field(default_factory=dict)
    forward_velocities: dict[int, Any] = field(default_factory=dict)
    final_state: Any = None
    schedule_points: dict[int, tuple[float, float]] = field(default_factory=dict)
    final_schedule_point: tuple[float, float] = (0.0, 0.0)


@contextmanager
def capture_states(
    pipeline: Any, *, points: tuple[int, ...], num_steps: int, width: int, height: int
):
    if (
        not points
        or points != tuple(sorted(set(points)))
        or points[0] < 0
        or points[-1] >= num_steps
    ):
        raise ValueError("capture points must be unique, increasing forward indices")
    trace = QwenImage21Trace()
    scheduler = pipeline.scheduler
    original_prepare, original_step = pipeline.prepare_latents, scheduler.step
    prepare_override = pipeline.__dict__.get("prepare_latents")
    step_override = scheduler.__dict__.get("step")
    calls = 0

    def snapshot(tokens):
        return unpack_state(tokens, width=width, height=height)

    def prepare(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        # prepare_latents returns (target noise, encoded reference), before
        # set_timesteps. The source is already in normalized model-space units.
        trace.source_state = snapshot(result[1])
        return result

    def step(model_output, timestep, sample, *args, **kwargs):
        nonlocal calls
        if calls == 0:
            if (
                len(scheduler.timesteps) != num_steps
                or len(scheduler.sigmas) != num_steps + 1
            ):
                raise RuntimeError("Qwen scheduler returned an unexpected schedule")
            trace.schedule_points = {
                k: (float(scheduler.sigmas[k]), float(scheduler.timesteps[k]))
                for k in points
            }
            trace.final_schedule_point = (float(scheduler.sigmas[-1]), 0.0)
        if calls >= num_steps or float(timestep) != float(scheduler.timesteps[calls]):
            raise RuntimeError("Qwen forward/scheduler indices diverged")
        if calls in points:
            trace.forward_states[calls] = snapshot(sample)
            trace.forward_velocities[calls] = snapshot(model_output)
        output = original_step(model_output, timestep, sample, *args, **kwargs)
        calls += 1
        if calls == num_steps:
            trace.final_state = snapshot(
                output[0] if isinstance(output, tuple) else output.prev_sample
            )
        return output

    pipeline.prepare_latents, scheduler.step = prepare, step
    try:
        yield trace
        if (
            calls != num_steps
            or trace.source_state is None
            or trace.final_state is None
            or set(trace.forward_states) != set(points)
        ):
            raise RuntimeError(f"incomplete Qwen trajectory: {calls}/{num_steps}")
    finally:
        for obj, name, override in (
            (pipeline, "prepare_latents", prepare_override),
            (scheduler, "step", step_override),
        ):
            if override is None:
                delattr(obj, name)
            else:
                setattr(obj, name, override)


def keep_mask_variants(change_mask, *, state_shape, variants):
    import numpy as np

    from tailsteer.models.masks import dilated_keep_mask, project_keep_mask

    channels, height, width = state_shape
    if (
        channels != 64
        or change_mask.shape != (height * 16, width * 16)
        or set(variants) - {"fine", "dilate5"}
    ):
        raise ValueError("invalid Qwen VAE grid or mask variants")
    result = {}
    if "fine" in variants:
        keep = project_keep_mask(change_mask, height, width)
        # No spatial subpositions are packed into Qwen's VAE channels.
        result["fine"] = np.broadcast_to(keep, state_shape).astype(np.uint8).copy()
    if "dilate5" in variants:
        result["dilate5"] = dilated_keep_mask(change_mask, height, width, radius=5)
    return result


@contextmanager
def model_load_slot():
    """Bound simultaneous CPU weight-loading peaks on eight-GPU workers."""
    slot = int(os.environ.get("LOCAL_RANK", "0")) % 2
    lock = Path("/tmp") / f"tailsteer-qwen21-{os.getuid()}-load-{slot}.lock"
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


class QwenImage21Runtime:
    def __init__(self, pipeline: Any, *, device: Any):
        self.pipeline = pipeline
        self.device = device
        self._lock = threading.RLock()

    @classmethod
    def load(cls, model: dict[str, Any], *, device: Any):
        import torch
        from diffusers import QwenImage21Pipeline

        if model.get("dtype", "bfloat16") != "bfloat16":
            raise ValueError("Qwen cache preserves the official BF16 configuration")
        if model.get("diffusers_revision", DIFFUSERS_REVISION) != DIFFUSERS_REVISION:
            raise ValueError("Qwen config differs from the pinned Diffusers revision")
        root = Path(model["path"]).resolve()
        if (
            json.loads((root / "model_index.json").read_text()).get("_class_name")
            != "QwenImage21Pipeline"
        ):
            raise ValueError("not a Qwen-Image 2.1 pipeline snapshot")
        # Two concurrent loaders bound host-memory peaks on 512-GiB workers.
        logger = logging.getLogger("tailsteer")
        logger.info("Qwen-Image 2.1 waiting for a model-load slot")
        with model_load_slot():
            pipeline = QwenImage21Pipeline.from_pretrained(
                str(root), torch_dtype=torch.bfloat16, local_files_only=True
            ).to(device)
        if pipeline.config.sample_sigmas is not None:
            raise ValueError("expected base Qwen-Image 2.1, not a saved Turbo schedule")
        for module in (pipeline.transformer, pipeline.text_encoder, pipeline.vae):
            module.eval().requires_grad_(False)
        pipeline.set_progress_bar_config(disable=True)
        logger.info("Qwen-Image 2.1 loaded from %s", root)
        return cls(pipeline, device=device)

    def edit(
        self,
        *,
        source_image: Any,
        prompt: str,
        seed: int,
        num_steps: int,
        capture_forward_steps: tuple[int, ...],
        true_cfg_scale: float = 1.0,
        use_kv_cache: bool = True,
    ) -> QwenImage21Trace:
        import torch

        if source_image.size != (1024, 1024) or num_steps != 40:
            raise ValueError("Qwen cache requires a 1024-square source and 40 steps")
        if true_cfg_scale != 1.0:
            raise ValueError("Qwen base cache uses official guidance-free sampling")
        with (
            self._lock,
            capture_states(
                self.pipeline,
                points=capture_forward_steps,
                num_steps=num_steps,
                width=1024,
                height=1024,
            ) as trace,
        ):
            trace.rgba_image = self.pipeline(
                image=source_image,
                prompt=prompt,
                height=1024,
                width=1024,
                output_resolution=1024,
                num_inference_steps=num_steps,
                true_cfg_scale=true_cfg_scale,
                use_kv_cache=use_kv_cache,
                generator=torch.Generator(device=self.device).manual_seed(seed),
            ).images[0]
        if trace.rgba_image.size != source_image.size:
            raise RuntimeError("Qwen output/source resolution mismatch")
        trace.image = composite_rgb(trace.rgba_image)
        return trace

    def unload(self) -> None:
        self.pipeline = None


class QwenImage21VaeDecoder:
    """Decode cached normalized states without loading the DiT or Qwen-VL."""

    @classmethod
    def load(cls, model: dict[str, Any], *, device: str):
        import torch
        from diffusers import AutoencoderKLQwenImage21
        from diffusers.image_processor import VaeImageProcessor

        result = cls()
        result.vae = AutoencoderKLQwenImage21.from_pretrained(
            model["path"],
            subfolder="vae",
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).to(device)
        result.processor = VaeImageProcessor(
            vae_scale_factor=16, vae_latent_channels=64
        )
        return result

    def decode_state(self, state: Any, *, width: int, height: int):
        import torch

        if tuple(state.shape) != (64, height // 16, width // 16):
            raise ValueError(f"unexpected Qwen cached state shape: {state.shape}")
        with torch.no_grad():
            z = state[None, :, None].to(device=self.vae.device, dtype=self.vae.dtype)
            mean = z.new_tensor(self.vae.config.latents_mean).view(1, 64, 1, 1, 1)
            std = z.new_tensor(self.vae.config.latents_std).view(1, 64, 1, 1, 1)
            pixels = self.vae.decode(z * std + mean, return_dict=False)[0][:, :, 0]
            rgba = self.processor.postprocess(pixels, output_type="pil")[0]
        return composite_rgb(rgba)

    def unload(self) -> None:
        self.vae = None
