"""Official Diffusers rollout capture used by strict FLUX.2 parity checks."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import wraps
from typing import Any


@dataclass(frozen=True)
class Flux2OfficialTrace:
    source_state: Any
    forward_states: dict[int, Any]
    forward_velocities: dict[int, Any]
    timesteps: dict[int, float]
    final_state: Any


def _unpack_tokens(tokens: Any, *, height: int, width: int) -> Any:
    batch, sequence, channels = tokens.shape
    latent_height, latent_width = height // 16, width // 16
    if sequence != latent_height * latent_width:
        raise RuntimeError("FLUX.2 official token count differs from output grid")
    return tokens.permute(0, 2, 1).reshape(batch, channels, latent_height, latent_width)


def capture_official_flux2_trace(
    backend: Any,
    *,
    source_image: Any,
    prompt: str,
    seed: int,
    width: int,
    height: int,
    num_steps: int,
    schedule_width: int | None = None,
    schedule_height: int | None = None,
) -> Flux2OfficialTrace:
    """Run the unmodified pipeline loop and expose its model-native states."""

    import torch

    pipe = backend.pipe
    if pipe.vae is None:
        raise RuntimeError("official FLUX.2 parity requires the VAE")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    initial_tokens, _ = pipe.prepare_latents(
        batch_size=1,
        num_latents_channels=int(pipe.transformer.config.in_channels) // 4,
        height=int(height),
        width=int(width),
        dtype=backend.dtype,
        device=backend.device,
        generator=generator,
    )
    initial = _unpack_tokens(initial_tokens, height=height, width=width)
    forward_states = {0: initial.detach().cpu().contiguous()}
    source_holder: dict[str, Any] = {}
    final_holder: dict[str, Any] = {}
    velocities: dict[int, Any] = {}
    timesteps: dict[int, float] = {}
    original_prepare_image_latents = pipe.prepare_image_latents
    original_set_timesteps = pipe.scheduler.set_timesteps
    original_step = pipe.scheduler.step
    schedule_width = int(width if schedule_width is None else schedule_width)
    schedule_height = int(height if schedule_height is None else schedule_height)
    if schedule_width % 16 or schedule_height % 16:
        raise ValueError("FLUX.2 reference schedule shape must be divisible by 16")
    module = importlib.import_module(type(pipe).__module__)
    empirical_mu = getattr(module, "compute_empirical_mu", None)
    if not callable(empirical_mu):
        raise RuntimeError("official FLUX.2 schedule function is unavailable")
    fixed_mu = empirical_mu(
        image_seq_len=(schedule_height // 16) * (schedule_width // 16),
        num_steps=int(num_steps),
    )

    def capture_source(*args: Any, **kwargs: Any):
        packed, image_ids = original_prepare_image_latents(*args, **kwargs)
        source_holder["state"] = (
            _unpack_tokens(packed, height=height, width=width)
            .detach()
            .cpu()
            .contiguous()
        )
        return packed, image_ids

    def capture_step(_pipe: Any, index: int, _timestep: Any, values: dict[str, Any]):
        state = _unpack_tokens(values["latents"], height=height, width=width)
        completed = int(index) + 1
        if completed < int(num_steps):
            forward_states[completed] = state.detach().cpu().contiguous()
        else:
            final_holder["state"] = state.detach().cpu().contiguous()
        return values

    @wraps(original_set_timesteps)
    def set_timesteps(*args: Any, **kwargs: Any) -> Any:
        kwargs["mu"] = fixed_mu
        return original_set_timesteps(*args, **kwargs)

    def scheduler_step(*args: Any, **kwargs: Any) -> Any:
        model_output = kwargs.get("model_output", args[0] if args else None)
        timestep = kwargs.get("timestep", args[1] if len(args) > 1 else None)
        if model_output is None:
            raise RuntimeError("official FLUX.2 scheduler did not expose velocity")
        if timestep is None:
            raise RuntimeError("official FLUX.2 scheduler did not expose timestep")
        index = len(velocities)
        velocities[index] = (
            _unpack_tokens(model_output, height=height, width=width)
            .detach()
            .cpu()
            .contiguous()
        )
        timesteps[index] = float(timestep.detach().float().cpu())
        return original_step(*args, **kwargs)

    pipe.prepare_image_latents = capture_source
    pipe.scheduler.set_timesteps = set_timesteps
    pipe.scheduler.step = scheduler_step
    try:
        pipe(
            image=source_image,
            prompt=str(prompt),
            height=int(height),
            width=int(width),
            num_inference_steps=int(num_steps),
            guidance_scale=float(backend.cfg_scale),
            generator=torch.Generator(device="cpu").manual_seed(int(seed)),
            latents=initial,
            output_type="latent",
            callback_on_step_end=capture_step,
            callback_on_step_end_tensor_inputs=["latents"],
        )
    finally:
        pipe.scheduler.step = original_step
        pipe.scheduler.set_timesteps = original_set_timesteps
        pipe.prepare_image_latents = original_prepare_image_latents
    if set(forward_states) != set(range(int(num_steps))):
        raise RuntimeError("official FLUX.2 rollout missed forward states")
    if "state" not in source_holder or "state" not in final_holder:
        raise RuntimeError("official FLUX.2 rollout missed source or final state")
    return Flux2OfficialTrace(
        source_state=source_holder["state"],
        forward_states=forward_states,
        forward_velocities=velocities,
        timesteps=timesteps,
        final_state=final_holder["state"],
    )
