"""Explicit segmented FLUX.2 rollout used by comparison inference."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any

from .model import ScheduleTransition

if TYPE_CHECKING:
    from tailsteer.inference.endpoint import EndpointExecutionPlan


@dataclass(frozen=True)
class RolloutResult:
    latent: Any
    actors: tuple[str, ...]
    elapsed_seconds: float
    spans: tuple[dict[str, Any], ...] = ()


class Flux2TextInference:
    """Native text-only pipeline; callback switches only the denoising actor."""

    def __init__(self, model: Any) -> None:
        self.model = model

    @staticmethod
    def conditioning_actor_for(policy: Any, *, num_steps: int) -> str:
        return "base"  # External text encoder remains frozen.

    def generate_text(self, *, prompt, seed, policy, config):
        import torch

        if config["schedule"]["shift_policy"] != "native":
            raise ValueError("FLUX.2 T2I uses the native resolution-dependent schedule")
        pipe = self.model.pipe
        original = pipe.transformer
        trace = []
        pipe.transformer = self.model._actor(policy.actor_at(0))

        def step_end(pipeline, step, timestep, tensors):
            trace.append(policy.actor_at(step))
            if step + 1 < policy.num_steps:
                pipeline.transformer = self.model._actor(policy.actor_at(step + 1))
            return tensors

        try:
            result = pipe(
                image=None,
                prompt=prompt,
                height=config["height"],
                width=config["width"],
                num_inference_steps=policy.num_steps,
                guidance_scale=self.model.cfg_scale,
                generator=torch.Generator(device="cpu").manual_seed(seed),
                callback_on_step_end=step_end,
            )
            return result.images[0], tuple(trace)
        finally:
            pipe.transformer = original


class Flux2DevTextInference(Flux2TextInference):
    """Native Dev T2I: no reference tokens, official noise and Euler schedule."""

    def generate_text(self, *, prompt, seed, policy, config):
        import torch
        from flux2.sampling import get_schedule

        from .model import Flux2Conditioning, Flux2Schedule, SchedulePoint

        if config["schedule"]["shift_policy"] != "native":
            raise ValueError("FLUX.2 Dev T2I requires the native schedule")
        height, width = int(config["height"]), int(config["width"])
        if min(height, width) < 16 or height % 16 or width % 16:
            raise ValueError("FLUX.2 Dev T2I dimensions must be multiples of 16")
        model = self.model
        with torch.no_grad():
            context, context_ids = model._batched_prc_txt(model.text_encoder([prompt]))
            latent = torch.randn(
                (1, 128, height // 16, width // 16),
                generator=torch.Generator(device=model.device).manual_seed(seed),
                device=model.device,
                dtype=model.dtype,
            )
            tokens, token_ids = model._batched_prc_img(latent)
            conditioning = Flux2Conditioning(
                payload={
                    "context": context,
                    "context_ids": context_ids,
                    "reference": tokens[:, :0],
                    "reference_ids": token_ids[:, :0],
                },
                batch_size=1,
            )
            sigmas = get_schedule(policy.num_steps, tokens.shape[1])
            schedule = Flux2Schedule(
                [
                    SchedulePoint(step, sigma, sigma * 1000)
                    for step, sigma in enumerate(sigmas[:-1])
                ]
            )
            result = run_policy(
                model,
                initial_latent=latent,
                conditioning=conditioning,
                schedule=schedule,
                policy=policy,
            )
            image = model.decode_latent(result.latent, height=height, width=width)
        return image, result.actors


def run_policy(
    model: Any,
    *,
    initial_latent: Any,
    conditioning: Any,
    schedule: Any,
    policy: Any,
) -> RolloutResult:
    import torch

    if policy.num_steps != len(schedule):
        raise ValueError("inference policy and schedule lengths differ")
    state = initial_latent
    actors = []
    started = perf_counter()
    with torch.no_grad():
        for step in range(len(schedule)):
            actor = policy.actor_at(step)
            transition = schedule.transition(step)
            velocity = model.predict_velocity(
                state, transition.current, conditioning, actor=actor
            )
            state = model.advance(state, velocity, transition)
            actors.append(actor)
    return RolloutResult(state, tuple(actors), perf_counter() - started)


def run_endpoint(
    model: Any,
    *,
    initial_latent: Any,
    conditioning: Any,
    schedule: Any,
    plan: EndpointExecutionPlan,
) -> RolloutResult:
    """Run an endpoint plan while preserving the official FLUX schedule."""

    if plan.num_steps != len(schedule):
        raise ValueError("endpoint plan and schedule lengths differ")
    state = initial_latent
    actors: list[str] = []
    started = perf_counter()
    import torch

    with torch.no_grad():
        for span in plan.spans:
            if span.execution == "native_step":
                for step in range(span.start, span.end):
                    transition = schedule.transition(step)
                    velocity = model.predict_velocity(
                        state,
                        transition.current,
                        conditioning,
                        actor=span.actor,
                    )
                    state = model.advance(state, velocity, transition)
                    actors.append(span.actor)
                continue
            if span.end != len(schedule):
                raise ValueError(
                    "endpoint skip must terminate at the FLUX schedule end"
                )
            transition = ScheduleTransition(
                current=schedule.transition(span.start).current,
                next_sigma=0.0,
            )
            velocity = model.predict_velocity(
                state,
                transition.current,
                conditioning,
                actor=span.actor,
            )
            state = model.advance(state, velocity, transition)
            actors.append(span.actor)
    return RolloutResult(
        state,
        tuple(actors),
        perf_counter() - started,
        tuple(span.to_dict() for span in plan.spans),
    )
