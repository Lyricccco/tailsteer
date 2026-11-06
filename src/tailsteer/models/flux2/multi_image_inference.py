"""Inference-only FLUX.2 multi-image conditioning and segmented rollout."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from .model import Flux2Conditioning


@dataclass(frozen=True)
class MultiImagePolicySegment:
    start: int
    end: int
    actor: str
    context: str = "full"

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid policy interval [{self.start}, {self.end})")
        if self.actor not in {"base", "refined"}:
            raise ValueError(f"invalid inference actor {self.actor!r}")
        if self.context not in {"full", "primary_only"}:
            raise ValueError(f"invalid conditioning context {self.context!r}")


class MultiImageInferencePolicy:
    def __init__(
        self,
        name: str,
        segments: Sequence[MultiImagePolicySegment],
        num_steps: int,
    ) -> None:
        self.name = str(name)
        self.segments = tuple(segments)
        self.num_steps = int(num_steps)
        expected = 0
        for segment in self.segments:
            if segment.start != expected:
                raise ValueError(f"policy {name!r} has a gap or overlap at {expected}")
            expected = segment.end
        if expected != self.num_steps:
            raise ValueError(
                f"policy {name!r} covers [0, {expected}), expected [0, {num_steps})"
            )

    def segment_at(self, step: int) -> MultiImagePolicySegment:
        for segment in self.segments:
            if segment.start <= step < segment.end:
                return segment
        raise IndexError(step)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "num_steps": self.num_steps,
            "segments": [segment.__dict__ for segment in self.segments],
        }


def build_multi_image_policies(
    config: Mapping[str, Any], *, num_steps: int
) -> dict[str, MultiImageInferencePolicy]:
    raw = config.get("policies")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("inference.policies must be a non-empty mapping")
    policies = {
        str(name): MultiImageInferencePolicy(
            str(name),
            [
                MultiImagePolicySegment(
                    start=int(segment["start"]),
                    end=int(segment["end"]),
                    actor=str(segment["actor"]),
                    context=str(segment.get("context", "full")),
                )
                for segment in value["segments"]
            ],
            num_steps,
        )
        for name, value in raw.items()
    }
    return policies


@dataclass(frozen=True)
class PreparedMultiImageConditioning:
    profiles: Mapping[str, Flux2Conditioning]
    condition_token_lengths: tuple[int, ...]
    condition_time_slots: tuple[int, ...]


def prepare_multi_image_conditioning(
    model: Any,
    *,
    primary_latent: Any,
    reference_latents: Sequence[Any],
    prompts: Sequence[str],
    height: int,
    width: int,
) -> PreparedMultiImageConditioning:
    """Extend the unchanged single-image payload with native FLUX.2 image slots."""

    import torch

    primary_only = model.prepare_conditioning(
        source_latent=primary_latent,
        prompts=prompts,
        height=height,
        width=width,
    )
    if not reference_latents:
        token_length = int(primary_only.payload["source_tokens"].shape[1])
        return PreparedMultiImageConditioning(
            {"full": primary_only, "primary_only": primary_only},
            (token_length,),
            (10,),
        )
    if primary_latent.ndim != 4:
        raise ValueError("primary latent must be rank four")
    batch_size = int(primary_latent.shape[0])
    condition_latents = [primary_latent, *reference_latents]
    packed = []
    token_lengths = []
    for index, latent in enumerate(condition_latents, 1):
        if latent.ndim != 4:
            raise ValueError(f"condition latent {index} must be rank four")
        if latent.shape[1] != primary_latent.shape[1]:
            raise ValueError("condition latent channel counts differ")
        if latent.shape[0] == 1 and batch_size != 1:
            latent = latent.expand(batch_size, -1, -1, -1)
        elif latent.shape[0] != batch_size:
            raise ValueError("condition latent batch sizes differ")
        tokens = model.pipe._pack_latents(latent)
        packed.append(tokens)
        token_lengths.append(int(tokens.shape[1]))

    source_tokens = torch.cat(packed, dim=1)
    source_ids = model.pipe._prepare_image_ids(
        [latent[:1] for latent in condition_latents]
    )
    source_ids = source_ids.to(primary_latent.device).expand(batch_size, -1, -1)
    if source_tokens.shape[:2] != source_ids.shape[:2]:
        raise RuntimeError("FLUX.2 condition tokens and image IDs differ in length")
    payload = dict(primary_only.payload)
    payload["source_tokens"] = source_tokens
    payload["source_ids"] = source_ids
    full = Flux2Conditioning(payload=payload, batch_size=batch_size)
    return PreparedMultiImageConditioning(
        profiles={"full": full, "primary_only": primary_only},
        condition_token_lengths=tuple(token_lengths),
        condition_time_slots=tuple(10 * index for index in range(1, len(packed) + 1)),
    )


@dataclass(frozen=True)
class MultiImageRolloutResult:
    latent: Any
    actors: tuple[str, ...]
    contexts: tuple[str, ...]
    elapsed_seconds: float


def run_multi_image_policy(
    model: Any,
    *,
    initial_latent: Any,
    conditionings: Mapping[str, Flux2Conditioning],
    schedule: Any,
    policy: MultiImageInferencePolicy,
) -> MultiImageRolloutResult:
    import torch

    if policy.num_steps != len(schedule):
        raise ValueError("inference policy and schedule lengths differ")
    state = initial_latent
    actors = []
    contexts = []
    started = perf_counter()
    with torch.no_grad():
        for step in range(len(schedule)):
            segment = policy.segment_at(step)
            try:
                conditioning = conditionings[segment.context]
            except KeyError as error:
                raise ValueError(
                    f"conditioning profile {segment.context!r} is unavailable"
                ) from error
            transition = schedule.transition(step)
            velocity = model.predict_velocity(
                state,
                transition.current,
                conditioning,
                actor=segment.actor,
            )
            state = model.advance(state, velocity, transition)
            actors.append(segment.actor)
            contexts.append(segment.context)
    return MultiImageRolloutResult(
        latent=state,
        actors=tuple(actors),
        contexts=tuple(contexts),
        elapsed_seconds=perf_counter() - started,
    )
