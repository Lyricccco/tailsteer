"""Small, validated mapping from denoising intervals to model actors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PolicySegment:
    start: int
    end: int
    actor: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid policy interval [{self.start}, {self.end})")
        if self.actor not in {"base", "refined"}:
            raise ValueError(f"invalid inference actor {self.actor!r}")


class InferencePolicy:
    def __init__(self, name: str, segments: Sequence[PolicySegment], num_steps: int):
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

    def actor_at(self, step: int) -> str:
        for segment in self.segments:
            if segment.start <= step < segment.end:
                return segment.actor
        raise IndexError(step)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "num_steps": self.num_steps,
            "segments": [segment.__dict__ for segment in self.segments],
        }


def build_policies(
    config: dict[str, Any], *, num_steps: int
) -> dict[str, InferencePolicy]:
    raw = config.get("policies")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("inference.policies must be a non-empty mapping")
    policies = {
        str(name): InferencePolicy(
            str(name),
            [
                PolicySegment(
                    int(segment["start"]),
                    int(segment["end"]),
                    str(segment["actor"]),
                )
                for segment in value["segments"]
            ],
            num_steps,
        )
        for name, value in raw.items()
    }
    return policies
