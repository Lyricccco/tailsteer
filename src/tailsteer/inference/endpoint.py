"""Execution plans for endpoint-preservation (A) inference.

The regular :mod:`tailsteer.inference.policies` module describes one actor
per native sampler interval.  Endpoint preservation is different: the model
is trained to predict one finite-difference velocity from its target point to
the terminal point.  This module keeps that variable-span execution contract
separate from the C/D actor policies.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

ENDPOINT_MODE_NAMES = ("base", "transition", "tailsteer")


@dataclass(frozen=True)
class EndpointSpan:
    """One contiguous execution span in actor-index coordinates."""

    start: int
    end: int
    actor: str
    execution: str
    calls: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid endpoint span [{self.start}, {self.end})")
        if self.actor not in {"base", "refined"}:
            raise ValueError(f"invalid endpoint actor {self.actor!r}")
        if self.execution not in {"native_step", "endpoint_skip"}:
            raise ValueError(f"invalid endpoint execution {self.execution!r}")
        if self.calls < 1:
            raise ValueError("endpoint span calls must be positive")
        expected = self.end - self.start if self.execution == "native_step" else 1
        if self.calls != expected:
            raise ValueError(
                f"endpoint span [{self.start}, {self.end}) with execution "
                f"{self.execution!r} expects {expected} calls, got {self.calls}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "actor": self.actor,
            "execution": self.execution,
            "calls": self.calls,
        }


@dataclass(frozen=True)
class EndpointExecutionPlan:
    """Validated actor and integration plan for one A-mode comparison."""

    name: str
    num_steps: int
    target_step: int
    target_actor_index: int
    spans: tuple[EndpointSpan, ...]

    def __post_init__(self) -> None:
        if self.name not in ENDPOINT_MODE_NAMES:
            raise ValueError(f"unknown endpoint mode {self.name!r}")
        if self.num_steps < 1:
            raise ValueError("endpoint num_steps must be positive")
        if not 0 <= self.target_actor_index < self.num_steps:
            raise ValueError("endpoint target actor index is outside the schedule")
        expected = 0
        for span in self.spans:
            if span.start != expected:
                raise ValueError(
                    f"endpoint plan {self.name!r} has a gap or overlap at {expected}"
                )
            expected = span.end
        if expected != self.num_steps:
            raise ValueError(
                f"endpoint plan {self.name!r} covers [0, {expected}), expected "
                f"[0, {self.num_steps})"
            )
        if self.name == "base" and any(
            span.execution != "native_step" or span.actor != "base"
            for span in self.spans
        ):
            raise ValueError(
                "pretrained endpoint mode must use native pretrained steps"
            )
        if self.name != "base":
            endpoint = self.spans[-1]
            if (
                endpoint.execution != "endpoint_skip"
                or endpoint.start != self.target_actor_index
            ):
                raise ValueError(
                    f"endpoint mode {self.name!r} must end in a terminal skip span"
                )

    @property
    def expected_calls(self) -> int:
        return sum(span.calls for span in self.spans)

    @property
    def actors(self) -> tuple[str, ...]:
        return tuple(span.actor for span in self.spans for _ in range(span.calls))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "num_steps": self.num_steps,
            "target_step": self.target_step,
            "target_actor_index": self.target_actor_index,
            "expected_calls": self.expected_calls,
            "spans": [span.to_dict() for span in self.spans],
        }

    def actor_policy(self) -> EndpointActorSequence:
        """Return a small actor-policy view for existing actor controllers."""

        return EndpointActorSequence(self.name, self.actors)


@dataclass(frozen=True)
class EndpointActorSequence:
    """Compatibility policy consumed by transformer actor controllers."""

    name: str
    _actors: tuple[str, ...]

    @property
    def num_steps(self) -> int:
        return len(self._actors)

    def actor_at(self, step: int) -> str:
        return self._actors[step]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "num_steps": self.num_steps,
            "actors": list(self._actors),
        }


def _native_span(start: int, end: int, actor: str) -> EndpointSpan | None:
    if start == end:
        return None
    return EndpointSpan(start, end, actor, "native_step", end - start)


def build_endpoint_plans(
    *,
    num_steps: int,
    target_step: int,
    target_actor_index: int,
    selected: Sequence[str] | None = None,
) -> dict[str, EndpointExecutionPlan]:
    """Build the three A-mode plans from a backend-resolved actor index."""

    names = tuple(selected or ENDPOINT_MODE_NAMES)
    if not names:
        raise ValueError("endpoint inference must select at least one mode")
    if len(set(names)) != len(names):
        raise ValueError("endpoint modes must be unique")
    unknown = set(names) - set(ENDPOINT_MODE_NAMES)
    if unknown:
        raise ValueError(f"unknown endpoint modes: {sorted(unknown)}")
    if "base" not in names:
        raise ValueError("endpoint comparison requires the pretrained mode")
    if num_steps < 1:
        raise ValueError("endpoint num_steps must be positive")
    if not 0 <= target_actor_index < num_steps:
        raise ValueError("endpoint target actor index is outside the schedule")

    result: dict[str, EndpointExecutionPlan] = {}
    for name in names:
        if name == "base":
            spans = (EndpointSpan(0, num_steps, "base", "native_step", num_steps),)
        else:
            prefix_actor = "base" if name == "transition" else "refined"
            prefix = _native_span(0, target_actor_index, prefix_actor)
            raw_spans = [] if prefix is None else [prefix]
            raw_spans.append(
                EndpointSpan(
                    target_actor_index,
                    num_steps,
                    "refined",
                    "endpoint_skip",
                    1,
                )
            )
            spans = tuple(raw_spans)
        result[name] = EndpointExecutionPlan(
            name=name,
            num_steps=num_steps,
            target_step=int(target_step),
            target_actor_index=int(target_actor_index),
            spans=tuple(spans),
        )
    return result


def plans_from_mapping(
    mapping: Mapping[str, Any], *, num_steps: int
) -> dict[str, EndpointExecutionPlan]:
    """Deserialize plans stored in a resolved inference config."""

    if not mapping:
        raise ValueError("endpoint execution requires at least one plan")
    result: dict[str, EndpointExecutionPlan] = {}
    for name, raw in mapping.items():
        if not isinstance(raw, Mapping):
            raise TypeError(f"endpoint plan {name!r} must be a mapping")
        plan_name = str(name)
        if "name" in raw and str(raw["name"]) != plan_name:
            raise ValueError(
                f"endpoint plan key/name mismatch: {plan_name!r} != {raw['name']!r}"
            )
        spans = tuple(
            EndpointSpan(
                int(item["start"]),
                int(item["end"]),
                str(item["actor"]),
                str(item["execution"]),
                int(item["calls"]),
            )
            for item in raw.get("spans", ())
        )
        plan = EndpointExecutionPlan(
            name=plan_name,
            num_steps=int(raw.get("num_steps", num_steps)),
            target_step=int(raw["target_step"]),
            target_actor_index=int(raw["target_actor_index"]),
            spans=spans,
        )
        if plan.num_steps != num_steps:
            raise ValueError(
                f"endpoint plan {name!r} has num_steps={plan.num_steps}, "
                f"runner expects {num_steps}"
            )
        expected_calls = raw.get("expected_calls")
        if expected_calls is not None and int(expected_calls) != plan.expected_calls:
            raise ValueError(
                f"endpoint plan {name!r} has expected_calls={expected_calls}, "
                f"computed {plan.expected_calls}"
            )
        result[plan_name] = plan
    if "base" not in result:
        raise ValueError("endpoint comparison requires a pretrained plan")
    return result
