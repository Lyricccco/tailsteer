"""Serializable transition schedules for teacher/preservation target mixing."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


def _linear(progress: float, _config: Mapping[str, Any]) -> float:
    return progress


def _constant(_progress: float, config: Mapping[str, Any]) -> float:
    return float(config.get("value", 1.0))


def _cosine(progress: float, _config: Mapping[str, Any]) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * progress)


def _smoothstep(progress: float, _config: Mapping[str, Any]) -> float:
    return progress * progress * (3.0 - 2.0 * progress)


def _power(progress: float, config: Mapping[str, Any]) -> float:
    exponent = float(config.get("exponent", 2.0))
    if exponent <= 0 or not math.isfinite(exponent):
        raise ValueError("power alpha schedule exponent must be finite and positive")
    return progress**exponent


def _odds(progress: float, config: Mapping[str, Any]) -> float:
    """Map progress through a single odds-ratio curvature parameter."""

    odds_ratio = float(config.get("odds_ratio", 1.0))
    if odds_ratio <= 0 or not math.isfinite(odds_ratio):
        raise ValueError("odds alpha schedule odds_ratio must be finite and positive")
    if progress <= 0.0:
        return 0.0
    if progress >= 1.0:
        return 1.0
    return progress / (progress + odds_ratio * (1.0 - progress))


def _piecewise(progress: float, config: Mapping[str, Any]) -> float:
    raw = config.get("knots")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TypeError("piecewise alpha schedule requires a knots sequence")
    knots = tuple((float(item[0]), float(item[1])) for item in raw)
    if len(knots) < 2:
        raise ValueError("piecewise alpha schedule requires at least two knots")
    if knots[0] != (0.0, 0.0) or knots[-1] != (1.0, 1.0):
        raise ValueError("piecewise alpha knots must start at (0,0) and end at (1,1)")
    if any(
        not 0.0 <= x <= 1.0
        or not 0.0 <= y <= 1.0
        or (index and x <= knots[index - 1][0])
        for index, (x, y) in enumerate(knots)
    ):
        raise ValueError("piecewise alpha knots must be finite and x-increasing")
    for (left_x, left_y), (right_x, right_y) in zip(knots, knots[1:], strict=False):
        if progress <= right_x:
            ratio = (progress - left_x) / (right_x - left_x)
            return left_y + ratio * (right_y - left_y)
    return 1.0


_SCHEDULES: dict[str, Callable[[float, Mapping[str, Any]], float]] = {
    "constant": _constant,
    "linear": _linear,
    "cosine": _cosine,
    "smoothstep": _smoothstep,
    "power": _power,
    "odds": _odds,
    "piecewise": _piecewise,
}


@dataclass(frozen=True)
class AlphaSchedule:
    name: str
    config: Mapping[str, Any]

    def __call__(self, progress: float) -> float:
        value = min(max(float(progress), 0.0), 1.0)
        result = float(_SCHEDULES[self.name](value, self.config))
        if not 0.0 <= result <= 1.0 or not math.isfinite(result):
            raise ValueError(f"alpha schedule {self.name!r} returned {result}")
        return result


def build_alpha_schedule(config: str | Mapping[str, Any]) -> AlphaSchedule:
    resolved = {"name": config} if isinstance(config, str) else dict(config)
    name = str(resolved.get("name", "cosine"))
    if name not in _SCHEDULES:
        raise ValueError(
            f"unknown alpha schedule {name!r}; available={sorted(_SCHEDULES)}"
        )
    schedule = AlphaSchedule(name, resolved)
    # Validate parameterized schedules during setup rather than during training.
    schedule(0.0)
    schedule(0.5)
    schedule(1.0)
    return schedule
