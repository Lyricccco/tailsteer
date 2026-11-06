"""Small model-neutral schedule values used by image-edit trajectory caches."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

# One-based cache anchors for SenseNova's 50 denoising forwards.
DENSE_50_STEP_FORWARD_POINTS = (
    1,
    7,
    14,
    23,
    29,
    35,
    36,
    38,
    40,
    42,
    45,
    47,
    49,
    50,
)


@dataclass(frozen=True)
class SchedulePoint:
    step_index: int
    sigma: float
    timestep: float

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("step_index cannot be negative")
        if self.sigma < 0 or not math.isfinite(self.sigma):
            raise ValueError("sigma must be finite and non-negative")
        if not math.isfinite(self.timestep):
            raise ValueError("timestep must be finite")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DiscreteSchedule:
    def __init__(
        self,
        points: Iterable[SchedulePoint],
        *,
        family: str,
        terminal_sigma: float = 0.0,
    ) -> None:
        self._points = tuple(points)
        self.family = str(family)
        self.terminal_sigma = float(terminal_sigma)
        if not self.family or not self._points:
            raise ValueError("schedule family and points must be non-empty")
        if any(point.step_index != index for index, point in enumerate(self._points)):
            raise ValueError("schedule points must use consecutive zero-based indices")
        if self.terminal_sigma < 0 or not math.isfinite(self.terminal_sigma):
            raise ValueError("terminal sigma must be finite and non-negative")

    def __len__(self) -> int:
        return len(self._points)

    def point(self, step_index: int) -> SchedulePoint:
        return self._points[step_index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "family": self.family,
            "terminal_sigma": self.terminal_sigma,
            "points": [point.to_dict() for point in self._points],
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()
