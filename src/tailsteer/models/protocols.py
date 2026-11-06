"""Small shared values for teacher-guided trajectory refinement.

The training algorithms operate on model-native tensors.  Model backends own
the interpretation of those tensors, their integration direction, and mask
layout; this module only defines the values exchanged at that boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrajectoryPoint:
    """One (possibly interpolated) point on a denoising trajectory."""

    step: float
    sigma: float
    timestep: float


@dataclass(frozen=True)
class PreparedTrajectoryBatch:
    """Model-native states plus the conditioning shared by both actors."""

    states: dict[str, Any]
    source_state: Any
    keep_mask: Any
    conditioning: Any
    prompts: tuple[str, ...]
    height: int
    width: int


@dataclass(frozen=True)
class TailVelocityBranches:
    """Named model-native velocities before backend-owned CFG composition."""

    values: Mapping[str, Any]
