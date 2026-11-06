"""Geometry for teacher-guided velocity targets.

The training algorithm decides where a state comes from (offline cache replay
or a detached block rollout).  This module decides which geometric target is
applied at that state. Keeping the two concerns separate lets all recipes and
their rollout policies share the same target implementation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from tailsteer.cache.trajectory_dataset import CachedPathReference
from tailsteer.training.endpoint_targets import (
    SOURCE_KEEP_TERMINAL_EDIT,
    endpoint_goal_and_mask,
)

TERMINAL_BRIDGE = "terminal_bridge"
INSTANT_VELOCITY = "instant_velocity"
STRAIGHT_TAIL = "straight_tail"
INSTANT_ENDPOINT_TARGETS = ("source", SOURCE_KEEP_TERMINAL_EDIT)
CACHED_PATH = "cached_path"
TRACKING_PATH = "tracking_path"
TARGET_GEOMETRIES = frozenset(
    {TERMINAL_BRIDGE, INSTANT_VELOCITY, STRAIGHT_TAIL, CACHED_PATH, TRACKING_PATH}
)


@dataclass(frozen=True)
class TargetGeometryResult:
    """Velocity target and native integration interval for one target point."""

    velocity: Any
    delta: float
    target_state: Any | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def target_velocity(self) -> Any:
        """Descriptive alias for callers that use the geometry vocabulary."""

        return self.velocity


@dataclass(frozen=True)
class TargetGeometryContext:
    current: Any
    source: Any
    keep: Any
    point: Any
    following_point: Any | None
    reference: CachedPathReference | None
    alpha: float
    alpha_fn: Any
    preservation_scale: float
    model: Any
    terminal: Any | None = None
    anchor: Any | None = None
    anchor_point: Any | None = None


class TargetGeometry:
    name = "unknown"
    requires_reference = False
    requires_terminal = False
    requires_anchor = False

    def build(
        self, teacher: Any, context: TargetGeometryContext
    ) -> TargetGeometryResult:
        raise NotImplementedError


class TerminalBridgeGeometry(TargetGeometry):
    """Legacy terminal bridge from the current state to source state."""

    name = TERMINAL_BRIDGE

    def build(
        self, teacher: Any, context: TargetGeometryContext
    ) -> TargetGeometryResult:
        delta = float(context.model.tail_terminal_delta(context.point))
        if delta == 0.0:
            raise ValueError("teacher-guided target cannot be evaluated at terminal")
        preservation = (context.source - context.current) / delta
        target = teacher + (
            context.alpha
            * context.preservation_scale
            * context.keep
            * (preservation - teacher)
        )
        return TargetGeometryResult(
            target,
            delta,
            target_state=context.current.float() + delta * target,
        )


class InstantVelocityGeometry(TargetGeometry):
    """Full-state endpoint feedback evaluated at the actual training state.

    Masking defines only the goal: g = M*source + (1-M)*cached_terminal.
    With remaining native interval R, line = (g-current)/R and
    target = (1-alpha*scale)*teacher + alpha*scale*line. No reference path
    or extra mask is applied to this velocity. Endpoint arrival under Euler
    is exact only at the last forward, with alpha=scale=1 and exact prediction.
    """

    name = INSTANT_VELOCITY

    def __init__(
        self,
        endpoint_mode: str = SOURCE_KEEP_TERMINAL_EDIT,
        velocity_scale: float = 1.0,
    ) -> None:
        if not isinstance(endpoint_mode, str):
            raise TypeError("instant_velocity endpoint.mode must be a string")
        if endpoint_mode not in INSTANT_ENDPOINT_TARGETS:
            raise ValueError(
                f"instant_velocity endpoint.mode must be one of "
                f"{INSTANT_ENDPOINT_TARGETS}"
            )
        if isinstance(velocity_scale, bool) or not isinstance(
            velocity_scale, (int, float)
        ):
            raise TypeError("velocity_scale must be a real number")
        if not math.isfinite(velocity_scale) or velocity_scale < 0:
            raise ValueError("velocity_scale must be finite and non-negative")
        self.endpoint_mode = endpoint_mode
        self.velocity_scale = float(velocity_scale)
        self.requires_terminal = endpoint_mode == SOURCE_KEEP_TERMINAL_EDIT

    def line(self, goal: Any, context: TargetGeometryContext) -> Any:
        delta = float(context.model.tail_terminal_delta(context.point))
        return (goal - context.current.float()) / delta

    def build(
        self, teacher: Any, context: TargetGeometryContext
    ) -> TargetGeometryResult:
        delta = float(context.model.tail_terminal_delta(context.point))
        if delta == 0.0:
            raise ValueError("teacher-guided target cannot be evaluated at terminal")
        goal = context.source.float()
        if self.requires_terminal:
            if context.terminal is None:
                raise ValueError("instant_velocity requires a cached terminal state")
            goal, _ = endpoint_goal_and_mask(
                goal, context.keep, terminal=context.terminal.float()
            )
        goal = goal.detach()
        line = self.line(goal, context)
        mix = context.alpha * self.velocity_scale
        target = (1.0 - mix) * teacher + mix * line
        target_state = context.current.float() + delta * target
        return TargetGeometryResult(
            target,
            delta,
            target_state=target_state,
            metrics={
                "target_geometry_instant_velocity": target.new_tensor(
                    float(self.name == INSTANT_VELOCITY)
                ),
                "target_geometry_straight_tail": target.new_tensor(
                    float(self.name == STRAIGHT_TAIL)
                ),
                "target_velocity_scale": target.new_tensor(self.velocity_scale),
                "instant_line_velocity_rms": line.square().mean().sqrt().detach(),
                "instant_correction_rms": (
                    (target - teacher).square().mean().sqrt().detach()
                ),
                "instant_target_endpoint_mse": (
                    (target_state - goal).square().mean().detach()
                ),
            },
        )


class StraightTailGeometry(InstantVelocityGeometry):
    """B ablation: keep the anchor-to-goal direction fixed throughout rollout."""

    name = STRAIGHT_TAIL
    requires_anchor = True

    def line(self, goal: Any, context: TargetGeometryContext) -> Any:
        if context.anchor is None or context.anchor_point is None:
            raise ValueError("straight_tail requires the cached anchor and its point")
        delta = float(context.model.tail_terminal_delta(context.anchor_point))
        if delta == 0:
            raise ValueError("straight_tail anchor must be a model forward")
        return (goal - context.anchor.float()) / delta


def requires_teacher(transition: Mapping[str, Any], target: Mapping[str, Any]) -> bool:
    """Decide once per configuration, never independently per FSDP rank/step."""
    alpha = transition.get("alpha_schedule", {})
    return not (
        isinstance(alpha, Mapping)
        and alpha.get("name") == "constant"
        and float(alpha.get("value", 1.0)) == 1.0
        and target.get("mode", "composed") == "composed"
        and target.get("geometry") in {INSTANT_VELOCITY, STRAIGHT_TAIL}
        and float(target.get("velocity_scale", 1.0)) == 1.0
    )


class CachedPathGeometry(TargetGeometry):
    """Fixed cached trajectory with a smoothly introduced terminal correction.

    Let ``q_k`` and ``q_{k+1}`` be cached teacher states and ``q_T`` the cached
    terminal state.  The reference path is

    ``y_k = q_k + alpha_k * M * (source - q_T)``.

    The local target is the native velocity ``(y_{k+1}-y_k)/delta``.  On changed
    regions the frozen teacher remains the target; on kept regions this path
    target is blended with the teacher by ``preservation_scale``.  The latter
    keeps the existing target-config knob meaningful and avoids changing the
    default terminal-bridge behavior when this geometry is not selected.
    """

    name = CACHED_PATH
    requires_reference = True

    @staticmethod
    def _path_components(
        context: TargetGeometryContext,
    ) -> tuple[Any, float, Any]:
        import torch

        reference = context.reference
        following = context.following_point
        if reference is None or following is None:
            raise ValueError(
                "cached_path target requires current/following cached reference states"
            )
        delta_method = getattr(context.model, "tail_step_delta", None)
        if not callable(delta_method):
            raise TypeError(
                f"{type(context.model).__name__} does not expose tail_step_delta"
            )
        delta = float(delta_method(context.point, following))
        if delta == 0.0:
            raise ValueError("cached_path target has a zero integration interval")

        device = context.current.device
        cached_current = reference.current_state.to(device=device, dtype=torch.float32)
        cached_following = reference.following_state.to(
            device=device, dtype=torch.float32
        )
        cached_terminal = reference.terminal_state.to(
            device=device, dtype=torch.float32
        )
        correction = context.keep * (context.source.float() - cached_terminal)
        following_alpha = float(context.alpha_fn(following))
        path_current = cached_current + context.alpha * correction
        path_following = cached_following + following_alpha * correction
        path_velocity = (path_following - path_current) / delta
        return path_velocity, delta, path_current

    @staticmethod
    def _metrics(
        context: TargetGeometryContext,
        *,
        delta: float,
        deviation: Any | None = None,
        retention: float = 0.0,
    ) -> dict[str, Any]:
        import torch

        if deviation is None:
            tracking_deviation = context.current.new_tensor(0.0, dtype=torch.float32)
        else:
            tracking_deviation = deviation.float().square().flatten(1).mean().sqrt()
        return {
            "path_beta": context.current.new_tensor(context.alpha, dtype=torch.float32),
            "path_delta": context.current.new_tensor(delta, dtype=torch.float32),
            "tracking_deviation": tracking_deviation.detach(),
            "tracking_retention": context.current.new_tensor(
                retention, dtype=torch.float32
            ),
        }

    def build(
        self, teacher: Any, context: TargetGeometryContext
    ) -> TargetGeometryResult:
        path_velocity, delta, _path_current = self._path_components(context)
        target = teacher + context.preservation_scale * context.keep * (
            path_velocity - teacher
        )
        return TargetGeometryResult(
            target,
            delta,
            target_state=context.current.float() + delta * target,
            metrics=self._metrics(context, delta=delta),
        )


class TrackingPathGeometry(CachedPathGeometry):
    """Cached path target with a contraction toward the reference path."""

    name = TRACKING_PATH

    def __init__(self, retention: float) -> None:
        if (
            isinstance(retention, bool)
            or not math.isfinite(retention)
            or not 0.0 <= retention <= 1.0
        ):
            raise ValueError("tracking.retention must be finite and in [0, 1]")
        self.retention = float(retention)

    def build(
        self, teacher: Any, context: TargetGeometryContext
    ) -> TargetGeometryResult:
        path_velocity, delta, path_current = self._path_components(context)
        target = teacher + context.preservation_scale * context.keep * (
            path_velocity - teacher
        )
        deviation = context.current.float() - path_current
        target = target - ((1.0 - self.retention) / delta) * deviation
        return TargetGeometryResult(
            target,
            delta,
            target_state=context.current.float() + delta * target,
            metrics=self._metrics(
                context,
                delta=delta,
                deviation=deviation,
                retention=self.retention,
            ),
        )


def build_target_geometry(config: Mapping[str, Any] | None) -> TargetGeometry:
    """Build a validated geometry object; omitted config keeps legacy behavior."""

    values = {} if config is None else dict(config)
    name = str(values.get("geometry", TERMINAL_BRIDGE))
    if name != TRACKING_PATH and "tracking" in values:
        raise ValueError(
            "teacher-guided target tracking config requires geometry tracking_path"
        )
    if (
        name not in {INSTANT_VELOCITY, STRAIGHT_TAIL}
        and {"endpoint", "velocity_scale"} & values.keys()
    ):
        raise ValueError("endpoint/velocity_scale require geometry instant_velocity")
    if name == TERMINAL_BRIDGE:
        return TerminalBridgeGeometry()
    if name in {INSTANT_VELOCITY, STRAIGHT_TAIL}:
        endpoint = values.get("endpoint", {})
        if not isinstance(endpoint, Mapping):
            raise TypeError("instant_velocity endpoint must be a mapping")
        if set(endpoint) - {"mode"}:
            raise ValueError("instant_velocity endpoint only accepts mode")
        if values.get("preservation_scale", 1.0) != 1.0:
            raise ValueError(
                "instant_velocity uses velocity_scale, not preservation_scale"
            )
        geometry = (
            StraightTailGeometry if name == STRAIGHT_TAIL else InstantVelocityGeometry
        )
        return geometry(
            endpoint_mode=endpoint.get("mode", SOURCE_KEEP_TERMINAL_EDIT),
            velocity_scale=values.get("velocity_scale", 1.0),
        )
    if name == CACHED_PATH:
        return CachedPathGeometry()
    if name == TRACKING_PATH:
        raw_tracking = values.get("tracking", {})
        if not isinstance(raw_tracking, Mapping):
            raise TypeError("tracking_path target requires a tracking mapping")
        unknown = set(raw_tracking) - {"retention"}
        if unknown:
            raise ValueError(
                "tracking_path only accepts tracking.retention; "
                f"unknown={sorted(unknown)}"
            )
        retention = raw_tracking.get("retention", 0.5)
        if isinstance(retention, bool) or not isinstance(retention, (int, float)):
            raise TypeError("tracking.retention must be a real number")
        return TrackingPathGeometry(float(retention))
    raise ValueError(
        "teacher_guided_target geometry must be one of "
        f"{sorted(TARGET_GEOMETRIES)}, got {name!r}"
    )
