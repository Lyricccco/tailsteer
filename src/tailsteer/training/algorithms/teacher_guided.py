"""Shared teacher-guided target for offline and block-local state sources."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from tailsteer.cache.trajectory_dataset import (
    CachedPathReferenceResolver,
    TrajectoryContract,
    materialize_trajectory_state,
)
from tailsteer.models.protocols import (
    PreparedTrajectoryBatch,
    TailVelocityBranches,
    TrajectoryPoint,
)
from tailsteer.training.alpha_schedules import build_alpha_schedule
from tailsteer.training.target_geometry import (
    CACHED_PATH,
    INSTANT_VELOCITY,
    STRAIGHT_TAIL,
    TERMINAL_BRIDGE,
    TRACKING_PATH,
    TargetGeometryContext,
    build_target_geometry,
    requires_teacher,
)

from .common import AlgorithmResult

V_TARGET = "v-target"
X_TARGET = "x-target"
TARGET_LOSSES = frozenset({V_TARGET, X_TARGET})


def validate_endpoint_loss_weight(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("endpoint_loss_weight must be a real number")
    if not math.isfinite(value) or value < 0:
        raise ValueError("endpoint_loss_weight must be finite and non-negative")
    return float(value)


class TeacherGuidedTarget:
    def __init__(
        self,
        model: Any,
        contract: TrajectoryContract,
        transition: Mapping[str, Any],
        *,
        target_config: Mapping[str, Any] | None = None,
        weight: float = 1.0,
    ) -> None:
        self.model = model
        self.contract = contract
        self.start_step = int(transition["start_step"])
        self.supervision_step = int(transition["supervision_step"])
        self.alpha_schedule = build_alpha_schedule(
            transition.get("alpha_schedule", {"name": "cosine"})
        )
        constant_alpha = self.alpha_schedule.name == "constant"
        if self.start_step > self.supervision_step or (
            self.start_step == self.supervision_step and not constant_alpha
        ):
            raise ValueError("transition start_step must precede supervision_step")
        self.start_point = contract.point(self.start_step)
        self.supervision_point = contract.point(self.supervision_step)
        if self.supervision_point.state_kind == "final":
            raise ValueError("supervision_step must be a model forward")
        if self.start_point.state_kind == "final":
            raise ValueError("start_step must be a model forward")
        if (
            not constant_alpha
            and self.start_point.sigma <= self.supervision_point.sigma
        ):
            raise ValueError("transition points must use decreasing remaining sigma")
        self.requires_teacher = requires_teacher(transition, target_config or {})
        self.weight = float(weight)
        if self.weight < 0:
            raise ValueError("teacher-guided loss weight must be non-negative")
        self.endpoint_loss_weight = validate_endpoint_loss_weight(
            (target_config or {}).get("endpoint_loss_weight", 1.0)
        )
        (
            self.target_loss,
            self.target_mode,
            self.branch_names,
            self.branch_weights,
            self.preservation_scale,
        ) = self._parse_target_config(target_config)
        self.target_geometry = build_target_geometry(target_config)
        if (
            self.target_geometry.name in {INSTANT_VELOCITY, STRAIGHT_TAIL}
            and self.target_loss != V_TARGET
        ):
            raise ValueError("instant_velocity requires v-target supervision")
        self.terminal_point = None
        if self.target_geometry.requires_terminal:
            terminal = contract.terminal_point
            self.terminal_point = TrajectoryPoint(
                float(terminal.step_index), terminal.sigma, terminal.timestep
            )

    def _parse_target_config(
        self, config: Mapping[str, Any] | None
    ) -> tuple[str, str, tuple[str, ...], dict[str, float], float]:
        values = {} if config is None else dict(config)
        target_loss = str(values.get("loss", V_TARGET))
        if target_loss not in TARGET_LOSSES:
            raise ValueError(
                "teacher_guided_target loss must be "
                f"one of {sorted(TARGET_LOSSES)}, got {target_loss!r}"
            )
        raw_scale = values.get("preservation_scale", 1.0)
        if isinstance(raw_scale, bool) or not isinstance(raw_scale, (int, float)):
            raise TypeError("preservation_scale must be a real number")
        preservation_scale = float(raw_scale)
        if (
            not math.isfinite(preservation_scale)
            or not 0.0 <= preservation_scale <= 1.0
        ):
            raise ValueError("preservation_scale must be finite and in [0, 1]")
        mode = str(values.get("mode", "composed"))
        if mode not in {"composed", "branchwise"}:
            raise ValueError(
                "teacher_guided_target mode must be composed or branchwise"
            )
        if mode == "composed":
            if values.get("branch_weights") is not None:
                raise ValueError("branch_weights are only valid for branchwise targets")
            return target_loss, mode, (), {}, preservation_scale

        branch_names_method = getattr(self.model, "tail_velocity_branch_names", None)
        if branch_names_method is None:
            raise ValueError(
                "branchwise teacher-guided targets require a model backend "
                "with tail_velocity_branch_names"
            )
        branch_names = tuple(str(name) for name in branch_names_method())
        if len(branch_names) < 2 or len(set(branch_names)) != len(branch_names):
            raise ValueError(
                "branchwise teacher-guided targets require at least two "
                "distinct velocity branches"
            )
        configured = values.get("branch_weights")
        weights = (
            {name: 1.0 / len(branch_names) for name in branch_names}
            if configured is None
            else {str(name): float(value) for name, value in dict(configured).items()}
        )
        if set(weights) != set(branch_names):
            raise ValueError(
                "branch_weights must exactly match model branches: "
                f"expected={sorted(branch_names)} actual={sorted(weights)}"
            )
        if any(value < 0 for value in weights.values()) or not any(
            value > 0 for value in weights.values()
        ):
            raise ValueError(
                "branch_weights must be non-negative with at least one positive value"
            )
        return target_loss, mode, branch_names, weights, preservation_scale

    def alpha(self, point: TrajectoryPoint) -> float:
        if self.alpha_schedule.name == "constant":
            return self.alpha_schedule(0.0)
        if (
            self.target_geometry.name == INSTANT_VELOCITY
            and float(point.step) == self.supervision_step
        ):
            return 1.0
        progress = (self.start_point.sigma - float(point.sigma)) / (
            self.start_point.sigma - self.supervision_point.sigma
        )
        return self.alpha_schedule(progress)

    def prepare_batch(
        self, batch: Mapping[str, Any], materialized: Mapping[str, Any]
    ) -> PreparedTrajectoryBatch:
        """Prepare native states, including the optional endpoint and anchor.

        The endpoint passes through the same native layout/dtype conversion as
        the input state. D keeps this cached goal fixed while its state rolls out.
        """

        state_keys = ("latent",)
        if self.terminal_point is not None:
            terminal = materialize_trajectory_state(batch, self.terminal_point)
            materialized = {**materialized, "terminal_latent": terminal["latent"]}
            state_keys += ("terminal_latent",)
        if self.target_geometry.requires_anchor:
            point = TrajectoryPoint(
                float(self.start_step),
                self.start_point.sigma,
                self.start_point.timestep,
            )
            anchor = materialize_trajectory_state(batch, point)
            materialized = {**materialized, "anchor_latent": anchor["latent"]}
            state_keys += ("anchor_latent",)
        return self.model.prepare_tail_batch(materialized, state_keys=state_keys)

    @staticmethod
    def _per_sample(value: Any) -> Any:
        return value.flatten(1).mean(dim=1)

    @staticmethod
    def _masked_per_sample(value: Any, mask: Any) -> Any:
        expanded = mask.expand_as(value).float()
        dimensions = tuple(range(1, value.ndim))
        denominator = expanded.sum(dim=dimensions).clamp_min(1.0)
        return (value * expanded).sum(dim=dimensions) / denominator

    def _validate_branches(self, prediction: Any, *, actor: str) -> dict[str, Any]:
        if not isinstance(prediction, TailVelocityBranches):
            raise TypeError(f"{actor} branch prediction must be TailVelocityBranches")
        values = dict(prediction.values)
        if set(values) != set(self.branch_names):
            raise ValueError(
                f"{actor} velocity branches differ from backend declaration: "
                f"expected={sorted(self.branch_names)} actual={sorted(values)}"
            )
        return values

    def _actor_memory_boundary(self) -> None:
        boundary = getattr(self.model, "actor_memory_boundary", None)
        if callable(boundary):
            boundary()

    def cached_path_inputs(
        self,
        batch: Mapping[str, Any],
        schedule: Mapping[int, TrajectoryPoint],
        point: TrajectoryPoint,
    ) -> tuple[TrajectoryPoint | None, Any | None]:
        """Resolve the dense following point and cache states when requested."""

        if not self.target_geometry.requires_reference:
            return None, None
        next_step = int(point.step) + 1
        following = schedule.get(next_step)
        if following is None and next_step == self.contract.terminal_point.step_index:
            terminal = self.contract.terminal_point
            following = TrajectoryPoint(
                float(terminal.step_index), terminal.sigma, terminal.timestep
            )
        if following is None:
            raise ValueError(
                "cached_path target requires a live following schedule point: "
                f"step={next_step}"
            )
        reference = CachedPathReferenceResolver(batch).resolve(point, following)
        return following, reference

    def compute(
        self,
        prepared: PreparedTrajectoryBatch,
        state: Any,
        point: TrajectoryPoint,
        *,
        following_point: TrajectoryPoint | None = None,
        reference: Any | None = None,
    ) -> AlgorithmResult:
        """Weight the last supervised forward locally, before rank reduction.

        Targets and raw error metrics remain unchanged for both composed and
        branchwise supervision. The terminal state itself is not a forward.
        """
        result = self._compute_unweighted(
            prepared,
            state,
            point,
            following_point=following_point,
            reference=reference,
        )
        scale = (
            self.endpoint_loss_weight
            if float(point.step) == self.supervision_step
            else 1.0
        )
        return AlgorithmResult(
            loss=result.loss * scale,
            per_sample=result.per_sample * scale,
            metrics={
                **result.metrics,
                "loss": result.metrics["loss"] * scale,
                "endpoint_loss_weight": result.loss.detach().new_tensor(scale),
            },
        )

    def _compute_unweighted(
        self,
        prepared: PreparedTrajectoryBatch,
        state: Any,
        point: TrajectoryPoint,
        *,
        following_point: TrajectoryPoint | None = None,
        reference: Any | None = None,
    ) -> AlgorithmResult:
        import torch

        current = state.float()
        source = prepared.source_state.float()
        terminal_delta = float(self.model.tail_terminal_delta(point))
        if terminal_delta == 0.0:
            raise ValueError("teacher-guided target cannot be evaluated at terminal")
        keep = self.model.tail_keep_mask(prepared.keep_mask, current).float()
        alpha = self.alpha(point)
        effective_alpha = alpha * getattr(
            self.target_geometry, "velocity_scale", self.preservation_scale
        )

        if self.target_mode == "branchwise":
            return self._compute_branchwise(
                prepared,
                state,
                point,
                current=current,
                source=source,
                keep=keep,
                terminal_delta=terminal_delta,
                alpha=alpha,
                effective_alpha=effective_alpha,
                following_point=following_point,
                reference=reference,
            )

        self._actor_memory_boundary()
        with torch.no_grad():
            teacher = (
                self.model.predict_tail_velocity(
                    state, point, prepared.conditioning, actor="base"
                ).float()
                if self.requires_teacher
                else torch.zeros_like(current)
            )
            geometry = self.target_geometry.build(
                teacher,
                TargetGeometryContext(
                    current=current,
                    source=source,
                    keep=keep,
                    point=point,
                    following_point=following_point,
                    reference=reference,
                    alpha=alpha,
                    alpha_fn=self.alpha,
                    preservation_scale=self.preservation_scale,
                    model=self.model,
                    terminal=prepared.states.get("terminal_latent"),
                    anchor=prepared.states.get("anchor_latent"),
                    anchor_point=TrajectoryPoint(
                        float(self.start_step),
                        self.start_point.sigma,
                        self.start_point.timestep,
                    ),
                ),
            )
            target = geometry.velocity
        self._actor_memory_boundary()
        student = self.model.predict_tail_velocity(
            state, point, prepared.conditioning, actor="refined"
        ).float()

        velocity_error = (student - target).square()
        velocity_per_sample = self._per_sample(velocity_error)
        predicted_terminal = current + student * geometry.delta
        target_terminal = current + target * geometry.delta
        state_per_sample = self._per_sample(
            (predicted_terminal - target_terminal).square()
        )
        per_sample = (
            velocity_per_sample if self.target_loss == V_TARGET else state_per_sample
        )
        raw_loss = per_sample.mean()
        prior_mse = self._per_sample((student - teacher).square()).mean()
        prior_state_mse = prior_mse * geometry.delta**2
        preservation_target = (
            (source - current) / terminal_delta
            if self.target_geometry.name == TERMINAL_BRIDGE
            else target
        )
        preservation_mse = self._masked_per_sample(
            (student - preservation_target).square(), keep
        ).mean()
        predicted_terminal_for_metric = current + student * terminal_delta
        terminal_mse = self._masked_per_sample(
            (predicted_terminal_for_metric - source).square(), keep
        ).mean()
        return AlgorithmResult(
            loss=raw_loss * self.weight,
            per_sample=per_sample * self.weight,
            metrics={
                "loss": raw_loss.detach() * self.weight,
                "target_velocity_mse": velocity_per_sample.mean().detach(),
                "target_state_mse": state_per_sample.mean().detach(),
                **(
                    {
                        "prior_velocity_mse": prior_mse.detach(),
                        "prior_state_mse": prior_state_mse.detach(),
                    }
                    if self.requires_teacher
                    else {}
                ),
                "teacher_evaluated": current.new_tensor(float(self.requires_teacher)),
                "keep_preservation_velocity_mse": preservation_mse.detach(),
                "keep_terminal_mse": terminal_mse.detach(),
                "target_loss_is_x": state.new_tensor(
                    float(self.target_loss == X_TARGET), dtype=torch.float32
                ),
                "target_alpha": state.new_tensor(alpha, dtype=torch.float32),
                "target_preservation_scale": state.new_tensor(
                    self.preservation_scale, dtype=torch.float32
                ),
                "target_effective_alpha": state.new_tensor(
                    effective_alpha, dtype=torch.float32
                ),
                "target_geometry_cached_path": state.new_tensor(
                    float(self.target_geometry.name == CACHED_PATH),
                    dtype=torch.float32,
                ),
                "target_geometry_tracking_path": state.new_tensor(
                    float(self.target_geometry.name == TRACKING_PATH),
                    dtype=torch.float32,
                ),
                "target_geometry_terminal_bridge": state.new_tensor(
                    float(self.target_geometry.name == TERMINAL_BRIDGE),
                    dtype=torch.float32,
                ),
                "target_step_delta": state.new_tensor(
                    geometry.delta, dtype=torch.float32
                ),
                "path_beta": state.new_tensor(alpha, dtype=torch.float32),
                "path_delta": state.new_tensor(geometry.delta, dtype=torch.float32),
                "tracking_deviation": state.new_tensor(0.0, dtype=torch.float32),
                "tracking_retention": state.new_tensor(0.0, dtype=torch.float32),
                **geometry.metrics,
                "sampled_step": state.new_tensor(float(point.step)),
                "sampled_sigma": state.new_tensor(float(point.sigma)),
                "keep_fraction": keep.mean().detach(),
            },
        )

    def _compute_branchwise(
        self,
        prepared: PreparedTrajectoryBatch,
        state: Any,
        point: TrajectoryPoint,
        *,
        current: Any,
        source: Any,
        keep: Any,
        terminal_delta: float,
        alpha: float,
        effective_alpha: float,
        following_point: TrajectoryPoint | None,
        reference: Any | None,
    ) -> AlgorithmResult:
        import torch

        self._actor_memory_boundary()
        with torch.no_grad():
            teacher_values = self._validate_branches(
                self.model.predict_tail_velocity_branches(
                    state, point, prepared.conditioning, actor="base"
                ),
                actor="base",
            )
            context = TargetGeometryContext(
                current=current,
                source=source,
                keep=keep,
                point=point,
                following_point=following_point,
                reference=reference,
                alpha=alpha,
                alpha_fn=self.alpha,
                preservation_scale=self.preservation_scale,
                model=self.model,
                terminal=prepared.states.get("terminal_latent"),
            )
            geometry_results = {
                name: self.target_geometry.build(teacher_values[name].float(), context)
                for name in self.branch_names
            }
            targets = {
                name: geometry_results[name].velocity for name in self.branch_names
            }
            geometry_delta = next(iter(geometry_results.values())).delta
            if any(
                result.delta != geometry_delta for result in geometry_results.values()
            ):
                raise ValueError("target geometry returned inconsistent branch deltas")
        self._actor_memory_boundary()
        student_values = self._validate_branches(
            self.model.predict_tail_velocity_branches(
                state, point, prepared.conditioning, actor="refined"
            ),
            actor="refined",
        )

        student_values = {
            name: student_values[name].float() for name in self.branch_names
        }
        teacher_values = {
            name: teacher_values[name].float() for name in self.branch_names
        }
        branch_velocity_errors = {
            name: self._per_sample((student_values[name] - targets[name]).square())
            for name in self.branch_names
        }
        branch_state_errors = {
            name: value * geometry_delta**2
            for name, value in branch_velocity_errors.items()
        }
        selected_errors = (
            branch_velocity_errors
            if self.target_loss == V_TARGET
            else branch_state_errors
        )
        per_sample = sum(
            self.branch_weights[name] * selected_errors[name]
            for name in self.branch_names
        )
        raw_loss = per_sample.mean()

        student = self.model.compose_tail_velocity_branches(
            TailVelocityBranches(student_values)
        ).float()
        with torch.no_grad():
            teacher = self.model.compose_tail_velocity_branches(
                TailVelocityBranches(teacher_values)
            ).float()
            target = self.model.compose_tail_velocity_branches(
                TailVelocityBranches(targets)
            ).float()
        branch_prior = {
            name: self._per_sample(
                (student_values[name] - teacher_values[name]).square()
            ).mean()
            for name in self.branch_names
        }
        prior_mse = sum(
            self.branch_weights[name] * branch_prior[name] for name in self.branch_names
        )
        prior_state_mse = prior_mse * geometry_delta**2
        composed_target_mse = self._per_sample((student - target).square()).mean()
        composed_target_state_mse = composed_target_mse * geometry_delta**2
        composed_prior_mse = self._per_sample((student - teacher).square()).mean()
        target_velocity_mse = sum(
            self.branch_weights[name] * branch_velocity_errors[name]
            for name in self.branch_names
        ).mean()
        target_state_mse = sum(
            self.branch_weights[name] * branch_state_errors[name]
            for name in self.branch_names
        ).mean()
        preservation_target = (
            (source - current) / terminal_delta
            if self.target_geometry.name == TERMINAL_BRIDGE
            else target
        )
        preservation_mse = self._masked_per_sample(
            (student - preservation_target).square(), keep
        ).mean()
        predicted_terminal = current + student * terminal_delta
        terminal_mse = self._masked_per_sample(
            (predicted_terminal - source).square(), keep
        ).mean()
        metrics = {
            "loss": raw_loss.detach() * self.weight,
            "target_velocity_mse": target_velocity_mse.detach(),
            "target_state_mse": target_state_mse.detach(),
            "prior_velocity_mse": prior_mse.detach(),
            "prior_state_mse": prior_state_mse.detach(),
            "composed_target_velocity_mse": composed_target_mse.detach(),
            "composed_target_state_mse": composed_target_state_mse.detach(),
            "composed_prior_velocity_mse": composed_prior_mse.detach(),
            "keep_preservation_velocity_mse": preservation_mse.detach(),
            "keep_terminal_mse": terminal_mse.detach(),
            "target_loss_is_x": state.new_tensor(
                float(self.target_loss == X_TARGET), dtype=torch.float32
            ),
            "target_alpha": state.new_tensor(alpha, dtype=torch.float32),
            "target_preservation_scale": state.new_tensor(
                self.preservation_scale, dtype=torch.float32
            ),
            "target_effective_alpha": state.new_tensor(
                effective_alpha, dtype=torch.float32
            ),
            "target_geometry_cached_path": state.new_tensor(
                float(self.target_geometry.name == CACHED_PATH),
                dtype=torch.float32,
            ),
            "target_geometry_tracking_path": state.new_tensor(
                float(self.target_geometry.name == TRACKING_PATH),
                dtype=torch.float32,
            ),
            "target_geometry_terminal_bridge": state.new_tensor(
                float(self.target_geometry.name == TERMINAL_BRIDGE),
                dtype=torch.float32,
            ),
            "target_step_delta": state.new_tensor(geometry_delta, dtype=torch.float32),
            "path_beta": state.new_tensor(alpha, dtype=torch.float32),
            "path_delta": state.new_tensor(geometry_delta, dtype=torch.float32),
            "tracking_deviation": state.new_tensor(0.0, dtype=torch.float32),
            "tracking_retention": state.new_tensor(0.0, dtype=torch.float32),
            **next(iter(geometry_results.values())).metrics,
            "sampled_step": state.new_tensor(float(point.step)),
            "sampled_sigma": state.new_tensor(float(point.sigma)),
            "keep_fraction": keep.mean().detach(),
        }
        for name in self.branch_names:
            metrics[f"branch_{name}_target_velocity_mse"] = (
                branch_velocity_errors[name].mean().detach()
            )
            metrics[f"branch_{name}_target_state_mse"] = (
                branch_state_errors[name].mean().detach()
            )
            metrics[f"branch_{name}_prior_velocity_mse"] = branch_prior[name].detach()
        return AlgorithmResult(
            loss=raw_loss * self.weight,
            per_sample=per_sample * self.weight,
            metrics=metrics,
        )
