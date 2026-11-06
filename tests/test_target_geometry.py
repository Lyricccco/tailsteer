from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

from tailsteer.cache.format import (
    TRAJECTORY_GROUPED_LAYOUT,
    CacheExample,
    TrajectoryCacheWriter,
)
from tailsteer.cache.trajectory_dataset import (
    CachedPathReference,
    CachedPathReferenceResolver,
    TrajectorySourceDataset,
    trajectory_source_collate,
)
from tailsteer.models.protocols import TrajectoryPoint
from tailsteer.training.target_geometry import (
    CACHED_PATH,
    INSTANT_VELOCITY,
    TERMINAL_BRIDGE,
    TRACKING_PATH,
    TargetGeometryContext,
    build_target_geometry,
)


class _GeometryModel:
    @staticmethod
    def tail_terminal_delta(point):
        return -float(point.sigma)

    @staticmethod
    def tail_step_delta(current, following):
        return float(following.sigma) - float(current.sigma)


def _context(*, geometry_reference=None, alpha=1.0, scale=1.0):
    current = torch.tensor([[[[2.0, 2.0]]]])
    return TargetGeometryContext(
        current=current,
        source=torch.zeros_like(current),
        keep=torch.ones_like(current),
        point=TrajectoryPoint(49.0, 0.1, 0.9),
        following_point=TrajectoryPoint(50.0, 0.0, 1.0),
        reference=geometry_reference,
        alpha=alpha,
        alpha_fn=lambda point: 1.0 if point.step >= 50 else alpha,
        preservation_scale=scale,
        model=_GeometryModel(),
    )


def test_terminal_bridge_is_the_default_geometry():
    geometry = build_target_geometry(None)
    result = geometry.build(torch.ones((1, 1, 1, 2)), _context())

    assert geometry.name == TERMINAL_BRIDGE
    torch.testing.assert_close(result.velocity, torch.full((1, 1, 1, 2), 20.0))
    assert result.delta == pytest.approx(-0.1)


@pytest.mark.parametrize("alpha", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("scale", [0.0, 0.5, 1.0, 2.0])
def test_instant_velocity_masks_only_the_goal(alpha, scale):
    context = replace(
        _context(alpha=alpha),
        keep=torch.tensor([[[[1.0, 0.0]]]]),
        terminal=torch.tensor([[[[9.0, 3.0]]]]),
    )
    geometry = build_target_geometry(
        {"geometry": INSTANT_VELOCITY, "velocity_scale": scale}
    )
    teacher = torch.full_like(context.current, 4.0)
    result = geometry.build(teacher, context)

    # g=[source=0, terminal=3], x=[2,2], R=-.1; edit velocity is -10,
    # NOT the teacher's +4. There is no outer keep mask on the correction.
    line = torch.tensor([[[[20.0, -10.0]]]])
    expected = (1 - alpha * scale) * teacher + alpha * scale * line
    torch.testing.assert_close(result.velocity, expected)
    assert result.delta == pytest.approx(-0.1)
    assert not geometry.requires_reference
    assert geometry.requires_terminal
    if alpha == scale == 1.0:
        torch.testing.assert_close(result.target_state, torch.tensor([[[[0.0, 3.0]]]]))


def test_instant_velocity_source_goal_does_not_require_terminal():
    geometry = build_target_geometry(
        {"geometry": INSTANT_VELOCITY, "endpoint": {"mode": "source"}}
    )
    context = replace(_context(), keep=torch.zeros((1, 1, 1, 2)))
    result = geometry.build(torch.zeros_like(context.current), context)
    assert not geometry.requires_terminal
    torch.testing.assert_close(result.target_state, context.source)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("exponent", [0.5, 1.0, 3.0])
def test_instant_velocity_ideal_rollout_reaches_goal(sign, exponent):
    class NativeModel:
        @staticmethod
        def tail_terminal_delta(point):
            return sign * float(point.sigma)

    geometry = build_target_geometry({"geometry": INSTANT_VELOCITY})
    context = replace(
        _context(),
        model=NativeModel(),
        keep=torch.tensor([[[[1.0, 0.0]]]]),
        terminal=torch.tensor([[[[8.0, 3.0]]]]),
    )
    sigmas = [0.9, 0.53, 0.3, 0.07, 0.0]
    state = context.current
    for k, (sigma, following) in enumerate(zip(sigmas, sigmas[1:], strict=False)):
        progress = (sigmas[0] - sigma) / (sigmas[0] - sigmas[-2])
        context = replace(
            context,
            current=state,
            point=TrajectoryPoint(float(k), sigma, sigma),
            alpha=progress**exponent,
        )
        teacher = state.tanh() + 0.2
        velocity = geometry.build(teacher, context).velocity
        if k == 0:
            torch.testing.assert_close(velocity, teacher)
        state = state + sign * (sigma - following) * velocity
    torch.testing.assert_close(state, torch.tensor([[[[0.0, 3.0]]]]))


def test_instant_velocity_uses_remaining_interval_not_next_step():
    context = replace(
        _context(),
        point=TrajectoryPoint(47.0, 0.4, 0.6),
        following_point=TrajectoryPoint(48.0, 0.3, 0.7),
    )
    geometry = build_target_geometry(
        {"geometry": INSTANT_VELOCITY, "endpoint": {"mode": "source"}}
    )
    result = geometry.build(torch.zeros_like(context.current), context)
    assert result.delta == pytest.approx(-0.4)
    torch.testing.assert_close(result.velocity, torch.full_like(context.current, 5.0))


@pytest.mark.parametrize("scale", [-1.0, float("nan"), float("inf"), True, "1"])
def test_instant_velocity_rejects_invalid_scale(scale):
    with pytest.raises((ValueError, TypeError), match="velocity_scale"):
        build_target_geometry({"geometry": INSTANT_VELOCITY, "velocity_scale": scale})


def test_instant_velocity_requires_goal_and_does_not_reinterpret_old_scale():
    geometry = build_target_geometry({"geometry": INSTANT_VELOCITY})
    with pytest.raises(ValueError, match="cached terminal state"):
        geometry.build(torch.ones((1, 1, 1, 2)), _context())
    with pytest.raises(ValueError, match="not preservation_scale"):
        build_target_geometry({"geometry": INSTANT_VELOCITY, "preservation_scale": 0.2})
    with pytest.raises(ValueError, match="endpoint.mode"):
        build_target_geometry(
            {"geometry": INSTANT_VELOCITY, "endpoint": {"mode": "source_keep"}}
        )
    with pytest.raises(ValueError, match="require geometry instant_velocity"):
        build_target_geometry({"geometry": CACHED_PATH, "velocity_scale": 2.0})


def test_cached_path_uses_local_cached_segment_and_terminal_correction():
    reference = CachedPathReference(
        current_state=torch.full((1, 1, 1, 2), 4.0),
        following_state=torch.full((1, 1, 1, 2), 5.0),
        terminal_state=torch.full((1, 1, 1, 2), 5.0),
        current_point=TrajectoryPoint(49.0, 0.1, 0.9),
        following_point=TrajectoryPoint(50.0, 0.0, 1.0),
    )
    geometry = build_target_geometry({"geometry": CACHED_PATH})
    result = geometry.build(
        torch.zeros((1, 1, 1, 2)), _context(geometry_reference=reference)
    )

    # y_49 = 4 + (0 - 5), y_50 = 5 + (0 - 5), so the local velocity is -10.
    torch.testing.assert_close(result.velocity, torch.full((1, 1, 1, 2), -10.0))
    assert result.delta == pytest.approx(-0.1)


def test_cached_path_requires_reference_and_known_geometry():
    geometry = build_target_geometry({"geometry": CACHED_PATH})
    with pytest.raises(ValueError, match="requires current/following"):
        geometry.build(torch.zeros((1, 1, 1, 2)), _context())
    with pytest.raises(ValueError, match="geometry must be one of"):
        build_target_geometry({"geometry": "unknown"})


def test_tracking_path_contracts_student_deviation_to_reference():
    reference = CachedPathReference(
        current_state=torch.full((1, 1, 1, 2), 4.0),
        following_state=torch.full((1, 1, 1, 2), 5.0),
        terminal_state=torch.full((1, 1, 1, 2), 5.0),
        current_point=TrajectoryPoint(49.0, 0.1, 0.9),
        following_point=TrajectoryPoint(50.0, 0.0, 1.0),
    )
    geometry = build_target_geometry(
        {"geometry": TRACKING_PATH, "tracking": {"retention": 0.5}}
    )
    result = geometry.build(
        torch.zeros((1, 1, 1, 2)),
        _context(geometry_reference=reference),
    )

    # y_49=-1, current=2, u=-10, and rho=.5 gives -10-.5/(-.1)*3=5.
    torch.testing.assert_close(result.velocity, torch.full((1, 1, 1, 2), 5.0))
    assert result.metrics["tracking_retention"].item() == pytest.approx(0.5)
    assert result.metrics["tracking_deviation"].item() == pytest.approx(3.0)


def test_cached_path_resolver_includes_terminal_state(tmp_path):
    cache = tmp_path / "cache"
    examples = []
    for step, sigma, state_kind in (
        (0, 1.0, "forward_input"),
        (1, 0.5, "forward_input"),
        (2, 0.0, "final"),
    ):
        examples.append(
            CacheExample(
                anchor_latent=torch.full((1, 1, 2), float(step)),
                source_latent=torch.zeros((1, 1, 2)),
                keep_mask=torch.ones((1, 1, 2), dtype=torch.uint8),
                metadata={
                    "sample_id": f"source-a:{step}",
                    "source_id": "source-a",
                    "source_identity": "dataset:source-a",
                    "prompt": "edit",
                    "seed": 1,
                    "step_index": step,
                    "sigma": sigma,
                    "timestep": 1.0 - sigma,
                    "state_kind": state_kind,
                    "state_representation": "toy",
                    "height": 16,
                    "width": 32,
                    "task": "local_edit",
                },
            )
        )
    TrajectoryCacheWriter(cache).write(examples)
    (cache / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "storage_layout": TRAJECTORY_GROUPED_LAYOUT,
                "model": "toy",
                "representation": "toy",
                "state_points": [0, 1, 2],
                "sources": 1,
                "records": 3,
                "creation_config": {"cache": {"num_steps": 2}},
            }
        ),
        encoding="utf-8",
    )
    (cache / "COMPLETE").write_text("complete\n", encoding="utf-8")

    dataset = TrajectorySourceDataset(cache, supervision_step=1)
    batch = trajectory_source_collate([dataset[0]])
    reference = CachedPathReferenceResolver(batch).resolve(
        TrajectoryPoint(1.0, 0.5, 0.5), TrajectoryPoint(2.0, 0.0, 1.0)
    )
    torch.testing.assert_close(reference.current_state, torch.ones((1, 1, 1, 2)))
    torch.testing.assert_close(reference.following_state, torch.full((1, 1, 1, 2), 2.0))
    torch.testing.assert_close(reference.terminal_state, torch.full((1, 1, 1, 2), 2.0))
