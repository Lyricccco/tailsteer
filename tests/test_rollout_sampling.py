from __future__ import annotations

import pytest

from tailsteer.cache.trajectory_dataset import (
    CachedTrajectoryPoint,
    TrajectoryContract,
)
from tailsteer.training.rollout_sampling import (
    CACHED_SEGMENT,
    LOCAL_WINDOW,
    RolloutStepSampler,
    parse_rollout_policy,
)
from tailsteer.training.target_step_sampling import TargetStepSample


def _contract(steps: tuple[int, ...], *, supervision_step: int) -> TrajectoryContract:
    points = tuple(
        CachedTrajectoryPoint(
            step,
            1.0 - step / (supervision_step + 1),
            step / (supervision_step + 1),
            "forward_input",
        )
        for step in steps
    ) + (
        CachedTrajectoryPoint(
            supervision_step + 1,
            0.0,
            1.0,
            "final",
        ),
    )
    return TrajectoryContract(
        model="toy",
        representation="toy",
        points=points,
        num_steps=supervision_step + 1,
        height=16,
        width=16,
    )


def _sample(sampler: RolloutStepSampler, step: int):
    return sampler.sample(TargetStepSample(step=step, region="prior"), training=False)


def test_original_rollout_schema_remains_local_window() -> None:
    parsed = parse_rollout_policy({"min_rollout_steps": 1, "max_rollout_steps": 3})

    assert parsed.strategy == LOCAL_WINDOW
    assert parsed.min_steps == 1
    assert parsed.max_steps == 3


def test_explicit_local_window_keeps_fixed_forward_budget() -> None:
    sampler = RolloutStepSampler(
        _contract((0, 4, 7), supervision_step=7),
        {"start_step": 4, "supervision_step": 7},
        {
            "strategy": LOCAL_WINDOW,
            "min_rollout_steps": 1,
            "max_rollout_steps": 3,
        },
    )

    assert _sample(sampler, 0).hops == 0
    assert _sample(sampler, 2).hops == 2
    assert _sample(sampler, 7).hops == 3
    assert sampler.fixed_forward_count == 3


def test_flux_cached_segment_uses_strict_prior_anchor_and_transition_boundary() -> None:
    sampler = RolloutStepSampler(
        _contract(
            (0, 23, 33, 38, 42, 43, 44, 45, 46, 47, 48, 49),
            supervision_step=49,
        ),
        {"start_step": 42, "supervision_step": 49},
        {"strategy": CACHED_SEGMENT},
    )

    assert (_sample(sampler, 0).start_step, _sample(sampler, 0).hops) == (0, 0)
    assert (_sample(sampler, 20).start_step, _sample(sampler, 20).hops) == (
        0,
        20,
    )
    assert (_sample(sampler, 23).start_step, _sample(sampler, 23).hops) == (
        0,
        23,
    )
    assert (_sample(sampler, 30).start_step, _sample(sampler, 30).hops) == (
        23,
        7,
    )
    assert (_sample(sampler, 42).start_step, _sample(sampler, 42).hops) == (
        38,
        4,
    )
    assert (_sample(sampler, 43).start_step, _sample(sampler, 43).hops) == (
        42,
        1,
    )
    assert (_sample(sampler, 49).start_step, _sample(sampler, 49).hops) == (
        42,
        7,
    )
    assert sampler.fixed_forward_count is None


def test_sensenova_cached_segment_is_derived_from_its_contract() -> None:
    sampler = RolloutStepSampler(
        _contract(
            (1, 14, 23, 29, 35, 36, 38, 40, 42, 45, 47, 49, 50),
            supervision_step=50,
        ),
        {"start_step": 42, "supervision_step": 50},
        {"strategy": CACHED_SEGMENT},
    )

    assert (_sample(sampler, 14).start_step, _sample(sampler, 14).hops) == (
        1,
        13,
    )
    assert (_sample(sampler, 41).start_step, _sample(sampler, 41).hops) == (
        40,
        1,
    )
    assert (_sample(sampler, 50).start_step, _sample(sampler, 50).hops) == (
        42,
        8,
    )


def test_cached_segment_rejects_local_window_fields() -> None:
    with pytest.raises(ValueError, match="only accepts strategy"):
        parse_rollout_policy(
            {
                "strategy": CACHED_SEGMENT,
                "max_rollout_steps": 3,
            }
        )
