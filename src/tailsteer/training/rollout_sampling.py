"""Model-neutral rollout-start policies for cached trajectory training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tailsteer.cache.trajectory_dataset import TrajectoryContract
from tailsteer.training.target_step_sampling import TargetStepSample

LOCAL_WINDOW = "local_window"
CACHED_SEGMENT = "cached_segment"
FIXED_TAIL = "fixed_tail"
ROLLOUT_STRATEGIES = (LOCAL_WINDOW, CACHED_SEGMENT, FIXED_TAIL)


@dataclass(frozen=True)
class RolloutPolicySpec:
    strategy: str
    min_steps: int | None = None
    max_steps: int | None = None


@dataclass(frozen=True)
class RolloutStepSample:
    start_step: int
    target_step: int
    hops: int
    strategy: str


def parse_rollout_policy(config: Mapping[str, Any]) -> RolloutPolicySpec:
    """Validate a rollout policy while accepting the original D-local schema."""

    if not isinstance(config, Mapping):
        raise TypeError("D state_source must be a mapping")
    values = dict(config)
    strategy = str(values.get("strategy", LOCAL_WINDOW))
    if strategy not in ROLLOUT_STRATEGIES:
        raise ValueError(
            f"unknown rollout strategy {strategy!r}; "
            f"available={list(ROLLOUT_STRATEGIES)}"
        )

    if strategy in {CACHED_SEGMENT, FIXED_TAIL}:
        unknown = set(values) - {"strategy"}
        if unknown:
            raise ValueError(
                "cached_segment state_source only accepts strategy; "
                f"unknown={sorted(unknown)}"
            )
        return RolloutPolicySpec(strategy=strategy)

    unknown = set(values) - {
        "strategy",
        "min_rollout_steps",
        "max_rollout_steps",
    }
    if unknown:
        raise ValueError(
            f"local_window state_source has unknown fields: {sorted(unknown)}"
        )
    minimum = values.get("min_rollout_steps")
    maximum = values.get("max_rollout_steps")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (minimum, maximum)
    ):
        raise TypeError("D rollout bounds must be integers")
    if minimum < 1 or maximum < minimum:
        raise ValueError("D rollout bounds require 1 <= min <= max")
    return RolloutPolicySpec(
        strategy=strategy,
        min_steps=int(minimum),
        max_steps=int(maximum),
    )


class RolloutStepSampler:
    """Select a cached/offline start for one already-sampled target step."""

    def __init__(
        self,
        contract: TrajectoryContract,
        transition: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> None:
        self.spec = parse_rollout_policy(config)
        self.start_step = int(transition["start_step"])
        self.supervision_step = int(transition["supervision_step"])
        self.cached_steps = tuple(
            sorted(
                int(point.step_index)
                for point in contract.forward_points
                if int(point.step_index) <= self.supervision_step
            )
        )
        if not self.cached_steps:
            raise ValueError("rollout sampling requires cached forward states")
        self.first_step = self.cached_steps[0]
        if self.start_step not in self.cached_steps:
            raise ValueError(
                "cached_segment boundary must be an exact cached forward state: "
                f"step={self.start_step} cached={list(self.cached_steps)}"
            )
        if self.cached_steps[-1] != self.supervision_step:
            raise ValueError(
                "rollout supervision step must be an exact cached forward state"
            )

    @property
    def strategy(self) -> str:
        return self.spec.strategy

    @property
    def fixed_forward_count(self) -> int | None:
        if self.spec.strategy == LOCAL_WINDOW:
            return self.spec.max_steps
        return None

    def sample(
        self,
        target: TargetStepSample,
        *,
        training: bool,
    ) -> RolloutStepSample:
        target_step = int(target.step)
        if not self.first_step <= target_step <= self.supervision_step:
            raise ValueError(
                f"target step {target_step} is outside rollout support "
                f"[{self.first_step}, {self.supervision_step}]"
            )
        if self.spec.strategy == LOCAL_WINDOW:
            start_step = target_step - self._local_hops(target_step, training=training)
        elif self.spec.strategy == FIXED_TAIL:
            if target_step < self.start_step:
                raise ValueError("fixed_tail targets must not precede the anchor")
            start_step = self.start_step
        else:
            start_step = self._segment_start(target_step)
        return RolloutStepSample(
            start_step=start_step,
            target_step=target_step,
            hops=target_step - start_step,
            strategy=self.spec.strategy,
        )

    def _local_hops(self, target_step: int, *, training: bool) -> int:
        import torch

        assert self.spec.min_steps is not None
        assert self.spec.max_steps is not None
        available = target_step - self.first_step
        if available < self.spec.min_steps:
            return 0
        upper = min(self.spec.max_steps, available)
        if not training:
            return upper
        return int(torch.randint(self.spec.min_steps, upper + 1, ()).item())

    def _segment_start(self, target_step: int) -> int:
        if target_step == self.first_step:
            return self.first_step
        if target_step > self.start_step:
            return self.start_step
        candidates = tuple(step for step in self.cached_steps if step < target_step)
        if not candidates:
            raise ValueError(
                f"cached_segment target {target_step} has no preceding cache anchor"
            )
        return candidates[-1]


def build_rollout_sampler(
    contract: TrajectoryContract,
    transition: Mapping[str, Any],
    config: Mapping[str, Any],
) -> RolloutStepSampler:
    return RolloutStepSampler(contract, transition, config)
