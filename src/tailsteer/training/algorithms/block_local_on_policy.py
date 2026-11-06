"""Cached-start, block-local student on-policy teacher guidance."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tailsteer.cache.trajectory_dataset import TrajectoryContract
from tailsteer.training.rollout_sampling import (
    ROLLOUT_STRATEGIES,
    build_rollout_sampler,
)
from tailsteer.training.rollout_start_state import (
    CACHED,
    materialize_rollout_start,
    validate_rollout_start_state,
)
from tailsteer.training.target_step_sampling import (
    build_target_step_sampler,
    validate_inference_schedule,
)

from ..start_state_augmentation import StartStateAugmentation
from .common import AlgorithmResult
from .teacher_guided import TeacherGuidedTarget


class BlockLocalOnPolicy:
    def __init__(
        self,
        model: Any,
        contract: TrajectoryContract,
        transition: Mapping[str, Any],
        rollout: Mapping[str, Any],
        sampling: Mapping[str, Any] | None = None,
        target_config: Mapping[str, Any] | None = None,
        *,
        weight: float = 1.0,
        seed: int = 0,
        distributed_context: Any | None = None,
        start_state_augmentation: StartStateAugmentation | None = None,
        rollout_start_state: str = CACHED,
    ) -> None:
        self.model = model
        self.start_state_augmentation = (
            start_state_augmentation or StartStateAugmentation()
        )
        self.contract = contract
        self.rollout_start_state = validate_rollout_start_state(
            rollout_start_state, contract
        )
        self.target = TeacherGuidedTarget(
            model,
            contract,
            transition,
            target_config=target_config,
            weight=weight,
        )
        self.rollout_sampler = build_rollout_sampler(contract, transition, rollout)
        self.first_step = self.rollout_sampler.first_step
        self.distributed_context = distributed_context
        self.schedule = model.tail_schedule(
            num_steps=contract.num_steps,
            height=contract.height,
            width=contract.width,
        )
        validate_inference_schedule(
            contract,
            self.schedule,
            supervision_step=self.target.supervision_step,
        )
        self.sampler = build_target_step_sampler(
            contract, transition, sampling, seed=seed, schedule=self.schedule
        )
        missing = sorted(
            step
            for steps in self.sampler.steps.values()
            for step in steps
            if step not in self.schedule
        )
        if missing:
            raise ValueError(f"model schedule lacks sampled steps: {missing}")

    def _rollout_steps(self, target_step: int, *, training: bool) -> int:
        from tailsteer.training.target_step_sampling import TargetStepSample

        return self.rollout_sampler.sample(
            TargetStepSample(step=target_step, region="prior"), training=training
        ).hops

    def _forward_count(self, hops: int) -> int:
        fixed = self.rollout_sampler.fixed_forward_count
        if fixed is not None:
            return int(fixed)
        if self.distributed_context is None:
            return int(hops)
        return int(self.distributed_context.reduce_max_int(hops))

    def compute_loss(
        self, batch: Mapping[str, Any], *, training: bool = True
    ) -> AlgorithmResult:
        import torch

        metadata = batch["metadata"]
        if len(metadata) != 1:
            raise ValueError(
                "block-local on-policy training currently requires batch_size=1"
            )
        sample = self.sampler.sample(
            training=training, key=str(metadata[0]["source_identity"])
        )
        rollout = self.rollout_sampler.sample(sample, training=training)
        hops = rollout.hops
        start_step = rollout.start_step
        forward_count = self._forward_count(hops)
        target_point = self.schedule[sample.step]
        start_point = self.schedule[start_step]
        materialized = materialize_rollout_start(
            batch, start_point, self.contract, mode=self.rollout_start_state
        )
        materialized["latent"] = self.start_state_augmentation.apply(
            materialized["latent"], start_point.sigma, training=training
        )
        prepared = self.target.prepare_batch(batch, materialized)
        state = prepared.states["latent"]

        # FSDP ranks independently sample target steps. All ranks execute the
        # same per-batch forward count; calls after a local target are padding.
        with torch.no_grad():
            for rollout_index in range(forward_count):
                if rollout_index < hops:
                    current_step = start_step + rollout_index
                    current = self.schedule[current_step]
                    following = self.schedule[current_step + 1]
                    velocity = self.model.predict_tail_velocity(
                        state, current, prepared.conditioning, actor="refined"
                    )
                    state = self.model.tail_advance(
                        state, velocity, current=current, following=following
                    ).detach()
                else:
                    self.model.predict_tail_velocity(
                        state, target_point, prepared.conditioning, actor="refined"
                    )

        following_point, reference = self.target.cached_path_inputs(
            batch, self.schedule, target_point
        )
        result = self.target.compute(
            prepared,
            state,
            target_point,
            following_point=following_point,
            reference=reference,
        )
        return AlgorithmResult(
            loss=result.loss,
            per_sample=result.per_sample,
            metrics={
                **result.metrics,
                **self.sampler.metrics(sample, state),
                "endpoint_sample": state.new_tensor(
                    float(sample.step == self.target.supervision_step)
                ),
                "rollout_start_step": state.new_tensor(float(start_step)),
                "rollout_target_step": state.new_tensor(float(sample.step)),
                "rollout_hops": state.new_tensor(float(hops)),
                "rollout_forward_count": state.new_tensor(float(forward_count)),
                "rollout_padding_hops": state.new_tensor(float(forward_count - hops)),
                **{
                    f"rollout_strategy_{name}": state.new_tensor(
                        float(rollout.strategy == name)
                    )
                    for name in ROLLOUT_STRATEGIES
                },
            },
        )
