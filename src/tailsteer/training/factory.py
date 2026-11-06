"""Build the source-level dataset and TailSteer training objective."""

from dataclasses import dataclass
from typing import Any
from tailsteer.cache.trajectory_dataset import (
    compose_trajectory_source_datasets,
    trajectory_source_collate,
)
from .algorithms.block_local_on_policy import BlockLocalOnPolicy
from .target_geometry import requires_teacher


@dataclass(frozen=True)
class TrainingDatasetSpec:
    dataset: Any
    collate_fn: Any
    contract: Any


def algorithm_name(config):
    return "trajectory_preservation"


def requires_independent_teacher(config):
    t = config["training"]
    return requires_teacher(t["transition"], t["teacher_guided_target"])


def build_training_regularizer(config, model, *, distributed_context=None):
    return None


def build_training_dataset(config):
    t = config["training"]
    if t["batch_size"] != 1:
        raise ValueError(
            "TailSteer uses one trajectory per rank; use gradient accumulation"
        )
    dataset, contract = compose_trajectory_source_datasets(
        config["data"]["caches"],
        supervision_step=int(t["transition"]["supervision_step"]),
    )
    return TrainingDatasetSpec(dataset, trajectory_source_collate, contract)


def build_algorithm(config, model, contract, *, distributed_context=None):
    t = config["training"]
    return BlockLocalOnPolicy(
        model,
        contract,
        t["transition"],
        t["state_source"],
        t["target_step_sampling"],
        t["teacher_guided_target"],
        weight=1.0,
        seed=int(config.get("seed", 42)),
        distributed_context=distributed_context,
        rollout_start_state="cached",
    )
