"""Optional D ablation: start on the cached noise-to-terminal straight line."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from tailsteer.cache.trajectory_dataset import (
    TrajectoryContract,
    TrajectorySourceSample,
    materialize_trajectory_state,
)
from tailsteer.models.sensenova.identity import SENSENOVA_MODEL_NAMES

CACHED = "cached"
NOISE_TERMINAL_INTERPOLATION = "noise_terminal_interpolation"
ROLLOUT_START_STATES = (CACHED, NOISE_TERMINAL_INTERPOLATION)


def validate_rollout_start_state(
    mode: str, contract: TrajectoryContract | None = None
) -> str:
    if mode not in ROLLOUT_START_STATES:
        raise ValueError(f"rollout_start_state must be one of {ROLLOUT_START_STATES}")
    if mode == CACHED or contract is None:
        return mode
    terminal = contract.terminal_point
    initial_step = terminal.step_index - contract.num_steps
    try:
        initial = contract.point(initial_step)
    except KeyError as exc:
        raise ValueError(
            f"noise_terminal_interpolation requires cached step {initial_step}"
        ) from exc
    if (
        initial_step not in {0, 1}
        or initial.state_kind == "final"
        or initial.sigma != 1.0
        or terminal.sigma != 0.0
    ):
        raise ValueError(
            "noise_terminal_interpolation requires the initial step at sigma=1 and "
            "the final state at sigma=0"
        )
    return mode


def noise_terminal_endpoints(
    source: TrajectorySourceSample, contract: TrajectoryContract
) -> dict[int, Mapping[str, Any]]:
    """Select paired noise/final states in the model's native step numbering."""

    validate_rollout_start_state(NOISE_TERMINAL_INTERPOLATION, contract)
    terminal_step = contract.terminal_point.step_index
    initial_step = terminal_step - contract.num_steps
    if initial_step not in source.points or terminal_step not in source.points:
        raise ValueError("noise_terminal_interpolation requires both cached endpoints")
    noise_mode = source.points[initial_step].get("noise_mode")
    # Legacy SenseNova caches omit this field; their first forward always takes
    # the native scaled Gaussian noise (not a source-image/noise mixture).
    if noise_mode is None and contract.model in SENSENOVA_MODEL_NAMES:
        noise_mode = "pure_noise"
    if noise_mode != "pure_noise":
        raise ValueError("noise_terminal_interpolation requires cached pure_noise")
    return {
        initial_step: source.points[initial_step],
        terminal_step: source.points[terminal_step],
    }


def materialize_rollout_start(
    batch: Mapping[str, Any],
    point: Any,
    contract: TrajectoryContract,
    *,
    mode: str = CACHED,
) -> dict[str, Any]:
    """Keep the selected start index; change only its state, without drawing RNG."""

    validate_rollout_start_state(mode, contract)
    if mode == CACHED:
        return materialize_trajectory_state(batch, point)
    sources = batch["trajectory_sources"]
    if len(sources) != 1:
        raise ValueError("rollout start materialization requires batch_size=1")
    source = sources[0]
    if not 0.0 <= float(point.sigma) <= 1.0:
        raise ValueError("noise_terminal_interpolation requires sigma in [0, 1]")

    # Restrict only this read to the endpoints. The original contract/locator
    # still controls segment selection and teacher targets. The shared reader
    # interpolates in FP32 then restores dtype, and reads source/mask unchanged.
    endpoints = replace(
        source,
        points=noise_terminal_endpoints(source, contract),
    )
    materialized = materialize_trajectory_state(
        {**batch, "trajectory_sources": [endpoints]}, point
    )
    for metadata in materialized["metadata"]:
        metadata["state_kind"] = NOISE_TERMINAL_INTERPOLATION
        metadata["rollout_start_state"] = mode
    return materialized
