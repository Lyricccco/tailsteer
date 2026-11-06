"""Strict numerical checks for official-state TailSteer forwards and rollouts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TensorParity:
    max_abs_error: float
    mean_abs_error: float


def assert_tensor_parity(
    actual: Any,
    expected: Any,
    *,
    label: str,
    rtol: float,
    atol: float,
) -> TensorParity:
    """Assert shape/finite/value parity and return compact diagnostics."""

    import torch

    if tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError(
            f"{label} shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    actual_float = actual.detach().float().cpu()
    expected_float = expected.detach().float().cpu()
    if (
        not torch.isfinite(actual_float).all()
        or not torch.isfinite(expected_float).all()
    ):
        raise AssertionError(f"{label} contains non-finite values")
    difference = (actual_float - expected_float).abs()
    try:
        torch.testing.assert_close(
            actual_float,
            expected_float,
            rtol=float(rtol),
            atol=float(atol),
            equal_nan=False,
        )
    except AssertionError as error:
        raise AssertionError(
            f"{label} failed parity: max_abs={float(difference.max()):.8g}, "
            f"mean_abs={float(difference.mean()):.8g}, rtol={rtol}, atol={atol}"
        ) from error
    return TensorParity(
        max_abs_error=float(difference.max()),
        mean_abs_error=float(difference.mean()),
    )


def state_like(value: Any, reference: Any) -> Any:
    """Move one captured official state to the prepared training-state boundary."""

    result = value
    while result.ndim < reference.ndim:
        result = result.unsqueeze(0)
    while result.ndim > reference.ndim and int(result.shape[0]) == 1:
        result = result[0]
    if tuple(result.shape) != tuple(reference.shape):
        raise ValueError(
            "captured official state cannot be aligned with training state: "
            f"{tuple(value.shape)} -> {tuple(result.shape)} != {tuple(reference.shape)}"
        )
    return result.to(device=reference.device, dtype=reference.dtype)


def check_training_rollout_parity(
    model: Any,
    *,
    conditioning: Any,
    schedule: Mapping[int, Any],
    official_states: Mapping[int, Any],
    official_final: Any,
    official_velocities: Mapping[int, Any] | None = None,
    reference_state: Any,
    exact_steps: Sequence[int],
    actor: str = "base",
    rtol: float = 0.0,
    atol: float = 0.0,
) -> dict[str, Any]:
    """Check one-step forwards on official states and a complete chained rollout."""

    import torch

    ordered = tuple(sorted(int(step) for step in official_states))
    if not ordered or ordered != tuple(range(ordered[0], ordered[-1] + 1)):
        raise ValueError("official parity states must cover a dense forward interval")
    terminal_step = ordered[-1] + 1
    if terminal_step not in schedule:
        raise ValueError("model schedule lacks the official terminal state")
    states = {
        step: state_like(official_states[step], reference_state) for step in ordered
    }
    final = state_like(official_final, reference_state)
    exact = tuple(int(step) for step in exact_steps)
    if any(step not in states for step in exact):
        raise ValueError("exact parity steps must be official forward states")

    one_step: dict[str, dict[str, float]] = {}
    velocity_metrics: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for step in exact:
            following_state = states.get(step + 1, final)
            velocity = model.predict_tail_velocity(
                states[step], schedule[step], conditioning, actor=actor
            )
            if official_velocities is not None:
                expected_velocity = state_like(
                    official_velocities[step], reference_state
                )
                velocity_metrics[str(step)] = assert_tensor_parity(
                    velocity,
                    expected_velocity,
                    label=f"training velocity {step}",
                    rtol=rtol,
                    atol=atol,
                ).__dict__
            advanced = model.tail_advance(
                states[step],
                velocity,
                current=schedule[step],
                following=schedule[step + 1],
            )
            result = assert_tensor_parity(
                advanced,
                following_state,
                label=f"training one-step {step}->{step + 1}",
                rtol=rtol,
                atol=atol,
            )
            one_step[str(step)] = result.__dict__

        chained = states[ordered[0]].clone()
        chained_errors: dict[str, dict[str, float]] = {}
        for step in ordered:
            velocity = model.predict_tail_velocity(
                chained, schedule[step], conditioning, actor=actor
            )
            chained = model.tail_advance(
                chained,
                velocity,
                current=schedule[step],
                following=schedule[step + 1],
            )
            expected = states.get(step + 1, final)
            difference = (chained.float() - expected.float()).abs()
            chained_errors[str(step + 1)] = {
                "max_abs_error": float(difference.max()),
                "mean_abs_error": float(difference.mean()),
            }
        complete = assert_tensor_parity(
            chained,
            final,
            label="training chained rollout final",
            rtol=rtol,
            atol=atol,
        )
    return {
        "one_step": one_step,
        "velocity": velocity_metrics,
        "chained": chained_errors,
        "final": complete.__dict__,
    }
