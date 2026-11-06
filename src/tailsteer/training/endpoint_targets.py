"""Terminal goals shared by A/B and the opt-in C/D instant velocity target."""

from __future__ import annotations

from typing import Any

SOURCE_KEEP = "source_keep"
SOURCE_KEEP_TERMINAL_EDIT = "source_keep_terminal_edit"
ENDPOINT_TARGETS = (SOURCE_KEEP, SOURCE_KEEP_TERMINAL_EDIT)


def validate_endpoint_target(value: str) -> str:
    if value not in ENDPOINT_TARGETS:
        raise ValueError(
            f"unknown endpoint_target {value!r}; available={ENDPOINT_TARGETS}"
        )
    return value


def endpoint_goal_and_mask(
    source: Any, keep: Any, *, terminal: Any | None = None
) -> tuple[Any, Any]:
    """Return the goal and loss support; terminal is an opt-in cached target.

    Without a terminal, retain the historical keep-only objective. With one,
    mask=1 selects the source and mask=0 selects the same trajectory's terminal;
    supervise the resulting single velocity target over the full state.
    """

    if terminal is None:
        return source, keep
    if source.shape != terminal.shape:
        raise ValueError("endpoint source and cached terminal shapes must match")
    import torch

    goal = keep * source + (1.0 - keep) * terminal
    return goal.detach(), torch.ones_like(keep)
