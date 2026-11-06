"""Small, model-agnostic vocabulary for multi-round feedback state."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

FEEDBACK_MODES = frozenset(
    {
        "latent",
        "model_native",
        "continuous_image",
        "image",
    }
)


def feedback_config(mode: str) -> dict[str, str]:
    """Build the canonical config form while keeping parsing backwards compatible."""

    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized not in FEEDBACK_MODES:
        raise ValueError(
            f"unsupported feedback mode {mode!r}; expected {sorted(FEEDBACK_MODES)}"
        )
    return {"mode": normalized}


def feedback_mode(value: Any, *, default: str) -> str:
    """Read ``feedback: mode`` and the canonical ``feedback: {mode: ...}``."""

    raw = default if value is None else value
    if isinstance(raw, Mapping):
        unknown = set(raw) - {"mode"}
        if unknown:
            raise ValueError(f"unsupported feedback fields: {sorted(unknown)}")
        raw = raw.get("mode", default)
    return feedback_config(str(raw))["mode"]
