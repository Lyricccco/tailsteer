"""Stable identities for checkpoints sharing the SenseNova U1.5 backend."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

SENSENOVA_MODEL_VARIANTS: Mapping[str, str] = MappingProxyType(
    {
        "sensenova_u1_5_8b_mot": "sensenova-u1.5-8b-mot",
    }
)
SENSENOVA_MODEL_NAMES = frozenset(SENSENOVA_MODEL_VARIANTS)


def resolve_sensenova_variant(
    model_name: str, configured_variant: object | None = None
) -> str:
    """Resolve and validate the checkpoint identity for one shared backend."""

    try:
        expected = SENSENOVA_MODEL_VARIANTS[str(model_name)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported SenseNova model identity {model_name!r}"
        ) from exc
    if configured_variant is None:
        return expected
    configured = str(configured_variant)
    if configured != expected:
        raise ValueError(
            f"SenseNova model {model_name!r} requires variant {expected!r}, "
            f"got {configured!r}"
        )
    return configured
