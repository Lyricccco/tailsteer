"""World-size- and ordering-independent source split."""

from __future__ import annotations

import hashlib


def source_partition(
    source_identity: str, *, seed: int, validation_fraction: float
) -> str:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    token = f"{int(seed)}\n{source_identity}".encode()
    bucket = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
    return "validation" if bucket / float(1 << 64) < validation_fraction else "train"
