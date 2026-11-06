"""Rank-aware console and file logging."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(
    *, rank: int = 0, output_dir: str | Path | None = None
) -> logging.Logger:
    logger = logging.getLogger("tailsteer")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        f"%(asctime)s | rank={rank:05d} | %(levelname)s | %(message)s"
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if output_dir is not None:
        logs = Path(output_dir).expanduser().resolve() / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(logs / f"rank-{rank:05d}.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger
