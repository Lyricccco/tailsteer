"""Small shared helpers, intentionally independent of torch and model packages."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

DEFAULT_ROOT = Path(
    os.environ.get(
        "TAILSTEER_EVAL_ROOT", Path(__file__).resolve().parents[3] / "data/evaluation"
    )
)
EDITING_BENCHMARKS = ("gedit", "rise", "imgedit-basic", "mse", "edival")
BENCHMARKS = (*EDITING_BENCHMARKS, "qwenimagebench", "bizgeneval")
PROJECT = Path(__file__).resolve().parents[3]
TRAINING_PYTHON = sys.executable


def cache_environment(root: Path) -> dict[str, str]:
    hf = root / "cache/huggingface"
    return {
        "HF_HOME": str(hf),
        "HF_HUB_CACHE": str(hf / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(hf / "hub"),
        "HF_DATASETS_CACHE": str(hf / "datasets"),
        "TORCH_HOME": str(root / "cache/torch"),
    }


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    atomic_text(
        path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as f:
        name = Path(f.name)
        f.write(text)
    try:
        os.replace(name, path)
    finally:
        name.unlink(missing_ok=True)


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes its root: {relative!r}")
    return path


def load_sources() -> dict:
    import yaml

    return yaml.safe_load(
        (PROJECT / "configs/evaluation/sources.lock.yaml").read_text()
    )["benchmarks"]
