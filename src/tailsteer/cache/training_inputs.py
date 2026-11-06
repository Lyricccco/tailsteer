"""Task selection and lightweight preflight for native training caches."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .format import POINT_EXTENSION_MARKER, mask_tensor_key

DATA_TASKS = ("local_edit", "noop")
RECIPE_TASKS = {
    "local-edit-filtered": ("local_edit",),
    "noop-filtered": ("noop",),
    "annotated-mixed": DATA_TASKS,
}


def validate_tasks(tasks: Sequence[str]) -> tuple[str, ...]:
    if isinstance(tasks, str) or not isinstance(tasks, (list, tuple)):
        raise TypeError("cache tasks must be a list of local_edit/noop")
    values = tuple(tasks)
    if not values or any(task not in DATA_TASKS for task in values):
        raise ValueError(f"cache tasks must select from {DATA_TASKS}")
    if len(values) != len(set(values)):
        raise ValueError("cache tasks must be unique")
    return values


def configured_cache_tasks(cache: Mapping[str, Any]) -> tuple[str, ...]:
    task = cache.get("task")
    available = DATA_TASKS if task == "mixed" else (task,)
    if any(value not in DATA_TASKS for value in available):
        raise ValueError("each training cache requires task: local_edit/noop/mixed")
    selected = validate_tasks(cache.get("tasks", list(available)))
    if not set(selected).issubset(available):
        raise ValueError("cache tasks conflict with its declared task")
    return selected


def select_task_records(records, tasks: Sequence[str] | None):
    if tasks is None:
        return records
    selected = set(validate_tasks(tasks))
    result = tuple(record for record in records if record["task"] in selected)
    if not result:
        raise ValueError(f"cache has no records for tasks {sorted(selected)}")
    return result


def validate_training_cache_inputs(config: Mapping[str, Any]):
    """Check declared inputs without loading weights or scanning every shard.

    Dataset readers still perform the full tensor/sidecar validation at startup.
    Returns resolved entries so submit and shell launchers share one preflight.
    """
    from tailsteer.utils import cache_root, workspace_root

    caches = config.get("data", {}).get("caches")
    if not isinstance(caches, list) or not caches:
        raise ValueError("training data.caches must be a non-empty list")
    result = []
    for cache in caches:
        if not isinstance(cache, Mapping) or not isinstance(cache.get("path"), str):
            raise TypeError("each training cache requires a path")
        text = cache["path"].replace("${TAILSTEER_CACHE_ROOT}", str(cache_root()))
        text = text.replace("${TAILSTEER_ROOT}", str(workspace_root()))
        path = Path(os.path.expandvars(os.path.expanduser(text))).resolve()
        if not (path / "COMPLETE").is_file() or not (path / "manifest.json").is_file():
            raise FileNotFoundError(f"trajectory cache is incomplete: {path}")
        if (path / POINT_EXTENSION_MARKER).exists():
            raise RuntimeError(f"cache point extension is in progress: {path}")
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        expected_model = config.get("model", {}).get("name")
        if manifest.get("model") != expected_model:
            raise ValueError(
                f"cache {path} model differs from training: "
                f"{manifest.get('model')!r} != {expected_model!r}"
            )
        creation = manifest.get("creation_config", {}).get("cache", {})
        available = RECIPE_TASKS.get(creation.get("recipe"))
        declared = cache.get("task")
        expected = DATA_TASKS if declared == "mixed" else (declared,)
        if available != expected:
            raise ValueError(
                f"trajectory cache {path} task differs from config: "
                f"manifest={available!r} config={declared!r}"
            )
        configured_cache_tasks(cache)
        variant = cache.get("mask_variant", "default")
        mask_tensor_key(variant)
        if variant != "default" and variant not in creation.get("mask_variants", []):
            raise ValueError(
                f"cache {path} does not contain requested mask_variant: {variant}"
            )
        result.append((cache, path, manifest))
    return result
