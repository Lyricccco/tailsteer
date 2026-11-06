"""Frozen trajectory selections, applied after the existing source split."""

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from .dataset import cache_record
from .split import source_partition
from .training_inputs import configured_cache_tasks


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_selection(path, data, *, sha256=None):
    path = Path(path).expanduser().resolve()
    payload = path.read_bytes()
    if sha256 is not None and hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError("Training selection changed after experiment preparation")
    selection = json.loads(payload)
    if selection.get("schema_version") != 1:
        raise ValueError("Unsupported training selection schema")
    caches = data["caches"]
    if len(caches) != 1:
        raise ValueError("Training selection requires one explicitly identified cache")
    from tailsteer.utils import cache_root

    configured = str(caches[0]["path"]).replace(
        "${TAILSTEER_CACHE_ROOT}", str(cache_root())
    )
    cache = Path(os.path.expandvars(configured)).expanduser().resolve()
    if cache != Path(selection["cache"]).resolve():
        raise ValueError("Training selection refers to another cache")
    if file_sha256(cache / "manifest.json") != selection["cache_manifest_sha256"]:
        raise ValueError("Selected cache manifest changed")
    split = {
        "seed": int(data.get("split_seed", 42)),
        "validation_fraction": float(data.get("validation_fraction", 0.01)),
    }
    if split != selection["split"]:
        raise ValueError("Training selection source split differs from config")
    rows = selection["trajectories"]
    if not rows or len({r["source_id"] for r in rows}) != len(rows):
        raise ValueError("Training selection must contain unique trajectory IDs")
    allowed = configured_cache_tasks(caches[0])
    for row in rows:
        if row["task"] not in allowed:
            raise ValueError("Training selection conflicts with --data-task")
        if source_partition(row["source_identity"], **split) != "train":
            raise ValueError("Training selection contains a validation source")
    return selection


def pin_selection(path, data):
    path = Path(path).expanduser().resolve()
    sha256 = file_sha256(path)
    read_selection(path, data, sha256=sha256)
    return {"path": str(path), "sha256": sha256}


def select_training_dataset(dataset, data):
    """Keep all states of selected trajectories; never touch validation items."""
    spec = data.get("train_selection")
    if spec is None:
        return dataset, None
    from torch.utils.data import Subset

    selection = read_selection(spec["path"], data, sha256=spec["sha256"])
    requested = {r["source_id"]: r for r in selection["trajectories"]}
    found = {key: [] for key in requested}
    for index in range(len(dataset)):
        record = cache_record(dataset, index)
        key = record["source_id"]
        if key not in requested:
            continue
        if any(
            record.get(field) != requested[key][field]
            for field in ("source_identity", "task", "sampling_category")
        ):
            raise ValueError(f"Training selection metadata differs: {key}")
        found[key].append(index)
    missing = [key for key, indices in found.items() if not indices]
    if missing:
        raise ValueError(f"Selected trajectories missing from training: {missing[:5]}")
    # Manifest order makes the three paired ablations use the same image order.
    indices = [
        index for row in selection["trajectories"] for index in found[row["source_id"]]
    ]
    summary = {
        **spec,
        "original_items": len(dataset),
        "selected_items": len(indices),
        "trajectories": len(requested),
        "source_images": len({r["source_identity"] for r in requested.values()}),
        "tasks": dict(Counter(r["task"] for r in requested.values())),
        "categories": dict(Counter(r["sampling_category"] for r in requested.values())),
    }
    return Subset(dataset, indices), summary
