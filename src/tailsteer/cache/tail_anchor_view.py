"""Strict reader and one-time canonicalizer for zero-copy tail-anchor views."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .sample import CacheSample, normalize_metadata

CANONICAL_TASKS = frozenset({"local_edit", "noop"})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _read_view_records(path: Path) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for sidecar in sorted(path.glob("metadata-*.jsonl")):
        for line_number, line in enumerate(
            sidecar.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"invalid view record at {sidecar}:{line_number}")
            records.append(value)
    return tuple(records)


def _metadata_object(record: Mapping[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        raise TypeError("tail-anchor view record lacks nested metadata")
    return dict(metadata)


def _flatten_record(record: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _metadata_object(record)
    point = metadata.get("point")
    if not isinstance(point, Mapping):
        raise TypeError("tail-anchor view metadata lacks schedule point")
    extra = metadata.get("extra", {})
    if not isinstance(extra, Mapping):
        raise TypeError("tail-anchor view metadata.extra must be a mapping")
    source_id = extra.get("source_sample_id")
    if source_id is None:
        raise KeyError("tail-anchor view metadata lacks extra.source_sample_id")
    return normalize_metadata(
        {
            **metadata,
            "source_id": source_id,
            "step_index": point["step_index"],
            "sigma": point["sigma"],
            "timestep": point["timestep"],
            "shard": str(record["shard"]),
            "offset": int(record["offset"]),
            "has_reference": bool(record.get("has_reference", False)),
            "mask_key": str(metadata.get("mask_key", "m_keep_latent")),
        }
    )


def validate_tail_anchor_view(
    path: str | Path,
    *,
    shard_root: str | Path,
    expected_records: int | None = None,
    expected_task: str | None = None,
    verify_shard_hashes: bool = False,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Validate a canonical, committed view and its external tensor shards."""

    from safetensors import safe_open

    root = Path(path).expanduser().resolve()
    tensors = Path(shard_root).expanduser().resolve()
    if not (root / "COMPLETE").is_file():
        raise RuntimeError(f"tail-anchor view is incomplete: {root}")
    commit_path = root / "commit.json"
    if not commit_path.is_file():
        raise FileNotFoundError(commit_path)
    commit = _read_json(commit_path)
    if commit.get("kind") != "tail_anchor_cache_view":
        raise ValueError(f"unsupported tail-anchor view kind: {commit.get('kind')!r}")

    objects = commit.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("tail-anchor view commit has no metadata objects")
    for descriptor in objects:
        if not isinstance(descriptor, Mapping):
            raise TypeError("tail-anchor view object descriptor must be a mapping")
        object_path = root / str(descriptor["relative_path"])
        if not object_path.is_file():
            raise FileNotFoundError(object_path)
        if object_path.stat().st_size != int(descriptor["size"]):
            raise ValueError(f"tail-anchor view object size differs: {object_path}")
        if _sha256_file(object_path) != str(descriptor["sha256"]):
            raise ValueError(f"tail-anchor view object checksum differs: {object_path}")

    raw_records = _read_view_records(root)
    metadata = commit.get("metadata")
    if not isinstance(metadata, Mapping):
        raise TypeError("tail-anchor view commit metadata must be a mapping")
    committed_records = int(metadata.get("records", -1))
    if len(raw_records) != committed_records:
        raise ValueError("tail-anchor view record count differs from commit")
    if expected_records is not None and len(raw_records) != int(expected_records):
        raise ValueError(
            f"tail-anchor view has {len(raw_records)} records, "
            f"expected {int(expected_records)}"
        )

    records = tuple(_flatten_record(record) for record in raw_records)
    sample_ids = [str(record["sample_id"]) for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("tail-anchor view contains duplicate sample_id values")
    source_counts = Counter(str(record["source_identity"]) for record in records)
    if len(source_counts) != int(metadata.get("source_samples", -1)):
        raise ValueError("tail-anchor view source count differs from commit")
    records_per_source = int(metadata.get("records_per_source", 1))
    if set(source_counts.values()) != {records_per_source}:
        raise ValueError("tail-anchor view records-per-source contract differs")
    state_points = {int(record["step_index"]) for record in records}
    committed_points = {int(point) for point in metadata.get("state_points", [])}
    if state_points != committed_points:
        raise ValueError("tail-anchor view state points differ from commit")
    task_counts = Counter(str(record["task"]) for record in records)
    invalid_tasks = sorted(set(task_counts) - CANONICAL_TASKS)
    if invalid_tasks:
        raise ValueError(
            "tail-anchor view contains non-canonical tasks "
            f"{invalid_tasks}; rebuild it with scripts/prepare_flux2_training_views.py"
        )
    if expected_task is not None:
        if expected_task not in CANONICAL_TASKS:
            raise ValueError(f"expected_task is not canonical: {expected_task!r}")
        if task_counts != Counter({expected_task: len(records)}):
            raise ValueError(
                f"tail-anchor view tasks are {dict(task_counts)}, "
                f"expected only {expected_task!r}"
            )

    external = metadata.get("external_shards")
    if not isinstance(external, list):
        raise TypeError("tail-anchor view commit lacks external_shards")
    descriptors = {
        str(item["relative_path"]): item
        for item in external
        if isinstance(item, Mapping)
    }
    by_shard: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_shard.setdefault(str(record["shard"]), []).append(record)
    missing_descriptors = sorted(set(by_shard) - set(descriptors))
    if missing_descriptors:
        raise ValueError(
            f"tail-anchor view lacks descriptors for shards: {missing_descriptors}"
        )
    for shard_name, shard_records in by_shard.items():
        shard = tensors / shard_name
        if not shard.is_file():
            raise FileNotFoundError(shard)
        descriptor = descriptors[shard_name]
        if shard.stat().st_size != int(descriptor["size"]):
            raise ValueError(f"external shard size differs: {shard}")
        if verify_shard_hashes and _sha256_file(shard) != str(
            descriptor["source_sha256"]
        ):
            raise ValueError(f"external shard checksum differs: {shard}")
        with safe_open(shard, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            required = {"anchor_latent", "source_latent"}
            required.update(f"mask.{item['mask_key']}" for item in shard_records)
            missing = sorted(required - keys)
            if missing:
                raise ValueError(f"external shard {shard} lacks tensors: {missing}")
            largest_offset = max(int(item["offset"]) for item in shard_records)
            for key in required:
                if largest_offset >= int(handle.get_slice(key).get_shape()[0]):
                    raise ValueError(
                        f"external shard offset exceeds tensor {key}: {shard}"
                    )
    return commit, records


class TailAnchorViewDataset:
    """Read canonical JSONL metadata while tensors stay in immutable shards."""

    cache_format = "tail_anchor_view"

    def __init__(
        self,
        path: str | Path,
        *,
        shard_root: str | Path,
        steps: Sequence[int] | None = None,
        expected_records: int | None = None,
        expected_task: str | None = None,
        verify_shard_hashes: bool = False,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.shard_root = Path(shard_root).expanduser().resolve()
        _, records = validate_tail_anchor_view(
            self.path,
            shard_root=self.shard_root,
            expected_records=expected_records,
            expected_task=expected_task,
            verify_shard_hashes=verify_shard_hashes,
        )
        if steps is not None:
            allowed = {int(step) for step in steps}
            records = tuple(
                record for record in records if int(record["step_index"]) in allowed
            )
        if not records:
            raise ValueError(f"cache dataset has no selected records: {self.path}")
        if expected_records is not None and len(records) != int(expected_records):
            raise ValueError(
                f"tail-anchor view selects {len(records)} records, "
                f"expected {int(expected_records)}"
            )
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> CacheSample:
        from safetensors import safe_open

        record = self.records[index]
        with safe_open(
            self.shard_root / str(record["shard"]), framework="pt", device="cpu"
        ) as handle:
            offset = int(record["offset"])
            reference = (
                handle.get_tensor("reference_latent")[offset]
                if record["has_reference"]
                else None
            )
            return CacheSample(
                latent=handle.get_tensor("anchor_latent")[offset],
                source_latent=handle.get_tensor("source_latent")[offset],
                keep_mask=handle.get_tensor(f"mask.{record['mask_key']}")[offset],
                reference_latent=reference,
                metadata=record,
            )


def canonicalize_tail_anchor_view(
    source: str | Path,
    output: str | Path,
    *,
    shard_root: str | Path,
    canonical_task: str,
    accepted_source_tasks: Sequence[str],
    expected_records: int | None = None,
) -> Path:
    """Publish a new immutable view with a canonical training task label."""

    if canonical_task not in CANONICAL_TASKS:
        raise ValueError(f"canonical_task is not supported: {canonical_task!r}")
    source_root = Path(source).expanduser().resolve()
    output_root = Path(output).expanduser().resolve()
    tensor_root = Path(shard_root).expanduser().resolve()
    source_commit_path = source_root / "commit.json"
    source_commit = _read_json(source_commit_path)
    if source_commit.get("kind") != "tail_anchor_cache_view":
        raise ValueError(f"unsupported source view kind: {source_commit.get('kind')!r}")
    source_objects = source_commit.get("objects")
    if not isinstance(source_objects, list) or not source_objects:
        raise ValueError("source view commit has no metadata objects")
    for descriptor in source_objects:
        if not isinstance(descriptor, Mapping):
            raise TypeError("source view object descriptor must be a mapping")
        source_object = source_root / str(descriptor["relative_path"])
        if source_object.stat().st_size != int(descriptor["size"]):
            raise ValueError(f"source view object size differs: {source_object}")
        if _sha256_file(source_object) != str(descriptor["sha256"]):
            raise ValueError(f"source view object checksum differs: {source_object}")
    raw_records = _read_view_records(source_root)
    source_metadata = source_commit.get("metadata")
    if not isinstance(source_metadata, Mapping):
        raise TypeError("source view commit metadata must be a mapping")
    if len(raw_records) != int(source_metadata.get("records", -1)):
        raise ValueError("source view record count differs from commit")
    if expected_records is not None and len(raw_records) != int(expected_records):
        raise ValueError(
            f"source view has {len(raw_records)} records, "
            f"expected {int(expected_records)}"
        )
    accepted = set(accepted_source_tasks)
    source_task_counts: Counter[str] = Counter()
    canonical_records = []
    for record in raw_records:
        copied = dict(record)
        metadata = _metadata_object(record)
        source_task = str(metadata.get("task"))
        if source_task not in accepted:
            raise ValueError(
                f"source view has unexpected task {source_task!r}; "
                f"accepted={sorted(accepted)}"
            )
        source_task_counts[source_task] += 1
        extra = metadata.get("extra", {})
        if not isinstance(extra, Mapping):
            raise TypeError("source view metadata.extra must be a mapping")
        metadata["extra"] = {**dict(extra), "source_task": source_task}
        metadata["task"] = canonical_task
        copied["metadata"] = metadata
        canonical_records.append(copied)

    if output_root.exists():
        existing_commit, _ = validate_tail_anchor_view(
            output_root,
            shard_root=tensor_root,
            expected_records=len(canonical_records),
            expected_task=canonical_task,
        )
        existing_metadata = existing_commit.get("metadata", {})
        existing_view = (
            existing_metadata.get("training_view", {})
            if isinstance(existing_metadata, Mapping)
            else {}
        )
        expected_source_sha = _sha256_file(source_commit_path)
        if not isinstance(existing_view, Mapping) or (
            existing_view.get("source_view_commit_sha256") != expected_source_sha
        ):
            raise ValueError(
                f"existing canonical view was built from another source: {output_root}"
            )
        return output_root
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        sidecar = temporary / "metadata-000000.jsonl"
        content = "".join(
            json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
            for item in canonical_records
        )
        sidecar.write_text(content, encoding="utf-8")

        metadata = dict(source_metadata)
        external = []
        for descriptor in source_metadata.get("external_shards", []):
            if not isinstance(descriptor, Mapping):
                raise TypeError("source external shard descriptor must be a mapping")
            item = dict(descriptor)
            shard = tensor_root / str(item["relative_path"])
            if not shard.is_file():
                raise FileNotFoundError(shard)
            stat = shard.stat()
            if stat.st_size != int(item["size"]):
                raise ValueError(f"external shard size differs: {shard}")
            item.update(
                {
                    "source_path": str(shard),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "ctime_ns": stat.st_ctime_ns,
                }
            )
            external.append(item)
        metadata["external_shards"] = external
        metadata["dataset_content_fingerprint"] = _sha256_file(sidecar)
        training_view = dict(metadata.get("training_view", {}))
        training_view.update(
            {
                "canonical_task": canonical_task,
                "source_task_counts": dict(sorted(source_task_counts.items())),
                "source_view": str(source_root),
                "source_view_commit_sha256": _sha256_file(source_commit_path),
                "task_contract": "tailsteer_canonical_v1",
            }
        )
        metadata["training_view"] = training_view
        commit = {
            "schema_version": 1,
            "kind": "tail_anchor_cache_view",
            "run": {
                "run_id": output_root.name,
                "run_name": output_root.name,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            "metadata": metadata,
            "objects": [
                {
                    "relative_path": sidecar.name,
                    "media_type": "application/jsonl",
                    "size": sidecar.stat().st_size,
                    "sha256": _sha256_file(sidecar),
                }
            ],
        }
        (temporary / "commit.json").write_text(
            json.dumps(commit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "COMPLETE").write_text("complete\n", encoding="utf-8")
        os.replace(temporary, output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    validate_tail_anchor_view(
        output_root,
        shard_root=tensor_root,
        expected_records=len(canonical_records),
        expected_task=canonical_task,
    )
    return output_root
