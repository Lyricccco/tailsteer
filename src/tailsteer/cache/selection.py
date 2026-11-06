"""Export accepted cache sources as a deterministic trajectory replay manifest."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .format import read_cache_records

SELECTION_MODE = "accepted_cache_replay"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(path: Path) -> dict[str, Any]:
    if not (path / "COMPLETE").is_file():
        raise RuntimeError(f"cache is incomplete: {path}")
    manifest_path = path / "manifest.json"
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"cache manifest must be a mapping: {manifest_path}")
    return value


def _consistent_value(
    source_id: str,
    records: Sequence[Mapping[str, Any]],
    key: str,
) -> Any:
    encoded = {
        json.dumps(record.get(key), sort_keys=True, separators=(",", ":"))
        for record in records
    }
    if len(encoded) != 1:
        raise ValueError(f"cache source {source_id!r} has conflicting {key!r}")
    return records[0].get(key)


def _canonical_sample(
    source_id: str, records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    source_identity = _consistent_value(source_id, records, "source_identity")
    source_path = _consistent_value(source_id, records, "source_path")
    edited_path = _consistent_value(source_id, records, "edited_path")
    prompt = _consistent_value(source_id, records, "prompt")
    seed = _consistent_value(source_id, records, "seed")
    if not source_identity or not source_path:
        raise ValueError(f"cache source {source_id!r} lacks source provenance")
    if prompt is None or seed is None:
        raise ValueError(f"cache source {source_id!r} lacks prompt or seed")
    return {
        "id": source_id,
        "source_identity": str(source_identity),
        "source": str(source_path),
        "edited": None if edited_path is None else str(edited_path),
        "prompt": str(prompt),
        "seed": int(seed),
    }


def _selection_samples(
    cache: Path, manifest: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], str]:
    records = read_cache_records(cache)
    if len(records) != int(manifest["records"]):
        raise ValueError("cache manifest record count differs from sidecars")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tasks: set[str] = set()
    for record in records:
        source_id = str(record["source_id"])
        grouped[source_id].append(record)
        tasks.add(str(record.get("task", "")))
    if len(grouped) != int(manifest["sources"]):
        raise ValueError("cache manifest source count differs from sidecars")
    if len(tasks) != 1 or tasks == {""}:
        raise ValueError(f"cache selection requires one source task, got {tasks}")
    required_points = set(map(int, manifest["state_points"]))
    samples = []
    for source_id, source_records in grouped.items():
        points = {int(record["step_index"]) for record in source_records}
        if points != required_points:
            raise ValueError(f"cache source {source_id!r} has incomplete state points")
        samples.append(_canonical_sample(source_id, source_records))
    samples.sort(key=lambda value: (value["source_identity"], value["id"]))
    return samples, tasks.pop()


def validate_cache_selection(
    metadata_path: str | Path,
    *,
    samples_path: str | Path | None = None,
    expected_sources: int | None = None,
) -> dict[str, Any]:
    """Validate an exported replay selection and return its metadata."""

    metadata_file = Path(metadata_path).expanduser().resolve()
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("mode") != SELECTION_MODE:
        raise ValueError(f"invalid cache selection metadata: {metadata_file}")
    declared_samples = (metadata_file.parent / str(metadata["samples_file"])).resolve()
    resolved_samples = (
        declared_samples
        if samples_path is None
        else Path(samples_path).expanduser().resolve()
    )
    if resolved_samples != declared_samples:
        raise ValueError("cache selection metadata refers to another samples file")
    if _sha256(resolved_samples) != str(metadata["samples_sha256"]):
        raise ValueError("cache selection samples checksum mismatch")
    lines = sum(
        bool(line.strip())
        for line in resolved_samples.read_text(encoding="utf-8").splitlines()
    )
    if lines != int(metadata["sources"]):
        raise ValueError("cache selection source count differs from samples file")
    if expected_sources is not None and lines != int(expected_sources):
        raise ValueError(
            f"cache selection has {lines} sources, expected {expected_sources}"
        )
    return metadata


def export_cache_selection(
    source_cache: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Publish one canonical JSONL record per accepted cache source."""

    source = Path(source_cache).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = _manifest(source)
    samples, task = _selection_samples(source, manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.tmp-{uuid.uuid4().hex}")
    staging.mkdir()
    try:
        samples_file = staging / "samples.jsonl"
        samples_file.write_text(
            "".join(json.dumps(sample, sort_keys=True) + "\n" for sample in samples),
            encoding="utf-8",
        )
        source_manifest = source / "manifest.json"
        metadata = {
            "schema_version": 1,
            "mode": SELECTION_MODE,
            "source_cache": str(source),
            "source_cache_manifest_sha256": _sha256(source_manifest),
            "source_cache_model": str(manifest.get("model", "")),
            "source_cache_storage_layout": str(
                manifest.get("storage_layout", "step_records_v1")
            ),
            "source_cache_state_points": list(map(int, manifest["state_points"])),
            "source_task": task,
            "sources": len(samples),
            "samples_file": samples_file.name,
            "samples_sha256": _sha256(samples_file),
        }
        (staging / "selection.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_cache_selection(
        output / "selection.json", expected_sources=len(samples)
    )
