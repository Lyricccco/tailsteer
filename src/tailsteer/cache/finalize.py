"""Publish complete trajectories already committed to cache staging.

This module intentionally does not load a model.  It is the recovery path for a
trajectory job that stopped after committing useful shards but before writing
the final manifest (for example, after a distributed shutdown failure).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .format import (
    STEP_RECORDS_LAYOUT,
    TRAJECTORY_GROUPED_LAYOUT,
    read_cache_records,
    validate_cache,
)
from .trajectory import _git_commit, _load_records, _read_progress, _source_id

_SHARD_NAME = re.compile(r"^shard-(\d+)\.safetensors$")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_build_config(staging: Path) -> dict[str, Any]:
    path = staging / "build-config.json"
    if not path.is_file():
        raise FileNotFoundError(f"cache staging lacks build-config.json: {staging}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"cache build config must be a mapping: {path}")
    if not isinstance(value.get("model"), Mapping):
        raise TypeError(f"cache build config lacks model mapping: {path}")
    if not isinstance(value.get("data"), Mapping):
        raise TypeError(f"cache build config lacks data mapping: {path}")
    if not isinstance(value.get("cache"), Mapping):
        raise TypeError(f"cache build config lacks cache mapping: {path}")
    return value


def _shard_rank(name: str) -> int:
    match = _SHARD_NAME.fullmatch(name)
    if match is None:
        raise ValueError(f"invalid cache shard name: {name!r}")
    return int(match.group(1)) // 1_000_000


def _stopped_candidate_coverage(staging, config, accepted_ids):
    """Separate filter rejection from unfinished work in exhaustive jobs.

    Committed trajectories are authoritative even if shutdown interrupted the
    following journal append. Journals cannot stand in for missing tensors.
    """
    cache, data = config["cache"], config["data"]
    manifest = Path(data["manifest"])
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if data.get("manifest_sha256", digest) != digest:
        raise ValueError("candidate manifest checksum changed")
    start, end = int(cache.get("candidate_start", 0)), int(cache["candidate_end"])
    candidates = _load_records(manifest, start=start, end=end)
    if len(candidates) != end - start:
        raise ValueError("candidate range extends beyond the immutable manifest")
    rows = {}
    for _, row in candidates:
        sid, _ = _source_id(row, Path(f"line-{row['_line_number']}"))
        if sid in rows:
            raise ValueError(f"duplicate candidate ID: {sid}")
        rows[sid] = row
    outcomes = {}
    for journal in sorted(staging.glob("progress-rank-*.jsonl")):
        for value in _read_progress(journal):
            outcome = value["outcome"]
            if outcome not in {"accepted", "rejected", "failed"}:
                raise ValueError(f"unknown candidate outcome: {outcome}")
            outcomes[str(value["source_id"])] = outcome
    unknown = (set(outcomes) | accepted_ids) - set(rows)
    if unknown:
        raise ValueError(f"cache contains unknown candidate IDs: {sorted(unknown)[:8]}")
    missing = {
        sid for sid, status in outcomes.items() if status == "accepted"
    } - accepted_ids
    if missing:
        raise ValueError(
            "accepted journal entries lack committed trajectories: "
            f"{sorted(missing)[:8]}"
        )
    outcomes.update(dict.fromkeys(accepted_ids, "accepted"))

    def summarize(items):
        counts = Counter(outcomes.get(sid, "unprocessed") for sid, _ in items)
        accepted = [(sid, row) for sid, row in items if sid in accepted_ids]
        return {
            "candidates": len(items),
            **{
                name: counts[name]
                for name in ("accepted", "rejected", "failed", "unprocessed")
            },
            "source_images": len(
                {row.get("source_identity", sid) for sid, row in items}
            ),
            "accepted_source_images": len(
                {row.get("source_identity", sid) for sid, row in accepted}
            ),
            "accepted_tasks": dict(
                Counter(row.get("task", "unspecified") for _, row in accepted)
            ),
        }

    categories = defaultdict(list)
    for sid, row in rows.items():
        categories[row.get("sampling_category", "unspecified")].append((sid, row))
    result = summarize(list(rows.items()))
    result["complete"] = not (result["failed"] or result["unprocessed"])
    result["categories"] = {
        key: summarize(items) for key, items in sorted(categories.items())
    }
    return result


def inspect_trajectory_staging(staging_dir: str | Path) -> dict[str, Any]:
    """Validate committed shard/sidecar topology without writing any files."""

    staging = Path(staging_dir).expanduser().resolve()
    if not staging.is_dir():
        raise FileNotFoundError(staging)
    config = _load_build_config(staging)
    cache = config["cache"]
    output = Path(str(cache["output_dir"])).expanduser().resolve()
    expected_staging = output.with_name(output.name + ".building")
    if staging != expected_staging:
        raise ValueError(
            "staging path does not match cache.output_dir: "
            f"{staging} != {expected_staging}"
        )
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing cache output: {output}")
    if (staging / "COMPLETE").exists() or (staging / "manifest.json").exists():
        raise RuntimeError(
            f"staging already has publication markers; inspect it manually: {staging}"
        )
    temporary_files = sorted(
        path.name
        for path in staging.iterdir()
        if path.name.startswith(".") and "tmp" in path.name and path.is_file()
    )
    if temporary_files:
        raise RuntimeError(
            f"staging has uncommitted temporary files: {temporary_files[:8]}"
        )

    shard_names = {path.name for path in staging.glob("shard-*.safetensors")}
    sidecar_names = {path.name for path in staging.glob("metadata-*.jsonl")}
    expected_sidecars = {
        name.replace("shard-", "metadata-", 1).replace(".safetensors", ".jsonl")
        for name in shard_names
    }
    if sidecar_names != expected_sidecars:
        missing = sorted(expected_sidecars - sidecar_names)
        orphaned = sorted(sidecar_names - expected_sidecars)
        raise ValueError(
            "cache shard/metadata pairs differ: "
            f"missing_sidecars={missing[:8]}, orphaned_sidecars={orphaned[:8]}"
        )
    if not shard_names:
        raise ValueError(f"cache staging has no committed shards: {staging}")

    records = read_cache_records(staging)
    if not records:
        raise ValueError(f"cache staging has no metadata records: {staging}")
    referenced_shards = {str(record["shard"]) for record in records}
    if referenced_shards != shard_names:
        missing = sorted(shard_names - referenced_shards)
        unknown = sorted(referenced_shards - shard_names)
        raise ValueError(
            "cache metadata/shard references differ: "
            f"unreferenced={missing[:8]}, missing_shards={unknown[:8]}"
        )

    expected_points = [
        *map(int, cache["forward_points"]),
        int(cache["final_point"]),
    ]
    required_points = set(expected_points)
    source_steps: dict[str, set[int]] = defaultdict(set)
    source_ranks: dict[str, set[int]] = defaultdict(set)
    sample_ids: list[str] = []
    for record in records:
        source_id = str(record["source_id"])
        source_steps[source_id].add(int(record["step_index"]))
        source_ranks[source_id].add(_shard_rank(str(record["shard"])))
        sample_ids.append(str(record["sample_id"]))
    duplicates = [
        sample_id for sample_id, count in Counter(sample_ids).items() if count > 1
    ]
    if duplicates:
        raise ValueError(f"duplicate cache sample ids: {duplicates[:8]}")
    incomplete = {
        source_id: sorted(points)
        for source_id, points in source_steps.items()
        if points != required_points
    }
    if incomplete:
        preview = list(incomplete.items())[:8]
        raise ValueError(f"cache contains incomplete source trajectories: {preview}")
    split_sources = [
        source_id for source_id, ranks in source_ranks.items() if len(ranks) != 1
    ]
    if split_sources:
        raise ValueError(f"source trajectories span cache ranks: {split_sources[:8]}")

    sources_by_rank = Counter(
        next(iter(source_ranks[source_id])) for source_id in source_steps
    )
    exhaustive = cache.get("completion_mode") == "candidates_exhausted"
    coverage = (
        _stopped_candidate_coverage(staging, config, set(source_steps))
        if exhaustive
        else None
    )
    requested_sources = None if exhaustive else int(cache["target_accepted"])
    published_sources = len(source_steps)
    return {
        "staging": str(staging),
        "output": str(output),
        "model": str(config["model"]["name"]),
        "storage_layout": str(cache.get("storage_layout", STEP_RECORDS_LAYOUT)),
        "state_points": expected_points,
        "requested_sources": requested_sources,
        "sources": published_sources,
        "shortfall": (
            coverage["failed"] + coverage["unprocessed"]
            if coverage is not None
            else max(0, requested_sources - published_sources)
        ),
        **(
            {"candidate_coverage": coverage, "shortfall_unit": "candidates"}
            if exhaustive
            else {}
        ),
        "records": len(records),
        "shards": len(shard_names),
        "sources_by_rank": {
            str(rank): sources_by_rank[rank] for rank in sorted(sources_by_rank)
        },
        "config": config,
    }


def finalize_trajectory_staging(
    staging_dir: str | Path,
    *,
    allow_shortfall: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Atomically publish complete trajectories already present in staging.

    A target shortfall is never hidden: it requires explicit permission and is
    recorded in the final manifest.  Visualizations are deliberately omitted so
    they can be rendered later with the decoder-only backfill path.
    """

    inspection = inspect_trajectory_staging(staging_dir)
    if (
        inspection["requested_sources"] is not None
        and inspection["sources"] > inspection["requested_sources"]
    ):
        raise ValueError(
            "staging has more sources than the configured target: "
            f"{inspection['sources']} > {inspection['requested_sources']}"
        )
    if inspection["shortfall"] and not allow_shortfall:
        raise RuntimeError(
            "cache has unfinished work; rerun with --allow-shortfall to "
            f"publish the accepted trajectories: shortfall={inspection['shortfall']}"
        )
    public_inspection = {
        key: value for key, value in inspection.items() if key != "config"
    }
    if dry_run:
        return {**public_inspection, "dry_run": True, "published": False}

    staging = Path(inspection["staging"])
    output = Path(inspection["output"])
    config = inspection["config"]
    cache = config["cache"]
    model = config["model"]
    data = config["data"]
    lock = staging / ".finalize-existing.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise RuntimeError(f"another finalizer owns cache staging: {staging}") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
    finally:
        os.close(descriptor)

    storage_layout = str(inspection["storage_layout"])
    if storage_layout not in {STEP_RECORDS_LAYOUT, TRAJECTORY_GROUPED_LAYOUT}:
        lock.unlink(missing_ok=True)
        raise ValueError(f"unsupported cache storage layout: {storage_layout!r}")
    sources_by_rank = {
        int(rank): int(count) for rank, count in inspection["sources_by_rank"].items()
    }
    inferred_world_size = max(sources_by_rank, default=-1) + 1
    requested_sources = inspection["requested_sources"]
    rank_status = [
        {
            "rank": rank,
            "global_rank": rank,
            "accepted": count,
            **(
                {
                    "requested_quota": requested_sources // inferred_world_size
                    + int(rank < requested_sources % inferred_world_size),
                    "shortfall": max(
                        0,
                        requested_sources // inferred_world_size
                        + int(rank < requested_sources % inferred_world_size)
                        - count,
                    ),
                }
                if requested_sources is not None
                else {}
            ),
            "finalized_from_existing": True,
        }
        for rank, count in sorted(sources_by_rank.items())
    ]
    repository = Path(__file__).resolve().parents[3]
    manifest = {
        "schema_version": (2 if storage_layout == TRAJECTORY_GROUPED_LAYOUT else 1),
        "storage_layout": storage_layout,
        "model": str(model["name"]),
        "model_path": str(model["path"]),
        "model_code_path": str(model["code_path"]),
        "model_code_revision": _git_commit(Path(str(model["code_path"]))),
        "source_dataset": str(data["manifest"]),
        "state_points": inspection["state_points"],
        "representation": str(model.get("state_representation", "native")),
        "sources": int(inspection["sources"]),
        "requested_sources": requested_sources,
        "records": int(inspection["records"]),
        "tailsteer_commit": _git_commit(repository),
        "world_size": inferred_world_size,
        "global_world_size": inferred_world_size,
        "task_shard_id": int(cache.get("task_shard_id", 0)),
        "num_task_shards": int(cache.get("num_task_shards", 1)),
        "rank_status": rank_status,
        "finalization": {
            "schema_version": 1,
            "mode": "accepted_existing_trajectories",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "requested_sources": requested_sources,
            "published_sources": int(inspection["sources"]),
            "shortfall": int(inspection["shortfall"]),
            "model_loaded": False,
            "visualizations_deferred": True,
        },
        "creation_config": config,
    }
    if "candidate_coverage" in inspection:
        manifest["candidate_coverage"] = inspection["candidate_coverage"]
        manifest["finalization"].update(
            mode="early_stopped_candidates"
            if inspection["shortfall"]
            else "accepted_existing_trajectories",
            shortfall_unit="candidates",
            requested_candidates=inspection["candidate_coverage"]["candidates"],
            candidate_coverage_complete=inspection["candidate_coverage"]["complete"],
        )
    manifest_path = staging / "manifest.json"
    complete_path = staging / "COMPLETE"
    try:
        _atomic_json(manifest_path, manifest)
        complete_path.write_text("complete\n", encoding="utf-8")
        validate_cache(staging)
        (staging / ".trajectory-rendezvous.json").unlink(missing_ok=True)
        for status in staging.glob(".trajectory-rank-*.json"):
            status.unlink(missing_ok=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output)
        (output / lock.name).unlink(missing_ok=True)
    except BaseException:
        complete_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)
        raise
    return {
        **public_inspection,
        "dry_run": False,
        "published": True,
        "manifest": validate_cache(output),
    }
