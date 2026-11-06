"""Build immutable candidate pools from old caches or priority selections."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .format import read_cache_records
from .selection import validate_cache_selection

CONTINUATION_MODE = "partial_cache_continuation"
CONTINUATION_MARKER = "_tailsteer_continuation"
REPLAY_FILTER_POLICY = "revalidate_and_replace"


def is_continuation_replay(marker: Mapping[str, Any]) -> bool:
    """Return whether a continuation candidate came from an accepted old cache.

    ``must_accept`` is retained as a read-only compatibility alias for candidate
    pools published before replay candidates were revalidated by the current
    filter.  It must not be interpreted as a requirement to accept the sample.
    """

    return (
        str(marker.get("origin", "")) == "accepted_replay"
        or bool(marker.get("replay_source", False))
        or bool(marker.get("must_accept", False))
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_lines(
    path: Path, *, tolerate_truncated_tail: bool
) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        lines = handle.readlines() if tolerate_truncated_tail else handle
        total = len(lines) if tolerate_truncated_tail else None
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                if tolerate_truncated_tail and index + 1 == total:
                    break
                raise
            if not isinstance(value, dict):
                raise TypeError(f"JSONL record must be an object: {path}:{index + 1}")
            yield value


def _source_key(record: Mapping[str, Any], source: Any) -> tuple[str, str]:
    identity = str(
        record.get("source_identity")
        or record.get("openimages_image_id")
        or record.get("id")
        or source
    )
    value = str(record.get("id") or record.get("sample_id") or "")
    if not value:
        value = hashlib.sha256(identity.encode()).hexdigest()[:20]
    return value.replace("/", "_"), identity


def _consistent_value(
    source_id: str, records: Sequence[Mapping[str, Any]], key: str
) -> Any:
    encoded = {
        json.dumps(record.get(key), sort_keys=True, separators=(",", ":"))
        for record in records
    }
    if len(encoded) != 1:
        raise ValueError(f"partial cache source {source_id!r} has conflicting {key!r}")
    return records[0].get(key)


def _partial_roots(paths: Sequence[str | Path]) -> tuple[Path, ...]:
    roots: list[Path] = []
    for value in paths:
        path = Path(value).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        if (path / "config.yaml").is_file():
            roots.append(path)
            continue
        children = sorted(
            child
            for child in path.iterdir()
            if child.is_dir() and (child / "config.yaml").is_file()
        )
        if not children:
            raise ValueError(f"partial-cache root contains no cache parts: {path}")
        roots.extend(children)
    resolved = tuple(dict.fromkeys(roots))
    if not resolved:
        raise ValueError("continuation requires at least one partial cache root")
    return resolved


def _load_partial_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"partial cache config must be a mapping: {path}")
    return value


def _validate_partial_contract(
    config: Mapping[str, Any],
    *,
    root: Path,
    model_name: str,
    recipe: str,
    width: int,
    height: int,
    num_steps: int,
) -> tuple[int, ...]:
    model = config.get("model")
    cache = config.get("cache")
    if not isinstance(model, Mapping) or not isinstance(cache, Mapping):
        raise TypeError(f"partial cache lacks model/cache mappings: {root}")
    actual = (
        str(model.get("name")),
        str(cache.get("recipe")),
        int(cache.get("width", -1)),
        int(cache.get("height", -1)),
        int(cache.get("num_steps", -1)),
    )
    expected = (model_name, recipe, width, height, num_steps)
    if actual != expected:
        raise ValueError(
            f"partial cache contract differs at {root}: {actual} != {expected}"
        )
    points = tuple(
        [*map(int, cache.get("forward_points", ())), int(cache["final_point"])]
    )
    if not points or points != tuple(sorted(set(points))):
        raise ValueError(f"partial cache has invalid state points: {root}")
    return points


def _canonical_accepted_sample(
    source_id: str,
    records: Sequence[Mapping[str, Any]],
    *,
    root: Path,
    require_edited: bool,
) -> dict[str, Any]:
    source_identity = _consistent_value(source_id, records, "source_identity")
    source_path = _consistent_value(source_id, records, "source_path")
    edited_path = _consistent_value(source_id, records, "edited_path")
    prompt = _consistent_value(source_id, records, "prompt")
    seed = _consistent_value(source_id, records, "seed")
    if not source_identity or not source_path or (require_edited and not edited_path):
        raise ValueError(f"partial cache source {source_id!r} lacks image provenance")
    if prompt is None or seed is None:
        raise ValueError(f"partial cache source {source_id!r} lacks prompt or seed")
    return {
        "id": source_id,
        "source_identity": str(source_identity),
        "source": str(source_path),
        "edited": None if edited_path is None else str(edited_path),
        "prompt": str(prompt),
        "seed": int(seed),
        CONTINUATION_MARKER: {
            "origin": "accepted_replay",
            "replay_source": True,
            "partial_cache": str(root),
        },
    }


def _accepted_and_attempted(
    roots: Sequence[Path],
    *,
    model_name: str,
    recipe: str,
    width: int,
    height: int,
    num_steps: int,
    require_edited: bool,
) -> tuple[list[dict[str, Any]], set[str], list[dict[str, Any]]]:
    accepted: dict[str, dict[str, Any]] = {}
    attempted: set[str] = set()
    provenance: list[dict[str, Any]] = []
    for root in roots:
        config_path = root / "config.yaml"
        config = _load_partial_config(root)
        required_points = set(
            _validate_partial_contract(
                config,
                root=root,
                model_name=model_name,
                recipe=recipe,
                width=width,
                height=height,
                num_steps=num_steps,
            )
        )
        records = read_cache_records(root)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            source_id = str(record["source_id"])
            grouped[source_id].append(record)
            attempted.add(source_id)
            shard = root / str(record["shard"])
            if not shard.is_file():
                raise FileNotFoundError(shard)
        for source_id, source_records in grouped.items():
            points = {int(record["step_index"]) for record in source_records}
            if points != required_points:
                raise ValueError(
                    f"partial cache source {source_id!r} has incomplete state points"
                )
            sample = _canonical_accepted_sample(
                source_id, source_records, root=root, require_edited=require_edited
            )
            previous = accepted.get(source_id)
            if previous is not None:
                comparable = lambda item: {  # noqa: E731
                    key: value
                    for key, value in item.items()
                    if key != CONTINUATION_MARKER
                }
                if comparable(previous) != comparable(sample):
                    raise ValueError(
                        f"partial caches disagree about accepted source {source_id!r}"
                    )
            else:
                accepted[source_id] = sample
        journals = sorted(root.glob("progress-rank-*.jsonl"))
        journal_records = 0
        outcomes: dict[str, int] = defaultdict(int)
        for journal in journals:
            for value in _json_lines(journal, tolerate_truncated_tail=True):
                source_id = str(value["source_id"])
                attempted.add(source_id)
                outcomes[str(value.get("outcome", "unknown"))] += 1
                journal_records += 1
        provenance.append(
            {
                "root": str(root),
                "config_sha256": _sha256(config_path),
                "state_points": sorted(required_points),
                "accepted_sources": len(grouped),
                "progress_records": journal_records,
                "progress_outcomes": dict(sorted(outcomes.items())),
            }
        )
    samples = sorted(
        accepted.values(), key=lambda item: (item["source_identity"], item["id"])
    )
    return samples, attempted, provenance


def _raw_value(
    record: Mapping[str, Any], field: str, alternatives: Sequence[str]
) -> Any:
    for name in (field, *alternatives):
        value = record.get(name)
        if value not in {None, ""}:
            return value
    return None


def _selection_candidates(
    metadata_path: str | Path, *, recipe: str, require_edited: bool
) -> tuple[list[dict[str, Any]], set[str], dict[str, Any]]:
    """Reuse only source records, never another model's tensors or rejections."""
    path = Path(metadata_path).expanduser().resolve()
    metadata = validate_cache_selection(path)
    task = "local_edit" if recipe == "local-edit-filtered" else "noop"
    if metadata.get("source_task") != task:
        raise ValueError("seed selection task differs from the cache recipe")
    samples_path = path.parent / metadata["samples_file"]
    samples, seen = [], set()
    for record in _json_lines(samples_path, tolerate_truncated_tail=False):
        if any(
            record.get(key) in (None, "")
            for key in ("id", "source_identity", "source", "prompt", "seed")
        ):
            raise ValueError("seed selection record lacks source provenance")
        if require_edited and not record.get("edited"):
            raise ValueError("local seed selection lacks edited image")
        source_id = str(record["id"])
        if source_id in seen:
            raise ValueError(f"duplicate seed selection source ID: {source_id}")
        seen.add(source_id)
        samples.append(
            {
                **record,
                CONTINUATION_MARKER: {
                    "origin": "accepted_replay",
                    "replay_source": True,
                    "seed_selection": str(path),
                },
            }
        )
    return (
        samples,
        seen,
        {
            "metadata_path": str(path),
            "metadata_sha256": _sha256(path),
            "samples_sha256": metadata["samples_sha256"],
            "source_cache_model": metadata.get("source_cache_model"),
        },
    )


def _discovery_samples(
    raw_manifest: Path,
    *,
    fields: Mapping[str, Any],
    attempted: set[str],
    default_seed: int,
    require_edited: bool,
) -> tuple[list[dict[str, Any]], int]:
    samples: list[dict[str, Any]] = []
    seen = set(attempted)
    raw_records = 0
    for raw_ordinal, record in enumerate(
        _json_lines(raw_manifest, tolerate_truncated_tail=False)
    ):
        raw_records += 1
        source = _raw_value(
            record, str(fields.get("source", "source")), ("source_path", "image")
        )
        edited = _raw_value(
            record,
            str(fields.get("edited", "edited")),
            ("edited_path", "reference"),
        )
        prompt = _raw_value(
            record,
            str(fields.get("prompt", "prompt")),
            ("instruction", "text"),
        )
        if source is None or prompt is None or (require_edited and edited is None):
            required = "source/edited/prompt" if require_edited else "source/prompt"
            raise ValueError(
                f"raw continuation record lacks {required}: "
                f"{raw_manifest}:{raw_ordinal + 1}"
            )
        source_id, source_identity = _source_key(record, source)
        if source_id in seen:
            continue
        seen.add(source_id)
        seed_field = str(fields.get("seed", "seed"))
        samples.append(
            {
                "id": source_id,
                "source_identity": source_identity,
                "source": str(source),
                "edited": None if edited is None else str(edited),
                "prompt": str(prompt),
                "seed": int(record.get(seed_field, default_seed)),
                CONTINUATION_MARKER: {
                    "origin": "discovery",
                    "raw_candidate_ordinal": raw_ordinal,
                },
            }
        )
    return samples, raw_records


def validate_cache_continuation(
    metadata_path: str | Path,
    *,
    samples_path: str | Path | None = None,
    expected_model: str | None = None,
    expected_target: int | None = None,
    expected_reference_mode: str | None = None,
) -> dict[str, Any]:
    """Validate one immutable partial-cache continuation snapshot."""

    metadata_file = Path(metadata_path).expanduser().resolve()
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("mode") != CONTINUATION_MODE:
        raise ValueError(f"invalid cache continuation metadata: {metadata_file}")
    declared = (metadata_file.parent / str(metadata["samples_file"])).resolve()
    resolved = declared if samples_path is None else Path(samples_path).resolve()
    if resolved != declared:
        raise ValueError("cache continuation metadata refers to another samples file")
    if _sha256(resolved) != str(metadata["samples_sha256"]):
        raise ValueError("cache continuation samples checksum mismatch")
    samples = list(_json_lines(resolved, tolerate_truncated_tail=False))
    if len(samples) != int(metadata["candidates"]):
        raise ValueError("cache continuation candidate count differs from samples file")
    source_ids = [str(sample["id"]) for sample in samples]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("cache continuation samples contain duplicate source IDs")
    replay = sum(
        is_continuation_replay(sample.get(CONTINUATION_MARKER, {}))
        for sample in samples
    )
    if replay != int(metadata["replay_sources"]):
        raise ValueError("cache continuation replay count differs from samples file")
    if expected_model is not None and str(metadata["model"]) != expected_model:
        raise ValueError("cache continuation was built for another model")
    if expected_target is not None and int(metadata["target_sources"]) != int(
        expected_target
    ):
        raise ValueError("cache continuation was built for another target")
    if (
        expected_reference_mode is not None
        and metadata.get("reference_mode", "paired") != expected_reference_mode
    ):
        raise ValueError("cache continuation was built for another reference_mode")
    return metadata


def prepare_cache_continuation(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Publish the continuation snapshot declared by a trajectory config."""

    data = config.get("data")
    model = config.get("model")
    cache = config.get("cache")
    if not isinstance(data, Mapping):
        raise TypeError("trajectory config requires a data mapping")
    continuation = data.get("continuation")
    if continuation is None:
        return None
    if not isinstance(continuation, Mapping):
        raise TypeError("data.continuation must be a mapping")
    if continuation.get("mode") != CONTINUATION_MODE:
        raise ValueError(
            f"unsupported data.continuation mode: {continuation.get('mode')!r}"
        )
    if not isinstance(model, Mapping) or not isinstance(cache, Mapping):
        raise TypeError("trajectory config requires model and cache mappings")
    reference_mode = str(cache.get("filter", {}).get("reference_mode", "paired"))
    if reference_mode not in {"paired", "none"}:
        raise ValueError("cache.filter.reference_mode must be paired or none")
    require_edited = (
        cache["recipe"] == "local-edit-filtered" and reference_mode == "paired"
    )
    seed_selection = continuation.get("seed_selection")
    partial_paths = continuation.get("partial_cache_roots")
    if bool(seed_selection) == bool(partial_paths):
        raise ValueError(
            "continuation requires exactly one of seed_selection or partial_cache_roots"
        )
    samples_file = Path(str(data["manifest"])).expanduser().resolve()
    metadata_file = Path(str(continuation["metadata"])).expanduser().resolve()
    if samples_file.parent != metadata_file.parent:
        raise ValueError("continuation samples and metadata must share one directory")
    if samples_file.name != "samples.jsonl":
        raise ValueError("continuation manifest must be named samples.jsonl")
    output = samples_file.parent
    expected_model = str(model["name"])
    target = int(cache["target_accepted"])
    if output.exists():
        metadata = validate_cache_continuation(
            metadata_file,
            samples_path=samples_file,
            expected_model=expected_model,
            expected_target=target,
            expected_reference_mode=reference_mode,
        )
        if "preparation_config" in metadata and metadata["preparation_config"] != dict(
            continuation
        ):
            raise ValueError(
                "candidate pool has another preparation config; choose a new pool path"
            )
        return metadata
    seed_provenance = None
    if seed_selection:
        accepted, attempted, seed_provenance = _selection_candidates(
            seed_selection, recipe=str(cache["recipe"]), require_edited=require_edited
        )
        provenance = []
    else:
        roots = _partial_roots(partial_paths)
        accepted, attempted, provenance = _accepted_and_attempted(
            roots,
            model_name=expected_model,
            recipe=str(cache["recipe"]),
            width=int(cache["width"]),
            height=int(cache["height"]),
            num_steps=int(cache["num_steps"]),
            require_edited=require_edited,
        )
        if len(accepted) >= target:
            raise ValueError(
                f"partial cache already has {len(accepted)} accepted sources; "
                "use accepted_cache_replay instead of continuation"
            )
    raw_manifest = Path(str(continuation["raw_manifest"])).expanduser().resolve()
    if not raw_manifest.is_file():
        raise FileNotFoundError(raw_manifest)
    raw_fields = continuation.get("raw_fields")
    if not isinstance(raw_fields, Mapping):
        raise TypeError("data.continuation.raw_fields must be a mapping")
    discovery, raw_records = _discovery_samples(
        raw_manifest,
        fields=raw_fields,
        attempted=attempted,
        default_seed=int(cache.get("seed", 0)),
        require_edited=require_edited,
    )
    samples = [*accepted, *discovery]
    if len(samples) < target:
        raise RuntimeError(
            "continuation has only "
            f"{len(samples)} unique candidates for target {target}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.tmp-{uuid.uuid4().hex}")
    staging.mkdir()
    try:
        staged_samples = staging / samples_file.name
        staged_samples.write_text(
            "".join(json.dumps(sample, sort_keys=True) + "\n" for sample in samples),
            encoding="utf-8",
        )
        metadata = {
            "schema_version": 2,
            "mode": CONTINUATION_MODE,
            "replay_filter_policy": REPLAY_FILTER_POLICY,
            "model": expected_model,
            "recipe": str(cache["recipe"]),
            "reference_mode": reference_mode,
            "target_sources": target,
            "replay_sources": len(accepted),
            # Priority selections have no accepted/attempted samples under the
            # new model yet, even when their length already reaches the target.
            "remaining_acceptances": target
            if seed_selection
            else target - len(accepted),
            "attempted_source_ids": 0 if seed_selection else len(attempted),
            "discovery_sources": len(discovery),
            "candidates": len(samples),
            "raw_records": raw_records,
            "raw_manifest": str(raw_manifest),
            "raw_manifest_sha256": _sha256(raw_manifest),
            "partial_caches": provenance,
            "seed_selection": seed_provenance,
            "preparation_config": dict(continuation),
            "samples_file": staged_samples.name,
            "samples_sha256": _sha256(staged_samples),
        }
        (staging / metadata_file.name).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        try:
            os.replace(staging, output)
        except OSError:
            if not output.is_dir():
                raise
            shutil.rmtree(staging, ignore_errors=True)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_cache_continuation(
        metadata_file,
        samples_path=samples_file,
        expected_model=expected_model,
        expected_target=target,
        expected_reference_mode=reference_mode,
    )
