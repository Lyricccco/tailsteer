"""Local, resumable trajectory caches for official image-edit models."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tailsteer.models.flux2.cache import (
    cache_mask_decision,
    fit_rgb,
    flux2_keep_mask_variants,
    largest_true_square,
)
from tailsteer.models.sensenova.identity import (
    SENSENOVA_MODEL_NAMES,
    resolve_sensenova_variant,
)
from tailsteer.utils import configure_logging, save_resolved_config

from .continuation import (
    CONTINUATION_MARKER,
    CONTINUATION_MODE,
    is_continuation_replay,
    validate_cache_continuation,
)
from .format import (
    STEP_RECORDS_LAYOUT,
    TRAJECTORY_GROUPED_LAYOUT,
    CacheExample,
    CacheWriter,
    TrajectoryCacheWriter,
    read_cache_records,
    validate_cache,
)
from .selection import SELECTION_MODE, validate_cache_selection
from .visualization import (
    build_visualization_contact_sheets,
    sample_visualization_complete,
    save_sample_visualization,
    validate_visualizations,
    write_visualization_manifest,
)

SUPPORTED_MODELS = {
    "flux2_klein_9b",
    "flux1_kontext_dev",
    "qwen_image_21",
    "sensenova_u1_5_8b_mot",
}


@dataclass(frozen=True)
class CaptureContract:
    num_steps: int
    first_forward: int
    forward_count: int
    final_point: int
    width: int
    height: int

    @property
    def last_forward(self) -> int:
        return self.first_forward + self.forward_count - 1


CAPTURE_CONTRACTS = {
    "flux1_kontext_dev": CaptureContract(30, 0, 30, 30, 1024, 1024),
    "flux2_klein_9b": CaptureContract(50, 0, 50, 50, 1024, 1024),
    "qwen_image_21": CaptureContract(40, 0, 40, 40, 1024, 1024),
    "sensenova_u1_5_8b_mot": CaptureContract(50, 1, 50, 51, 1024, 1024),
}


@dataclass(frozen=True)
class TrajectoryResult:
    image: Any
    source_state: Any
    forward_states: Mapping[int, Any]
    final_state: Any
    schedule_points: Mapping[int, tuple[float, float]]
    final_schedule_point: tuple[float, float]
    grid: tuple[int, int]
    extra: Mapping[str, Any]
    forward_velocities: Mapping[int, Any] | None = None
    parity_conditioning: Mapping[str, Any] | None = None


class _Distributed:
    def __init__(self) -> None:
        import torch
        import torch.distributed as dist

        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if not torch.cuda.is_available():
            raise RuntimeError("trajectory cache generation requires CUDA")
        torch.cuda.set_device(self.local_rank)
        self.owns_group = False
        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group("nccl")
            self.owns_group = True

    @property
    def device(self) -> Any:
        import torch

        return torch.device("cuda", self.local_rank)

    def barrier(self) -> None:
        if self.world_size > 1:
            import torch.distributed as dist

            dist.barrier(device_ids=[self.local_rank])

    def close(self) -> None:
        if self.owns_group:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
            self.owns_group = False


def _torch_dtype(name: str) -> Any:
    import torch

    normalized = str(name).removeprefix("torch.")
    if normalized not in {"float32", "float16", "bfloat16"}:
        raise ValueError(f"unsupported dtype {name!r}")
    return getattr(torch, normalized)


def _git_commit(repository: Path) -> str | None:
    """Return optional source provenance without making Git a runtime dependency."""

    repository = repository.expanduser().resolve()
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repository}",
                "rev-parse",
                "--verify",
                "HEAD",
            ],
            cwd=repository,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and revision else None


def _config_identity(config: Mapping[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _validate_completed_cache(
    output: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    manifest = validate_cache(output)
    recorded = manifest.get("creation_config")
    if not isinstance(recorded, Mapping):
        raise ValueError(f"completed cache has no creation_config: {output}")
    if _config_identity(recorded) != _config_identity(config):
        raise ValueError(
            "completed cache was built with another resolved config; "
            f"choose a new output_dir or use the original config: {output}"
        )
    return manifest


def validate_trajectory_config(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model")
    data = config.get("data")
    cache = config.get("cache")
    if not all(isinstance(value, dict) for value in (model, data, cache)):
        raise TypeError("trajectory config requires model, data, and cache mappings")
    name = str(model.get("name", ""))
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported trajectory model {name!r}")
    model_root = Path(model["path"]).expanduser().resolve()
    code_root = Path(model.get("code_path", ".")).expanduser().resolve()
    if not model_root.is_dir() or (
        name in {"flux1_kontext_dev", "sensenova_u1_5_8b_mot"}
        and not code_root.is_dir()
    ):
        raise FileNotFoundError(
            f"model/code path is missing: {model_root}, {code_root}"
        )
    recipe = str(cache.get("recipe", ""))
    if recipe not in {"local-edit-filtered", "noop-filtered", "annotated-mixed"}:
        raise ValueError("unsupported cache.recipe")
    completion_mode = str(cache.get("completion_mode", "target_accepted"))
    if completion_mode not in {"target_accepted", "candidates_exhausted"}:
        raise ValueError("unsupported cache.completion_mode")
    if cache.get("sample_error_policy", "fail") not in {"fail", "reject"}:
        raise ValueError("cache.sample_error_policy must be fail or reject")
    exhaustive = completion_mode == "candidates_exhausted"
    if recipe == "annotated-mixed" and not exhaustive:
        raise ValueError("annotated-mixed requires candidates_exhausted completion")
    filter_config = cache.get("filter", {})
    if not isinstance(filter_config, dict):
        raise TypeError("cache.filter must be a mapping")
    if filter_config.get("reference_mode", "paired") not in {"paired", "none"}:
        raise ValueError("cache.filter.reference_mode must be paired or none")
    if filter_config.get("component_policy", "largest") not in {"largest", "all"}:
        raise ValueError("cache.filter.component_policy must be largest or all")
    forward_points = tuple(int(value) for value in cache["forward_points"])
    if not forward_points or forward_points != tuple(sorted(set(forward_points))):
        raise ValueError(
            "cache.forward_points must be non-empty, unique, and increasing"
        )
    num_steps = int(cache["num_steps"])
    contract = CAPTURE_CONTRACTS[name]
    if num_steps != contract.num_steps:
        raise ValueError(
            f"{name} cache requires num_steps={contract.num_steps}, got {num_steps}"
        )
    if (
        min(forward_points) < contract.first_forward
        or max(forward_points) > contract.last_forward
    ):
        raise ValueError(
            f"{name} forward points must be inside "
            f"[{contract.first_forward}, {contract.last_forward}]"
        )
    final_point = int(cache["final_point"])
    if final_point != contract.final_point:
        raise ValueError(
            f"{name} final_point must be {contract.final_point}, got {final_point}"
        )
    width, height = int(cache["width"]), int(cache["height"])
    if (width, height) != (contract.width, contract.height):
        raise ValueError(
            f"{name} cache size must be {contract.width}x{contract.height}, "
            f"got {width}x{height}"
        )
    storage_layout = str(cache.get("storage_layout", STEP_RECORDS_LAYOUT))
    if storage_layout not in {STEP_RECORDS_LAYOUT, TRAJECTORY_GROUPED_LAYOUT}:
        raise ValueError(f"unsupported cache.storage_layout: {storage_layout!r}")
    variants = cache.get("mask_variants", [])
    if (
        not isinstance(variants, list)
        or any(
            not isinstance(value, str) or value not in {"fine", "dilate5"}
            for value in variants
        )
        or len(variants) != len(set(variants))
    ):
        raise ValueError("cache.mask_variants must be a unique list of fine/dilate5")
    if name == "qwen_image_21" and (
        float(model.get("true_cfg_scale", 1.0)) != 1.0
        or not isinstance(model.get("use_kv_cache", True), bool)
    ):
        raise ValueError(
            "Qwen cache requires true_cfg_scale=1 and boolean use_kv_cache"
        )
    if variants and int(filter_config.get("dilation_pixels", 48)) != 48:
        raise ValueError("multi-mask cache retains the default legacy 48px mask")
    if not exhaustive and int(cache["target_accepted"]) < 1:
        raise ValueError("cache.target_accepted must be positive")
    selection = data.get("selection")
    continuation = data.get("continuation")
    if selection is not None and continuation is not None:
        raise ValueError("data.selection and data.continuation are mutually exclusive")
    if exhaustive and (selection is not None or continuation is not None):
        raise ValueError(
            "candidates_exhausted consumes a fixed manifest, not replay/continuation"
        )
    if (
        selection is None
        and continuation is None
        and cache.get("candidate_end") == "auto"
    ):
        with Path(data["manifest"]).open(encoding="utf-8") as handle:
            cache["candidate_end"] = sum(bool(line.strip()) for line in handle)
    if exhaustive:
        # Pin input bytes in the resume identity; do not follow a growing export.
        digest = hashlib.sha256(Path(data["manifest"]).read_bytes()).hexdigest()
        if data.get("manifest_sha256", digest) != digest:
            raise ValueError("candidate manifest checksum changed")
        data["manifest_sha256"] = digest
        cache["target_accepted"] = int(cache["candidate_end"]) - int(
            cache.get("candidate_start", 0)
        )
    if selection is not None:
        if not isinstance(selection, dict):
            raise TypeError("data.selection must be a mapping")
        if selection.get("mode") != SELECTION_MODE:
            raise ValueError(
                f"unsupported data.selection mode: {selection.get('mode')!r}"
            )
        selection_metadata = validate_cache_selection(
            selection["metadata"],
            samples_path=data["manifest"],
        )
        selection_sources = int(selection_metadata["sources"])
        if int(cache["target_accepted"]) > selection_sources:
            raise ValueError(
                "cache target exceeds the accepted-source selection: "
                f"{cache['target_accepted']} > {selection_sources}"
            )
        if int(cache.get("candidate_end", 0)) > selection_sources:
            raise ValueError(
                "cache candidate_end exceeds the accepted-source selection: "
                f"{cache['candidate_end']} > {selection_sources}"
            )
    if continuation is not None:
        if not isinstance(continuation, dict):
            raise TypeError("data.continuation must be a mapping")
        if continuation.get("mode") != CONTINUATION_MODE:
            raise ValueError(
                f"unsupported data.continuation mode: {continuation.get('mode')!r}"
            )
        continuation_metadata = validate_cache_continuation(
            continuation["metadata"],
            samples_path=data["manifest"],
            expected_model=name,
            expected_reference_mode=str(filter_config.get("reference_mode", "paired")),
        )
        continuation_candidates = int(continuation_metadata["candidates"])
        continuation_target = int(continuation_metadata["target_sources"])
        if int(cache["target_accepted"]) > continuation_target:
            raise ValueError(
                "cache target exceeds the continuation snapshot target: "
                f"{cache['target_accepted']} > {continuation_target}"
            )
        if cache.get("candidate_end") == "auto":
            cache["candidate_end"] = continuation_candidates
        if int(cache["target_accepted"]) > continuation_candidates:
            raise ValueError(
                "cache target exceeds the continuation candidate pool: "
                f"{cache['target_accepted']} > {continuation_candidates}"
            )
        if int(cache.get("candidate_end", 0)) > continuation_candidates:
            raise ValueError(
                "cache candidate_end exceeds the continuation candidate pool: "
                f"{cache['candidate_end']} > {continuation_candidates}"
            )
    num_task_shards = int(cache.get("num_task_shards", 1))
    task_shard_id = int(cache.get("task_shard_id", 0))
    if num_task_shards < 1:
        raise ValueError("cache.num_task_shards must be positive")
    if exhaustive and num_task_shards != 1:
        raise ValueError(
            "candidates_exhausted uses one torchrun job, not independent task shards"
        )
    if not 0 <= task_shard_id < num_task_shards:
        raise ValueError("cache.task_shard_id must be in [0, cache.num_task_shards)")
    if int(cache.get("candidate_end", 0)) <= int(cache.get("candidate_start", 0)):
        raise ValueError("cache candidate range is empty")
    return config


def _load_records(
    path: str | Path, *, start: int, end: int
) -> list[tuple[int, dict[str, Any]]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    records: list[tuple[int, dict[str, Any]]] = []
    ordinal = 0
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if ordinal >= end:
                break
            if ordinal >= start:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(
                        f"dataset record must be an object: {source}:{line_number}"
                    )
                value["_line_number"] = line_number
                records.append((ordinal, value))
            ordinal += 1
    if not records:
        raise ValueError(f"dataset candidate range is empty: {source} [{start}, {end})")
    return records


def _field(record: Mapping[str, Any], name: str, alternatives: Sequence[str]) -> Any:
    for candidate in (name, *alternatives):
        value = record.get(candidate)
        if value not in {None, ""}:
            return value
    return None


def _local_path(value: Any, root: Path, *, anchor: str | None = None) -> Path:
    if value is None:
        raise ValueError("dataset record lacks a required image path")
    path = Path(str(value)).expanduser()
    result = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if result.is_file():
        return result
    # Pico-Banana metadata retains absolute paths from the machine that built
    # it. Rebind the suffix below a declared legacy directory name; root.name
    # remains the backwards-compatible default for snapshots with unchanged
    # directory names.
    anchors = [value for value in (anchor, root.name) if value]
    for path_anchor in dict.fromkeys(anchors):
        if Path(path_anchor).name != path_anchor or path_anchor in {".", ".."}:
            raise ValueError(f"invalid dataset path anchor: {path_anchor!r}")
        for root_index, part in enumerate(path.parts):
            if part != path_anchor:
                continue
            rebound = root.joinpath(*path.parts[root_index + 1 :]).resolve()
            if root not in rebound.parents or not rebound.is_file():
                continue
            return rebound
    raise FileNotFoundError(result)


def _source_id(record: Mapping[str, Any], source: Path) -> tuple[str, str]:
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


def _append_progress(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (json.dumps(dict(value), sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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


def _rank_status_path(building: Path, global_rank: int) -> Path:
    return building / f".trajectory-rank-{global_rank:05d}.json"


def _write_rank_status(
    building: Path, *, session_id: str, status: Mapping[str, Any]
) -> dict[str, Any]:
    value = {
        **dict(status),
        "schema_version": 1,
        "session_id": session_id,
        "updated_at_unix": time.time(),
    }
    _atomic_json(_rank_status_path(building, int(value["global_rank"])), value)
    return value


def _wait_for_rank_statuses(
    building: Path,
    *,
    session_id: str,
    global_ranks: Sequence[int],
    timeout_seconds: float,
    poll_seconds: float = 2.0,
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    if timeout_seconds <= 0:
        raise ValueError("rank status timeout must be positive")
    if poll_seconds <= 0:
        raise ValueError("rank status poll interval must be positive")
    expected = tuple(int(rank) for rank in global_ranks)
    deadline = time.monotonic() + timeout_seconds
    next_log = time.monotonic()
    while True:
        statuses: dict[int, dict[str, Any]] = {}
        for global_rank in expected:
            path = _rank_status_path(building, global_rank)
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                continue
            if value.get("session_id") != session_id:
                continue
            if int(value.get("global_rank", -1)) != global_rank:
                continue
            statuses[global_rank] = value
        missing = [rank for rank in expected if rank not in statuses]
        if not missing:
            return [statuses[rank] for rank in expected]
        now = time.monotonic()
        if now >= deadline:
            raise TimeoutError(
                "timed out waiting for trajectory-cache rank status files: "
                f"session={session_id}, missing={missing}"
            )
        if logger is not None and now >= next_log:
            logger.info(
                "waiting for cache ranks via shared storage: ready=%d/%d missing=%s",
                len(statuses),
                len(expected),
                missing,
            )
            next_log = now + 60.0
        time.sleep(min(poll_seconds, max(0.0, deadline - now)))


def _read_progress(path: Path) -> tuple[dict[str, Any], ...]:
    values = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if index + 1 == len(lines):
                break
            raise
        values.append(value)
    return tuple(values)


def _completed_sources(building: Path, *, retry_failed: bool = False) -> set[str]:
    complete = {str(value["source_id"]) for value in read_cache_records(building)}
    for journal in building.glob("progress-rank-*.jsonl"):
        complete.update(
            str(value["source_id"])
            for value in _read_progress(journal)
            if not retry_failed or value.get("outcome") in {"accepted", "rejected"}
        )
    return complete


def resolve_cache_recipe(cache: Mapping[str, Any], record: Mapping[str, Any]) -> str:
    recipe = str(cache["recipe"])
    if recipe != "annotated-mixed":
        return recipe
    task = record.get("task")
    if task not in {"local_edit", "noop"}:
        raise ValueError(f"annotated-mixed candidate has invalid task: {task!r}")
    return {"local_edit": "local-edit-filtered", "noop": "noop-filtered"}[task]


def _candidate_coverage(
    candidates, accepted_ids: set[str], completed_ids: set[str]
) -> dict:
    """Report input balance separately from post-filter acceptance balance."""
    from collections import Counter

    ids = [
        _source_id(row, Path(f"line-{row['_line_number']}"))[0] for _, row in candidates
    ]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate manifest contains duplicate trajectory IDs")
    missing = set(ids) - completed_ids
    if missing:
        raise RuntimeError(f"{len(missing)} candidates remain failed or unprocessed")
    groups = {}
    for category in sorted(
        {row.get("sampling_category", "unspecified") for _, row in candidates}
    ):
        rows = [
            (sid, row)
            for sid, (_, row) in zip(ids, candidates, strict=True)
            if row.get("sampling_category", "unspecified") == category
        ]
        groups[category] = {
            "source_images": len(
                {row.get("source_identity", sid) for sid, row in rows}
            ),
            "candidates": len(rows),
            "accepted": sum(sid in accepted_ids for sid, _ in rows),
            "rejected": sum(sid not in accepted_ids for sid, _ in rows),
            "tasks": dict(Counter(row.get("task", "unspecified") for _, row in rows)),
        }
    return {
        "candidates": len(ids),
        "accepted": len(set(ids) & accepted_ids),
        "rejected": len(set(ids) - accepted_ids),
        "failed": 0,
        "categories": groups,
    }


def _accepted_for_rank(building: Path, rank: int) -> int:
    return len(_accepted_sources_for_rank(building, rank))


def _accepted_sources_for_rank(building: Path, rank: int) -> set[str]:
    prefix = rank * 1_000_000
    upper = prefix + 1_000_000
    return {
        str(record["source_id"])
        for record in read_cache_records(building)
        if prefix <= int(str(record["shard"])[6:13]) < upper
    }


def _rank_quota(total: int, rank: int, world_size: int) -> int:
    return total // world_size + int(rank < total % world_size)


def _task_shard_layout(
    cache: Mapping[str, Any], *, rank: int, world_size: int
) -> tuple[int, int, int]:
    """Map one process to its stable rank in an independently submitted fanout."""

    num_task_shards = int(cache.get("num_task_shards", 1))
    task_shard_id = int(cache.get("task_shard_id", 0))
    global_world_size = num_task_shards * world_size
    global_rank = task_shard_id * world_size + rank
    return global_rank, global_world_size, task_shard_id


def _load_runtime(config: dict[str, Any], device: Any) -> Any:
    model = config["model"]
    name = model["name"]
    if name == "qwen_image_21":
        from tailsteer.models.qwen_image_21 import QwenImage21Runtime

        return QwenImage21Runtime.load(model, device=device)
    if name == "flux1_kontext_dev":
        from tailsteer.models.flux1_kontext import KontextRuntime

        return KontextRuntime.load(model, device=device)
    if name == "flux2_klein_9b":
        from tailsteer.models.flux2 import Flux2Model

        return Flux2Model.load(model, device=device, include_vae=True)
    from tailsteer.models.sensenova import load_sensenova_runtime

    return load_sensenova_runtime(
        model_root=model["path"],
        model_code_repo=model["code_path"],
        device=str(device),
        torch_dtype=_torch_dtype(model.get("dtype", "bfloat16")),
        variant=resolve_sensenova_variant(name, model.get("variant")),
    )


def _run_trajectory(
    runtime: Any,
    *,
    config: dict[str, Any],
    source: Any,
    prompt: str,
    seed: int,
) -> TrajectoryResult:
    cache = config["cache"]
    model = config["model"]
    name = str(model["name"])
    forwards = tuple(int(value) for value in cache["forward_points"])
    if name == "qwen_image_21":
        import numpy as np

        trace = runtime.edit(
            source_image=source,
            prompt=prompt,
            seed=seed,
            num_steps=int(cache["num_steps"]),
            capture_forward_steps=forwards,
            true_cfg_scale=float(model.get("true_cfg_scale", 1.0)),
            use_kv_cache=model.get("use_kv_cache", True),
        )
        alpha = np.asarray(trace.rgba_image.convert("RGBA"))[..., 3]
        return TrajectoryResult(
            image=trace.image,
            source_state=trace.source_state,
            forward_states=trace.forward_states,
            forward_velocities=trace.forward_velocities,
            final_state=trace.final_state,
            schedule_points=trace.schedule_points,
            final_schedule_point=trace.final_schedule_point,
            grid=tuple(trace.final_state.shape[-2:]),
            extra={
                "true_cfg_scale": float(model.get("true_cfg_scale", 1.0)),
                "use_kv_cache": model.get("use_kv_cache", True),
                "source_state_origin": "native_normalized_qwen21_rgba_vae",
                "conditioning": "source_image_qwen3vl_and_vae_reference",
                "timestep_convention": "scheduler_native_0_1000",
                "display_composite": "white",
                "output_alpha_min": int(alpha.min()),
                "output_transparent_fraction": float((alpha < 254).mean()),
            },
        )
    if name in {"flux2_klein_9b"}:
        import torch

        from tailsteer.models.flux2 import build_flux2_schedule

        width, height = int(cache["width"]), int(cache["height"])
        schedule = build_flux2_schedule(
            num_steps=int(cache["num_steps"]),
            width=width,
            height=height,
        )
        source_state = runtime.encode_image(source)
        conditioning = runtime.prepare_conditioning(
            source_latent=source_state,
            prompts=[prompt],
            width=width,
            height=height,
        )
        state = runtime.initialize_latent(
            source_state,
            seed=seed,
            sigma=schedule.point(0).sigma,
            noise_mode=str(cache.get("noise_mode", "pure_noise")),
        )
        captured = {}
        captured_velocities = {}
        with torch.no_grad():
            for step in range(len(schedule)):
                if step in forwards:
                    captured[step] = state[0].detach().cpu().contiguous()
                transition = schedule.transition(step)
                velocity = runtime.predict_velocity(
                    state,
                    transition.current,
                    conditioning,
                    actor="base",
                )
                if step in forwards:
                    captured_velocities[step] = velocity[0].detach().cpu().contiguous()
                state = runtime.advance(state, velocity, transition)
        points = {
            step: (
                float(schedule.point(step).sigma),
                float(schedule.point(step).timestep),
            )
            for step in forwards
        }
        return TrajectoryResult(
            image=runtime.decode_latent(state, height=height, width=width),
            source_state=source_state[0].detach().cpu().contiguous(),
            forward_states=captured,
            final_state=state[0].detach().cpu().contiguous(),
            schedule_points=points,
            final_schedule_point=(0.0, 0.0),
            grid=(int(state.shape[-2]), int(state.shape[-1])),
            extra={"noise_mode": str(cache.get("noise_mode", "pure_noise"))},
            forward_velocities=captured_velocities,
        )
    if name in SENSENOVA_MODEL_NAMES:
        from tailsteer.models.sensenova import run_sensenova_edit

        width, height = int(cache["width"]), int(cache["height"])
        trace = run_sensenova_edit(
            runtime,
            source_image=source,
            prompt=prompt,
            seed=seed,
            width=width,
            height=height,
            num_inference_steps=int(cache["num_steps"]),
            cfg_scale=float(model.get("cfg_scale", 4.0)),
            img_cfg_scale=float(model.get("img_cfg_scale", 1.0)),
            cfg_norm=str(model.get("cfg_norm", "none")),
            shift=float(model.get("timestep_shift", 3.0)),
            cfg_interval=tuple(
                float(value) for value in model.get("cfg_interval", (0.0, 1.0))
            ),
            capture_forward_steps=forwards,
        )
        points = {
            step: (
                1.0 - float(trace.timesteps[step - 1]),
                float(trace.timesteps[step - 1]),
            )
            for step in forwards
        }
        patch = int(runtime.model.patch_size) * int(
            1 / float(runtime.model.downsample_ratio)
        )
        return TrajectoryResult(
            image=trace.image,
            source_state=trace.source_state,
            forward_states=trace.forward_states,
            final_state=trace.final_state,
            schedule_points=points,
            final_schedule_point=(0.0, 1.0),
            grid=(height // patch, width // patch),
            extra={
                "cfg_scale": float(model.get("cfg_scale", 4.0)),
                "img_cfg_scale": float(model.get("img_cfg_scale", 1.0)),
                "timestep_shift": float(model.get("timestep_shift", 3.0)),
            },
        )
    if name == "flux1_kontext_dev":
        trace = runtime.edit(
            source_image=source,
            prompt=prompt,
            seed=seed,
            num_steps=int(cache["num_steps"]),
            guidance_scale=float(config["model"].get("guidance_scale", 2.5)),
            capture_forward_steps=forwards,
        )
        return TrajectoryResult(
            image=trace.image,
            source_state=trace.source_state,
            forward_states=trace.forward_states,
            final_state=trace.final_state,
            schedule_points=trace.schedule_points,
            final_schedule_point=trace.final_schedule_point,
            grid=tuple(trace.final_state.shape[-2:]),
            extra={"guidance_type": "distilled", "timestep_convention": "native_sigma"},
        )
    raise ValueError(f"Unsupported model: {name}")


def load_trajectory_runtime(config: dict[str, Any], device: Any) -> Any:
    """Load the same official runtime used by cache generation.

    This public boundary exists for parity validation; callers remain
    responsible for unloading the returned runtime.
    """

    validate_trajectory_config(config)
    return _load_runtime(config, device)


def capture_trajectory(
    runtime: Any,
    *,
    config: dict[str, Any],
    source: Any,
    prompt: str,
    seed: int,
) -> TrajectoryResult:
    """Capture one trajectory through the cache generator's exact forward path."""

    return _run_trajectory(
        runtime,
        config=config,
        source=source,
        prompt=prompt,
        seed=seed,
    )


def prepare_trajectory_examples(
    *,
    runtime: Any,
    record: dict[str, Any],
    config: dict[str, Any],
    image_root: Path,
    candidate_ordinal: int | None = None,
    visualization_root: Path | None = None,
) -> tuple[str, list[CacheExample], dict[str, Any]]:
    import numpy as np
    import torch

    data, cache = config["data"], config["cache"]
    fields = data.get("fields", {})
    source_root = Path(data.get("source_root", image_root)).expanduser().resolve()
    edited_root = Path(data.get("edited_root", image_root)).expanduser().resolve()
    source_path = _local_path(
        _field(record, str(fields.get("source", "source")), ("source_path", "image")),
        source_root,
        anchor=data.get("source_path_anchor"),
    )
    source_id, source_identity = _source_id(record, source_path)
    recipe = resolve_cache_recipe(cache, record)
    edited_value = _field(
        record,
        str(fields.get("edited", "edited")),
        ("edited_path", "reference"),
    )
    reference_mode = str(cache.get("filter", {}).get("reference_mode", "paired"))
    if recipe == "local-edit-filtered" and reference_mode == "none":
        # This mode does not consult a paired target, even if the manifest has one.
        edited_value = None
    if (
        recipe == "local-edit-filtered"
        and reference_mode == "paired"
        and edited_value is None
    ):
        raise ValueError(f"local-edit record lacks edited image: {source_id}")
    edited_path = (
        None
        if edited_value is None
        else _local_path(
            edited_value,
            edited_root,
            anchor=data.get("edited_path_anchor"),
        )
    )
    width, height = int(cache["width"]), int(cache["height"])
    source = fit_rgb(source_path, width, height)
    edited = fit_rgb(edited_path, width, height) if edited_path is not None else None
    prompt = (
        str(cache.get("noop_prompt", "Keep the image exactly unchanged."))
        if recipe == "noop-filtered"
        else str(
            _field(record, str(fields.get("prompt", "prompt")), ("instruction", "text"))
            or ""
        )
    )
    if not prompt:
        raise ValueError(f"local-edit record lacks prompt: {source_id}")
    seed = int(record.get(str(fields.get("seed", "seed")), cache.get("seed", 0)))
    result = _run_trajectory(
        runtime,
        config=config,
        source=source,
        prompt=prompt,
        seed=seed,
    )
    generated = result.image
    if isinstance(generated, (list, tuple)):
        if len(generated) != 1:
            raise RuntimeError("official model returned multiple generated images")
        generated = generated[0]
    decision = cache_mask_decision(
        recipe=recipe,
        source=source,
        generated=generated,
        edited=edited if recipe == "local-edit-filtered" else None,
        latent_height=result.grid[0],
        latent_width=result.grid[1],
        filter_config=dict(cache.get("filter", {})),
    )
    if not bool(cache.get("filter", {}).get("enabled", True)):
        decision["accepted"] = True
    metrics = dict(decision["metrics"])
    if not decision["accepted"]:
        return source_id, [], metrics
    keep = torch.from_numpy(
        np.asarray(decision["keep_mask"], dtype=np.uint8)
    ).unsqueeze(0)
    mask_variants = None
    if cache.get("mask_variants"):
        change_mask = (
            np.zeros((height, width), dtype=bool)
            if recipe == "noop-filtered"
            else decision["change_mask"]
        )
        if config["model"]["name"] in SENSENOVA_MODEL_NAMES:
            from tailsteer.models.sensenova.cache import sensenova_keep_mask_variants

            extra_masks = sensenova_keep_mask_variants(
                change_mask,
                state_shape=tuple(result.source_state.shape),
                grid=result.grid,
                variants=cache["mask_variants"],
            )
        elif config["model"]["name"] == "flux1_kontext_dev":
            from tailsteer.models.flux1_kontext.cache import keep_mask_variants

            extra_masks = keep_mask_variants(
                change_mask,
                state_shape=tuple(result.source_state.shape),
                variants=cache["mask_variants"],
            )
        elif config["model"]["name"] == "qwen_image_21":
            from tailsteer.models.qwen_image_21.official import keep_mask_variants

            extra_masks = keep_mask_variants(
                change_mask,
                state_shape=tuple(result.source_state.shape),
                variants=cache["mask_variants"],
            )
        else:
            extra_masks = flux2_keep_mask_variants(
                change_mask,
                latent_shape=tuple(result.source_state.shape),
                variants=cache["mask_variants"],
            )
        mask_variants = {
            name: torch.from_numpy(np.ascontiguousarray(value))
            for name, value in extra_masks.items()
        }
    keep_patch = (
        largest_true_square(
            np.asarray(decision["keep_mask"], dtype=bool),
            multiple=int(cache.get("filter", {}).get("patch_multiple", 8)),
        )
        if recipe == "local-edit-filtered"
        else None
    )
    base = {
        "source_id": source_id,
        "source_identity": source_identity,
        "source_path": str(source_path),
        "edited_path": None if edited_path is None else str(edited_path),
        "prompt": prompt,
        "seed": seed,
        "height": height,
        "width": width,
        "task": "local_edit" if recipe == "local-edit-filtered" else "noop",
        "mask_key": "m_keep_latent",
        "state_representation": str(
            config["model"].get("state_representation", "native")
        ),
        "filter_metrics": metrics,
        **dict(result.extra),
    }
    # Optional annotation provenance; old manifests and model adapters are unchanged.
    base.update(
        {
            key: record[key]
            for key in (
                "sampling_category",
                "source_ordinal",
                "category",
                "origin",
                "source_group",
                "prompt_id",
                "edit_type",
                "target",
            )
            if key in record
        }
    )
    if keep_patch is not None:
        base.update(
            {
                "patch_top": int(keep_patch[0]),
                "patch_left": int(keep_patch[1]),
                "patch_side": int(keep_patch[2]),
            }
        )
    if candidate_ordinal is not None:
        base["candidate_ordinal"] = int(candidate_ordinal)
    continuation = record.get(CONTINUATION_MARKER)
    if continuation is not None:
        if not isinstance(continuation, Mapping):
            raise TypeError(f"invalid continuation marker for source {source_id}")
        base["continuation_origin"] = str(continuation.get("origin", "unknown"))
        base["continuation_replay"] = is_continuation_replay(continuation)
        if continuation.get("raw_candidate_ordinal") is not None:
            base["raw_candidate_ordinal"] = int(continuation["raw_candidate_ordinal"])
    examples = []
    for point in cache["forward_points"]:
        point = int(point)
        sigma, timestep = result.schedule_points[point]
        examples.append(
            CacheExample(
                anchor_latent=result.forward_states[point].detach().cpu(),
                source_latent=result.source_state.detach().cpu(),
                keep_mask=keep,
                mask_variants=mask_variants,
                metadata={
                    **base,
                    "sample_id": f"{source_id}:forward-{point}",
                    "step_index": point,
                    "state_kind": "forward_input",
                    "sigma": float(sigma),
                    "timestep": float(timestep),
                },
            )
        )
    final_point = int(cache["final_point"])
    sigma, timestep = result.final_schedule_point
    examples.append(
        CacheExample(
            anchor_latent=result.final_state.detach().cpu(),
            source_latent=result.source_state.detach().cpu(),
            keep_mask=keep,
            mask_variants=mask_variants,
            metadata={
                **base,
                "sample_id": f"{source_id}:final-{final_point}",
                "step_index": final_point,
                "state_kind": "final",
                "sigma": float(sigma),
                "timestep": float(timestep),
            },
        )
    )
    if visualization_root is not None:
        if candidate_ordinal is None:
            raise ValueError("visualization output requires candidate_ordinal")
        states = {
            "z_src": result.source_state,
            **{
                f"z_forward_{point:03d}": result.forward_states[point]
                for point in map(int, cache["forward_points"])
            },
            "z_final": result.final_state,
        }
        save_sample_visualization(
            root=visualization_root,
            ordinal=int(candidate_ordinal),
            source_id=source_id,
            source=source,
            reference=edited if recipe == "local-edit-filtered" else None,
            generated=generated,
            keep_mask=np.asarray(decision["keep_mask"], dtype=bool),
            states=states,
            grid=result.grid,
            prompt=prompt,
            metrics=metrics,
            change_threshold=float(
                cache.get("filter", {}).get("generated_change_threshold", 0.22)
            ),
            keep_patch=keep_patch,
        )
    return source_id, examples, metrics


def build_trajectory_cache(config: dict[str, Any]) -> dict[str, Any]:
    validate_trajectory_config(config)
    runtime = _Distributed()
    cache = config["cache"]
    exhaustive = (
        cache.get("completion_mode", "target_accepted") == "candidates_exhausted"
    )
    output = Path(cache["output_dir"]).expanduser().resolve()
    repository = Path(__file__).resolve().parents[3]
    if output == repository or repository in output.parents:
        raise ValueError("generated caches must live outside the TailSteer repository")
    building = output.with_name(output.name + ".building")
    logger = configure_logging(rank=runtime.rank, output_dir=building)
    model_runtime = None
    try:
        if (output / "COMPLETE").is_file():
            return _validate_completed_cache(output, config)
        if output.exists():
            raise RuntimeError(f"refusing an unmarked cache directory: {output}")
        building.mkdir(parents=True, exist_ok=True)
        session_path = building / ".trajectory-rendezvous.json"
        if runtime.rank == 0:
            identity = _config_identity(config)
            identity_path = building / "build-config.json"
            if identity_path.exists() and identity_path.read_text() != identity:
                raise ValueError("existing cache staging directory has another config")
            identity_path.write_text(identity, encoding="utf-8")
            save_resolved_config(config, building)
            _atomic_json(
                session_path,
                {
                    "schema_version": 1,
                    "session_id": uuid.uuid4().hex,
                    "world_size": runtime.world_size,
                    "created_at_unix": time.time(),
                },
            )
        runtime.barrier()
        session = json.loads(session_path.read_text(encoding="utf-8"))
        if int(session["world_size"]) != runtime.world_size:
            raise RuntimeError("trajectory rendezvous world size does not match launch")
        session_id = str(session["session_id"])
        # Cache workers are data-parallel and perform no model collectives. Tear
        # down NCCL immediately after the short startup rendezvous so an early
        # worker cannot trigger the process-group watchdog while peers process
        # a longer candidate tail.
        runtime.close()
        data = config["data"]
        start = int(cache.get("candidate_start", 0))
        end = int(cache["candidate_end"])
        candidates = _load_records(data["manifest"], start=start, end=end)
        if exhaustive:
            if len(candidates) != end - start:
                raise ValueError(
                    "candidate range extends beyond the immutable manifest"
                )
            candidate_ids = set()
            for _ordinal, record in candidates:
                source_id = _source_id(record, Path(f"line-{record['_line_number']}"))[
                    0
                ]
                if source_id in candidate_ids:
                    raise ValueError(f"duplicate candidate ID: {source_id}")
                candidate_ids.add(source_id)
                resolve_cache_recipe(cache, record)
        image_root = (
            Path(data.get("image_root", Path(data["manifest"]).parent))
            .expanduser()
            .resolve()
        )
        target = int(cache["target_accepted"])
        global_rank, global_world_size, task_shard_id = _task_shard_layout(
            cache, rank=runtime.rank, world_size=runtime.world_size
        )
        quota = _rank_quota(target, global_rank, global_world_size)
        task_target = sum(
            _rank_quota(
                target,
                task_shard_id * runtime.world_size + local_rank,
                global_world_size,
            )
            for local_rank in range(runtime.world_size)
        )
        accepted_sources = _accepted_sources_for_rank(building, global_rank)
        accepted = len(accepted_sources)
        completed = _completed_sources(building, retry_failed=exhaustive)
        progress = building / f"progress-rank-{global_rank:05d}.jsonl"
        storage_layout = str(cache.get("storage_layout", STEP_RECORDS_LAYOUT))
        writer_type = (
            TrajectoryCacheWriter
            if storage_layout == TRAJECTORY_GROUPED_LAYOUT
            else CacheWriter
        )
        writer = writer_type(building, rank=global_rank)
        attempted = rejected = failed = 0
        missing_visual_sources = {
            _source_id(record, Path(f"line-{record['_line_number']}"))[0]
            for ordinal, record in candidates
            if ordinal % global_world_size == global_rank
            and _source_id(record, Path(f"line-{record['_line_number']}"))[0]
            in accepted_sources
            and not sample_visualization_complete(
                building,
                ordinal,
                _source_id(record, Path(f"line-{record['_line_number']}"))[0],
            )
        }
        pending_local = any(
            ordinal % global_world_size == global_rank
            and _source_id(record, Path(f"line-{record['_line_number']}"))[0]
            not in completed
            for ordinal, record in candidates
        )
        if (
            pending_local if exhaustive else accepted < quota
        ) or missing_visual_sources:
            logger.info("loading %s from local weights", config["model"]["name"])
            model_runtime = _load_runtime(config, runtime.device)
            logger.info("model ready")
        for ordinal, record in candidates:
            if not exhaustive and accepted >= quota and not missing_visual_sources:
                break
            if ordinal % global_world_size != global_rank:
                continue
            source_hint = _source_id(record, Path(f"line-{record['_line_number']}"))[0]
            continuation = record.get(CONTINUATION_MARKER, {})
            if continuation and not isinstance(continuation, Mapping):
                raise TypeError(f"invalid continuation marker for source {source_hint}")
            replay_source = is_continuation_replay(continuation)
            progress_context = {
                "candidate_ordinal": ordinal,
                "continuation_origin": continuation.get("origin"),
                "continuation_replay": replay_source,
                "raw_candidate_ordinal": continuation.get("raw_candidate_ordinal"),
            }
            needs_visual_recovery = source_hint in missing_visual_sources
            if source_hint in completed and not needs_visual_recovery:
                continue
            attempted += 1
            try:
                source_id, examples, metrics = prepare_trajectory_examples(
                    runtime=model_runtime,
                    record=record,
                    config=config,
                    image_root=image_root,
                    candidate_ordinal=ordinal,
                    visualization_root=building,
                )
                if not examples:
                    if needs_visual_recovery:
                        raise RuntimeError(
                            "previously accepted source now fails the cache filter: "
                            f"{source_id}"
                        )
                    if replay_source:
                        logger.warning(
                            "continuation replay source rejected by the current "
                            "filter; replacing it from the discovery pool: %s",
                            source_id,
                        )
                    rejected += 1
                    _append_progress(
                        progress,
                        {
                            "source_id": source_id,
                            "outcome": "rejected",
                            "metrics": metrics,
                            **progress_context,
                        },
                    )
                    continue
                if needs_visual_recovery:
                    if source_id != source_hint:
                        raise RuntimeError(
                            "visualization recovery resolved another source: "
                            f"{source_hint} != {source_id}"
                        )
                    missing_visual_sources.remove(source_hint)
                    logger.info(
                        "recovered visualization source=%s remaining=%d",
                        source_id,
                        len(missing_visual_sources),
                    )
                    continue
                writer.write(examples)
                accepted += 1
                accepted_sources.add(source_id)
                _append_progress(
                    progress,
                    {
                        "source_id": source_id,
                        "outcome": "accepted",
                        "records": len(examples),
                        "metrics": metrics,
                        **progress_context,
                    },
                )
                logger.info(
                    "accepted %d/%d source=%s attempted=%d",
                    accepted,
                    quota,
                    source_id,
                    attempted,
                )
            except Exception as exc:
                reject_error = cache.get("sample_error_policy", "fail") == "reject"
                if reject_error:
                    rejected += 1
                else:
                    failed += 1
                _append_progress(
                    progress,
                    {
                        "source_id": source_hint,
                        "outcome": "rejected" if reject_error else "failed",
                        "reason": "sample_error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        **progress_context,
                    },
                )
                logger.exception("cache source failed: %s", source_hint)
                if not reject_error and failed >= int(
                    cache.get("max_sample_failures", 32)
                ):
                    raise RuntimeError(
                        "too many sample failures on one rank: "
                        f"rank={runtime.rank}, failures={failed}"
                    ) from exc
        local_error = None if accepted >= quota else f"accepted {accepted}/{quota}"
        if exhaustive:
            terminal = _completed_sources(building, retry_failed=True)
            unresolved = [
                _source_id(record, Path(f"line-{record['_line_number']}"))[0]
                for ordinal, record in candidates
                if ordinal % global_world_size == global_rank
                and _source_id(record, Path(f"line-{record['_line_number']}"))[0]
                not in terminal
            ]
            local_error = (
                f"unresolved candidates={len(unresolved)}, "
                f"missing visuals={len(missing_visual_sources)}"
                if unresolved or missing_visual_sources
                else None
            )
        local_status = {
            "rank": runtime.rank,
            "global_rank": global_rank,
            "accepted": accepted,
            "quota": quota,
            "attempted": attempted,
            "rejected": rejected,
            "failed": failed,
            "error": local_error,
        }
        local_status = _write_rank_status(
            building, session_id=session_id, status=local_status
        )
        if local_status["error"]:
            raise RuntimeError(
                f"trajectory cache did not reach local rank quota: {local_status}"
            )
        if runtime.rank != 0:
            return {
                "worker_completed": True,
                "worker_rank": runtime.rank,
                "publication_delegated_to_rank": 0,
            }
        first_global_rank = task_shard_id * runtime.world_size
        statuses = _wait_for_rank_statuses(
            building,
            session_id=session_id,
            global_ranks=range(
                first_global_rank, first_global_rank + runtime.world_size
            ),
            timeout_seconds=float(
                os.environ.get("TAILSTEER_CACHE_RENDEZVOUS_TIMEOUT_SECONDS", "86400")
            ),
            logger=logger,
        )
        errors = [value for value in statuses if value["error"]]
        if errors:
            raise RuntimeError(f"trajectory cache did not reach rank quotas: {errors}")
        cache_records = read_cache_records(building)
        sources = {str(value["source_id"]) for value in cache_records}
        expected_points = [
            *map(int, cache["forward_points"]),
            int(cache["final_point"]),
        ]
        if exhaustive:
            task_target = sum(int(status["accepted"]) for status in statuses)
        if len(sources) != task_target:
            raise RuntimeError(
                f"cache part source count {len(sources)} != target {task_target}"
            )
        logger.info(
            "building visualization contact sheets for %d sources", len(sources)
        )
        visualization_summary = build_visualization_contact_sheets(building)
        validate_visualizations(
            building, visualization_summary, expected_samples=len(sources)
        )
        write_visualization_manifest(building, visualization_summary)
        manifest = {
            "schema_version": (2 if storage_layout == TRAJECTORY_GROUPED_LAYOUT else 1),
            "storage_layout": storage_layout,
            "model": str(config["model"]["name"]),
            "model_path": str(config["model"]["path"]),
            "model_code_path": str(config["model"]["code_path"]),
            "model_code_revision": _git_commit(Path(config["model"]["code_path"])),
            "source_dataset": str(data["manifest"]),
            "state_points": expected_points,
            "representation": str(
                config["model"].get("state_representation", "native")
            ),
            "sources": len(sources),
            "records": len(cache_records),
            "tailsteer_commit": _git_commit(repository),
            "world_size": runtime.world_size,
            "global_world_size": global_world_size,
            "task_shard_id": task_shard_id,
            "num_task_shards": int(cache.get("num_task_shards", 1)),
            "rank_status": statuses,
            "visualizations": visualization_summary,
            "creation_config": config,
        }
        if exhaustive:
            manifest["candidate_coverage"] = _candidate_coverage(
                candidates, sources, _completed_sources(building, retry_failed=True)
            )
        (building / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        (building / "COMPLETE").write_text("complete\n", encoding="utf-8")
        session_path.unlink(missing_ok=True)
        for status in statuses:
            _rank_status_path(building, int(status["global_rank"])).unlink(
                missing_ok=True
            )
        validate_cache(building)
        os.replace(building, output)
        logger.info("published complete cache: %s", output)
        return _validate_completed_cache(output, config)
    finally:
        try:
            if model_runtime is not None and hasattr(model_runtime, "unload"):
                model_runtime.unload()
        finally:
            runtime.close()
