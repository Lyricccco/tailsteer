"""Prompt-only comparison using native model generation and shared actors.

No source-image preprocessing, prompt rewriting, or editing preservation metrics.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from tailsteer.models import create_model
from tailsteer.models.sensenova.inference import SenseNovaPolicyInference
from tailsteer.utils import save_resolved_config

from .panels import output_comparison_panel
from .policies import build_policies


@dataclass(frozen=True)
class T2ITask:
    task_id: str
    prompt: str
    width: int
    height: int
    seed: int
    record: dict[str, Any]


def load_t2i_tasks(path: str | Path, *, default_seed: int = 42) -> list[T2ITask]:
    """Read ID-keyed JSON, a JSON list, or JSONL, regardless of file suffix."""

    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict) and isinstance(data.get("prompt"), str):
        rows = [(None, data)]  # A JSONL file containing just one record.
    elif isinstance(data, dict):
        rows = list(data.items())
    elif isinstance(data, list):
        rows = [(None, row) for row in data]
    else:
        raise TypeError("T2I tasks must be an ID-keyed mapping or a list")
    tasks: list[T2ITask] = []
    seen: set[str] = set()
    for key, row in rows:
        if not isinstance(row, dict):
            raise TypeError("each T2I record must be a mapping")
        task_id = str(row.get("id", key if key is not None else ""))
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", task_id):
            raise ValueError(f"invalid T2I task ID {task_id!r}")
        if key is not None and str(key) != task_id:
            raise ValueError(f"T2I mapping key {key!r} differs from ID {task_id!r}")
        if task_id in seen:
            raise ValueError(f"duplicate T2I task ID {task_id!r}")
        seen.add(task_id)
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"T2I task {task_id} has no prompt")
        if row.get("category", "t2i") != "t2i":
            raise ValueError(f"T2I task {task_id} is not category=t2i")
        if "width" in row or "height" in row:
            width, height = int(row["width"]), int(row["height"])
        else:
            resolution = int(row["resolution"])
            rw, rh = (int(v) for v in str(row["ratio"]).split(":"))
            if min(resolution, rw, rh) <= 0:
                raise ValueError(f"invalid T2I resolution/ratio for {task_id}")
            # Match this benchmark's existing area-budget convention, not a
            # long-edge resize. Never cap 4K tasks or crop to a square silently.
            width = round(resolution * math.sqrt(rw / rh) / 32) * 32
            height = round(resolution * math.sqrt(rh / rw) / 32) * 32
        if min(width, height) <= 0 or width % 32 or height % 32:
            raise ValueError(
                f"T2I dimensions must be positive multiples of 32: {task_id}"
            )
        seed = int(row.get("seed", default_seed))
        if not 0 <= seed < 2**63:
            raise ValueError(f"T2I seed must be in [0, 2**63): {task_id}")
        tasks.append(T2ITask(task_id, prompt, width, height, seed, dict(row)))
    if not tasks:
        raise ValueError("T2I task set must not be empty")
    return tasks


def balanced_t2i_indices(tasks: list[T2ITask], world_size: int) -> list[list[int]]:
    """Assign whole comparisons greedily by pixel budget; stable across resume."""

    if world_size < 1:
        raise ValueError("world_size must be positive")
    shards: list[list[int]] = [[] for _ in range(world_size)]
    loads = [0] * world_size
    for index in sorted(
        range(len(tasks)), key=lambda i: -(tasks[i].width * tasks[i].height)
    ):
        rank = min(range(world_size), key=lambda r: loads[r])
        shards[rank].append(index)
        loads[rank] += tasks[index].width * tasks[index].height
    return [sorted(shard) for shard in shards]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _commit_file(temporary, path)


def _commit_file(temporary: Path, path: Path) -> None:
    """Verify the published bytes before marking work complete on shared AFS.

    A successful rename alone is insufficient: interrupted/lagging shared-FS
    writes have left valid temporary PNGs but no destination on other workers.
    Retry only this file publication, not the expensive model generation.
    """
    expected = temporary.read_bytes()
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    for attempt in range(4):
        if not temporary.exists():
            with temporary.open("wb") as handle:
                handle.write(expected)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            if path.read_bytes() == expected:
                return
        except FileNotFoundError:
            pass
        time.sleep(0.25 * (attempt + 1))
    raise OSError(f"T2I file publication did not persist: {path}")


def _task_complete(directory: Path, policies: dict[str, Any]) -> bool:
    files_present = (
        (directory / "COMPLETE").is_file()
        and (directory / "task.json").is_file()
        and (directory / "comparison.jpg").is_file()
        and (directory / "prompt.txt").is_file()
        and all((directory / f"{name}.png").is_file() for name in policies)
    )
    if not files_present:
        return False
    try:
        results = json.loads((directory / "task.json").read_text())["results"]
        names = [item["policy"] for item in results]
        return len(names) == len(policies) and set(names) == set(policies)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _run_task(
    task: T2ITask,
    *,
    engine: Any,
    inference: dict[str, Any],
    policies: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    from PIL import Image

    directory = output / "tasks" / task.task_id
    directory.mkdir(parents=True, exist_ok=True)
    task_input = {
        "record": task.record,
        "seed": task.seed,
        "width": task.width,
        "height": task.height,
    }
    manifest = {"id": task.task_id, "input": task_input, "results": []}
    manifest_path = directory / "task.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["input"] != task_input:
            raise ValueError(f"resume T2I input changed: {task.task_id}")
        if _task_complete(directory, policies):
            return manifest
    (directory / "prompt.txt").write_text(task.prompt, encoding="utf-8")
    records = {item["policy"]: item for item in manifest["results"]}
    previews = {}
    preview_side = int(inference.get("preview_max_side", 512))
    for name, policy in policies.items():
        image_path = directory / f"{name}.png"
        if name not in records or not image_path.is_file():
            print(
                f"[t2i] id={task.task_id} policy={name} "
                f"size={task.width}x{task.height} seed={task.seed}",
                flush=True,
            )
            start = time.monotonic()
            image, trace = engine.generate_text(
                prompt=task.prompt,
                seed=task.seed,
                policy=policy,
                config={**inference, "width": task.width, "height": task.height},
            )
            expected = tuple(policy.actor_at(i) for i in range(policy.num_steps))
            if tuple(trace) != expected or image.size != (task.width, task.height):
                raise RuntimeError(
                    f"T2I actor trace/size mismatch: {task.task_id}/{name}"
                )
            temporary = image_path.with_name(f".{name}.tmp-{os.getpid()}.png")
            image.save(temporary)
            _commit_file(temporary, image_path)
            image.close()
            records[name] = {
                "policy": name,
                "image": str(image_path.relative_to(output)),
                "actor_trace": list(trace),
                "conditioning_actor": engine.conditioning_actor_for(
                    policy, num_steps=policy.num_steps
                ),
                "elapsed_seconds": time.monotonic() - start,
            }
            # Commit after each policy, so a 4K failure does not discard earlier
            # completed images of this prompt on resume.
            manifest["results"] = [records[n] for n in policies if n in records]
            _write_json(manifest_path, manifest)
        with Image.open(image_path) as image:
            preview = image.convert("RGB")
        preview.thumbnail((preview_side, preview_side), Image.Resampling.LANCZOS)
        previews[name] = preview
    panel = output_comparison_panel(
        previews, list(policies), prompt=f"ID {task.task_id}; full text: prompt.txt"
    )
    panel.save(directory / "comparison.jpg", quality=95)
    manifest["comparison"] = str((directory / "comparison.jpg").relative_to(output))
    _write_json(manifest_path, manifest)
    (directory / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return manifest


def run_t2i_inference(
    config: dict[str, Any], *, resume: bool = False
) -> dict[str, Any]:
    import torch
    import torch.distributed as dist
    import yaml

    inference = {**config["inference"], "schedule": dict(config["schedule"])}
    digest = hashlib.sha256(Path(inference["tasks"]).read_bytes()).hexdigest()
    if digest != inference["tasks_sha256"]:
        raise ValueError(
            "T2I task file changed after plan creation; prepare a new plan"
        )
    tasks = load_t2i_tasks(inference["tasks"], default_seed=int(config["seed"]))
    policies = build_policies(inference, num_steps=int(inference["num_steps"]))
    if any(not re.fullmatch(r"[a-zA-Z0-9_-]+", name) for name in policies):
        raise ValueError("invalid T2I policy filename")
    if int(inference.get("preview_max_side", 512)) < 1:
        raise ValueError("preview_max_side must be positive")
    output = Path(config["run"]["output_dir"]).expanduser().resolve()
    repository = Path(__file__).resolve().parents[3]
    if output == repository or repository in output.parents:
        raise ValueError("inference results must live outside the TailSteer repository")
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("invalid torchrun coordinates")
    shards = balanced_t2i_indices(tasks, world_size)
    owned_group = world_size > 1 and not dist.is_initialized()
    if owned_group:
        # Independent 4K comparisons can take much longer than small 2K ones.
        dist.init_process_group(backend="gloo", timeout=timedelta(hours=24))
    try:
        error = [None]
        if rank == 0:
            try:
                if output.exists() and any(output.iterdir()) and not resume:
                    raise FileExistsError(
                        f"inference output exists; pass --resume: {output}"
                    )
                output.mkdir(parents=True, exist_ok=True)
                saved = output / "config.yaml"
                if saved.is_file():
                    if yaml.safe_load(saved.read_text(encoding="utf-8")) != config:
                        raise ValueError(
                            "resume config differs from saved inference config"
                        )
                else:
                    if resume and any(output.iterdir()):
                        raise ValueError(
                            "cannot resume nonempty output without config.yaml"
                        )
                    save_resolved_config(config, output)
            except Exception as exc:
                if world_size == 1:
                    raise
                error[0] = f"{type(exc).__name__}: {exc}"
        if world_size > 1:
            dist.broadcast_object_list(error, src=0)
            if error[0] is not None:
                raise RuntimeError(error[0])
        pending = [
            tasks[i]
            for i in shards[rank]
            if not _task_complete(output / "tasks" / tasks[i].task_id, policies)
        ]
        if pending:
            torch.cuda.set_device(local_rank)
            flux2 = config["model"]["name"] in {"flux2_klein_9b"}
            device = torch.device("cuda", local_rank)
            if config["model"]["name"] == "qwen_image_21":
                from tailsteer.models.qwen_image_21.inference import (
                    QwenImage21InferenceRuntime,
                )

                engine = QwenImage21InferenceRuntime.load(
                    config["model"], inference.get("export_path"), device=device
                )
            elif config["model"]["name"] == "flux1_kontext_dev":
                from tailsteer.models.flux1_kontext.inference import (
                    KontextInferenceRuntime,
                )

                engine = KontextInferenceRuntime.load(
                    config["model"], inference.get("export_path"), device=device
                )
            else:
                model = create_model(
                    config["model"],
                    device=device,
                    include_vae=flux2,
                    **(
                        {
                            "independent_actors": True,
                            "export_path": inference.get("export_path"),
                        }
                        if flux2
                        else {}
                    ),
                )
                if flux2:
                    from tailsteer.models.flux2.inference import Flux2TextInference

                    engine = Flux2TextInference(model)
                else:
                    engine = SenseNovaPolicyInference(
                        model, inference.get("export_path")
                    )
            for task in pending:
                _run_task(
                    task,
                    engine=engine,
                    inference=inference,
                    policies=policies,
                    output=output,
                )
        _write_json(
            output / f"rank-{rank:02d}.json",
            {
                "rank": rank,
                "world_size": world_size,
                "task_ids": [tasks[i].task_id for i in shards[rank]],
            },
        )
        if world_size > 1:
            dist.barrier()
        if rank == 0:
            missing = [
                task.task_id
                for task in tasks
                if not _task_complete(output / "tasks" / task.task_id, policies)
            ]
            if missing:
                raise RuntimeError(
                    f"T2I outputs missing at final validation: {missing}; resume to repair"
                )
            manifest = {
                "schema_version": 1,
                "product": (
                    "qwen_image_21_t2i_comparison"
                    if config["model"]["name"] == "qwen_image_21"
                    else "kontext_t2i_comparison"
                    if config["model"]["name"] == "flux1_kontext_dev"
                    else "flux2_t2i_comparison"
                    if config["model"]["name"].startswith("flux2")
                    else "sensenova_t2i_comparison"
                ),
                "policy_order": list(policies),
                "policies": {
                    name: policy.to_dict() for name, policy in policies.items()
                },
                "num_steps": inference["num_steps"],
                "schedule": inference["schedule"],
                "export_path": inference.get("export_path"),
                "evaluation": config.get("evaluation", {}),
                "tasks_sha256": digest,
                "resolution_policy": "area_budget_align32_no_cap",
                "prompt_rewriting": False,
                "distributed": {
                    "world_size": world_size,
                    "strategy": "whole_comparison_balanced_by_pixels",
                },
                "tasks": [
                    json.loads(
                        (output / "tasks" / task.task_id / "task.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    for task in tasks
                ],
            }
            _write_json(output / "manifest.json", manifest)
            (output / "COMPLETE").write_text("complete\n", encoding="utf-8")
        if world_size > 1:
            dist.barrier()
        return json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    finally:
        if owned_group:
            dist.destroy_process_group()
