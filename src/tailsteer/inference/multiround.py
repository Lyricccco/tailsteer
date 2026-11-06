"""Task parsing for independent multi-round editing chains."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EditRound:
    prompt: str
    seed: int
    steps: int
    denoising_strength: float = 1.0
    noise_mode: str = "pure_noise"


@dataclass(frozen=True)
class EditTask:
    task_id: str
    image: str
    rounds: tuple[EditRound, ...]
    chain_rounds: bool = True


def load_tasks(
    path: str | Path,
    *,
    default_steps: int,
    default_seed: int,
) -> tuple[EditTask, ...]:
    value = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    raw_tasks = value.get("tasks") if isinstance(value, dict) else value
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("task JSON must contain a non-empty list")
    tasks = []
    for task_index, raw_task in enumerate(raw_tasks, 1):
        image = str(raw_task["image"])
        subfolder = str(raw_task.get("subfolder", "")).strip("/")
        if subfolder and not image.startswith(subfolder + "/"):
            image = str(Path(subfolder) / image)
        raw_rounds = (
            raw_task.get("rounds") or raw_task.get("edits") or raw_task.get("prompts")
        )
        if not isinstance(raw_rounds, list) or not raw_rounds:
            raise ValueError(f"task {task_index} has no edit rounds")
        rounds = []
        for round_index, raw_round in enumerate(raw_rounds):
            item: Mapping[str, Any] = (
                {"prompt": raw_round} if isinstance(raw_round, str) else raw_round
            )
            if item.get("condition_images"):
                raise ValueError("multi-image tasks require multi_condition config")
            strength = float(item.get("denoising_strength", 1.0))
            noise_mode = str(item.get("noise_mode", "pure_noise")).replace(
                "add_noise_to_current", "add_noise"
            )
            if not 0 < strength <= 1 or noise_mode not in {
                "pure_noise",
                "add_noise",
                "auto",
            }:
                raise ValueError("invalid round denoising strength or noise mode")
            rounds.append(
                EditRound(
                    prompt=str(item["prompt"]),
                    seed=int(item.get("seed", default_seed + round_index)),
                    steps=int(item.get("steps", default_steps)),
                    denoising_strength=strength,
                    noise_mode=noise_mode,
                )
            )
        task_id = str(raw_task.get("id") or f"{task_index:04d}-{Path(image).stem}")
        if "/" in task_id or task_id in {"", ".", ".."}:
            raise ValueError(f"unsafe task id {task_id!r}")
        tasks.append(
            EditTask(
                task_id,
                image,
                tuple(rounds),
                bool(raw_task.get("chain_rounds", True)),
            )
        )
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("task ids must be unique")
    return tuple(tasks)


def balanced_task_indices(
    tasks: tuple[EditTask, ...], world_size: int
) -> tuple[tuple[int, ...], ...]:
    """Assign whole edit chains while balancing their generation rounds."""

    if world_size < 1:
        raise ValueError("world_size must be positive")
    shards: list[list[int]] = [[] for _ in range(world_size)]
    loads = [0] * world_size
    ordered = sorted(
        enumerate(tasks),
        key=lambda item: (-len(item[1].rounds), item[0]),
    )
    for task_index, task in ordered:
        rank = min(range(world_size), key=lambda value: (loads[value], value))
        shards[rank].append(task_index)
        loads[rank] += len(task.rounds)
    return tuple(tuple(sorted(shard)) for shard in shards)
