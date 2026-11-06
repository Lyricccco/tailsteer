"""Task parsing for isolated chained multi-image inference."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _mapping_tuple(value: Any, *, field: str) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(
        isinstance(item, Mapping) for item in value
    ):
        raise ValueError(f"{field} must be a list of mappings")
    return tuple(dict(item) for item in value)


def _string_tuple(value: Any, *, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a list of strings")
    return tuple(value)


@dataclass(frozen=True)
class MultiImageEditRound:
    prompt: str
    condition_images: tuple[str, ...]
    primary_source: str
    seed: int
    steps: int
    denoising_strength: float = 1.0
    noise_mode: str = "pure_noise"
    reference_roles: tuple[Mapping[str, Any], ...] = ()
    constraints_hard: tuple[str, ...] = ()
    constraints_soft: tuple[str, ...] = ()
    checkpoints: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "condition_images": list(self.condition_images),
            "primary_source": self.primary_source,
            "seed": self.seed,
            "steps": self.steps,
            "denoising_strength": self.denoising_strength,
            "noise_mode": self.noise_mode,
            "reference_roles": [dict(item) for item in self.reference_roles],
            "constraints_hard": list(self.constraints_hard),
            "constraints_soft": list(self.constraints_soft),
            "checkpoints": [dict(item) for item in self.checkpoints],
        }


@dataclass(frozen=True)
class MultiImageEditTask:
    task_id: str
    task_type: str
    image: str
    rounds: tuple[MultiImageEditRound, ...]
    goal: str = ""
    chain_rounds: bool = True
    story_subtype: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "id": self.task_id,
            "task_type": self.task_type,
            "image": self.image,
            "goal": self.goal,
            "chain_rounds": self.chain_rounds,
            "rounds": [item.to_dict() for item in self.rounds],
        }
        if self.story_subtype is not None:
            value["story_subtype"] = self.story_subtype
        return value


def load_multi_image_tasks(
    path: str | Path,
    *,
    default_steps: int,
    default_seed: int,
    max_static_references: int = 2,
) -> tuple[MultiImageEditTask, ...]:
    value = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    raw_tasks = value.get("tasks") if isinstance(value, dict) else value
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("multi-image task JSON must contain a non-empty task list")
    if max_static_references < 1:
        raise ValueError("max_static_references must be positive")

    tasks = []
    for task_index, raw_task in enumerate(raw_tasks, 1):
        if not isinstance(raw_task, Mapping):
            raise ValueError(f"task {task_index} must be a mapping")
        if not bool(raw_task.get("chain_rounds", True)):
            raise ValueError("multi-image inference requires chain_rounds=true")
        image = str(raw_task["image"])
        raw_rounds = raw_task.get("rounds")
        if not isinstance(raw_rounds, list) or not raw_rounds:
            raise ValueError(f"task {task_index} has no edit rounds")

        rounds = []
        for round_offset, raw_round in enumerate(raw_rounds):
            if not isinstance(raw_round, Mapping):
                raise ValueError("multi-image rounds must be mappings")
            round_number = round_offset + 1
            condition_images = _string_tuple(
                raw_round.get("condition_images"), field="condition_images"
            )
            if not 1 <= len(condition_images) <= max_static_references:
                raise ValueError(
                    f"task {task_index} round {round_number} requires 1 to "
                    f"{max_static_references} static references"
                )
            expected_count = int(
                raw_round.get("expected_input_count", 1 + len(condition_images))
            )
            if expected_count != 1 + len(condition_images):
                raise ValueError("expected_input_count disagrees with condition_images")
            expected_primary = (
                "task.image" if round_offset == 0 else f"round_{round_offset}.output"
            )
            primary_source = str(raw_round.get("primary_source", expected_primary))
            if primary_source != expected_primary:
                raise ValueError(
                    f"task {task_index} round {round_number} primary_source must be "
                    f"{expected_primary!r}"
                )
            roles = _mapping_tuple(
                raw_round.get("reference_roles"), field="reference_roles"
            )
            if roles and len(roles) != len(condition_images):
                raise ValueError("reference_roles and condition_images lengths differ")
            if roles:
                expected_indices = tuple(range(2, 2 + len(condition_images)))
                actual_indices = tuple(int(item["image_index"]) for item in roles)
                if actual_indices != expected_indices:
                    raise ValueError(
                        "reference_roles image indices must be ordered from 2"
                    )

            strength = float(raw_round.get("denoising_strength", 1.0))
            noise_mode = str(raw_round.get("noise_mode", "pure_noise")).replace(
                "add_noise_to_current", "add_noise"
            )
            if not 0 < strength <= 1 or noise_mode not in {
                "pure_noise",
                "add_noise",
                "auto",
            }:
                raise ValueError("invalid round denoising strength or noise mode")
            rounds.append(
                MultiImageEditRound(
                    prompt=str(raw_round["prompt"]),
                    condition_images=condition_images,
                    primary_source=primary_source,
                    seed=int(raw_round.get("seed", default_seed + round_offset)),
                    steps=int(raw_round.get("steps", default_steps)),
                    denoising_strength=strength,
                    noise_mode=noise_mode,
                    reference_roles=roles,
                    constraints_hard=_string_tuple(
                        raw_round.get("constraints_hard"), field="constraints_hard"
                    ),
                    constraints_soft=_string_tuple(
                        raw_round.get("constraints_soft"), field="constraints_soft"
                    ),
                    checkpoints=_mapping_tuple(
                        raw_round.get("checkpoints"), field="checkpoints"
                    ),
                )
            )

        task_id = str(raw_task.get("id") or f"{task_index:04d}-{Path(image).stem}")
        if "/" in task_id or task_id in {"", ".", ".."}:
            raise ValueError(f"unsafe task id {task_id!r}")
        tasks.append(
            MultiImageEditTask(
                task_id=task_id,
                task_type=str(raw_task.get("task_type", "multi_image_composition")),
                image=image,
                rounds=tuple(rounds),
                goal=str(raw_task.get("goal", "")),
                chain_rounds=True,
                story_subtype=(
                    None
                    if raw_task.get("story_subtype") is None
                    else str(raw_task["story_subtype"])
                ),
            )
        )
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("task ids must be unique")
    return tuple(tasks)


def balanced_multi_image_task_indices(
    tasks: tuple[MultiImageEditTask, ...], world_size: int
) -> tuple[tuple[int, ...], ...]:
    """Balance chains using their number of total input images as a cost proxy."""

    if world_size < 1:
        raise ValueError("world_size must be positive")
    shards: list[list[int]] = [[] for _ in range(world_size)]
    loads = [0] * world_size
    weighted = [
        (index, sum(1 + len(item.condition_images) for item in task.rounds))
        for index, task in enumerate(tasks)
    ]
    for task_index, cost in sorted(weighted, key=lambda item: (-item[1], item[0])):
        rank = min(range(world_size), key=lambda value: (loads[value], value))
        shards[rank].append(task_index)
        loads[rank] += cost
    return tuple(tuple(sorted(shard)) for shard in shards)
