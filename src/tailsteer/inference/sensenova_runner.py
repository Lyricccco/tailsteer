"""SenseNova pretrained vs trained independent multi-round comparison."""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

from tailsteer.models import create_model
from tailsteer.models.sensenova.inference import SenseNovaPolicyInference
from tailsteer.utils import save_resolved_config

from .endpoint import plans_from_mapping
from .feedback import feedback_mode
from .metrics import image_comparison_metrics
from .multi_image_multiround import (
    balanced_multi_image_task_indices,
    load_multi_image_tasks,
)
from .multiround import balanced_task_indices, load_tasks
from .panels import (
    COMPARISON_LAYOUT,
    output_comparison_panel,
    source_comparison_panel,
)
from .policies import InferencePolicy, build_policies

SENSENOVA_REFERENCE_MIN_PIXELS = 512 * 512
SENSENOVA_REFERENCE_MAX_PIXELS = 2048 * 2048
SENSENOVA_FIXED_SCHEDULE = {
    "shift_policy": "fixed_reference_shape",
    "reference_height": 1024,
    "reference_width": 1024,
}


def _distributed_coordinates() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError(
            "invalid torchrun coordinates: "
            f"rank={rank} local_rank={local_rank} world_size={world_size}"
        )
    return rank, world_size, local_rank


def sensenova_native_resolution(
    size: tuple[int, int],
    *,
    factor: int = 32,
    min_pixels: int = SENSENOVA_REFERENCE_MIN_PIXELS,
    max_pixels: int = SENSENOVA_REFERENCE_MAX_PIXELS,
) -> tuple[int, int]:
    """Return the exact size selected by official SenseNova ``smart_resize``.

    The public repository keeps this helper inside its model package, which is
    unavailable to lightweight inference tests. Keep this small arithmetic
    equivalent here so source preprocessing and its tests do not require model
    weights or CUDA.
    """

    width, height = (int(value) for value in size)
    if width <= 0 or height <= 0:
        raise ValueError(f"image dimensions must be positive, got {size}")
    if factor <= 0:
        raise ValueError(f"resolution factor must be positive, got {factor}")
    if min_pixels <= 0 or max_pixels < min_pixels:
        raise ValueError("invalid SenseNova pixel bounds")
    aspect = max(height, width) / min(height, width)
    if aspect > 200:
        raise ValueError(f"absolute aspect ratio must be at most 200, got {aspect}")

    def nearest(value: float) -> int:
        return round(value / factor) * factor

    def ceil_factor(value: float) -> int:
        return math.ceil(value / factor) * factor

    def floor_factor(value: float) -> int:
        return math.floor(value / factor) * factor

    aligned_height = max(factor, nearest(height))
    aligned_width = max(factor, nearest(width))
    if aligned_height * aligned_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        aligned_height = max(factor, floor_factor(height / beta))
        aligned_width = max(factor, floor_factor(width / beta))
    elif aligned_height * aligned_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        aligned_height = ceil_factor(height * beta)
        aligned_width = ceil_factor(width * beta)
    return aligned_width, aligned_height


def _prepare_model_input(
    original: Any,
    *,
    inference: dict[str, Any],
    image_multiple: int,
) -> tuple[Any, dict[str, Any]]:
    """Keep the original input; compute only the official native output size."""

    requested_width = inference.get("width")
    requested_height = inference.get("height")
    if requested_width is None and requested_height is None:
        target_size = sensenova_native_resolution(original.size, factor=image_multiple)
        selection = "official_native"
    else:
        if requested_width is None or requested_height is None:
            raise ValueError(
                "SenseNova inference width and height must be set together"
            )
        target_size = (int(requested_width), int(requested_height))
        native_target = sensenova_native_resolution(target_size, factor=image_multiple)
        if native_target != target_size:
            raise ValueError(
                "explicit SenseNova inference size would be resized again by the "
                "reference encoder: "
                f"requested={target_size}, official_native={native_target}"
            )
        selection = "explicit_native"

    original_width, original_height = original.size
    target_width, target_height = target_size
    metadata = {
        "policy": "official_native_no_crop",
        "selection": selection,
        "original_size": [original_width, original_height],
        "reference_size": [target_width, target_height],
        "generation_size": [target_width, target_height],
        "token_grid": {
            "width": target_width // image_multiple,
            "height": target_height // image_multiple,
        },
        "image_multiple": image_multiple,
        "crop_fraction": 0.0,
        "aspect_ratio_preserved": True,
    }
    return original.copy(), metadata


def _display_image(image: Any, size: tuple[int, int]) -> Any:
    from PIL import Image

    if image.size == size:
        return image
    return image.resize(size, Image.Resampling.LANCZOS)


def _safe_image(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"task image escapes image_root: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _conditioning_actor(engine: Any, policy: Any, *, num_steps: int) -> str:
    actors = getattr(policy, "actors", None)
    if actors is not None:
        return "refined" if "refined" in actors else "base"
    resolver = getattr(engine, "conditioning_actor_for", None)
    if not callable(resolver):
        # Lightweight test/dry-run engines predate conditioning actors.
        return "base"
    actor = str(resolver(policy, num_steps=num_steps))
    if actor not in {"base", "refined"}:
        raise ValueError(f"invalid SenseNova conditioning actor {actor!r}")
    return actor


def _runtime_policy(policy: Any) -> Any:
    factory = getattr(policy, "actor_policy", None)
    return factory() if callable(factory) else policy


def _runtime_config(inference: dict[str, Any], policy: Any) -> dict[str, Any]:
    endpoint = getattr(policy, "to_dict", None)
    if callable(endpoint) and hasattr(policy, "spans"):
        return {**inference, "endpoint_plan": endpoint()}
    return dict(inference)


def _run_task(
    *,
    task: Any,
    engine: Any,
    inference: dict[str, Any],
    image_root: Path,
    output: Path,
    refined_policy: str,
) -> dict[str, Any]:
    from PIL import Image

    task_dir = output / "tasks" / task.task_id
    task_input = {
        "id": task.task_id,
        "image": task.image,
        "chain_rounds": task.chain_rounds,
        "rounds": [item.__dict__ for item in task.rounds],
        "refined_policy": refined_policy,
        "num_steps": int(inference.get("num_steps", 50)),
        "activation_step": (
            0
            if refined_policy == "all_steps"
            else int(
                inference.get(
                    "activation_step", int(inference.get("num_steps", 50)) - 1
                )
            )
        ),
        "resolution_policy": "official_native_no_crop",
        "requested_width": inference.get("width"),
        "requested_height": inference.get("height"),
        "schedule": dict(inference.get("schedule", SENSENOVA_FIXED_SCHEDULE)),
    }
    if (task_dir / "COMPLETE").is_file():
        existing = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
        if existing.get("input") != task_input:
            raise ValueError(f"completed task has different inputs: {task.task_id}")
        return existing
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True)
    with Image.open(_safe_image(image_root, task.image)) as opened:
        original = opened.convert("RGB")
    model_input, resolution = _prepare_model_input(
        original,
        inference=inference,
        image_multiple=int(getattr(engine, "image_multiple", 32)),
    )
    display_size = tuple(resolution["generation_size"])
    display_source = _display_image(model_input, display_size)
    display_source.save(task_dir / "source.png")
    policies = ("base", refined_policy)
    conditioning_actors = {
        policy: _conditioning_actor(
            engine,
            policy,
            num_steps=int(inference.get("num_steps", 50)),
        )
        for policy in policies
    }
    comparison_dir = task_dir / "comparison"
    comparison_dir.mkdir(exist_ok=True)
    source_panel_path = comparison_dir / "source.png"
    source_comparison_panel(display_source, policies).save(source_panel_path)
    previous = {policy: model_input for policy in policies}
    records = []
    for round_index, round_spec in enumerate(task.rounds, 1):
        if float(round_spec.denoising_strength) != 1.0:
            raise ValueError("SenseNova official sampler requires denoising_strength=1")
        if round_spec.noise_mode not in {"pure_noise", "auto"}:
            raise ValueError(
                "SenseNova official sampler uses pure-noise initialization"
            )
        round_input = {
            policy: (
                model_input
                if round_index == 1 or not task.chain_rounds
                else previous[policy]
            )
            for policy in policies
        }
        outputs: dict[str, Any] = {}
        traces: dict[str, tuple[str, ...]] = {}
        for policy in policies:
            outputs[policy], traces[policy] = engine.generate(
                round_input[policy],
                prompt=round_spec.prompt,
                seed=round_spec.seed,
                policy=_runtime_policy(policy),
                config={
                    **_runtime_config(inference, policy),
                    "num_steps": round_spec.steps,
                    "width": display_size[0],
                    "height": display_size[1],
                },
            )
            expected_calls = getattr(policy, "expected_calls", int(round_spec.steps))
            if len(traces[policy]) != expected_calls:
                raise RuntimeError(
                    f"{policy} executed {len(traces[policy])} actors, expected "
                    f"{expected_calls}"
                )
            policy_dir = task_dir / policy
            policy_dir.mkdir(exist_ok=True)
            outputs[policy].save(policy_dir / f"round-{round_index:02d}.png")
        panel_path = comparison_dir / f"round-{round_index:02d}.png"
        output_comparison_panel(
            outputs,
            policies,
            prompt=round_spec.prompt,
        ).save(panel_path)
        metrics = image_comparison_metrics(outputs[refined_policy], outputs["base"])
        records.append(
            {
                "round": round_index,
                "prompt": round_spec.prompt,
                "seed": round_spec.seed,
                "pretrained_image": str(
                    (task_dir / "base" / f"round-{round_index:02d}.png").relative_to(
                        output
                    )
                ),
                "refined_image": str(
                    (
                        task_dir / refined_policy / f"round-{round_index:02d}.png"
                    ).relative_to(output)
                ),
                "comparison_panel": str(panel_path.relative_to(output)),
                "pretrained_actors": list(traces["base"]),
                "refined_actors": list(traces[refined_policy]),
                "pretrained_conditioning_actor": conditioning_actors["base"],
                "refined_conditioning_actor": conditioning_actors[refined_policy],
                "metrics_pretrained_vs_input": image_comparison_metrics(
                    outputs["base"],
                    _display_image(round_input["base"], display_size),
                ),
                "metrics_refined_vs_input": image_comparison_metrics(
                    outputs[refined_policy],
                    _display_image(round_input[refined_policy], display_size),
                ),
                "metrics_refined_vs_pretrained": metrics,
            }
        )
        previous = outputs
    manifest = {
        "id": task.task_id,
        "input": task_input,
        "source": str((task_dir / "source.png").relative_to(output)),
        "source_comparison": str(source_panel_path.relative_to(output)),
        "resolution": resolution,
        "comparison_layout": COMPARISON_LAYOUT,
        "independent_feedback_chains": True,
        "results": records,
    }
    (task_dir / "task.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (task_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return manifest


def _run_segmented_task(
    *,
    task: Any,
    engine: Any,
    inference: dict[str, Any],
    policies: dict[str, InferencePolicy],
    image_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Run arbitrary pretrained/refined segments through the official sampler."""

    from PIL import Image

    task_dir = output / "tasks" / task.task_id
    conditioning_actors = {
        name: _conditioning_actor(
            engine, policy, num_steps=int(inference.get("num_steps", 50))
        )
        for name, policy in policies.items()
    }
    task_input = {
        "id": task.task_id,
        "image": task.image,
        "chain_rounds": task.chain_rounds,
        "rounds": [
            item.to_dict() if hasattr(item, "to_dict") else item.__dict__
            for item in task.rounds
        ],
        "num_steps": int(inference.get("num_steps", 50)),
        "policies": {name: policy.to_dict() for name, policy in policies.items()},
        "resolution_policy": "official_native_no_crop",
        "requested_width": inference.get("width"),
        "requested_height": inference.get("height"),
        "schedule": dict(inference.get("schedule", SENSENOVA_FIXED_SCHEDULE)),
        "conditioning_actors": conditioning_actors,
    }
    if "multi_condition" in inference:
        # Shared multi-image manifest fields are also consumed by animations.
        task_input["task"] = task.to_dict()
        task_input["inference"] = {
            "image_root": str(image_root),
            "multi_condition": dict(inference["multi_condition"]),
        }
    if (task_dir / "COMPLETE").is_file():
        existing = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
        if existing.get("input") != task_input:
            raise ValueError(f"completed task has different inputs: {task.task_id}")
        return existing
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True)
    with Image.open(_safe_image(image_root, task.image)) as opened:
        original = opened.convert("RGB")
    model_input, resolution = _prepare_model_input(
        original,
        inference=inference,
        image_multiple=int(getattr(engine, "image_multiple", 32)),
    )
    display_size = tuple(resolution["generation_size"])
    display_source = _display_image(model_input, display_size)
    display_source.save(task_dir / "source.png")
    policy_names = tuple(policies)
    comparison_dir = task_dir / "comparison"
    comparison_dir.mkdir(exist_ok=True)
    source_panel_path = comparison_dir / "source.png"
    source_comparison_panel(display_source, policy_names).save(source_panel_path)
    previous = {name: model_input for name in policy_names}
    records: list[dict[str, Any]] = []
    for round_index, round_spec in enumerate(task.rounds, 1):
        if float(round_spec.denoising_strength) != 1.0:
            raise ValueError("SenseNova official sampler requires denoising_strength=1")
        if round_spec.noise_mode not in {"pure_noise", "auto"}:
            raise ValueError(
                "SenseNova official sampler uses pure-noise initialization"
            )
        round_inputs = {
            name: (
                model_input
                if round_index == 1 or not task.chain_rounds
                else previous[name]
            )
            for name in policy_names
        }
        outputs: dict[str, Any] = {}
        traces: dict[str, tuple[str, ...]] = {}
        reference_images = []
        for path in getattr(round_spec, "condition_images", ()):
            with Image.open(_safe_image(image_root, path)) as opened:
                reference_images.append(opened.convert("RGB"))
        for name, policy in policies.items():
            outputs[name], traces[name] = engine.generate(
                round_inputs[name],
                prompt=round_spec.prompt,
                seed=round_spec.seed,
                policy=_runtime_policy(policy),
                config={
                    **_runtime_config(inference, policy),
                    "num_steps": round_spec.steps,
                    "width": display_size[0],
                    "height": display_size[1],
                },
                **(
                    {"condition_images": tuple(reference_images)}
                    if reference_images
                    else {}
                ),
            )
            expected_calls = getattr(policy, "expected_calls", int(round_spec.steps))
            if len(traces[name]) != expected_calls:
                raise RuntimeError(
                    f"{name} executed {len(traces[name])} actors, expected "
                    f"{expected_calls}"
                )
            policy_dir = task_dir / name
            policy_dir.mkdir(exist_ok=True)
            outputs[name].save(policy_dir / f"round-{round_index:02d}.png")

        panel_path = comparison_dir / f"round-{round_index:02d}.png"
        panel = output_comparison_panel(
            outputs,
            policy_names,
            prompt=round_spec.prompt,
        )
        panel.save(panel_path)
        baseline = outputs.get("base")
        for name in policy_names:
            image_path = task_dir / name / f"round-{round_index:02d}.png"
            records.append(
                {
                    **(round_spec.to_dict() if reference_images else {}),
                    "round": round_index,
                    "policy": name,
                    "prompt": round_spec.prompt,
                    "seed": round_spec.seed,
                    "image": str(image_path.relative_to(output)),
                    "actors": list(traces[name]),
                    "execution_spans": (
                        [span.to_dict() for span in policy.spans]
                        if hasattr(policy, "spans")
                        else None
                    ),
                    "conditioning_actor": conditioning_actors[name],
                    "comparison_panel": str(panel_path.relative_to(output)),
                    "metrics_vs_pretrained": image_comparison_metrics(
                        outputs[name], baseline
                    ),
                    "metrics_vs_input": image_comparison_metrics(
                        outputs[name], _display_image(round_inputs[name], display_size)
                    ),
                    "metrics_vs_source": image_comparison_metrics(
                        outputs[name], display_source
                    ),
                }
            )
        previous = outputs
    manifest = {
        "id": task.task_id,
        "input": task_input,
        "source": str((task_dir / "source.png").relative_to(output)),
        "source_comparison": str(source_panel_path.relative_to(output)),
        "resolution": resolution,
        "feedback": "image",
        **(
            {"chain_rounds": True, "task_type": task.task_type}
            if "multi_condition" in inference
            else {}
        ),
        "comparison_layout": COMPARISON_LAYOUT,
        "independent_feedback_chains": True,
        "results": records,
    }
    (task_dir / "task.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (task_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return manifest


def _run_segmented_model_native_task(
    *,
    task: Any,
    engine: Any,
    inference: dict[str, Any],
    policies: dict[str, InferencePolicy],
    image_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Keep round feedback continuous; decode saved states after the chain."""

    from PIL import Image
    from safetensors.torch import load_file, save_file

    from tailsteer.models.sensenova.official import decode_sensenova_state

    task_dir = output / "tasks" / task.task_id
    conditioning_actors = {
        name: _conditioning_actor(
            engine, policy, num_steps=int(inference.get("num_steps", 50))
        )
        for name, policy in policies.items()
    }
    task_input = {
        "id": task.task_id,
        "image": task.image,
        "chain_rounds": task.chain_rounds,
        "rounds": [item.__dict__ for item in task.rounds],
        "num_steps": int(inference.get("num_steps", 50)),
        "policies": {name: policy.to_dict() for name, policy in policies.items()},
        "resolution_policy": "official_native_no_crop",
        "requested_width": inference.get("width"),
        "requested_height": inference.get("height"),
        "schedule": dict(inference.get("schedule", SENSENOVA_FIXED_SCHEDULE)),
        "feedback": "model_native",
        "conditioning_actors": conditioning_actors,
    }
    if (task_dir / "COMPLETE").is_file():
        existing = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
        if existing.get("input") != task_input:
            raise ValueError(f"completed task has different inputs: {task.task_id}")
        return existing
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True)
    with Image.open(_safe_image(image_root, task.image)) as opened:
        original = opened.convert("RGB")
    model_input, resolution = _prepare_model_input(
        original,
        inference=inference,
        image_multiple=int(getattr(engine, "image_multiple", 32)),
    )
    width, height = tuple(resolution["generation_size"])
    display_source = _display_image(model_input, (width, height))
    display_source.save(task_dir / "source.png")
    policy_names = tuple(policies)
    comparison_dir = task_dir / "comparison"
    comparison_dir.mkdir(exist_ok=True)
    source_panel_path = comparison_dir / "source.png"
    source_comparison_panel(display_source, policy_names).save(source_panel_path)

    initial_by_actor = {
        actor: engine.encode_reference_image(model_input, actor=actor)
        for actor in set(conditioning_actors.values())
    }
    initial_references = {
        name: initial_by_actor[conditioning_actors[name]] for name in policy_names
    }
    references = dict(initial_references)
    state_paths: dict[tuple[int, str], Path] = {}
    traces: dict[tuple[int, str], tuple[str, ...]] = {}
    for round_index, round_spec in enumerate(task.rounds, 1):
        if float(round_spec.denoising_strength) != 1.0:
            raise ValueError("SenseNova official sampler requires denoising_strength=1")
        if round_spec.noise_mode not in {"pure_noise", "auto"}:
            raise ValueError(
                "SenseNova official sampler uses pure-noise initialization"
            )
        for name, policy in policies.items():
            reference = (
                initial_references[name]
                if round_index == 1 or not task.chain_rounds
                else references[name]
            )
            state, next_reference, trace = engine.generate_model_native(
                model_input,
                reference_state=reference,
                prompt=round_spec.prompt,
                seed=round_spec.seed,
                policy=_runtime_policy(policy),
                config={
                    **_runtime_config(inference, policy),
                    "num_steps": round_spec.steps,
                    "width": width,
                    "height": height,
                },
                encode_next_reference=(
                    task.chain_rounds and round_index < len(task.rounds)
                ),
            )
            expected_calls = getattr(policy, "expected_calls", int(round_spec.steps))
            if len(trace) != expected_calls:
                raise RuntimeError(
                    f"{name} executed {len(trace)} actors, expected {expected_calls}"
                )
            policy_dir = task_dir / name
            policy_dir.mkdir(exist_ok=True)
            state_path = policy_dir / f"round-{round_index:02d}-state.safetensors"
            save_file({"z_final": state}, state_path)
            state_paths[(round_index, name)] = state_path
            traces[(round_index, name)] = trace
            if next_reference is not None:
                references[name] = next_reference

    # No output state is quantized while the chain is active. Decode every
    # saved z_final only after all reference embeddings have been consumed.
    decoded: dict[tuple[int, str], Any] = {}
    for round_index, _round_spec in enumerate(task.rounds, 1):
        for name in policy_names:
            state_path = state_paths[(round_index, name)]
            state = load_file(state_path, device="cpu")["z_final"]
            image = decode_sensenova_state(state, width=width, height=height)
            image.save(task_dir / name / f"round-{round_index:02d}.png")
            decoded[(round_index, name)] = image

    records: list[dict[str, Any]] = []
    for round_index, round_spec in enumerate(task.rounds, 1):
        outputs = {name: decoded[(round_index, name)] for name in policy_names}
        panel_path = comparison_dir / f"round-{round_index:02d}.png"
        output_comparison_panel(
            outputs,
            policy_names,
            prompt=round_spec.prompt,
        ).save(panel_path)
        baseline = outputs.get("base")
        for name in policy_names:
            round_input = (
                model_input
                if round_index == 1 or not task.chain_rounds
                else decoded[(round_index - 1, name)]
            )
            image_path = task_dir / name / f"round-{round_index:02d}.png"
            state_path = state_paths[(round_index, name)]
            records.append(
                {
                    "round": round_index,
                    "policy": name,
                    "prompt": round_spec.prompt,
                    "seed": round_spec.seed,
                    "state": str(state_path.relative_to(output)),
                    "image": str(image_path.relative_to(output)),
                    "actors": list(traces[(round_index, name)]),
                    "execution_spans": (
                        [span.to_dict() for span in policy.spans]
                        if hasattr(policy, "spans")
                        else None
                    ),
                    "conditioning_actor": conditioning_actors[name],
                    "comparison_panel": str(panel_path.relative_to(output)),
                    "metrics_vs_pretrained": image_comparison_metrics(
                        outputs[name], baseline
                    ),
                    "metrics_vs_input": image_comparison_metrics(
                        outputs[name], _display_image(round_input, (width, height))
                    ),
                    "metrics_vs_source": image_comparison_metrics(
                        outputs[name], display_source
                    ),
                }
            )
    manifest = {
        "id": task.task_id,
        "input": task_input,
        "source": str((task_dir / "source.png").relative_to(output)),
        "source_comparison": str(source_panel_path.relative_to(output)),
        "resolution": resolution,
        "feedback": "model_native",
        "native_state": "z_final_pixels+reference_embedding",
        "inter_round_feedback_quantized": False,
        "output_materialization": "all_rounds_after_chain",
        "state_format": "safetensors:z_final",
        "decoded_after_chain": "all_rounds",
        "comparison_layout": COMPARISON_LAYOUT,
        "independent_feedback_chains": True,
        "results": records,
    }
    (task_dir / "task.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (task_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return manifest


def run_sensenova_inference(
    config: dict[str, Any], *, refined_policy: str | None = None, resume: bool = False
) -> dict[str, Any]:
    import torch

    inference = {**config["inference"], "schedule": dict(config["schedule"])}
    num_steps = int(inference.get("num_steps", 50))
    execution = inference.get("execution")
    endpoint_mode = isinstance(execution, dict) and execution.get("kind") == "endpoint"
    segmented = endpoint_mode or isinstance(inference.get("policies"), dict)
    feedback = feedback_mode(inference.get("feedback"), default="model_native")
    if feedback not in {"image", "model_native"}:
        raise ValueError(f"unsupported SenseNova feedback mode {feedback!r}")
    if feedback == "model_native" and not segmented:
        raise ValueError("model-native feedback requires segmented policies")
    policy_map: dict[str, Any] | None = None
    selected_policy: str | None = None
    activation: int | None = None
    if segmented:
        if refined_policy is not None:
            raise ValueError(
                "--refined-policy cannot override a segmented policy config"
            )
        if endpoint_mode:
            policy_map = plans_from_mapping(
                execution.get("plans", {}), num_steps=num_steps
            )
        else:
            policy_map = build_policies(inference, num_steps=num_steps)
    else:
        selected_policy = str(
            refined_policy or inference.get("refined_policy", "late_step")
        )
        if selected_policy not in {"late_step", "all_steps"}:
            raise ValueError("refined_policy must be late_step or all_steps")
        activation = int(inference.get("activation_step", num_steps - 1))
        if selected_policy == "late_step" and activation != num_steps - 1:
            raise ValueError(
                "the trained step-50 policy requires activation_step=num_steps-1"
            )
    multi_image = "multi_condition" in inference
    if multi_image:
        if not segmented or endpoint_mode or feedback != "image":
            raise ValueError(
                "SenseNova multi-image inference requires stepwise policies "
                "and feedback=image"
            )
        tasks = load_multi_image_tasks(
            inference["tasks"],
            default_steps=num_steps,
            default_seed=int(config.get("seed", 0)),
            max_static_references=int(
                inference["multi_condition"].get("max_static_references", 2)
            ),
        )
    else:
        # Never silently drop static references through the single-image parser.
        raw = json.loads(Path(inference["tasks"]).read_text(encoding="utf-8"))
        rows = raw.get("tasks", []) if isinstance(raw, dict) else raw
        if any(
            isinstance(item, dict) and item.get("condition_images")
            for row in rows
            for item in row.get("rounds", [])
        ):
            raise ValueError(
                "SenseNova multi-image tasks require multi_condition config"
            )
        tasks = load_tasks(
            inference["tasks"],
            default_steps=num_steps,
            default_seed=int(config.get("seed", 0)),
        )
    mismatched = [
        (task.task_id, round_index, item.steps)
        for task in tasks
        for round_index, item in enumerate(task.rounds, 1)
        if item.steps != num_steps
    ]
    if mismatched:
        raise ValueError(
            "every SenseNova round must use the configured step count; "
            f"mismatches={mismatched[:3]}"
        )
    image_root = Path(inference["image_root"]).expanduser().resolve()
    for task in tasks:
        _safe_image(image_root, task.image)
        for item in task.rounds:
            for path in getattr(item, "condition_images", ()):
                _safe_image(image_root, path)

    output_root = Path(config["run"]["output_dir"]).expanduser().resolve()
    output = (
        output_root / str(selected_policy)
        if not segmented and bool(config["run"].get("policy_subdir", True))
        else output_root
    )
    repository = Path(__file__).resolve().parents[3]
    if output == repository or repository in output.parents:
        raise ValueError("inference results must live outside the TailSteer repository")
    rank, world_size, local_rank = _distributed_coordinates()
    distributed = world_size > 1
    process_group_owned = False
    dist = None
    if distributed:
        import torch.distributed as torch_dist

        dist = torch_dist
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo")
            process_group_owned = True

    resolved = (
        config
        if segmented
        else {
            **config,
            "inference": {**inference, "refined_policy": selected_policy},
        }
    )
    setup_error: list[str | None] = [None]
    if rank == 0:
        try:
            if output.exists() and not resume and any(output.iterdir()):
                raise FileExistsError(
                    f"inference output exists; pass --resume: {output}"
                )
            output.mkdir(parents=True, exist_ok=True)
            (output / "tasks").mkdir(exist_ok=True)
            config_path = output / "config.yaml"
            if config_path.is_file():
                import yaml

                if yaml.safe_load(config_path.read_text(encoding="utf-8")) != resolved:
                    raise ValueError(
                        "resume config differs from saved inference config"
                    )
            else:
                save_resolved_config(resolved, output)
        except Exception as error:
            if not distributed:
                raise
            setup_error[0] = f"{type(error).__name__}: {error}"
    if distributed:
        assert dist is not None
        dist.broadcast_object_list(setup_error, src=0)
        if setup_error[0] is not None:
            if process_group_owned:
                dist.destroy_process_group()
            raise RuntimeError(setup_error[0])
        dist.barrier()

    if resume and (output / "COMPLETE").is_file():
        # Completed tasks already contain their panels. Resume must not rewrite
        # all images on rank 0 while other workers wait for a collective.
        # Legacy layout migration is explicit: scripts/rebuild_comparison_panels.py.
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if distributed and process_group_owned:
            assert dist is not None
            dist.destroy_process_group()
        return manifest

    device_index = local_rank if distributed else int(inference.get("device", 0))
    # The official SenseNova runtime and its fused CUDA kernels consult the
    # process-local current device in addition to tensor.device. torchrun does
    # not set it automatically, so every rank must bind before model creation.
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    model = create_model(config["model"], device=device, include_vae=False)
    engine = SenseNovaPolicyInference(model, inference.get("export_path"))
    shards = (
        balanced_multi_image_task_indices(tasks, world_size)
        if multi_image
        else balanced_task_indices(tasks, world_size)
    )
    if segmented:
        assert policy_map is not None
        task_runner = (
            _run_segmented_model_native_task
            if feedback == "model_native"
            else _run_segmented_task
        )
        local_task_manifests = [
            task_runner(
                task=tasks[task_index],
                engine=engine,
                inference=inference,
                policies=policy_map,
                image_root=image_root,
                output=output,
            )
            for task_index in shards[rank]
        ]
    else:
        assert selected_policy is not None
        local_task_manifests = [
            _run_task(
                task=tasks[task_index],
                engine=engine,
                inference=inference,
                image_root=image_root,
                output=output,
                refined_policy=selected_policy,
            )
            for task_index in shards[rank]
        ]
    (output / f"rank-{rank:02d}.json").write_text(
        json.dumps(
            {
                "rank": rank,
                "world_size": world_size,
                "task_ids": [item["id"] for item in local_task_manifests],
                "round_count": sum(
                    len(item["results"]) for item in local_task_manifests
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if distributed:
        assert dist is not None
        dist.barrier()
    if rank == 0:
        task_manifests = [
            json.loads(
                (output / "tasks" / task.task_id / "task.json").read_text(
                    encoding="utf-8"
                )
            )
            for task in tasks
        ]
        manifest = {
            "schema_version": (
                6
                if endpoint_mode
                else 5
                if segmented and feedback == "model_native"
                else 4
                if segmented
                else 3
            ),
            "product": "sensenova_independent_multiround_comparison",
            "num_steps": num_steps,
            "feedback": feedback,
            "schedule": dict(inference["schedule"]),
            "export_path": inference.get("export_path"),
            "export_step": engine.actors.manifest.get("step"),
            "export_schema_version": getattr(
                engine.actors.export, "schema_version", None
            ),
            "actor_parameter_count": engine.actors.parameter_count,
            "generation_parameter_count": (engine.actors.generation_parameter_count),
            "conditioning_parameter_count": (
                engine.actors.conditioning_parameter_count
            ),
            "comparison_layout": COMPARISON_LAYOUT,
            "distributed": {
                "world_size": world_size,
                "strategy": (
                    "whole_chain_balanced_by_input_count"
                    if multi_image
                    else "whole_chain_balanced_by_round_count"
                ),
                "rank_round_counts": [
                    sum(len(tasks[index].rounds) for index in shard) for shard in shards
                ],
            },
            "tasks": task_manifests,
        }
        if segmented:
            assert policy_map is not None
            manifest["policies"] = {
                name: policy.to_dict() for name, policy in policy_map.items()
            }
            if endpoint_mode:
                manifest["execution"] = "endpoint"
            manifest["evaluation"] = config.get("evaluation", {})
        else:
            manifest["refined_policy"] = selected_policy
            manifest["activation_step"] = (
                activation if selected_policy == "late_step" else 0
            )
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (output / "COMPLETE").write_text("complete\n", encoding="utf-8")
        # Each task runner saved source/round panels on its own worker; there is
        # no image rendering left to repeat at this distributed finalization.
    if distributed:
        assert dist is not None
        dist.barrier()
        if rank != 0:
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
        if process_group_owned:
            dist.destroy_process_group()
    return manifest
