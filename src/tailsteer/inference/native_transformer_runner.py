"""Shared policy runner for official transformer-native pipelines."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

from tailsteer.utils import save_resolved_config

from .endpoint import plans_from_mapping
from .feedback import feedback_mode
from .metrics import image_comparison_metrics
from .multiround import balanced_task_indices, load_tasks
from .multi_image_multiround import (
    balanced_multi_image_task_indices,
    load_multi_image_tasks,
)
from .panels import (
    COMPARISON_LAYOUT,
    output_comparison_panel,
    source_comparison_panel,
)
from .policies import build_policies

_PROCESS_GROUP_TIMEOUT = timedelta(hours=6)


def _rank_zero_call(action: Callable[[], Any], *, rank: int, dist: Any) -> Any:
    """Publish rank-zero I/O errors before peers leave the collective."""
    result = None
    error: list[str | None] = [None]
    if rank == 0:
        try:
            result = action()
        except Exception as exc:
            if dist is None:
                raise
            error[0] = f"{type(exc).__name__}: {exc}"
    if dist is not None:
        dist.broadcast_object_list(error, src=0)
        if error[0]:
            raise RuntimeError(error[0])
    return result


def _coordinates() -> tuple[int, int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world < 1 or not 0 <= rank < world or local < 0:
        raise ValueError("invalid native-transformer torchrun coordinates")
    return rank, world, local


def _safe_image(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"unsafe native-transformer image path: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_source(path: Path) -> Any:
    from PIL import Image

    with Image.open(path) as opened:
        return opened.convert("RGB")


def cap_source_pixels(source, max_pixels):
    """Optional benchmark area limit; never upscale or crop to a square."""
    from PIL import Image

    if max_pixels is None:
        return source
    if int(max_pixels) < 32 * 32:
        raise ValueError("input_max_pixels must be at least 1024")
    width, height = source.size
    scale = min(1.0, math.sqrt(int(max_pixels) / (width * height)))
    if scale == 1.0:
        return source
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return source.resize(size, Image.Resampling.LANCZOS)


def _seed(value: int) -> None:
    import numpy as np
    import torch

    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _runtime_class(model_name: str) -> Any:
    if model_name == "qwen_image_21":
        from tailsteer.models.qwen_image_21.inference import (
            QwenImage21InferenceRuntime,
        )

        return QwenImage21InferenceRuntime
    if model_name == "flux1_kontext_dev":
        from tailsteer.models.flux1_kontext.inference import KontextInferenceRuntime

        return KontextInferenceRuntime
    raise ValueError(f"unsupported native transformer runtime {model_name!r}")


def _round_prompts(runtime: Any, inputs: dict[str, Any], prompt: str, mode: str):
    """Prepare captions before seeding generation; never change image feedback."""
    prepare = getattr(runtime, "prepare_round_prompt", None)
    if not callable(prepare):
        if mode != "pretrained_if_available":
            raise ValueError("runtime does not support prompt upsampling overrides")
        return {}
    if mode == "disabled":
        raw = {
            "prompt": prompt,
            "input_prompt": prompt,
            "prompt_contract": "raw_instruction_v1",
            "prompt_upsampling": False,
        }
        return {name: {"prepared_prompt": raw} for name in inputs}
    if mode == "per_policy":
        return {
            name: {"prepared_prompt": prepare(source, prompt=prompt)}
            for name, source in inputs.items()
        }
    if mode != "pretrained_if_available":
        raise ValueError(f"invalid prompt reference policy: {mode!r}")
    reference = "base" if "base" in inputs else next(iter(inputs))
    shared = prepare(inputs[reference], prompt=prompt)
    return {name: {"prepared_prompt": shared} for name in inputs}


def _run_task(
    *,
    task: Any,
    runtime: Any,
    inference: dict[str, Any],
    policies: dict[str, Any],
    image_root: Path,
    output: Path,
) -> dict[str, Any]:
    from PIL import Image
    from safetensors.torch import save_file

    task_dir = output / "tasks" / task.task_id
    execution = inference.get("execution")
    endpoint_mode = isinstance(execution, dict) and execution.get("kind") == "endpoint"
    if endpoint_mode:
        plans = plans_from_mapping(
            execution.get("plans", {}), num_steps=int(inference["actor_steps"])
        )
        policies = {}
    else:
        plans = {}
    policy_names = tuple(plans if endpoint_mode else policies)
    feedback = feedback_mode(inference.get("feedback"), default="model_native")
    if feedback not in {"model_native", "latent", "continuous_image", "image"}:
        raise ValueError(f"unsupported feedback mode {feedback!r}")
    save_round_images = bool(inference.get("save_round_images", False))
    if save_round_images and (
        feedback != "image" or not inference.get("input_preprocessed", False)
    ):
        raise ValueError(
            "save_round_images requires image feedback and prepared inputs"
        )
    source = _load_source(_safe_image(image_root, task.image))
    original_size = source.size
    source = cap_source_pixels(source, inference.get("input_max_pixels"))
    geometry = getattr(runtime, "source_geometry", None)
    resolution = (
        geometry(
            source.size, input_preprocessed=inference.get("input_preprocessed", False)
        )
        if callable(geometry)
        else None
    )
    expected_size = (
        tuple(resolution["generation_size"])
        if resolution is not None
        else (source.size if inference.get("input_preprocessed", False) else None)
    )
    task_input = {
        "id": task.task_id,
        "image": task.image,
        "chain_rounds": task.chain_rounds,
        "rounds": [
            item.to_dict() if "multi_condition" in inference else item.__dict__
            for item in task.rounds
        ],
        "num_steps": int(inference["num_steps"]),
        "actor_steps": int(inference["actor_steps"]),
        "policies": (
            {name: plan.to_dict() for name, plan in plans.items()}
            if endpoint_mode
            else {name: policy.to_dict() for name, policy in policies.items()}
        ),
        "execution": "endpoint" if endpoint_mode else "native_policy",
        "original_size": list(source.size),
        "feedback": feedback,
        "schedule": dict(inference["schedule"]),
    }
    prompt_policy = inference.get("prompt_reference_policy", "pretrained_if_available")
    if inference.get("input_max_pixels") is not None:
        task_input["input_preprocessing"] = {
            "policy": "area_cap_lanczos_v1",
            "max_pixels": int(inference["input_max_pixels"]),
            "original_size": list(original_size),
            "resized_size": list(source.size),
        }
    if prompt_policy != "pretrained_if_available":
        # Preserve old shared-caption receipts while distinguishing ablations.
        task_input["prompt_reference_policy"] = prompt_policy
    if inference.get("input_preprocessed", False):
        task_input["input_preprocessed"] = True
    if save_round_images:
        task_input["save_round_images"] = True
    if resolution is not None:
        task_input["resolution"] = resolution
    if "multi_condition" in inference:
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
    if save_round_images:
        source.save(task_dir / "source.png")
    previous_feedback = {name: source for name in policy_names}
    references = {
        path: cap_source_pixels(
            _load_source(_safe_image(image_root, path)),
            inference.get("input_max_pixels"),
        )
        for item in task.rounds
        for path in getattr(item, "condition_images", ())
    }
    round_data: list[dict[str, Any]] = []
    for round_index, round_spec in enumerate(task.rounds, 1):
        if float(round_spec.denoising_strength) != 1.0:
            raise ValueError("official native pipeline requires denoising_strength=1")
        if round_spec.noise_mode not in {"pure_noise", "auto"}:
            raise ValueError("official native pipeline uses pure-noise initialization")
        round_inputs = {
            name: (
                source
                if round_index == 1 or not task.chain_rounds
                else previous_feedback[name]
            )
            for name in policy_names
        }
        results: dict[str, Any] = {}
        prepared_prompts = _round_prompts(
            runtime, round_inputs, round_spec.prompt, prompt_policy
        )
        for name, policy in plans.items() if endpoint_mode else policies.items():
            _seed(int(round_spec.seed))
            actor_policy = policy.actor_policy() if endpoint_mode else policy
            result = runtime.generate(
                round_inputs[name],
                prompt=round_spec.prompt,
                seed=round_spec.seed,
                policy=actor_policy,
                **(
                    {
                        "condition_images": tuple(
                            references[p] for p in round_spec.condition_images
                        )
                    }
                    if "multi_condition" in inference
                    else {}
                ),
                inference={
                    **inference,
                    "feedback_mode": feedback,
                    **prepared_prompts.get(name, {}),
                    **({"endpoint_plan": policy.to_dict()} if endpoint_mode else {}),
                },
            )
            expected_calls = (
                policy.expected_calls
                if endpoint_mode
                else int(inference["actor_steps"])
            )
            if (
                expected_size is not None
                and result.image is not None
                and result.image.size != expected_size
            ):
                raise RuntimeError(
                    "Prepared input canvas changed during native inference"
                )
            if len(result.trace) != expected_calls:
                raise RuntimeError(
                    f"{name} executed {len(result.trace)} actors, expected "
                    f"{expected_calls}"
                )
            policy_dir = task_dir / name
            policy_dir.mkdir(exist_ok=True)
            save_file(
                {"z_final": result.state},
                policy_dir / f"round-{round_index:02d}-state.safetensors",
            )
            if save_round_images:
                if result.image is None:
                    raise RuntimeError("image feedback requires a decoded round image")
                result.image.save(policy_dir / f"round-{round_index:02d}.png")
            results[name] = result
        round_data.append(
            {
                "index": round_index,
                "spec": round_spec,
                "results": results,
            }
        )
        previous_feedback = {
            name: (
                result.native_state
                if feedback in {"model_native", "latent"}
                else (
                    result.continuous_image
                    if feedback == "continuous_image"
                    else result.image
                )
            )
            for name, result in results.items()
        }

    # Float pixels are converted to observations only after the state chain.
    decoded: dict[tuple[int, str], Any] = {}
    for item in round_data:
        round_index = int(item["index"])
        for name, result in item["results"].items():
            image = result.image
            if image is None:
                value = (
                    result.native_state
                    if feedback == "latent"
                    else result.continuous_image
                )
                image = runtime.materialize(value)
            if expected_size is not None and image.size != expected_size:
                raise RuntimeError(
                    "Prepared input canvas changed during native inference"
                )
            decoded[(round_index, name)] = image
    inference_size = decoded[(1, policy_names[0])].size
    if any(image.size != inference_size for image in decoded.values()):
        raise RuntimeError("native pipeline changed resolution inside one chain")
    prepare_source = getattr(runtime, "prepare_source", None)
    display_source = (
        source
        if inference.get("input_preprocessed", False)
        else (
            prepare_source(source)
            if callable(prepare_source)
            else source.resize(inference_size, Image.Resampling.LANCZOS)
        )
    )
    if not save_round_images:
        display_source.save(task_dir / "source.png")
    comparison = task_dir / "comparison"
    comparison.mkdir()
    source_panel = comparison / "source.png"
    source_comparison_panel(display_source, policy_names).save(source_panel)
    records: list[dict[str, Any]] = []
    for item in round_data:
        round_index = int(item["index"])
        round_spec = item["spec"]
        outputs = {name: decoded[(round_index, name)] for name in policy_names}
        state_paths: dict[str, Path] = {}
        image_paths: dict[str, Path] = {}
        for name in policy_names:
            policy_dir = task_dir / name
            image_path = policy_dir / f"round-{round_index:02d}.png"
            state_path = policy_dir / f"round-{round_index:02d}-state.safetensors"
            if not save_round_images:
                outputs[name].save(image_path)
            image_paths[name] = image_path
            state_paths[name] = state_path
        panel_path = comparison / f"round-{round_index:02d}.png"
        output_comparison_panel(outputs, policy_names, prompt=round_spec.prompt).save(
            panel_path
        )
        baseline = outputs.get("base")
        for name in policy_names:
            input_image = (
                display_source
                if round_index == 1 or not task.chain_rounds
                else decoded[(round_index - 1, name)]
            )
            records.append(
                {
                    "round": round_index,
                    "policy": name,
                    "prompt": round_spec.prompt,
                    "seed": round_spec.seed,
                    **(
                        {
                            "condition_images": list(round_spec.condition_images),
                            "primary_source": round_spec.primary_source,
                            "reference_roles": list(round_spec.reference_roles),
                        }
                        if "multi_condition" in inference
                        else {}
                    ),
                    "image": str(image_paths[name].relative_to(output)),
                    "state": str(state_paths[name].relative_to(output)),
                    "actors": list(item["results"][name].trace),
                    **(
                        {"model_metadata": item["results"][name].metadata}
                        if hasattr(item["results"][name], "metadata")
                        else {}
                    ),
                    "execution_spans": (
                        [span.to_dict() for span in plans[name].spans]
                        if endpoint_mode
                        else None
                    ),
                    "comparison_panel": str(panel_path.relative_to(output)),
                    "metrics_vs_pretrained": image_comparison_metrics(
                        outputs[name], baseline
                    ),
                    "metrics_vs_input": image_comparison_metrics(
                        outputs[name], input_image
                    ),
                    "metrics_vs_source": image_comparison_metrics(
                        outputs[name], display_source
                    ),
                }
            )
    manifest = {
        "id": task.task_id,
        **(
            {"chain_rounds": task.chain_rounds}
            if "multi_condition" in inference
            else {}
        ),
        "input": task_input,
        "source": str((task_dir / "source.png").relative_to(output)),
        "source_comparison": str(source_panel.relative_to(output)),
        "original_size": list(source.size),
        "inference_size": list(inference_size),
        "feedback": feedback,
        "inter_round_feedback_quantized": feedback == "image",
        "inter_round_latent_reused": feedback in {"model_native", "latent"},
        "output_materialization": (
            "each_round_with_image_feedback"
            if save_round_images
            else "all_rounds_after_chain"
        ),
        "comparison_layout": COMPARISON_LAYOUT,
        "independent_feedback_chains": True,
        "state_format": "safetensors:z_final",
        "results": records,
    }
    if resolution is not None:
        manifest["resolution"] = resolution
    (task_dir / "task.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (task_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return manifest


def run_native_transformer_inference(
    config: dict[str, Any], *, resume: bool = False
) -> dict[str, Any]:
    import torch

    model = config["model"]
    model_name = str(model["name"])
    runtime_type = _runtime_class(model_name)
    contract = getattr(runtime_type, "preprocessing_contract", None)
    prompt_default = getattr(runtime_type, "default_prompt_reference_policy", None)
    if contract is not None or prompt_default is not None:
        config = deepcopy(config)
    if contract is not None:
        # Stamp even direct worker invocations, before the COMPLETE fast path.
        # A pre-fix square run must never be resumed as native-size inference.
        declared = config["inference"].get("preprocessing")
        if declared is not None and declared != contract:
            raise ValueError("inference preprocessing contract differs from runtime")
        config["inference"]["preprocessing"] = dict(contract)
    if prompt_default is not None:
        # Direct worker calls must use the same default as prepared plans.
        # Explicit shared/disabled configs retain their original semantics.
        config["inference"].setdefault("prompt_reference_policy", prompt_default)
    inference = {**config["inference"], "schedule": dict(config["schedule"])}
    num_steps = int(inference.get("num_steps", 50))
    actor_steps = int(inference.get("actor_steps", num_steps))
    execution = inference.get("execution")
    endpoint_mode = isinstance(execution, dict) and execution.get("kind") == "endpoint"
    if not endpoint_mode and actor_steps != num_steps:
        raise ValueError("native transformer pipelines require one actor per step")
    if endpoint_mode:
        plans = plans_from_mapping(execution.get("plans", {}), num_steps=actor_steps)
        policies = {}
    else:
        plans = {}
        policies = build_policies(inference, num_steps=actor_steps)
    multi_image = "multi_condition" in inference
    if multi_image:
        if not getattr(runtime_type, "supports_multi_image", False):
            raise ValueError(f"{model_name} does not support multi-image inference")
        if (
            endpoint_mode
            or feedback_mode(inference.get("feedback"), default="image") != "image"
        ):
            raise ValueError(
                "multi-image inference requires stepwise policies and feedback=image"
            )
    else:
        raw = json.loads(Path(inference["tasks"]).read_text())
        for task in raw.get("tasks", []) if isinstance(raw, dict) else raw:
            if any(
                isinstance(item, dict) and item.get("condition_images")
                for item in task.get("rounds", [])
            ):
                raise ValueError("multi-image tasks require multi_condition config")
    tasks = (load_multi_image_tasks if multi_image else load_tasks)(
        inference["tasks"],
        default_steps=num_steps,
        default_seed=int(config.get("seed", 0)),
        **(
            {
                "max_static_references": int(
                    inference["multi_condition"].get("max_static_references", 2)
                )
            }
            if multi_image
            else {}
        ),
    )
    if inference.get("override_round_steps", False):
        tasks = tuple(
            replace(
                task,
                rounds=tuple(replace(item, steps=num_steps) for item in task.rounds),
            )
            for task in tasks
        )
    mismatched = [
        (task.task_id, index, item.steps)
        for task in tasks
        for index, item in enumerate(task.rounds, 1)
        if item.steps != num_steps
    ]
    if mismatched:
        raise ValueError(
            f"every native round must use {num_steps} steps: {mismatched[:3]}"
        )
    image_root = Path(inference["image_root"]).expanduser().resolve()
    for task in tasks:
        _safe_image(image_root, task.image)
        for item in task.rounds:
            for path in getattr(item, "condition_images", ()):
                _safe_image(image_root, path)
    output = Path(config["run"]["output_dir"]).expanduser().resolve()
    rank, world, local = _coordinates()
    distributed = world > 1
    dist = None
    owned = False
    if distributed:
        import torch.distributed as torch_dist

        dist = torch_dist
        if not dist.is_initialized():
            dist.init_process_group("gloo", timeout=_PROCESS_GROUP_TIMEOUT)
            owned = True
    error: list[str | None] = [None]
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

                if yaml.safe_load(config_path.read_text(encoding="utf-8")) != config:
                    raise ValueError(
                        "resume config differs from saved inference config"
                    )
            else:
                save_resolved_config(config, output)
        except Exception as exc:
            if not distributed:
                raise
            error[0] = f"{type(exc).__name__}: {exc}"
    if distributed:
        assert dist is not None
        dist.broadcast_object_list(error, src=0)
        if error[0]:
            if owned:
                dist.destroy_process_group()
            raise RuntimeError(error[0])
        dist.barrier()
    if resume and (output / "COMPLETE").is_file():
        # Panels already belong to the completed tasks. Visual migrations are
        # explicit via rebuild_comparison_panels.py, not part of model resume.
        try:
            manifest = _rank_zero_call(
                lambda: json.loads(
                    (output / "manifest.json").read_text(encoding="utf-8")
                ),
                rank=rank,
                dist=dist,
            )
            if rank != 0:
                manifest = json.loads(
                    (output / "manifest.json").read_text(encoding="utf-8")
                )
            return manifest
        finally:
            if owned:
                dist.destroy_process_group()

    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    runtime = runtime_type.load(model, inference.get("export_path"), device=device)
    feedback = feedback_mode(inference.get("feedback"), default="model_native")
    shards = (
        balanced_multi_image_task_indices if multi_image else balanced_task_indices
    )(tasks, world)
    local_manifests = [
        _run_task(
            task=tasks[index],
            runtime=runtime,
            inference=inference,
            policies=plans if endpoint_mode else policies,
            image_root=image_root,
            output=output,
        )
        for index in shards[rank]
    ]
    (output / f"rank-{rank:02d}.json").write_text(
        json.dumps(
            {
                "rank": rank,
                "world_size": world,
                "task_ids": [item["id"] for item in local_manifests],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if distributed:
        assert dist is not None
        dist.barrier()

    def finalize() -> dict[str, Any]:
        task_manifests = [
            json.loads(
                (output / "tasks" / task.task_id / "task.json").read_text(
                    encoding="utf-8"
                )
            )
            for task in tasks
        ]
        manifest = {
            "schema_version": 1,
            "product": runtime.product,
            "num_steps": num_steps,
            "actor_steps": actor_steps,
            "feedback": feedback,
            "schedule": dict(inference["schedule"]),
            "export_path": inference.get("export_path"),
            "export_step": runtime.actors.manifest.get("step"),
            "actor_parameter_count": runtime.actors.parameter_count,
            "comparison_layout": COMPARISON_LAYOUT,
            "policies": (
                {name: plan.to_dict() for name, plan in plans.items()}
                if endpoint_mode
                else {name: policy.to_dict() for name, policy in policies.items()}
            ),
            "execution": "endpoint" if endpoint_mode else "native_policy",
            "evaluation": config.get("evaluation", {}),
            "distributed": {
                "world_size": world,
                "strategy": "whole_chain_balanced_by_input_count"
                if multi_image
                else "whole_chain_balanced_by_round_count",
            },
            "tasks": task_manifests,
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (output / "COMPLETE").write_text("complete\n", encoding="utf-8")
        # _run_task already wrote source/round panels on each worker. Do not
        # decode and re-encode the entire benchmark again on rank zero.
        return manifest

    try:
        manifest = _rank_zero_call(finalize, rank=rank, dist=dist)
        if rank != 0:
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
        return manifest
    finally:
        if owned:
            dist.destroy_process_group()
        runtime.unload()
