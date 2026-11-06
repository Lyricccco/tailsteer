"""Independent multi-image FLUX.2 chains with latent or image feedback."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from tailsteer.models import create_model
from tailsteer.models.flux2 import (
    build_flux2_schedule,
    flux2_reference_max_pixels,
)
from tailsteer.models.flux2.multi_image_inference import (
    build_multi_image_policies,
    prepare_multi_image_conditioning,
    run_multi_image_policy,
)
from tailsteer.utils import save_resolved_config

from .feedback import feedback_mode
from .metrics import (
    comparison_metrics,
    image_comparison_metrics,
    latent_comparison_metrics,
)
from .multi_image_multiround import (
    balanced_multi_image_task_indices,
    load_multi_image_tasks,
)
from .panels import (
    COMPARISON_LAYOUT,
    output_comparison_panel,
    rebuild_output_comparison_panels,
    source_comparison_panel,
)


def _safe_image(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"task image escapes image_root: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _schedule(
    config: dict[str, Any], *, steps: int, height: int, width: int, strength: float
) -> Any:
    schedule = config.get("schedule", {})
    policy = str(schedule.get("shift_policy", "fixed_reference_shape"))
    if policy == "fixed_reference_shape":
        schedule_height = int(schedule.get("reference_height", 1024))
        schedule_width = int(schedule.get("reference_width", 1024))
    elif policy in {"actual_resolution", "native"}:
        schedule_height, schedule_width = height, width
    else:
        raise ValueError("schedule.shift_policy is invalid")
    return build_flux2_schedule(
        num_steps=steps,
        height=schedule_height,
        width=schedule_width,
        denoising_strength=strength,
    )


def _decode_round(
    *,
    model: Any,
    round_index: int,
    round_spec: Any,
    task_dir: Path,
    output: Path,
    model_input: Any,
    task_records: list[dict[str, Any]],
    metrics_records: list[dict[str, Any]],
    policy_names: tuple[str, ...],
    infer_height: int,
    infer_width: int,
    previous_images: dict[str, Any],
    after_chain: bool,
) -> dict[str, Any]:
    """Save one paired round, either immediately or during final observation."""

    from safetensors.torch import load_file

    records = {
        str(item["policy"]): item
        for item in task_records
        if int(item["round"]) == round_index
    }
    if set(records) != set(policy_names):
        raise ValueError(f"round {round_index} does not contain every policy")
    latents: dict[str, Any] = {}
    images: dict[str, Any] = {}
    for name in policy_names:
        record = records[name]
        latents[name] = load_file(str(output / record["latent"]), device="cpu")[
            "latent"
        ]
        images[name] = model.decode_latent(
            latents[name], height=infer_height, width=infer_width
        )
        image_path = task_dir / name / f"round-{round_index:02d}.png"
        images[name].save(image_path)
        record["image"] = str(image_path.relative_to(output))
        record["decoded_after_chain"] = after_chain

    baseline_latent = latents.get("base")
    baseline_image = images.get("base")
    for name in policy_names:
        metrics = comparison_metrics(
            latents[name], baseline_latent, images[name], baseline_image
        )
        records[name]["metrics_vs_pretrained"] = metrics
        records[name]["metrics_vs_input"] = image_comparison_metrics(
            images[name], previous_images[name]
        )
        records[name]["metrics_vs_source"] = image_comparison_metrics(
            images[name], model_input
        )
        for metric_record in metrics_records:
            if (
                int(metric_record["round"]) == round_index
                and metric_record["policy"] == name
            ):
                metric_record.update(metrics)
                break

    comparison_dir = task_dir / "comparison"
    comparison_dir.mkdir(exist_ok=True)
    panel_path = comparison_dir / f"round-{round_index:02d}.png"
    output_comparison_panel(
        images,
        policy_names,
        prompt=round_spec.prompt,
    ).save(panel_path)
    for record in records.values():
        record["comparison_panel"] = str(panel_path.relative_to(output))
    return images


def _decode_chain_once(*, task: Any, **kwargs: Any) -> int:
    """Observe a latent-only chain after all round feedback has completed."""
    previous = {name: kwargs["model_input"] for name in kwargs["policy_names"]}
    for index, round_spec in enumerate(task.rounds, 1):
        previous = _decode_round(
            round_index=index,
            round_spec=round_spec,
            previous_images=previous,
            after_chain=True,
            **kwargs,
        )
    return len(task.rounds) * len(kwargs["policy_names"])


def _run_multi_image_task(
    *,
    task: Any,
    model: Any,
    config: dict[str, Any],
    image_root: Path,
    output: Path,
) -> dict[str, Any]:
    from PIL import Image
    from safetensors.torch import save_file

    inference_config = config["inference"]
    feedback = feedback_mode(inference_config.get("feedback"), default="latent")
    task_dir = output / "tasks" / task.task_id
    task_input = {
        "task": task.to_dict(),
        "model": config.get("model", {}),
        "schedule": config.get("schedule", {}),
        "inference": inference_config,
        "flux2_contract": {
            "numerics": "v2_bf16",
            "reference_preprocessing": "native_v1",
        },
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
    original_size = original.size
    reference_count = max(
        1 + len(round_spec.condition_images) for round_spec in task.rounds
    )
    model_input = model.prepare_inference_image(
        original,
        reference_count=reference_count,
    )
    infer_width, infer_height = model_input.size
    model_input.save(task_dir / "source.png")

    # Static pixel inputs are encoded once and shared across policies. Only image
    # feedback re-encodes the per-policy outputs inside the round loop.
    original_latent = model.encode_image(model_input)
    reference_latents: dict[str, Any] = {task.image: original_latent}
    reference_images: dict[str, Any] = {task.image: model_input}
    reference_shapes: dict[str, list[int]] = {task.image: list(original_latent.shape)}
    reference_paths = dict.fromkeys(
        path for round_spec in task.rounds for path in round_spec.condition_images
    )
    for path in reference_paths:
        if path in reference_latents:
            continue
        with Image.open(_safe_image(image_root, path)) as opened:
            prepared = model.prepare_inference_image(
                opened,
                reference_count=reference_count,
            )
        reference_images[path] = prepared
        reference_latents[path] = model.encode_image(prepared)
        reference_shapes[path] = list(reference_latents[path].shape)

    source_latent_path = task_dir / "source.safetensors"
    save_file(
        {"latent": original_latent.detach().cpu().contiguous()}, source_latent_path
    )
    source_latents: dict[str, Any] = {}
    task_records: list[dict[str, Any]] = []
    metrics_records: list[dict[str, Any]] = []
    policy_names: tuple[str, ...] | None = None
    previous_images: dict[str, Any] = {}
    feedback_encode_calls = 0
    round_decode_calls = 0

    for round_index, round_spec in enumerate(task.rounds, 1):
        policies = build_multi_image_policies(
            inference_config, num_steps=round_spec.steps
        )
        current_policy_names = tuple(policies)
        if policy_names is None:
            policy_names = current_policy_names
            previous_images = {name: model_input for name in policy_names}
        elif policy_names != current_policy_names:
            raise ValueError("policy names differ between rounds")
        round_latents: dict[str, Any] = {}
        round_results: dict[str, Any] = {}
        round_metadata: dict[str, Any] = {}
        static_latents = [
            reference_latents[path] for path in round_spec.condition_images
        ]
        schedule = _schedule(
            config,
            steps=round_spec.steps,
            height=infer_height,
            width=infer_width,
            strength=round_spec.denoising_strength,
        )
        for name, policy in policies.items():
            if round_index == 1:
                primary_latent = original_latent.clone()
            elif feedback == "image":
                primary_image = model.prepare_inference_image(
                    previous_images[name], reference_count=reference_count
                )
                if primary_image.size != (infer_width, infer_height):
                    raise RuntimeError("FLUX multi-image feedback changed resolution")
                primary_latent = model.encode_image(primary_image)
                feedback_encode_calls += 1
            else:
                primary_latent = source_latents[name]
            prepared = prepare_multi_image_conditioning(
                model,
                primary_latent=primary_latent,
                reference_latents=static_latents,
                prompts=[round_spec.prompt],
                height=infer_height,
                width=infer_width,
            )
            initial = model.initialize_latent(
                primary_latent,
                seed=round_spec.seed,
                sigma=schedule.point(0).sigma,
                noise_mode=round_spec.noise_mode,
            )
            result = run_multi_image_policy(
                model,
                initial_latent=initial,
                conditionings=prepared.profiles,
                schedule=schedule,
                policy=policy,
            )
            round_latents[name] = result.latent.detach()
            round_results[name] = result
            round_metadata[name] = prepared

        baseline_latent = round_latents.get("base")
        for name in policies:
            policy_dir = task_dir / name
            policy_dir.mkdir(exist_ok=True)
            latent_path = policy_dir / f"round-{round_index:02d}.safetensors"
            save_file({"latent": round_latents[name].cpu().contiguous()}, latent_path)
            metrics = latent_comparison_metrics(round_latents[name], baseline_latent)
            metrics_records.append({"round": round_index, "policy": name, **metrics})
            prepared = round_metadata[name]
            task_records.append(
                {
                    "round": round_index,
                    "policy": name,
                    "prompt": round_spec.prompt,
                    "primary_source": round_spec.primary_source,
                    "condition_images": list(round_spec.condition_images),
                    "reference_roles": [
                        dict(item) for item in round_spec.reference_roles
                    ],
                    "constraints_hard": list(round_spec.constraints_hard),
                    "constraints_soft": list(round_spec.constraints_soft),
                    "checkpoints": [dict(item) for item in round_spec.checkpoints],
                    "seed": round_spec.seed,
                    "actors": list(round_results[name].actors),
                    "conditioning_contexts": list(round_results[name].contexts),
                    "condition_token_lengths": list(prepared.condition_token_lengths),
                    "condition_time_slots": list(prepared.condition_time_slots),
                    "elapsed_seconds": round_results[name].elapsed_seconds,
                    "latent": str(latent_path.relative_to(output)),
                    "metrics_vs_pretrained": metrics,
                }
            )
        source_latents = round_latents
        if feedback == "image":
            previous_images = _decode_round(
                model=model,
                round_index=round_index,
                round_spec=round_spec,
                task_dir=task_dir,
                output=output,
                model_input=model_input,
                task_records=task_records,
                metrics_records=metrics_records,
                policy_names=policy_names,
                infer_height=infer_height,
                infer_width=infer_width,
                previous_images=previous_images,
                after_chain=False,
            )
            round_decode_calls += len(policy_names)

    assert policy_names is not None
    comparison_dir = task_dir / "comparison"
    comparison_dir.mkdir(exist_ok=True)
    source_panel_path = comparison_dir / "source.png"
    source_comparison_panel(model_input, policy_names).save(source_panel_path)
    post_chain_decode_calls = (
        _decode_chain_once(
            model=model,
            task=task,
            task_dir=task_dir,
            output=output,
            model_input=model_input,
            task_records=task_records,
            metrics_records=metrics_records,
            policy_names=policy_names,
            infer_height=infer_height,
            infer_width=infer_width,
        )
        if feedback == "latent"
        else 0
    )
    input_encode_calls = len(reference_latents)
    scale = max(infer_width / original.width, infer_height / original.height)
    retained_fraction = (infer_width * infer_height) / (
        original.width * scale * original.height * scale
    )
    task_manifest = {
        "id": task.task_id,
        "task_type": task.task_type,
        "goal": task.goal,
        "input": task_input,
        "source": str((task_dir / "source.png").relative_to(output)),
        "source_latent": str(source_latent_path.relative_to(output)),
        "source_comparison": str(source_panel_path.relative_to(output)),
        "original_width": original_size[0],
        "original_height": original_size[1],
        "inference_width": infer_width,
        "inference_height": infer_height,
        "resolution": {
            "policy": "flux2_native_reference_geometry",
            "original_size": list(original_size),
            "primary_size": [infer_width, infer_height],
            "reference_count": reference_count,
            "max_pixels_per_reference": flux2_reference_max_pixels(reference_count),
            "crop_fraction": max(0.0, 1.0 - retained_fraction),
            "aspect_ratio_preserved": True,
        },
        "reference_latent_shapes": reference_shapes,
        "feedback": feedback,
        "comparison_layout": COMPARISON_LAYOUT,
        "vae_usage": {
            "input_encode_calls": input_encode_calls,
            "unique_input_images": input_encode_calls,
            "inter_round_encode_calls": feedback_encode_calls,
            "inter_round_decode_calls": max(0, round_decode_calls - len(policy_names)),
            "post_chain_decode_calls": post_chain_decode_calls,
            "each_generated_state_decoded_once": True,
            "strict_latent_only_chain": feedback == "latent",
            **(
                {
                    "per_round_decode_calls": round_decode_calls,
                    "total_encode_calls": input_encode_calls + feedback_encode_calls,
                }
                if feedback == "image"
                else {}
            ),
        },
        "chain_rounds": True,
        "results": task_records,
        "metrics": metrics_records,
    }
    (task_dir / "task.json").write_text(
        json.dumps(task_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (task_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return task_manifest


def run_multi_image_inference(
    config: dict[str, Any],
    *,
    resume: bool = False,
    shard_index: int = 0,
    num_shards: int = 1,
) -> dict[str, Any]:
    import torch

    if not 0 <= shard_index < num_shards:
        raise ValueError("invalid inference shard index/count")
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(env_rank)))
    distributed = env_world_size > 1
    if distributed and (shard_index, num_shards) != (env_rank, env_world_size):
        raise ValueError("torchrun rank/world size disagree with shard arguments")

    inference = config["inference"]
    feedback = feedback_mode(inference.get("feedback"), default="latent")
    if feedback not in {"latent", "image"}:
        raise ValueError("multi-image inference requires latent or image feedback")
    image_feedback = feedback == "image"
    if bool(inference.get("decode_intermediate", image_feedback)) != image_feedback:
        raise ValueError("multi-image decode_intermediate must match image feedback")
    expected_decode = "none" if image_feedback else "all_rounds"
    if str(inference.get("decode_after_chain", expected_decode)) != expected_decode:
        raise ValueError(
            f"multi-image {feedback} requires decode_after_chain={expected_decode}"
        )
    default_steps = int(inference.get("num_steps", 50))
    multi_config = inference.get("multi_condition", {})
    tasks = load_multi_image_tasks(
        inference["tasks"],
        default_steps=default_steps,
        default_seed=int(config.get("seed", 0)),
        max_static_references=int(multi_config.get("max_static_references", 2)),
    )
    selected_ids = inference.get("task_ids")
    if selected_ids is not None:
        selected = {str(value) for value in selected_ids}
        unknown = selected - {task.task_id for task in tasks}
        if unknown:
            raise ValueError(f"unknown multi-image task ids: {sorted(unknown)}")
        tasks = tuple(task for task in tasks if task.task_id in selected)
        if not tasks:
            raise ValueError("task_ids selected no tasks")
    image_root = Path(inference["image_root"]).expanduser().resolve()
    for task in tasks:
        _safe_image(image_root, task.image)
        for round_spec in task.rounds:
            for path in round_spec.condition_images:
                _safe_image(image_root, path)

    output = Path(config["run"]["output_dir"]).expanduser().resolve()
    repository = Path(__file__).resolve().parents[3]
    if output == repository or repository in output.parents:
        raise ValueError("inference results must live outside the TailSteer repository")

    dist = None
    process_group_owned = False
    if distributed:
        import torch.distributed as torch_dist

        dist = torch_dist
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo")
            process_group_owned = True

    setup_error: list[str | None] = [None]
    if not distributed or env_rank == 0:
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

                saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                if saved != config:
                    raise ValueError("resume config differs from saved config")
            else:
                save_resolved_config(config, output)
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
        rebuild_error: list[str | None] = [None]
        if not distributed or env_rank == 0:
            try:
                rebuild_output_comparison_panels(output)
            except Exception as error:
                if not distributed:
                    raise
                rebuild_error[0] = f"{type(error).__name__}: {error}"
        if distributed:
            assert dist is not None
            dist.broadcast_object_list(rebuild_error, src=0)
            if rebuild_error[0] is not None:
                if process_group_owned:
                    dist.destroy_process_group()
                raise RuntimeError(rebuild_error[0])
            dist.barrier()
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if distributed and process_group_owned:
            assert dist is not None
            dist.destroy_process_group()
        return manifest

    shards = balanced_multi_image_task_indices(tasks, num_shards)
    assigned = [tasks[index] for index in shards[shard_index]]
    policies = build_multi_image_policies(inference, num_steps=default_steps)
    needs_pretrained = any(
        segment.actor == "base"
        for policy in policies.values()
        for segment in policy.segments
    )
    needs_refined = any(
        segment.actor == "refined"
        for policy in policies.values()
        for segment in policy.segments
    )
    if needs_refined and not inference.get("export_path"):
        raise ValueError("a refined policy requires inference.export_path")
    device_index = (
        local_rank
        if distributed
        else int(os.environ.get("LOCAL_RANK", inference.get("device", 0)))
    )
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    model = create_model(
        config["model"],
        device=device,
        include_vae=True,
        independent_actors=needs_pretrained and needs_refined,
        export_path=inference.get("export_path") if needs_refined else None,
    )
    task_manifests = [
        _run_multi_image_task(
            task=task,
            model=model,
            config=config,
            image_root=image_root,
            output=output,
        )
        for task in assigned
    ]
    manifest = {
        "schema_version": 1,
        "product": f"flux2_multi_image_{feedback}_chain_comparison",
        "feedback": feedback,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "comparison_layout": COMPARISON_LAYOUT,
        "policies": {name: policy.to_dict() for name, policy in policies.items()},
        "tasks": task_manifests,
    }
    shard_path = output / f"manifest-shard-{shard_index:05d}.json"
    shard_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / f"COMPLETE-shard-{shard_index:05d}").write_text(
        "complete\n", encoding="utf-8"
    )
    if distributed:
        assert dist is not None
        dist.barrier()
        if env_rank == 0:
            merge_multi_image_inference_shards(output, num_shards=num_shards)
        dist.barrier()
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if process_group_owned:
            dist.destroy_process_group()
        return manifest
    if num_shards == 1:
        return merge_multi_image_inference_shards(output, num_shards=1)
    return manifest


def merge_multi_image_inference_shards(
    output_dir: str | Path, *, num_shards: int
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    shards = []
    for index in range(num_shards):
        if not (output / f"COMPLETE-shard-{index:05d}").is_file():
            raise RuntimeError(f"multi-image inference shard {index} is incomplete")
        shards.append(
            json.loads(
                (output / f"manifest-shard-{index:05d}.json").read_text(
                    encoding="utf-8"
                )
            )
        )
    tasks = [task for shard in shards for task in shard["tasks"]]
    if any(shard["num_shards"] != num_shards for shard in shards):
        raise ValueError("inference shards disagree on shard count")
    if {shard["shard_index"] for shard in shards} != set(range(num_shards)):
        raise ValueError("inference shard indices are incomplete or duplicated")
    if any(shard["policies"] != shards[0]["policies"] for shard in shards[1:]):
        raise ValueError("inference shards used different policies")
    if any(shard["product"] != shards[0]["product"] for shard in shards[1:]):
        raise ValueError("inference shards used different feedback modes")
    if len({task["id"] for task in tasks}) != len(tasks):
        raise ValueError("inference shards contain duplicate task ids")
    manifest = {
        "schema_version": 2,
        "product": shards[0]["product"],
        "feedback": shards[0].get("feedback", "latent"),
        "num_shards": num_shards,
        "comparison_layout": COMPARISON_LAYOUT,
        "strategy": "whole_chain_balanced_by_total_input_image_count",
        "rank_round_counts": [
            sum(
                len(task["results"]) // len(shards[0]["policies"])
                for task in shard["tasks"]
            )
            for shard in shards
        ],
        "policies": shards[0]["policies"],
        "tasks": sorted(tasks, key=lambda item: item["id"]),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "COMPLETE").write_text("complete\n", encoding="utf-8")
    rebuild_output_comparison_panels(output)
    return json.loads((output / "manifest.json").read_text(encoding="utf-8"))
