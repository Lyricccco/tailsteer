"""FLUX.2 base/TailSteer/transition comparison with isolated chains."""

from __future__ import annotations

import json
import os
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Any

from tailsteer.models import create_model
from tailsteer.models.flux2 import build_flux2_schedule
from tailsteer.models.flux2.inference import run_endpoint, run_policy
from tailsteer.utils import save_resolved_config

from .endpoint import plans_from_mapping
from .feedback import feedback_mode
from .metrics import (
    comparison_metrics,
    image_comparison_metrics,
    latent_comparison_metrics,
)
from .multiround import balanced_task_indices, load_tasks
from .panels import (
    COMPARISON_LAYOUT,
    output_comparison_panel,
    rebuild_output_comparison_panels,
    source_comparison_panel,
)
from .policies import build_policies

_PROCESS_GROUP_TIMEOUT = timedelta(hours=6)


def _safe_image(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"task image escapes image_root: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _schedule(
    config: dict[str, Any], *, steps: int, height: int, width: int, strength: float
):
    schedule = config.get("schedule", {})
    policy = str(schedule.get("shift_policy", "fixed_reference_shape"))
    if policy == "fixed_reference_shape":
        schedule_height = int(schedule.get("reference_height", 1024))
        schedule_width = int(schedule.get("reference_width", 1024))
    elif policy in {"native", "actual_resolution"}:
        schedule_height, schedule_width = height, width
    else:
        raise ValueError("schedule.shift_policy is invalid")
    return build_flux2_schedule(
        num_steps=steps,
        height=schedule_height,
        width=schedule_width,
        denoising_strength=strength,
    )


def _decode_saved_task_latents(
    *,
    model: Any,
    task_dir: Path,
    output: Path,
    model_input: Any,
    task_records: list[dict[str, Any]],
    metrics_records: list[dict[str, Any]],
    policy_names: tuple[str, ...],
    infer_height: int,
    infer_width: int,
    decode_policy: str,
    reuse_existing: bool = False,
) -> int:
    """Decode observations after a latent chain without feeding them back."""

    from PIL import Image
    from safetensors.torch import load_file

    if decode_policy not in {"all_rounds", "final_only", "none"}:
        raise ValueError(
            "inference.decode_after_chain must be all_rounds, final_only, or none"
        )
    if decode_policy == "none" or not task_records:
        return 0
    final_round = max(int(item["round"]) for item in task_records)
    selected_rounds = (
        range(1, final_round + 1) if decode_policy == "all_rounds" else (final_round,)
    )
    previous_images = {name: model_input for name in policy_names}
    decode_calls = 0
    for round_index in selected_rounds:
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
            latent_path = output / record["latent"]
            latents[name] = load_file(str(latent_path), device="cpu")["latent"]
            image_path = task_dir / name / f"round-{round_index:02d}.png"
            image = None
            if reuse_existing and image_path.is_file():
                try:
                    with Image.open(image_path) as opened:
                        image = opened.convert("RGB")
                except (OSError, ValueError):
                    # Interrupted PNG writes are observations, not chain states.
                    # Recover from the persisted latent without another rollout.
                    pass
            if image is None:
                image = model.decode_latent(
                    latents[name], height=infer_height, width=infer_width
                )
                temporary = image_path.with_suffix(".tmp.png")
                image.save(temporary)
                temporary.replace(image_path)
                decode_calls += 1
            images[name] = image
            record["image"] = str(image_path.relative_to(output))
            record["decoded_after_chain"] = True

        baseline_latent = latents.get("base")
        baseline_image = images.get("base")
        for name in policy_names:
            record = records[name]
            metrics = comparison_metrics(
                latents[name], baseline_latent, images[name], baseline_image
            )
            record["metrics_vs_pretrained"] = metrics
            record["metrics_vs_input"] = image_comparison_metrics(
                images[name], previous_images[name]
            )
            record["metrics_vs_source"] = image_comparison_metrics(
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
        prompts = {str(record["prompt"]) for record in records.values()}
        if len(prompts) != 1:
            raise ValueError(f"round {round_index} policies used different prompts")
        output_comparison_panel(
            images,
            policy_names,
            prompt=prompts.pop(),
        ).save(panel_path)
        for record in records.values():
            record["comparison_panel"] = str(panel_path.relative_to(output))
        previous_images = images
    return decode_calls


def _load_saved_round_latents(
    *,
    task_dir: Path,
    policy_names: tuple[str, ...],
    round_index: int,
    reference: Any,
) -> dict[str, Any] | None:
    """Load one fully persisted round, or mark it for recomputation."""

    from safetensors.torch import load_file

    paths = {
        name: task_dir / name / f"round-{round_index:02d}.safetensors"
        for name in policy_names
    }
    if not all(path.is_file() for path in paths.values()):
        return None
    try:
        return {
            name: load_file(str(path), device="cpu")["latent"].to(
                device=reference.device,
                dtype=reference.dtype,
            )
            for name, path in paths.items()
        }
    except Exception:
        # A process may have been interrupted while writing the last file.
        return None


def _run_task(
    *,
    task: Any,
    model: Any,
    config: dict[str, Any],
    image_root: Path,
    output: Path,
    resume: bool = False,
) -> dict[str, Any]:
    from PIL import Image
    from safetensors.torch import save_file

    task_dir = output / "tasks" / task.task_id
    task_input = {
        "task": {
            "id": task.task_id,
            "image": task.image,
            "chain_rounds": task.chain_rounds,
            "rounds": [round_spec.__dict__ for round_spec in task.rounds],
        },
        "model": config.get("model", {}),
        "schedule": config.get("schedule", {}),
        "inference": config["inference"],
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
    if task_dir.exists() and not resume:
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(_safe_image(image_root, task.image)) as opened:
        original = opened.convert("RGB")
    original_size = original.size
    display_source = model.prepare_inference_image(
        original,
        reference_count=1,
    )
    infer_width, infer_height = display_source.size
    display_source.save(task_dir / "source.png")
    scale = max(infer_width / original.width, infer_height / original.height)
    retained_fraction = (infer_width * infer_height) / (
        original.width * scale * original.height * scale
    )
    # The saved source is exactly the PIL image seen by the VAE. Geometry is
    # prepared once; the Diffusers image processor only converts/normalizes it.
    original_latent = model.encode_image(display_source)
    source_latent_path = task_dir / "source.safetensors"
    save_file(
        {"latent": original_latent.detach().cpu().contiguous()}, source_latent_path
    )
    inference_config = config["inference"]
    feedback = feedback_mode(inference_config.get("feedback"), default="latent")
    if feedback not in {"image", "latent"}:
        raise ValueError("inference.feedback must be image or latent")
    decode_intermediate = bool(
        inference_config.get("decode_intermediate", feedback == "image")
    )
    if feedback == "image" and not decode_intermediate:
        raise ValueError("image feedback requires inference.decode_intermediate=true")
    decode_after_chain = str(
        inference_config.get(
            "decode_after_chain", "all_rounds" if feedback == "latent" else "none"
        )
    )
    if decode_after_chain not in {"all_rounds", "final_only", "none"}:
        raise ValueError(
            "inference.decode_after_chain must be all_rounds, final_only, or none"
        )
    source_latents: dict[str, Any] = {}
    previous_images: dict[str, Any] = {}
    task_records = []
    metrics_records = []
    vae_decode_calls = 0
    inter_round_vae_feedback_calls = 0
    reuse_latent_prefix = resume and feedback == "latent"

    for round_index, round_spec in enumerate(task.rounds, 1):
        execution = inference_config.get("execution")
        endpoint_mode = (
            isinstance(execution, dict) and execution.get("kind") == "endpoint"
        )
        if endpoint_mode:
            plans = plans_from_mapping(
                execution.get("plans", {}), num_steps=round_spec.steps
            )
            policies = {}
            policy_names = tuple(plans)
        else:
            plans = {}
            policies = build_policies(inference_config, num_steps=round_spec.steps)
            policy_names = tuple(policies)
        should_decode = feedback == "image"
        round_latents, round_output_images, round_info = (
            {},
            {},
            {},
        )
        round_inputs: dict[str, Any | None] = {}
        saved_latents = (
            _load_saved_round_latents(
                task_dir=task_dir,
                policy_names=policy_names,
                round_index=round_index,
                reference=original_latent,
            )
            if reuse_latent_prefix
            else None
        )
        if saved_latents is not None:
            round_latents = saved_latents
            reused_round = True
        else:
            reuse_latent_prefix = False
            reused_round = False
            for name, policy in plans.items() if endpoint_mode else policies.items():
                if round_index == 1 or not task.chain_rounds:
                    source_latent = original_latent.clone()
                    round_inputs[name] = display_source
                elif feedback == "image":
                    round_inputs[name] = model.prepare_inference_image(
                        previous_images[name],
                        reference_count=1,
                    )
                    if round_inputs[name].size != (infer_width, infer_height):
                        raise RuntimeError(
                            "FLUX.2 image feedback changed inference resolution"
                        )
                    source_latent = model.encode_image(round_inputs[name])
                    inter_round_vae_feedback_calls += 1
                else:
                    round_inputs[name] = None
                    source_latent = source_latents[name]
                schedule = _schedule(
                    config,
                    steps=round_spec.steps,
                    height=infer_height,
                    width=infer_width,
                    strength=round_spec.denoising_strength,
                )
                conditioning = model.prepare_conditioning(
                    source_latent=source_latent,
                    prompts=[round_spec.prompt],
                    height=infer_height,
                    width=infer_width,
                )
                initial = model.initialize_latent(
                    source_latent,
                    seed=round_spec.seed,
                    sigma=schedule.point(0).sigma,
                    noise_mode=round_spec.noise_mode,
                )
                result = (
                    run_endpoint(
                        model,
                        initial_latent=initial,
                        conditioning=conditioning,
                        schedule=schedule,
                        plan=policy,
                    )
                    if endpoint_mode
                    else run_policy(
                        model,
                        initial_latent=initial,
                        conditioning=conditioning,
                        schedule=schedule,
                        policy=policy,
                    )
                )
                if endpoint_mode and len(result.actors) != policy.expected_calls:
                    raise RuntimeError(
                        f"{name} executed {len(result.actors)} actors, expected "
                        f"{policy.expected_calls}"
                    )
                round_latents[name] = result.latent.detach()
                if should_decode:
                    round_output_images[name] = model.decode_latent(
                        result.latent, height=infer_height, width=infer_width
                    )
                    vae_decode_calls += 1
                round_info[name] = result
        baseline_latent = round_latents.get("base")
        baseline_image = round_output_images.get("base")
        for name, policy in plans.items() if endpoint_mode else policies.items():
            policy_dir = task_dir / name
            policy_dir.mkdir(exist_ok=True)
            image_path = policy_dir / f"round-{round_index:02d}.png"
            latent_path = policy_dir / f"round-{round_index:02d}.safetensors"
            if not reused_round:
                save_file(
                    {"latent": round_latents[name].cpu().contiguous()}, latent_path
                )
            metrics = latent_comparison_metrics(round_latents[name], baseline_latent)
            if should_decode:
                round_output_images[name].save(image_path)
                metrics = comparison_metrics(
                    round_latents[name],
                    baseline_latent,
                    round_output_images[name],
                    baseline_image,
                )
            metrics_records.append({"round": round_index, "policy": name, **metrics})
            record = {
                "round": round_index,
                "policy": name,
                "prompt": round_spec.prompt,
                "seed": round_spec.seed,
                "actors": list(
                    round_info[name].actors
                    if name in round_info
                    else (
                        list(policy.actors)
                        if endpoint_mode
                        else tuple(
                            policy.actor_at(step) for step in range(policy.num_steps)
                        )
                    )
                ),
                "execution": (
                    list(round_info[name].spans)
                    if endpoint_mode and name in round_info
                    else (
                        [span.to_dict() for span in plans[name].spans]
                        if endpoint_mode
                        else None
                    )
                ),
                "elapsed_seconds": (
                    round_info[name].elapsed_seconds if name in round_info else None
                ),
                "latent": str(latent_path.relative_to(output)),
                "metrics_vs_pretrained": metrics,
            }
            if should_decode:
                record["image"] = str(image_path.relative_to(output))
                record["decoded_after_chain"] = feedback == "latent"
                input_image = round_inputs[name]
                if input_image is not None:
                    if input_image.size != (infer_width, infer_height):
                        input_image = display_source
                    record["metrics_vs_input"] = image_comparison_metrics(
                        round_output_images[name], input_image
                    )
                if feedback == "latent":
                    record["metrics_vs_source"] = image_comparison_metrics(
                        round_output_images[name], display_source
                    )
            task_records.append(record)
        if should_decode:
            comparison_dir = task_dir / "comparison"
            comparison_dir.mkdir(exist_ok=True)
            panel_path = comparison_dir / f"round-{round_index:02d}.png"
            output_comparison_panel(
                round_output_images,
                policy_names,
                prompt=round_spec.prompt,
            ).save(panel_path)
            for item in task_records[-len(policies) :]:
                item["comparison_panel"] = str(panel_path.relative_to(output))
        source_latents = round_latents
        if feedback == "image":
            previous_images = round_output_images
    comparison_dir = task_dir / "comparison"
    comparison_dir.mkdir(exist_ok=True)
    source_panel_path = comparison_dir / "source.png"
    source_comparison_panel(display_source, policy_names).save(source_panel_path)
    post_chain_decode_calls = 0
    if feedback == "latent":
        post_chain_decode_calls = _decode_saved_task_latents(
            model=model,
            task_dir=task_dir,
            output=output,
            model_input=display_source,
            task_records=task_records,
            metrics_records=metrics_records,
            policy_names=policy_names,
            infer_height=infer_height,
            infer_width=infer_width,
            decode_policy=decode_after_chain,
            reuse_existing=resume,
        )
        vae_decode_calls += post_chain_decode_calls
    task_manifest = {
        "id": task.task_id,
        "input": task_input,
        "source": str((task_dir / "source.png").relative_to(output)),
        "source_latent": str(source_latent_path.relative_to(output)),
        "source_comparison": str(source_panel_path.relative_to(output)),
        "original_width": original_size[0],
        "original_height": original_size[1],
        "inference_width": infer_width,
        "inference_height": infer_height,
        "resolution": {
            "policy": "flux2_official_native_input",
            "original_size": list(original_size),
            "reference_size": [infer_width, infer_height],
            "generation_size": [infer_width, infer_height],
            "crop_fraction": max(0.0, 1.0 - retained_fraction),
            "aspect_ratio_preserved": True,
        },
        "feedback": feedback,
        "comparison_layout": COMPARISON_LAYOUT,
        "vae_usage": {
            "initial_encode_calls": 1,
            "decode_calls": vae_decode_calls,
            "decode_policy": (
                "every_round_with_image_feedback"
                if feedback == "image"
                else f"{decode_after_chain}_after_chain"
            ),
            "post_chain_decode_calls": post_chain_decode_calls,
            "inter_round_vae_feedback_calls": inter_round_vae_feedback_calls,
            "strict_no_inter_round_vae": inter_round_vae_feedback_calls == 0,
        },
        "chain_rounds": task.chain_rounds,
        "results": task_records,
        "metrics": metrics_records,
    }
    (task_dir / "task.json").write_text(
        json.dumps(task_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (task_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return task_manifest


def run_inference(
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
        raise ValueError(
            "torchrun rank/world size disagree with inference shard arguments"
        )

    inference = config["inference"]
    default_steps = int(inference.get("num_steps", 50))
    tasks = load_tasks(
        inference["tasks"],
        default_steps=default_steps,
        default_seed=int(config.get("seed", 0)),
    )
    image_root = Path(inference["image_root"]).expanduser().resolve()
    for task in tasks:
        _safe_image(image_root, task.image)

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
            dist.init_process_group(
                backend="gloo",
                timeout=_PROCESS_GROUP_TIMEOUT,
            )
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

                if yaml.safe_load(config_path.read_text(encoding="utf-8")) != config:
                    raise ValueError("resume config differs from the saved config")
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

    shards = balanced_task_indices(tasks, num_shards)
    assigned = [tasks[index] for index in shards[shard_index]]
    execution = inference.get("execution")
    endpoint_mode = isinstance(execution, dict) and execution.get("kind") == "endpoint"
    if endpoint_mode:
        plans = plans_from_mapping(execution.get("plans", {}), num_steps=default_steps)
        policies = {}
        needs_pretrained = any(
            span.actor == "base" for plan in plans.values() for span in plan.spans
        )
        needs_refined = any(
            span.actor == "refined" for plan in plans.values() for span in plan.spans
        )
    else:
        plans = {}
        policies = build_policies(inference, num_steps=default_steps)
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
        raise ValueError("a refined inference policy requires inference.export_path")
    device_index = (
        local_rank
        if distributed
        else int(os.environ.get("LOCAL_RANK", config["inference"].get("device", 0)))
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
        _run_task(
            task=task,
            model=model,
            config=config,
            image_root=image_root,
            output=output,
            resume=resume,
        )
        for task in assigned
    ]
    manifest = {
        "schema_version": 1,
        "product": "flux2_independent_multiround_comparison",
        "shard_index": shard_index,
        "num_shards": num_shards,
        "feedback": feedback_mode(inference.get("feedback"), default="latent"),
        "schedule": dict(config["schedule"]),
        "comparison_layout": COMPARISON_LAYOUT,
        "policies": (
            {name: plan.to_dict() for name, plan in plans.items()}
            if endpoint_mode
            else {name: policy.to_dict() for name, policy in policies.items()}
        ),
        "execution": "endpoint" if endpoint_mode else "native_policy",
        "evaluation": config.get("evaluation", {}),
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
            merge_inference_shards(output, num_shards=num_shards)
        dist.barrier()
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if process_group_owned:
            dist.destroy_process_group()
        return manifest
    if num_shards == 1:
        return merge_inference_shards(output, num_shards=1)
    return manifest


def merge_inference_shards(
    output_dir: str | Path, *, num_shards: int
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    shards = []
    for index in range(num_shards):
        if not (output / f"COMPLETE-shard-{index:05d}").is_file():
            raise RuntimeError(f"inference shard {index} is incomplete")
        shards.append(
            json.loads(
                (output / f"manifest-shard-{index:05d}.json").read_text(
                    encoding="utf-8"
                )
            )
        )
    tasks = [task for shard in shards for task in shard["tasks"]]
    if any(shard["num_shards"] != num_shards for shard in shards):
        raise ValueError("inference shards disagree on the shard count")
    if {shard["shard_index"] for shard in shards} != set(range(num_shards)):
        raise ValueError("inference shard indices are incomplete or duplicated")
    if any(shard["policies"] != shards[0]["policies"] for shard in shards[1:]):
        raise ValueError("inference shards used different policies")
    if any(shard["schedule"] != shards[0]["schedule"] for shard in shards[1:]):
        raise ValueError("inference shards used different schedules")
    if any(shard["feedback"] != shards[0]["feedback"] for shard in shards[1:]):
        raise ValueError("inference shards used different feedback modes")
    if any(
        shard.get("evaluation", {}) != shards[0].get("evaluation", {})
        for shard in shards[1:]
    ):
        raise ValueError("inference shards used different evaluation plans")
    if len({task["id"] for task in tasks}) != len(tasks):
        raise ValueError("inference shards contain duplicate task ids")
    manifest = {
        "schema_version": 2,
        "product": "flux2_independent_multiround_comparison",
        "num_shards": num_shards,
        "feedback": shards[0]["feedback"],
        "schedule": shards[0]["schedule"],
        "comparison_layout": COMPARISON_LAYOUT,
        "strategy": "whole_chain_balanced_by_round_count",
        "rank_round_counts": [
            sum(
                len(task["results"]) // len(shards[0]["policies"])
                for task in shard["tasks"]
            )
            for shard in shards
        ],
        "policies": shards[0]["policies"],
        "evaluation": shards[0].get("evaluation", {}),
        "tasks": sorted(tasks, key=lambda item: item["id"]),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "COMPLETE").write_text("complete\n", encoding="utf-8")
    rebuild_output_comparison_panels(output)
    return json.loads((output / "manifest.json").read_text(encoding="utf-8"))
