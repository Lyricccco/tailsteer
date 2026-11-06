"""FLUX.2 adapter for experiment-aware comparison inference."""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

from ..feedback import feedback_config, feedback_mode
from .base import InferenceBuildContext


class Flux2InferenceBackend:
    key = "flux2"
    model_names = frozenset({"flux2_klein_9b"})

    @staticmethod
    def actor_step_count(num_steps: int) -> int:
        return int(num_steps)

    @staticmethod
    def forward_step_to_actor_index(step: int) -> int:
        # FLUX cache forward points are already zero based: 0..49.
        return int(step)

    @staticmethod
    def validate_transition(
        *, start_step: int, supervision_step: int, num_steps: int
    ) -> None:
        if not 0 <= start_step < supervision_step:
            raise ValueError("invalid FLUX.2 transition boundaries")
        if supervision_step != num_steps - 1:
            raise ValueError(
                "FLUX.2 supervision_step must identify the final zero-based "
                f"model forward: {supervision_step} != {num_steps - 1}"
            )

    def build_config(self, context: InferenceBuildContext) -> dict[str, Any]:
        model = deepcopy(context.training_config["model"])
        t2i = context.suite.get("task_type") == "t2i"
        backend = dict(context.suite.get("backends", {}).get(self.key, {}))
        schedule = {
            "shift_policy": "native" if t2i else "fixed_reference_shape",
            "reference_height": 1024,
            "reference_width": 1024,
            **dict(backend.get("schedule", {})),
        }
        inference = {
            "tasks": str(context.suite["tasks"]),
            **({} if t2i else {"image_root": str(context.suite["image_root"])}),
            "export_path": str(context.export_path),
            "num_steps": int(context.suite["num_steps"]),
            "feedback": feedback_config("image"),
            **dict(backend.get("inference", {})),
            **deepcopy(context.inference_overrides),
        }
        # Policies and resolved experiment paths are governed by the planner,
        # not mutable suite/backend defaults.
        inference["export_path"] = str(context.export_path)
        if t2i:
            import hashlib
            from pathlib import Path
            from ..t2i_runner import load_t2i_tasks

            if context.execution is not None:
                raise ValueError("T2I requires stepwise policies")
            load_t2i_tasks(inference["tasks"])
            inference.pop("feedback", None)
            inference["task_type"] = "t2i"
            inference["tasks_sha256"] = hashlib.sha256(
                Path(inference["tasks"]).read_bytes()
            ).hexdigest()
        if context.execution is None:
            inference["policies"] = deepcopy(context.policies)
        else:
            inference["execution"] = deepcopy(context.execution)
        mode = feedback_mode(inference.get("feedback"), default="image")
        if mode not in {"latent", "image"}:
            raise ValueError(
                "FLUX.2 feedback must be latent or image; continuous_image and "
                "model_native are not implemented by its inference runner"
            )
        if not t2i:
            inference["feedback"] = feedback_config(mode)
            if "feedback" in context.inference_overrides or (
                "multi_condition" in inference and mode == "image"
            ):
                # A feedback override must also replace the suite's decode defaults.
                inference["decode_intermediate"] = mode == "image"
                inference["decode_after_chain"] = (
                    "none" if mode == "image" else "all_rounds"
                )
        if "multi_condition" in inference:
            if context.execution is not None:
                raise ValueError(
                    "Multi-image FLUX inference requires stepwise policies"
                )
            if mode == "latent" and (
                inference.get("decode_intermediate", False)
                or inference.get("decode_after_chain", "all_rounds") != "all_rounds"
            ):
                raise ValueError(
                    "Multi-image FLUX inference decodes all rounds after the chain"
                )
        self._validate_schedule(schedule)
        seed = context.suite.get("seed", context.training_config.get("seed", 0))
        return {
            "seed": int(seed),
            "model": model,
            "schedule": schedule,
            "inference": inference,
            "evaluation": deepcopy(context.evaluation),
            "run": {"output_dir": str(context.output_dir)},
        }

    @staticmethod
    def _validate_schedule(schedule: dict[str, Any]) -> None:
        policy = str(schedule.get("shift_policy", ""))
        if policy not in {"fixed_reference_shape", "native"}:
            raise ValueError(
                "FLUX.2 shift_policy must be fixed_reference_shape or native"
            )
        if policy == "fixed_reference_shape":
            height = int(schedule.get("reference_height", 0))
            width = int(schedule.get("reference_width", 0))
            if height <= 0 or width <= 0 or height % 16 or width % 16:
                raise ValueError(
                    "FLUX.2 reference schedule shape must be positive and "
                    "divisible by 16"
                )

    @staticmethod
    def run(config: dict[str, Any], *, resume: bool) -> dict[str, Any]:
        if config["inference"].get("task_type") == "t2i":
            from tailsteer.inference.t2i_runner import run_t2i_inference

            return run_t2i_inference(config, resume=resume)
        from tailsteer.inference.runner import run_inference

        if "multi_condition" in config["inference"]:
            from tailsteer.inference.multi_image_runner import (
                run_multi_image_inference,
            )

            run_inference = run_multi_image_inference
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        return run_inference(
            config,
            resume=resume,
            shard_index=rank,
            num_shards=world_size,
        )
