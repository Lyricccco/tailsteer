"""SenseNova adapter for experiment-aware comparison inference."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..feedback import feedback_config, feedback_mode
from .base import InferenceBuildContext


class SenseNovaInferenceBackend:
    key = "sensenova"
    model_names = frozenset(
        {
            "sensenova_u1_5_8b_mot",
            "sensenova_u1_5_8b_mot_preview",
        }
    )

    @staticmethod
    def actor_step_count(num_steps: int) -> int:
        return int(num_steps)

    @staticmethod
    def forward_step_to_actor_index(step: int) -> int:
        # The official SenseNova cache labels model forwards 1..50, while the
        # sampler actor hook observes the same calls at zero-based indices 0..49.
        return int(step) - 1

    @staticmethod
    def validate_transition(
        *, start_step: int, supervision_step: int, num_steps: int
    ) -> None:
        if not 1 <= start_step < supervision_step:
            raise ValueError("invalid SenseNova transition boundaries")
        if supervision_step != num_steps:
            raise ValueError(
                "SenseNova supervision_step must identify the final one-based "
                f"model forward: {supervision_step} != {num_steps}"
            )

    def build_config(self, context: InferenceBuildContext) -> dict[str, Any]:
        model = deepcopy(context.training_config["model"])
        t2i = context.suite.get("task_type") == "t2i"
        backend = dict(context.suite.get("backends", {}).get(self.key, {}))
        schedule = {
            "shift_policy": "fixed_reference_shape",
            "reference_height": 1024,
            "reference_width": 1024,
            **dict(backend.get("schedule", {})),
        }
        inference = {
            "tasks": str(context.suite["tasks"]),
            **({} if t2i else {"image_root": str(context.suite["image_root"])}),
            "export_path": str(context.export_path),
            "num_steps": int(context.suite["num_steps"]),
            "cfg_scale": float(model.get("cfg_scale", 4.0)),
            **(
                {} if t2i else {"img_cfg_scale": float(model.get("img_cfg_scale", 1.0))}
            ),
            "cfg_norm": str(model.get("cfg_norm", "none")),
            "timestep_shift": float(model.get("timestep_shift", 3.0)),
            **({} if t2i else {"feedback": feedback_config("model_native")}),
            **dict(backend.get("inference", {})),
            **deepcopy(context.inference_overrides),
        }
        inference["export_path"] = str(context.export_path)
        if context.execution is None:
            inference["policies"] = deepcopy(context.policies)
        else:
            inference["execution"] = deepcopy(context.execution)
        self._validate_schedule(schedule)
        seed = context.suite.get("seed", context.training_config.get("seed", 0))
        if t2i:
            import hashlib
            from pathlib import Path

            from ..t2i_runner import load_t2i_tasks

            if context.execution is not None:
                raise ValueError("T2I requires native stepwise policies")
            unsupported = {"feedback", "image_root", "img_cfg_scale"} & inference.keys()
            if unsupported:
                raise ValueError(
                    f"T2I has no image conditioning/feedback: {sorted(unsupported)}"
                )
            # Validate the dataset during dry-run too; its digest protects resume
            # against an in-place edit of the shared prompt file.
            load_t2i_tasks(inference["tasks"], default_seed=int(seed))
            inference["task_type"] = "t2i"
            inference["tasks_sha256"] = hashlib.sha256(
                Path(inference["tasks"]).read_bytes()
            ).hexdigest()
        else:
            mode = feedback_mode(inference.get("feedback"), default="model_native")
            if mode not in {"model_native", "image"}:
                raise ValueError("SenseNova feedback must be model_native or image")
            inference["feedback"] = feedback_config(mode)
            if "multi_condition" in inference:
                if context.execution is not None or mode != "image":
                    raise ValueError(
                        "SenseNova multi-image inference requires stepwise "
                        "policies and feedback=image"
                    )
                from ..multi_image_multiround import load_multi_image_tasks

                load_multi_image_tasks(
                    inference["tasks"],
                    default_steps=inference["num_steps"],
                    default_seed=int(seed),
                    max_static_references=int(
                        inference["multi_condition"].get("max_static_references", 2)
                    ),
                )
        return {
            "seed": int(seed),
            "model": model,
            "schedule": schedule,
            "inference": inference,
            "evaluation": deepcopy(context.evaluation),
            "run": {
                "output_dir": str(context.output_dir),
                "policy_subdir": False,
            },
        }

    @staticmethod
    def _validate_schedule(schedule: dict[str, Any]) -> None:
        policy = str(schedule.get("shift_policy", ""))
        if policy not in {"fixed_reference_shape", "native"}:
            raise ValueError(
                "SenseNova shift_policy must be fixed_reference_shape or native"
            )
        if policy == "fixed_reference_shape":
            height = int(schedule.get("reference_height", 0))
            width = int(schedule.get("reference_width", 0))
            if height <= 0 or width <= 0 or height % 32 or width % 32:
                raise ValueError(
                    "SenseNova reference schedule shape must be positive and "
                    "divisible by 32"
                )

    @staticmethod
    def run(config: dict[str, Any], *, resume: bool) -> dict[str, Any]:
        if config["inference"].get("task_type") == "t2i":
            from tailsteer.inference.t2i_runner import run_t2i_inference

            return run_t2i_inference(config, resume=resume)

        from tailsteer.inference.sensenova_runner import run_sensenova_inference

        return run_sensenova_inference(config, resume=resume)
