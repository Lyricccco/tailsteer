"""Inference-plan adapters for official transformer pipelines."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..feedback import feedback_config, feedback_mode
from .base import InferenceBuildContext


class NativeTransformerInferenceBackend:
    key = ""
    model_names: frozenset[str] = frozenset()
    default_feedback = "image"
    allowed_feedback = frozenset({"image"})
    schedule_multiple = 16
    default_shift_policy = "fixed_reference_shape"
    supports_t2i = False

    @staticmethod
    def actor_step_count(num_steps: int) -> int:
        return int(num_steps)

    @staticmethod
    def forward_step_to_actor_index(step: int) -> int:
        return int(step) - 1

    @staticmethod
    def validate_transition(
        *, start_step: int, supervision_step: int, num_steps: int
    ) -> None:
        if not 1 <= start_step < supervision_step:
            raise ValueError("invalid native-transformer transition boundaries")
        if supervision_step != num_steps:
            raise ValueError(
                "native-transformer supervision_step must identify the final "
                f"one-based forward: {supervision_step} != {num_steps}"
            )

    def build_config(self, context: InferenceBuildContext) -> dict[str, Any]:
        t2i = context.suite.get("task_type") == "t2i"
        if t2i and not self.supports_t2i:
            raise ValueError(f"{self.key} does not support T2I inference")
        model = deepcopy(context.training_config["model"])
        backend = dict(context.suite.get("backends", {}).get(self.key, {}))
        schedule = {
            "shift_policy": self.default_shift_policy,
            "reference_height": 1024,
            "reference_width": 1024,
            **dict(backend.get("schedule", {})),
        }
        inference = {
            "tasks": str(context.suite["tasks"]),
            **({} if t2i else {"image_root": str(context.suite["image_root"])}),
            "export_path": str(context.export_path),
            "num_steps": int(context.suite["num_steps"]),
            "actor_steps": self.actor_step_count(int(context.suite["num_steps"])),
            "feedback": feedback_config(self.default_feedback),
            **dict(backend.get("inference", {})),
            **deepcopy(context.inference_overrides),
        }
        inference["export_path"] = str(context.export_path)
        if "num_steps" in backend:
            # A shared task list can be evaluated on a different native schedule.
            inference["override_round_steps"] = True
        if context.execution is None:
            inference["policies"] = deepcopy(context.policies)
        else:
            inference["execution"] = deepcopy(context.execution)
        mode = feedback_mode(inference.get("feedback"), default=self.default_feedback)
        if mode not in self.allowed_feedback:
            raise ValueError(
                f"{self.key} feedback must be one of "
                f"{sorted(self.allowed_feedback)}, got {mode!r}"
            )
        inference["feedback"] = feedback_config(mode)
        if t2i:
            import hashlib
            from pathlib import Path
            from ..t2i_runner import load_t2i_tasks

            if context.execution is not None:
                raise ValueError("T2I requires stepwise policies")
            load_t2i_tasks(inference["tasks"])
            inference.pop("feedback")
            inference["task_type"] = "t2i"
            inference["tasks_sha256"] = hashlib.sha256(
                Path(inference["tasks"]).read_bytes()
            ).hexdigest()
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

    def _validate_schedule(self, schedule: dict[str, Any]) -> None:
        policy = str(schedule.get("shift_policy", ""))
        if policy not in {"fixed_reference_shape", "native"}:
            raise ValueError(
                f"{self.key} shift_policy must be fixed_reference_shape or native"
            )
        if policy == "fixed_reference_shape":
            height = int(schedule.get("reference_height", 0))
            width = int(schedule.get("reference_width", 0))
            if (
                height <= 0
                or width <= 0
                or height % self.schedule_multiple
                or width % self.schedule_multiple
            ):
                raise ValueError(
                    f"{self.key} reference schedule shape must be positive and "
                    f"divisible by {self.schedule_multiple}"
                )

    @staticmethod
    def run(config: dict[str, Any], *, resume: bool) -> dict[str, Any]:
        if config["inference"].get("task_type") == "t2i":
            from tailsteer.inference.t2i_runner import run_t2i_inference

            return run_t2i_inference(config, resume=resume)
        from tailsteer.inference.native_transformer_runner import (
            run_native_transformer_inference,
        )

        return run_native_transformer_inference(config, resume=resume)


class QwenImage21InferenceBackend(NativeTransformerInferenceBackend):
    key = "qwen21"
    model_names = frozenset({"qwen_image_21"})
    default_num_steps = 40
    schedule_multiple = 32
    default_shift_policy = "native"
    supports_t2i = True

    def build_config(self, context):
        from tailsteer.models.qwen_image_21.preprocessing import PREPROCESSING

        config = super().build_config(context)
        if config["inference"].get("task_type") != "t2i":
            config["inference"]["preprocessing"] = dict(PREPROCESSING)
        return config

    @staticmethod
    def forward_step_to_actor_index(step: int) -> int:
        return int(step)

    @staticmethod
    def validate_transition(*, start_step, supervision_step, num_steps):
        if not 0 <= start_step < supervision_step == num_steps - 1:
            raise ValueError("Qwen supervision must be the last zero-based forward")


class KontextInferenceBackend(NativeTransformerInferenceBackend):
    key = "kontext"
    model_names = frozenset({"flux1_kontext_dev"})
    default_num_steps = 30
    supports_t2i = True
    allowed_feedback = frozenset({"image", "latent"})

    def build_config(self, context):
        from tailsteer.models.flux1_kontext.preprocessing import PREPROCESSING

        config = super().build_config(context)
        if context.suite.get("task_type") == "t2i":
            config["inference"]["conditioning"] = "text_only_no_reference_v1"
            schedule = (
                context.suite.get("backends", {}).get(self.key, {}).get("schedule", {})
            )
            if "shift_policy" not in schedule:
                config["schedule"]["shift_policy"] = "native"
        else:
            config["inference"]["preprocessing"] = dict(PREPROCESSING)
        return config

    @staticmethod
    def forward_step_to_actor_index(step: int) -> int:
        return int(step)

    @staticmethod
    def validate_transition(*, start_step, supervision_step, num_steps):
        if not 0 <= start_step < supervision_step == num_steps - 1:
            raise ValueError("Kontext supervision must be the last zero-based forward")
