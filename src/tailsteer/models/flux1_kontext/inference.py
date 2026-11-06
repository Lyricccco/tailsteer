"""Stepwise actors around BFL's unchanged Kontext denoise function."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tailsteer.models.native_transformer import TransformerActors

from .model import KontextModel
from .official import KontextRuntime, decode_image
from .preprocessing import (
    PREPROCESSING,
    prepare_latent_inputs,
    prepare_shared_inputs,
    prepare_source,
    source_geometry,
)


def inference_schedule(inference, *, height, width):
    """Select schedule geometry independently of the actual image canvas."""
    from flux.sampling import get_schedule

    schedule = inference.get("schedule", {})
    policy = schedule.get("shift_policy", "fixed_reference_shape")
    if policy == "fixed_reference_shape":
        height = int(schedule.get("reference_height", 1024))
        width = int(schedule.get("reference_width", 1024))
    elif policy != "native":
        raise ValueError(f"unsupported Kontext shift_policy: {policy!r}")
    if min(height, width) <= 0 or height % 16 or width % 16:
        raise ValueError("Kontext schedule dimensions must be positive multiples of 16")
    return get_schedule(int(inference["num_steps"]), height * width // 256, shift=True)


@dataclass(frozen=True)
class KontextGenerationResult:
    state: Any
    image: Any
    trace: tuple[str, ...]
    continuous_image: Any = None
    native_state: Any = None


class KontextInferenceRuntime:
    model_name = "flux1_kontext_dev"
    product = "kontext_independent_multiround_comparison"
    preprocessing_contract = PREPROCESSING

    def __init__(self, runtime, actors, model):
        self.runtime, self.actors, self.model = runtime, actors, model

    @classmethod
    def load(cls, model, export_path, *, device):
        runtime = KontextRuntime.load(model, device=device)
        actors = TransformerActors(
            runtime.model,
            export_path,
            model_name=cls.model_name,
            export_format=KontextModel.export_format,
        )
        return cls(runtime, actors, model)

    prepare_source = staticmethod(prepare_source)
    source_geometry = staticmethod(source_geometry)

    def generate(self, source, *, prompt, seed, policy, inference):
        import torch

        feedback = inference.get("feedback_mode", "image")
        if feedback not in {"image", "latent"}:
            raise ValueError("Kontext feedback must be image or latent")
        input_preprocessed = bool(inference.get("input_preprocessed", False))
        runtime = self.runtime
        with runtime._lock, torch.inference_mode():
            if feedback == "latent" and isinstance(source, torch.Tensor):
                inputs, height, width = prepare_latent_inputs(
                    runtime, source, prompt=prompt, seed=seed
                )
            else:
                source = self.prepare_source(
                    source, input_preprocessed=input_preprocessed
                )
                inputs, height, width = prepare_shared_inputs(
                    runtime, source, prompt=prompt, seed=seed
                )
            state, trace = self._denoise(
                inputs, height=height, width=width, policy=policy, inference=inference
            )
            image = self.materialize(state) if feedback == "image" else None
        return KontextGenerationResult(
            state=state,
            image=image,
            trace=trace,
            native_state=state if feedback == "latent" else None,
        )

    @staticmethod
    def conditioning_actor_for(policy, *, num_steps):
        # T5/CLIP remain frozen for both transformer actors.
        return "base"

    def generate_text(self, *, prompt, seed, policy, config):
        """Use Kontext weights with BFL's text-only path, without fake references."""
        import torch
        from flux.sampling import get_noise, prepare

        height, width = int(config["height"]), int(config["width"])
        if min(height, width) <= 0 or height % 16 or width % 16:
            raise ValueError("Kontext T2I dimensions must be positive multiples of 16")
        inference = {"schedule": {"shift_policy": "native"}, **config}
        runtime = self.runtime
        with runtime._lock, torch.inference_mode():
            noise = get_noise(
                1, height, width, runtime.device, torch.bfloat16, int(seed)
            )
            inputs = prepare(runtime.t5, runtime.clip, noise, prompt)
            state, trace = self._denoise(
                inputs, height=height, width=width, policy=policy, inference=inference
            )
            return self.materialize(state), trace

    def _denoise(self, inputs, *, height, width, policy, inference):
        from flux.sampling import denoise, unpack

        sigmas = inference_schedule(inference, height=height, width=width)
        endpoint = inference.get("endpoint_plan")
        if endpoint and endpoint["name"] != "base":
            # A: native prefix, then one target-step jump to sigma=0.
            sigmas = sigmas[: int(endpoint["target_actor_index"]) + 1] + [0.0]
        with self.actors.use_policy(policy):
            handle = self.runtime.model.register_forward_hook(
                lambda module, args, output: self.actors.complete_step()
            )
            try:
                final = denoise(
                    self.runtime.model,
                    **inputs,
                    timesteps=sigmas,
                    guidance=float(self.model.get("guidance_scale", 2.5)),
                )
            finally:
                handle.remove()
            trace = self.actors.trace
        return unpack(final, height, width)[0].detach().cpu().contiguous(), trace

    def materialize(self, value):
        import torch

        with self.runtime._lock, torch.inference_mode():
            return decode_image(self.runtime.ae, value, device=self.runtime.device)

    def unload(self):
        self.runtime.unload()
