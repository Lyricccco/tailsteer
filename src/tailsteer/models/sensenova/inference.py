"""Pretrained/refined actor switching inside official SenseNova sampling."""

from __future__ import annotations

import importlib
from collections import deque
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .conditioning import PrefixKVAdapterConfig, attach_prefix_kv_adapter
from .export import SenseNovaExportSpec
from .training import (
    configure_sensenova_trainable,
    is_sensenova_generation_state,
    is_sensenova_prefix_adapter_state,
)


def _raw_state_name(export_name: str) -> str:
    if not export_name.startswith("model."):
        raise ValueError(f"SenseNova export key lacks model prefix: {export_name}")
    return export_name.removeprefix("model.")


@contextmanager
def _bounded_prefix_attention(
    raw_model: Any, *, method: str = "_t2i_prefix_forward"
) -> Any:
    """Bound eager attention's temporary matrix, without truncating text/KV.

    Only wraps the selected inference prefix calls. Denoising and training
    retain their attention paths. Query chunks still attend to all allowed keys.
    """

    import torch

    original_prefix = getattr(raw_model, method, None)
    if original_prefix is None:
        yield
        return
    module = importlib.import_module(type(raw_model.language_model.model).__module__)
    eager = module.eager_attention_forward

    def bounded(attention, query, key, value, attention_mask, **kwargs):
        chunk_size = 512
        if (
            query.shape[-2] <= chunk_size
            or torch.is_grad_enabled()
            or attention.training
            or kwargs.get("output_attentions", False)
        ):
            return eager(attention, query, key, value, attention_mask, **kwargs)
        outputs = []
        for start in range(0, query.shape[-2], chunk_size):
            stop = start + chunk_size
            mask = attention_mask
            if mask is not None and mask.shape[-2] != 1:
                mask = mask[..., start:stop, :]
            output, weights = eager(
                attention, query[..., start:stop, :], key, value, mask, **kwargs
            )
            del weights
            outputs.append(output)
        # Official eager output layout is [batch, query, heads, head_dim].
        return torch.cat(outputs, dim=1), None

    def prefix(*args, **kwargs):
        module.eager_attention_forward = bounded
        try:
            return original_prefix(*args, **kwargs)
        finally:
            module.eager_attention_forward = eager

    setattr(raw_model, method, prefix)
    try:
        yield
    finally:
        setattr(raw_model, method, original_prefix)


class _ActorState:
    actor = "base"


@dataclass(frozen=True)
class SenseNovaReferenceState:
    """Model-native source conditioning, before construction of prefix KV."""

    embeddings: Any
    grid_hw: Any
    width: int
    height: int


class _ActorParameter:
    @staticmethod
    def build(refined: Any, actor_state: _ActorState) -> Any:
        import torch

        class _Parametrization(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("refined", refined)
                self.actor_state = actor_state

            def forward(self, pretrained: Any) -> Any:
                return (
                    self.refined if self.actor_state.actor == "refined" else pretrained
                )

        return _Parametrization()


class SenseNovaGenerationActors:
    """Switch denoiser steps and source conditioning without a second model."""

    POLICIES = ("base", "late_step", "all_steps")

    def __init__(self, raw_model: Any, export_path: str | Path) -> None:
        import torch
        from ..checkpoint import open_weights
        from torch.nn.utils import parametrize

        self.raw_model = raw_model
        self.export = None
        self.export_path = None
        self.manifest = {}
        self.prefix_kv_adapter = None
        self.denoiser_actor_state = _ActorState()
        self.conditioning_actor_state = _ActorState()
        self.actor_state = self.denoiser_actor_state
        self._generation_parameters = []
        self._conditioning_parameters = []
        self._adapter_parameter_count = 0
        self._loaded_parameter_count = 0
        raw_model.requires_grad_(False)
        if export_path is not None:
            self.export = SenseNovaExportSpec.load(export_path)
            self.export_path = self.export.root
            self.manifest = self.export.manifest
            adapter_config = PrefixKVAdapterConfig.from_mapping(
                self.export.trainable.get("prefix_kv_adapter")
            )
            self.prefix_kv_adapter = attach_prefix_kv_adapter(raw_model, adapter_config)
            if self.prefix_kv_adapter is not None:
                self.prefix_kv_adapter.requires_grad_(False)
                self.prefix_kv_adapter.eval()
            parameters = dict(raw_model.named_parameters())
            expected = set(
                configure_sensenova_trainable(raw_model, self.export.trainable)
            )
            raw_model.requires_grad_(False)
            self.denoiser_actor_state = _ActorState()
            self.conditioning_actor_state = _ActorState()
            # Compatibility for callers that inspected the old generation-only state.
            self.actor_state = self.denoiser_actor_state
            self._generation_parameters: list[tuple[Any, str]] = []
            self._conditioning_parameters: list[tuple[Any, str]] = []
            self._adapter_parameter_count = 0
            with open_weights(
                self.export.weights, prefix=self.manifest.get("weight_prefix", "")
            ) as handle:
                export_keys = tuple(handle.keys())
                raw_names = {_raw_state_name(name) for name in export_keys}
                if raw_names != expected:
                    missing = sorted(expected - raw_names)
                    extra = sorted(raw_names - expected)
                    raise RuntimeError(
                        "SenseNova inference delta differs from its trainable policy: "
                        f"missing={missing[:3]} extra={extra[:3]}"
                    )
                for export_name in export_keys:
                    raw_name = _raw_state_name(export_name)
                    if raw_name not in parameters:
                        raise KeyError(
                            f"export tensor is not a model parameter: {raw_name}"
                        )
                    pretrained = parameters[raw_name]
                    refined = handle.get_tensor(export_name).to(
                        device=pretrained.device,
                        dtype=pretrained.dtype,
                    )
                    if refined.shape != pretrained.shape:
                        raise ValueError(f"export tensor shape mismatch: {raw_name}")
                    if is_sensenova_prefix_adapter_state(raw_name):
                        with torch.no_grad():
                            pretrained.copy_(refined)
                        self._adapter_parameter_count += 1
                        continue
                    parent_name, _, attribute = raw_name.rpartition(".")
                    parent = (
                        raw_model.get_submodule(parent_name)
                        if parent_name
                        else raw_model
                    )
                    actor_state = (
                        self.denoiser_actor_state
                        if is_sensenova_generation_state(raw_name)
                        else self.conditioning_actor_state
                    )
                    parametrize.register_parametrization(
                        parent,
                        attribute,
                        _ActorParameter.build(refined, actor_state),
                        unsafe=True,
                    )
                    target = (
                        self._generation_parameters
                        if is_sensenova_generation_state(raw_name)
                        else self._conditioning_parameters
                    )
                    target.append((parent, attribute))
            if not self._generation_parameters:
                raise RuntimeError("SenseNova actor export contains no parameters")
            self._loaded_parameter_count = len(export_keys)
        self._original_extract_feature = raw_model.extract_feature
        self._original_build_it2i_inputs = getattr(
            raw_model, "_build_it2i_inputs", None
        )
        self._original_prefix_forward = getattr(raw_model, "_it2i_prefix_forward", None)
        self._active = False
        self._policy = "base"
        self._num_steps = 0
        self._activation_step = 0
        self._actor_sequence: tuple[str, ...] = ()
        self._step_index = 0
        self._trace: list[str] = []
        self._reference_state: SenseNovaReferenceState | None = None
        self._conditioning_actor = "base"
        self._captured_reference_embeddings: Any | None = None
        self._pending_prefix_contexts: deque[Any | None] = deque()

        def actor_extract_feature(*args: Any, **kwargs: Any) -> Any:
            gen_model = bool(
                kwargs.get("gen_model", args[1] if len(args) > 1 else False)
            )
            if not gen_model and self._active and self._reference_state is not None:
                grid_hw = kwargs.get("grid_hw", args[2] if len(args) > 2 else None)
                if grid_hw is None:
                    raise RuntimeError("SenseNova reference call omitted grid_hw")
                expected = self._reference_state.grid_hw
                if tuple(grid_hw.shape) != tuple(expected.shape) or not bool(
                    grid_hw.to(device=expected.device).eq(expected).all()
                ):
                    raise RuntimeError(
                        "model-native reference grid differs from generation input"
                    )
                result = self._reference_state.embeddings
                self._captured_reference_embeddings = result
                return result
            if gen_model and self._active:
                if self._step_index >= self._num_steps:
                    raise RuntimeError(
                        "SenseNova sampler executed too many denoise steps"
                    )
                actor = self._actor_for_step(self._step_index)
                self.denoiser_actor_state.actor = actor
                self._trace.append(actor)
                self._step_index += 1
            result = self._original_extract_feature(*args, **kwargs)
            if not gen_model and self._active:
                self._captured_reference_embeddings = result
            return result

        raw_model.extract_feature = actor_extract_feature

        if self.prefix_kv_adapter is not None:
            if not callable(self._original_build_it2i_inputs) or not callable(
                self._original_prefix_forward
            ):
                raise RuntimeError(
                    "SenseNova prefix adapter requires official prefix methods"
                )

            def actor_build_it2i_inputs(*args: Any, **kwargs: Any) -> Any:
                self._captured_reference_embeddings = None
                result = self._original_build_it2i_inputs(*args, **kwargs)
                if not self._active:
                    return result
                tokenizer = kwargs.get("tokenizer", args[0] if args else None)
                query = kwargs.get("query", args[1] if len(args) > 1 else None)
                pixel_values = kwargs.get(
                    "pixel_values", args[2] if len(args) > 2 else None
                )
                context = None
                if pixel_values is not None:
                    if tokenizer is None or query is None:
                        raise RuntimeError(
                            "SenseNova prefix input omitted tokenizer or query"
                        )
                    reference = self._captured_reference_embeddings
                    if reference is None:
                        raise RuntimeError(
                            "SenseNova prefix input did not encode its reference"
                        )
                    input_ids = tokenizer(query, return_tensors="pt")["input_ids"].to(
                        raw_model.device
                    )
                    image_mask = input_ids.reshape(-1).eq(
                        raw_model.img_context_token_id
                    )
                    context = (image_mask, reference)
                self._pending_prefix_contexts.append(context)
                return result

            def actor_prefix_forward(*args: Any, **kwargs: Any) -> Any:
                result = self._original_prefix_forward(*args, **kwargs)
                if not self._active:
                    return result
                if not self._pending_prefix_contexts:
                    raise RuntimeError(
                        "SenseNova prefix forward has no matching input context"
                    )
                context = self._pending_prefix_contexts.popleft()
                if context is not None and self._conditioning_actor == "refined":
                    image_mask, reference = context
                    self.prefix_kv_adapter.apply(result[0], image_mask, reference)
                return result

            raw_model._build_it2i_inputs = actor_build_it2i_inputs
            raw_model._it2i_prefix_forward = actor_prefix_forward

    @property
    def parameter_count(self) -> int:
        return self._loaded_parameter_count

    @property
    def generation_parameter_count(self) -> int:
        return len(self._generation_parameters)

    @property
    def conditioning_parameter_count(self) -> int:
        return len(self._conditioning_parameters) + self._adapter_parameter_count

    def _actor_for_step(self, step_index: int) -> str:
        if self._actor_sequence:
            return self._actor_sequence[step_index]
        if self._policy == "base":
            return "base"
        if self._policy == "all_steps":
            return "refined"
        return "refined" if step_index >= self._activation_step else "base"

    def conditioning_actor_for(self, policy: Any, *, num_steps: int) -> str:
        """Choose the once-per-round source actor for one denoiser policy."""

        if not (
            (self.export is not None and self.export.has_reference_actor)
            or (self.export is not None and self.export.has_prefix_adapter)
        ):
            return "base"
        if isinstance(policy, str):
            if policy not in self.POLICIES:
                raise ValueError(f"unsupported SenseNova actor policy {policy!r}")
            return "base" if policy == "base" else "refined"
        policy_steps = int(getattr(policy, "num_steps", -1))
        actor_at = getattr(policy, "actor_at", None)
        if policy_steps != num_steps or not callable(actor_at):
            raise TypeError(
                "segmented SenseNova policies require actor_at() and a "
                "matching num_steps"
            )
        return (
            "refined"
            if any(str(actor_at(step)) == "refined" for step in range(num_steps))
            else "base"
        )

    @contextmanager
    def use_conditioning_actor(self, actor: str) -> Any:
        if actor not in {"base", "refined"}:
            raise ValueError(f"invalid SenseNova conditioning actor {actor!r}")
        if self._active:
            raise RuntimeError(
                "cannot encode an external reference during active sampling"
            )
        previous = self.conditioning_actor_state.actor
        self.conditioning_actor_state.actor = actor
        try:
            yield
        finally:
            self.conditioning_actor_state.actor = previous

    def begin(
        self,
        policy: Any,
        *,
        num_steps: int,
        activation_step: int | None = None,
        reference_state: SenseNovaReferenceState | None = None,
    ) -> None:
        if self._active:
            raise RuntimeError("SenseNova actor controller is already active")
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        self._actor_sequence = ()
        if isinstance(policy, str):
            if policy not in self.POLICIES:
                raise ValueError(f"unsupported SenseNova actor policy {policy!r}")
            activation = (
                num_steps - 1 if activation_step is None else int(activation_step)
            )
            if policy == "late_step" and not 0 <= activation < num_steps:
                raise ValueError(
                    "late_step activation must be inside the denoise schedule"
                )
            self._policy = policy
            self._activation_step = activation
        else:
            policy_steps = int(getattr(policy, "num_steps", -1))
            actor_at = getattr(policy, "actor_at", None)
            if policy_steps != num_steps or not callable(actor_at):
                raise TypeError(
                    "segmented SenseNova policies require actor_at() and a "
                    "matching num_steps"
                )
            actors = tuple(str(actor_at(step)) for step in range(num_steps))
            if any(actor not in {"base", "refined"} for actor in actors):
                raise ValueError("segmented SenseNova policy has an invalid actor")
            self._policy = str(getattr(policy, "name", "segmented"))
            self._activation_step = 0
            self._actor_sequence = actors
        self._num_steps = int(num_steps)
        self._step_index = 0
        self._trace = []
        self._reference_state = reference_state
        self._conditioning_actor = self.conditioning_actor_for(
            policy, num_steps=num_steps
        )
        self.conditioning_actor_state.actor = self._conditioning_actor
        self.denoiser_actor_state.actor = "base"
        self._captured_reference_embeddings = None
        self._pending_prefix_contexts.clear()
        self._active = True

    def finish(self) -> tuple[str, ...]:
        self._active = False
        self._reference_state = None
        self.denoiser_actor_state.actor = "base"
        self.conditioning_actor_state.actor = "base"
        if self._step_index != self._num_steps:
            raise RuntimeError(
                "SenseNova sampler denoise count mismatch: "
                f"{self._step_index} != {self._num_steps}"
            )
        if self._pending_prefix_contexts:
            raise RuntimeError("SenseNova sampler left unmatched prefix input contexts")
        return tuple(self._trace)

    def abort(self) -> None:
        self._active = False
        self._reference_state = None
        self.denoiser_actor_state.actor = "base"
        self.conditioning_actor_state.actor = "base"
        self._captured_reference_embeddings = None
        self._pending_prefix_contexts.clear()


class SenseNovaPolicyInference:
    """Run an official edit while recording the actor used by every step."""

    def __init__(self, model: Any, export_path: str | Path) -> None:
        self.model = model
        self.actors = SenseNovaGenerationActors(model.runtime.model, export_path)
        raw = model.runtime.model
        self.reference_patch = int(raw.patch_size)
        self.image_multiple = int(raw.patch_size) * int(1 / raw.downsample_ratio)

    def _reference_from_patches(
        self,
        patches: Any,
        grid_hw: Any,
        *,
        width: int,
        height: int,
        actor: str,
    ) -> SenseNovaReferenceState:
        import torch

        raw = self.model.runtime.model
        parameter = next(raw.parameters())
        patches = patches.to(device=parameter.device, dtype=torch.bfloat16)
        grid_hw = grid_hw.to(device=parameter.device)
        with self.actors.use_conditioning_actor(actor):
            embeddings = self.actors._original_extract_feature(
                patches,
                gen_model=False,
                grid_hw=grid_hw,
            )
        return SenseNovaReferenceState(
            embeddings=embeddings.detach(),
            grid_hw=grid_hw.detach(),
            width=int(width),
            height=int(height),
        )

    def conditioning_actor_for(self, policy: Any, *, num_steps: int) -> str:
        return self.actors.conditioning_actor_for(policy, num_steps=num_steps)

    def encode_reference_image(
        self, image: Any, *, actor: str = "base"
    ) -> SenseNovaReferenceState:
        """Run the official PIL reference preprocessing exactly once."""

        raw = self.model.runtime.model
        module = importlib.import_module(type(raw).__module__)
        load_image_native = getattr(module, "load_image_native", None)
        if not callable(load_image_native):
            raise RuntimeError("official SenseNova image loader is unavailable")
        patches, grid_hw = load_image_native(
            image,
            self.reference_patch,
            float(raw.downsample_ratio),
            min_pixels=512 * 512,
            max_pixels=2048 * 2048,
            upscale=False,
        )
        encoded_height = int(grid_hw[0, 0]) * self.reference_patch
        encoded_width = int(grid_hw[0, 1]) * self.reference_patch
        return self._reference_from_patches(
            patches,
            grid_hw,
            width=encoded_width,
            height=encoded_height,
            actor=actor,
        )

    @contextmanager
    def _schedule_context(self, config: dict[str, Any]) -> Any:
        """Override only schedule geometry, never source/output geometry."""

        raw = self.model.runtime.model
        schedule = dict(config.get("schedule", {}))
        policy = str(schedule.get("shift_policy", "fixed_reference_shape"))
        if policy == "native":
            yield
            return
        if policy != "fixed_reference_shape":
            raise ValueError(f"unsupported SenseNova shift_policy {policy!r}")
        reference_height = int(schedule.get("reference_height", 1024))
        reference_width = int(schedule.get("reference_width", 1024))
        if (
            reference_height <= 0
            or reference_width <= 0
            or reference_height % self.image_multiple
            or reference_width % self.image_multiple
        ):
            raise ValueError(
                "SenseNova reference schedule shape must be positive and "
                f"divisible by {self.image_multiple}"
            )
        fixed_tokens = (reference_height // self.image_multiple) * (
            reference_width // self.image_multiple
        )
        original = raw._apply_time_schedule
        was_instance_attribute = "_apply_time_schedule" in vars(raw)
        instance_value = vars(raw).get("_apply_time_schedule")

        def fixed_schedule(
            t: Any, _image_seq_len: int, *args: Any, **kwargs: Any
        ) -> Any:
            return original(t, fixed_tokens, *args, **kwargs)

        raw._apply_time_schedule = fixed_schedule
        try:
            yield
        finally:
            if was_instance_attribute:
                raw._apply_time_schedule = instance_value
            else:
                del raw._apply_time_schedule

    def encode_reference_pixels(
        self, pixels: Any, *, actor: str = "base"
    ) -> SenseNovaReferenceState:
        """Encode continuous sampler pixels without clamp, uint8, or PIL."""

        import torch

        if pixels.ndim != 4 or pixels.shape[0] != 1 or pixels.shape[1] != 3:
            raise ValueError("SenseNova continuous pixels must have shape [1,3,H,W]")
        value = pixels.to(dtype=torch.bfloat16)
        mean = torch.tensor(
            [0.485, 0.456, 0.406], dtype=value.dtype, device=value.device
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.229, 0.224, 0.225], dtype=value.dtype, device=value.device
        ).view(1, 3, 1, 1)
        normalized = (value * 0.5 + 0.5 - mean) / std
        height, width = int(value.shape[2]), int(value.shape[3])
        if height % self.reference_patch or width % self.reference_patch:
            raise ValueError("continuous SenseNova pixels are not patch aligned")
        raw = self.model.runtime.model
        patches = raw.patchify(
            normalized,
            self.reference_patch,
            channel_first=True,
        ).reshape(-1, 3 * self.reference_patch**2)
        grid_hw = torch.tensor(
            [[height // self.reference_patch, width // self.reference_patch]],
            device=value.device,
        )
        return self._reference_from_patches(
            patches,
            grid_hw,
            width=width,
            height=height,
            actor=actor,
        )

    def pack_pixels(self, pixels: Any) -> Any:
        state = self.model.runtime.model.patchify(pixels, self.image_multiple)
        return state.detach().cpu().contiguous()

    @contextmanager
    def _endpoint_timestep_context(self, config: dict[str, Any]) -> Any:
        """Use the full official timetable with one terminal interval skipped."""

        import torch

        plan = config.get("endpoint_plan")
        if not isinstance(plan, dict) or str(plan.get("name")) == "base":
            yield
            return
        raw = self.model.runtime.model
        original = raw._apply_time_schedule
        logical_steps = int(config["num_steps"])
        target = int(plan["target_actor_index"])
        execution_steps = int(plan["expected_calls"])
        if execution_steps != target + 1:
            raise ValueError(
                "SenseNova endpoint plan must execute prefix plus one terminal call"
            )

        def apply(t: Any, image_seq_len: int, *args: Any, **kwargs: Any) -> Any:
            if int(t.numel()) != execution_steps + 1:
                return original(t, image_seq_len, *args, **kwargs)
            full = torch.linspace(
                0.0,
                1.0,
                logical_steps + 1,
                device=t.device,
                dtype=t.dtype,
            )
            full = original(full, image_seq_len, *args, **kwargs)
            return torch.cat((full[: target + 1], full[-1:]), dim=0)

        raw._apply_time_schedule = apply
        try:
            yield
        finally:
            raw._apply_time_schedule = original

    @staticmethod
    def _execution_steps(config: dict[str, Any]) -> int:
        plan = config.get("endpoint_plan")
        if isinstance(plan, dict):
            return int(plan["expected_calls"])
        return int(config.get("num_steps", 50))

    def generate(
        self,
        source_image: Any,
        *,
        prompt: str,
        seed: int,
        policy: Any,
        config: dict[str, Any],
        condition_images: tuple[Any, ...] = (),
    ) -> tuple[Any, tuple[str, ...]]:
        execution_steps = self._execution_steps(config)
        activation = int(config.get("activation_step", execution_steps - 1))
        width = int(config.get("width", source_image.size[0]))
        height = int(config.get("height", source_image.size[1]))
        if width % self.image_multiple or height % self.image_multiple:
            raise ValueError(
                f"SenseNova input size {(width, height)} must be divisible by "
                f"{self.image_multiple}"
            )
        self.actors.begin(
            policy,
            num_steps=execution_steps,
            activation_step=activation,
        )
        try:
            runtime_config = {**config, "num_steps": execution_steps}
            prefix_context = (
                _bounded_prefix_attention(
                    self.model.runtime.model, method="_it2i_prefix_forward"
                )
                if condition_images
                else nullcontext()
            )
            with (
                self._schedule_context(config),
                self._endpoint_timestep_context(config),
                prefix_context,
            ):
                image = self.model.generate_image(
                    source_image,
                    prompt=prompt,
                    seed=seed,
                    width=width,
                    height=height,
                    config=runtime_config,
                    **(
                        {"condition_images": condition_images}
                        if condition_images
                        else {}
                    ),
                )
            trace = self.actors.finish()
        except Exception:
            self.actors.abort()
            raise
        return image, trace

    def generate_text(
        self,
        *,
        prompt: str,
        seed: int,
        policy: Any,
        config: dict[str, Any],
    ) -> tuple[Any, tuple[str, ...]]:
        """Use the official T2I sampler, with the existing full-model actors."""

        import torch
        from PIL import Image

        if config.get("endpoint_plan") is not None:
            raise ValueError("T2I requires a native stepwise policy")
        width, height = int(config["width"]), int(config["height"])
        if min(width, height) <= 0 or any(
            side % self.image_multiple for side in (width, height)
        ):
            raise ValueError(
                f"T2I size {(width, height)} must be positive and divisible by "
                f"{self.image_multiple}"
            )
        runtime = self.model.runtime
        if not callable(getattr(runtime.model, "t2i_generate", None)):
            raise RuntimeError("SenseNova runtime lacks the official t2i_generate")
        steps = int(config.get("num_steps", 50))
        self.actors.begin(policy, num_steps=steps)
        try:
            with (
                torch.inference_mode(),
                self._schedule_context(config),
                _bounded_prefix_attention(runtime.model),
            ):
                pixels = runtime.model.t2i_generate(
                    runtime.tokenizer,
                    prompt=prompt,
                    image_size=(width, height),
                    cfg_scale=float(config.get("cfg_scale", 4.0)),
                    cfg_norm=str(config.get("cfg_norm", "none")),
                    timestep_shift=float(config.get("timestep_shift", 3.0)),
                    enable_timestep_shift=True,
                    cfg_interval=(0.0, 1.0),
                    num_steps=steps,
                    batch_size=1,
                    think_mode=False,
                    seed=int(seed),
                )
            trace = self.actors.finish()
        except Exception:
            self.actors.abort()
            raise
        if tuple(pixels.shape) != (1, 3, height, width):
            raise ValueError(f"unexpected T2I output shape: {tuple(pixels.shape)}")
        # Match official examples/t2i/inference.py: denormalize, clamp, round.
        pixels = (pixels[0].detach().float().cpu() * 0.5 + 0.5).clamp(0, 1)
        array = pixels.permute(1, 2, 0).mul(255).round().byte().numpy()
        return Image.fromarray(array), trace

    def generate_model_native(
        self,
        geometry_image: Any,
        *,
        reference_state: SenseNovaReferenceState,
        prompt: str,
        seed: int,
        policy: Any,
        config: dict[str, Any],
        encode_next_reference: bool = True,
    ) -> tuple[Any, SenseNovaReferenceState | None, tuple[str, ...]]:
        """Generate and feed the continuous output through the reference ViT."""

        execution_steps = self._execution_steps(config)
        activation = int(config.get("activation_step", execution_steps - 1))
        conditioning_actor = self.conditioning_actor_for(
            policy, num_steps=execution_steps
        )
        width, height = reference_state.width, reference_state.height
        self.actors.begin(
            policy,
            num_steps=execution_steps,
            activation_step=activation,
            reference_state=reference_state,
        )
        try:
            runtime_config = {**config, "num_steps": execution_steps}
            with (
                self._schedule_context(config),
                self._endpoint_timestep_context(config),
            ):
                pixels = self.model.generate_pixels(
                    geometry_image,
                    prompt=prompt,
                    seed=seed,
                    width=width,
                    height=height,
                    config=runtime_config,
                )
            trace = self.actors.finish()
            next_reference = (
                self.encode_reference_pixels(pixels, actor=conditioning_actor)
                if encode_next_reference
                else None
            )
        except Exception:
            self.actors.abort()
            raise
        return self.pack_pixels(pixels), next_reference, trace
