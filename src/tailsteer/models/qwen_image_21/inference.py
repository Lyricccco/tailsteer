"""Official Qwen sampler with actor-local conditioning and prefix KV."""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from functools import wraps
from math import sqrt

from tailsteer.models.native_transformer import TransformerActors

from .model import QwenImage21Model
from .official import QwenImage21Runtime, composite_rgb, unpack_state
from .preprocessing import (
    PREPROCESSING,
    prepare_source,
    reference_canvas_context,
    source_geometry,
)


@contextmanager
def schedule_context(scheduler, schedule):
    """Override only the official scheduler's shift, never the image geometry."""
    policy = schedule.get("shift_policy", "native")
    if policy == "native":
        yield
        return
    if policy != "fixed_reference_shape":
        raise ValueError(f"unsupported Qwen shift_policy: {policy!r}")
    height = int(schedule.get("reference_height", 1024))
    width = int(schedule.get("reference_width", 1024))
    if min(height, width) <= 0 or height % 32 or width % 32:
        raise ValueError("Qwen schedule reference dimensions must be divisible by 32")
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift

    cfg = scheduler.config
    mu = calculate_shift(
        height * width // 256,
        cfg.get("base_image_seq_len", 256),
        cfg.get("max_image_seq_len", 4096),
        cfg.get("base_shift", 0.5),
        cfg.get("max_shift", 1.15),
    )
    original = scheduler.set_timesteps
    override = scheduler.__dict__.get("set_timesteps")

    @wraps(original)  # retrieve_timesteps inspects the sigmas parameter.
    def set_timesteps(*args, **kwargs):
        return original(*args, **{**kwargs, "mu": mu})

    scheduler.set_timesteps = set_timesteps
    try:
        yield
    finally:
        if override is None:
            del scheduler.set_timesteps
        else:
            scheduler.set_timesteps = override


@contextmanager
def actor_prefix_cache(transformer, actors, *, cache_factory):
    """Re-prefill on every weight switch, including switches back to pretrained.

    The official pipeline otherwise prefills only at step zero. A hook local to
    this generation keeps its sampler unchanged and never shares KV across rounds.
    """
    previous_actor, cache = None, None

    def pre_forward(module, args, kwargs):
        nonlocal previous_actor, cache
        if kwargs.get("kv_cache") is not None:
            actor = actors.state.actor
            if actor != previous_actor:
                cache = cache_factory()
                mode = "extract"
            else:
                mode = "cached"
            previous_actor = actor
            kwargs = {**kwargs, "kv_cache": cache, "kv_cache_mode": mode}
        return args, kwargs

    handle = transformer.register_forward_pre_hook(pre_forward, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def actor_conditioning(pipe, actors):
    """Re-encode on actor switches; keep both embeddings local to this round.

    The official pipeline normally encodes only once, before the denoising loop.
    The original encoder arguments already contain its exact source preprocessing.
    """
    import torch

    original = pipe.encode_prompt
    override = pipe.__dict__.get("encode_prompt")
    inputs, encoded, trace = None, {}, []

    def encode(*args, **kwargs):
        nonlocal inputs
        inputs = (args, kwargs)
        value = original(*args, **kwargs)
        encoded[actors.state.actor] = value
        return value

    def before_forward(module, args, kwargs):
        actor = actors.state.actor
        if inputs is None:
            raise RuntimeError("Qwen joint actor has no conditioning inputs")
        if actor not in encoded:
            value = original(*inputs[0], **inputs[1])
            reference = next(iter(encoded.values()))
            if not torch.equal(value[2], reference[2]):
                raise RuntimeError("Qwen actor changed the image-token layout")
            encoded[actor] = value
        embeds, mask, _ = encoded[actor]
        trace.append(actor)
        return args, {
            **kwargs,
            "encoder_hidden_states": embeds,
            "encoder_hidden_states_mask": mask,
        }

    pipe.encode_prompt = encode
    handle = pipe.transformer.register_forward_pre_hook(
        before_forward, with_kwargs=True
    )
    try:
        yield trace
    finally:
        handle.remove()
        if override is None:
            del pipe.encode_prompt
        else:
            pipe.encode_prompt = override


@dataclass(frozen=True)
class QwenImage21GenerationResult:
    state: object
    image: object
    trace: tuple[str, ...]
    metadata: dict = field(default_factory=dict)


class QwenImage21InferenceRuntime:
    supports_multi_image = True
    joint = False
    model_name = "qwen_image_21"
    product = "qwen_image_21_independent_multiround_comparison"
    preprocessing_contract = PREPROCESSING
    source_geometry = staticmethod(source_geometry)
    prepare_source = staticmethod(prepare_source)

    @classmethod
    def load(cls, model, export_path, *, device):
        if float(model.get("true_cfg_scale", 1.0)) != 1.0:
            raise ValueError("Qwen training/inference uses true_cfg_scale=1")
        result = cls()
        result.runtime = QwenImage21Runtime.load(model, device=device)
        result.model = model
        from ..checkpoint import checkpoint_manifest

        manifest = checkpoint_manifest(export_path) if export_path else {}
        from .joint import JOINT_EXPORT_FORMAT, ConditionEncoder, JointDenoiser

        result.joint = manifest.get("format") == JOINT_EXPORT_FORMAT
        pipe = result.runtime.pipeline
        actor_root = pipe.transformer
        if result.joint:
            encoder = ConditionEncoder(pipe.text_encoder.model)
            actor_root = JointDenoiser(pipe.transformer, encoder, pipe)
            pipe.text_encoder = encoder
        result.actors = TransformerActors(
            actor_root,
            export_path,
            model_name=cls.model_name,
            export_format=JOINT_EXPORT_FORMAT
            if result.joint
            else QwenImage21Model.export_format,
            offload_weights=True,
        )
        return result

    def generate(self, source, *, prompt, seed, policy, inference, condition_images=()):
        import torch
        from diffusers.models.transformers.transformer_qwenimage21 import (
            QwenImage21KVCache,
        )
        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import (
            calculate_dimensions,
        )

        if inference.get("feedback_mode", "image") != "image":
            raise ValueError("Qwen currently supports image feedback only")
        if int(inference["num_steps"]) != 40 or policy.num_steps != 40:
            raise ValueError("Qwen uses the native 40-step schedule")
        if inference.get("endpoint_plan"):
            raise ValueError("Qwen final recipes use full-schedule actor policies")
        if condition_images and source is None:
            raise ValueError("Qwen multi-image editing requires a primary source")
        if source is None:
            width, height = int(inference["width"]), int(inference["height"])
            if min(width, height) <= 0 or width % 32 or height % 32:
                raise ValueError("Qwen T2I dimensions must be positive multiples of 32")
        else:
            source = self.prepare_source(
                source,
                input_preprocessed=inference.get("input_preprocessed", False),
            )
            width, height = source.size
        # The official pipeline also resizes the reference according to this
        # area, independently of the explicit output width/height.
        resolution = round(sqrt(width * height))
        if source is not None and calculate_dimensions(resolution**2, width / height)[
            :2
        ] != (width, height):
            raise ValueError("Qwen reference would be resized away from its canvas")
        pipe = self.runtime.pipeline
        final = None

        def after_step(pipeline, index, timestep, values):
            nonlocal final
            if index == 39:
                final = unpack_state(values["latents"], width=width, height=height)
            self.actors.complete_step()
            return {}

        with (
            self.runtime._lock,
            torch.no_grad(),
            reference_canvas_context(pipe) if source is not None else nullcontext(),
            self.actors.use_policy(policy),
            (
                actor_conditioning(pipe, self.actors) if self.joint else nullcontext([])
            ) as condition_trace,
            schedule_context(pipe.scheduler, inference.get("schedule", {})),
            actor_prefix_cache(
                pipe.transformer,
                self.actors,
                cache_factory=lambda: QwenImage21KVCache(
                    len(pipe.transformer.transformer_blocks)
                ),
            ),
        ):
            rgba = pipe(
                image=[source, *condition_images] if condition_images else source,
                prompt=prompt,
                height=height,
                width=width,
                output_resolution=resolution,
                num_inference_steps=40,
                true_cfg_scale=1.0,
                use_kv_cache=bool(self.model.get("use_kv_cache", True)),
                generator=torch.Generator(device=self.runtime.device).manual_seed(
                    int(seed)
                ),
                callback_on_step_end=after_step,
                callback_on_step_end_tensor_inputs=["latents"],
            ).images[0]
            trace = self.actors.trace
        if final is None:
            raise RuntimeError("Qwen produced no final state")
        if rgba.size != (width, height):
            raise RuntimeError("Qwen output dimensions differ from the source canvas")
        if self.joint and tuple(condition_trace) != trace:
            raise RuntimeError("Qwen encoder and DiT actor traces differ")
        return QwenImage21GenerationResult(
            final,
            composite_rgb(rgba),
            trace,
            {
                **({"conditioning_actors": condition_trace} if self.joint else {}),
                **(
                    {"input_image_count": 1 + len(condition_images)}
                    if condition_images
                    else {}
                ),
            },
        )

    def generate_text(self, *, prompt, seed, policy, config):
        """Use the official image=None path, with the same actor/schedule handling."""
        result = self.generate(
            None, prompt=prompt, seed=seed, policy=policy, inference=config
        )
        return result.image, result.trace

    def conditioning_actor_for(self, policy, *, num_steps):
        if not self.joint:
            return "base"
        actors = {policy.actor_at(step) for step in range(num_steps)}
        return next(iter(actors)) if len(actors) == 1 else "per_step"

    def unload(self):
        self.actors = None
        self.runtime.unload()
