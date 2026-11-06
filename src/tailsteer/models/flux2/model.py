"""Thin FLUX.2 Klein integration over the installed Diffusers checkout.

Canonical TailSteer latents use unpacked ``[B, C, H/16, W/16]`` layout.
Packing, four-axis position IDs, reference concatenation, CFG, and timestep
scaling stay here because they are FLUX.2 semantics rather than trainer logic.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tailsteer.models.protocols import TailVelocityBranches

from .output_adapter import (
    attach_flux2_output_adapter,
    flux2_output_adapter_config,
)
from .policy import (
    canonical_flux2_policy,
    configure_flux2_structured_trainable,
    is_flux2_structured_policy,
)


@dataclass(frozen=True)
class SchedulePoint:
    step_index: int
    sigma: float
    timestep: float


@dataclass(frozen=True)
class ScheduleTransition:
    current: SchedulePoint
    next_sigma: float


class Flux2Schedule:
    def __init__(self, points: Sequence[SchedulePoint]) -> None:
        self.points = tuple(points)
        if not self.points or any(
            point.step_index != index for index, point in enumerate(self.points)
        ):
            raise ValueError("FLUX.2 schedule must be consecutive and non-empty")

    def __len__(self) -> int:
        return len(self.points)

    def point(self, index: int) -> SchedulePoint:
        return self.points[index]

    def transition(self, index: int) -> ScheduleTransition:
        following = self.points[index + 1].sigma if index + 1 < len(self) else 0.0
        return ScheduleTransition(self.points[index], following)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": "flux2_flow_match",
            "terminal_sigma": 0.0,
            "points": [point.__dict__ for point in self.points],
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def _empirical_mu(image_sequence_length: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_sequence_length > 4300:
        return float(a2 * image_sequence_length + b2)
    mu_200 = a2 * image_sequence_length + b2
    mu_10 = a1 * image_sequence_length + b1
    slope = (mu_200 - mu_10) / 190.0
    return float(mu_200 - 200.0 * slope + num_steps * slope)


def build_flux2_schedule(
    *,
    num_steps: int,
    height: int,
    width: int,
    denoising_strength: float = 1.0,
) -> Flux2Schedule:
    """Reproduce the BF16 rollout contract used by the FLUX.2 v2 cache."""

    import torch

    if num_steps < 1 or height < 16 or width < 16:
        raise ValueError("invalid FLUX.2 schedule dimensions or step count")
    if not 0.0 < denoising_strength <= 1.0:
        raise ValueError("denoising_strength must be in (0, 1]")
    sigma_min = 1.0 / num_steps
    sigma_start = sigma_min + (1.0 - sigma_min) * denoising_strength
    sigmas = torch.linspace(sigma_start, sigma_min, num_steps)
    sequence_length = (height // 16) * (width // 16)
    shift = math.exp(_empirical_mu(sequence_length, num_steps))
    sigmas = shift / (shift + (1.0 / sigmas - 1.0))
    return Flux2Schedule(
        [
            SchedulePoint(index, float(sigma), float(sigma * 1000.0))
            for index, sigma in enumerate(sigmas)
        ]
    )


@dataclass(frozen=True)
class Flux2Conditioning:
    payload: Mapping[str, Any]
    batch_size: int


def _module_device_dtype(module: Any, device: Any, dtype: Any) -> tuple[Any, Any]:
    try:
        parameter = next(module.parameters())
    except (AttributeError, StopIteration):
        return device, dtype
    return parameter.device, parameter.dtype


def _unwrap(module: Any) -> Any:
    seen: set[int] = set()
    while hasattr(module, "module"):
        if id(module) in seen or module.module is module:
            raise RuntimeError("cyclic module wrapper")
        seen.add(id(module))
        module = module.module
    return module


def _promote_flux_master_parameters(module: Any) -> None:
    """Promote the student FLUX parameter storage to FP32.

    FLUX is loaded in BF16 so that the 9B model fits the training workers.  For
    the opt-in FP32 path, every parameter in each student FSDP group must have
    the same original dtype.  Frozen parameters are therefore promoted too,
    while ``requires_grad`` still selects the only tensors seen by the
    optimizer.  FSDP's mixed-precision policy casts these FP32 parameters to
    BF16 for the forward pass.
    """

    import torch

    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.is_floating_point() and parameter.dtype != torch.float32:
                parameter.data = parameter.data.float()


def _component(root: Path, name: str, marker: str) -> Path | None:
    nested = root / name
    if nested.is_dir():
        return nested
    return root if (root / marker).is_file() else None


def _required_component(root: Path, name: str, marker: str) -> Path:
    value = _component(root, name, marker)
    if value is None:
        raise FileNotFoundError(f"missing FLUX.2 component {name!r} under {root}")
    return value


def torch_dtype(name: str) -> Any:
    import torch

    normalized = str(name).removeprefix("torch.")
    if normalized not in {"float32", "float16", "bfloat16"}:
        raise ValueError(f"unsupported dtype {name!r}")
    return getattr(torch, normalized)


def _jsonable_export(value: Any) -> Any:
    """Convert PEFT config values to the JSON types used by export manifests."""

    if isinstance(value, Mapping):
        return {str(key): _jsonable_export(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable_export(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class Flux2Model:
    """Research operations required by cache, offline training, and inference."""

    def __init__(
        self,
        *,
        pipe: Any,
        refined_transformer: Any,
        pretrained_transformer: Any,
        device: Any,
        dtype: Any,
        cfg_scale: float,
        negative_prompt: str,
        cfg_training: str,
    ) -> None:
        if cfg_training not in {"guided", "positive_only"}:
            raise ValueError("cfg_training must be guided or positive_only")
        self.pipe = pipe
        self.refined_transformer = refined_transformer
        self.pretrained_transformer = pretrained_transformer
        self.device = device
        self.dtype = dtype
        self.cfg_scale = float(cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.cfg_training = cfg_training
        self.trainable_policy = "full"
        # ``model`` is the legacy behavior: parameters stay in the dtype used
        # to load the checkpoint (BF16 in the shipped FLUX configs).  The
        # explicit ``float32`` option is opt-in.  It promotes student storage
        # for FSDP dtype uniformity; the policy still controls gradients and
        # which parameters enter AdamW.
        self.trainable_config: dict[str, Any] = {
            "policy": "full",
            "master_dtype": "model",
        }
        self.trainable_groups: tuple[str, ...] = ("transformer[:]",)
        self.trainable_parameter_names: tuple[str, ...] = ()
        self._fsdp_master_parameters_prepared = False
        for frozen in (getattr(pipe, "text_encoder", None), getattr(pipe, "vae", None)):
            if frozen is not None:
                frozen.requires_grad_(False)
                frozen.eval()
        if pretrained_transformer is not refined_transformer:
            pretrained_transformer.requires_grad_(False)
            pretrained_transformer.eval()

    @classmethod
    def load(
        cls,
        config: dict[str, Any],
        *,
        device: Any | None = None,
        include_vae: bool = False,
        independent_actors: bool = False,
        gradient_checkpointing: bool = False,
        export_path: str | Path | None = None,
    ) -> Flux2Model:
        import torch
        from diffusers import (
            AutoencoderKLFlux2,
            FlowMatchEulerDiscreteScheduler,
            Flux2KleinPipeline,
            Flux2Transformer2DModel,
        )
        from transformers import AutoTokenizer, Qwen3ForCausalLM

        model_root = Path(config["path"]).expanduser().resolve()
        if not model_root.is_dir():
            raise FileNotFoundError(model_root)
        support_value = config.get("support_path")
        support_root = (
            None
            if support_value is None
            else Path(support_value).expanduser().resolve()
        )
        if support_root is not None and not support_root.is_dir():
            raise FileNotFoundError(support_root)
        dtype = torch_dtype(str(config.get("dtype", "bfloat16")))
        device = device or torch.device(
            "cuda", int(__import__("os").environ.get("LOCAL_RANK", "0"))
        )
        load_kwargs = {"dtype": dtype, "local_files_only": True}

        if support_root is None:
            pipe = Flux2KleinPipeline.from_pretrained(
                str(model_root), torch_dtype=dtype, local_files_only=True
            )
            refined = pipe.transformer
            transformer_root = _required_component(
                model_root, "transformer", "config.json"
            )
            pretrained = (
                Flux2Transformer2DModel.from_pretrained(
                    str(transformer_root), **load_kwargs
                )
                if independent_actors
                else refined
            )
            if not include_vae:
                pipe.vae = None
        else:
            transformer_root = _required_component(
                model_root, "transformer", "config.json"
            )
            text_root = _required_component(support_root, "text_encoder", "config.json")
            tokenizer_root = _required_component(
                support_root, "tokenizer", "tokenizer_config.json"
            )
            vae_source = Path(config.get("vae_path", model_root)).expanduser().resolve()
            vae_root = _component(vae_source, "vae", "config.json")
            if vae_root is None:
                vae_root = _component(support_root, "vae", "config.json")
            refined = Flux2Transformer2DModel.from_pretrained(
                str(transformer_root), **load_kwargs
            )
            pretrained = (
                Flux2Transformer2DModel.from_pretrained(
                    str(transformer_root), **load_kwargs
                )
                if independent_actors
                else refined
            )
            text_encoder = Qwen3ForCausalLM.from_pretrained(
                str(text_root), **load_kwargs
            )
            tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_root), local_files_only=True
            )
            vae = (
                AutoencoderKLFlux2.from_pretrained(str(vae_root), **load_kwargs)
                if include_vae and vae_root is not None
                else None
            )
            scheduler_root = _component(
                support_root, "scheduler", "scheduler_config.json"
            ) or _component(model_root, "scheduler", "scheduler_config.json")
            scheduler = (
                FlowMatchEulerDiscreteScheduler.from_pretrained(
                    str(scheduler_root), local_files_only=True
                )
                if scheduler_root is not None
                else FlowMatchEulerDiscreteScheduler()
            )
            pipe = Flux2KleinPipeline(
                scheduler=scheduler,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                transformer=refined,
                is_distilled=bool(config.get("is_distilled", False)),
            )
        if gradient_checkpointing:
            refined.enable_gradient_checkpointing()
        pipe.to(device)
        if pretrained is not refined:
            pretrained.to(device=device, dtype=dtype)
        pipe.set_progress_bar_config(disable=True)
        result = cls(
            pipe=pipe,
            refined_transformer=refined,
            pretrained_transformer=pretrained,
            device=device,
            dtype=dtype,
            cfg_scale=float(config.get("cfg_scale", 4.0)),
            negative_prompt=str(config.get("negative_prompt", "")),
            cfg_training=str(config.get("cfg_training", "guided")),
        )
        if export_path is not None:
            result.load_export(export_path)
        return result

    def unload(self) -> None:
        """Release FLUX.2 modules without copying 9B weights back to host RAM.

        Cache workers call this exactly once at shutdown, but keeping the method
        idempotent also makes failure recovery safe.  Breaking every strong
        module reference before collecting cycles avoids the large temporary CPU
        allocation that ``module.to("cpu")`` can introduce in a 64 GiB worker.
        """

        import gc

        import torch

        pipe = getattr(self, "pipe", None)
        self.refined_transformer = None
        self.pretrained_transformer = None
        self.pipe = None
        if pipe is not None:
            for name in ("transformer", "text_encoder", "vae"):
                if hasattr(pipe, name):
                    setattr(pipe, name, None)
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_trainable_module(self) -> Any:
        return self.refined_transformer

    def parameter_anchor_reference(self) -> Any:
        """Return the frozen pretrained actor used by parameter regularizers."""

        return self.pretrained_transformer

    def replace_trainable_module(self, module: Any) -> None:
        self.refined_transformer = module
        self.pipe.transformer = module

    @staticmethod
    def _transformer_sharding_units(transformer: Any) -> tuple[Any, ...]:
        """Return the native FLUX block units used by both actors.

        The independent teacher is frozen, but still needs to be sharded in
        FSDP2.  Keeping the block discovery in one helper prevents the
        student and teacher from silently acquiring different layouts as the
        upstream transformer evolves.
        """

        root = _unwrap(transformer)
        units: list[Any] = []
        for name in ("transformer_blocks", "single_transformer_blocks"):
            blocks = tuple(getattr(root, name, ()) or ())
            if not blocks:
                raise ValueError(f"FLUX.2 transformer has no non-empty {name}")
            units.extend(blocks)
        return tuple(units)

    def sharding_units(self) -> tuple[Any, ...]:
        return self._transformer_sharding_units(self.refined_transformer)

    def auxiliary_sharding_groups(
        self,
    ) -> tuple[tuple[str, Any, tuple[Any, ...]], ...]:
        """Expose the frozen FLUX teacher to the FSDP2 preparation path.

        Teacher-guided C/D experiments instantiate a second 9B transformer.
        Leaving it replicated on every rank defeats FSDP's memory savings and
        can OOM before the first update, especially with an FP32 student
        master.  The generic distributed wrapper shards this group without
        adding teacher parameters to the optimizer.
        """

        if self.pretrained_transformer is self.refined_transformer:
            return ()
        return (
            (
                "pretrained_teacher",
                self.pretrained_transformer,
                self._transformer_sharding_units(self.pretrained_transformer),
            ),
        )

    def replace_auxiliary_sharded_module(self, name: str, module: Any) -> None:
        if name != "pretrained_teacher":
            raise KeyError(f"unknown FLUX.2 auxiliary module {name!r}")
        self.pretrained_transformer = module

    def actor_memory_boundary(self) -> None:
        """Release allocator blocks between independent student/teacher calls."""

        if self.pretrained_transformer is self.refined_transformer:
            return
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def prepare_sharding_unit(self, unit: Any) -> None:
        """Prepare one FLUX FSDP unit before it is sharded."""

        if str(self.trainable_config.get("master_dtype", "model")) == "float32":
            _promote_flux_master_parameters(unit)

    def prepare_sharding_root(self) -> None:
        """Promote the student parameters before the root FSDP wrapper."""

        if str(self.trainable_config.get("master_dtype", "model")) == "float32":
            _promote_flux_master_parameters(_unwrap(self.refined_transformer))
            self._fsdp_master_parameters_prepared = True

    @staticmethod
    def _local_parameter_tensor(parameter: Any) -> Any:
        to_local = getattr(parameter, "to_local", None)
        return to_local() if callable(to_local) else parameter

    def prepare_optimizer_parameters(self) -> None:
        """Ensure FP32 student storage remains after distributed preparation."""

        import torch

        if str(self.trainable_config.get("master_dtype", "model")) != "float32":
            return
        if not self._fsdp_master_parameters_prepared:
            _promote_flux_master_parameters(_unwrap(self.refined_transformer))
            self._fsdp_master_parameters_prepared = True
        mismatched = [
            name
            for name, parameter in self.refined_transformer.named_parameters()
            if parameter.requires_grad
            and (
                parameter.dtype != torch.float32
                or self._local_parameter_tensor(parameter).dtype != torch.float32
            )
        ]
        if mismatched:
            raise RuntimeError(
                f"FLUX trainable parameters lack FP32 masters: {mismatched[:3]}"
            )

    def configure_trainable(self, config: Mapping[str, Any]) -> None:
        root = _unwrap(self.refined_transformer)
        requested_policy = str(config.get("policy", "full"))
        policy = canonical_flux2_policy(requested_policy)
        master_dtype = str(config.get("master_dtype", "model"))
        if master_dtype not in {"float32", "model"}:
            raise ValueError("FLUX.2 master_dtype must be float32 or model")
        output_adapter = flux2_output_adapter_config(config.get("output_adapter"))
        if output_adapter.enabled:
            if policy == "lora":
                raise ValueError(
                    "FLUX.2 output_adapter is not supported with the lora policy"
                )
            attach_flux2_output_adapter(root, output_adapter.to_dict())
        if policy == "full":
            root.requires_grad_(True)
            selection = None
        elif policy == "lora":
            from peft import LoraConfig

            root.requires_grad_(False)
            lora = config.get("lora", {})
            adapter_config = LoraConfig(
                r=int(lora.get("rank", 16)),
                lora_alpha=int(lora.get("alpha", 16)),
                lora_dropout=float(lora.get("dropout", 0.0)),
                target_modules=list(
                    lora.get("target_modules", ["to_q", "to_k", "to_v", "to_out.0"])
                ),
            )
            root.add_adapter(adapter_config)
            selection = None
        elif is_flux2_structured_policy(policy):
            selection = configure_flux2_structured_trainable(root, config)
        else:
            raise ValueError(
                "FLUX.2 trainable policy must be full, lora, or one of "
                "output_head, output_head_norm, tail_single, single_stream, dit_full"
            )
        self.trainable_policy = policy
        self.trainable_config = dict(config)
        self.trainable_config.setdefault("policy", policy)
        self.trainable_config["policy"] = policy
        self.trainable_config["master_dtype"] = master_dtype
        if output_adapter.enabled:
            self.trainable_config["output_adapter"] = output_adapter.to_dict()
        if selection is None:
            self.trainable_groups = ("lora" if policy == "lora" else "transformer[:]",)
            self.trainable_parameter_names = tuple(
                name
                for name, parameter in root.named_parameters()
                if parameter.requires_grad
            )
        else:
            self.trainable_groups = selection.groups
            self.trainable_parameter_names = selection.parameter_names
            self.trainable_config.update(selection.config())
        if output_adapter.enabled:
            self.trainable_groups = (*self.trainable_groups, "output_residual_adapter")
            self.trainable_config["trainable_groups"] = list(self.trainable_groups)
        self._fsdp_master_parameters_prepared = False
        root.train()
        if not any(parameter.requires_grad for parameter in root.parameters()):
            raise RuntimeError("trainable policy selected no parameters")

    def trainable_parameters(self) -> tuple[Any, ...]:
        return tuple(
            parameter
            for parameter in self.refined_transformer.parameters()
            if parameter.requires_grad
        )

    def inference_export_dtype(self) -> Any:
        """Keep portable FLUX exports in the native model dtype.

        This is particularly important for the opt-in FP32-storage experiment:
        training parameters may stay FP32, while inference exports remain BF16
        and retain the storage/loader contract of existing FLUX runs.
        """

        return getattr(self, "dtype", None)

    def prepare_inference_export(
        self, state: Mapping[str, Any], *, step: int
    ) -> tuple[dict[str, Any], str, dict[str, Any]]:
        """Prepare a portable FLUX.2 export without changing its state format.

        Structured H0--H4 policies intentionally export the complete
        transformer state.  The manifest records the selected trainable groups;
        loading the complete state keeps the existing FLUX inference contract
        and makes an export self-contained for actor switching.
        """

        policy = canonical_flux2_policy(self.trainable_policy)
        if policy == "lora":
            root = _unwrap(self.get_trainable_module())
            selected = {key: value for key, value in state.items() if "lora_" in key}
            if not selected:
                raise RuntimeError("FLUX.2 LoRA export selected no adapter tensors")
            peft_config = next(iter(root.peft_config.values()))
            manifest = {
                "schema_version": 1,
                "model": "flux2_klein_9b",
                "step": int(step),
                "trainable_policy": policy,
                "weights": "adapter_model.safetensors",
                "lora_config": _jsonable_export(peft_config.to_dict()),
            }
            return selected, "adapter_model.safetensors", manifest

        if policy != "full" and not is_flux2_structured_policy(policy):
            raise ValueError(f"unsupported FLUX.2 export policy {policy!r}")
        groups = tuple(getattr(self, "trainable_groups", ("transformer[:]",)))
        trainable = deepcopy(
            getattr(
                self,
                "trainable_config",
                {"policy": policy, "master_dtype": "model"},
            )
        )
        trainable["policy"] = policy
        manifest = {
            "schema_version": 1,
            "model": "flux2_klein_9b",
            "step": int(step),
            "trainable_policy": policy,
            "weights": "transformer.safetensors",
            "lora_config": None,
            "trainable": trainable,
            "trainable_groups": list(groups),
        }
        return dict(state), "transformer.safetensors", manifest

    def _uses_cfg(self) -> bool:
        distilled = bool(
            getattr(getattr(self.pipe, "config", None), "is_distilled", False)
        )
        return self.cfg_training == "guided" and self.cfg_scale > 1.0 and not distilled

    def tail_velocity_branch_names(self) -> tuple[str, ...]:
        """Branches available under the active FLUX.2 CFG policy."""

        return ("base", "conditioned") if self._uses_cfg() else ("conditioned",)

    def prepare_conditioning(
        self,
        *,
        source_latent: Any,
        prompts: Sequence[str],
        height: int,
        width: int,
    ) -> Flux2Conditioning:
        import torch

        if source_latent.ndim != 4 or len(prompts) != source_latent.shape[0]:
            raise ValueError("source latent and prompt batch shapes differ")
        if height <= 0 or width <= 0 or height % 16 or width % 16:
            raise ValueError("FLUX.2 dimensions must be positive multiples of 16")
        with torch.no_grad():
            positive, positive_ids = self.pipe.encode_prompt(
                prompt=list(prompts), device=self.device, num_images_per_prompt=1
            )
            if self._uses_cfg():
                negative, negative_ids = self.pipe.encode_prompt(
                    prompt=[self.negative_prompt] * len(prompts),
                    device=self.device,
                    num_images_per_prompt=1,
                )
            else:
                negative = negative_ids = None
            target = source_latent.new_empty(
                (
                    source_latent.shape[0],
                    source_latent.shape[1],
                    height // 16,
                    width // 16,
                )
            )
            target_ids = self.pipe._prepare_latent_ids(target).to(source_latent.device)
            source_ids = self.pipe._prepare_image_ids([source_latent[:1]])
            source_ids = source_ids.to(source_latent.device).expand(
                source_latent.shape[0], -1, -1
            )
            source_tokens = self.pipe._pack_latents(source_latent)
        return Flux2Conditioning(
            payload={
                "positive": positive,
                "positive_ids": positive_ids,
                "negative": negative,
                "negative_ids": negative_ids,
                "target_ids": target_ids,
                "source_ids": source_ids,
                "source_tokens": source_tokens,
            },
            batch_size=int(source_latent.shape[0]),
        )

    def _actor(self, actor: str) -> Any:
        if actor == "base":
            return self.pretrained_transformer
        if actor == "refined":
            return self.refined_transformer
        raise ValueError(f"unknown FLUX.2 actor {actor!r}")

    def _branch(
        self,
        transformer: Any,
        model_input: Any,
        image_ids: Any,
        point: SchedulePoint,
        prompt: Any,
        text_ids: Any,
        cache_name: str,
    ) -> Any:
        import torch

        owner = _unwrap(transformer)
        context = (
            owner.cache_context(cache_name)
            if hasattr(owner, "cache_context")
            else nullcontext()
        )
        timestep = torch.full(
            (model_input.shape[0],),
            point.timestep / 1000.0,
            device=model_input.device,
            dtype=model_input.dtype,
        )
        with context:
            output = transformer(
                hidden_states=model_input,
                timestep=timestep,
                guidance=None,
                encoder_hidden_states=prompt,
                txt_ids=text_ids,
                img_ids=image_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )
        return output[0] if isinstance(output, tuple) else output.sample

    def predict_velocity_branches(
        self,
        latent: Any,
        point: SchedulePoint,
        conditioning: Flux2Conditioning,
        *,
        actor: str,
    ) -> TailVelocityBranches:
        import torch

        transformer = self._actor(actor)
        if conditioning.batch_size != latent.shape[0]:
            raise ValueError("conditioning and latent batch sizes differ")
        device, parameter_dtype = _module_device_dtype(
            transformer, latent.device, self.dtype
        )
        # FP32 student storage is not a request for FP32 activations. FSDP's
        # mixed-precision policy still gathers BF16 compute parameters, so keep
        # cached inputs in the native model dtype (the legacy path already has
        # this value).
        dtype = (
            self.dtype
            if str(self.trainable_config.get("master_dtype", "model")) == "float32"
            else parameter_dtype
        )
        target_tokens = self.pipe._pack_latents(latent).to(device=device, dtype=dtype)
        payload = conditioning.payload
        source_tokens = payload["source_tokens"].to(device=device, dtype=dtype)
        model_input = torch.cat([target_tokens, source_tokens], dim=1)
        image_ids = torch.cat(
            [payload["target_ids"].to(device), payload["source_ids"].to(device)], dim=1
        )
        conditioned = self._branch(
            transformer,
            model_input,
            image_ids,
            point,
            payload["positive"].to(device=device, dtype=dtype),
            payload["positive_ids"].to(device),
            "cond",
        )[:, : target_tokens.shape[1]]
        token_values = {"conditioned": conditioned}
        if self._uses_cfg():
            base = self._branch(
                transformer,
                model_input,
                image_ids,
                point,
                payload["negative"].to(device=device, dtype=dtype),
                payload["negative_ids"].to(device),
                "uncond",
            )[:, : target_tokens.shape[1]]
        batch, channels, latent_height, latent_width = latent.shape
        if conditioned.shape != target_tokens.shape:
            raise RuntimeError("FLUX.2 transformer output shape differs from target")
        if self._uses_cfg():
            if base.shape != target_tokens.shape:
                raise RuntimeError(
                    "FLUX.2 transformer output shape differs from target"
                )
            token_values["base"] = base
        values = {
            name: value.reshape(batch, latent_height, latent_width, channels).permute(
                0, 3, 1, 2
            )
            for name, value in token_values.items()
        }
        return TailVelocityBranches(values)

    def compose_tail_velocity_branches(self, prediction: TailVelocityBranches) -> Any:
        values = dict(prediction.values)
        expected = set(self.tail_velocity_branch_names())
        if set(values) != expected:
            raise ValueError(
                "FLUX.2 velocity branches differ from active CFG policy: "
                f"expected={sorted(expected)} actual={sorted(values)}"
            )
        conditioned = values["conditioned"]
        if "base" not in values:
            return conditioned
        base = values["base"]
        return base + self.cfg_scale * (conditioned - base)

    def predict_velocity(
        self,
        latent: Any,
        point: SchedulePoint,
        conditioning: Flux2Conditioning,
        *,
        actor: str,
    ) -> Any:
        return self.compose_tail_velocity_branches(
            self.predict_velocity_branches(latent, point, conditioning, actor=actor)
        )

    @staticmethod
    def advance(latent: Any, velocity: Any, transition: ScheduleTransition) -> Any:
        return latent + velocity * (transition.next_sigma - transition.current.sigma)

    def prepare_training_batch(self, batch: Mapping[str, Any]) -> Any:
        from .training import prepare_flux2_training_batch

        return prepare_flux2_training_batch(batch, device=self.device, dtype=self.dtype)

    def prepare_tail_batch(
        self, batch: Mapping[str, Any], *, state_keys: Sequence[str]
    ) -> Any:
        """Prepare model-native cached states for teacher-guided algorithms."""

        from tailsteer.models.protocols import PreparedTrajectoryBatch

        metadata = tuple(batch["metadata"])
        if not metadata:
            raise ValueError("FLUX.2 tail batch has no metadata")
        first = metadata[0]
        for field in ("height", "width"):
            if len({int(item[field]) for item in metadata}) != 1:
                raise ValueError(f"FLUX.2 tail batch mixes {field}")
        source = batch["source_latent"]
        if source.ndim != 4:
            raise ValueError("FLUX.2 source tail states must use NCHW layout")
        states = {}
        for key in state_keys:
            value = batch[key]
            if value.shape != source.shape:
                raise ValueError(f"FLUX.2 tail state {key!r} differs from source shape")
            states[key] = value.to(device=self.device, dtype=self.dtype)
        source = source.to(device=self.device, dtype=self.dtype)
        prompts = tuple(str(item["prompt"]) for item in metadata)
        height, width = int(first["height"]), int(first["width"])
        return PreparedTrajectoryBatch(
            states=states,
            source_state=source,
            keep_mask=batch["keep_mask"].to(device=self.device),
            conditioning=self.prepare_conditioning(
                source_latent=source,
                prompts=prompts,
                height=height,
                width=width,
            ),
            prompts=prompts,
            height=height,
            width=width,
        )

    def predict_tail_velocity(
        self, latent: Any, point: Any, conditioning: Any, *, actor: str
    ) -> Any:
        return self.predict_velocity(latent, point, conditioning, actor=actor)

    def flow_matching_training_point(self, *, generator: Any = None) -> Any:
        """Sample the native training grid, independently of inference scheduling.

        Matches Diffusers' Klein img2img SFT ``weighting_scheme=none``.
        Reconstructing from config avoids reusing a scheduler mutated by rollout.
        """
        import torch

        from tailsteer.models.protocols import TrajectoryPoint

        scheduler = getattr(self, "_flow_matching_scheduler", None)
        if scheduler is None:
            scheduler = type(self.pipe.scheduler).from_config(
                self.pipe.scheduler.config
            )
            self._flow_matching_scheduler = scheduler
        index = int(
            (
                torch.rand((), device=self.device, generator=generator)
                * len(scheduler.timesteps)
            ).item()
        )
        return TrajectoryPoint(
            float(index),
            float(scheduler.sigmas[index]),
            float(scheduler.timesteps[index]),
        )

    def predict_tail_velocity_branches(
        self, latent: Any, point: Any, conditioning: Any, *, actor: str
    ) -> TailVelocityBranches:
        return self.predict_velocity_branches(latent, point, conditioning, actor=actor)

    def tail_schedule(
        self, *, num_steps: int, height: int, width: int
    ) -> dict[int, Any]:
        from tailsteer.models.protocols import TrajectoryPoint

        schedule = build_flux2_schedule(num_steps=num_steps, height=height, width=width)
        points = {
            point.step_index: TrajectoryPoint(
                float(point.step_index), float(point.sigma), float(point.timestep)
            )
            for point in schedule.points
        }
        points[num_steps] = TrajectoryPoint(float(num_steps), 0.0, 0.0)
        return points

    @staticmethod
    def tail_advance(
        latent: Any,
        velocity: Any,
        *,
        current: Any,
        following: Any,
    ) -> Any:
        return latent + velocity * (float(following.sigma) - float(current.sigma))

    @staticmethod
    def tail_step_delta(current: Any, following: Any) -> float:
        """Native integration interval for one cached-path segment."""

        return float(following.sigma) - float(current.sigma)

    @staticmethod
    def tail_terminal_delta(point: Any) -> float:
        return -float(point.sigma)

    @staticmethod
    def tail_keep_mask(mask: Any, state: Any) -> Any:
        if mask.ndim == state.ndim - 1:
            mask = mask.unsqueeze(1)
        if mask.ndim != state.ndim or mask.shape[0] != state.shape[0]:
            raise ValueError("FLUX.2 tail mask rank/batch differs from state")
        if (
            mask.shape[1] not in {1, state.shape[1]}
            or mask.shape[2:] != state.shape[2:]
        ):
            raise ValueError("FLUX.2 tail mask shape differs from state")
        return mask.to(device=state.device, dtype=state.dtype)

    def training_forward(self, batch: Any) -> Any:
        from .training import Flux2TrainingBatch

        prepared = (
            batch
            if isinstance(batch, Flux2TrainingBatch)
            else self.prepare_training_batch(batch)
        )
        conditioning = self.prepare_conditioning(
            source_latent=prepared.source_latent,
            prompts=prepared.prompts,
            height=prepared.height,
            width=prepared.width,
        )
        point = SchedulePoint(
            prepared.step_index,
            prepared.sigma,
            prepared.timestep,
        )
        return self.predict_velocity(
            prepared.latent, point, conditioning, actor="refined"
        )

    def compute_training_loss(
        self, batch: Mapping[str, Any], *, preservation_weight: float
    ) -> Any:
        from .objective import flux2_preservation_velocity

        prepared = self.prepare_training_batch(batch)
        prediction = self.training_forward(prepared)
        return flux2_preservation_velocity(
            prediction, prepared, weight=preservation_weight
        )

    def validation_inference_config(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """Describe the exact TailSteer sampler used by visual validation."""

        return {
            "sampler": "flux2_flow_match_tailsteer",
            "num_steps": int(config.get("num_steps", 50)),
            "denoising_strength": float(config.get("denoising_strength", 1.0)),
            "noise_mode": str(config.get("noise_mode", "auto")),
            "cfg_scale": float(config.get("cfg_scale", self.cfg_scale)),
        }

    def generate_validation_sample(
        self, sample: Any, config: Mapping[str, Any]
    ) -> tuple[Any, Any]:
        """Run a complete TailSteer FLUX.2 edit from one held-out source."""

        import torch

        if getattr(self.pipe, "vae", None) is None:
            raise RuntimeError("FLUX.2 visual validation requires include_vae=True")
        metadata = sample.metadata
        height, width = int(metadata["height"]), int(metadata["width"])
        source_latent = sample.source_latent
        if source_latent.ndim == 3:
            source_latent = source_latent.unsqueeze(0)
        if source_latent.ndim != 4 or source_latent.shape[0] != 1:
            raise ValueError("FLUX.2 validation source must have shape [C,H,W]")
        source_latent = source_latent.to(device=self.device, dtype=self.dtype)
        inference = self.validation_inference_config(config)
        schedule = build_flux2_schedule(
            num_steps=int(inference["num_steps"]),
            height=height,
            width=width,
            denoising_strength=float(inference["denoising_strength"]),
        )
        source_image = self.decode_latent(source_latent, height=height, width=width)
        previous_cfg = self.cfg_scale
        self.cfg_scale = float(inference["cfg_scale"])
        try:
            conditioning = self.prepare_conditioning(
                source_latent=source_latent,
                prompts=[str(metadata["prompt"])],
                height=height,
                width=width,
            )
            state = self.initialize_latent(
                source_latent,
                seed=int(metadata["seed"]),
                sigma=schedule.point(0).sigma,
                noise_mode=str(inference["noise_mode"]),
            )
            with torch.no_grad():
                for step in range(len(schedule)):
                    transition = schedule.transition(step)
                    velocity = self.predict_velocity(
                        state,
                        transition.current,
                        conditioning,
                        actor="refined",
                    )
                    state = self.advance(state, velocity, transition)
            generated = self.decode_latent(state, height=height, width=width)
        finally:
            self.cfg_scale = previous_cfg
        return source_image, generated

    def inference_size(self, image: Any) -> tuple[int, int]:
        return self.prepare_inference_image(image).size

    @staticmethod
    def prepare_inference_image(image: Any, *, reference_count: int = 1) -> Any:
        from .preprocessing import prepare_flux2_reference_image

        return prepare_flux2_reference_image(
            image,
            reference_count=reference_count,
        )

    def encode_image(
        self, image: Any, *, width: int | None = None, height: int | None = None
    ) -> Any:
        import torch

        vae = self.pipe.vae
        if vae is None:
            raise RuntimeError("FLUX.2 image encoding requires a VAE")
        device, dtype = _module_device_dtype(vae, self.device, self.dtype)
        with torch.no_grad():
            preprocess_kwargs: dict[str, Any] = {}
            if width is not None or height is not None:
                if width is None or height is None:
                    raise ValueError(
                        "FLUX.2 encode width and height must be set together"
                    )
                if image.size != (int(width), int(height)):
                    raise ValueError(
                        "FLUX.2 encode_image expects an already prepared PIL image; "
                        f"got {image.size}, expected {(int(width), int(height))}"
                    )
            pixels = self.pipe.image_processor.preprocess(
                image, **preprocess_kwargs
            ).to(device=device, dtype=dtype)
            generator = torch.Generator(device="cpu").manual_seed(0)
            return self.pipe._encode_vae_image(pixels, generator=generator).to(
                self.dtype
            )

    def decode_latent(self, latent: Any, *, height: int, width: int) -> Any:
        import torch

        vae = self.pipe.vae
        if vae is None:
            raise RuntimeError("FLUX.2 image decoding requires a VAE")
        device, dtype = _module_device_dtype(vae, self.device, self.dtype)
        with torch.no_grad():
            latent = latent.to(device=device, dtype=dtype)
            mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latent)
            std = torch.sqrt(
                vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
            ).to(latent)
            unpacked = self.pipe._unpatchify_latents(latent * std + mean)
            decoded = vae.decode(unpacked, return_dict=False)[0]
            images = self.pipe.image_processor.postprocess(decoded, output_type="pil")
        return images[0]

    def initialize_latent(
        self, source_latent: Any, *, seed: int, sigma: float, noise_mode: str
    ) -> Any:
        import torch

        if noise_mode == "auto":
            noise_mode = "pure_noise" if sigma >= 1.0 else "add_noise"
        if noise_mode not in {"pure_noise", "add_noise"}:
            raise ValueError("noise_mode must be pure_noise, add_noise, or auto")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.randn(
            tuple(source_latent.shape),
            generator=generator,
            dtype=self.dtype,
            device="cpu",
        ).to(source_latent.device)
        return (
            noise
            if noise_mode == "pure_noise"
            else source_latent * (1 - sigma) + noise * sigma
        )

    def load_export(self, export_path: str | Path) -> None:
        from safetensors.torch import load_file

        export = Path(export_path).expanduser().resolve()
        if (
            not (export / "tailrefine_release.json").is_file()
            and not (export / "COMPLETE").is_file()
        ):
            raise RuntimeError(f"inference export is incomplete: {export}")
        from ..checkpoint import checkpoint_manifest, open_weights

        manifest = checkpoint_manifest(export)
        policy = canonical_flux2_policy(str(manifest["trainable_policy"]))
        trainable = manifest.get("trainable")
        if isinstance(trainable, Mapping):
            attach_flux2_output_adapter(
                _unwrap(self.refined_transformer),
                trainable.get("output_adapter"),
            )
        if policy == "lora":
            from peft import LoraConfig, set_peft_model_state_dict

            root = _unwrap(self.refined_transformer)
            root.add_adapter(LoraConfig(**manifest["lora_config"]))
            set_peft_model_state_dict(root, load_file(export / manifest["weights"]))
        elif policy == "full" or is_flux2_structured_policy(policy):
            with open_weights(export / manifest["weights"]) as handle:
                state = {name: handle.get_tensor(name) for name in handle.keys()}
            _unwrap(self.refined_transformer).load_state_dict(state, strict=True)
        else:
            raise ValueError(f"unsupported export policy {policy!r}")
        _unwrap(self.refined_transformer).requires_grad_(False)
        _unwrap(self.refined_transformer).eval()
        self.trainable_policy = policy
        if isinstance(trainable, Mapping):
            self.trainable_config = deepcopy(dict(trainable))
        groups = manifest.get("trainable_groups")
        if isinstance(groups, list):
            self.trainable_groups = tuple(str(group) for group in groups)
