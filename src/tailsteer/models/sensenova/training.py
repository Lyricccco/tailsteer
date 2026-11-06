"""SenseNova-specific cache layout and official denoiser training forward."""

from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import nullcontext
from copy import copy
from dataclasses import dataclass
from typing import Any

from .conditioning import PrefixKVAdapterConfig, attach_prefix_kv_adapter
from .policy import (
    configure_sensenova_structured_trainable,
    is_sensenova_structured_policy,
)

SENSENOVA_GENERATION_MODULES = (
    "vision_model_mot_gen",
    "timestep_embedder",
    "noise_scale_embedder",
    "fm_head",
)


def is_sensenova_generation_state(name: str) -> bool:
    normalized = str(name).removeprefix("model.")
    return "mot_gen" in normalized or any(
        normalized == f"fm_modules.{module}"
        or normalized.startswith(f"fm_modules.{module}.")
        for module in SENSENOVA_GENERATION_MODULES
    )


def is_sensenova_reference_state(name: str) -> bool:
    normalized = str(name).removeprefix("model.")
    return normalized == "vision_model" or normalized.startswith("vision_model.")


def is_sensenova_prefix_adapter_state(name: str) -> bool:
    normalized = str(name).removeprefix("model.")
    return normalized == "tailsteer_prefix_kv_adapter" or normalized.startswith(
        "tailsteer_prefix_kv_adapter."
    )


def configure_sensenova_trainable(
    model: Any, config: str | Mapping[str, Any]
) -> tuple[str, ...]:
    """Select SenseNova parameters while preserving the narrow default policy.

    ``full_model`` follows the released U1 8B SFT module scope:
    ``freeze_llm=false``, ``freeze_backbone=false``, and
    ``unfreeze_mot_gen=true``. TailSteer's loss and cached-state sampling are
    unchanged; only the set of trainable parameters differs.
    """

    raw = {"policy": config} if isinstance(config, str) else dict(config)
    normalized = str(raw.get("policy", "generation_denoiser"))
    train_reference = bool(raw.get("reference_vision_model", False))
    adapter_config = PrefixKVAdapterConfig.from_mapping(raw.get("prefix_kv_adapter"))
    if train_reference and not adapter_config.enabled:
        raise ValueError(
            "training the SenseNova reference vision model requires the "
            "image-prefix KV adapter gradient path"
        )
    model.requires_grad_(False)
    model._tailsteer_trainable_groups = ()
    if is_sensenova_structured_policy(normalized):
        selection = configure_sensenova_structured_trainable(model, raw)
        model._tailsteer_trainable_groups = selection.groups
        normalized = selection.policy
    elif normalized == "generation_denoiser":
        for name, parameter in model.named_parameters():
            if is_sensenova_generation_state(name):
                parameter.requires_grad_(True)
    elif normalized == "full_denoiser":
        model.language_model.model.requires_grad_(True)
        model.language_model.model.embed_tokens.requires_grad_(False)
        model.fm_modules.requires_grad_(True)
    elif normalized == "full_model":
        model.requires_grad_(True)
    else:
        raise ValueError(
            "SenseNova trainable policy must be generation_denoiser, "
            "full_denoiser, full_model, output_head, or output_head_norm"
        )
    if train_reference:
        model.vision_model.requires_grad_(True)
    adapter = attach_prefix_kv_adapter(model, adapter_config)
    if adapter is not None:
        adapter.requires_grad_(True)
    conditioning_gradients = raw.get("conditioning_gradients")
    if conditioning_gradients is None:
        conditioning_gradients = normalized == "full_model"
    if not isinstance(conditioning_gradients, bool):
        raise TypeError("SenseNova conditioning_gradients must be a boolean")
    model._tailsteer_conditioning_gradients = conditioning_gradients
    master_dtype = str(raw.get("master_dtype", "float32"))
    if master_dtype not in {"float32", "model"}:
        raise ValueError("SenseNova master_dtype must be float32 or model")
    model._tailsteer_master_dtype = master_dtype
    selected = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    if not selected:
        raise RuntimeError(f"SenseNova policy {normalized!r} selected no parameters")
    return selected


def promote_sensenova_master_parameters(module: Any) -> None:
    """Promote a student FSDP unit to authoritative FP32 parameters."""

    import torch

    if getattr(module, "_tailsteer_master_dtype", None) == "model":
        return
    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.is_floating_point() and parameter.dtype != torch.float32:
                parameter.data = parameter.data.float()


@dataclass(frozen=True)
class SenseNovaReferenceInputs:
    pixel_values: Any
    grid_hw: Any


@dataclass(frozen=True)
class SenseNovaConditioning:
    references: tuple[SenseNovaReferenceInputs, ...]
    prompts: tuple[str, ...]
    width: int
    height: int
    batch_size: int


@dataclass(frozen=True)
class SenseNovaTrainingBatch:
    latent: Any
    source_latent: Any
    keep_mask: Any
    conditioning: SenseNovaConditioning
    step_index: int
    sigma: float
    timestep: float


def prepare_sensenova_training_batch(
    batch: Mapping[str, Any], *, runtime: Any, device: Any, dtype: Any
) -> SenseNovaTrainingBatch:
    import torch

    latent = batch["latent"]
    source = batch["source_latent"]
    if latent.ndim != 4 or latent.shape[1] != 1:
        raise ValueError("SenseNova cache state must have shape [B,1,L,D]")
    if source.shape != latent.shape:
        raise ValueError("SenseNova source and anchor token shapes differ")
    metadata = tuple(batch["metadata"])
    if len(metadata) != latent.shape[0]:
        raise ValueError("SenseNova metadata and tensor batch sizes differ")
    first = metadata[0]
    for field in ("step_index", "sigma", "timestep", "height", "width"):
        if len({item[field] for item in metadata}) != 1:
            raise ValueError(f"SenseNova batch mixes {field}")
    latent = latent[:, 0].to(device=device, dtype=dtype)
    source = source[:, 0].to(device=device, dtype=dtype)
    references = tuple(
        prepare_sensenova_reference_inputs(runtime, metadata_item=item, device=device)
        for item in metadata
    )
    return SenseNovaTrainingBatch(
        latent=latent,
        source_latent=source,
        keep_mask=batch["keep_mask"].to(device=device, dtype=torch.float32),
        conditioning=SenseNovaConditioning(
            references=references,
            prompts=tuple(str(item["prompt"]) for item in metadata),
            width=int(first["width"]),
            height=int(first["height"]),
            batch_size=int(latent.shape[0]),
        ),
        step_index=int(first["step_index"]),
        sigma=float(first["sigma"]),
        timestep=float(first["timestep"]),
    )


def prepare_sensenova_reference_inputs(
    runtime: Any, *, metadata_item: Mapping[str, Any], device: Any
) -> SenseNovaReferenceInputs:
    """Prepare conditioning from the original source through official code."""

    from tailsteer.models.flux2.cache import fit_rgb

    from .official import prepare_sensenova_reference_inputs as official_prepare

    image = fit_rgb(
        metadata_item["source_path"],
        int(metadata_item["width"]),
        int(metadata_item["height"]),
    )
    pixel_values, grid_hw = official_prepare(runtime, image)
    return SenseNovaReferenceInputs(
        pixel_values=pixel_values.to(device=device),
        grid_hw=grid_hw.to(device=device),
    )


class SenseNovaDenoiserTrainingModule:
    """Factory that keeps torch out of package discovery and owns native CFG."""

    @staticmethod
    def build(
        runtime: Any,
        *,
        cfg_scale: float,
        activation_offload: bool = False,
        flash_kv_preallocation: bool = True,
    ) -> Any:
        import torch

        class _StagedFullModelBranches(torch.autograd.Function):
            @staticmethod
            def forward(
                ctx: Any,
                state: Any,
                reference: SenseNovaReferenceInputs,
                module: Any,
                prompt: str,
                timestep: float,
                width: int,
                height: int,
            ) -> tuple[Any, Any]:
                ctx.module = module
                ctx.state = state.detach()
                ctx.reference = reference
                ctx.prompt = prompt
                ctx.timestep = float(timestep)
                ctx.width = int(width)
                ctx.height = int(height)
                return module._predict_one(
                    state.detach(),
                    reference,
                    prompt,
                    timestep=float(timestep),
                    width=int(width),
                    height=int(height),
                )

            @staticmethod
            def backward(
                ctx: Any, image_gradient: Any, condition_gradient: Any
            ) -> tuple[None, ...]:
                module = ctx.module
                module._run_staged_branch_backward(
                    ctx.state,
                    ctx.reference,
                    ctx.prompt,
                    timestep=ctx.timestep,
                    width=ctx.width,
                    height=ctx.height,
                    branch="conditioned",
                    output_gradient=condition_gradient,
                )
                module._run_staged_branch_backward(
                    ctx.state,
                    ctx.reference,
                    ctx.prompt,
                    timestep=ctx.timestep,
                    width=ctx.width,
                    height=ctx.height,
                    branch="image_conditioned",
                    output_gradient=image_gradient,
                )
                return (None,) * 7

        class _Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = runtime.model
                self.tokenizer = runtime.tokenizer
                self.cfg_scale = float(cfg_scale)
                self.activation_offload = bool(activation_offload)
                self.flash_kv_preallocation = bool(flash_kv_preallocation)
                self.reference_vision_trainable = False
                self.conditioning_gradients = False
                self.full_model_trainable = False
                self.prefix_kv_adapter = None
                if self.cfg_scale != 4.0:
                    raise ValueError("SenseNova cached training requires cfg_scale=4")
                self.model.requires_grad_(False)

            def configure_trainable(self, config: Mapping[str, Any]) -> tuple[str, ...]:
                selected = configure_sensenova_trainable(self.model, config)
                self.reference_vision_trainable = any(
                    parameter.requires_grad
                    for parameter in self.model.vision_model.parameters()
                )
                self.conditioning_gradients = bool(
                    getattr(self.model, "_tailsteer_conditioning_gradients", False)
                )
                self.full_model_trainable = (
                    str(config.get("policy", "generation_denoiser")) == "full_model"
                )
                adapter_config = PrefixKVAdapterConfig.from_mapping(
                    config.get("prefix_kv_adapter")
                )
                self.prefix_kv_adapter = (
                    self.model.tailsteer_prefix_kv_adapter
                    if adapter_config.enabled
                    else None
                )
                return selected

            def forward(
                self,
                state: Any,
                conditioning: SenseNovaConditioning,
                timestep: float,
                *,
                return_branches: bool = False,
            ) -> Any:
                autocast = torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=state.device.type == "cuda",
                )
                with autocast:
                    return self._forward_native(
                        state,
                        conditioning,
                        timestep,
                        return_branches=return_branches,
                    )

            def _forward_native(
                self,
                state: Any,
                conditioning: SenseNovaConditioning,
                timestep: float,
                *,
                return_branches: bool = False,
            ) -> Any:
                outputs = []
                for index in range(conditioning.batch_size):
                    item_state = state[index : index + 1]
                    arguments = (
                        item_state,
                        conditioning.references[index],
                        self,
                        conditioning.prompts[index],
                        float(timestep),
                        int(conditioning.width),
                        int(conditioning.height),
                    )
                    if self.full_model_trainable and torch.is_grad_enabled():
                        anchor = item_state.detach().requires_grad_(True)
                        outputs.append(
                            _StagedFullModelBranches.apply(anchor, *arguments[1:])
                        )
                    else:
                        outputs.append(
                            self._predict_one(
                                item_state,
                                conditioning.references[index],
                                conditioning.prompts[index],
                                timestep=float(timestep),
                                width=conditioning.width,
                                height=conditioning.height,
                            )
                        )
                image_conditioned = torch.cat([output[0] for output in outputs], dim=0)
                conditioned = torch.cat([output[1] for output in outputs], dim=0)
                if return_branches:
                    return image_conditioned, conditioned
                return image_conditioned + self.cfg_scale * (
                    conditioned - image_conditioned
                )

            def _predict_one(
                self,
                state: Any,
                reference: SenseNovaReferenceInputs,
                prompt: str,
                *,
                timestep: float,
                width: int,
                height: int,
                only_branch: str | None = None,
                cache_boundary: bool = False,
            ) -> Any:
                from sensenova_u1.models.neo_unify.modeling_neo_chat import (
                    clear_flash_kv_cache,
                    prepare_flash_kv_cache,
                )
                from sensenova_u1.models.neo_unify.modeling_qwen3 import (
                    create_block_causal_mask,
                )
                from sensenova_u1.models.neo_unify.utils import SYSTEM_MESSAGE_FOR_GEN

                model = self.model
                if only_branch not in {None, "conditioned", "image_conditioned"}:
                    raise ValueError(f"unknown SenseNova CFG branch {only_branch!r}")
                if cache_boundary and only_branch is None:
                    raise ValueError("cache-boundary forward requires one CFG branch")
                merge = int(1 / float(model.downsample_ratio))
                state_patch = int(model.patch_size) * merge
                token_h, token_w = height // state_patch, width // state_patch
                image_tokens = token_h * token_w
                expected = (image_tokens, state_patch * state_patch * 3)
                if tuple(state.shape[1:]) != expected:
                    raise ValueError(
                        "SenseNova cached token shape differs from configured "
                        "image size"
                    )

                flattened = reference.pixel_values
                grid_hw = reference.grid_hw
                grid_h = height // int(model.patch_size)
                grid_w = width // int(model.patch_size)

                model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(
                    "<IMG_CONTEXT>"
                )
                model.config.t_eps = 0.02
                think = "<think>\n\n</think>\n\n<img>"
                prompt_with_image = str(prompt)
                if "<image>" not in prompt_with_image:
                    prompt_with_image = "<image>\n" + prompt_with_image
                condition_query = model._build_t2i_query(
                    prompt_with_image,
                    system_message=SYSTEM_MESSAGE_FOR_GEN,
                    append_text=think,
                )
                image_query = model._build_t2i_query("<image>", append_text="<img>")
                image_sequence = "<img>" + "<IMG_CONTEXT>" * image_tokens + "</img>"
                condition_query = condition_query.replace("<image>", image_sequence, 1)
                image_query = image_query.replace("<image>", image_sequence, 1)

                gradients_enabled = torch.is_grad_enabled()
                differentiable_prefix = gradients_enabled and (
                    self.conditioning_gradients
                    or self.reference_vision_trainable
                    or self.prefix_kv_adapter is not None
                )
                prefix_context = (
                    nullcontext()
                    if self.reference_vision_trainable and gradients_enabled
                    else torch.no_grad()
                )
                saved_prefix = (
                    self._saved_tensor_context(
                        torch,
                        image_tokens,
                        prefix_activations=True,
                    )
                    if differentiable_prefix and self.reference_vision_trainable
                    else nullcontext()
                )
                with prefix_context, saved_prefix:
                    reference_embeddings = model.extract_feature(
                        flattened, grid_hw=grid_hw
                    )
                # The narrow default keeps the ordinary prefix LLM outside
                # autograd. The full-model SFT ablation keeps this graph live.
                prefix_gradients = gradients_enabled and self.conditioning_gradients
                prefix_forward_context = (
                    nullcontext() if prefix_gradients else torch.no_grad()
                )
                saved_condition_prefix = (
                    self._saved_tensor_context(
                        torch,
                        image_tokens,
                        prefix_activations=True,
                    )
                    if prefix_gradients
                    else nullcontext()
                )
                saved_image_prefix = (
                    self._saved_tensor_clone_context(
                        torch,
                        image_tokens,
                        prefix_activations=True,
                    )
                    if prefix_gradients and self.activation_offload
                    else self._saved_tensor_context(
                        torch,
                        image_tokens,
                        prefix_activations=True,
                    )
                )
                prefix_reference = (
                    reference_embeddings
                    if prefix_gradients
                    else reference_embeddings.detach()
                )
                condition_cache = None
                condition_indexes = None
                condition_image_mask = None
                if only_branch in {None, "conditioned"}:
                    with prefix_forward_context, saved_condition_prefix:
                        condition_inputs, condition_image_mask = (
                            self._build_prefix_inputs(
                                torch,
                                create_block_causal_mask,
                                condition_query,
                                prefix_reference,
                                grid_hw,
                            )
                        )
                        condition_cache, _ = model._it2i_prefix_forward(
                            *condition_inputs
                        )
                        condition_indexes = model._build_t2i_image_indexes(
                            token_h,
                            token_w,
                            condition_inputs[1][0].max() + 1,
                            device=state.device,
                        )
                image_cache = None
                image_indexes = None
                image_image_mask = None
                if only_branch in {None, "image_conditioned"}:
                    with prefix_forward_context, saved_image_prefix:
                        image_inputs, image_image_mask = self._build_prefix_inputs(
                            torch,
                            create_block_causal_mask,
                            image_query,
                            prefix_reference,
                            grid_hw,
                        )
                        image_cache, _ = model._it2i_prefix_forward(*image_inputs)
                        image_indexes = model._build_t2i_image_indexes(
                            token_h,
                            token_w,
                            image_inputs[1][0].max() + 1,
                            device=state.device,
                        )
                if self.prefix_kv_adapter is not None:
                    if condition_cache is not None:
                        self.prefix_kv_adapter.apply(
                            condition_cache,
                            condition_image_mask,
                            reference_embeddings,
                        )
                    if image_cache is not None:
                        self.prefix_kv_adapter.apply(
                            image_cache,
                            image_image_mask,
                            reference_embeddings,
                        )
                # The official preallocated Flash KV buffer performs in-place
                # copies. Keep it for the unchanged baseline and validation, but
                # use the differentiable torch.cat fallback whenever gradients
                # must reach reference features or the KV adapter.
                if not differentiable_prefix and self.flash_kv_preallocation:
                    if condition_cache is not None:
                        prepare_flash_kv_cache(
                            condition_cache, current_len=image_tokens, batch_size=1
                        )
                    if image_cache is not None:
                        prepare_flash_kv_cache(
                            image_cache, current_len=image_tokens, batch_size=1
                        )

                original_cache_tensors: tuple[Any, ...] = ()
                detached_cache_tensors: tuple[Any, ...] = ()
                if cache_boundary:
                    selected_cache = (
                        condition_cache if only_branch == "conditioned" else image_cache
                    )
                    (
                        selected_cache,
                        original_cache_tensors,
                        detached_cache_tensors,
                    ) = self._detach_prefix_cache(selected_cache)
                    if only_branch == "conditioned":
                        condition_cache = selected_cache
                    else:
                        image_cache = selected_cache

                try:
                    pixels = model.unpatchify(state, state_patch, height, width)
                    image_input = model.patchify(
                        pixels, int(model.patch_size), channel_first=True
                    )
                    image_embeds = model.extract_feature(
                        image_input.view(grid_h * grid_w, -1),
                        gen_model=True,
                        grid_hw=grid_hw,
                    ).view(1, image_tokens, -1)
                    t = torch.tensor(
                        float(timestep), device=state.device, dtype=torch.float32
                    )
                    expanded_t = t.expand(image_tokens)
                    time_embedding = model.fm_modules["timestep_embedder"](
                        expanded_t
                    ).view(1, image_tokens, -1)
                    noise_scale = float(model.noise_scale)
                    if model.noise_scale_mode in {
                        "resolution",
                        "dynamic",
                        "dynamic_sqrt",
                    }:
                        scale = math.sqrt(
                            (grid_h * grid_w)
                            / (merge**2)
                            / float(model.noise_scale_base_image_seq_len)
                        )
                        noise_scale *= scale
                        if model.noise_scale_mode == "dynamic_sqrt":
                            noise_scale = math.sqrt(noise_scale)
                    noise_scale = min(noise_scale, float(model.noise_scale_max_value))
                    if model.add_noise_scale_embedding:
                        normalized = noise_scale / float(model.noise_scale_max_value)
                        noise_t = torch.full_like(expanded_t, normalized)
                        time_embedding = time_embedding + model.fm_modules[
                            "noise_scale_embedder"
                        ](noise_t).view(1, image_tokens, -1)
                    image_embeds = image_embeds + time_embedding
                    attention = {"full_attention": None}
                    saved_conditioned = self._saved_tensor_context(
                        torch,
                        image_tokens,
                    )
                    saved_image_conditioned = self._saved_tensor_context(
                        torch,
                        image_tokens,
                    )
                    conditioned = None
                    if only_branch in {None, "conditioned"}:
                        with saved_conditioned:
                            conditioned = model._t2i_predict_v(
                                image_embeds,
                                condition_indexes,
                                attention,
                                condition_cache,
                                t,
                                state,
                                image_token_num=image_tokens,
                                timestep_embeddings=time_embedding,
                                image_size=(width, height),
                            )
                    image_conditioned = None
                    if only_branch in {None, "image_conditioned"}:
                        with saved_image_conditioned:
                            image_conditioned = model._t2i_predict_v(
                                image_embeds,
                                image_indexes,
                                attention,
                                image_cache,
                                t,
                                state,
                                image_token_num=image_tokens,
                                timestep_embeddings=time_embedding,
                                image_size=(width, height),
                            )
                    if only_branch is None:
                        return image_conditioned, conditioned
                    selected = (
                        conditioned
                        if only_branch == "conditioned"
                        else image_conditioned
                    )
                    if cache_boundary:
                        return (
                            selected,
                            original_cache_tensors,
                            detached_cache_tensors,
                        )
                    return selected
                finally:
                    if condition_cache is not None:
                        clear_flash_kv_cache(condition_cache)
                    if image_cache is not None:
                        clear_flash_kv_cache(image_cache)

            def _detach_prefix_cache(
                self,
                cache: Any,
            ) -> tuple[Any, tuple[Any, ...], tuple[Any, ...]]:
                if cache is None:
                    raise ValueError("cannot detach an absent SenseNova prefix cache")
                from torch.distributed._composable.fsdp import fully_shard

                detached_cache = copy(cache)
                detached_layers = []
                originals = []
                detached = []
                model_layers = tuple(self.model.language_model.model.layers)
                if len(cache.layers) != len(model_layers):
                    raise RuntimeError(
                        "SenseNova prefix cache and FSDP layer counts differ"
                    )
                for cache_layer, model_layer in zip(
                    cache.layers, model_layers, strict=True
                ):
                    detached_layer = copy(cache_layer)
                    fsdp_state = fully_shard.state(model_layer)
                    for attribute in ("keys", "values"):
                        value = getattr(cache_layer, attribute)
                        if value is None:
                            setattr(detached_layer, attribute, None)
                            continue
                        if value.requires_grad:

                            def pre_backward(
                                gradient: Any, *, state: Any = fsdp_state
                            ) -> Any:
                                group = state._fsdp_param_group
                                if group is not None and group.is_sharded:
                                    return state._pre_backward(gradient)
                                return gradient

                            value.register_hook(pre_backward)
                        leaf = value.detach().requires_grad_(value.requires_grad)
                        setattr(detached_layer, attribute, leaf)
                        if value.requires_grad:
                            originals.append(value)
                            detached.append(leaf)
                    detached_layers.append(detached_layer)
                detached_cache.layers = detached_layers
                if not originals:
                    raise RuntimeError(
                        "SenseNova differentiable prefix has no KV tensors"
                    )
                return detached_cache, tuple(originals), tuple(detached)

            def _reset_staged_fsdp(self) -> None:
                from torch.distributed._composable.fsdp import fully_shard
                from torch.distributed._composable.fsdp._fsdp_common import (
                    TrainingState,
                )

                layers = tuple(self.model.language_model.model.layers)
                for layer in layers:
                    state = fully_shard.state(layer)
                    group = state._fsdp_param_group
                    if group is None:
                        continue
                    state._training_state = TrainingState.IDLE
                    group._training_state = TrainingState.IDLE
                    group._post_forward_indices.clear()
                if layers:
                    state = fully_shard.state(layers[0])
                    state._comm_ctx.post_forward_order.clear()

            def _run_staged_branch_backward(
                self,
                state: Any,
                reference: SenseNovaReferenceInputs,
                prompt: str,
                *,
                timestep: float,
                width: int,
                height: int,
                branch: str,
                output_gradient: Any,
            ) -> None:
                if output_gradient is None:
                    return
                self._reset_staged_fsdp()
                autocast = torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=state.device.type == "cuda",
                )
                with torch.enable_grad(), autocast:
                    output, original_cache, detached_cache = self._predict_one(
                        state,
                        reference,
                        prompt,
                        timestep=timestep,
                        width=width,
                        height=height,
                        only_branch=branch,
                        cache_boundary=True,
                    )
                    torch.autograd.backward(output, output_gradient)
                    cache_backward = tuple(
                        (original, leaf.grad)
                        for original, leaf in zip(
                            original_cache, detached_cache, strict=True
                        )
                        if leaf.grad is not None
                    )
                self._reset_staged_fsdp()
                if cache_backward:
                    torch.autograd.backward(
                        tuple(item[0] for item in cache_backward),
                        tuple(item[1] for item in cache_backward),
                    )
                self._reset_staged_fsdp()

            def _build_prefix_inputs(
                self,
                torch: Any,
                create_block_causal_mask: Any,
                query: str,
                reference_embeddings: Any,
                grid_hw: Any,
            ) -> tuple[tuple[Any, Any, Any], Any]:
                """Build a prefix once while retaining exact image-token slots."""

                model = self.model
                tokenized = self.tokenizer(query, return_tensors="pt")
                input_ids = tokenized["input_ids"].to(model.device)
                indexes = model.get_thw_indexes(input_ids[0], grid_hw)
                attention = {"full_attention": create_block_causal_mask(indexes[0])}
                input_embeddings = model.language_model.get_input_embeddings()(
                    input_ids
                )
                batch, sequence, width = input_embeddings.shape
                flat_ids = input_ids.reshape(batch * sequence)
                flat_mask = flat_ids == model.img_context_token_id
                positions = flat_mask.nonzero(as_tuple=False).flatten()
                reference = reference_embeddings.reshape(-1, width).to(
                    device=input_embeddings.device, dtype=input_embeddings.dtype
                )
                if int(positions.numel()) != int(reference.shape[0]):
                    raise ValueError(
                        "reference embedding count differs from IMG_CONTEXT slots: "
                        f"{reference.shape[0]} != {positions.numel()}"
                    )
                flat_embeddings = input_embeddings.reshape(batch * sequence, width)
                flat_embeddings = flat_embeddings.index_copy(0, positions, reference)
                return (
                    flat_embeddings.reshape(batch, sequence, width),
                    indexes,
                    attention,
                ), flat_mask.reshape(batch, sequence)[0]

            def _saved_tensor_context(
                self,
                torch: Any,
                image_tokens: int,
                *,
                prefix_activations: bool = False,
                enabled: bool = True,
            ) -> Any:
                if (
                    not enabled
                    or not self.activation_offload
                    or not torch.is_grad_enabled()
                ):
                    return nullcontext()

                def pack(tensor: Any) -> tuple[Any | None, Any]:
                    selected = image_tokens in tensor.shape or (
                        prefix_activations and tensor.ndim >= 3
                    )
                    if tensor.device.type != "cuda" or not selected:
                        return None, tensor
                    host = torch.empty(
                        tensor.size(),
                        dtype=tensor.dtype,
                        layout=tensor.layout,
                        pin_memory=not tensor.is_sparse,
                    )
                    host.copy_(tensor)
                    return tensor.device, host

                def unpack(value: tuple[Any | None, Any]) -> Any:
                    device, tensor = value
                    return (
                        tensor
                        if device is None
                        else tensor.to(device, non_blocking=True)
                    )

                return torch.autograd.graph.saved_tensors_hooks(pack, unpack)

            @staticmethod
            def _saved_tensor_clone_context(
                torch: Any,
                image_tokens: int,
                *,
                prefix_activations: bool = False,
            ) -> Any:
                """Keep selected FSDP activations in owned GPU storage.

                The official model saves views that can otherwise outlive an
                FSDP unit's all-gathered storage. Cloning only the retained CFG
                branch avoids that invalid view without doubling CPU offload.
                """

                if not torch.is_grad_enabled():
                    return nullcontext()

                def pack(tensor: Any) -> Any:
                    selected = image_tokens in tensor.shape or (
                        prefix_activations and tensor.ndim >= 3
                    )
                    if tensor.device.type != "cuda" or not selected:
                        return tensor
                    return tensor.detach().clone()

                def unpack(tensor: Any) -> Any:
                    return tensor

                return torch.autograd.graph.saved_tensors_hooks(pack, unpack)

        return _Module()


def unwrap(module: Any) -> Any:
    seen: set[int] = set()
    while hasattr(module, "module"):
        if id(module) in seen or module.module is module:
            raise RuntimeError("SenseNova wrapper contains a module cycle")
        seen.add(id(module))
        module = module.module
    return module
