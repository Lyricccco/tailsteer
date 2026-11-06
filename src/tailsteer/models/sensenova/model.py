"""Minimal SenseNova U1.5 model backend for native-cache offline training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from tailsteer.models.protocols import TailVelocityBranches

from .export import SenseNovaExportSpec
from .identity import resolve_sensenova_variant
from .objective import sensenova_preservation_velocity
from .official import load_sensenova_runtime, sensenova_timesteps
from .policy import canonical_sensenova_policy
from .training import (
    SenseNovaDenoiserTrainingModule,
    SenseNovaTrainingBatch,
    configure_sensenova_trainable,
    is_sensenova_generation_state,
    prepare_sensenova_reference_inputs,
    prepare_sensenova_training_batch,
    promote_sensenova_master_parameters,
    unwrap,
)


def _torch_dtype(name: str) -> Any:
    import torch

    normalized = str(name).removeprefix("torch.")
    if normalized not in {"float32", "float16", "bfloat16"}:
        raise ValueError(f"unsupported SenseNova dtype {name!r}")
    return getattr(torch, normalized)


class SenseNovaModel:
    """Official-source full-denoiser backend; storage stays model-native."""

    def __init__(
        self,
        runtime: Any,
        *,
        device: Any,
        dtype: Any,
        cfg_scale: float,
        activation_offload: bool,
        model_name: str,
        pretrained_runtime: Any | None = None,
        timestep_shift: float = 3.0,
    ) -> None:
        self.runtime = runtime
        self.device = device
        self.dtype = dtype
        self.cfg_scale = float(cfg_scale)
        self.trainable_policy = "generation_denoiser"
        self.trainable_config: dict[str, Any] = {
            "policy": "generation_denoiser",
            "trainable_groups": [],
            "master_dtype": "float32",
            "reference_vision_model": False,
            "prefix_kv_adapter": {"enabled": False},
            "conditioning_gradients": False,
        }
        self.model_name = str(model_name)
        self.timestep_shift = float(timestep_shift)
        self._root = SenseNovaDenoiserTrainingModule.build(
            runtime,
            cfg_scale=cfg_scale,
            activation_offload=activation_offload,
        )
        self.pretrained_runtime = (
            runtime if pretrained_runtime is None else pretrained_runtime
        )
        self._pretrained_root = (
            self._root
            if self.pretrained_runtime is runtime
            else SenseNovaDenoiserTrainingModule.build(
                self.pretrained_runtime,
                cfg_scale=cfg_scale,
                activation_offload=False,
                flash_kv_preallocation=False,
            )
        )
        if self._pretrained_root is not self._root:
            self._pretrained_root.requires_grad_(False)
            self._pretrained_root.eval()
        self._export_names: frozenset[str] = frozenset()
        self._fsdp_master_parameters_prepared = False

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
    ) -> SenseNovaModel:
        import torch

        del include_vae
        if gradient_checkpointing:
            raise ValueError(
                "SenseNova mutable prefix KV is incompatible with FSDP "
                "gradient checkpointing; use bounded activation offload"
            )
        device = device or torch.device(
            "cuda", int(__import__("os").environ.get("LOCAL_RANK", "0"))
        )
        dtype = _torch_dtype(str(config.get("dtype", "bfloat16")))
        model_name = str(config.get("name", ""))
        variant = resolve_sensenova_variant(model_name, config.get("variant"))
        code_path = config.get("code_path", config.get("support_path"))
        if code_path is None:
            raise KeyError("SenseNova model config requires code_path")
        runtime = load_sensenova_runtime(
            model_root=config["path"],
            model_code_repo=code_path,
            variant=variant,
            device=str(device),
            torch_dtype=dtype,
        )
        pretrained_runtime = (
            load_sensenova_runtime(
                model_root=config["path"],
                model_code_repo=code_path,
                variant=variant,
                device=str(device),
                torch_dtype=dtype,
            )
            if independent_actors
            else runtime
        )
        result = cls(
            runtime,
            device=device,
            dtype=dtype,
            cfg_scale=float(config.get("cfg_scale", 4.0)),
            activation_offload=bool(config.get("activation_offload", False)),
            model_name=model_name,
            pretrained_runtime=pretrained_runtime,
            timestep_shift=float(config.get("timestep_shift", 3.0)),
        )
        if export_path is not None:
            result.load_export(export_path)
        return result

    def configure_trainable(self, config: Mapping[str, Any]) -> None:
        policy = canonical_sensenova_policy(
            str(config.get("policy", "generation_denoiser"))
        )
        root = unwrap(self._root)
        selected = root.configure_trainable(config)
        self.trainable_policy = policy
        self.trainable_config = deepcopy(dict(config))
        self.trainable_config["policy"] = policy
        groups = tuple(getattr(root.model, "_tailsteer_trainable_groups", ()))
        if groups:
            self.trainable_config["trainable_groups"] = list(groups)
        self.trainable_config.setdefault("master_dtype", "float32")
        self.trainable_config.setdefault("reference_vision_model", False)
        self.trainable_config.setdefault("prefix_kv_adapter", {"enabled": False})
        self.trainable_config.setdefault(
            "conditioning_gradients", policy == "full_model"
        )
        self._export_names = frozenset(f"model.{name}" for name in selected)
        root.train()

    @staticmethod
    def _is_generation_state(name: str) -> bool:
        return is_sensenova_generation_state(name)

    def _is_export_state(self, name: str) -> bool:
        return name in self._export_names

    def prepare_inference_export(
        self, state: Mapping[str, Any], *, step: int
    ) -> tuple[dict[str, Any], str, dict[str, Any]]:
        selected = {
            name: tensor
            for name, tensor in state.items()
            if self._is_export_state(name)
        }
        if set(selected) != set(self._export_names):
            missing = sorted(set(self._export_names) - set(selected))
            raise RuntimeError(
                "SenseNova inference export is missing trainable tensors: "
                f"{missing[:3]}"
            )
        if not selected:
            raise RuntimeError("SenseNova inference export selected no tensors")
        extended = any(
            self._is_export_state(name) and not self._is_generation_state(name)
            for name in selected
        )
        manifest = {
            "schema_version": (
                3 if self.trainable_policy == "full_model" else 2 if extended else 1
            ),
            "format": "tailsteer_sensenova_delta",
            "model": self.model_name,
            "variant": self.runtime.variant,
            "base_model_path": str(self.runtime.model_root),
            "source_revision": "f71dfb098226d01edc0e4c67b3917a2af71a30ef",
            "step": int(step),
            "trainable_policy": self.trainable_policy,
            "trainable": deepcopy(self.trainable_config),
            "weights": "model.safetensors",
        }
        dtypes = {
            str(tensor.dtype).removeprefix("torch.")
            for tensor in selected.values()
            if hasattr(tensor, "dtype")
        }
        if len(dtypes) == 1:
            manifest["weight_dtype"] = next(iter(dtypes))
        return selected, "model.safetensors", manifest

    def load_export(self, export_path: str | Path) -> None:
        from ..checkpoint import open_weights

        spec = SenseNovaExportSpec.load(export_path)
        manifest = spec.manifest
        manifest_path = spec.root / "manifest.json"
        if manifest.get("model") != self.model_name:
            raise ValueError(f"incompatible inference export: {manifest_path}")
        if manifest.get("variant") != self.runtime.variant:
            raise ValueError(
                f"export variant {manifest.get('variant')!r} does not match "
                f"{self.runtime.variant!r}"
            )
        trainable = spec.trainable
        root = unwrap(self._root)
        configure = getattr(root, "configure_trainable", None)
        if callable(configure):
            selected = configure(trainable)
        else:
            selected = configure_sensenova_trainable(root.model, trainable)
        self.trainable_policy = str(trainable.get("policy", "generation_denoiser"))
        self.trainable_config = deepcopy(trainable)
        self._export_names = frozenset(f"model.{name}" for name in selected)
        with open_weights(
            spec.weights, prefix=manifest.get("weight_prefix", "")
        ) as handle:
            state = {name: handle.get_tensor(name) for name in handle.keys()}
        if not state or set(state) != set(self._export_names):
            raise ValueError("SenseNova export contains invalid trainable-state keys")
        incompatible = self._root.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                "SenseNova export has unexpected keys: "
                f"{incompatible.unexpected_keys[:3]}"
            )
        loaded = set(state)
        expected = set(self._export_names)
        if loaded != expected:
            missing = sorted(expected - loaded)
            extra = sorted(loaded - expected)
            raise RuntimeError(
                "SenseNova inference delta is incomplete: "
                f"missing={missing[:3]} extra={extra[:3]}"
            )
        root.requires_grad_(False)
        root.eval()

    def get_trainable_module(self) -> Any:
        return self._root

    def parameter_anchor_reference(self) -> Any:
        """Return the frozen pretrained denoiser for parameter anchoring."""

        return self._pretrained_root

    def replace_trainable_module(self, module: Any) -> None:
        self._root = module

    def sharding_units(self) -> tuple[Any, ...]:
        # Each transformer block is an independent FSDP unit; the root shards
        # embeddings, heads, and the small generation/reference modules.
        root = unwrap(self._root)
        layers = tuple(root.model.language_model.model.layers)
        if not layers:
            raise ValueError("SenseNova language backbone has no sharding layers")
        return layers

    def prepare_sharding_unit(self, unit: Any) -> None:
        if str(self.trainable_config.get("master_dtype", "float32")) == "float32":
            promote_sensenova_master_parameters(unit)

    def prepare_sharding_root(self) -> None:
        if str(self.trainable_config.get("master_dtype", "float32")) == "float32":
            promote_sensenova_master_parameters(unwrap(self._root).model)
            self._fsdp_master_parameters_prepared = True

    @staticmethod
    def _local_parameter_tensor(parameter: Any) -> Any:
        to_local = getattr(parameter, "to_local", None)
        return to_local() if callable(to_local) else parameter

    def prepare_optimizer_parameters(self) -> None:
        import torch

        if str(self.trainable_config.get("master_dtype", "float32")) != "float32":
            return
        if not self._fsdp_master_parameters_prepared:
            promote_sensenova_master_parameters(unwrap(self._root).model)
        mismatched = [
            name
            for name, parameter in self._root.named_parameters()
            if parameter.requires_grad
            and (
                parameter.dtype != torch.float32
                or self._local_parameter_tensor(parameter).dtype != torch.float32
            )
        ]
        if mismatched:
            raise RuntimeError(
                f"SenseNova trainable parameters lack FP32 masters: {mismatched[:3]}"
            )

    def auxiliary_sharding_groups(
        self,
    ) -> tuple[tuple[str, Any, tuple[Any, ...]], ...]:
        """Expose the frozen teacher so FSDP does not replicate it per rank."""

        if self._pretrained_root is self._root:
            return ()
        root = unwrap(self._pretrained_root)
        layers = tuple(root.model.language_model.model.layers)
        if not layers:
            raise ValueError("SenseNova teacher has no sharding layers")
        return (("pretrained_teacher", self._pretrained_root, layers),)

    def replace_auxiliary_sharded_module(self, name: str, module: Any) -> None:
        if name != "pretrained_teacher":
            raise KeyError(f"unknown SenseNova auxiliary module {name!r}")
        self._pretrained_root = module

    def trainable_parameters(self) -> tuple[Any, ...]:
        return tuple(
            parameter
            for parameter in self._root.parameters()
            if parameter.requires_grad
        )

    def prepare_training_batch(
        self, batch: Mapping[str, Any]
    ) -> SenseNovaTrainingBatch:
        return prepare_sensenova_training_batch(
            batch, runtime=self.runtime, device=self.device, dtype=self.dtype
        )

    def prepare_tail_batch(
        self, batch: Mapping[str, Any], *, state_keys: Sequence[str]
    ) -> Any:
        from tailsteer.models.protocols import PreparedTrajectoryBatch

        from .training import SenseNovaConditioning

        metadata = tuple(batch["metadata"])
        if not metadata:
            raise ValueError("SenseNova tail batch has no metadata")
        first = metadata[0]
        for field in ("height", "width"):
            if len({int(item[field]) for item in metadata}) != 1:
                raise ValueError(f"SenseNova tail batch mixes {field}")
        source = batch["source_latent"]
        if source.ndim != 4 or source.shape[1] != 1:
            raise ValueError("SenseNova source tail states must have shape [B,1,L,D]")
        source = source[:, 0].to(device=self.device, dtype=self.dtype)
        states = {}
        for key in state_keys:
            value = batch[key]
            if value.ndim != 4 or value.shape[1] != 1:
                raise ValueError(f"SenseNova tail state {key!r} has invalid shape")
            value = value[:, 0]
            if value.shape != source.shape:
                raise ValueError(f"SenseNova tail state {key!r} differs from source")
            states[key] = value.to(device=self.device, dtype=self.dtype)
        prompts = tuple(str(item["prompt"]) for item in metadata)
        height, width = int(first["height"]), int(first["width"])
        references = tuple(
            prepare_sensenova_reference_inputs(
                self.runtime,
                metadata_item=item,
                device=self.device,
            )
            for item in metadata
        )
        conditioning = SenseNovaConditioning(
            references=references,
            prompts=prompts,
            width=width,
            height=height,
            batch_size=int(source.shape[0]),
        )
        return PreparedTrajectoryBatch(
            states=states,
            source_state=source,
            keep_mask=batch["keep_mask"].to(device=self.device, dtype=self.dtype),
            conditioning=conditioning,
            prompts=prompts,
            height=height,
            width=width,
        )

    def predict_tail_velocity(
        self, latent: Any, point: Any, conditioning: Any, *, actor: str
    ) -> Any:
        return self.compose_tail_velocity_branches(
            self.predict_tail_velocity_branches(
                latent, point, conditioning, actor=actor
            )
        )

    @staticmethod
    def tail_velocity_branch_names() -> tuple[str, ...]:
        return ("base", "conditioned")

    def predict_tail_velocity_branches(
        self, latent: Any, point: Any, conditioning: Any, *, actor: str
    ) -> TailVelocityBranches:
        if actor == "refined":
            root = self._root
        elif actor == "base":
            root = self._pretrained_root
        else:
            raise ValueError(f"unknown SenseNova actor {actor!r}")
        base, conditioned = root(
            latent,
            conditioning,
            float(point.timestep),
            return_branches=True,
        )
        return TailVelocityBranches({"base": base, "conditioned": conditioned})

    def actor_memory_boundary(self) -> None:
        """Release cached blocks between the independently sharded actors."""

        if self._pretrained_root is self._root:
            return
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def compose_tail_velocity_branches(self, prediction: TailVelocityBranches) -> Any:
        values = dict(prediction.values)
        expected = set(self.tail_velocity_branch_names())
        if set(values) != expected:
            raise ValueError(
                "SenseNova velocity branches differ from the two-branch CFG policy: "
                f"expected={sorted(expected)} actual={sorted(values)}"
            )
        base = values["base"]
        return base + self.cfg_scale * (values["conditioned"] - base)

    def tail_schedule(
        self, *, num_steps: int, height: int, width: int
    ) -> dict[int, Any]:
        from tailsteer.models.protocols import TrajectoryPoint

        values = (
            sensenova_timesteps(
                self.runtime,
                width=width,
                height=height,
                steps=num_steps,
                shift=self.timestep_shift,
            )
            .detach()
            .float()
            .cpu()
        )
        return {
            index + 1: TrajectoryPoint(
                float(index + 1), 1.0 - float(timestep), float(timestep)
            )
            for index, timestep in enumerate(values)
        }

    @staticmethod
    def tail_advance(
        latent: Any,
        velocity: Any,
        *,
        current: Any,
        following: Any,
    ) -> Any:
        import torch

        current_t = torch.tensor(
            float(current.timestep), device=latent.device, dtype=torch.float32
        )
        following_t = torch.tensor(
            float(following.timestep), device=latent.device, dtype=torch.float32
        )
        return latent + velocity * (following_t - current_t)

    @staticmethod
    def tail_step_delta(current: Any, following: Any) -> float:
        """SenseNova integrates velocity in its increasing native timestep."""

        return float(following.timestep) - float(current.timestep)

    @staticmethod
    def tail_terminal_delta(point: Any) -> float:
        return float(point.sigma)

    @staticmethod
    def tail_keep_mask(mask: Any, state: Any) -> Any:
        if mask.ndim == 4 and mask.shape[1] == 1:
            mask = mask[:, 0]
        if state.ndim != 3 or mask.ndim != 3 or mask.shape[0] != state.shape[0]:
            raise ValueError("SenseNova tail mask/state batch shapes differ")
        if mask.shape == state.shape:
            # Fine is already patchified per RGB pixel; do not average channels.
            return mask.to(device=state.device, dtype=state.dtype)
        if mask.shape[1] * mask.shape[2] != state.shape[1]:
            raise ValueError("SenseNova tail mask grid differs from token state")
        return mask.reshape(mask.shape[0], state.shape[1], 1).to(
            device=state.device, dtype=state.dtype
        )

    def training_forward(self, batch: Any) -> Any:
        prepared = (
            batch
            if isinstance(batch, SenseNovaTrainingBatch)
            else self.prepare_training_batch(batch)
        )
        return self._root(
            prepared.latent,
            prepared.conditioning,
            prepared.timestep,
        )

    def compute_training_loss(
        self, batch: Mapping[str, Any], *, preservation_weight: float
    ) -> Any:
        prepared = self.prepare_training_batch(batch)
        prediction = self.training_forward(prepared)
        return sensenova_preservation_velocity(
            prediction,
            prepared.latent,
            prepared.source_latent,
            prepared.keep_mask,
            sigma=prepared.sigma,
            weight=preservation_weight,
        )

    def generate_validation_sample(
        self, sample: Any, config: Mapping[str, Any]
    ) -> tuple[Any, Any]:
        """Run the official complete image-editing sampler for one cache source."""

        import numpy as np
        from PIL import Image

        metadata = sample.metadata
        height, width = int(metadata["height"]), int(metadata["width"])
        source = sample.source_latent
        if source.ndim == 3 and source.shape[0] == 1:
            source = source[0]
        if source.ndim != 2:
            raise ValueError("SenseNova validation source must have shape [1,L,D]")
        source = source.unsqueeze(0).to(device=self.device, dtype=self.dtype)
        raw = self.runtime.model
        merge = int(1 / float(raw.downsample_ratio))
        patch = int(raw.patch_size) * merge
        source_pixels = raw.unpatchify(source, patch, height, width)
        source_uint8 = (
            (source_pixels[0].float().clamp(-1, 1) * 0.5 + 0.5)
            .permute(1, 2, 0)
            .mul(255)
            .round()
            .byte()
            .cpu()
            .numpy()
        )
        source_image = Image.fromarray(np.asarray(source_uint8), mode="RGB")
        generated_image = self.generate_image(
            source_image,
            prompt=str(metadata["prompt"]),
            seed=int(metadata["seed"]),
            width=width,
            height=height,
            config=config,
        )
        return source_image, generated_image

    def generate_image(
        self,
        source_image: Any,
        *,
        prompt: str,
        seed: int,
        width: int,
        height: int,
        config: Mapping[str, Any],
        condition_images: Sequence[Any] = (),
    ) -> Any:
        """Edit the primary PIL image with optional ordered static references."""

        import numpy as np
        from PIL import Image

        generated = self.generate_pixels(
            source_image,
            prompt=prompt,
            seed=seed,
            width=width,
            height=height,
            config=config,
            condition_images=condition_images,
        )
        generated_uint8 = (
            (generated[0].float().clamp(-1, 1) * 0.5 + 0.5)
            .permute(1, 2, 0)
            .mul(255)
            .round()
            .byte()
            .cpu()
            .numpy()
        )
        return Image.fromarray(np.asarray(generated_uint8), mode="RGB")

    def generate_pixels(
        self,
        source_image: Any,
        *,
        prompt: str,
        seed: int,
        width: int,
        height: int,
        config: Mapping[str, Any],
        condition_images: Sequence[Any] = (),
    ) -> Any:
        """Return the sampler's continuous ``[-1, 1]`` pixel state."""

        prompt = str(prompt)
        images = [source_image, *condition_images]
        if prompt.count("<image>") > len(images):
            raise ValueError("SenseNova prompt has more image placeholders than inputs")
        # Preserve single-image prompts; let the official multi-image sampler
        # insert its Image-1/Image-2 labels for an untagged editing instruction.
        if len(images) == 1 and "<image>" not in prompt:
            prompt = "<image>\n" + prompt
        return self.runtime.model.it2i_generate(
            self.runtime.tokenizer,
            prompt,
            images,
            cfg_scale=float(config.get("cfg_scale", 4.0)),
            img_cfg_scale=float(config.get("img_cfg_scale", 1.0)),
            cfg_norm=str(config.get("cfg_norm", "none")),
            enable_timestep_shift=True,
            timestep_shift=float(config.get("timestep_shift", 3.0)),
            image_size=(width, height),
            num_steps=int(config.get("num_steps", 50)),
            batch_size=1,
            think_mode=False,
            seed=int(seed),
        )
