"""Common training/export lifecycle for native diffusion transformers.

Model-specific subclasses still own conditioning, CFG, state layout, and time
integration.  Keeping those semantics out of this class is intentional: in
particular, supported models use
decreasing remaining sigma.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tailsteer.models.protocols import PreparedTrajectoryBatch


@dataclass(frozen=True)
class TransformerConditioning:
    payload: Mapping[str, Any]
    batch_size: int


def unwrap(module: Any) -> Any:
    seen: set[int] = set()
    while hasattr(module, "module"):
        if id(module) in seen or module.module is module:
            raise RuntimeError("cyclic module wrapper")
        seen.add(id(module))
        module = module.module
    return module


def torch_dtype(name: str) -> Any:
    import torch

    normalized = str(name).removeprefix("torch.")
    if normalized not in {"float32", "float16", "bfloat16"}:
        raise ValueError(f"unsupported model dtype {name!r}")
    return getattr(torch, normalized)


def fit_source_image(path: str | Path, *, width: int, height: int) -> Any:
    from PIL import Image, ImageOps

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with Image.open(source) as opened:
        return ImageOps.fit(
            opened.convert("RGB"),
            (int(width), int(height)),
            method=Image.Resampling.LANCZOS,
        )


class NativeTransformerTailModel:
    """Reusable model/optimizer/export mechanics for a transformer denoiser."""

    export_format = "tailsteer_native_transformer"
    source_revision = "unknown"
    trainable_policies = frozenset({"denoiser_full"})

    def __init__(
        self,
        *,
        refined_transformer: Any,
        pretrained_transformer: Any,
        device: Any,
        dtype: Any,
        model_name: str,
        model_path: str | Path,
        layer_checkpointing: bool,
    ) -> None:
        self.refined_transformer = refined_transformer
        self.pretrained_transformer = pretrained_transformer
        self.device = device
        self.dtype = dtype
        self.model_name = str(model_name)
        self.model_path = Path(model_path).expanduser().resolve()
        self.layer_checkpointing = bool(layer_checkpointing)
        self.trainable_policy = "denoiser_full"
        self.trainable_config: dict[str, Any] = {
            "policy": "denoiser_full",
            "additional_modules": [],
            "master_dtype": "float32",
            "conditioning_gradients": False,
        }
        self._export_names: frozenset[str] = frozenset()
        if self.pretrained_transformer is not self.refined_transformer:
            self.pretrained_transformer.requires_grad_(False)
            self.pretrained_transformer.eval()

    def get_trainable_module(self) -> Any:
        return self.refined_transformer

    def parameter_anchor_reference(self) -> Any:
        """Return the frozen pretrained transformer for parameter anchoring."""

        return self.pretrained_transformer

    def replace_trainable_module(self, module: Any) -> None:
        self.refined_transformer = module
        self._replace_pipeline_transformer(module)

    def _replace_pipeline_transformer(self, module: Any) -> None:
        raise NotImplementedError

    def configure_trainable(self, config: Mapping[str, Any]) -> None:
        values = deepcopy(dict(config))
        values.setdefault("policy", "denoiser_full")
        values.setdefault("additional_modules", [])
        values.setdefault("master_dtype", "float32")
        values.setdefault("conditioning_gradients", False)
        policy = str(values["policy"])
        if policy not in self.trainable_policies:
            raise ValueError(
                f"{self.model_name} trainable policy must be one of "
                f"{sorted(self.trainable_policies)}"
            )
        additional = tuple(str(name) for name in values["additional_modules"])
        if additional:
            raise ValueError(
                f"{self.model_name} does not yet expose additional trainable "
                f"module groups: {list(additional)}"
            )
        root = unwrap(self.refined_transformer)
        self._configure_trainable_parameters(root, policy)
        # Portable actors require a complete transformer, including frozen
        # parameters when a backend exposes a narrower training policy.
        self._export_names = frozenset(dict(root.named_parameters()))
        root.train()
        self.trainable_policy = policy
        self.trainable_config = values
        if not any(parameter.requires_grad for parameter in root.parameters()):
            raise RuntimeError("trainable policy selected no transformer parameters")

    def _configure_trainable_parameters(self, root: Any, policy: str) -> None:
        root.requires_grad_(policy == "denoiser_full")

    def trainable_parameters(self) -> tuple[Any, ...]:
        return tuple(
            parameter
            for parameter in self.refined_transformer.parameters()
            if parameter.requires_grad
        )

    def transformer_blocks(self, transformer: Any) -> tuple[Any, ...]:
        raise NotImplementedError

    def sharding_units(self) -> tuple[Any, ...]:
        units = self.transformer_blocks(unwrap(self.refined_transformer))
        if not units:
            raise ValueError(f"{self.model_name} transformer exposes no shard units")
        return units

    def auxiliary_sharding_groups(
        self,
    ) -> tuple[tuple[str, Any, tuple[Any, ...]], ...]:
        """Shard an independent frozen teacher instead of replicating it per rank."""

        if self.pretrained_transformer is self.refined_transformer:
            return ()
        root = unwrap(self.pretrained_transformer)
        units = self.transformer_blocks(root)
        if not units:
            raise ValueError(f"{self.model_name} teacher exposes no shard units")
        return (("pretrained_teacher", self.pretrained_transformer, units),)

    def replace_auxiliary_sharded_module(self, name: str, module: Any) -> None:
        if name != "pretrained_teacher":
            raise KeyError(f"unknown {self.model_name} auxiliary module {name!r}")
        self.pretrained_transformer = module

    def prepare_activation_checkpointing(self) -> None:
        if not self.layer_checkpointing:
            return
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            checkpoint_wrapper,
        )

        for parent, attribute, index, layer in self._checkpoint_locations():
            if getattr(layer, "_tailsteer_activation_checkpoint", False):
                continue
            wrapped = checkpoint_wrapper(
                layer, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            )
            wrapped._tailsteer_activation_checkpoint = True
            getattr(parent, attribute)[index] = wrapped

    def _checkpoint_locations(self) -> tuple[tuple[Any, str, int, Any], ...]:
        raise NotImplementedError

    def _uses_fp32_master(self) -> bool:
        return str(self.trainable_config.get("master_dtype", "float32")) == "float32"

    @staticmethod
    def _promote_trainable(module: Any) -> None:
        import torch

        with torch.no_grad():
            for parameter in module.parameters():
                if parameter.requires_grad and parameter.dtype != torch.float32:
                    parameter.data = parameter.data.float()

    def prepare_sharding_unit(self, unit: Any) -> None:
        if self._uses_fp32_master():
            self._promote_trainable(unit)

    def prepare_sharding_root(self) -> None:
        if self._uses_fp32_master():
            self._promote_trainable(unwrap(self.refined_transformer))

    def prepare_optimizer_parameters(self) -> None:
        import torch

        if self._uses_fp32_master():
            self._promote_trainable(unwrap(self.refined_transformer))
            mismatched = [
                name
                for name, parameter in self.refined_transformer.named_parameters()
                if parameter.requires_grad and parameter.dtype != torch.float32
            ]
            if mismatched:
                raise RuntimeError(
                    "trainable transformer parameters lack FP32 masters: "
                    f"{mismatched[:3]}"
                )

    def actor_transformer(self, actor: str) -> Any:
        if actor == "refined":
            return self.refined_transformer
        if actor == "base":
            return self.pretrained_transformer
        raise ValueError(f"unknown {self.model_name} actor {actor!r}")

    def autocast_context(self, transformer: Any) -> Any:
        import torch

        parameter = next(unwrap(transformer).parameters())
        device = parameter.device
        if device.type != "cuda" or self.dtype not in {torch.float16, torch.bfloat16}:
            return nullcontext()
        # Inference uses parameters stored in the requested compute dtype. Enabling
        # autocast again changes FP32 runtime inputs such as RoPE tables. Only
        # FP32-master students require an explicit compute autocast.
        if parameter.dtype == self.dtype:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.dtype)

    @staticmethod
    def _strip_cached_singleton(value: Any, *, expected_rank: int, label: str) -> Any:
        while value.ndim > expected_rank and value.shape[1] == 1:
            value = value.squeeze(1)
        if value.ndim != expected_rank:
            raise ValueError(
                f"{label} expected rank {expected_rank}, got {tuple(value.shape)}"
            )
        return value

    def state_rank(self) -> int:
        raise NotImplementedError

    def state_dtype(self) -> Any:
        """State precision is model-owned, independently of parameter precision."""
        return self.dtype

    def prepare_conditioning_from_metadata(
        self,
        *,
        source_state: Any,
        metadata: Sequence[Mapping[str, Any]],
        prompts: Sequence[str],
        height: int,
        width: int,
    ) -> TransformerConditioning:
        raise NotImplementedError

    def prepare_tail_batch(
        self, batch: Mapping[str, Any], *, state_keys: Sequence[str]
    ) -> PreparedTrajectoryBatch:
        metadata = tuple(batch["metadata"])
        if not metadata:
            raise ValueError(f"{self.model_name} tail batch has no metadata")
        for field in ("height", "width"):
            if len({int(item[field]) for item in metadata}) != 1:
                raise ValueError(f"{self.model_name} tail batch mixes {field}")
        rank = self.state_rank()
        source = self._strip_cached_singleton(
            batch["source_latent"], expected_rank=rank, label="source state"
        ).to(device=self.device, dtype=self.state_dtype())
        states = {}
        for key in state_keys:
            value = self._strip_cached_singleton(
                batch[key], expected_rank=rank, label=f"tail state {key!r}"
            )
            if value.shape != source.shape:
                raise ValueError(f"tail state {key!r} differs from source shape")
            states[key] = value.to(device=self.device, dtype=self.state_dtype())
        prompts = tuple(str(item["prompt"]) for item in metadata)
        height, width = int(metadata[0]["height"]), int(metadata[0]["width"])
        conditioning = self.prepare_conditioning_from_metadata(
            source_state=source,
            metadata=metadata,
            prompts=prompts,
            height=height,
            width=width,
        )
        return PreparedTrajectoryBatch(
            states=states,
            source_state=source,
            keep_mask=batch["keep_mask"].to(device=self.device),
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

    def prepare_inference_export(
        self, state: Mapping[str, Any], *, step: int
    ) -> tuple[dict[str, Any], str, dict[str, Any]]:
        import torch

        if self.trainable_policy not in self.trainable_policies:
            raise ValueError(f"unsupported export policy: {self.trainable_policy!r}")
        # Actor switching is parameter-only. Scheduler/rotary buffers belong to
        # the official base runtime and are neither optimized nor duplicated.
        expected = set(self._export_names)
        if not expected:
            raise RuntimeError("native transformer export names were not configured")
        selected = {
            name: tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
            for name, tensor in state.items()
            if name in expected
        }
        if set(selected) != expected:
            missing = sorted(expected - set(selected))
            extra = sorted(set(selected) - expected)
            raise RuntimeError(
                "native transformer export is incomplete: "
                f"missing={missing[:3]} extra={extra[:3]}"
            )
        manifest = {
            "schema_version": 1,
            "format": self.export_format,
            "model": self.model_name,
            "base_model_path": str(self.model_path),
            "source_revision": self.source_revision,
            "step": int(step),
            "trainable_policy": self.trainable_policy,
            "trainable": deepcopy(self.trainable_config),
            "weight_dtype": "bfloat16",
            "weights": "transformer.safetensors",
        }
        return selected, "transformer.safetensors", manifest

    def unload(self) -> None:
        import gc

        import torch

        self.refined_transformer = None
        self.pretrained_transformer = None
        self._release_runtime()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _release_runtime(self) -> None:
        raise NotImplementedError
