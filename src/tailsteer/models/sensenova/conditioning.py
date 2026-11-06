"""Trainable adapters for SenseNova's frozen source-conditioning prefix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PrefixKVAdapterConfig:
    """Shape-independent configuration for an image-only prefix-KV adapter."""

    enabled: bool = False
    target: str = "image_tokens"
    last_n_layers: int = 8
    rank: int = 8
    scale: float = 1.0

    @classmethod
    def from_mapping(cls, raw: Any) -> PrefixKVAdapterConfig:
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise TypeError("prefix_kv_adapter must be a mapping")
        unknown = set(raw) - {
            "enabled",
            "target",
            "last_n_layers",
            "rank",
            "scale",
        }
        if unknown:
            raise ValueError(f"unknown prefix_kv_adapter fields: {sorted(unknown)}")
        result = cls(
            enabled=bool(raw.get("enabled", False)),
            target=str(raw.get("target", "image_tokens")),
            last_n_layers=int(raw.get("last_n_layers", 8)),
            rank=int(raw.get("rank", 8)),
            scale=float(raw.get("scale", 1.0)),
        )
        if result.target != "image_tokens":
            raise ValueError("SenseNova prefix KV adapters must target image_tokens")
        if result.last_n_layers < 1:
            raise ValueError("prefix KV adapter last_n_layers must be positive")
        if result.rank < 1:
            raise ValueError("prefix KV adapter rank must be positive")
        if result.scale <= 0:
            raise ValueError("prefix KV adapter scale must be positive")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _LowRankResidual(torch.nn.Module):
    """Zero-output low-rank residual with a live first-step gradient."""

    def __init__(self, width: int, rank: int, scale: float) -> None:
        super().__init__()
        self.down = torch.nn.Linear(width, rank, bias=False)
        self.up = torch.nn.Linear(rank, width, bias=False)
        self.scale = float(scale)
        torch.nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        torch.nn.init.zeros_(self.up.weight)

    def forward(
        self,
        value: torch.Tensor,
        reference_low_rank: torch.Tensor,
    ) -> torch.Tensor:
        if reference_low_rank.shape != (*value.shape[:-1], self.down.out_features):
            raise ValueError("reference low-rank features differ from prefix K/V")
        return self.up(self.down(value) + reference_low_rank) * self.scale


class _LayerKVResidual(torch.nn.Module):
    def __init__(
        self,
        width: int,
        reference_width: int,
        head_count: int,
        rank: int,
        scale: float,
    ) -> None:
        super().__init__()
        self.key = _LowRankResidual(width, rank, scale)
        self.value = _LowRankResidual(width, rank, scale)
        self.head_count = int(head_count)
        self.rank = int(rank)
        self.key_reference = torch.nn.Linear(
            reference_width, self.head_count * self.rank, bias=False
        )
        self.value_reference = torch.nn.Linear(
            reference_width, self.head_count * self.rank, bias=False
        )

    def reference_features(
        self,
        reference: torch.Tensor,
        projection: torch.nn.Linear,
    ) -> torch.Tensor:
        tokens = int(reference.shape[0])
        return (
            projection(reference)
            .reshape(tokens, self.head_count, self.rank)
            .permute(1, 0, 2)
            .unsqueeze(0)
        )


class ImagePrefixKVAdapter(torch.nn.Module):
    """Fuse base K/V and reference features into image-token-only residuals."""

    def __init__(
        self,
        *,
        layer_count: int,
        head_dim: int,
        reference_width: int,
        head_count: int,
        config: PrefixKVAdapterConfig,
    ) -> None:
        super().__init__()
        if not config.enabled:
            raise ValueError("cannot construct a disabled prefix KV adapter")
        if config.last_n_layers > layer_count:
            raise ValueError(
                "prefix KV adapter selects more layers than the language model: "
                f"{config.last_n_layers} > {layer_count}"
            )
        self.config = config
        self.layer_count = int(layer_count)
        self.first_layer = self.layer_count - int(config.last_n_layers)
        self.layers = torch.nn.ModuleDict(
            {
                str(index): _LayerKVResidual(
                    int(head_dim),
                    int(reference_width),
                    int(head_count),
                    int(config.rank),
                    float(config.scale),
                )
                for index in range(self.first_layer, self.layer_count)
            }
        )

    def apply(
        self,
        past_key_values: Any,
        image_token_mask: torch.Tensor,
        reference_embeddings: torch.Tensor,
    ) -> None:
        """Replace cache tensors with differentiable image-only residuals.

        Prefix caches use ``[B, H_kv, S, D]`` before the official FlashAttention
        preparation step. Each residual receives both the frozen base K/V and
        the matching reference-vision embedding. The same source-image positions
        are present in every layer, while text-token positions remain bit-identical.
        """

        if image_token_mask.ndim != 1 or image_token_mask.dtype != torch.bool:
            raise ValueError("image_token_mask must be a one-dimensional bool tensor")
        positions = image_token_mask.nonzero(as_tuple=False).flatten()
        if positions.numel() == 0:
            raise ValueError("prefix KV adapter received no image-token positions")
        reference = reference_embeddings.reshape(
            -1, int(reference_embeddings.shape[-1])
        )
        if int(reference.shape[0]) != int(positions.numel()):
            raise ValueError(
                "reference embedding count differs from image-token positions: "
                f"{reference.shape[0]} != {positions.numel()}"
            )
        cache_layers = past_key_values.layers
        if len(cache_layers) != self.layer_count:
            raise ValueError(
                "prefix cache layer count differs from adapter: "
                f"{len(cache_layers)} != {self.layer_count}"
            )
        for raw_index, adapter in self.layers.items():
            layer = cache_layers[int(raw_index)]
            key, value = layer.keys, layer.values
            if key is None or value is None:
                raise ValueError(f"prefix cache layer {raw_index} has no K/V")
            if key.ndim != 4 or value.shape != key.shape:
                raise ValueError(
                    f"prefix cache layer {raw_index} has invalid K/V shapes"
                )
            if int(key.shape[2]) != int(image_token_mask.numel()):
                raise ValueError(
                    "image-token mask length differs from prefix cache sequence: "
                    f"{image_token_mask.numel()} != {key.shape[2]}"
                )
            selected_key = key.index_select(2, positions)
            selected_value = value.index_select(2, positions)
            key_reference = adapter.reference_features(
                reference, adapter.key_reference
            ).to(dtype=selected_key.dtype)
            value_reference = adapter.reference_features(
                reference, adapter.value_reference
            ).to(dtype=selected_value.dtype)
            key_delta = torch.zeros_like(key).index_copy(
                2,
                positions,
                adapter.key(selected_key, key_reference),
            )
            value_delta = torch.zeros_like(value).index_copy(
                2,
                positions,
                adapter.value(selected_value, value_reference),
            )
            layer.keys = key + key_delta
            layer.values = value + value_delta

    def extra_repr(self) -> str:
        return (
            f"layers=[{self.first_layer},{self.layer_count}), "
            f"rank={self.config.rank}, scale={self.config.scale}"
        )


def attach_prefix_kv_adapter(
    model: Any, config: PrefixKVAdapterConfig
) -> ImagePrefixKVAdapter | None:
    """Attach one stable, checkpoint-visible adapter to the official model."""

    name = "tailsteer_prefix_kv_adapter"
    existing = getattr(model, name, None)
    if not config.enabled:
        if existing is not None:
            raise ValueError("a prefix KV adapter is attached but disabled by config")
        return None
    if existing is not None:
        if not isinstance(existing, ImagePrefixKVAdapter):
            raise TypeError(f"model.{name} has an incompatible type")
        if existing.config != config:
            raise ValueError("attached prefix KV adapter config differs from request")
        return existing
    layers = tuple(model.language_model.model.layers)
    if not layers:
        raise ValueError("SenseNova language model has no layers")
    head_dim = int(layers[0].self_attn.head_dim)
    head_count = int(layers[0].self_attn.config.num_key_value_heads)
    reference_width = int(layers[0].self_attn.q_proj.in_features)
    adapter = ImagePrefixKVAdapter(
        layer_count=len(layers),
        head_dim=head_dim,
        reference_width=reference_width,
        head_count=head_count,
        config=config,
    )
    parameter = next(model.parameters())
    adapter.to(device=parameter.device, dtype=parameter.dtype)
    model.add_module(name, adapter)
    return adapter
